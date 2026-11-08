"""哥哥，用可控时钟验证模拟运行时长和跨注册中心恢复。"""

import asyncio
import copy
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

import devices.simulator_driver as simulator_module
from core.device_registry_factory import (
    close_registry, load_custom_devices, save_custom_devices, setup_registry,
)
from devices.base import DeviceCapability, DeviceCommand
from devices.simulator_driver import SimulatorDriver


@pytest.fixture
def clock(monkeypatch):
    current = [2_000_000_000.0]

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(current[0], tz=tz)

    monkeypatch.setattr(simulator_module, "datetime", FixedDatetime)
    return current


def _driver(initial_state=None):
    driver = SimulatorDriver(simulated_latency_ms=0)
    driver.add_virtual_device(
        "pump_01", "演示水泵", [DeviceCapability.IRRIGATE],
        initial_state=initial_state or {
            "power": False, "status": "powered_off", "flow_rate": 0, "rpm": 0,
        },
    )
    asyncio.run(driver.connect())
    return driver


def _execute(driver, command="start", **params):
    result = asyncio.run(driver.execute("pump_01", DeviceCommand(command, params)))
    assert result.success, result


def _state(driver, device_id="pump_01"):
    return asyncio.run(driver.read_state(device_id))


def test_运行时长按分钟计且到期停止保持通电(clock):
    driver = _driver()
    _execute(driver, duration=1, flow_rate=8)
    clock[0] += 59
    assert _state(driver)["status"] == "running"
    clock[0] += 1

    state = _state(driver)

    assert state["status"] == "standby"
    assert state["power"] is True
    assert state["flow_rate"] == 0
    assert state["rpm"] == 0
    assert state.get("run_until") is None


@pytest.mark.parametrize("entrypoint", ["discover", "read", "execute"])
def test_截止时间持久化且重建注册中心后从正常入口校正(clock, entrypoint):
    username = f"duration_{uuid.uuid4().hex}"
    save_custom_devices(username, [{
        "device_id": "pump_01", "name": "演示水泵", "driver": "simulator",
        "capabilities": ["irrigate"],
        "initial_state": {"power": False, "status": "powered_off"},
    }])
    registry, loop = setup_registry(username)
    try:
        loop.run_until_complete(registry.discover_all())
        assert loop.run_until_complete(registry.execute(
            "pump_01", DeviceCommand("start", {"duration": 1}),
        )).success
    finally:
        close_registry(loop, registry)
    stored = load_custom_devices(username)[0]["initial_state"]
    assert stored.get("run_until") == clock[0] + 60
    clock[0] += 60

    registry, loop = setup_registry(username)
    try:
        simulator = registry._drivers["simulator"]
        if entrypoint == "discover":
            loop.run_until_complete(simulator.discover())
        elif entrypoint == "read":
            loop.run_until_complete(simulator.read_state("pump_01"))
        else:
            assert loop.run_until_complete(simulator.execute(
                "pump_01", DeviceCommand("power_on"),
            )).success
        restored = load_custom_devices(username)[0]["initial_state"]
        assert restored["status"] == "standby"
        assert restored["power"] is True
        assert restored.get("run_until") is None
    finally:
        close_registry(loop, registry)


def test_旧运行记录没有截止时间时保持运行(clock):
    driver = _driver({"power": True, "status": "running", "last_duration": 5})
    clock[0] += 86_400

    assert _state(driver)["status"] == "running"


@pytest.mark.parametrize("command", ["stop", "power_off", "shutdown"])
def test_停止和断电清除计时且无时长重启后持续运行(clock, command):
    driver = _driver()
    _execute(driver, duration=1)
    clock[0] += 30
    _execute(driver, command)
    assert _state(driver).get("run_until") is None
    _execute(driver)
    clock[0] += 120

    assert _state(driver)["status"] == "running"


def test_重新指定时长从本次启动重新计时(clock):
    driver = _driver()
    _execute(driver, duration=1)
    clock[0] += 30
    _execute(driver, duration=2)
    clock[0] += 30
    assert _state(driver)["status"] == "running"
    clock[0] += 90

    assert _state(driver)["status"] == "standby"


def test_无时长启动取消原计时(clock):
    driver = _driver()
    _execute(driver, duration=1)
    clock[0] += 30
    _execute(driver)
    clock[0] += 300

    assert _state(driver)["status"] == "running"
    assert _state(driver).get("run_until") is None


def test_读取土壤前先停止过期灌溉设备(clock, monkeypatch):
    monkeypatch.setattr(simulator_module.random, "uniform", lambda low, high: (low + high) / 2)
    driver = _driver()
    driver.add_virtual_device(
        "soil_01", "演示传感器", [DeviceCapability.READ_SENSOR],
        sensors=["soil_moisture"], initial_state={"soil_moisture": 45.0},
    )
    _execute(driver, duration=1)
    clock[0] += 60

    assert _state(driver, "soil_01")["soil_moisture"] == 44.8


@pytest.mark.parametrize("params,seconds", [
    ({"duration": "0.5"}, 30), ({"duration_minutes": 1}, 60), ({"duration": 0}, 0),
])
def test_支持小数分钟别名和零时长(clock, params, seconds):
    driver = _driver()
    _execute(driver, **params)
    clock[0] += seconds

    assert _state(driver)["status"] == "standby"


@pytest.mark.parametrize("duration", ["invalid", -1, float("nan"), float("inf"), 1e308])
def test_非法时长不启动设备(clock, duration):
    driver = _driver()

    result = asyncio.run(driver.execute("pump_01", DeviceCommand("start", {"duration": duration})))

    assert result.success is False
    assert result.error_code == "INVALID_PARAMS"
    assert _state(driver)["status"] == "powered_off"


def test_故障复位后清除旧截止时间(clock):
    driver = _driver({
        "status": "error", "power": True, "run_until": clock[0] + 60,
    })

    _execute(driver, "reset")

    assert _state(driver)["status"] == "standby"
    assert _state(driver).get("run_until") is None


def test_现有两参数状态保存回调仍可保存到期状态(clock):
    stored = {"pump_01": {"power": False, "status": "powered_off"}}

    def save(device_id, changes):
        stored[device_id].update(copy.deepcopy(changes))

    driver = SimulatorDriver(
        simulated_latency_ms=0, state_loader=lambda: copy.deepcopy(stored), state_saver=save,
    )
    driver.add_virtual_device("pump_01", "演示水泵", [DeviceCapability.IRRIGATE])
    asyncio.run(driver.connect())
    _execute(driver, duration=1)
    assert stored["pump_01"]["run_until"] == clock[0] + 60
    clock[0] += 60

    assert _state(driver)["status"] == "standby"
    assert stored["pump_01"]["status"] == "standby"
    assert stored["pump_01"]["run_until"] is None


def test_过期旧快照不能覆盖另一个注册中心新启动的计时(clock):
    username = f"duration_race_{uuid.uuid4().hex}"
    save_custom_devices(username, [{
        "device_id": "pump_01", "name": "演示水泵", "driver": "simulator",
        "capabilities": ["irrigate"],
        "initial_state": {"power": False, "status": "powered_off"},
    }])
    reader_registry, reader_loop = setup_registry(username)
    writer_registry, writer_loop = setup_registry(username)
    reader = reader_registry._drivers["simulator"]
    writer = writer_registry._drivers["simulator"]
    snapshot_ready = threading.Event()
    release_reader = threading.Event()
    load = reader._state_loader
    first_load = [True]

    def paused_load():
        state = load()
        if first_load[0]:
            first_load[0] = False
            snapshot_ready.set()
            assert release_reader.wait(timeout=8)
        return state

    try:
        _execute(writer, duration=1)
        clock[0] += 60
        reader._state_loader = paused_load
        with ThreadPoolExecutor(max_workers=1) as pool:
            reading = pool.submit(_state, reader)
            try:
                assert snapshot_ready.wait(timeout=8)
                _execute(writer, duration=2)
            finally:
                release_reader.set()
            result = reading.result(timeout=8)
        saved = load_custom_devices(username)[0]["initial_state"]
        assert saved["status"] == "running"
        assert saved["run_until"] == clock[0] + 120
        assert result["status"] == "running"
    finally:
        close_registry(reader_loop, reader_registry)
        close_registry(writer_loop, writer_registry)
