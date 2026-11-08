"""哥哥，模拟设备状态要跨请求保存，不能让成功指令悄悄丢掉。"""

import asyncio
import copy
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api_routes import register_routes
from core.device_registry_factory import (
    close_registry,
    invalidate_registry_cache,
    load_custom_devices,
    save_custom_devices,
    setup_registry,
)
from devices.base import DeviceCapability, DeviceCommand
from devices.simulator_driver import SimulatorDriver


@pytest.fixture
def simulator_client(monkeypatch):
    monkeypatch.setenv("AUTONOMY_LEVEL", "medium")
    username = f"simulator_{uuid.uuid4().hex}"
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        yield client, username
    invalidate_registry_cache(username)


def _create_pump(client, username, device_id="pump_01"):
    response = client.post(
        f"/api/devices?username={username}",
        json={
            "device_id": device_id,
            "name": "测试水泵",
            "driver": "simulator",
            "capabilities": ["irrigate"],
            "initial_state": {"power": False, "status": "powered_off"},
        },
    )
    assert response.json()["success"] is True


def _state(client, username, device_id="pump_01"):
    return client.get(
        f"/api/devices/{device_id}/state?username={username}"
    ).json()


@pytest.mark.parametrize("expire_cache", [False, True])
def test_模拟器指令状态在刷新和缓存过期后仍保留(
    simulator_client, monkeypatch, expire_cache
):
    import core.device_registry_factory as factory

    client, username = simulator_client
    _create_pump(client, username)
    result = client.post(
        f"/api/devices/pump_01/command?username={username}",
        json={"command": "start", "params": {"duration": 5}},
    ).json()
    assert result["success"] is True
    assert _state(client, username)["status"] == "running"

    if expire_cache:
        monkeypatch.setattr(factory, "_CACHE_TTL_SECONDS", -1)
    else:
        assert client.post(
            f"/api/devices/refresh?username={username}"
        ).json()["success"] is True

    restored = _state(client, username)
    assert restored["status"] == "running"
    assert restored["power"] is True
    assert restored["last_duration"] == 5


def test_待确认执行成功后设备接口立即显示实际状态(
    simulator_client, monkeypatch
):
    client, username = simulator_client
    _create_pump(client, username)
    assert _state(client, username)["status"] == "powered_off"
    monkeypatch.setenv("AUTONOMY_LEVEL", "low")
    requested = client.post(
        f"/api/devices/pump_01/command?username={username}",
        json={"command": "start", "params": {"duration": 5}},
    ).json()
    assert requested["decision"] == "need_confirm"
    confirmed = client.post(
        f"/api/actions/{requested['pending_id']}/confirm?username={username}"
    ).json()
    assert confirmed["success"] is True
    assert confirmed["action_status"] == "executed"
    assert _state(client, username)["status"] == "running"
    assert client.get(
        f"/api/actions/pending?username={username}"
    ).json() == []


def test_独立注册中心执行后已缓存设备也能恢复最新状态(simulator_client):
    client, username = simulator_client
    _create_pump(client, username)
    assert _state(client, username)["status"] == "powered_off"
    registry, loop = setup_registry(username)
    try:
        loop.run_until_complete(registry.discover_all())
        result = loop.run_until_complete(
            registry.execute("pump_01", DeviceCommand(command="start"))
        )
        assert result.success is True
    finally:
        close_registry(loop, registry)
    assert _state(client, username)["status"] == "running"


def test_模拟状态只写回当前用户且不改真实协议配置(simulator_client):
    client, username = simulator_client
    other_username = f"other_{uuid.uuid4().hex}"
    _create_pump(client, username)
    _create_pump(client, other_username)
    real_config = {
        "device_id": "configured_real_pump",
        "name": "仅保存配置，不连接硬件",
        "driver": "http",
        "capabilities": ["irrigate"],
        "initial_state": {"status": "unchanged"},
    }
    other_configs = load_custom_devices(other_username)
    save_custom_devices(other_username, [*other_configs, real_config])
    assert client.post(
        f"/api/devices/pump_01/command?username={username}",
        json={"command": "start", "params": {"duration": 5}},
    ).json()["success"] is True

    saved = load_custom_devices(username)[0]
    assert saved["initial_state"]["status"] == "running"
    other = {item["device_id"]: item for item in load_custom_devices(other_username)}
    assert other["pump_01"]["initial_state"]["status"] == "powered_off"
    assert other["configured_real_pump"]["initial_state"] == {"status": "unchanged"}


def test_虚拟传感器漂移在注册中心重建后继续累积(
    simulator_client, monkeypatch
):
    client, username = simulator_client
    monkeypatch.setattr(
        "devices.simulator_driver.random.uniform", lambda lo, hi: (lo + hi) / 2
    )
    assert client.post(
        f"/api/devices?username={username}",
        json={
            "device_id": "soil_01",
            "name": "测试土壤传感器",
            "driver": "simulator",
            "capabilities": ["read_sensor"],
            "sensors": ["soil_moisture"],
            "initial_state": {"soil_moisture": 45},
        },
    ).json()["success"] is True
    assert _state(client, username, "soil_01")["soil_moisture"] == 44.8
    invalidate_registry_cache(username)
    assert _state(client, username, "soil_01")["soil_moisture"] == 44.6


def test_旧读取快照不会覆盖另一个驱动刚执行的启动状态(monkeypatch):
    monkeypatch.setattr("devices.simulator_driver.random.uniform", lambda lo, hi: hi)
    stored = {"pump_01": {"power": False, "status": "powered_off", "temperature": 20.0}}
    store_lock = threading.Lock()
    snapshot_ready = threading.Event()
    release_reader = threading.Event()

    def load():
        with store_lock:
            return copy.deepcopy(stored)

    def paused_reader_load():
        snapshot = load()
        snapshot_ready.set()
        assert release_reader.wait(timeout=5)
        return snapshot

    def save(device_id, changes):
        with store_lock:
            stored[device_id].update(copy.deepcopy(changes))

    reader = SimulatorDriver(0, state_loader=paused_reader_load, state_saver=save)
    writer = SimulatorDriver(0, state_loader=load, state_saver=save)
    for driver in (reader, writer):
        driver.add_virtual_device(
            "pump_01", "测试复合设备",
            [DeviceCapability.IRRIGATE, DeviceCapability.READ_SENSOR],
            sensors=["temperature"], initial_state=stored["pump_01"],
        )
        asyncio.run(driver.connect())

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending_read = pool.submit(asyncio.run, reader.read_state("pump_01"))
        try:
            assert snapshot_ready.wait(timeout=5)
            result = asyncio.run(writer.execute("pump_01", DeviceCommand(command="start")))
            assert result.success is True
            assert stored["pump_01"]["status"] == "running"
        finally:
            release_reader.set()
        pending_read.result(timeout=5)

    assert {key: stored["pump_01"][key] for key in ("power", "status", "temperature")} == {
        "power": True, "status": "running", "temperature": 20.5,
    }
    assert stored["pump_01"]["run_id"]
    assert stored["pump_01"]["run_started_at"]


def test_并发模拟状态写入保留不同字段的更新():
    from sqlalchemy import event
    from core.database import engine as db_engine
    from core.device_registry_factory import _save_simulator_state

    username = f"simulator_cas_{uuid.uuid4().hex}"
    save_custom_devices(username, [{
        "device_id": "pump_01", "name": "测试水泵", "driver": "simulator",
        "capabilities": ["irrigate", "read_sensor"],
        "initial_state": {
            "power": False, "status": "powered_off", "temperature": 20, "humidity": 65,
        },
    }])
    both_read = threading.Barrier(2, timeout=5)
    worker_state = threading.local()

    def pause_first_state_read(connection, cursor, statement, parameters, context, many):
        if (statement.lstrip().upper().startswith("SELECT")
                and "device_configs" in statement and not getattr(worker_state, "read", False)):
            worker_state.read = True
            both_read.wait()

    event.listen(db_engine._engine, "after_cursor_execute", pause_first_state_read)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            writers = [
                pool.submit(_save_simulator_state, username, "pump_01", {"temperature": 20.5}),
                pool.submit(_save_simulator_state, username, "pump_01", {"humidity": 66}),
            ]
            for writer in writers:
                writer.result(timeout=10)
    finally:
        event.remove(db_engine._engine, "after_cursor_execute", pause_first_state_read)

    assert load_custom_devices(username)[0]["initial_state"] == {
        "power": False, "status": "powered_off", "temperature": 20.5, "humidity": 66,
    }


@pytest.mark.parametrize(
    "pump_plot,pump_zone,sensor_plot,sensor_zone,expected",
    [
        (101, "east", 202, "east", 44.8),
        (101, "east", 101, "west", 44.8),
        (101, "east", 101, "east", 46.0),
    ],
)
@pytest.mark.parametrize("location", ["同一个园区", ""])
def test_真实注册中心按地块分区隔离同名位置的灌溉反馈(
    monkeypatch, pump_plot, pump_zone, sensor_plot, sensor_zone, expected, location
):
    monkeypatch.setattr(
        "devices.simulator_driver.random.uniform", lambda lo, hi: (lo + hi) / 2
    )
    username = f"simulator_scope_{uuid.uuid4().hex}"
    save_custom_devices(username, [
        {
            "device_id": "pump_01", "name": "测试水泵", "driver": "simulator",
            "capabilities": ["irrigate"], "location": location,
            "plot_id": pump_plot, "zone_id": pump_zone,
            "initial_state": {"power": False, "status": "powered_off"},
        },
        {
            "device_id": "soil_01", "name": "测试土壤传感器", "driver": "simulator",
            "capabilities": ["read_sensor"], "sensors": ["soil_moisture"],
            "location": location, "plot_id": sensor_plot, "zone_id": sensor_zone,
            "initial_state": {"soil_moisture": 45.0},
        },
    ])
    registry, loop = setup_registry(username)
    try:
        loop.run_until_complete(registry.discover_all())
        assert loop.run_until_complete(
            registry.execute("pump_01", DeviceCommand(command="start"))
        ).success
        state = loop.run_until_complete(registry.read_state("soil_01"))
        assert state["soil_moisture"] == expected
    finally:
        close_registry(loop, registry)
