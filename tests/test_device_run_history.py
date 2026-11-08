"""哥哥，后续作业不能抹掉旧任务的结束证据，也不能重新执行旧编号。"""

import asyncio
import uuid
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


def _start(driver, run_id):
    result = asyncio.run(driver.execute(
        "pump", DeviceCommand("start", {"duration": 1}, execution_id=run_id),
    ))
    assert result.success, result
    return result


def _driver(initial_state=None):
    driver = SimulatorDriver(simulated_latency_ms=0)
    driver.add_virtual_device(
        "pump", "麦田水泵", [DeviceCapability.IRRIGATE],
        initial_state=initial_state or {"power": False, "status": "powered_off"},
    )
    asyncio.run(driver.connect())
    return driver


def test_结束证据在多轮作业和注册中心重建后仍按编号保存(clock):
    username = f"run_history_{uuid.uuid4().hex}"
    save_custom_devices(username, [{
        "device_id": "pump", "name": "麦田水泵", "driver": "simulator",
        "capabilities": ["irrigate"],
        "initial_state": {"power": False, "status": "powered_off"},
    }])
    for run_id, completed in (("run-a", True), ("run-b", False), ("run-c", True)):
        registry, loop = setup_registry(username)
        try:
            loop.run_until_complete(registry.discover_all())
            assert loop.run_until_complete(registry.execute(
                "pump", DeviceCommand("start", {"duration": 1}, execution_id=run_id),
            )).success
            clock[0] += 60 if completed else 10
            if not completed:
                assert loop.run_until_complete(registry.execute(
                    "pump", DeviceCommand("stop"),
                )).success
            loop.run_until_complete(registry.read_state("pump"))
        finally:
            close_registry(loop, registry)

    stored = load_custom_devices(username)[0]["initial_state"]
    assert stored["run_history"]["run-a"]["reason"] == "completed"
    assert stored["run_history"]["run-b"]["reason"] == "interrupted"
    assert stored["run_history"]["run-c"]["reason"] == "completed"
    assert stored["run_history"]["run-a"]["ended_at"] == datetime.fromtimestamp(2_000_000_060).isoformat()
    assert stored["last_ended_run_id"] == "run-c"
    registry, loop = setup_registry(username)
    try:
        loop.run_until_complete(registry.discover_all())
        restored = loop.run_until_complete(registry.read_state("pump"))
        assert restored["run_history"] == stored["run_history"]
    finally:
        close_registry(loop, registry)


@pytest.mark.parametrize("first_completed", [True, False])
@pytest.mark.parametrize("second_running", [True, False])
def test_旧执行编号在后续作业后重试不会重启或中断当前作业(
    clock, first_completed, second_running,
):
    driver = _driver()
    first = _start(driver, "run-a")
    clock[0] += 60 if first_completed else 10
    if not first_completed:
        assert asyncio.run(driver.execute("pump", DeviceCommand("stop"))).success
    asyncio.run(driver.read_state("pump"))
    _start(driver, "run-b")
    if not second_running:
        clock[0] += 60
    before = asyncio.run(driver.read_state("pump"))
    # 哥哥，重建驱动复用已保存状态，重试保障不能只藏在旧实例的内存里。
    rebuilt = _driver(before)

    replay = _start(rebuilt, "run-a")

    after = asyncio.run(rebuilt.read_state("pump"))
    assert after == before
    assert replay.actual_params["run_id"] == "run-a"
    assert replay.actual_params["run_started_at"] == first.actual_params["run_started_at"]
    assert replay.actual_params["run_until"] == first.actual_params["run_until"]


def test_结束历史有界且保留最近一百轮供任务对账(clock):
    driver = _driver()
    for index in range(103):
        _start(driver, f"run-{index}")
        assert asyncio.run(driver.execute("pump", DeviceCommand("stop"))).success
    state = asyncio.run(driver.read_state("pump"))
    assert len(state["run_history"]) == 100
    assert "run-0" not in state["run_history"]
    assert state["run_history"]["run-3"]["reason"] == "interrupted"
    assert state["run_history"]["run-102"]["reason"] == "interrupted"
