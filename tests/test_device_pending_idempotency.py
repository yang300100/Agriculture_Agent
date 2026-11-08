"""哥哥，规则和任务的待确认操作要跨生产者原子去重。"""

import asyncio
import copy
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from core.device_executor import DeviceExecutor
from core.device_rule_engine import RuleEngine
from devices.base import DeviceCapability, DeviceCommand
from devices.registry import DeviceDriverRegistry
from devices.simulator_driver import SimulatorDriver


@pytest.fixture
def queue_pair(tmp_path, monkeypatch):
    import core.device_executor as executor_module
    from core.database.engine import init_db
    from core.database.repository.users import UserRepository

    init_db()
    monkeypatch.setattr(executor_module, "DEFAULT_DATA_DIR", str(tmp_path))
    username = f"pending_{uuid.uuid4().hex}"
    UserRepository().create(username=username, password_hash="")
    registry = DeviceDriverRegistry()
    simulator = SimulatorDriver(simulated_latency_ms=0)
    simulator.add_virtual_device("pump", "演示水泵", [DeviceCapability.IRRIGATE])
    registry.register("simulator", simulator)
    asyncio.run(simulator.connect())
    asyncio.run(registry.discover_all())
    rule_id = RuleEngine(username).add_rule({
        "name": "待确认去重规则", "trigger": {"conditions": []},
        "action": {
            "device_id": "pump", "capability": "irrigate", "command": "stop",
        },
    })
    return SimpleNamespace(
        first=DeviceExecutor(registry, username),
        second=DeviceExecutor(registry, username),
        simulator=simulator, rule_id=rule_id,
    )


def _action(queue_pair):
    return {
        "device_id": "pump", "command": "stop", "params": {},
        "rule_id": queue_pair.rule_id, "capability": "irrigate",
    }


def test_确认后的执行日志保留原触发规则(queue_pair):
    action_id = queue_pair.first.add_pending(_action(queue_pair))
    result = queue_pair.first.confirm_pending(action_id)
    assert result["success"]
    log = queue_pair.first.get_logs(limit=1)[0]
    assert log["trigger"] == "confirmed"
    assert str(log["rule_id"]) == str(queue_pair.rule_id)


def test_两实例并发记录同一规则只产生一个待确认操作(queue_pair):
    barrier = threading.Barrier(2)

    def record(executor, trigger):
        barrier.wait(timeout=5)
        return executor.record_decision(
            "pump", DeviceCommand(command="stop"),
            decision="need_confirm", reason="等待哥哥确认",
            trigger=trigger, rule_id=queue_pair.rule_id, add_pending=True,
            capability="irrigate",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(record, queue_pair.first, "rule")
        second = pool.submit(record, queue_pair.second, "autonomous")
        results = [first.result(timeout=10), second.result(timeout=10)]
    assert len({result["pending_id"] for result in results}) == 1
    assert len(queue_pair.first.list_pending()) == 1


def test_正在确认的规则被其他生产者再次触发时不会新增下发(queue_pair, monkeypatch):
    action = _action(queue_pair)
    original_id = queue_pair.first.add_pending(action)
    entered = threading.Event()
    resume = threading.Event()
    original_execute = queue_pair.simulator.execute

    async def delayed_execute(device_id, command):
        entered.set()
        assert resume.wait(timeout=10)
        return await original_execute(device_id, command)

    monkeypatch.setattr(queue_pair.simulator, "execute", delayed_execute)
    with ThreadPoolExecutor(max_workers=1) as pool:
        confirmation = pool.submit(queue_pair.first.confirm_pending, original_id)
        assert entered.wait(timeout=10)
        try:
            repeated_id = queue_pair.second.add_pending(action)
            assert repeated_id == original_id
            assert queue_pair.second.confirm_pending(repeated_id)["success"] is False
        finally:
            resume.set()
        assert confirmation.result(timeout=10)["success"] is True
    assert len(queue_pair.simulator.get_history()) == 1


def test_失败规则操作保留原编号供人工重试(queue_pair, monkeypatch):
    import core.device_executor as executor_module

    monkeypatch.setattr(executor_module, "MAX_RETRIES", 1)
    action = {**_action(queue_pair), "device_id": "missing_pump"}
    first_id = queue_pair.first.add_pending(action)
    assert queue_pair.first.confirm_pending(first_id)["action_status"] == "failed"
    assert queue_pair.second.add_pending(action) == first_id
    pending = queue_pair.second.list_pending()
    assert len(pending) == 1
    assert pending[0]["status"] == "failed"


@pytest.mark.parametrize("terminal", ["executed", "rejected"])
def test_终态之后同一规则新事件可以重新入队(queue_pair, terminal):
    action = _action(queue_pair)
    first_id = queue_pair.first.add_pending(action)
    if terminal == "executed":
        assert queue_pair.first.confirm_pending(first_id)["success"] is True
    else:
        assert queue_pair.first.reject_pending(first_id) is True
    second_id = queue_pair.second.add_pending(action)
    assert second_id != first_id
    assert len(queue_pair.second.list_pending()) == 1


@pytest.mark.parametrize("field,value", [
    ("rule_id", "different_rule"), ("device_id", "other_pump"),
    ("command", "start"), ("params", {"duration": 5}),
    ("policy_context", {"task_id": "different_task"}),
])
def test_规则设备指令参数或任务不同的操作分别保留(queue_pair, field, value):
    action = {**_action(queue_pair), "policy_context": {"task_id": "original_task"}}
    first_id = queue_pair.first.add_pending(action)
    different = copy.deepcopy(action)
    different[field] = value
    assert queue_pair.second.add_pending(different) != first_id
    assert len(queue_pair.second.list_pending()) == 2


def test_只有任务编号的重复操作也复用原编号(queue_pair):
    action = _action(queue_pair)
    action.pop("rule_id")
    action["policy_context"] = {"task_id": "7"}
    first_id = queue_pair.first.add_pending(action)
    action["policy_context"] = {"task_id": 7, "sensor_data": {"soil_moisture": 20}}
    assert queue_pair.second.add_pending(action) == first_id


def test_无规则或任务编号的手动操作仍分别保留(queue_pair):
    action = _action(queue_pair)
    action.pop("rule_id")
    first_id = queue_pair.first.add_pending(action)
    assert queue_pair.second.add_pending(action) != first_id
    assert len(queue_pair.second.list_pending()) == 2
