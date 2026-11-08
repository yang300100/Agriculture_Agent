"""哥哥，规则的冷却和每日用量要包含人工确认，并且经得起重启。"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from core.database.engine import get_session
from core.database.models import DeviceActionLog, User
from core.device_executor import DeviceExecutor
from core.device_rule_engine import RuleEngine
from devices.base import DeviceCapability, DeviceCommand
from devices.registry import DeviceDriverRegistry
from devices.simulator_driver import SimulatorDriver


@pytest.fixture
def accounting_demo(tmp_path, monkeypatch):
    import core.device_executor as executor_module
    import core.device_rule_engine as rule_module

    monkeypatch.setattr(executor_module, "DEFAULT_DATA_DIR", str(tmp_path))
    username = f"usage_{uuid.uuid4().hex}"
    with get_session() as session:
        user = User(username=username, password_hash="")
        session.add(user)
        session.commit()
        user_id = user.id
    registry = DeviceDriverRegistry()
    simulator = SimulatorDriver(simulated_latency_ms=0)
    simulator.add_virtual_device("pump", "演示水泵", [DeviceCapability.IRRIGATE])
    registry.register("simulator", simulator)
    asyncio.run(simulator.connect())
    asyncio.run(registry.discover_all())
    demo = SimpleNamespace(
        username=username, user_id=user_id, simulator=simulator,
        executor=DeviceExecutor(registry, username), engine=RuleEngine(username),
        rule_module=rule_module,
    )
    yield demo
    rule_module._global_history.pop(username, None)


def _confirm_start(demo, duration=5):
    pending_id = demo.executor.add_pending({
        "device_id": "pump", "command": "start", "params": {"duration": duration},
        "capability": "irrigate",
    })
    assert demo.executor.confirm_pending(pending_id)["success"] is True
    assert len(demo.simulator.get_history()) == 1


@pytest.mark.parametrize("restart", [False, True])
def test_人工确认后规则仍执行自身最小间隔(accounting_demo, restart):
    demo = accounting_demo
    _confirm_start(demo)
    if restart:
        demo.rule_module._global_history.pop(demo.username)
    engine = RuleEngine(demo.username)

    allowed, reason = engine._check_constraints(
        {"min_interval_minutes": 10}, {"duration": 5}, "pump", {},
    )

    assert allowed is False
    assert "不足最小间隔" in reason


@pytest.mark.parametrize("restart", [False, True])
def test_人工确认后规则每日累计在重启前后均生效(accounting_demo, restart):
    demo = accounting_demo
    _confirm_start(demo)
    if restart:
        demo.rule_module._global_history.pop(demo.username)
    engine = RuleEngine(demo.username)

    allowed, reason = engine._check_constraints(
        {"max_duration_per_day": 8}, {"duration": 5}, "pump", {},
    )

    assert allowed is False
    assert "每日上限" in reason


def test_自动成功日志与旧记账调用不会重复累计(accounting_demo):
    demo = accounting_demo
    result = demo.executor.execute_sync(
        "pump", DeviceCommand(command="start", params={"duration": 5}),
        capability="irrigate",
    )
    assert result["success"] is True
    demo.engine.record_execution("pump", {"duration": 5})

    allowed, _ = demo.engine._check_constraints(
        {"max_duration_per_day": 11}, {"duration": 5}, "pump", {},
    )

    assert allowed is True


def test_没有日志的旧记账接口仍约束规则(accounting_demo):
    demo = accounting_demo
    demo.engine.record_execution("pump", {"duration": 5})

    assert demo.engine._check_constraints(
        {"max_duration_per_day": 8}, {"duration": 5}, "pump", {},
    )[0] is False
    assert demo.engine._check_constraints(
        {"min_interval_minutes": 10}, {"duration": 5}, "pump", {},
    )[0] is False


def test_规则用量只累计当前用户设备的今日成功日志(accounting_demo):
    demo = accounting_demo
    with get_session() as session:
        other = User(username=f"other_{uuid.uuid4().hex}", password_hash="")
        session.add(other)
        session.flush()
        for user_id, device_id, success, created_at in [
            (demo.user_id, "pump", 0, datetime.now()),
            (demo.user_id, "other_pump", 1, datetime.now()),
            (other.id, "pump", 1, datetime.now()),
            (demo.user_id, "pump", 1, datetime.now() - timedelta(days=1)),
        ]:
            session.add(DeviceActionLog(
                user_id=user_id, device_id=device_id, command="start",
                params=json.dumps({"duration": 120}), success=success,
                created_at=created_at,
            ))
        session.commit()

    assert demo.engine._check_constraints(
        {"max_duration_per_day": 10}, {"duration": 5}, "pump", {},
    )[0] is True


@pytest.mark.parametrize("command", ["stop", "set_param"])
def test_成功停止与参数设置不会消耗规则启动额度或刷新冷却(accounting_demo, command):
    demo = accounting_demo
    with get_session() as session:
        session.add(DeviceActionLog(
            user_id=demo.user_id, device_id="pump", command=command,
            params=json.dumps({"duration": 120}), success=1,
            created_at=datetime.now(),
        ))
        session.commit()

    assert demo.engine._check_constraints(
        {"max_duration_per_day": 8, "min_interval_minutes": 10},
        {"duration": 5}, "pump", {},
    )[0] is True
