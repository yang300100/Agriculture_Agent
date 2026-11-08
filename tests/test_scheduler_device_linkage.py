"""哥哥，用隔离数据库验证虚拟设备的规则和周期任务联动。"""

import os
import threading
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app import scheduler_jobs
from core.device_executor import DeviceExecutor
from core.device_registry_factory import (
    close_registry, invalidate_registry_cache, load_custom_devices,
    save_custom_devices, setup_registry,
)
from core.device_rule_engine import RuleDecision, RuleEngine
from core.planting_tracker import PlantingTracker


@pytest.fixture
def linkage(monkeypatch):
    monkeypatch.setenv("AUTONOMY_LEVEL", "medium")
    username = f"linkage_{uuid.uuid4().hex}"
    tracker = PlantingTracker(os.path.join(scheduler_jobs.DEFAULT_DATA_DIR, username))
    save_custom_devices(username, [
        {
            "device_id": "soil_01", "name": "演示土壤传感器",
            "driver": "simulator", "capabilities": ["read_sensor"],
            "sensors": ["soil_moisture"],
            "initial_state": {"soil_moisture": 20},
        },
        {
            "device_id": "pump_01", "name": "演示水泵",
            "driver": "simulator", "capabilities": ["irrigate"],
            "initial_state": {"status": "powered_off", "power": False},
        },
        {
            "device_id": "pump_02", "name": "另一个水泵",
            "driver": "simulator", "capabilities": ["irrigate"],
            "initial_state": {"status": "powered_off", "power": False},
        },
    ])
    engine = RuleEngine(username)
    env = SimpleNamespace(username=username, tracker=tracker, engine=engine)
    yield env
    current_engine = RuleEngine(username)
    for rule in current_engine.list_rules():
        current_engine.delete_rule(rule["id"])
    for task in tracker.get_tasks():
        tracker.delete_task(task.id)
    invalidate_registry_cache(username)


def _rule(env, **updates):
    rule = {
        "name": "低湿度联动水泵", "enabled": True,
        "trigger": {"conditions": [
            {"type": "sensor", "field": "soil_01.soil_moisture", "op": "<", "value": 30},
        ]},
        "action": {
            "device_id": "pump_01", "capability": "irrigate",
            "command": "start", "params": {"duration": 5},
        },
        "constraints": {}, "execution_mode": "auto",
    }
    rule.update(updates)
    return env.engine.add_rule(rule)


def _task(env, **updates):
    payload = {
        "title": "演示设备任务", "device_id": "pump_01",
        "device_command": "start", "device_params": {"duration": 5},
    }
    payload.update(updates)
    return env.tracker.create_task(payload)


def _snapshot(env):
    registry, loop = setup_registry(env.username)
    try:
        loop.run_until_complete(registry.discover_all())
        state = loop.run_until_complete(registry.read_state("pump_01"))
        executor = DeviceExecutor(registry, env.username)
        return state, executor.list_pending(), executor.get_logs()
    finally:
        close_registry(loop, registry)


def _all_day_trigger():
    return {"conditions": [
        {"type": "time", "op": "between", "value": ["00:00", "23:59"]},
    ]}


def _legacy_task_marker(env):
    path = os.path.join(scheduler_jobs.DEFAULT_DATA_DIR, env.username, "planting_tasks.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("[]")


def test_传感器规则自动执行会保存设备状态及审计日志(linkage):
    rule_id = _rule(linkage)
    scheduler_jobs.check_device_rules_job()
    state, pending, logs = _snapshot(linkage)
    assert state["status"] == "running"
    assert state["last_duration"] == 5
    assert pending == []
    successful = [log for log in logs if log["success"]]
    assert len(successful) == 1
    assert str(successful[0]["rule_id"]) == rule_id
    assert successful[0]["trigger"] == "rule"


def test_连续周期不会重复创建相同规则待确认操作(linkage):
    _rule(linkage, execution_mode="confirm")
    scheduler_jobs.check_device_rules_job()
    scheduler_jobs.check_device_rules_job()
    state, pending, logs = _snapshot(linkage)
    assert state["status"] == "powered_off"
    assert len(pending) == 1
    assert len(logs) == 1


@pytest.mark.parametrize("updates", [{"enabled": False}, {"execution_mode": "notify"}])
def test_关闭或仅通知规则不改变设备状态(linkage, updates):
    _rule(linkage, **updates)
    scheduler_jobs.check_device_rules_job()
    state, pending, logs = _snapshot(linkage)
    assert state["status"] == "powered_off"
    assert pending == []
    assert not any(log["success"] for log in logs)


def test_数据库设备任务无需旧文件即可读取传感器并执行(linkage):
    _rule(linkage)
    task = _task(linkage)
    scheduler_jobs.check_task_execution_job()
    assert linkage.tracker.get_task_by_id(task.id).status == "运行中"
    state, pending, logs = _snapshot(linkage)
    assert state["status"] == "running"
    assert len([log for log in logs if log["success"]]) == 1
    scheduler_jobs.check_task_execution_job()
    assert len([log for log in _snapshot(linkage)[2] if log["success"]]) == 1


@pytest.mark.parametrize("mismatch", ["device", "command"])
def test_其他设备或指令的规则不能授权当前任务(linkage, mismatch):
    action = {
        "device_id": "pump_02" if mismatch == "device" else "pump_01",
        "capability": "irrigate", "command": "stop" if mismatch == "command" else "start",
        "params": {} if mismatch == "command" else {"duration": 5},
    }
    _rule(linkage, trigger=_all_day_trigger(), action=action)
    _legacy_task_marker(linkage)
    task = _task(linkage)
    scheduler_jobs.check_task_execution_job()
    assert linkage.tracker.get_task_by_id(task.id).status == "待办"
    assert _snapshot(linkage)[0]["status"] == "powered_off"


@pytest.mark.parametrize("mode,autonomy", [("confirm", "high"), ("notify", "high"), ("auto", "low")])
def test_设备任务尊重执行模式和低自主权(linkage, monkeypatch, mode, autonomy):
    monkeypatch.setenv("AUTONOMY_LEVEL", autonomy)
    _rule(linkage, trigger=_all_day_trigger(), execution_mode=mode)
    _legacy_task_marker(linkage)
    task = _task(linkage)
    scheduler_jobs.check_task_execution_job()
    assert linkage.tracker.get_task_by_id(task.id).status == "待办"
    assert _snapshot(linkage)[0]["status"] == "powered_off"


def test_未到开始日期的设备任务保持待办(linkage):
    _rule(linkage, trigger=_all_day_trigger())
    _legacy_task_marker(linkage)
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    task = _task(linkage, start_date=tomorrow)
    scheduler_jobs.check_task_execution_job()
    assert linkage.tracker.get_task_by_id(task.id).status == "待办"
    assert _snapshot(linkage)[0]["status"] == "powered_off"


def test_任务下发规则评估后实际参数(linkage):
    _rule(linkage, trigger=_all_day_trigger(), ai_enhance={
        "enabled": True, "can_adjust": ["duration"],
        "absolute_range": {"duration": [1, 3]},
    })
    _legacy_task_marker(linkage)
    task = _task(linkage)
    scheduler_jobs.check_task_execution_job()
    assert linkage.tracker.get_task_by_id(task.id).status == "运行中"
    assert _snapshot(linkage)[0]["last_duration"] == 3


def test_停止规则不受刚执行后的运行间隔阻挡(linkage):
    linkage.engine.record_execution("pump_01", {"duration": 5})
    rule = {
        "action": {"device_id": "pump_01", "capability": "irrigate", "command": "stop"},
        "constraints": {"min_interval_minutes": 120, "forbidden_hours": list(range(24))},
    }
    decision, _, params = linkage.engine.evaluate_action(rule, {}, {})
    assert decision == RuleDecision.AUTO_EXECUTE
    assert params == {}


@pytest.mark.parametrize("qualified", [False, True])
def test_规则不能使用其他地块的传感器数据(linkage, qualified):
    from core.database.repository.fields import FieldRepository

    fields = FieldRepository()
    first = fields.create(user_id=linkage.tracker._uid, name="干燥地块", coordinates="[]")
    second = fields.create(user_id=linkage.tracker._uid, name="湿润地块", coordinates="[]")
    configs = load_custom_devices(linkage.username)
    for config in configs:
        config["plot_id"] = first.id if config["device_id"] == "soil_01" else second.id
    configs.append({
        "device_id": "soil_02", "name": "当前地块湿润传感器",
        "driver": "simulator", "capabilities": ["read_sensor"],
        "sensors": ["soil_moisture"], "plot_id": second.id,
        "initial_state": {"soil_moisture": 80},
    })
    save_custom_devices(linkage.username, configs)
    field = "soil_01.soil_moisture" if qualified else "soil_moisture"
    _rule(linkage, trigger={"conditions": [
        {"type": "sensor", "field": field, "op": "<", "value": 30},
    ]})
    scheduler_jobs.check_device_rules_job()
    assert _snapshot(linkage)[0]["status"] == "powered_off"


def test_显式指定传感器时不被同名短字段覆盖(linkage):
    _rule(linkage, trigger={"conditions": [
        {"type": "sensor", "device_id": "soil_01", "field": "soil_moisture", "op": "<", "value": 30},
    ]})
    engine = RuleEngine(linkage.username)
    assert engine.find_matching_rules({"sensor_data": {
        "soil_moisture": 80, "soil_01.soil_moisture": 20,
    }})


def test_真实短周期调度重复启动共用实例并能停止(linkage, monkeypatch):
    from core import scheduler_service

    _rule(linkage, execution_mode="confirm")
    scheduler = scheduler_service.create_scheduler()
    scheduler.remove_all_jobs()
    completed = threading.Event()
    unexpected_cycle = threading.Event()
    count = 0

    def poll():
        nonlocal count
        scheduler_jobs.check_device_rules_job()
        count += 1
        if count == 2:
            completed.set()
        elif count > 2:
            unexpected_cycle.set()

    scheduler.add_job(poll, "interval", seconds=0.35, id="device_rules")
    monkeypatch.setattr(scheduler_service, "create_scheduler", lambda: scheduler)
    try:
        first = scheduler_service.start_scheduler()
        assert scheduler_service.start_scheduler() is first
        assert completed.wait(10)
        scheduler_service.stop_scheduler()
        assert not unexpected_cycle.wait(0.8)
        assert len(_snapshot(linkage)[1]) == 1
    finally:
        scheduler_service.stop_scheduler()


def test_已有待确认任务不会被周期调度绕过(linkage):
    _rule(linkage)
    task = _task(linkage)
    registry, loop = setup_registry(linkage.username)
    try:
        DeviceExecutor(registry, linkage.username).add_pending({
            "device_id": "pump_01", "command": "start", "params": {"duration": 5},
            "policy_context": {"task_id": task.id},
        })
    finally:
        close_registry(loop, registry)
    scheduler_jobs.check_task_execution_job()
    state, pending, logs = _snapshot(linkage)
    assert linkage.tracker.get_task_by_id(task.id).status == "待办"
    assert state["status"] == "powered_off"
    assert len(pending) == 1
    assert not any(log["success"] for log in logs)


def test_失败任务保留失败且显式重试后运行(linkage, monkeypatch):
    from core import device_executor

    monkeypatch.setattr(device_executor, "MAX_RETRIES", 1)
    _rule(linkage)
    task = _task(linkage)
    configs = load_custom_devices(linkage.username)
    save_custom_devices(linkage.username, [item for item in configs if item["device_id"] != "pump_01"])
    scheduler_jobs.check_task_execution_job()
    assert linkage.tracker.get_task_by_id(task.id).status == "失败"

    save_custom_devices(linkage.username, configs)
    scheduler_jobs.check_task_execution_job()
    assert linkage.tracker.get_task_by_id(task.id).status == "失败"
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api_routes import register_routes
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        result = client.post(f"/api/tasks/{task.id}/execute?username={linkage.username}")
        assert result.json()["success"] is True
    assert linkage.tracker.get_task_by_id(task.id).status == "运行中"
    state, pending, logs = _snapshot(linkage)
    assert state["status"] == "running"
    assert len([log for log in logs if log["success"]]) == 1
    assert any(not log["success"] for log in logs)


def test_种子规则默认关闭且显式传感器支持同地块跨分区(linkage):
    from core.database.engine import get_session
    from scripts.seed_farm_data import seed_farm_data

    with get_session() as session:
        seed_farm_data(session, linkage.username)
        session.commit()
    configs = load_custom_devices(linkage.username)
    sensor = next(item for item in configs if item["device_id"] == "wheat_soil_sensor")
    sensor["initial_state"]["soil_moisture"] = 20
    save_custom_devices(linkage.username, configs)

    scheduler_jobs.check_device_rules_job()
    assert _snapshot(linkage)[1] == []
    engine = RuleEngine(linkage.username)
    rule = next(item for item in engine.rules if item["name"] == "麦田墒情灌溉")
    engine.toggle_rule(rule["id"], True)
    scheduler_jobs.check_device_rules_job()
    pending = _snapshot(linkage)[1]
    assert len(pending) == 1
    assert pending[0]["device_id"] == "wheat_irrigation"
    assert pending[0]["command"] == "start"


def test_同地块其他分区读数不能冒充短名传感器(linkage):
    from core.database.repository.fields import FieldRepository

    field = FieldRepository().create(
        user_id=linkage.tracker._uid, name="分区测试地块", coordinates="[]"
    )
    configs = load_custom_devices(linkage.username)
    for config in configs:
        config["plot_id"] = field.id
        config["zone_id"] = "west" if config["device_id"] == "soil_01" else "east"
    save_custom_devices(linkage.username, configs)
    _rule(linkage, trigger={"conditions": [
        {"type": "sensor", "field": "soil_moisture", "op": "<", "value": 30},
    ]})
    scheduler_jobs.check_device_rules_job()
    assert _snapshot(linkage)[0]["status"] == "powered_off"


@pytest.fixture
def rule_clock(monkeypatch):
    current = [datetime.now().timestamp()]

    class RuleDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(current[0], tz=tz)

    # 哥哥，让设备计时与规则冷却一起前进，不等待真实的十分钟。
    for module_name in (
        "devices.simulator_driver", "core.device_rule_engine", "core.device_safety_policy",
    ):
        monkeypatch.setattr(f"{module_name}.datetime", RuleDatetime)
    return current


def test_持续命中周期规则不会延长正在运行设备的截止时间(linkage, rule_clock):
    _rule(linkage, trigger=_all_day_trigger(), action={
        "device_id": "pump_01", "capability": "irrigate", "command": "start",
        "params": {"duration": 10},
    })
    scheduler_jobs.check_device_rules_job()
    started, _, _ = _snapshot(linkage)
    deadline = started["run_until"]

    rule_clock[0] += 300
    scheduler_jobs.check_device_rules_job()

    state, _, logs = _snapshot(linkage)
    assert state["run_until"] == deadline
    assert len([log for log in logs if log["success"]]) == 1
    rule_clock[0] += 300
    assert _snapshot(linkage)[0]["status"] == "standby"


def test_同轮多条自动规则不会重复启动同一设备(linkage, monkeypatch):
    from core.device_safety_policy import ABSOLUTE_CEILINGS

    _rule(linkage, trigger=_all_day_trigger())
    _rule(linkage, name="同轮第二条规则", trigger=_all_day_trigger())
    # 哥哥，暂时放开冷却，才能单独验证本轮使用了执行后的最新状态。
    monkeypatch.setitem(ABSOLUTE_CEILINGS["irrigate"], "min_interval_seconds", 0)

    scheduler_jobs.check_device_rules_job()

    assert len([log for log in _snapshot(linkage)[2] if log["success"]]) == 1


@pytest.mark.parametrize("command,params,status,flow", [
    ("stop", {}, "standby", 0),
    ("set_param", {"flow_rate": 9}, "running", 9),
])
def test_运行状态保护仍允许停止和设置参数(linkage, command, params, status, flow):
    configs = load_custom_devices(linkage.username)
    pump = next(item for item in configs if item["device_id"] == "pump_01")
    pump["initial_state"] = {
        "status": "running", "power": True, "flow_rate": 3,
        "run_until": datetime.now().timestamp() + 600,
    }
    save_custom_devices(linkage.username, configs)
    _rule(linkage, trigger=_all_day_trigger(), action={
        "device_id": "pump_01", "capability": "irrigate", "command": command,
        "params": params,
    })

    scheduler_jobs.check_device_rules_job()

    state, _, logs = _snapshot(linkage)
    assert state["status"] == status
    assert state["flow_rate"] == flow
    assert len([log for log in logs if log["success"]]) == 1
