"""哥哥，从任务 API 验证模拟设备和数据库中的真实执行结果。"""

import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api_routes import register_routes
from core.device_registry_factory import invalidate_registry_cache, load_custom_devices
from devices.simulator_driver import SimulatorDriver


@pytest.fixture
def task_client(monkeypatch):
    monkeypatch.setenv("AUTONOMY_LEVEL", "medium")
    username = f"task_device_{uuid.uuid4().hex}"
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        yield client, username
    invalidate_registry_cache(username)


def _create_pump(client, username):
    response = client.post(f"/api/devices?username={username}", json={
        "device_id": "pump_01", "name": "演示水泵", "driver": "simulator",
        "capabilities": ["irrigate"],
        "initial_state": {"power": False, "status": "powered_off"},
    })
    assert response.json()["success"] is True


def _create_task(client, username, command="start", duration=5):
    response = client.post(f"/api/tasks?username={username}", json={
        "crop": "小麦", "title": f"水泵{command}", "device_id": "pump_01",
        "device_command": command, "device_params": {"duration": duration},
    })
    assert response.json()["success"] is True
    tasks = client.get(f"/api/tasks?username={username}").json()
    return next(task for task in tasks if task["title"] == f"水泵{command}")


def _task(client, username, task_id):
    tasks = client.get(f"/api/tasks?username={username}").json()
    return next(task for task in tasks if task["id"] == task_id)


def _execute(client, username, task_id):
    return client.post(f"/api/tasks/{task_id}/execute?username={username}").json()


def test_任务执行后设备和数据库一致且重复执行不再下发(task_client):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)

    result = _execute(client, username, task["id"])

    assert result["success"] is True
    saved = _task(client, username, task["id"])
    assert saved["status"] == "运行中"
    assert saved["progress"] < 100
    state = client.get(f"/api/devices/pump_01/state?username={username}").json()
    assert state["status"] == "running"
    assert state["last_duration"] == 5
    assert load_custom_devices(username)[0]["initial_state"]["status"] == "running"
    assert _execute(client, username, task["id"])["success"] is False
    logs = client.get(f"/api/actions/log?username={username}").json()
    assert len([log for log in logs if log["success"]]) == 1


@pytest.mark.parametrize("stored_status", [None, ""])
def test_旧任务空状态仍可按待办执行(task_client, stored_status):
    from core.database.repository.planting import PlantingTaskRepository

    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    PlantingTaskRepository().update(int(task["id"]), status=stored_status)

    result = _execute(client, username, task["id"])

    assert result["success"] is True, result
    assert _task(client, username, task["id"])["status"] == "运行中"


def test_设备缺失执行失败后任务可重试(task_client):
    client, username = task_client
    task = _create_task(client, username)

    assert _execute(client, username, task["id"])["success"] is False
    assert _task(client, username, task["id"])["status"] == "失败"
    assert _task(client, username, task["id"])["progress"] == 0
    _create_pump(client, username)

    assert _execute(client, username, task["id"])["success"] is True
    assert _task(client, username, task["id"])["status"] == "运行中"


def test_低自主模式停止任务仍直接生效(task_client, monkeypatch):
    client, username = task_client
    _create_pump(client, username)
    start = _create_task(client, username)
    assert _execute(client, username, start["id"])["success"] is True
    monkeypatch.setenv("AUTONOMY_LEVEL", "low")
    stop = _create_task(client, username, command="stop")

    assert _execute(client, username, stop["id"])["success"] is True

    state = client.get(f"/api/devices/pump_01/state?username={username}").json()
    assert state["status"] == "standby"
    assert state["power"] is True
    assert _task(client, username, stop["id"])["status"] == "已完成"


def test_任务需要确认时返回队列编号且重复点击复用同一操作(task_client, monkeypatch):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    monkeypatch.setenv("AUTONOMY_LEVEL", "low")

    first = _execute(client, username, task["id"])

    assert first.get("decision") == "need_confirm", first
    assert first["pending_id"]
    assert "error" not in first
    second = _execute(client, username, task["id"])
    assert second["pending_id"] == first["pending_id"]
    pending = client.get(f"/api/actions/pending?username={username}").json()
    assert len(pending) == 1
    assert pending[0]["policy_context"]["task_id"] == task["id"]
    assert _task(client, username, task["id"])["status"] == "待办"
    assert load_custom_devices(username)[0]["initial_state"]["status"] == "powered_off"


def test_物理上限拒绝时保留决策且不产生待确认(task_client):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username, duration=121)

    result = _execute(client, username, task["id"])

    assert result.get("decision") == "rejected", result
    assert result["success"] is False
    assert client.get(f"/api/actions/pending?username={username}").json() == []
    assert _task(client, username, task["id"])["status"] == "待办"
    assert load_custom_devices(username)[0]["initial_state"]["status"] == "powered_off"


def test_任务确认成功后同步为运行中(task_client, monkeypatch):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    monkeypatch.setenv("AUTONOMY_LEVEL", "low")
    _execute(client, username, task["id"])
    pending = client.get(f"/api/actions/pending?username={username}").json()[0]

    result = client.post(
        f"/api/actions/{pending['id']}/confirm?username={username}"
    ).json()

    assert result["success"] is True
    assert _task(client, username, task["id"])["status"] == "运行中"
    assert _task(client, username, task["id"])["progress"] < 100
    assert load_custom_devices(username)[0]["initial_state"]["status"] == "running"


def _interrupt_confirmation(username, task_id):
    from core.device_executor import DeviceExecutor
    from core.planting_tracker import PlantingTracker
    from core.storage_paths import DEFAULT_DATA_DIR
    from devices.registry import DeviceDriverRegistry
    from pathlib import Path

    tracker = PlantingTracker(str(Path(DEFAULT_DATA_DIR) / username))
    assert tracker.claim_device_task(task_id)
    executor = DeviceExecutor(DeviceDriverRegistry(), username=username)
    with executor._pending_transaction():
        executor.pending_actions[0]["status"] = "executing"
        executor._save_pending()


def test_确认进程中断结果未知时不得再次下发(task_client, monkeypatch):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    monkeypatch.setenv("AUTONOMY_LEVEL", "low")
    pending_id = _execute(client, username, task["id"])["pending_id"]
    _interrupt_confirmation(username, task["id"])
    pending = client.get(f"/api/actions/pending?username={username}").json()[0]
    assert pending["status"] == "uncertain"

    result = client.post(f"/api/actions/{pending_id}/confirm?username={username}")

    assert result.status_code == 409
    state = client.get(f"/api/devices/pump_01/state?username={username}").json()
    assert state["status"] == "powered_off"
    logs = client.get(f"/api/actions/log?username={username}").json()
    assert not [log for log in logs if log["success"]]
    reconcile = client.post(f"/api/devices/pump_01/reconcile?username={username}")
    assert reconcile.status_code == 409
    assert "记录" in reconcile.json()["detail"]



def test_旧中断操作不能覆盖后来领取的任务(task_client, monkeypatch):
    from datetime import datetime, timedelta
    from core.database.repository.planting import PlantingTaskRepository

    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    monkeypatch.setenv("AUTONOMY_LEVEL", "low")
    pending_id = _execute(client, username, task["id"])["pending_id"]
    _interrupt_confirmation(username, task["id"])
    client.get(f"/api/actions/pending?username={username}")
    PlantingTaskRepository().update(int(task["id"]), updated_at=datetime.now() + timedelta(seconds=1))

    result = client.post(f"/api/actions/{pending_id}/confirm?username={username}")

    assert result.status_code == 409
    assert _task(client, username, task["id"])["status"] == "进行中"
    assert load_custom_devices(username)[0]["initial_state"]["status"] == "powered_off"


def test_任务确认失败后可在同一队列操作上重试(task_client, monkeypatch):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    monkeypatch.setenv("AUTONOMY_LEVEL", "low")
    _execute(client, username, task["id"])
    pending = client.get(f"/api/actions/pending?username={username}").json()[0]
    assert client.delete(f"/api/devices/pump_01?username={username}").json()["success"]

    result = client.post(
        f"/api/actions/{pending['id']}/confirm?username={username}"
    ).json()

    assert result["success"] is False
    assert result["action_status"] == "failed"
    assert _task(client, username, task["id"])["status"] == "失败"
    _create_pump(client, username)
    retry = client.post(
        f"/api/actions/{pending['id']}/confirm?username={username}"
    ).json()
    assert retry["success"] is True
    assert _task(client, username, task["id"])["status"] == "运行中"


def test_拒绝任务设备操作后仍可重新申请执行(task_client, monkeypatch):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    monkeypatch.setenv("AUTONOMY_LEVEL", "low")
    pending_id = _execute(client, username, task["id"])["pending_id"]

    rejected = client.post(
        f"/api/actions/{pending_id}/reject?username={username}"
    ).json()

    assert rejected["success"] is True
    assert _task(client, username, task["id"])["status"] == "待办"
    assert client.get(f"/api/actions/pending?username={username}").json() == []
    assert load_custom_devices(username)[0]["initial_state"]["status"] == "powered_off"
    requested = _execute(client, username, task["id"])
    assert requested["decision"] == "need_confirm"
    assert requested["pending_id"] != pending_id


def test_任务已手动完成后旧确认操作不能再次启动设备(task_client, monkeypatch):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    monkeypatch.setenv("AUTONOMY_LEVEL", "low")
    pending_id = _execute(client, username, task["id"])["pending_id"]
    assert client.post(
        f"/api/tasks/{task['id']}/complete?username={username}"
    ).json()["success"] is True

    result = client.post(f"/api/actions/{pending_id}/confirm?username={username}")

    assert result.status_code == 409
    assert _task(client, username, task["id"])["status"] == "已完成"
    assert load_custom_devices(username)[0]["initial_state"]["status"] == "powered_off"


def test_任务执行并发请求只下发一次(task_client, monkeypatch):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    started = threading.Event()
    release = threading.Event()
    original_execute = SimulatorDriver.execute

    async def paused_execute(driver, device_id, command):
        started.set()
        assert release.wait(timeout=8)
        return await original_execute(driver, device_id, command)

    monkeypatch.setattr(SimulatorDriver, "execute", paused_execute)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(_execute, client, username, task["id"])
        try:
            assert started.wait(timeout=8)
            assert _task(client, username, task["id"]).get("retry_available", False) is False
            second = _execute(client, username, task["id"])
        finally:
            release.set()
        assert first.result(timeout=8)["success"] is True

    assert second["success"] is False
    assert _task(client, username, task["id"])["status"] == "运行中"
    logs = client.get(f"/api/actions/log?username={username}").json()
    assert len([log for log in logs if log["success"]]) == 1


@pytest.mark.parametrize("operation", ["confirm", "reject"])
def test_兼容缺少任务上下文的旧待确认操作(task_client, operation):
    from core.device_executor import DeviceExecutor
    from devices.registry import DeviceDriverRegistry

    client, username = task_client
    _create_pump(client, username)
    executor = DeviceExecutor(DeviceDriverRegistry(), username=username)
    pending_id = executor.add_pending({
        "device_id": "pump_01", "command": "start", "params": {"duration": 5},
        "policy_context": None,
    })

    response = client.post(f"/api/actions/{pending_id}/{operation}?username={username}")

    assert response.status_code == 200, response.json()
    assert response.json()["success"] is True


def test_到期设备确实结束后任务跨注册中心重建完成(task_client, monkeypatch):
    import devices.simulator_driver as simulator
    from datetime import datetime, timedelta

    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username, duration=1)
    assert _execute(client, username, task["id"])["success"]
    saved = _task(client, username, task["id"])
    assert saved["status"] == "运行中"
    assert saved["started_at"] and saved["run_until"]
    future = datetime.now() + timedelta(minutes=2)

    class FutureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return future

    invalidate_registry_cache(username)
    monkeypatch.setattr(simulator, "datetime", FutureClock)
    finished = _task(client, username, task["id"])
    assert finished["status"] == "已完成"
    assert finished["progress"] == 100
    assert load_custom_devices(username)[0]["initial_state"]["status"] == "standby"
    assert _execute(client, username, task["id"])["success"] is False


def test_提前停止与故障不伪装成完成(task_client):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    assert _execute(client, username, task["id"])["success"]
    stop = _create_task(client, username, command="stop")
    assert _execute(client, username, stop["id"])["success"]
    assert _task(client, username, task["id"])["status"] == "已中止"


def test_中止后的任务可显式重试但不会被自动重启(task_client):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    assert _execute(client, username, task["id"])["success"]
    first_run = load_custom_devices(username)[0]["initial_state"]["run_id"]
    stop = _create_task(client, username, command="stop")
    assert _execute(client, username, stop["id"])["success"]
    assert _task(client, username, task["id"])["status"] == "已中止"
    from core.planting_tracker import PlantingTracker
    from core.storage_paths import DEFAULT_DATA_DIR
    from pathlib import Path
    tracker = PlantingTracker(str(Path(DEFAULT_DATA_DIR) / username))
    assert not tracker.get_pending_device_tasks()
    # 完成领取即可证明中止任务不再被误报为仍在执行。
    assert tracker.claim_device_task(task["id"]) is True
    saved = tracker.get_task_by_id(task["id"])
    assert saved.execution_state["run_id"] != first_run


def test_领取后下发前中断可显式恢复且沿用执行编号(task_client):
    from core.planting_tracker import PlantingTracker
    from core.storage_paths import DEFAULT_DATA_DIR
    from pathlib import Path

    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    tracker = PlantingTracker(str(Path(DEFAULT_DATA_DIR) / username))
    assert tracker.claim_device_task(task["id"])
    run_id = tracker.get_task_by_id(task["id"]).execution_state["run_id"]
    assert _task(client, username, task["id"])["retry_available"] is True
    result = _execute(client, username, task["id"])
    assert result["success"] is True
    state = client.get(f"/api/devices/pump_01/state?username={username}").json()
    assert state["run_id"] == run_id
    assert _task(client, username, task["id"])["status"] == "运行中"


def test_领取后确认前中断可从原确认操作恢复(task_client, monkeypatch):
    from core.planting_tracker import PlantingTracker
    from core.storage_paths import DEFAULT_DATA_DIR
    from pathlib import Path

    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    monkeypatch.setenv("AUTONOMY_LEVEL", "low")
    pending = _execute(client, username, task["id"])
    tracker = PlantingTracker(str(Path(DEFAULT_DATA_DIR) / username))
    assert tracker.claim_device_task(task["id"])
    run_id = tracker.get_task_by_id(task["id"]).execution_state["run_id"]
    response = client.post(f"/api/actions/{pending['pending_id']}/confirm?username={username}")
    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    state = client.get(f"/api/devices/pump_01/state?username={username}").json()
    assert state["run_id"] == run_id


def test_多轮设备作业后仍按原运行记录结算任务(task_client, monkeypatch):
    from datetime import datetime, timedelta
    from core.device_registry_factory import RegistrySession
    from devices.base import DeviceCommand
    import devices.simulator_driver as simulator

    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username, duration=1)
    assert _execute(client, username, task["id"])["success"]
    run_id = load_custom_devices(username)[0]["initial_state"]["run_id"]
    current = [datetime.now() + timedelta(minutes=2)]
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return current[0]
    monkeypatch.setattr(simulator, "datetime", Clock)
    with RegistrySession(username) as (registry, loop):
        loop.run_until_complete(registry.discover_all())
        state = loop.run_until_complete(registry.read_state("pump_01"))
        original_end = state["run_history"][run_id]["ended_at"]
        result = loop.run_until_complete(registry.execute("pump_01", DeviceCommand(command="start", params={"duration": 1})))
        assert result.success
        current[0] += timedelta(minutes=2)
        state = loop.run_until_complete(registry.read_state("pump_01"))
        assert state["last_ended_run_id"] != run_id
    saved = _task(client, username, task["id"])
    assert saved["status"] == "已完成"
    assert saved["progress"] == 100
    assert saved["ended_at"] == original_end


def test_预期时间过了但设备仍运行不可完成(task_client):
    from core.database.repository.planting import PlantingTaskRepository
    import json

    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    assert _execute(client, username, task["id"])["success"]
    row = PlantingTaskRepository().get_by_id(int(task["id"]))
    runtime = json.loads(row.execution_state)
    runtime["run_until"] = 1
    PlantingTaskRepository().update(row.id, execution_state=json.dumps(runtime))
    assert _task(client, username, task["id"])["status"] == "运行中"


def test_运行中的设备任务不能勾选为已完成(task_client):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    assert _execute(client, username, task["id"])["success"]
    response = client.post(f"/api/tasks/{task['id']}/complete?username={username}")
    assert response.status_code == 409


def test_运行中的设备任务不能删除丢失作业记录(task_client):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    assert _execute(client, username, task["id"])["success"]
    response = client.delete(f"/api/tasks/{task['id']}?username={username}")
    assert response.status_code == 409
    assert _task(client, username, task["id"])["status"] == "运行中"
    assert _task(client, username, task["id"])["status"] == "运行中"


def test_任务编辑保存日期和内容但不改运行参数(task_client):
    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    assert _execute(client, username, task["id"])["success"]
    response = client.put(f"/api/tasks/{task['id']}?username={username}", json={
        "title": "麦田灌溉", "crop": "玉米", "task_type": "浇水",
        "description": "调整作业安排", "priority": "high",
        "start_date": "2026-11-08", "end_date": "2026-11-09",
    })
    assert response.status_code == 200, response.json()
    assert response.json()["success"]
    updated = _task(client, username, task["id"])
    assert updated["title"] == "麦田灌溉"
    assert updated["task_type"] == "浇水"
    assert updated["start_date"] == "2026-11-08"
    assert updated["end_date"] == "2026-11-09"
    assert updated["status"] == "运行中"
    assert updated["device_params"] == {"duration": 5}
    assert load_custom_devices(username)[0]["initial_state"]["last_duration"] == 5


@pytest.mark.parametrize("payload", [
    {"start_date": "2026-02-30"}, {"end_date": "明天"},
    {"start_date": "2026-11-10", "end_date": "2026-11-09"},
    {"priority": "invalid"}, {"status": "已完成"},
    {"device_params": {"duration": 30}},
])
def test_任务编辑拒绝非法日期和越权字段(task_client, payload):
    client, username = task_client
    task = _create_task(client, username)
    response = client.put(f"/api/tasks/{task['id']}?username={username}", json=payload)
    assert response.status_code == 422
    saved = _task(client, username, task["id"])
    assert saved["title"] == task["title"]
    assert saved["status"] == "待办"


def test_任务编辑不能修改其他用户任务(task_client):
    client, username = task_client
    task = _create_task(client, username)
    response = client.put(f"/api/tasks/{task['id']}?username=another_user", json={"title": "改错了"})
    assert response.status_code == 404
    assert _task(client, username, task["id"])["title"] == task["title"]


def test_确认已启动后进程中断重试不能再启动设备(task_client, monkeypatch):
    from core.device_registry_factory import RegistrySession
    from core.database.repository.planting import PlantingTaskRepository
    from devices.base import DeviceCommand
    import json

    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    monkeypatch.setenv("AUTONOMY_LEVEL", "low")
    pending_id = _execute(client, username, task["id"])["pending_id"]
    _interrupt_confirmation(username, task["id"])
    row = PlantingTaskRepository().get_by_id(int(task["id"]))
    run_id = json.loads(row.execution_state)["run_id"]
    with RegistrySession(username) as (registry, loop):
        loop.run_until_complete(registry.discover_all())
        result = loop.run_until_complete(registry.execute("pump_01", DeviceCommand(
            command="start", params={"duration": 5}, execution_id=run_id,
        )))
        assert result.success
    original_run = load_custom_devices(username)[0]["initial_state"]["run_id"]
    invalidate_registry_cache(username)
    response = client.post(f"/api/actions/{pending_id}/confirm?username={username}")
    assert response.status_code == 409
    assert load_custom_devices(username)[0]["initial_state"]["run_id"] == original_run
    assert _task(client, username, task["id"])["status"] == "运行中"


@pytest.mark.parametrize("device_status,expected", [("error", "失败"), ("offline", "运行中"), ("unknown", "运行中")])
def test_读取故障与未知状态分别处理(task_client, device_status, expected):
    from core.device_registry_factory import _save_simulator_state

    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    assert _execute(client, username, task["id"])["success"]
    _save_simulator_state(username, "pump_01", {"status": device_status})
    assert _task(client, username, task["id"])["status"] == expected


def test_上一轮完成证据不能完成本轮任务(task_client):
    from core.device_registry_factory import _save_simulator_state

    client, username = task_client
    _create_pump(client, username)
    _save_simulator_state(username, "pump_01", {
        "last_ended_run_id": "older-run", "last_run_end_reason": "completed",
    })
    task = _create_task(client, username)
    assert _execute(client, username, task["id"])["success"]
    assert _task(client, username, task["id"])["status"] == "运行中"


def test_设备返回故障状态和错误详情时任务仍标记失败(task_client):
    from core.device_registry_factory import _save_simulator_state

    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username)
    assert _execute(client, username, task["id"])["success"]
    _save_simulator_state(username, "pump_01", {"status": "error", "error": "电机故障"})
    assert _task(client, username, task["id"])["status"] == "失败"


def test_任务调度启动后保持运行并可定时完成(task_client, monkeypatch):
    from app.scheduler_jobs import check_task_execution_job
    from core.device_rule_engine import RuleEngine
    from datetime import datetime, timedelta
    import devices.simulator_driver as simulator

    client, username = task_client
    _create_pump(client, username)
    task = _create_task(client, username, duration=1)
    RuleEngine(username=username).add_rule({
        "name": "麦田定时灌溉", "enabled": True,
        "trigger": {"conditions": [{"field": "temperature", "op": ">", "value": 0}]},
        "action": {"device_id": "pump_01", "command": "start", "params": {"duration": 1}},
    })
    from app import scheduler_jobs
    collect = scheduler_jobs._collect_device_rule_context

    def context_with_temperature(*args):
        return {**collect(*args), "temperature": 20}

    monkeypatch.setattr(scheduler_jobs, "_collect_device_rule_context", context_with_temperature)
    check_task_execution_job()
    assert _task(client, username, task["id"])["status"] == "运行中"
    current_run = load_custom_devices(username)[0]["initial_state"]["run_id"]
    check_task_execution_job()
    assert load_custom_devices(username)[0]["initial_state"]["run_id"] == current_run
    future = datetime.now() + timedelta(minutes=2)

    class FutureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return future

    monkeypatch.setattr(simulator, "datetime", FutureClock)
    check_task_execution_job()
    assert _task(client, username, task["id"])["status"] == "已完成"
