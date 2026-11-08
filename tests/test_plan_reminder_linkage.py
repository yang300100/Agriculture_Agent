"""哥哥，整季进度、未来计划和关联提醒要表达同一份农事安排。"""

import uuid
from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.api_routes import register_routes
from core.database.migrations import apply_migrations
from core.planting_planner import CropDatabase, PlantingPlanner
from core.planting_tracker import PlantingTracker
from core.reminder_scheduler import ReminderScheduler
from core.reminder_system import ReminderStorage, ReminderSystem


@pytest.fixture
def linked(tmp_path):
    username = f"linked_{uuid.uuid4().hex}"
    return PlantingTracker(str(tmp_path / username)), ReminderSystem(username=username), username


def _reminder(system, task=None, **changes):
    values = {"crop": "小麦", "reminder_type": "浇水", "frequency": "单次", "start_date": "2099-01-02", "time_of_day": "15:30"}
    if task:
        values["task_id"] = task.id
    values.update(changes)
    return system.create_reminder(values)


def test_未来计划保持待开始且不能提前推进(linked):
    tracker, _, _ = linked
    progress = tracker.create_progress({"crop": "小麦", "start_date": "2099-01-01", "total_stages": 5})
    assert progress.status == "待开始"
    assert tracker.advance_to_next_stage(progress.id)["success"] is False
    card = tracker.get_progress_cards()[0]
    assert card["status"] == "待开始"
    assert card["overall_progress"] == card["stage_progress"] == 0


def test_阶段推进不让整季进度倒退(linked):
    tracker, _, _ = linked
    start = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=80)).strftime("%Y-%m-%d")
    progress = tracker.create_progress({"crop": "未知试验作物", "start_date": start, "expected_end_date": end, "total_stages": 8, "progress_percent": 85})
    before = tracker.get_progress_cards()[0]
    assert before["overall_progress"] == 20
    tracker.advance_to_next_stage(progress.id)
    after = tracker.get_progress_cards()[0]
    assert after["overall_progress"] == 20
    assert after["stage_progress"] == after["progress"] == 0


def test_按日期推算的阶段文案不声称检测真实生长(linked):
    tracker, _, _ = linked
    tracker.create_progress({"crop": "小麦", "start_date": datetime.now().strftime("%Y-%m-%d")})
    assert "预计阶段" in tracker.get_progress_cards()[0]["stage"]


@pytest.mark.parametrize("stored_total", [1, 8])
@pytest.mark.parametrize("seconds_after_creation", [0, 11])
def test_推进基于卡片有效阶段并在重载后持续保留(linked, monkeypatch, stored_total, seconds_after_creation):
    import core.planting_tracker as module

    class Clock(datetime):
        current = datetime(2030, 1, 21, 12, 0)

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(module, "datetime", Clock)
    tracker, _, _ = linked
    progress = tracker.create_progress({
        "crop": "未知试验作物", "start_date": "2030-01-01",
        "stage_number": 1, "total_stages": stored_total,
    })
    tracker.update_progress(progress.id, {})
    Clock.current += timedelta(seconds=seconds_after_creation)
    before = tracker.get_progress_cards()[0]
    assert before["stage_number"] == 3
    assert before["total_stages"] == 8

    result = tracker.advance_to_next_stage(progress.id)
    assert result["is_completed"] is False
    assert result["stage_number"] == 4
    Clock.current += timedelta(seconds=11)
    reloaded = PlantingTracker(tracker._username)
    card = reloaded.get_progress_cards()[0]
    assert card["stage_number"] == 4
    assert card["stage_progress"] == 0
    assert card["total_stages"] == 8
    assert reloaded._load_progress_by_id(progress.id).stage_number == 4


def test_预计全季结束时推进确认完成有效末阶段(linked):
    tracker, _, _ = linked
    progress = tracker.create_progress({"crop": "未知试验作物", "start_date": "2020-01-01", "stage_number": 1, "total_stages": 1})
    card = tracker.get_progress_cards()[0]
    assert card["stage_number"] == 8
    assert card["overall_progress"] == 100
    result = tracker.advance_to_next_stage(progress.id)
    assert result["is_completed"] is True
    assert result["stage_number"] == 8
    saved = tracker._load_progress_by_id(progress.id)
    assert saved.status == "已完成"
    assert saved.stage_number == saved.total_stages == 8
    assert saved.actual_end_date


@pytest.mark.parametrize("status,stage_number,stage_percent,overall", [("进行中", 4, 0, 20), ("已完成", 8, 100, 100)])
def test_自动计算返回实际生效阶段而不覆盖人工确认(linked, status, stage_number, stage_percent, overall):
    tracker, _, _ = linked
    tracker.create_progress({
        "crop": "未知试验作物", "start_date": (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d"),
        "expected_end_date": (datetime.now() + timedelta(days=80)).strftime("%Y-%m-%d"),
        "stage": "人工确认阶段", "stage_number": stage_number, "total_stages": 8,
        "progress_percent": stage_percent, "status": status,
    })
    progress = tracker.get_progress()[0]
    result = tracker.auto_calculate_progress(progress.id)
    assert result["stage_number"] == stage_number
    assert result["stage_name"] == "人工确认阶段"
    assert result["progress_percent"] == result["stage_progress"] == stage_percent
    assert result["overall_progress"] == overall
    assert result["status"] == status


def test_仪表盘和聊天展示同一份有效阶段与整季进度(linked, monkeypatch):
    from app.agent.nodes.progress import progress_tracking_node
    from app.agent.state import AgentState

    monkeypatch.setattr("core.weather_alerts.check_weather_alert_for_region", lambda *args, **kwargs: None)

    tracker, _, username = linked
    tracker.create_progress({
        "crop": "未知试验作物", "start_date": (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d"),
        "expected_end_date": (datetime.now() + timedelta(days=80)).strftime("%Y-%m-%d"),
        "stage": "准备期", "stage_number": 1, "total_stages": 8,
    })
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        response = client.get(f"/api/dashboard?username={username}")
        assert response.status_code == 200
        card = response.json()["progress"][0]
    assert card["stage_number"] == 3
    assert card["overall_progress"] == 20
    assert card["stage_progress"] == card["progress_percent"] == 10
    answer = progress_tracking_node(AgentState(username=username, user_question="查看种植进度")).final_answer
    assert "3/8" in answer
    assert "整季 20%" in answer
    assert "阶段 10%" in answer


def test_指定开始日期统一平移计划阶段和任务():
    plan = PlantingPlanner().generate_plan({"region": "河北", "crop": "小麦", "start_date": "2099-03-12"})
    duration = sum(stage["duration_days"] for stage in CropDatabase().get_crop("小麦").growth_stages)
    assert plan.schedule["sowing_time"] == "2099-03-12"
    assert plan.schedule["stages"][0]["start_date"] == "2099-03-12"
    assert plan.schedule["harvest_time"] == (datetime(2099, 3, 12) + timedelta(days=duration)).strftime("%Y-%m-%d")
    assert plan.tasks[0]["start_date"] == "2099-03-12"
    assert all(task["date"] >= "2099-03-12" for stage in plan.tasks for task in stage["tasks"])


def test_向导未来计划的任务提醒关联可重载():
    username = f"wizard_{uuid.uuid4().hex}"
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        result = client.post(f"/api/plan?username={username}", json={"region": "河北", "crop": "小麦", "start_date": "2099-03-12"})
        assert result.status_code == 200
        card = client.get(f"/api/progress?username={username}").json()[0]
        assert card["start_date"] == "2099-03-12"
        assert card["status"] == "待开始"
        reminders = ReminderStorage(username=username).load_reminders()
        tasks = client.get(f"/api/tasks?username={username}").json()
        assert reminders
        assert all(r["progress_id"] == card["id"] and r["task_id"] for r in reminders)
        assert {r["task_id"] for r in reminders} <= {t["id"] for t in tasks}
        assert all(r["start_date"] >= "2099-03-12" for r in reminders)


@pytest.mark.parametrize("operation", ["complete", "delete"])
def test_完成或删除任务取消关联未通知单次提醒且保留独立与循环提醒(linked, operation):
    tracker, system, _ = linked
    task = tracker.create_task({"crop": "小麦", "title": "浇水"})
    linked_reminder = _reminder(system, task)
    independent = _reminder(system)
    repeating = _reminder(system, task, frequency="每天")
    if operation == "complete":
        assert tracker.update_task_status(task.id, "已完成") is True
    else:
        assert tracker.delete_task(task.id) is True
    saved = {r["id"]: r for r in ReminderStorage(username=tracker._username).load_reminders()}
    assert saved[linked_reminder.id]["status"] == "cancelled"
    assert not saved[linked_reminder.id]["next_trigger"]
    assert saved[independent.id]["status"] == "active"
    assert saved[repeating.id]["status"] == "active"


def test_任务改期同步关联提醒日期并保留通知时间(linked):
    tracker, system, username = linked
    task = tracker.create_task({"crop": "小麦", "title": "浇水", "start_date": "2099-01-02", "end_date": "2099-01-02"})
    reminder = _reminder(system, task)
    tracker.update_task(task.id, {"start_date": "2099-04-03", "end_date": "2099-04-03"})
    saved = next(r for r in ReminderStorage(username=username).load_reminders() if r["id"] == reminder.id)
    assert saved["start_date"] == "2099-04-03"
    assert saved["next_trigger"] == "2099-04-03 15:30"


def test_循环提醒已通知一次后完成关联任务仍继续下个周期(linked):
    tracker, system, username = linked
    task = tracker.create_task({"title": "每天浇水"})
    reminder = _reminder(system, task, frequency="每天")
    system.storage.update_reminder(reminder.id, {"next_trigger": "2020-01-01 15:30"})
    assert ReminderScheduler(username=username).check_and_fire(username)
    assert tracker.update_task_status(task.id, "已完成") is True
    saved = ReminderStorage(username=username).load_reminders()[0]
    assert saved["status"] == "active"
    assert saved["last_triggered"]
    assert datetime.strptime(saved["next_trigger"], "%Y-%m-%d %H:%M") > datetime.now()


def test_改期只同步尚未通知单次提醒且保留暂停状态(linked):
    tracker, system, username = linked
    task = tracker.create_task({"title": "浇水", "start_date": "2099-01-02", "end_date": "2099-01-02"})
    paused = _reminder(system, task)
    system.storage.update_reminder(paused.id, {"status": "paused"})
    notified = _reminder(system, task)
    system.storage.update_reminder(notified.id, {"status": "completed", "last_triggered": "2020-01-01 15:30"})
    repeating = _reminder(system, task, frequency="每天")
    independent = _reminder(system)
    tracker.update_task(task.id, {"end_date": "2099-04-03"})
    saved = {r["id"]: r for r in ReminderStorage(username=username).load_reminders()}
    assert saved[paused.id]["status"] == "paused"
    assert saved[paused.id]["next_trigger"] == "2099-04-03 15:30"
    for reminder in (notified, repeating, independent):
        assert saved[reminder.id]["next_trigger"] == "2099-01-02 15:30"


def test_关联编号不能指向其他用户任务(linked, tmp_path):
    _, system, _ = linked
    other = PlantingTracker(str(tmp_path / f"other_{uuid.uuid4().hex}"))
    task = other.create_task({"title": "其他人的任务"})
    with pytest.raises(ValueError, match="任务"):
        _reminder(system, task)


def test_单次通知状态兼容completed但展示已通知(linked):
    _, system, username = linked
    reminder = _reminder(system, start_date="2020-01-01")
    fired = ReminderScheduler(username=username).check_and_fire(username)
    saved = next(r for r in ReminderStorage(username=username).load_reminders() if r["id"] == reminder.id)
    assert fired[0]["status"] == "completed"
    assert saved["status_label"] == "已通知"


def test_旧提醒迁移保留原记录和空关联(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old-reminders.db'}")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE reminders (id INTEGER PRIMARY KEY, crop TEXT)"))
            connection.execute(text("INSERT INTO reminders VALUES (8, '小麦')"))
        apply_migrations(engine)
        apply_migrations(engine)
        with engine.connect() as connection:
            assert connection.execute(text("SELECT id, crop, task_id, progress_id, start_date FROM reminders")).one() == (8, "小麦", None, None, None)
    finally:
        engine.dispose()


@pytest.mark.parametrize("endpoint,payload", [("/api/plan", {"crop": "小麦", "start_date": "2099-02-30"}), ("/api/progress", {"crop": "小麦", "start_date": "2099-13-01"})])
def test_服务端拒绝不存在的计划日期(endpoint, payload):
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        assert client.post(endpoint, json=payload).status_code == 422


def test_未来计划到达起点自动显示进行中(linked, monkeypatch):
    import core.planting_tracker as module

    tracker, _, _ = linked
    tracker.create_progress({"crop": "小麦", "start_date": "2099-01-01"})

    class Arrived(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2099, 1, 1, 12, 0)

    monkeypatch.setattr(module, "datetime", Arrived)
    card = tracker.get_progress_cards()[0]
    assert card["status"] == "进行中"
    assert card["overall_progress"] == 0


def test_聊天创建单次提醒明确绑定日历任务并同步完成(tmp_path, monkeypatch):
    from app.agent.nodes import reminder as node
    from app.agent.state import AgentState

    username = f"chat_link_{uuid.uuid4().hex}"
    monkeypatch.setattr(node, "DEFAULT_DATA_DIR", str(tmp_path))
    state = node.reminder_management_node(AgentState(username=username, intent_type="reminder_setup", user_question="明天上午8点提醒我给小麦除草"))
    assert not state.error_code
    saved = ReminderStorage(username=username).load_reminders()[0]
    assert saved["task_id"] == state.reminders[0]["task_id"]
    tracker = PlantingTracker(str(tmp_path / username))
    assert tracker.update_task_status(saved["task_id"], "已完成") is True
    assert ReminderStorage(username=username).load_reminders()[0]["status"] == "cancelled"


@pytest.mark.parametrize("operation", ["complete", "reschedule"])
def test_提醒同步失败时任务变更一并回滚(linked, monkeypatch, operation):
    tracker, system, _ = linked
    task = tracker.create_task({"title": "浇水", "start_date": "2099-01-02", "end_date": "2099-01-02"})
    _reminder(system, task)

    def unavailable(*args, **kwargs):
        raise RuntimeError("提醒存储暂不可用")

    monkeypatch.setattr(ReminderSystem, "sync_task_reminders", unavailable)
    with pytest.raises(RuntimeError):
        if operation == "complete":
            tracker.update_task_status(task.id, "已完成")
        else:
            tracker.update_task(task.id, {"end_date": "2099-01-05"})
    saved = tracker.get_task_by_id(task.id)
    assert saved.status == "待办"
    assert saved.end_date == "2099-01-02"
