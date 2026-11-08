"""哥哥，对话建任务要真正落库，并保留约定的日期和用户归属。"""

from datetime import datetime, timedelta
import uuid

import pytest

from app.agent.state import AgentState
from app.agent.nodes import classify_intent, extract_tasks, reminder
from core.database.repository.reminders import ReminderRepository
from core.database.repository.users import UserRepository
from core.planting_tracker import PlantingTracker
from core.reminder_scheduler import ReminderScheduler
from core.reminder_system import ReminderSystem


@pytest.mark.parametrize("question,profile_crop,expected_crop", [
    ("明天下午3点提醒我给小麦除草", "", "小麦"),
    ("明天下午3点提醒我给番茄浇水", "水稻", "番茄"),
    ("明天上午8点提醒我为油菜施肥", "", "油菜"),
    ("明天下午3点提醒我给紫苏除草", "", "紫苏"),
    ("明天下午3点提醒我为试验藜麦施肥", "", "试验藜麦"),
    ("明天下午3点提醒我给黑小麦除草", "", "黑小麦"),
    ("明天下午3点提醒我除草", "水稻", "水稻"),
    ("明天下午3点提醒我启动水泵", "", "未指定作物"),
    ("明天上午8点提醒我浇水", "", "未指定作物"),
    ("明天下午3点提醒我给水泵浇水", "", "未指定作物"),
    ("明天下午3点提醒我用水泵给番茄浇水", "", "番茄"),
])
def test_提醒作物提取保留明确名称且不混入动作设备或时间(question, profile_crop, expected_crop, tmp_path, monkeypatch):
    from app.agent.nodes.parse_input import parse_user_input

    monkeypatch.setattr(reminder, "DEFAULT_DATA_DIR", str(tmp_path))
    username = "chat_crop_" + uuid.uuid4().hex
    state = parse_user_input(AgentState(
        username=username, intent_type="reminder_setup", user_question=question,
        user_profile={"crop": profile_crop},
    ))
    result = reminder.reminder_management_node(state)
    assert not result.error_code
    saved = ReminderSystem(username=username).storage.load_reminders()[0]
    task = PlantingTracker(str(tmp_path / username)).get_tasks()[0]
    assert saved["crop"] == task.crop == expected_crop
    assert saved["task_id"] == task.id
    assert result.reminders[0]["crop"] == expected_crop


@pytest.mark.parametrize("question", [
    "你好，帮我创建明天给小麦除草的任务",
    "谢谢，请添加明天下午3点给番茄浇水10分钟的任务",
    "明天上午8点提醒我给黄瓜施肥",
])
def test_明确建任务请求优先于问候且不依赖模型(question, monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError("哥哥，明确请求无需再问模型。")

    monkeypatch.setattr(classify_intent, "_llm_classify_intent", unavailable)
    state = classify_intent.classify_intent(AgentState(user_question=question))
    assert state.intent_type == "reminder_setup"
    assert state.need_clarification is False


@pytest.mark.parametrize("text,offset", [("明天", 1), ("后天", 2), ("一天内", 1)])
def test_中文日期按日历计算(text, offset):
    assert extract_tasks.calculate_end_date(text) == (
        datetime.now().date() + timedelta(days=offset)
    ).isoformat()


def test_指定截止日期完整保留():
    assert extract_tasks.calculate_end_date("2030-02-09上午8点") == "2030-02-09"
    assert extract_tasks.calculate_end_date("2030年2月9日") == "2030-02-09"


@pytest.mark.parametrize("time_text,expected_time", [("上午8点", "08:00"), ("下午3点半", "15:30")])
def test_提醒及任务属于当前用户且约定明天时间(time_text, expected_time, tmp_path, monkeypatch):
    username = "chat_date_" + expected_time.replace(":", "")
    monkeypatch.setattr(reminder, "DEFAULT_DATA_DIR", str(tmp_path))
    state = AgentState(
        username=username, intent_type="reminder_setup",
        user_question=f"帮我创建明天{time_text}给小麦除草的任务",
        short_term_facts={"crop": "小麦"},
    )
    reminder.reminder_management_node(state)
    saved = ReminderSystem(username=username).storage.load_reminders()
    assert len(saved) == 1
    tomorrow = (datetime.now().date() + timedelta(days=1)).isoformat()
    assert saved[0]["next_trigger"] == f"{tomorrow} {expected_time}"
    assert str(saved[0]["id"]) == str(state.reminders[0]["id"])
    assert PlantingTracker(str(tmp_path / username)).get_tasks()[0].end_date == tomorrow


def test_提醒节点已建任务后提取节点不再重复(tmp_path, monkeypatch):
    username = "chat_once"
    monkeypatch.setattr(reminder, "DEFAULT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(extract_tasks, "DEFAULT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(extract_tasks, "extract_suggestions_from_answer", lambda *args, **kwargs: [{
        "crop": "小麦", "task_type": "除草", "title": "小麦除草", "end_date": "2030-01-02",
    }])
    state = AgentState(username=username, intent_type="reminder_setup", user_question="创建明天给小麦除草的任务")
    reminder.reminder_management_node(state)
    extract_tasks.extract_and_create_tasks_node(state)
    assert len(PlantingTracker(str(tmp_path / username)).get_tasks()) == 1


@pytest.mark.parametrize("device_id", ["custom_irrigation_17", None])
def test_设备建议动态绑定且未发现设备也保留任务(device_id, tmp_path, monkeypatch):
    from app.agent.agents import device_agent

    username = "chat_bind_" + str(bool(device_id))
    monkeypatch.setattr(extract_tasks, "DEFAULT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(device_agent, "_discover_device_for_capability", lambda capability, username: device_id)
    monkeypatch.setattr(extract_tasks, "extract_suggestions_from_answer", lambda *args, **kwargs: [{
        "crop": "小麦", "task_type": "浇水", "title": "给小麦浇水10分钟",
        "end_date": "2030-01-02", "device_action": {"action": "irrigate", "params": {"duration": 10}},
    }])
    state = AgentState(username=username, intent_type="planting_method", final_answer="给小麦浇水10分钟")
    extract_tasks.extract_and_create_tasks_node(state)
    tasks = PlantingTracker(str(tmp_path / username)).get_tasks()
    assert len(tasks) == 1
    assert tasks[0].device_id == device_id
    if device_id:
        assert tasks[0].device_command == "start"
        assert tasks[0].device_params == {"duration": 10}


def test_对话创建浇水任务保存参数且不执行设备(tmp_path, monkeypatch):
    from app.agent.agents import device_agent

    monkeypatch.setattr(reminder, "DEFAULT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(device_agent, "_discover_device_for_capability", lambda capability, username, hint="": "custom_pump")
    state = AgentState(username="chat_irrigation", intent_type="reminder_setup", user_question="创建明天给小麦浇水10分钟的任务")
    reminder.reminder_management_node(state)
    tasks = PlantingTracker(str(tmp_path / state.username)).get_tasks()
    assert len(tasks) == 1
    assert tasks[0].device_id == "custom_pump"
    assert tasks[0].device_params == {"duration": 10}
    assert state.device_result is None


def test_提醒返回真实ID且添加其他用户后更新不重建旧记录():
    first = ReminderSystem(username="reminder_owner_first")
    reminder_a = first.create_reminder({"crop": "小麦", "reminder_type": "除草", "start_date": "2020-01-01"})
    row_a = first.storage.load_reminders()[0]
    second = ReminderSystem(username="reminder_owner_second")
    reminder_b = second.create_reminder({"crop": "玉米", "reminder_type": "除草"})
    first.create_reminder({"crop": "番茄", "reminder_type": "修剪"})
    first.pause_reminder(str(reminder_a.id))
    rows = first.storage.load_reminders()
    saved_a = next(row for row in rows if str(row["id"]) == str(reminder_a.id))
    assert saved_a["status"] == "paused"
    assert saved_a["created_at"] == row_a["created_at"]
    assert str(second.storage.load_reminders()[0]["id"]) == str(reminder_b.id)


def test_已到期提醒可由当前用户查询并完成():
    username = "reminder_due_owner"
    system = ReminderSystem(username=username)
    created = system.create_reminder({"crop": "黄瓜", "reminder_type": "浇水", "start_date": "2020-01-01"})
    due = ReminderScheduler(username=username).get_due_reminders(username)
    assert len(due) == 1
    assert str(due[0]["id"]) == str(created.id)
    system.complete_reminder(str(created.id))
    assert ReminderScheduler(username=username).get_due_reminders(username) == []
    row = ReminderRepository().find_one(id=int(created.id), user_id=UserRepository().get_by_username(username).id)
    assert row.status == "completed"
    assert row.last_triggered is not None


def test_卡片写入失败撤回提醒并返回失败码(tmp_path, monkeypatch):
    monkeypatch.setattr(reminder, "DEFAULT_DATA_DIR", str(tmp_path))

    def fail_save(*args, **kwargs):
        raise RuntimeError("哥哥，模拟写入失败。")

    monkeypatch.setattr(PlantingTracker, "create_task", fail_save)
    state = AgentState(username="chat_failed_save", intent_type="reminder_setup", user_question="创建明天除草任务")
    reminder.reminder_management_node(state)
    assert state.error_code == "TASK_CREATION_FAILED"
    assert state.reminders == []
    assert ReminderSystem(username=state.username).storage.load_reminders() == []


def test_模型错误回答不会被再提取成任务(tmp_path, monkeypatch):
    monkeypatch.setattr(extract_tasks, "DEFAULT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(extract_tasks, "extract_suggestions_from_answer", lambda *args, **kwargs: [{
        "title": "错误生成的除草任务", "task_type": "除草",
    }])
    state = AgentState(username="chat_error_answer", intent_type="planting_method", final_answer="模型服务鉴权失败", error_code="LLM_AUTH_FAILED")
    extract_tasks.extract_and_create_tasks_node(state)
    assert state.final_answer == "模型服务鉴权失败"
    assert PlantingTracker(str(tmp_path / state.username)).get_tasks() == []


@pytest.mark.parametrize("answer", ["已浇水10分钟", "当前水泵已关闭", "操作等待确认", "该操作被拒绝"])
def test_设备代理回答不会再生成待办任务(answer, tmp_path, monkeypatch):
    monkeypatch.setattr(extract_tasks, "DEFAULT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(extract_tasks, "extract_suggestions_from_answer", lambda *args, **kwargs: [{
        "title": "错误生成的设备后续任务", "task_type": "巡查",
    }])
    state = AgentState(username="chat_device_answer_" + str(len(answer)), intent_type="device_control", final_answer=answer)
    extract_tasks.extract_and_create_tasks_node(state)
    assert state.final_answer == answer
    assert PlantingTracker(str(tmp_path / state.username)).get_tasks() == []


@pytest.mark.parametrize("question", ["查看当前任务", "查看明天的农事任务"])
def test_明确任务查询本地路由只读(question, tmp_path, monkeypatch):
    from app.agent.graph import build_agricultural_policy_agent
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    def unavailable(*args, **kwargs):
        raise AssertionError("哥哥，查看任务不需要调用模型。")

    username = "chat_read_tasks_" + str(len(question))
    monkeypatch.setattr(ChatOpenAI, "invoke", unavailable)
    monkeypatch.setattr(extract_tasks, "DEFAULT_DATA_DIR", str(tmp_path))
    graph = build_agricultural_policy_agent(object())
    result = graph.invoke(AgentState(username=username, messages=[HumanMessage(content=question)]))
    assert result["intent_type"] == "progress_tracking"
    assert PlantingTracker(str(tmp_path / username)).get_tasks() == []


@pytest.mark.parametrize("question", ["今天已经给小麦浇水了", "不要给小麦施肥", "小麦应该怎么浇水"])
def test_模型故障时状态记录否定及咨询不会降级成建提醒(question):
    result = classify_intent._fallback_intent_classification(question)
    assert result["intent_type"] != "reminder_setup"


def test_模型返回明确截止日期时不被默认天数覆盖(monkeypatch):
    class DateLLM:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            from langchain_core.messages import AIMessage
            return AIMessage(content='[{"title":"小麦除草","end_date":"2030-02-09"}]')

    monkeypatch.setattr(extract_tasks, "ChatOpenAI", DateLLM)
    suggestions = extract_tasks.extract_suggestions_from_answer("2030-02-09除草", "小麦")
    assert suggestions[0]["end_date"] == "2030-02-09"


def test_完整对话图在模型不可用时仍准确创建一条明确任务(tmp_path, monkeypatch):
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from app.agent.graph import build_agricultural_policy_agent
    from app.agent.agents import device_agent

    def unavailable(*args, **kwargs):
        raise AssertionError("哥哥，这条明确建任务请求无需联网。")

    monkeypatch.setattr(ChatOpenAI, "invoke", unavailable)
    monkeypatch.setattr(reminder, "DEFAULT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(extract_tasks, "DEFAULT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(device_agent, "_discover_device_for_capability", lambda capability, username, hint="": "offline_pump")
    graph = build_agricultural_policy_agent(object())
    state = graph.invoke(AgentState(username="chat_graph_owner", messages=[
        HumanMessage(content="你好，请帮我创建明天下午3点半给小麦浇水10分钟的任务")
    ]))
    tomorrow = (datetime.now().date() + timedelta(days=1)).isoformat()
    tasks = PlantingTracker(str(tmp_path / "chat_graph_owner")).get_tasks()
    assert len(tasks) == 1
    assert tasks[0].end_date == tomorrow
    assert tasks[0].device_id == "offline_pump"
    assert tasks[0].device_params == {"duration": 10}
    assert state["error_code"] == ""
    assert "已创建任务" in state["final_answer"]
    reminders = ReminderSystem(username="chat_graph_owner").storage.load_reminders()
    assert len(reminders) == 1
    assert reminders[0]["next_trigger"] == f"{tomorrow} 15:30"


@pytest.mark.parametrize("suffix", ["现在先别执行", "不要马上执行"])
def test_真实注册中心双泵预约按明确ID绑定且保持关闭(suffix, tmp_path, monkeypatch):
    import uuid
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from app.agent.graph import build_agricultural_policy_agent
    from core.device_registry_factory import (
        close_registry, invalidate_registry_cache, load_custom_devices, save_custom_devices, setup_registry,
    )

    username = "chat_real_pumps_" + uuid.uuid4().hex[:12]
    save_custom_devices(username, [{
        "device_id": device_id, "name": name, "driver": "simulator", "capabilities": ["irrigate"],
        "initial_state": {"power": False, "status": "powered_off"},
    } for device_id, name in [("wheat_irrigation", "小麦灌溉泵"), ("tomato_irrigation", "番茄灌溉泵")]])

    def unavailable(*args, **kwargs):
        raise AssertionError("哥哥，建任务无需模型或设备执行。")

    monkeypatch.setattr(ChatOpenAI, "invoke", unavailable)
    monkeypatch.setattr(reminder, "DEFAULT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(extract_tasks, "DEFAULT_DATA_DIR", str(tmp_path))
    graph = build_agricultural_policy_agent(object())
    result = graph.invoke(AgentState(username=username, messages=[HumanMessage(content=(
        "你好，请帮我创建明天下午3点半给小麦浇水10分钟的任务，使用wheat_irrigation，" + suffix + "。"
    ))]))
    tasks = PlantingTracker(str(tmp_path / username)).get_tasks()
    assert len(tasks) == 1
    assert tasks[0].device_id == "wheat_irrigation"
    assert tasks[0].device_command == "start"
    assert tasks[0].device_params == {"duration": 10}
    assert tasks[0].end_date == (datetime.now().date() + timedelta(days=1)).isoformat()
    assert result.get("device_result") is None
    assert all(config["initial_state"]["power"] is False for config in load_custom_devices(username))
    registry, loop = setup_registry(username)
    try:
        loop.run_until_complete(registry.discover_all())
        for device_id in ("wheat_irrigation", "tomato_irrigation"):
            state = loop.run_until_complete(registry.read_state(device_id))
            assert state["power"] is False
            assert state["status"] == "powered_off"
    finally:
        close_registry(loop, registry)
        invalidate_registry_cache(username)
