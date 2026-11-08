"""哥哥，真实模型不可用时也要保护设备指令和用户边界。"""

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.agents import device_agent as module
from app.agent.state import AgentState
from devices.base import DeviceCapability, DeviceInfo, DeviceResult


@pytest.fixture
def devices():
    return [
        DeviceInfo("pump-a", "一号水泵", "simulator", [DeviceCapability.IRRIGATE]),
        DeviceInfo("pump-b", "二号水泵", "simulator", [DeviceCapability.IRRIGATE]),
        DeviceInfo("lamp-a", "一号补光灯", "simulator", [DeviceCapability.LIGHT]),
    ]


@pytest.fixture
def control(monkeypatch, devices):
    engine = Mock()
    engine.find_matching_rules.return_value = []
    engine.evaluate_action.side_effect = lambda rule, params, context: (
        "auto_execute", "规则校验通过", params
    )
    monkeypatch.setattr("core.device_rule_engine.RuleEngine", lambda username: engine)
    monkeypatch.setattr("app.agent.config.get_autonomy_level", lambda username="default": "medium")
    monkeypatch.setattr(module, "_discover_devices", lambda username: devices, raising=False)
    agent = module.DeviceAgent()
    monkeypatch.setattr(agent, "_get_sensor_context", lambda *args: {})
    executed = []

    def execute(device_id, command, params, state, engine, **kwargs):
        executed.append((device_id, command, params))
        return agent._reply(state, "已下发设备指令")

    monkeypatch.setattr(agent, "_do_execute", execute)
    return agent, engine, executed


@pytest.mark.parametrize("question, capability", [
    ("停止一号水泵浇水", "irrigate"),
    ("关闭一号补光灯", "light"),
])
def test_keyword_stop_precedes_start(question, capability):
    parsed = module.DeviceAgent()._keyword_parse(question)
    assert parsed["action"] == "stop"
    assert parsed["capability"] == capability


@pytest.mark.parametrize("question", [
    "不要给番茄浇水", "别开补光灯", "我已经浇水了", "明天给番茄浇水10分钟",
    "请创建一个明天给番茄浇水10分钟的任务，现在先别执行",
    "怎样给番茄浇水", "现在是否适合浇水", "不停止浇水",
])
def test_keyword_does_not_execute_negated_or_deferred_requests(question):
    assert module.DeviceAgent()._keyword_parse(question) is None


def test_keyword_status_precedes_irrigation():
    parsed = module.DeviceAgent()._keyword_parse("查看可以浇水的设备列表和状态")
    assert parsed["action"] == "status"


def test_keyword_generic_start_preserves_target_resolution():
    parsed = module.DeviceAgent()._keyword_parse("打开pump-b")
    assert parsed["action"] == "start"


def test_keyword_spaced_watering_preserves_duration():
    parsed = module.DeviceAgent()._keyword_parse("给番茄浇10分钟水")
    assert parsed == {"action": "irrigate", "params": {"duration": 10}}


def test_keyword_fertilizer_preserves_explicit_amount():
    parsed = module.DeviceAgent()._keyword_parse("给番茄施肥2.5公斤，持续10分钟")
    assert parsed["params"] == {"duration": 10, "amount_kg": 2.5}


def test_discovery_accepts_explicit_device_in_task_sentence(monkeypatch, devices):
    monkeypatch.setattr(module, "_discover_devices", lambda username: devices)
    assert module._discover_device_for_capability("irrigate", "alice", "明天用pump-b浇水10分钟") == "pump-b"


def test_explicit_device_hint_selects_requested_pump(control, monkeypatch):
    agent, engine, executed = control
    monkeypatch.setattr(agent, "_parse_device_intent", lambda *args: {
        "action": "irrigate", "device_hint": "二号水泵", "params": {"duration": 1}
    })
    agent.invoke(AgentState(username="alice", user_question="请给二号水泵浇水1分钟"))
    assert executed == [("pump-b", "start", {"duration": 1})]


def test_ambiguous_pumps_request_clarification(control, monkeypatch):
    agent, engine, executed = control
    monkeypatch.setattr(agent, "_parse_device_intent", lambda *args: {
        "action": "irrigate", "params": {"duration": 1}
    })
    state = agent.invoke(AgentState(username="alice", user_question="浇水1分钟"))
    assert not executed
    assert "一号水泵" in state.final_answer and "二号水泵" in state.final_answer


def test_unrelated_or_other_device_rules_cannot_replace_request(control, monkeypatch):
    agent, engine, executed = control
    engine.find_matching_rules.return_value = [
        {"id": "light-rule", "action": {"device_id": "lamp-a", "capability": "light", "command": "start", "params": {"brightness_percent": 80}}},
        {"id": "pump-a-rule", "action": {"device_id": "pump-a", "capability": "irrigate", "command": "start", "params": {"duration": 20}}},
    ]
    monkeypatch.setattr(agent, "_parse_device_intent", lambda *args: {
        "action": "irrigate", "device_hint": "pump-b", "params": {"duration": 1}
    })
    agent.invoke(AgentState(username="alice", user_question="pump-b浇水1分钟"))
    assert executed == [("pump-b", "start", {"duration": 1})]


def test_matched_database_rule_ids_remain_valid_state(control, monkeypatch):
    agent, engine, executed = control
    engine.find_matching_rules.return_value = [{
        "id": 7, "name": "灌溉规则", "action": {
            "device_id": "pump-b", "capability": "irrigate", "command": "start", "params": {"duration": 1}
        },
    }]
    monkeypatch.setattr(agent, "_parse_device_intent", lambda *args: {
        "action": "irrigate", "device_hint": "pump-b", "params": {"duration": 1}
    })
    state = agent.invoke(AgentState(username="alice", user_question="pump-b浇水1分钟"))
    assert state.matched_rules == ["7"]
    AgentState.model_validate(state.model_dump())


def test_generic_start_resolves_device_capability(control, monkeypatch):
    agent, engine, executed = control
    monkeypatch.setattr("langchain_openai.ChatOpenAI", Mock(side_effect=RuntimeError("离线")))
    agent.invoke(AgentState(username="alice", user_question="打开pump-b"))
    assert executed == [("pump-b", "start", {"duration": 30})]
    assert engine.evaluate_action.call_args.args[0]["action"]["capability"] == "irrigate"


def test_stop_uses_latest_device_in_history(control, monkeypatch):
    agent, engine, executed = control
    monkeypatch.setattr("langchain_openai.ChatOpenAI", Mock(side_effect=RuntimeError("离线")))
    state = AgentState(username="alice", user_question="停止它", messages=[
        HumanMessage(content="请用pump-b浇水1分钟"),
        AIMessage(content="设备 pump-b 已启动"),
        HumanMessage(content="停止它"),
    ])
    agent.invoke(state)
    assert executed == [("pump-b", "stop", {})]
    assert not engine.find_matching_rules.called


def test_negated_request_is_blocked_even_if_model_parser_proposes_start(control, monkeypatch):
    agent, engine, executed = control
    monkeypatch.setattr(agent, "_parse_device_intent", lambda *args: {
        "action": "irrigate", "device_hint": "pump-a", "params": {"duration": 30}
    })
    agent.invoke(AgentState(username="alice", user_question="不要用pump-a浇水"))
    assert not executed


def test_pending_persistence_failure_does_not_offer_missing_confirmation(control, monkeypatch):
    agent, engine, executed = control
    engine.evaluate_action.return_value = ("need_confirm", "需要确认", {"duration": 90})
    engine.evaluate_action.side_effect = None
    monkeypatch.setattr(agent, "_parse_device_intent", lambda *args: {
        "action": "irrigate", "device_hint": "pump-b", "params": {"duration": 90}
    })
    monkeypatch.setattr(agent, "_write_decision_log", lambda *args, **kwargs: None)
    state = agent.invoke(AgentState(username="alice", user_question="pump-b浇水90分钟"))
    assert not executed and state.pending_action is None
    assert "保存失败" in state.final_answer


def test_device_discovery_keeps_concurrent_user_scope(control, monkeypatch, devices):
    agent, engine, executed = control
    barrier = threading.Barrier(2)
    observed = []

    def parse(question, state):
        barrier.wait(timeout=5)
        return {"action": "irrigate", "params": {"duration": 1}}

    def discover(username):
        observed.append(username)
        return [DeviceInfo(username + "-pump", username, "simulator", [DeviceCapability.IRRIGATE])]

    monkeypatch.setattr(agent, "_parse_device_intent", parse)
    monkeypatch.setattr(module, "_discover_devices", discover)
    with ThreadPoolExecutor(max_workers=2) as pool:
        states = list(pool.map(agent.invoke, [AgentState(username="alice", user_question="浇水1分钟"), AgentState(username="bob", user_question="浇水1分钟")]))
    assert set(observed) == {"alice", "bob"}
    assert {entry[0] for entry in executed} == {"alice-pump", "bob-pump"}


def test_final_executor_confirmation_is_exposed_to_chat(monkeypatch):
    from core.device_executor import DeviceExecutor
    from core import device_registry_factory

    loop = asyncio.new_event_loop()

    async def discover():
        return []

    registry = SimpleNamespace(discover_all=discover)
    monkeypatch.setattr(device_registry_factory, "setup_registry", lambda username: (registry, loop))
    monkeypatch.setattr(device_registry_factory, "close_registry", lambda active_loop, registry: active_loop.close())
    monkeypatch.setattr(DeviceExecutor, "execute_sync", lambda *args, **kwargs: {
        "success": False, "decision": "need_confirm", "pending_id": "pending-test",
        "result": DeviceResult(False, "pump-b", "start", message="需要哥哥确认", error_code="POLICY_CONFIRM_REQUIRED"),
    })
    state = AgentState(username="alice")
    module.DeviceAgent()._do_execute("pump-b", "start", {"duration": 1}, state, Mock())
    assert state.pending_action["id"] == "pending-test"
    assert "⚠️" in state.final_answer and "❌" not in state.final_answer


def test_offline_device_dialogue_persists_start_status_stop_and_tasks(monkeypatch):
    from core.device_executor import DeviceExecutor
    from core.device_registry_factory import save_custom_devices
    from core.planting_tracker import PlantingTracker
    from core.storage_paths import DEFAULT_DATA_DIR
    from devices.registry import DeviceDriverRegistry

    username = "chat_device_integration"
    save_custom_devices(username, [{
        "device_id": "dialogue-pump", "name": "对话验收水泵", "driver": "simulator",
        "capabilities": ["irrigate"], "sensors": [], "connection": {},
        "initial_state": {"power": False, "status": "powered_off"},
    }])
    monkeypatch.setattr("langchain_openai.ChatOpenAI", Mock(side_effect=RuntimeError("离线")))
    monkeypatch.setattr("app.agent.config.get_autonomy_level", lambda username="default": "medium")
    monkeypatch.setenv("AUTONOMY_LEVEL", "medium")
    # 哥哥，固定放行旧版进程时段约束，让这个回归在昼夜都验证同一条硬件链路。
    monkeypatch.setattr("core.device_rule_engine.RuleEngine._check_constraints", lambda *args: (True, ""))
    agent = module.DeviceAgent()
    started = agent.invoke(AgentState(username=username, user_question="请给dialogue-pump浇水1分钟"))
    assert started.device_result["success"] is True
    assert started.device_command["action"] == "irrigate"
    queried = agent.invoke(AgentState(username=username, user_question="查看dialogue-pump设备状态"))
    assert queried.device_command is None
    assert queried.device_result["devices"][0]["state"]["status"] == "running"
    stopped = agent.invoke(AgentState(username=username, user_question="停止dialogue-pump浇水"))
    assert stopped.device_result["success"] is True
    queried = agent.invoke(AgentState(username=username, user_question="查看dialogue-pump设备状态"))
    assert queried.device_result["devices"][0]["state"]["status"] == "standby"
    tracker = PlantingTracker(DEFAULT_DATA_DIR + "/" + username)
    tasks = tracker.get_tasks()
    assert len(tasks) == 2
    assert {task.task_type for task in tasks} == {"浇水", "停止设备"}
    assert all(task.status == "已完成" and task.device_id == "dialogue-pump" for task in tasks)
    logs = DeviceExecutor(DeviceDriverRegistry(), username=username).get_logs()
    assert [entry["command"] for entry in logs] == ["stop", "start"]


@pytest.mark.parametrize("autonomy", ["medium", "high"])
@pytest.mark.parametrize("mode", ["confirm", "notify"])
def test_聊天遵守规则执行方式且高自主权不能绕过(monkeypatch, request, autonomy, mode):
    from core.device_executor import DeviceExecutor
    from core.device_registry_factory import load_custom_devices, save_custom_devices
    from core.device_rule_engine import RuleEngine
    from devices.registry import DeviceDriverRegistry

    username = f"chat_rule_mode_{uuid.uuid4().hex}"
    save_custom_devices(username, [{
        "device_id": "mode-pump", "name": "规则模式测试水泵", "driver": "simulator",
        "capabilities": ["irrigate"], "sensors": [],
        "initial_state": {"power": False, "status": "powered_off"},
    }])
    engine = RuleEngine(username)
    rule_id = engine.add_rule({
        "name": "聊天模式保护", "execution_mode": mode,
        "trigger": {"conditions": [
            {"type": "time", "op": "between", "value": ["00:00", "23:59"]},
        ]},
        "action": {
            "device_id": "mode-pump", "capability": "irrigate", "command": "start",
            "params": {"duration": 1},
        },
    })
    request.addfinalizer(lambda: engine.delete_rule(rule_id))
    monkeypatch.setenv("AUTONOMY_LEVEL", autonomy)
    monkeypatch.setattr("app.agent.config.get_autonomy_level", lambda username="default": autonomy)
    agent = module.DeviceAgent()
    # 哥哥，只替换外部模型解析，规则、队列和设备下发仍走真实链路。
    monkeypatch.setattr(agent, "_parse_device_intent", lambda *args: {
        "action": "irrigate", "device_hint": "mode-pump", "params": {"duration": 1},
    })

    state = agent.invoke(AgentState(username=username, user_question="mode-pump浇水1分钟"))

    assert load_custom_devices(username)[0]["initial_state"]["status"] == "powered_off"
    executor = DeviceExecutor(DeviceDriverRegistry(), username)
    pending = executor.list_pending()
    assert len(pending) == (1 if mode == "confirm" else 0)
    if mode == "confirm":
        assert state.pending_action["id"] == pending[0]["id"]
    else:
        assert state.pending_action is None
        assert "仅通知" in state.final_answer
    assert not any(log["success"] for log in executor.get_logs())
