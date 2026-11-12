"""哥哥，对话卡片必须先留下确认候选，不能提前操作设备。"""

import os
import uuid
from unittest.mock import Mock

import pytest

from app.agent.agents.device_agent import DeviceAgent
from app.agent.state import AgentState
from core.device_executor import DeviceExecutor
from core.device_registry_factory import load_custom_devices, save_custom_devices
from devices.registry import DeviceDriverRegistry


@pytest.fixture
def chat_devices(monkeypatch):
    username = "card_agent_" + uuid.uuid4().hex
    save_custom_devices(username, [{
        "device_id": device_id, "name": name, "driver": "simulator",
        "capabilities": ["irrigate"], "sensors": [],
        "initial_state": {"power": False, "status": "powered_off"},
    } for device_id, name in [("pump-a", "一号水泵"), ("pump-b", "二号水泵")]])
    monkeypatch.setattr("langchain_openai.ChatOpenAI", Mock(side_effect=RuntimeError("离线")))
    monkeypatch.setattr("app.agent.config.get_autonomy_level", lambda username="default": "medium")
    monkeypatch.setattr("core.device_rule_engine.RuleEngine._check_constraints", lambda *args: (True, ""))
    return username


@pytest.mark.parametrize("command,question", [("start", "pump-b浇水1分钟"), ("stop", "停止pump-b浇水")])
def test_interactive_dialogue_persists_confirmation_without_execution(chat_devices, command, question):
    state = DeviceAgent().invoke(AgentState(
        username=chat_devices, user_question=question, interactive_cards=True,
    ))
    assert state.pending_action is not None
    assert state.pending_action["command"] == command
    assert state.chat_device_ids == ["pump-b"]
    assert state.device_command["device_id"] == "pump-b"
    executor = DeviceExecutor(DeviceDriverRegistry(), chat_devices)
    assert executor.list_pending()[0]["id"] == state.pending_action["id"]
    assert load_custom_devices(chat_devices)[1]["initial_state"]["status"] == "powered_off"
    assert not any(log["success"] for log in executor.get_logs())


def test_default_dialogue_still_executes_legacy_flow(chat_devices):
    state = DeviceAgent().invoke(AgentState(username=chat_devices, user_question="pump-b浇水1分钟"))
    assert state.device_result["success"] is True
    assert state.pending_action is None
    assert load_custom_devices(chat_devices)[1]["initial_state"]["status"] == "running"


def test_query_and_ambiguous_selection_record_real_device_ids(chat_devices):
    agent = DeviceAgent()
    queried = agent.invoke(AgentState(username=chat_devices, user_question="查看设备列表", interactive_cards=True))
    assert queried.chat_device_ids == ["pump-a", "pump-b"]
    assert queried.pending_action is None
    ambiguous = agent.invoke(AgentState(username=chat_devices, user_question="浇水1分钟", interactive_cards=True))
    assert ambiguous.chat_device_ids == ["pump-a", "pump-b"]
    assert ambiguous.pending_action is None


def test_model_generic_hint_cannot_select_first_real_device(chat_devices, monkeypatch):
    agent = DeviceAgent()
    monkeypatch.setattr(agent, "_parse_device_intent", lambda *args: {
        "action": "irrigate", "device_hint": "水泵", "params": {"duration": 1},
    })
    state = agent.invoke(AgentState(username=chat_devices, user_question="浇水1分钟", interactive_cards=True))
    assert state.pending_action is None
    assert state.chat_device_ids == ["pump-a", "pump-b"]


def test_interactive_query_uses_brief_markdown_without_raw_state(chat_devices):
    state = DeviceAgent().invoke(AgentState(
        username=chat_devices, user_question="查看设备列表", interactive_cards=True,
    ))
    assert "下方" in state.final_answer and "2" in state.final_answer
    assert '"power"' not in state.final_answer and "_observed_at" not in state.final_answer
    assert len(state.device_result["devices"]) == 2
    legacy = DeviceAgent().invoke(AgentState(username=chat_devices, user_question="查看设备列表"))
    assert '"power"' in legacy.final_answer


@pytest.mark.parametrize("with_rule", [False, True])
def test_interactive_policy_confirmation_points_to_inline_card(chat_devices, monkeypatch, with_rule):
    from core.device_rule_engine import RuleEngine

    if with_rule:
        RuleEngine(chat_devices).add_rule({
            "name": "卡片确认规则", "execution_mode": "confirm",
            "trigger": {"conditions": [{"type": "time", "op": "between", "value": ["00:00", "23:59"]}]},
            "action": {"device_id": "pump-b", "capability": "irrigate", "command": "start", "params": {"duration": 1}},
        })
    monkeypatch.setattr(RuleEngine, "evaluate_action", lambda *args: ("need_confirm", "请确认操作", {"duration": 1}))
    state = DeviceAgent().invoke(AgentState(
        username=chat_devices, user_question="pump-b浇水1分钟", interactive_cards=True,
    ))
    assert state.pending_action is not None
    assert "下方" in state.final_answer and "设备中心" not in state.final_answer
