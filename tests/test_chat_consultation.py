"""哥哥，咨询应回答当前问题，不能顺手创建整季计划和任务。"""

import pytest

from app.agent.agents.planting_agent import PlantingAgent
from app.agent.state import AgentState
from app.agent.nodes import extract_tasks, llm_response, planting_plan, field


@pytest.mark.parametrize("intent,question", [
    ("planting_method", "只回答一句话，小麦返青期检查土壤墒情有什么作用？"),
    ("planting_schedule", "河北小麦应该什么时候播种？"),
    ("harvest_planning", "怎么判断小麦能否收获？"),
    ("crop_selection", "河北壤土适合种什么？"),
])
def test_普通咨询不生成计划或提取任务(intent, question, monkeypatch):
    extraction_calls = []
    def unexpected(*args, **kwargs):
        raise AssertionError("咨询不能创建计划或任务")

    def answer(state):
        state.final_answer = "根据土壤含水量确定灌溉时机。"
        return state

    def extract(*args, **kwargs):
        extraction_calls.append(True)
        return []

    monkeypatch.setattr(planting_plan, "planting_plan_node", unexpected)
    monkeypatch.setattr(llm_response, "llm_expert_answer", answer)
    monkeypatch.setattr(extract_tasks, "extract_suggestions_from_answer", extract)
    state = PlantingAgent().invoke(AgentState(intent_type=intent, user_question=question))
    extract_tasks.extract_and_create_tasks_node(state)
    assert state.final_answer == "根据土壤含水量确定灌溉时机。"
    assert extraction_calls == []


def test_明确制定计划仍调用原计划入口且不重复提取(monkeypatch):
    calls = []
    extraction_calls = []

    def plan(state):
        calls.append(state.user_question)
        state.final_answer = "已生成种植计划"
        return state

    def extract(*args, **kwargs):
        extraction_calls.append(True)
        return []

    monkeypatch.setattr(planting_plan, "planting_plan_node", plan)
    monkeypatch.setattr(extract_tasks, "extract_suggestions_from_answer", extract)
    state = PlantingAgent().invoke(AgentState(intent_type="planting_schedule", user_question="帮我制定小麦种植计划"))
    extract_tasks.extract_and_create_tasks_node(state)
    assert calls == ["帮我制定小麦种植计划"]
    assert extraction_calls == []


def test_地区面积查询使用当前档案回答(monkeypatch):
    def answer(state):
        assert state.user_profile["region"] == "河北石家庄"
        state.final_answer = "地区为河北石家庄，面积为13亩。"
        return state

    monkeypatch.setattr(llm_response, "llm_response_node", answer)
    state = field.field_management_node(AgentState(
        intent_type="field_management", user_question="我的农场地区和面积分别是多少？只回答这两项。",
        user_profile={"region": "河北石家庄", "farm_size": 13},
    ))
    assert state.final_answer == "地区为河北石家庄，面积为13亩。"
