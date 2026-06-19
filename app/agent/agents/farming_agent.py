"""农事管理 Agent — 提醒设置、进度跟踪、任务管理、地块管理"""

from .base import BaseAgent
from ..state import AgentState


class FarmingAgent(BaseAgent):
    name = "farming"
    description = "农事管家，负责提醒设置、种植进度跟踪、任务管理和地块信息维护"
    system_prompt = """你是一位农事管理专家，专精田间日常管理。
你能设置和管理各类农事提醒（浇水、施肥、打药等），
跟踪作物生长进度，创建和管理农事任务，
管理地块信息和种植历史，提供收获倒计时。"""
    intent_types = ["reminder_setup", "progress_tracking", "field_management"]

    def invoke(self, state: AgentState) -> AgentState:
        if state.intent_type == "reminder_setup":
            from ..nodes.reminder import reminder_management_node
            return reminder_management_node(state)
        elif state.intent_type == "progress_tracking":
            from ..nodes.progress import progress_tracking_node
            return progress_tracking_node(state)
        elif state.intent_type == "field_management":
            from ..nodes.field import field_management_node
            return field_management_node(state)
        return state
