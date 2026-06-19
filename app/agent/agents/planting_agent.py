"""种植规划 Agent — 作物推荐、种植时间、方法指导、收获规划、轮作建议"""

from .base import BaseAgent
from ..state import AgentState


class PlantingAgent(BaseAgent):
    name = "planting"
    description = "种植规划专家，负责作物选择、种植时间、栽培方法、收获规划和轮作建议"
    system_prompt = """你是一位资深农艺师，专精作物种植规划。
你能根据地区气候、土壤条件、种植目标推荐最适宜的作物，
制定完整的种植时间表，指导栽培技术，评估种植风险，并提供科学的轮作建议。"""
    intent_types = ["crop_selection", "planting_schedule", "planting_method", "harvest_planning"]

    def invoke(self, state: AgentState) -> AgentState:
        from ..nodes.planting_plan import planting_plan_node
        return planting_plan_node(state)
