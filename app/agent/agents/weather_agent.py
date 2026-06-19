"""气象服务 Agent — 天气查询、灾害预警、施药建议、农历节气"""

from .base import BaseAgent
from ..state import AgentState


class WeatherAgent(BaseAgent):
    name = "weather"
    description = "气象专家，负责天气查询、灾害预警、施药气象分析和农历节气指导"
    system_prompt = """你是一位农业气象专家，专精天气对农业生产的影响分析。
你能提供实时天气和预报，评估霜冻/暴雨/高温/大风风险，
判断当前是否适合喷药施肥，并结合农历节气给出传统农事指导。"""
    intent_types = ["weather_query"]

    def invoke(self, state: AgentState) -> AgentState:
        from ..nodes.weather import weather_query_node
        return weather_query_node(state)
