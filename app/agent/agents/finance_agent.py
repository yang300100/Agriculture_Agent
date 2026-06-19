"""财务管理 Agent — 记账、报表、价格查询"""

from .base import BaseAgent
from ..state import AgentState


class FinanceAgent(BaseAgent):
    name = "finance"
    description = "财务专家，负责成本收入管理和市场价格查询"
    system_prompt = """你是一位农业经济师，专精农场财务管理。
你能帮助记录种植成本与销售收入，生成财务分析报表，
查询农产品市场行情和价格走势，计算投入产出比和亩均收益。"""
    intent_types = ["finance_query"]

    def invoke(self, state: AgentState) -> AgentState:
        if state.intent_type == "finance_query":
            from ..nodes.finance import finance_query_node
            return finance_query_node(state)
        return state
