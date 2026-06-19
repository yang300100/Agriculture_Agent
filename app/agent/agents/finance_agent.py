"""财务管理 Agent — 记账、报表、价格查询、政策补贴"""

from .base import BaseAgent
from ..state import AgentState


class FinanceAgent(BaseAgent):
    name = "finance"
    description = "财务与政策专家，负责成本收入管理、市场价格查询和农业补贴政策"
    system_prompt = """你是一位农业经济师，专精农场财务管理和惠农政策。
你能帮助记录种植成本与销售收入，生成财务分析报表，
查询农产品市场行情和价格走势，解读国家和地方农业补贴政策，
计算投入产出比和亩均收益。"""
    intent_types = ["finance_query", "policy_query"]

    def invoke(self, state: AgentState) -> AgentState:
        if state.intent_type == "finance_query":
            from ..nodes.finance import finance_query_node
            return finance_query_node(state)
        elif state.intent_type == "policy_query":
            from ..nodes.policy import policy_query_node
            return policy_query_node(state)
        return state
