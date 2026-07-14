"""Agent 基类 — 支持 Agent 间互调"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from langchain_core.messages import AIMessage
from ..state import AgentState

if TYPE_CHECKING:
    from .orchestrator import AgentOrchestrator


class BaseAgent:
    """专业智能体基类"""
    name: str = "base"
    description: str = ""
    system_prompt: str = "你是一位专业的农业助手。"
    intent_types: List[str] = []

    # 调度中心引用（由 orchestrator 注入）
    _orchestrator: Optional["AgentOrchestrator"] = None

    def can_handle(self, intent: str) -> bool:
        return intent in self.intent_types

    def invoke(self, state: AgentState) -> AgentState:
        """子类重写此方法实现领域逻辑"""
        return state

    def call_colleague(self, intent: str, state: AgentState) -> Optional[str]:
        """
        调用其他 Agent 获取信息片段

        用法：
            weather_info = self.call_colleague("weather_query", state)
            # weather_info 是气象 Agent 的回答文本
        """
        if self._orchestrator is None:
            return None
        return self._orchestrator.interop_call(intent, state)

    def _reply(self, state: AgentState, answer: str) -> AgentState:
        state.final_answer = answer
        state.messages.append(AIMessage(content=answer))
        return state

    def _get_context(self, state: AgentState) -> Dict[str, Any]:
        return {
            "region": state.short_term_facts.get("region") or state.user_profile.get("region", ""),
            "crop": state.short_term_facts.get("crop", "") or state.user_profile.get("crop", ""),
            "soil": state.user_profile.get("soil_type", ""),
            "area": state.user_profile.get("farm_size", 1.0),
            "goals": state.user_profile.get("goals", []),
        }
