"""更新长记忆节点：定期对对话历史做摘要并持久化"""

import logging

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from ..state import AgentState
from ..config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, OPENAI_BASE_URL, SUMMARY_TRIGGER_ROUNDS
from ..utils import generate_long_memory_summary

logger = logging.getLogger(__name__)


def update_long_memory(state: AgentState) -> AgentState:
    """更新长记忆节点"""
    current_round = state.long_term_profile.get("conversation_round", 0)

    if current_round % SUMMARY_TRIGGER_ROUNDS == 0 and current_round > 0:
        # Streamlit中用st.info替代print
        st.info(f"🔍 正在更新对话记忆（第 {current_round} 轮）...")

        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        new_summary = generate_long_memory_summary(state.messages, llm)

        # 合并新旧摘要
        old_summary = state.long_term_profile.get("summary", "")
        if old_summary:
            state.long_term_profile["summary"] = f"历史总结：{old_summary}\n最新总结：{new_summary}"
        else:
            state.long_term_profile["summary"] = new_summary

        st.success(f" 记忆更新完成：{state.long_term_profile['summary'][:100]}...")

    return state
