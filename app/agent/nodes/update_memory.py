"""更新长记忆节点：定期对对话历史做摘要并持久化"""

import logging
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from ..state import AgentState
from ..config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, OPENAI_BASE_URL, SUMMARY_TRIGGER_ROUNDS
from ..utils import generate_long_memory_summary

logger = logging.getLogger(__name__)

# 长记忆摘要最大字符数，防止无限膨胀
MAX_SUMMARY_LENGTH = 2000


def update_long_memory(state: AgentState) -> AgentState:
    """更新长记忆节点"""
    state.progress_message = "正在更新对话记忆..."
    current_round = state.long_term_profile.get("conversation_round", 0)

    # 防止 SUMMARY_TRIGGER_ROUNDS=0 导致 ZeroDivisionError
    if SUMMARY_TRIGGER_ROUNDS <= 0:
        return state

    if current_round % SUMMARY_TRIGGER_ROUNDS == 0 and current_round > 0:
        logger.info("正在更新对话记忆（第 %d 轮）...", current_round)

        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        try:
            new_summary = generate_long_memory_summary(state.messages, llm)
        except Exception as e:
            logger.error("长记忆摘要生成失败: %s", e, exc_info=True)
            return state

        # 合并新旧摘要并截断，防止指数增长
        old_summary = state.long_term_profile.get("summary", "")
        if old_summary:
            merged = f"历史总结：{old_summary}\n最新总结：{new_summary}"
        else:
            merged = new_summary

        # 截断到固定长度，保留最新内容
        if len(merged) > MAX_SUMMARY_LENGTH:
            merged = merged[-MAX_SUMMARY_LENGTH:]
        # 清理截断后可能残留的孤立前缀 "历史总结："（没有实际内容跟着）
        merged = re.sub(r'^历史总结[：:]\s*$', '', merged, flags=re.MULTILINE)
        merged = re.sub(r'^历史总结[：:]\s*最新总结[：:]', '', merged, flags=re.MULTILINE)
        state.long_term_profile["summary"] = merged

        logger.info("记忆更新完成：%s...", state.long_term_profile['summary'][:100])

    return state
