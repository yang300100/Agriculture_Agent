"""Agent 工具函数：记忆管理 + 事实提取 + 文本聚合"""

from typing import List, Dict, Any
from datetime import datetime
import re

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .config import SHORT_MEMORY_TOP_K, SUMMARY_PROMPT, LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, OPENAI_BASE_URL
from .state import AgentState


def _get_llm(temperature: float = None) -> ChatOpenAI:
    """获取 LLM 实例（供 image_analysis / crop_monitor 等节点调用）"""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


def trim_short_memory(messages: List[BaseMessage], top_k: int = SHORT_MEMORY_TOP_K) -> List[BaseMessage]:
    """手动修剪短记忆，保留最近 N 轮对话 + 系统消息"""
    if not messages:
        return []
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    conversation_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    keep_count = top_k * 2
    trimmed = conversation_messages if len(conversation_messages) <= keep_count else conversation_messages[-keep_count:]
    return system_messages + trimmed


def generate_long_memory_summary(messages: List[BaseMessage], llm: ChatOpenAI) -> str:
    """生成长记忆摘要"""
    conv_history = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conv_history += f"用户：{msg.content}\n"
        elif isinstance(msg, AIMessage):
            conv_history += f"AI：{msg.content}\n"
    prompt = PromptTemplate(template=SUMMARY_PROMPT, input_variables=["conversation_history", "current_time"])
    summary_input = prompt.format(
        conversation_history=conv_history,
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    try:
        response = llm.invoke([HumanMessage(content=summary_input)])
        return (response.content or "").strip()
    except Exception as e:
        # LLM 调用失败时返回空摘要，不阻断主流程
        import logging
        logging.getLogger(__name__).warning("长记忆摘要生成失败: %s", e)
        return ""


def extract_facts_from_conversation(state: AgentState) -> Dict[str, Any]:
    """从当前对话中提取关键事实"""
    user_question = state.user_question or ""
    facts = {}

    region_match = re.search(r'([一-龥]{2,10}(?:省|市|县|区|地区))', user_question)
    if region_match:
        facts["region"] = region_match.group(1)

    crops = ["小麦", "玉米", "水稻", "大豆", "棉花", "土豆", "红薯", "番茄", "黄瓜", "茄子",
             "辣椒", "白菜", "萝卜", "胡萝卜", "菠菜", "生菜", "芹菜", "韭菜", "大葱", "大蒜",
             "洋葱", "南瓜", "西瓜", "甜瓜", "草莓", "葡萄", "苹果", "梨", "桃", "李子", "杏",
             "樱桃", "枣", "柿子", "核桃", "板栗", "茶叶", "烟草", "花生", "油菜", "芝麻",
             "向日葵", "甘蔗", "甜菜"]
    for crop in crops:
        if crop in user_question:
            facts["crop"] = crop
            break

    area_match = re.search(r'(\d+(?:\.\d+)?)\s*[亩分]', user_question)
    if area_match:
        facts["farm_size"] = float(area_match.group(1))

    soils = ["壤土", "砂土", "粘土", "沙壤土", "黏壤土", "黑土", "黄土", "红土", "水稻土"]
    for soil in soils:
        if soil in user_question:
            facts["soil_type"] = soil
            break

    if facts.get("crop") and len(user_question) <= 5:
        for msg in state.messages[-4:]:
            if isinstance(msg, HumanMessage):
                content = msg.content
                if any(word in content for word in ["发黄", "病害", "虫害", "病", "虫", "叶子", "枯萎", "斑点"]):
                    facts["context_disease_discussion"] = True
                    break

    return facts


def aggregate_sentences(docs: List[Dict[str, Any]], window: int = 1) -> List[Dict[str, Any]]:
    """聚合命中句子为弱段落"""
    aggregated = []
    for i, doc in enumerate(docs):
        sentences = [doc["page_content"]]
        if i - window >= 0:
            sentences.insert(0, docs[i - window]["page_content"])
        if i + window < len(docs):
            sentences.append(docs[i + window]["page_content"])
        aggregated.append({
            "content": "\n".join(sentences),
            "evidence": doc["page_content"],
            "source": doc.get("source", "未知文件")
        })
    return aggregated
