"""解析用户输入节点：提取问题、同步档案、检测语音指令"""

import re, logging
from langchain_core.messages import HumanMessage
from ..state import AgentState
from ..utils import trim_short_memory

logger = logging.getLogger(__name__)

# 语音指令前缀 → (intent_type, 参数提取正则)
VOICE_COMMANDS = [
    # 记账：小麦 收入 5000
    ("记账：", "finance_query",
     re.compile(r"记账[：:]?\s*(\S+)\s*(收入|支出|成本)\s*(\d+\.?\d*)\s*元?")),
    ("记一笔：", "finance_query",
     re.compile(r"记一笔[：:]?\s*(\S+)\s*(收入|支出|成本)\s*(\d+\.?\d*)\s*元?")),
    # 提醒：明天 8点 小麦 浇水
    ("提醒：", "reminder_setup",
     re.compile(r"提醒[：:]?\s*(.+?)\s*(\S+)\s*(\S+)")),
    ("设置提醒：", "reminder_setup",
     re.compile(r"设置提醒[：:]?\s*(.+?)\s*(\S+)\s*(\S+)")),
    # 添加任务：小麦 施肥
    ("添加任务：", "reminder_setup",
     re.compile(r"添加任务[：:]?\s*(\S+)\s*(.+)")),
    # 记录进度：小麦 拔节期
    ("记录进度：", "progress_tracking",
     re.compile(r"记录进度[：:]?\s*(\S+)\s*(.+)")),
    # 查天气
    ("查天气：", "weather_query", None),
    ("查天气", "weather_query", None),
]


def _extract_voice_command(question: str) -> dict:
    """检测语音指令前缀，提取结构化参数"""
    for prefix, intent, pattern in VOICE_COMMANDS:
        if question.startswith(prefix) or prefix.rstrip("：") in question[:8]:
            if pattern is None:
                return {"intent_type": intent, "params": {}}
            m = pattern.search(question)
            if m:
                groups = m.groups()
                if intent == "finance_query" and len(groups) >= 3:
                    return {
                        "intent_type": intent,
                        "params": {"crop": groups[0], "trans_type": groups[1], "amount": float(groups[2])},
                    }
                elif intent == "reminder_setup" and len(groups) >= 2:
                    return {
                        "intent_type": intent,
                        "params": {"crop": groups[-2], "task": groups[-1].strip(),
                                   "detail": groups[0] if len(groups) >= 3 else groups[-1]},
                    }
                elif intent == "progress_tracking" and len(groups) >= 1:
                    return {
                        "intent_type": intent,
                        "params": {"crop": groups[0], "detail": groups[1].strip() if len(groups) >= 2 else ""},
                    }
                elif intent == "weather_query":
                    return {"intent_type": intent, "params": {}}
            else:
                # 有前缀但正则不匹配 → 把去掉前缀的文本当作普通问题
                clean = question[len(prefix):].strip() if question.startswith(prefix) else question
                return {"intent_type": intent, "params": {"raw": clean}}
    return {}


def parse_user_input(state: AgentState) -> AgentState:
    """解析用户输入，提取问题，同步用户信息到短期记忆"""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            state.user_question = msg.content.strip()
            break

    question = state.user_question or ""

    # 语音指令检测
    voice_cmd = _extract_voice_command(question)
    if voice_cmd:
        state.intent_type = voice_cmd["intent_type"]
        params = voice_cmd["params"]
        if params.get("crop"):
            state.short_term_facts["crop"] = params["crop"]
        if params.get("trans_type"):
            state.short_term_facts["trans_type"] = params["trans_type"]
        if params.get("amount"):
            state.short_term_facts["amount"] = params["amount"]
        if params.get("task"):
            state.short_term_facts["task_desc"] = params["task"]
        if params.get("raw"):
            state.user_question = params["raw"]
        logger.info("语音指令: intent=%s params=%s", state.intent_type, params)

    # 同步用户档案到短期记忆
    user_profile = state.user_profile
    if user_profile.get("region"):
        state.short_term_facts["region"] = user_profile["region"]
    if user_profile.get("soil_type"):
        state.short_term_facts["soil_type"] = user_profile["soil_type"]
    if user_profile.get("farm_size"):
        state.short_term_facts["farm_size"] = user_profile["farm_size"]
    if user_profile.get("experience"):
        state.short_term_facts["experience"] = user_profile["experience"]
    if user_profile.get("goals"):
        state.short_term_facts["goals"] = user_profile["goals"]

    # 提取用户提及的前季作物（用于轮作建议）
    for pattern in ["去年种了", "之前种了", "上季种了", "上茬种了", "前茬是", "种过"]:
        idx = question.find(pattern)
        if idx >= 0:
            after = question[idx + len(pattern):].strip()
            match = re.match(r'[一-鿿]{2,3}', after)
            if match:
                state.short_term_facts["previous_crop"] = match.group()
            break

    state.long_term_profile["conversation_round"] = state.long_term_profile.get("conversation_round", 0) + 1
    state.messages = trim_short_memory(state.messages, 8)
    return state
