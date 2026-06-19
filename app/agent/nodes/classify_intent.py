"""意图分类节点：关键词快速路径 + LLM 分类推理 + 降级关键词匹配"""

import json
import logging
import re
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from ..state import AgentState
from ..config import *

logger = logging.getLogger(__name__)


def classify_intent(state: AgentState) -> AgentState:
    """
    意图分类节点：使用LLM进行智能意图推理
    保留关键词匹配作为快速路径，复杂意图使用LLM推理
    """
    user_question = state.user_question or ""

    # 如果 parse_input 已通过语音指令预设了意图，跳过分类
    if state.intent_type and state.intent_type not in ("unclear",):
        if state.intent_type in ("finance_query", "weather_query", "reminder_setup", "progress_tracking"):
            state.need_rag = False
            state.need_clarification = False
            return state

    # 图片分析意图判断（优先）
    if state.has_image:
        state.intent_type = "image_analysis"
        state.need_rag = True
        return state

    # 通用意图快速判断（关键词匹配）
    if any(word in user_question for word in GREETING_KEYWORDS):
        state.intent_type = "greeting"
        state.need_rag = False
        state.need_clarification = False
        return state
    elif any(word in user_question for word in THANKS_KEYWORDS):
        state.intent_type = "thanks"
        state.need_rag = False
        state.need_clarification = False
        return state
    elif any(word in user_question for word in FAREWELL_KEYWORDS):
        state.intent_type = "farewell"
        state.need_rag = False
        state.need_clarification = False
        return state
    elif any(word in user_question for word in DEVICE_KEYWORDS):
        state.intent_type = "device_control"
        state.need_rag = False
        state.need_clarification = False
        return state

    # 使用LLM进行意图推理
    intent = _llm_classify_intent(user_question, state)
    state.intent_type = intent["intent_type"]
    state.need_rag = intent["need_rag"]
    state.need_clarification = intent["need_clarification"]

    return state


def _llm_classify_intent(user_question: str, state: AgentState) -> Dict[str, Any]:
    """
    使用LLM进行意图分类推理

    返回:
        {
            "intent_type": 意图类型,
            "need_rag": 是否需要RAG检索,
            "need_clarification": 是否需要澄清,
            "reasoning": 推理过程
        }
    """
    # 构建最近的对话历史
    recent_history = []
    for msg in state.messages[-6:]:
        if isinstance(msg, HumanMessage):
            recent_history.append(f"用户：{msg.content}")
        elif isinstance(msg, AIMessage):
            recent_history.append(f"助手：{msg.content[:80]}...")
    history_text = "\n".join(recent_history)

    # 构建意图分类提示词
    intent_prompt = f"""你是一位意图分类专家。请分析用户的输入，判断其意图类型。

可选的意图类型：
- greeting: 问候语（你好、您好、早上好等）
- thanks: 感谢语（谢谢、感谢等）
- farewell: 告别语（再见、拜拜等）
- identity: 询问身份（你是谁、你叫什么等）
- function: 询问功能（你能做什么、有什么功能等）
- crop_selection: 作物选择建议（种什么好、适合种什么、推荐作物等）
- planting_schedule: 种植时间规划（什么时候种、播种时间、几月份种等）
- planting_method: 种植方法指导（怎么种、种植技术、栽培方法等）
- reminder_setup: 提醒设置管理（设置提醒、浇水提醒、施肥提醒等）
- progress_tracking: 进度跟踪查询（查看进度、现在该做什么、生长情况等）
- disease_prevention: 病虫害防治（病虫害、叶子发黄、有虫害、作物病害等）
- harvest_planning: 收获规划建议（什么时候收、收获时间、成熟度等）
- image_analysis: 图片分析（上传了图片进行分析）
- weather_query: 天气查询（天气、气温、下雨、预报等）
- finance_query: 财务查询（成本、收入、价格、收益等）
- field_management: 地块管理（地块、农田、面积、位置、地图等）
- device_control: 设备控制（浇水、灌溉、施肥、通风、补光、加热、开关设备、控制设备等）
- unclear: 意图不明

【关键规则】：
1. 如果用户只输入一个作物名称（如"小麦"、"玉米"），而之前的对话正在讨论该作物的病虫害/病害问题，则意图应为 "disease_prevention"
2. 如果用户输入与之前对话主题相关，优先保持上下文连贯，不要视为"unclear"
3. 用户当前输入可能是对之前问题的补充确认

用户输入："{user_question}"

对话上下文：
- 当前已知作物：{state.short_term_facts.get("crop", "未指定")}
- 地区：{state.short_term_facts.get("region", "未指定")}
- 是否有图片：{"是" if state.has_image else "否"}

【最近对话历史】：
{history_text}

请分析：
1. 用户的核心意图是什么？（请结合对话历史判断）
2. 用户是否在继续之前的话题？
3. 是否需要查询农业知识库？
4. 是否需要进一步澄清？

请以JSON格式返回：
{{
    "intent_type": "意图类型",
    "need_rag": true/false,
    "need_clarification": true/false,
    "reasoning": "推理过程的简要说明",
    "confidence": 0.95
}}"""

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )

        response = llm.invoke([HumanMessage(content=intent_prompt)])
        content = response.content

        # 解析JSON结果
        # 提取JSON部分
        json_match = re.search(r'\{.*?\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "intent_type": result.get("intent_type", "unclear"),
                "need_rag": result.get("need_rag", True),
                "need_clarification": result.get("need_clarification", False),
                "reasoning": result.get("reasoning", "")
            }
    except Exception as e:
        logger.debug(f"LLM意图分类失败: {e}")

    # 降级到关键词匹配
    return _fallback_intent_classification(user_question)


def _fallback_intent_classification(user_question: str) -> Dict[str, Any]:
    """降级方案：使用关键词匹配进行意图分类"""
    # 通用意图
    if any(word in user_question for word in IDENTITY_KEYWORDS):
        return {"intent_type": "identity", "need_rag": False, "need_clarification": False, "reasoning": "关键词匹配"}
    elif any(word in user_question for word in FUNCTION_KEYWORDS):
        return {"intent_type": "function", "need_rag": False, "need_clarification": False, "reasoning": "关键词匹配"}

    # 种植规划意图
    if any(keyword in user_question for keyword in CROP_SELECTION_KEYWORDS):
        return {"intent_type": "crop_selection", "need_rag": True, "need_clarification": False, "reasoning": "关键词匹配"}
    elif any(keyword in user_question for keyword in PLANTING_SCHEDULE_KEYWORDS):
        return {"intent_type": "planting_schedule", "need_rag": True, "need_clarification": False, "reasoning": "关键词匹配"}
    elif any(keyword in user_question for keyword in PLANTING_METHOD_KEYWORDS):
        return {"intent_type": "planting_method", "need_rag": True, "need_clarification": False, "reasoning": "关键词匹配"}
    elif any(keyword in user_question for keyword in REMINDER_KEYWORDS):
        return {"intent_type": "reminder_setup", "need_rag": False, "need_clarification": False, "reasoning": "关键词匹配"}
    elif any(keyword in user_question for keyword in PROGRESS_KEYWORDS):
        return {"intent_type": "progress_tracking", "need_rag": True, "need_clarification": False, "reasoning": "关键词匹配"}
    elif any(keyword in user_question for keyword in DISEASE_KEYWORDS):
        return {"intent_type": "disease_prevention", "need_rag": True, "need_clarification": False, "reasoning": "关键词匹配"}
    elif any(keyword in user_question for keyword in HARVEST_KEYWORDS):
        return {"intent_type": "harvest_planning", "need_rag": True, "need_clarification": False, "reasoning": "关键词匹配"}

    # 天气查询意图
    elif any(keyword in user_question for keyword in WEATHER_KEYWORDS):
        return {"intent_type": "weather_query", "need_rag": False, "need_clarification": False, "reasoning": "关键词匹配"}

    # 财务查询意图
    elif any(keyword in user_question for keyword in FINANCE_KEYWORDS):
        return {"intent_type": "finance_query", "need_rag": False, "need_clarification": False, "reasoning": "关键词匹配"}

    # 地块管理意图
    elif any(keyword in user_question for keyword in FIELD_KEYWORDS):
        return {"intent_type": "field_management", "need_rag": False, "need_clarification": False, "reasoning": "关键词匹配"}

    # 作物监测意图（摄像头拍照分析）
    elif any(keyword in user_question for keyword in CROP_MONITOR_KEYWORDS):
        return {"intent_type": "crop_monitoring", "need_rag": False, "need_clarification": False, "reasoning": "关键词匹配"}

    # 默认：意图不明
    return {"intent_type": "unclear", "need_rag": False, "need_clarification": True, "reasoning": "无法识别意图"}
