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
    state.progress_message = "正在分析您的意图..."
    user_question = state.user_question or ""

    # 如果 parse_input 已通过语音指令预设了意图，跳过 LLM 分类
    # 所有非 unclear 预设意图都信任 parse_input 的判断，避免被 LLM 覆盖
    if state.intent_type and state.intent_type not in ("unclear",):
        state.need_clarification = False
        # 不需要 RAG 的意图：自行处理数据检索
        no_rag_intents = {"finance_query", "weather_query", "reminder_setup",
                          "progress_tracking", "field_management", "device_control",
                          "crop_monitoring", "image_analysis"}
        if state.intent_type in no_rag_intents:
            state.need_rag = False
        return state

    # 图片分析意图判断（优先）
    if state.has_image:
        state.intent_type = "image_analysis"
        state.need_rag = True
        return state

    # 通用意图快速判断（关键词匹配 — 仅用于不会误判的安全意图）
    # 注意：DEVICE_KEYWORDS 不在此处快速匹配，因为设备关键词极易在否定/描述语境中误触发
    # （如"近期未施肥"中的"施肥"会被错误识别为设备控制指令）
    # 设备控制意图统一走下方 LLM 分类，LLM 能理解对话上下文区分"帮我浇水"和"今天浇过水了"
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

    # 使用LLM进行意图推理（包含 device_control 等复杂意图）
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
- policy_query: 政策补贴查询（补贴、政策、补助、惠农、农业保险等）
- field_management: 地块管理（地块、农田、面积、位置、地图等）
- device_control: 设备控制（浇水、灌溉、施肥、通风、补光、加热、开关设备、控制设备等）
- unclear: 意图不明

【关键规则】：
1. 如果用户只输入一个作物名称（如"小麦"、"玉米"），而之前的对话正在讨论该作物的病虫害/病害问题，则意图应为 "disease_prevention"
2. 如果用户输入与之前对话主题相关，优先保持上下文连贯，不要视为"unclear"
3. 用户当前输入可能是对之前问题的补充确认
    4. 【重要】区分"创建任务/提醒"与"执行设备操作"：
       - "添加...任务"、"创建...任务"、"设置...提醒"、"帮我建一个..." → reminder_setup
       - "帮我浇水"、"开启灌溉"、"启动施肥"、"打开通风" → device_control
       - 关键信号词：出现"添加/创建/设置/新建 + 任务/提醒" → reminder_setup，不是 device_control！
       - 即使用户提到了浇水/施肥等词，只要前面有"添加任务"、"创建提醒"等词，意图就是 reminder_setup
       - reminder_setup 不需要澄清：如果用户明确表达了创建任务/提醒的意图（如"添加浇水任务"），即使没有指定作物，也不需要澄清（need_clarification=false）。系统会自动使用"未指定作物"创建任务。只有在完全无法判断用户想做什么时才设 need_clarification=true。

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
4. 是否需要进一步澄清？（注意：reminder_setup 中如果任务类型已明确（浇水/施肥/除草等），即使没有作物名也不需要澄清，系统会自动处理。只有完全无法判断用户意图时才设 need_clarification=true）

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
        # 使用平衡括号匹配提取 JSON 对象（避免非贪婪正则截断嵌套 JSON）
        json_str = _extract_balanced_json(content)
        if json_str:
            result = json.loads(json_str)
            return {
                "intent_type": result.get("intent_type", "unclear"),
                "need_rag": result.get("need_rag", True),
                "need_clarification": result.get("need_clarification", False),
                "reasoning": result.get("reasoning", "")
            }
    except Exception as e:
        logger.error(f"LLM意图分类失败: {e}", exc_info=True)

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

    # 政策补贴意图
    elif any(keyword in user_question for keyword in POLICY_KEYWORDS):
        return {"intent_type": "policy_query", "need_rag": True, "need_clarification": False, "reasoning": "关键词匹配"}

    # 地块管理意图
    elif any(keyword in user_question for keyword in FIELD_KEYWORDS):
        return {"intent_type": "field_management", "need_rag": False, "need_clarification": False, "reasoning": "关键词匹配"}

    # 作物监测意图（摄像头拍照分析）
    elif any(keyword in user_question for keyword in CROP_MONITOR_KEYWORDS):
        return {"intent_type": "crop_monitoring", "need_rag": False, "need_clarification": False, "reasoning": "关键词匹配"}

    # 设备控制意图（降级关键词 — 作为 LLM 分类失败后的最后兜底）
    elif any(keyword in user_question for keyword in DEVICE_KEYWORDS):
        return {"intent_type": "device_control", "need_rag": False, "need_clarification": False, "reasoning": "关键词匹配(降级)"}

    # 默认：意图不明
    return {"intent_type": "unclear", "need_rag": False, "need_clarification": True, "reasoning": "无法识别意图"}


def _extract_balanced_json(text: str) -> str | None:
    """使用括号平衡从文本中提取完整 JSON 对象，避免非贪婪正则截断嵌套 JSON"""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None
