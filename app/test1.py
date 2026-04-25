"""
智能种植规划与事件提醒 Agent
功能：
- 作物选择建议：根据地区、土壤、季节推荐适合的作物
- 种植时间规划：提供科学的种植时间表
- 种植方法指导：详细的农事操作指导
- 农事提醒设置：浇水、施肥、除草等提醒管理
- 进度跟踪查询：记录和管理种植进度
- 病虫害防治：诊断和防治建议
- 收获规划建议：最佳收获时间和方法
"""

import os
import sys
import re
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import streamlit as st

# LangChain 相关导入
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
import dotenv

# 添加父目录到路径以导入其他模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 使用简化版农业知识RAG（无需Embeddings）
from knowledge.simple_agriculture_rag import SimpleAgricultureRAG

# 导入种植规划和提醒系统
from core.planting_planner import PlantingPlanner, generate_planting_plan
from core.reminder_system import ReminderSystem, create_watering_reminder, create_fertilizing_reminder
from core.planting_tracker import PlantingTracker, create_planting_task, create_planting_progress

# 导入天气和财务模块
from core.weather_service import WeatherService, get_weather_advice_for_crop
from core.finance_manager import FinanceManager, get_crop_profit

# 导入地图管理模块
from core.map_manager import MapManager, create_folium_map, extract_polygon_from_map_data

# =========================
# 环境变量加载 & 配置项
# =========================
dotenv.load_dotenv()

# -------------------------------------------
# LLM对话模型配置
# -------------------------------------------
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")

# 兼容旧变量名
OPENAI_API_KEY = LLM_API_KEY
OPENAI_BASE_URL = LLM_BASE_URL
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

if not LLM_API_KEY:
    raise EnvironmentError("未检测到 LLM_API_KEY 环境变量！")

# -------------------------------------------
# Vision视觉模型配置（独立配置，默认使用LLM配置）
# -------------------------------------------
VISION_API_KEY = os.getenv("VISION_API_KEY") or LLM_API_KEY
VISION_BASE_URL = os.getenv("VISION_BASE_URL") or LLM_BASE_URL
VISION_MODEL = os.getenv("VISION_MODEL") or LLM_MODEL  # 如果不设置，默认使用LLM模型
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
AGRICULTURE_FAISS_PATH = os.getenv("AGRICULTURE_FAISS_PATH", "agriculture_faiss_index")
DATA_STORAGE_DIR = os.getenv("DATA_STORAGE_DIR", "data")

# 记忆配置
SHORT_MEMORY_TOP_K = int(os.getenv("SHORT_MEMORY_TOP_K", "5"))
SUMMARY_TRIGGER_ROUNDS = int(os.getenv("SUMMARY_TRIGGER_ROUNDS", "3"))

# 功能开关
ENABLE_IMAGE_ANALYSIS = os.getenv("ENABLE_IMAGE_ANALYSIS", "true").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# 通用意图关键词（保留）
GREETING_KEYWORDS = ["你好", "您好", "嗨", "哈喽", "早上好", "下午好", "晚上好"]
THANKS_KEYWORDS = ["谢谢", "感谢", "多谢", "辛苦了"]
FAREWELL_KEYWORDS = ["再见", "拜拜", "下次见", "回见"]
IDENTITY_KEYWORDS = ["你是谁", "你叫什么", "名字", "身份"]
FUNCTION_KEYWORDS = ["你能做什么", "功能", "能干什么", "帮助", "作用"]
GENERAL_KEYWORDS = GREETING_KEYWORDS + THANKS_KEYWORDS + FAREWELL_KEYWORDS + IDENTITY_KEYWORDS + FUNCTION_KEYWORDS

# 种植规划意图关键词（新增）
CROP_SELECTION_KEYWORDS = ["种什么", "适合种", "推荐作物", "种哪种", "作物选择", "种植品种", "适合种什么"]
PLANTING_SCHEDULE_KEYWORDS = ["什么时候种", "种植时间", "播种时间", "季节", "月份", "时机", "几月份种"]
PLANTING_METHOD_KEYWORDS = ["怎么种", "种植方法", "技术", "栽培", "管理", "步骤", "如何种植", "如何栽培"]
REMINDER_KEYWORDS = ["提醒", "通知", "浇水", "施肥", "除草", "打药", "防治", "设置提醒", "添加提醒"]
PROGRESS_KEYWORDS = ["进度", "记录", "生长", "阶段", "里程碑", "跟踪", "现在该做什么", "进展情况"]
DISEASE_KEYWORDS = ["病虫害", "病害", "虫害", "防治", "治疗", "预防", "生病了", "叶子发黄"]
HARVEST_KEYWORDS = ["收获", "收割", "采摘", "成熟", "产量", "收获期", "什么时候收"]

# 天气相关意图关键词（新增）
WEATHER_KEYWORDS = ["天气", "气温", "下雨", "预报", "气象", "霜冻", "台风", "干旱", "降水"]

# 财务相关意图关键词（新增）
FINANCE_KEYWORDS = ["成本", "收入", "花费", "赚钱", "盈亏", "财务", "记账", "支出", "收益", "利润", "报表"]

# 地块管理意图关键词（新增）
FIELD_KEYWORDS = ["地块", "农田", "位置", "面积", "边界", "定位", "地图", "田地", "土地", "测量"]

# 长记忆摘要提示词
SUMMARY_PROMPT = """
请总结以下种植规划对话的核心信息，要求：
1. 保留关键信息：用户所在地区、种植的作物、土壤类型、农场面积、当前生长阶段
2. 保留用户的种植目标和关注点
3. 去除冗余内容，只保留有价值的信息
4. 忽略无关的寒暄内容

对话历史：
{conversation_history}

当前时间：{current_time}

总结要求：仅输出总结内容，不要额外解释
"""

# =========================
# 工具函数
# =========================
def trim_short_memory(messages: List[BaseMessage], top_k: int = SHORT_MEMORY_TOP_K) -> List[BaseMessage]:
    """手动修剪短记忆，兼容所有 LangChain 版本"""
    if not messages:
        return []
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    conversation_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    keep_count = top_k * 2
    trimmed_conversation = conversation_messages if len(conversation_messages) <= keep_count else conversation_messages[-keep_count:]
    return system_messages + trimmed_conversation

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
    response = llm.invoke([HumanMessage(content=summary_input)])
    return response.content.strip()

# =========================
# 数据模型定义
# =========================
class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    user_question: Optional[str] = None
    intent_type: Optional[Literal[
        # 通用意图（保留）
        "greeting",          # 问候
        "thanks",            # 感谢
        "farewell",          # 告别
        "identity",          # 身份询问
        "function",          # 功能询问
        # 种植规划意图（新增）
        "crop_selection",     # 作物选择建议
        "planting_schedule",  # 种植时间规划
        "planting_method",    # 种植方法指导
        "reminder_setup",     # 提醒设置管理
        "progress_tracking",  # 进度跟踪查询
        "disease_prevention", # 病虫害防治
        "harvest_planning",   # 收获规划建议
        "image_analysis",     # 图片分析（新增）
        "weather_query",      # 天气查询（新增）
        "finance_query",      # 财务查询（新增）
        "field_management",   # 地块管理（新增）
        "unclear"             # 意图不明
    ]] = None
    short_term_facts: Dict[str, Any] = Field(default_factory=dict)
    long_term_profile: Dict[str, Any] = Field(default_factory=lambda: {
        "summary": "",
        "conversation_round": 0,
        "user_profile": {}  # 用户档案（扩展）
    })
    need_rag: bool = False
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    need_clarification: bool = False
    refuse_answer: bool = False
    final_answer: Optional[str] = None

    # 种植规划相关字段（新增）
    planting_plan: Dict[str, Any] = Field(default_factory=lambda: {
        "crops": [],           # 选择的作物列表
        "schedule": {},        # 种植时间表
        "methods": {},         # 种植方法
        "progress": {},        # 进度跟踪
        "created_at": None     # 计划创建时间
    })

    reminders: List[Dict[str, Any]] = Field(default_factory=list)  # 提醒列表

    user_profile: Dict[str, Any] = Field(default_factory=lambda: {
        "region": "",          # 地区
        "climate": "",         # 气候类型
        "soil_type": "",       # 土壤类型
        "farm_size": 0,        # 农场面积
        "experience": "",      # 种植经验
        "goals": []            # 种植目标
    })

    # 图片分析相关字段（新增）
    image_data: Optional[str] = None           # base64编码的图片
    image_mime_type: Optional[str] = None      # 图片类型
    image_analysis_result: Optional[Dict[str, Any]] = Field(default_factory=dict)  # 分析结果
    has_image: bool = False                     # 是否有图片输入

    # 地块管理相关字段（新增）
    fields_data: List[Dict[str, Any]] = Field(default_factory=list)  # 地块列表
    current_field_id: Optional[str] = None     # 当前选中的地块ID

# =========================
# LangGraph 节点函数
# =========================
def parse_user_input(state: AgentState) -> AgentState:
    """解析用户输入，提取问题，同步用户信息到短期记忆"""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            state.user_question = msg.content.strip()
            break

    # 同步用户档案到短期记忆（确保 LLM 能看到用户信息）
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

    # 递增对话轮数
    state.long_term_profile["conversation_round"] = state.long_term_profile.get("conversation_round", 0) + 1

    # 保留更多消息以维持上下文（从5增加到8）
    state.messages = trim_short_memory(state.messages, 8)
    return state

def classify_intent(state: AgentState) -> AgentState:
    """
    意图分类节点：使用LLM进行智能意图推理
    保留关键词匹配作为快速路径，复杂意图使用LLM推理
    """
    user_question = state.user_question or ""

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
- field_management: 地块管理（地块、农田、面积、位置、地图等）
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
            temperature=0.1,  # 低温度确保稳定的分类结果
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )

        response = llm.invoke([HumanMessage(content=intent_prompt)])
        content = response.content

        # 解析JSON结果
        import json
        import re

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
        if DEBUG_MODE:
            print(f"LLM意图分类失败: {e}")

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

    # 默认：意图不明
    return {"intent_type": "unclear", "need_rag": False, "need_clarification": True, "reasoning": "无法识别意图"}

def extract_facts_from_conversation(state: AgentState) -> Dict[str, Any]:
    """从当前对话中提取关键事实"""
    user_question = state.user_question or ""
    facts = {}

    # 提取地区
    region_match = re.search(r'([\u4e00-\u9fa5]{2,10}(?:省|市|县|区|地区))', user_question)
    if region_match:
        facts["region"] = region_match.group(1)

    # 提取作物
    crops = ["小麦", "玉米", "水稻", "大豆", "棉花", "土豆", "红薯", "番茄", "黄瓜", "茄子", "辣椒", "白菜", "萝卜", "胡萝卜", "菠菜", "生菜", "芹菜", "韭菜", "大葱", "大蒜", "洋葱", "南瓜", "西瓜", "甜瓜", "草莓", "葡萄", "苹果", "梨", "桃", "李子", "杏", "樱桃", "枣", "柿子", "核桃", "板栗", "茶叶", "烟草", "花生", "油菜", "芝麻", "向日葵", "甘蔗", "甜菜"]
    for crop in crops:
        if crop in user_question:
            facts["crop"] = crop
            break

    # 提取面积
    area_match = re.search(r'(\d+(?:\.\d+)?)\s*[亩分]', user_question)
    if area_match:
        facts["farm_size"] = float(area_match.group(1))

    # 提取土壤类型
    soils = ["壤土", "砂土", "粘土", "沙壤土", "黏壤土", "黑土", "黄土", "红土", "水稻土"]
    for soil in soils:
        if soil in user_question:
            facts["soil_type"] = soil
            break

    # 如果用户只输入了一个作物名称，检查之前的对话是否有病虫害相关的上下文
    # 如果是，则保持病虫害相关的意图
    if facts.get("crop") and len(user_question) <= 5:
        # 检查之前的对话是否涉及病虫害
        for msg in state.messages[-4:]:
            if isinstance(msg, HumanMessage):
                content = msg.content
                if any(word in content for word in ["发黄", "病害", "虫害", "病", "虫", "叶子", "枯萎", "斑点"]):
                    facts["context_disease_discussion"] = True
                    break

    return facts


def llm_response_node(state: AgentState) -> AgentState:
    """
    统一 LLM 回答节点 - 所有回复都通过 LLM 生成
    并自动提取和记忆关键信息
    """
    intent = state.intent_type
    user_question = state.user_question or ""
    long_memory = state.long_term_profile.get("summary", "")
    user_profile = state.user_profile

    # 从当前对话提取关键事实并更新 short_term_facts（累积，不覆盖已有信息）
    new_facts = extract_facts_from_conversation(state)
    for key, value in new_facts.items():
        # 只在值为空时才更新，避免覆盖已有信息
        if not state.short_term_facts.get(key):
            state.short_term_facts[key] = value
            # 同时更新用户档案
            if key in ["region", "soil_type", "farm_size", "experience", "goals"]:
                user_profile[key] = value

    # 构建已收集的信息摘要
    collected_info = []
    if state.short_term_facts.get("region") or user_profile.get("region"):
        collected_info.append(f"地区：{state.short_term_facts.get('region', user_profile.get('region', ''))}")
    if state.short_term_facts.get("crop"):
        collected_info.append(f"作物：{state.short_term_facts.get('crop')}")
    if state.short_term_facts.get("farm_size") or user_profile.get("farm_size"):
        collected_info.append(f"面积：{state.short_term_facts.get('farm_size', user_profile.get('farm_size', ''))}亩")
    if state.short_term_facts.get("soil_type") or user_profile.get("soil_type"):
        collected_info.append(f"土壤：{state.short_term_facts.get('soil_type', user_profile.get('soil_type', ''))}")

    # 构建完整的对话历史（包括最近的几轮对话）
    recent_history = []
    for msg in state.messages[-10:]:  # 取最近10条消息
        if isinstance(msg, HumanMessage):
            recent_history.append(f"用户：{msg.content}")
        elif isinstance(msg, AIMessage):
            recent_history.append(f"助手：{msg.content[:100]}...")

    history_text = "\n".join(recent_history[-6:]) if recent_history else "暂无对话历史"

    # 构建系统提示词
    system_prompt = f"""你是一位专业的智能种植规划助手。请根据用户的意图和问题，提供自然、友好、专业的回答。

【当前用户意图类型】：{intent}

【用户档案】（来自用户填写的基础信息）：
- 地区：{user_profile.get('region', '未填写')}
- 土壤类型：{user_profile.get('soil_type', '未填写')}
- 种植面积：{user_profile.get('farm_size', '未填写')} 亩
- 种植经验：{user_profile.get('experience', '未填写')}
- 种植目标：{', '.join(user_profile.get('goals', [])) if user_profile.get('goals') else '未填写'}

【本次对话已收集的信息】：
{chr(10).join(collected_info) if collected_info else '暂无新信息'}

【对话历史】（最近几轮）：
{history_text}

【历史摘要】：
{long_memory if long_memory else '暂无历史摘要'}

【请遵循以下规则】：
1. 【自然对话】像真人一样自然交流，避免机械化的回复
2. 【记住信息】**这是最重要的规则**：
   - 如果【本次对话已收集的信息】中已经包含地区、作物等信息，绝对不要再次询问！
   - 直接基于已知信息给出专业建议
   - 只有在信息缺失时，才询问缺失的部分
3. 【上下文连贯性】**关键规则**：
   - 仔细看【对话历史】，理解之前讨论的主题
   - 用户输入简短内容（如只输入作物名）通常是对之前话题的确认或补充
   - 保持话题连贯，不要突然切换到新话题
   - 如果之前正在讨论病虫害问题，用户输入作物名，应继续讨论该作物的病虫害
4. 【专业准确】提供准确的农业种植知识和建议
5. 【个性化】根据用户的地区、作物等信息提供定制化建议

【意图类型说明】：
- greeting: 用户问候，请友好回应并询问有什么可以帮助
- thanks: 用户感谢，请礼貌回应并询问是否还有其他问题
- farewell: 用户告别，请友好道别并祝种植顺利
- identity: 用户询问身份，请介绍自己作为智能种植规划助手
- function: 用户询问功能，请介绍你能提供的种植相关服务
- crop_selection: 作物选择建议，请根据地区、土壤等推荐作物
- planting_schedule: 种植时间规划，请提供播种、收获等时间建议
- planting_method: 种植方法指导，请提供详细的栽培技术指导
- reminder_setup: 提醒设置管理，请帮助设置浇水、施肥等提醒
- progress_tracking: 进度跟踪查询，请查看并更新种植进度
- disease_prevention: 病虫害防治，请提供诊断和防治建议。如果用户只输入作物名且之前正在讨论病害，继续分析该作物的病害问题
- harvest_planning: 收获规划建议，请提供最佳收获时间和方法
- image_analysis: 图片分析，请分析上传的农作物图片
- unclear: 意图不明，请礼貌询问用户具体需求

现在请回复用户的问题："""

    # 如果有检索到的知识，添加到系统提示词中
    if state.retrieved_docs:
        knowledge_text = "\n".join([f"- {doc['page_content'][:200]}" for doc in state.retrieved_docs[:3]])
        system_prompt += f"\n\n【相关知识】：\n{knowledge_text}"

    # 构建消息 - 系统提示 + 历史对话 + 当前问题
    messages = [SystemMessage(content=system_prompt)]

    # 添加历史对话（最近6轮，不包括系统消息）
    history_messages = [msg for msg in state.messages if isinstance(msg, (HumanMessage, AIMessage))]

    # 找到当前问题的位置
    current_msg_index = -1
    for i, msg in enumerate(history_messages):
        if isinstance(msg, HumanMessage) and msg.content == user_question:
            current_msg_index = i
            break

    # 添加历史消息（最多6轮 = 12条消息，不包括当前这条）
    if current_msg_index > 0:
        history_start = max(0, current_msg_index - 12)
        for i in range(history_start, current_msg_index):
            messages.append(history_messages[i])
    elif len(history_messages) > 1:
        # 如果没找到当前问题，添加最近的历史（最后一条是当前的）
        for msg in history_messages[-13:-1]:  # 最多12条历史
            messages.append(msg)

    # 确保最后一条是当前用户问题
    if not (isinstance(messages[-1], HumanMessage) and messages[-1].content == user_question):
        messages.append(HumanMessage(content=user_question))

    # 调用 LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    try:
        response = llm.invoke(messages)
        state.final_answer = response.content
    except Exception as e:
        if DEBUG_MODE:
            print(f"LLM 调用失败: {e}")
        state.final_answer = "抱歉，我暂时无法回答，请稍后再试。"

    state.messages.append(AIMessage(content=state.final_answer))
    return state


# 保留原函数名兼容性，但内部调用 LLM
def general_response_node(state: AgentState) -> AgentState:
    """通用回复节点 - 现在通过 LLM 生成所有回复"""
    return llm_response_node(state)

def clarification_node(state: AgentState) -> AgentState:
    """
    使用 LLM 生成动态追问引导
    """
    user_question = state.user_question or ""
    intent = state.intent_type
    long_memory = state.long_term_profile.get("summary", "")
    user_profile = state.user_profile
    short_facts = state.short_term_facts

    # 检查已收集的信息
    has_region = bool(short_facts.get('region') or user_profile.get('region'))
    has_crop = bool(short_facts.get('crop'))
    has_soil = bool(short_facts.get('soil_type') or user_profile.get('soil_type'))
    has_area = bool(short_facts.get('farm_size') or user_profile.get('farm_size'))

    # 构建系统提示词让 LLM 生成追问
    clarify_prompt = f"""你是一位专业的种植规划顾问。用户的意图不够明确，需要你礼貌地询问更多信息。

【用户当前意图】：{intent}

【用户已提供的信息】：
- 地区：{'已提供：' + (short_facts.get('region') or user_profile.get('region', '')) if has_region else '未提供'}
- 作物：{'已提供：' + short_facts.get('crop', '') if has_crop else '未提供'}
- 土壤类型：{'已提供：' + (short_facts.get('soil_type') or user_profile.get('soil_type', '')) if has_soil else '未提供'}
- 种植面积：{'已提供：' + str(short_facts.get('farm_size') or user_profile.get('farm_size', '')) + '亩' if has_area else '未提供'}
- 种植经验：{user_profile.get('experience', '未提供')}
- 用户问题：{user_question}

【对话历史】：{long_memory}

【重要要求】：
1. **绝不要重复询问用户已经提供的信息**
2. 只询问缺失的关键信息（用"未提供"标记的）
3. 如果所有关键信息都已收集，直接说"请详细描述您的需求"
4. 语气友好自然，像真人顾问一样

【意图所需的最低信息】：
- crop_selection: 需要地区和作物（地区已从用户档案获取，如已提供则不再问）
- planting_schedule: 需要作物和地区
- planting_method: 需要作物
- reminder_setup: 需要作物和提醒类型
- progress_tracking: 需要作物
- disease_prevention: 需要作物和症状
- harvest_planning: 需要作物
- unclear: 询问具体想咨询哪方面的问题

请生成追问话术（只问缺失的信息）："""

    # 调用 LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.7,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    try:
        response = llm.invoke([HumanMessage(content=clarify_prompt)])
        clarify_msg = response.content.strip()
    except Exception as e:
        if DEBUG_MODE:
            print(f"LLM 追问生成失败: {e}")
        # 降级到简单追问
        clarify_msg = "为了更好地帮助您，能否告诉我更多信息？比如您所在的地区和想种植的作物？"

    state.final_answer = clarify_msg
    state.messages.append(AIMessage(content=clarify_msg))
    return state

def rag_retrieval_node(state: AgentState, rag_system: SimpleAgricultureRAG) -> AgentState:
    """RAG 检索节点 - 使用简化版农业知识检索，支持图片分析结果检索"""
    queries = []

    # 添加用户原始问题
    if state.user_question:
        queries.append(state.user_question)

    # 如果有图片分析结果，添加相关检索词
    if state.image_analysis_result:
        crop_type = state.image_analysis_result.get("crop_type", "")
        for issue in state.image_analysis_result.get("detected_issues", []):
            issue_name = issue.get("name", "")
            if crop_type and issue_name:
                queries.append(f"{crop_type}{issue_name}防治方法")

    # 执行检索
    all_results = []
    for query in queries[:2]:  # 最多检索2个查询
        if state.need_rag:
            try:
                results = rag_system.search(query, k=RAG_TOP_K)
                for result in results:
                    doc = {"page_content": result["content"], "source": result["metadata"].get("crop", "未知作物")}
                    if doc not in all_results:  # 去重
                        all_results.append(doc)
            except Exception as e:
                print(f"RAG 检索出错: {e}")

    state.retrieved_docs = all_results if all_results else []
    return state

def image_analysis_node(state: AgentState) -> AgentState:
    """图片分析节点 - 使用多模态LLM分析农作物图片"""
    if not state.has_image or not state.image_data:
        return state

    # 构建多模态提示词
    system_prompt = """你是一位专业的农业病虫害诊断专家。请仔细分析用户上传的农作物图片，识别以下问题：

1. **病虫害识别**：
   - 病害类型（真菌病、细菌病、病毒病等）
   - 虫害类型（蚜虫、红蜘蛛、螟虫等）
   - 严重程度（轻微/中等/严重）
   - 置信度评估

2. **生长阶段判断**：
   - 当前作物种类
   - 生长阶段（苗期、拔节期、开花期、成熟期等）

3. **营养/健康问题**：
   - 缺素症状（缺氮、缺磷、缺钾等）
   - 水分状况
   - 整体健康评分

请以JSON格式返回分析结果：
{
    "crop_type": "作物名称",
    "growth_stage": "生长阶段",
    "detected_issues": [
        {
            "type": "病害/虫害/营养问题",
            "name": "具体问题名称",
            "severity": "轻微/中等/严重",
            "confidence": 0.85,
            "description": "症状描述"
        }
    ],
    "overall_health": "良好/一般/较差",
    "recommendations": ["建议措施1", "建议措施2"],
    "urgency": "立即处理/近期处理/持续观察"
}"""

    # 调用多模态LLM
    llm = ChatOpenAI(
        model=VISION_MODEL,  # 使用环境变量配置的多模态模型
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    # 构建多模态消息
    from langchain_core.messages import HumanMessage

    message = HumanMessage(
        content=[
            {"type": "text", "text": system_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{state.image_mime_type};base64,{state.image_data}"
                }
            },
            {"type": "text", "text": f"用户问题：{state.user_question or '请分析这张农作物图片'}"}
        ]
    )

    try:
        response = llm.invoke([message])
        # 解析JSON结果
        import json
        import re

        # 提取JSON部分
        content = response.content
        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = content

        analysis_result = json.loads(json_str)
        state.image_analysis_result = analysis_result

    except Exception as e:
        state.image_analysis_result = {
            "error": f"图片分析失败: {str(e)}",
            "raw_response": response.content if 'response' in locals() else ""
        }

    return state

def llm_expert_answer(state: AgentState) -> AgentState:
    """种植专家回答节点"""
    long_memory = state.long_term_profile.get("summary", "")
    memory_context = f"\n【对话历史总结】：{long_memory}\n" if long_memory else ""

    # 从short_term_facts获取用户上下文
    user_context = ""
    if state.short_term_facts:
        user_context = "\n【用户已知信息】\n"
        for key, value in state.short_term_facts.items():
            user_context += f"  - {key}: {value}\n"

    system_prompt = f"""
你是一位经验丰富的农业种植专家，请遵循以下规则：

1. 【基于知识回答】优先使用提供的农业知识回答问题
2. 【科学实用】建议要科学、实用、可操作，适合农户实际执行
3. 【因地制宜】考虑用户的地区、土壤、气候等条件给出建议
4. 【风险提示】对可能的风险（病虫害、天气等）给出预警和防范建议
5. 【通俗易懂】使用朴实易懂的语言，避免过于学术化

【回答格式】
- 对于种植时间：明确说明播种期和收获期
- 对于种植方法：分步骤说明关键操作
- 对于病虫害：描述症状 + 防治方法
- 对于不确定的问题：诚实说明，建议咨询当地农技站

【对话上下文】
{memory_context}
{user_context}
"""

    # 构造知识证据
    evidence_blocks = ""
    if state.retrieved_docs:
        aggregated = aggregate_sentences(state.retrieved_docs)
        evidence_blocks = "\n【检索到的农业知识】\n"
        for i, item in enumerate(aggregated, 1):
            evidence_blocks += f"\n【知识{i}｜{item['source']}】\n{item['content']}\n"

    # 调用LLM - 正确传递对话历史
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    # 构建消息列表
    messages = [SystemMessage(content=system_prompt)]

    # 添加历史对话（最近6轮）
    user_question = state.user_question or ""
    current_msg_index = -1
    for i, msg in enumerate(state.messages):
        if isinstance(msg, HumanMessage) and msg.content == user_question:
            current_msg_index = i
            break

    history_start = max(0, current_msg_index - 12) if current_msg_index >= 0 else max(0, len(state.messages) - 12)
    for i in range(history_start, current_msg_index if current_msg_index >= 0 else len(state.messages)):
        msg = state.messages[i]
        if isinstance(msg, (HumanMessage, AIMessage)):
            messages.append(msg)

    # 添加知识证据
    if evidence_blocks:
        messages.append(HumanMessage(content=evidence_blocks))

    # 添加当前问题
    if not (isinstance(messages[-1], HumanMessage) and messages[-1].content == user_question):
        messages.append(HumanMessage(content=user_question))

    response = llm.invoke(messages)

    state.final_answer = response.content
    state.messages.append(AIMessage(content=response.content))
    return state

def image_analysis_answer_node(state: AgentState) -> AgentState:
    """图片分析回答节点 - 整合图片分析和RAG知识生成回答"""
    analysis = state.image_analysis_result
    docs = state.retrieved_docs

    # 构建分析结果展示
    answer_parts = []

    # 1. 图片分析结果摘要
    if analysis.get("crop_type"):
        answer_parts.append(f"🌾 **识别作物**: {analysis['crop_type']}")
    if analysis.get("growth_stage"):
        answer_parts.append(f"📈 **生长阶段**: {analysis['growth_stage']}")
    if analysis.get("overall_health"):
        answer_parts.append(f"💚 **整体健康**: {analysis['overall_health']}")

    # 2. 检测到的问题
    if analysis.get("detected_issues"):
        answer_parts.append("\n🔍 **检测到的问题**:")
        for issue in analysis["detected_issues"]:
            severity_emoji = {"轻微": "⚪", "中等": "🟡", "严重": "🔴"}
            emoji = severity_emoji.get(issue.get("severity", ""), "⚪")
            conf = issue.get("confidence", 0)
            answer_parts.append(
                f"  {emoji} **{issue['name']}** ({issue['type']})\\n"
                f"     严重程度: {issue['severity']} | 置信度: {conf:.0%}\\n"
                f"     {issue.get('description', '')}"
            )

    # 3. 处理建议
    if analysis.get("recommendations"):
        answer_parts.append("\n💡 **处理建议**:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            answer_parts.append(f"  {i}. {rec}")

    # 4. 结合RAG知识补充
    if docs:
        answer_parts.append("\n📚 **相关知识**:")
        for doc in docs[:2]:
            answer_parts.append(f"  • {doc['page_content'][:100]}...")

    # 5. 紧急程度
    if analysis.get("urgency"):
        urgency_emoji = {"立即处理": "🚨", "近期处理": "⚠️", "持续观察": "👁️"}
        emoji = urgency_emoji.get(analysis["urgency"], "")
        answer_parts.append(f"\n{emoji} **紧急程度**: {analysis['urgency']}")

    # 处理错误情况
    if analysis.get("error"):
        answer_parts.append(f"\n⚠️ **分析出错**: {analysis['error']}")
        if analysis.get("raw_response"):
            answer_parts.append(f"\n原始响应: {analysis['raw_response'][:200]}...")

    state.final_answer = "\n".join(answer_parts)
    state.messages.append(AIMessage(content=state.final_answer))

    # 清理图片数据（避免内存占用）
    state.image_data = None
    state.has_image = False

    return state

def planting_plan_node(state: AgentState) -> AgentState:
    """种植规划节点 - 生成个性化种植计划，同时创建进度卡片和任务"""
    if state.intent_type in ["crop_selection", "planting_schedule"]:
        # 提取用户信息
        user_info = {
            "region": state.short_term_facts.get("region") or state.user_profile.get("region", ""),
            "soil_type": state.short_term_facts.get("soil_type") or state.user_profile.get("soil_type", ""),
            "farm_size": state.short_term_facts.get("farm_size") or state.user_profile.get("farm_size", 1.0),
            "goals": state.short_term_facts.get("goals") or state.user_profile.get("goals", []),
            "experience": state.user_profile.get("experience", ""),
            "crop": state.short_term_facts.get("crop", "")
        }

        try:
            # 生成种植计划
            planner = PlantingPlanner()
            plan = planner.generate_plan(user_info)

            # 更新AgentState
            state.planting_plan = {
                "crops": [plan.crop],
                "schedule": plan.schedule,
                "methods": {},
                "progress": {},
                "created_at": plan.created_at
            }

            # 更新用户档案
            if plan.region:
                state.user_profile["region"] = plan.region
            if plan.soil_type:
                state.user_profile["soil_type"] = plan.soil_type

            # 创建进度卡片和任务
            try:
                tracker = PlantingTracker()

                # 1. 创建整体种植进度记录
                current_stage = "准备期"
                stage_number = 0
                total_stages = len(plan.schedule.get("stages", [])) if plan.schedule else 1

                if plan.schedule and plan.schedule.get("stages"):
                    first_stage = plan.schedule["stages"][0]
                    current_stage = first_stage.get("stage", "准备期")
                    stage_number = 1

                # 创建进度记录
                progress = tracker.create_progress({
                    "crop": plan.crop,
                    "stage": current_stage,
                    "stage_number": stage_number,
                    "total_stages": total_stages,
                    "start_date": datetime.now().strftime("%Y-%m-%d"),
                    "expected_end_date": plan.schedule.get("harvest_time", ""),
                    "progress_percent": 0,
                    "status": "进行中",
                    "tasks": [],
                    "notes": f"种植面积: {plan.farm_size}亩, 地区: {plan.region}"
                })

                # 2. 为每个阶段的关键任务创建任务卡片
                if plan.tasks:
                    for stage_info in plan.tasks:
                        stage_name = stage_info.get("stage", "")
                        for task_info in stage_info.get("tasks", [])[:2]:  # 每个阶段最多2个任务
                            task_date = task_info.get("date", "")
                            task_name = task_info.get("task", "")
                            priority = task_info.get("priority", "中")

                            tracker.create_task({
                                "crop": plan.crop,
                                "task_type": task_name[:4] if len(task_name) >= 4 else task_name,
                                "title": f"{stage_name} - {task_name}",
                                "description": f"{plan.crop}的{stage_name}阶段任务",
                                "status": "待办",
                                "priority": "high" if priority == "高" else "medium",
                                "end_date": task_date,
                                "progress_percent": 0
                            })

                # 3. 添加资源准备任务（种子、肥料等）
                if plan.resources:
                    if plan.resources.get("seeds"):
                        tracker.create_task({
                            "crop": plan.crop,
                            "task_type": "播种",
                            "title": f"准备{plan.crop}种子",
                            "description": f"需准备: {plan.resources['seeds'].get('amount', '适量')}",
                            "status": "待办",
                            "priority": "high",
                            "end_date": plan.schedule.get("sowing_time", datetime.now().strftime("%Y-%m-%d")),
                            "progress_percent": 0
                        })

            except Exception as e:
                print(f"创建进度卡片失败: {e}")

            # 格式化回答
            answer = planner.format_plan_as_text(plan)
            state.final_answer = answer
            state.messages.append(AIMessage(content=answer))

        except Exception as e:
            state.final_answer = f"生成种植计划时出现错误：{str(e)}。请稍后再试或联系技术支持。"
            state.messages.append(AIMessage(content=state.final_answer))

    return state


def reminder_management_node(state: AgentState) -> AgentState:
    """提醒管理节点 - 创建和管理农事提醒"""
    if state.intent_type == "reminder_setup":
        user_question = state.user_question or ""

        # 从问题中提取关键信息
        crop = state.short_term_facts.get("crop", "")
        reminder_type = "其他"

        # 识别提醒类型
        if "浇水" in user_question or "灌水" in user_question or "灌溉" in user_question:
            reminder_type = "浇水"
        elif "施肥" in user_question or "追肥" in user_question:
            reminder_type = "施肥"
        elif "除草" in user_question:
            reminder_type = "除草"
        elif "病" in user_question or "虫" in user_question or "防治" in user_question:
            reminder_type = "病虫害防治"
        elif "修剪" in user_question or "整枝" in user_question:
            reminder_type = "修剪"
        elif "收获" in user_question or "收割" in user_question or "采摘" in user_question:
            reminder_type = "收获"

        # 识别频率
        frequency = "单次"
        if "每天" in user_question or "每日" in user_question:
            frequency = "每天"
        elif "每周" in user_question:
            frequency = "每周"
        elif "每月" in user_question:
            frequency = "每月"

        # 识别时间
        time_of_day = "09:00"
        import re
        time_match = re.search(r'(\d{1,2})[:点](\d{0,2})', user_question)
        if time_match:
            hour = int(time_match.group(1))
            minute = time_match.group(2) or "00"
            if len(minute) < 2:
                minute += "0"
            time_of_day = f"{hour:02d}:{minute}"

        try:
            # 创建提醒
            system = ReminderSystem()
            reminder = system.create_reminder({
                "crop": crop or "未指定作物",
                "reminder_type": reminder_type,
                "task_description": f"给{crop or '作物'}{reminder_type}",
                "frequency": frequency,
                "time_of_day": time_of_day,
                "channels": ["app"]
            })

            # 添加到state
            state.reminders.append({
                "id": reminder.id,
                "crop": reminder.crop,
                "type": reminder.reminder_type,
                "next_trigger": reminder.next_trigger
            })

            # 同时创建任务卡片（用于前端展示）
            try:
                from datetime import timedelta
                tracker = PlantingTracker()
                end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                task_id = tracker.create_task({
                    "crop": crop or "未指定作物",
                    "task_type": reminder_type,
                    "title": f"{reminder_type} - {crop or '作物'}",
                    "description": f"给{crop or '作物'}{reminder_type}，频率：{frequency}",
                    "status": "待办",
                    "priority": "medium",
                    "end_date": end_date,
                    "progress_percent": 0
                })
            except Exception as e:
                print(f"创建任务卡片失败: {e}")

            # 生成确认回答
            confirmation = f"[OK] 已为您设置农事提醒\n\n"
            confirmation += f"**作物**: {reminder.crop}\n"
            confirmation += f"**任务**: {reminder.reminder_type}\n"
            confirmation += f"**提醒时间**: {reminder.next_trigger}\n"
            confirmation += f"**频率**: {reminder.frequency}\n\n"
            confirmation += "您可以在侧边栏的提醒管理中查看和管理所有提醒。"

            state.final_answer = confirmation
            state.messages.append(AIMessage(content=confirmation))

        except Exception as e:
            state.final_answer = f"设置提醒时出现错误：{str(e)}。请稍后重试。"
            state.messages.append(AIMessage(content=state.final_answer))

    return state


def weather_query_node(state: AgentState) -> AgentState:
    """
    天气查询节点 - 获取天气预报和农事建议
    """
    if state.intent_type == "weather_query":
        user_question = state.user_question or ""

        # 获取用户地区（优先使用用户档案中的地区）
        location = (state.short_term_facts.get("region") or
                   state.user_profile.get("region", "北京"))

        # 获取当前作物
        crop = state.short_term_facts.get("crop") or state.user_profile.get("crop", "")

        # 获取生长阶段
        growth_stage = state.short_term_facts.get("growth_stage", "")

        try:
            # 初始化天气服务
            weather_service = WeatherService()

            # 获取当前天气
            current = weather_service.get_current_weather(location)

            # 获取未来5天预报
            forecast = weather_service.get_forecast(location, 5)

            # 获取农事建议
            farming_advice = weather_service.get_farming_advice(location, crop, growth_stage)

            # 获取预警信息
            alerts = weather_service.check_weather_alerts(location, crop)

            # 构建回答
            answer_parts = []

            # 1. 当前天气
            if current:
                answer_parts.append(weather_service.format_weather_report(current))

            # 2. 天气预警
            if alerts:
                answer_parts.append(weather_service.format_alert_report(alerts))

            # 3. 农事建议
            if farming_advice:
                answer_parts.append(weather_service.format_farming_advice(farming_advice))

            # 4. 未来3天简要预报
            if forecast:
                answer_parts.append("\n📅 **未来3天预报**：")
                for w in forecast[:3]:
                    answer_parts.append(f"   {w.date}: {w.weather_desc} {w.temperature_low}℃~{w.temperature_high}℃")

            state.final_answer = "\n".join(answer_parts)
            state.messages.append(AIMessage(content=state.final_answer))

        except Exception as e:
            state.final_answer = f"获取天气信息时出现错误：{str(e)}。请检查天气服务配置。"
            state.messages.append(AIMessage(content=state.final_answer))

    return state


def finance_query_node(state: AgentState) -> AgentState:
    """
    财务查询节点 - 处理成本和收入查询、生成财务报表
    """
    if state.intent_type == "finance_query":
        user_question = state.user_question or ""

        # 获取当前作物
        crop = state.short_term_facts.get("crop") or state.user_profile.get("crop", "")

        # 解析查询意图（记账 vs 查询）
        is_record_request = any(word in user_question for word in ["记", "添加", "录入", "花了", "收入"])
        is_report_request = any(word in user_question for word in ["报表", "报告", "汇总", "统计"])

        try:
            # 初始化财务管理
            finance_manager = FinanceManager()

            if is_record_request:
                # 处理记账请求 - 这里简化处理，实际应该解析具体金额
                state.final_answer = """💰 **记账功能**

请在侧边栏"财务管理"中记录详细的成本和收入信息。

支持记录：
• 种子、肥料、农药等成本支出
• 作物销售收入
• 查看亩均成本和收益分析

您也可以导入CSV文件批量导入历史财务数据。"""

            elif is_report_request:
                # 生成年度报表
                report = finance_manager.get_annual_report()
                state.final_answer = finance_manager.format_annual_report(report)

            elif crop:
                # 查询特定作物的财务情况
                summary = finance_manager.get_crop_financial_summary(crop)
                if summary:
                    state.final_answer = finance_manager.format_summary_report(summary)
                else:
                    state.final_answer = f"📊 **{crop}财务记录**\n\n暂无{crop}的财务记录。\n\n请在侧边栏「财务管理」中添加成本或收入记录。"

            else:
                # 显示总体财务概况
                report = finance_manager.get_annual_report()
                if report['crop_reports']:
                    state.final_answer = finance_manager.format_annual_report(report)
                else:
                    state.final_answer = """📊 **财务概览**

暂无财务记录。

请在侧边栏"财务管理"中：
1. 记录各项成本支出（种子、肥料、人工等）
2. 记录作物销售收入
3. 查看收益分析报告

您也可以导入CSV文件批量导入历史数据。"""

            state.messages.append(AIMessage(content=state.final_answer))

        except Exception as e:
            state.final_answer = f"查询财务信息时出现错误：{str(e)}。请稍后重试。"
            state.messages.append(AIMessage(content=state.final_answer))

    return state


def field_management_node(state: AgentState) -> AgentState:
    """
    地块管理节点 - 处理地块查询和管理请求
    """
    if state.intent_type == "field_management":
        user_question = state.user_question or ""

        try:
            # 初始化地图管理器
            map_manager = MapManager()
            fields = map_manager.get_all_fields()

            # 检查是否是查询请求
            is_query = any(word in user_question for word in ["多少", "几个", "哪里", "在哪", "查询", "查看", "显示"])

            if is_query or not fields:
                # 生成地块信息报告
                if fields:
                    answer_parts = ["📍 **我的地块信息**\n"]
                    total_area = 0
                    for i, field in enumerate(fields, 1):
                        answer_parts.append(f"\n**地块{i}：{field.name}**")
                        answer_parts.append(f"- 面积：{field.area_mu:.2f}亩")
                        answer_parts.append(f"- 位置：{field.center_lat:.4f}°N, {field.center_lon:.4f}°E")
                        if field.soil_type:
                            answer_parts.append(f"- 土壤：{field.soil_type}")
                        if field.current_crop:
                            answer_parts.append(f"- 当前作物：{field.current_crop}")
                        total_area += field.area_mu

                    answer_parts.append(f"\n---")
                    answer_parts.append(f"**总计**：{len(fields)}个地块，共{total_area:.2f}亩")
                    answer_parts.append(f"\n💡 您可以在侧边栏「我的地块」中管理和添加新地块")

                    state.final_answer = "\n".join(answer_parts)
                else:
                    state.final_answer = """📍 **地块管理**

您还没有添加任何地块。

请在侧边栏「我的地块」中：
1. 点击「添加新地块」
2. 在地图上绘制地块边界
3. 系统自动计算面积
4. 填写地块信息并保存

地块信息将用于：
- 精准天气预测
- 分区种植规划
- 面积和成本核算"""
            else:
                # 一般性地块管理介绍
                state.final_answer = """📍 **地块管理功能**

您可以通过以下方式管理您的农田地块：

**1. 添加地块**
- 在侧边栏点击「我的地块」
- 点击「添加新地块」按钮
- 在地图上绘制多边形边界
- 系统自动计算面积

**2. 地块信息**
- 记录土壤类型
- 标注当前作物
- 查看总面积统计

**3. 应用场景**
- 基于位置获取精准天气
- 分地块管理种植计划
- 按地块记录成本和收入

请问您想查看已有地块信息还是添加新地块？"""

            state.messages.append(AIMessage(content=state.final_answer))

        except Exception as e:
            state.final_answer = f"地块管理功能出现错误：{str(e)}。请稍后重试。"
            state.messages.append(AIMessage(content=state.final_answer))

    return state


def extract_and_create_tasks_node(state: AgentState) -> AgentState:
    """
    从LLL回答中提取建议并自动创建农事任务
    只针对病虫害防治、种植方法、收获规划等会产生可操作建议的意图
    """
    # 只处理会产生建议的意图类型
    actionable_intents = [
        "disease_prevention",   # 病虫害防治
        "planting_method",      # 种植方法
        "harvest_planning",     # 收获规划
        "reminder_setup",       # 提醒设置
        "image_analysis"        # 图片分析
    ]

    if state.intent_type not in actionable_intents:
        return state

    if not state.final_answer:
        return state

    try:
        # 获取作物名称
        crop = state.short_term_facts.get("crop") or state.user_profile.get("crop", "")

        # 使用LLM提取建议
        suggestions = extract_suggestions_from_answer(state.final_answer, crop)

        if suggestions:
            # 创建任务
            tracker = PlantingTracker()
            created_tasks = []

            for suggestion in suggestions:
                try:
                    task = tracker.create_task({
                        "crop": suggestion.get("crop", crop or "未指定作物"),
                        "task_type": suggestion.get("task_type", "其他"),
                        "title": suggestion.get("title", "农事任务"),
                        "description": suggestion.get("description", ""),
                        "status": "待办",
                        "priority": suggestion.get("priority", "medium"),
                        "end_date": suggestion.get("end_date", ""),
                        "progress_percent": 0
                    })
                    created_tasks.append(task)
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"创建任务失败: {e}")

            # 如果有成功创建的任务，在回答中添加提示
            if created_tasks:
                task_notice = "\n\n---\n📋 **已为您自动生成农事任务**:\n"
                for i, task in enumerate(created_tasks[:3], 1):  # 最多显示3个
                    task_notice += f"{i}. {task.title}\n"
                if len(created_tasks) > 3:
                    task_notice += f"... 还有 {len(created_tasks) - 3} 个任务已添加到任务列表\n"
                task_notice += "\n💡 您可以在侧边栏查看和管理所有任务"

                # 追加到回答中
                state.final_answer += task_notice
                # 更新最后一条消息
                if state.messages and isinstance(state.messages[-1], AIMessage):
                    state.messages[-1] = AIMessage(content=state.final_answer)

    except Exception as e:
        if DEBUG_MODE:
            print(f"提取建议并创建任务时出错: {e}")

    return state


def extract_suggestions_from_answer(answer: str, crop: str = "") -> List[Dict[str, Any]]:
    """
    使用LLM从回答中提取可执行的建议并转换为任务格式

    返回:
        [
            {
                "crop": "作物名称",
                "task_type": "任务类型",
                "title": "任务标题",
                "description": "任务描述",
                "priority": "high/medium/low",
                "end_date": "截止日期(YYYY-MM-DD格式)"
            }
        ]
    """
    # 构建提取提示词
    extract_prompt = f"""请从以下农业建议文本中提取可执行的具体农事任务。

【作物名称】: {crop if crop else "从文本中识别"}

【建议文本】:
{answer}

【提取要求】:
1. 只提取具体的、可操作的农事任务（如浇水、施肥、喷药、除草等）
2. 忽略一般性建议、解释说明、警告提示等非操作性内容
3. 每个任务需要明确：
   - 任务类型（浇水、施肥、病虫害防治、除草、修剪、收获等）
   - 任务标题（简短明确，如"喷施叶面肥"、"浇灌透水"）
   - 任务描述（具体操作步骤）
   - 优先级（high-紧急重要/medium-一般/low-可延后）
   - 建议完成时间（如"3天内"、"1周内"、"立即"等）
4. 如果文本中没有可执行的具体任务，返回空数组 []

【当前时间】: {datetime.now().strftime("%Y-%m-%d")}

请以下面JSON格式返回，只返回JSON，不要其他说明:
[
  {{
    "crop": "作物名称",
    "task_type": "任务类型",
    "title": "任务标题",
    "description": "任务描述",
    "priority": "high/medium/low",
    "timeframe": "时间描述"
  }}
]"""

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.2,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )

        response = llm.invoke([HumanMessage(content=extract_prompt)])
        content = response.content.strip()

        # 解析JSON
        import json
        import re

        # 尝试提取JSON数组
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            suggestions = json.loads(json_match.group())
        else:
            suggestions = json.loads(content)

        # 处理时间描述，转换为具体日期
        processed_suggestions = []
        for suggestion in suggestions:
            timeframe = suggestion.get("timeframe", "")
            end_date = calculate_end_date(timeframe)
            suggestion["end_date"] = end_date
            processed_suggestions.append(suggestion)

        return processed_suggestions

    except Exception as e:
        if DEBUG_MODE:
            print(f"提取建议失败: {e}")
        return []


def calculate_end_date(timeframe: str) -> str:
    """
    根据时间描述计算具体的截止日期
    """
    from datetime import timedelta

    timeframe = timeframe.lower() if timeframe else ""
    now = datetime.now()

    # 立即/马上
    if any(word in timeframe for word in ["立即", "马上", "即刻", "今天"]):
        return now.strftime("%Y-%m-%d")

    # 1-3天
    if any(word in timeframe for word in ["1天", "2天", "3天", "三天", "两天", "24小时", "48小时", "72小时"]):
        return (now + timedelta(days=2)).strftime("%Y-%m-%d")

    # 1周内
    if any(word in timeframe for word in ["1周", "一周", "7天", "周内", "本周"]):
        return (now + timedelta(days=5)).strftime("%Y-%m-%d")

    # 2周内
    if any(word in timeframe for word in ["2周", "两周", "14天", "半月"]):
        return (now + timedelta(days=10)).strftime("%Y-%m-%d")

    # 1个月内
    if any(word in timeframe for word in ["1月", "一个月", "30天", "本月"]):
        return (now + timedelta(days=20)).strftime("%Y-%m-%d")

    # 默认3天后
    return (now + timedelta(days=3)).strftime("%Y-%m-%d")


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

# =========================
# 构建 LangGraph 工作流
# =========================
def build_agricultural_policy_agent(rag_system: SimpleAgricultureRAG):
    """构建带记忆和增强通用能力的Agent"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("parse_input", parse_user_input)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("general_response", general_response_node)
    workflow.add_node("extract_tasks", extract_and_create_tasks_node)  # 新增：提取建议并创建任务
    workflow.add_node("update_long_memory", update_long_memory)
    workflow.add_node("clarify", clarification_node)
    workflow.add_node("rag_retrieval", lambda s: rag_retrieval_node(s, rag_system))
    workflow.add_node("generate_answer", llm_expert_answer)
    workflow.add_node("planting_plan", planting_plan_node)
    workflow.add_node("reminder_management", reminder_management_node)
    workflow.add_node("image_analysis", image_analysis_node)           # 新增图片分析节点
    workflow.add_node("image_answer", image_analysis_answer_node)      # 新增图片回答节点
    workflow.add_node("weather_query", weather_query_node)             # 新增天气查询节点
    workflow.add_node("finance_query", finance_query_node)             # 新增财务查询节点
    workflow.add_node("field_management", field_management_node)       # 新增地块管理节点

    # 设置入口节点
    workflow.set_entry_point("parse_input")

    # 定义执行流程
    workflow.add_edge("parse_input", "classify_intent")

    # 意图路由函数 - 所有意图都通过 LLM 回复
    def route_intent(state: AgentState) -> str:
        if state.intent_type == "image_analysis":
            return "image_analysis"
        elif state.intent_type == "weather_query":
            return "weather_query"
        elif state.intent_type == "finance_query":
            return "finance_query"
        elif state.intent_type == "field_management":
            return "field_management"
        elif state.need_clarification:
            return "clarify"
        elif state.intent_type in ["crop_selection", "planting_schedule"]:
            return "planting_plan"
        elif state.intent_type == "reminder_setup":
            return "reminder_management"
        else:
            # 其他所有意图（greeting/thanks/farewell/identity/function等）都走 RAG + LLM
            return "rag_retrieval"

    # 条件分支
    workflow.add_conditional_edges(
        source="classify_intent",
        path=route_intent,
        path_map={
            "rag_retrieval": "rag_retrieval",
            "clarify": "clarify",
            "planting_plan": "planting_plan",
            "reminder_management": "reminder_management",
            "image_analysis": "image_analysis",
            "weather_query": "weather_query",
            "finance_query": "finance_query",
            "field_management": "field_management"
        }
    )

    # 后续流程 - 统一使用 LLM 回复节点
    workflow.add_edge("rag_retrieval", "general_response")
    workflow.add_edge("general_response", "extract_tasks")  # 新增：提取建议并创建任务
    workflow.add_edge("clarify", "update_long_memory")
    workflow.add_edge("planting_plan", "extract_tasks")     # 种植计划也提取建议
    workflow.add_edge("reminder_management", "update_long_memory")
    workflow.add_edge("image_analysis", "image_answer")
    workflow.add_edge("image_answer", "extract_tasks")      # 图片分析也提取建议
    workflow.add_edge("weather_query", "update_long_memory")  # 天气查询直接更新记忆
    workflow.add_edge("finance_query", "update_long_memory")  # 财务查询直接更新记忆
    workflow.add_edge("field_management", "update_long_memory")  # 地块管理直接更新记忆
    workflow.add_edge("extract_tasks", "update_long_memory")  # 所有路径最终都到更新记忆
    workflow.add_edge("update_long_memory", END)

    return workflow.compile()

# =========================
# Streamlit Web界面（核心新增）
# =========================
def streamlit_chat_interface():
    """
    替代CLI的Web可视化界面
    - 友好的对话界面
    - 保存对话历史
    - 适配多轮追问
    """
    # 页面基础配置
    st.set_page_config(
        page_title="智能种植规划助手",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 标题区域
    st.title("🌾 智能种植规划助手")
    st.markdown("### 为您提供作物选择、种植时间规划、农事提醒等全周期种植服务")

    st.divider()

    # 用户基础信息表单 - 在首次使用时显示
    if "user_profile_submitted" not in st.session_state:
        st.session_state.user_profile_submitted = False

    if not st.session_state.user_profile_submitted:
        with st.container():
            st.markdown("## 👋 欢迎使用智能种植规划助手")
            st.info("请填写以下基础信息，以便我们为您提供更精准的种植建议。这些信息可以随时在侧边栏修改。")

            # 基础信息表单
            with st.form("user_profile_form"):
                col1, col2 = st.columns(2)
                with col1:
                    region = st.text_input("📍 所在地区", placeholder="如：华北、山东、四川等",
                                          help="请填写您所在的省、市或地区")
                    soil_type = st.selectbox("🌍 土壤类型",
                                            ["请选择", "壤土", "砂土", "粘土", "沙壤土", "黏壤土", "其他"],
                                            help="选择您农田的主要土壤类型")
                with col2:
                    farm_size = st.number_input("📐 种植面积（亩）", min_value=0.0, max_value=10000.0, value=0.0, step=0.5,
                                               help="请输入您的种植面积，单位为亩")
                    experience = st.selectbox("🎓 种植经验",
                                             ["请选择", "新手（1年以下）", "初级（1-3年）", "中级（3-5年）", "高级（5-10年）", "专家（10年以上）"],
                                             help="选择您的种植经验水平")

                goals = st.multiselect("🎯 种植目标",
                                      ["高产", "优质", "省工", "节水", "有机", "多样化种植", "经济效益", "自用为主"],
                                      help="选择您的种植目标，可多选")

                st.markdown("---")
                st.markdown("💡 **提示**：这些信息将帮助我们为您提供更个性化的种植建议。您可以随时在侧边栏修改这些信息。")

                submitted = st.form_submit_button("🚀 开始使用", use_container_width=True)

                if submitted:
                    # 保存用户信息到 session_state
                    st.session_state.user_region = region if region else ""
                    st.session_state.user_soil_type = soil_type if soil_type != "请选择" else ""
                    st.session_state.user_farm_size = farm_size if farm_size > 0 else 1.0
                    st.session_state.user_experience = experience if experience != "请选择" else ""
                    st.session_state.user_goals = goals if goals else []
                    st.session_state.user_profile_submitted = True
                    st.success("✅ 信息已保存！正在启动助手...")
                    st.rerun()

        st.stop()  # 未提交表单前不显示后续内容

    st.success("✅ 基础信息已设置，您可以随时在侧边栏修改。")

    # 1. 初始化农业知识检索系统（只加载一次）
    @st.cache_resource
    def load_rag_system():
        try:
            # 使用简化版RAG（无需Embeddings）
            rag = SimpleAgricultureRAG()
            return rag
        except Exception as e:
            st.error(f"加载农业知识库失败：{e}")
            st.stop()

    # 2. 初始化Agent（只加载一次）
    @st.cache_resource
    def load_agent():
        rag_system = load_rag_system()
        return build_agricultural_policy_agent(rag_system)

    # 3. 初始化会话状态（保存对话历史和Agent状态）
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = AgentState(
            messages=[],
            user_profile={
                "region": st.session_state.get("user_region", ""),
                "soil_type": st.session_state.get("user_soil_type", ""),
                "farm_size": st.session_state.get("user_farm_size", 1.0),
                "experience": st.session_state.get("user_experience", ""),
                "goals": st.session_state.get("user_goals", [])
            }
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 初始化语音输入状态
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "voice_text" not in st.session_state:
        st.session_state.voice_text = None

    # 更新 AgentState 中的用户档案（防止侧边栏修改后未同步）
    st.session_state.agent_state.user_profile = {
        "region": st.session_state.get("user_region", ""),
        "soil_type": st.session_state.get("user_soil_type", ""),
        "farm_size": st.session_state.get("user_farm_size", 1.0),
        "experience": st.session_state.get("user_experience", ""),
        "goals": st.session_state.get("user_goals", [])
    }

    # 加载资源
    policy_agent = load_agent()

    # 显示历史对话
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # 图片上传区域（新增）
    with st.expander("📷 上传农作物图片进行诊断（可选）"):
        uploaded_image = st.file_uploader(
            "选择图片（支持 jpg, jpeg, png 格式）",
            type=["jpg", "jpeg", "png"],
            key="crop_image_uploader"
        )
        if uploaded_image:
            # 显示预览
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(uploaded_image, caption="预览", use_container_width=True)
            with col2:
                st.info("图片已加载，可以在下方输入问题（如：'请分析这张图片'），或直接发送空消息开始分析。")
            # 保存到session_state
            import base64
            image_bytes = uploaded_image.getvalue()
            st.session_state.uploaded_image_base64 = base64.b64encode(image_bytes).decode()
            st.session_state.uploaded_image_mime = uploaded_image.type

    # 用户输入框（支持语音输入）
    # 使用列布局放置输入框和语音按钮
    input_col, voice_col = st.columns([0.92, 0.08])

    with input_col:
        user_input = st.chat_input("请输入您的问题（或点击右侧🎤语音输入）...")

    with voice_col:
        # 语音输入按钮
        if st.button("🎤", help="点击开始语音输入（请允许麦克风权限）", key="voice_btn"):
            st.session_state.recording = True
            st.rerun()

    # 处理语音输入
    if st.session_state.get("recording"):
        with st.spinner("🎙️ 正在聆听，请说话..."):
            from voice_components import voice_input_button
            voice_text = voice_input_button(key="voice_recorder")

            if voice_text:
                st.session_state.recording = False
                if voice_text.startswith("ERROR:"):
                    st.error(f"语音识别失败: {voice_text[6:]}")
                    st.session_state.voice_text = None
                elif voice_text:
                    st.session_state.voice_text = voice_text
                st.rerun()

    # 如果有语音输入的文字，使用它
    if st.session_state.get("voice_text"):
        user_input = st.session_state.voice_text
        st.session_state.voice_text = None

    if user_input or st.session_state.get("uploaded_image_base64"):
        # 显示用户消息
        display_content = user_input if user_input else "📷 请分析这张农作物图片"
        with st.chat_message("user"):
            st.markdown(display_content)
        st.session_state.chat_history.append({"role": "user", "content": display_content})

        # 调用Agent处理
        try:
            # 更新Agent状态
            if user_input:
                st.session_state.agent_state.messages.append(HumanMessage(content=user_input))
            else:
                st.session_state.agent_state.messages.append(HumanMessage(content="请分析这张农作物图片"))

            # 如果有图片，添加到AgentState
            if st.session_state.get("uploaded_image_base64"):
                st.session_state.agent_state.image_data = st.session_state.uploaded_image_base64
                st.session_state.agent_state.image_mime_type = st.session_state.uploaded_image_mime
                st.session_state.agent_state.has_image = True
                # 清除已使用的图片
                st.session_state.uploaded_image_base64 = None
                st.session_state.uploaded_image_mime = None

            # 同步用户档案到 agent_state（确保 LLM 能看到最新信息）
            st.session_state.agent_state.user_profile = {
                "region": st.session_state.get("user_region", ""),
                "soil_type": st.session_state.get("user_soil_type", ""),
                "farm_size": st.session_state.get("user_farm_size", 1.0),
                "experience": st.session_state.get("user_experience", ""),
                "goals": st.session_state.get("user_goals", [])
            }

            # 执行Agent工作流
            result = policy_agent.invoke(st.session_state.agent_state)

            # 转换回AgentState对象
            if isinstance(result, dict):
                st.session_state.agent_state = AgentState(**result)
            else:
                st.session_state.agent_state = result

            # 从 agent_state 同步提取的信息回 session_state（持久化记忆）
            if st.session_state.agent_state.short_term_facts.get("region"):
                st.session_state.user_region = st.session_state.agent_state.short_term_facts["region"]
            if st.session_state.agent_state.short_term_facts.get("crop"):
                st.session_state.user_crop = st.session_state.agent_state.short_term_facts["crop"]

            # 获取回答并显示
            answer = st.session_state.agent_state.final_answer
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"回答生成出错：{str(e)}")
            # 回滚状态
            st.session_state.agent_state.messages.pop()

    # 主内容区：添加地块大地图界面
    if st.session_state.get("show_add_field", False):
        st.markdown("## 🗺️ 添加新地块")
        st.info("💡 **操作步骤**：1. 点击地图右上角定位按钮 📍 获取当前位置  →  2. 使用左侧绘制工具（多边形 ⬡ 或矩形 ▭）绘制地块边界  →  3. 填写地块信息并保存")

        try:
            map_manager = MapManager()
            fields = map_manager.get_all_fields()

            # 获取用户当前地区作为地图中心
            default_lat, default_lon = 39.9, 116.4  # 默认北京
            if st.session_state.get("user_region"):
                try:
                    from core.map_manager import get_location_from_address
                    coords = get_location_from_address(st.session_state["user_region"])
                    if coords:
                        default_lat, default_lon = coords
                except:
                    pass

            # 创建地图
            try:
                import folium
                from streamlit_folium import st_folium

                # 准备已有地块数据
                existing_shapes = []
                for field in fields:
                    if field.coordinates:
                        existing_shapes.append({
                            "name": field.name,
                            "coordinates": field.coordinates
                        })

                # 使用全屏宽度创建地图
                m = create_folium_map(
                    center_lat=default_lat,
                    center_lon=default_lon,
                    zoom=14,
                    drawn_shapes=existing_shapes
                )

                # 显示地图并获取绘制数据 - 使用更宽的尺寸
                col1, col2 = st.columns([3, 1])

                with col1:
                    # 大地图展示区域
                    map_data = st_folium(m, width=900, height=600, key="field_draw_map_main")

                    # 提取绘制的多边形
                    drawn_coordinates = None
                    if map_data:
                        drawn_coordinates = extract_polygon_from_map_data(map_data)
                        if drawn_coordinates:
                            # 实时计算面积
                            area_m2, area_mu = map_manager.calculate_area(drawn_coordinates)
                            st.success(f"📐 已绘制地块，预估面积: **{area_mu:.2f}亩** ({area_m2:.0f}平方米)，顶点数: {len(drawn_coordinates)}")

                with col2:
                    # 保存表单放在右侧
                    st.markdown("### 📋 地块信息")
                    with st.form("save_field_form_main"):
                        field_name = st.text_input("地块名称 *", value=f"地块{len(fields)+1}", placeholder="如：东地块、小麦田等")
                        field_soil = st.selectbox("土壤类型", ["", "壤土", "砂土", "粘土", "沙壤土", "黏壤土", "其他"],
                                                  index=0 if not st.session_state.get("user_soil_type") else
                                                  ["", "壤土", "砂土", "粘土", "沙壤土", "黏壤土", "其他"].index(st.session_state.get("user_soil_type", "")))
                        field_crop = st.text_input("当前作物", placeholder="如：小麦、玉米等（可选）")

                        st.markdown("---")
                        submit_field = st.form_submit_button("💾 保存地块", use_container_width=True, type="primary")
                        cancel_field = st.form_submit_button("❌ 取消", use_container_width=True)

                        if submit_field:
                            if not drawn_coordinates:
                                st.error("⚠️ 请先在地圖上绘制地块边界！")
                            elif not field_name:
                                st.error("⚠️ 请输入地块名称！")
                            else:
                                try:
                                    new_field = map_manager.create_field(
                                        name=field_name,
                                        coordinates=drawn_coordinates,
                                        soil_type=field_soil,
                                        current_crop=field_crop
                                    )
                                    st.success(f"✅ 地块'{field_name}'保存成功！\n\n面积: {new_field.area_mu:.2f}亩")
                                    st.session_state.show_add_field = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"保存失败: {e}")

                        if cancel_field:
                            st.session_state.show_add_field = False
                            st.rerun()

            except ImportError as e:
                st.error(f"缺少必要的地图组件: {e}")
                st.info("请安装: pip install folium streamlit-folium")
                if st.button("返回"):
                    st.session_state.show_add_field = False
                    st.rerun()

        except Exception as e:
            st.error(f"加载地图失败: {e}")
            if st.button("返回"):
                st.session_state.show_add_field = False
                st.rerun()

        # 显示已有地块列表（底部）
        st.markdown("---")
        st.markdown("### 📍 已有地块")
        if fields:
            cols = st.columns(min(len(fields), 4))
            for i, field in enumerate(fields):
                with cols[i % 4]:
                    st.caption(f"**{field.name}**")
                    st.caption(f"{field.area_mu:.2f}亩")
                    if field.current_crop:
                        st.caption(f"🌾 {field.current_crop}")
        else:
            st.info("暂无地块")

        st.stop()  # 阻止显示对话界面

    # 侧边栏：功能设置、卡片展示
    with st.sidebar:
        st.header("⚙️ 功能设置")
        if st.button("🗑️ 清空对话历史", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")

        # 显示已记住的对话信息
        if st.session_state.agent_state.short_term_facts:
            with st.expander("📝 已记住的信息", expanded=False):
                facts = st.session_state.agent_state.short_term_facts
                if facts.get("region"):
                    st.markdown(f"📍 **地区**: {facts['region']}")
                if facts.get("crop"):
                    st.markdown(f"🌾 **作物**: {facts['crop']}")
                if facts.get("soil_type"):
                    st.markdown(f"🌍 **土壤**: {facts['soil_type']}")
                if facts.get("farm_size"):
                    st.markdown(f"📐 **面积**: {facts['farm_size']}亩")
                if not any([facts.get(k) for k in ["region", "crop", "soil_type", "farm_size"]]):
                    st.info("暂无对话中提取的信息")

        st.markdown("---")

        # 用户基础信息编辑
        with st.expander("👤 我的信息", expanded=False):
            with st.form("edit_profile_form"):
                region = st.text_input("📍 所在地区", value=st.session_state.get("user_region", ""))
                soil_type = st.selectbox("🌍 土壤类型",
                                        ["请选择", "壤土", "砂土", "粘土", "沙壤土", "黏壤土", "其他"],
                                        index=["请选择", "壤土", "砂土", "粘土", "沙壤土", "黏壤土", "其他"].index(
                                            st.session_state.get("user_soil_type", "请选择")) if st.session_state.get("user_soil_type") else 0)
                farm_size = st.number_input("📐 种植面积（亩）", min_value=0.0, max_value=10000.0,
                                           value=st.session_state.get("user_farm_size", 1.0), step=0.5)
                experience = st.selectbox("🎓 种植经验",
                                         ["请选择", "新手（1年以下）", "初级（1-3年）", "中级（3-5年）", "高级（5-10年）", "专家（10年以上）"],
                                         index=["请选择", "新手（1年以下）", "初级（1-3年）", "中级（3-5年）", "高级（5-10年）", "专家（10年以上）"].index(
                                             st.session_state.get("user_experience", "请选择")) if st.session_state.get("user_experience") else 0)
                goals = st.multiselect("🎯 种植目标",
                                      ["高产", "优质", "省工", "节水", "有机", "多样化种植", "经济效益", "自用为主"],
                                      default=st.session_state.get("user_goals", []))

                if st.form_submit_button("💾 保存修改"):
                    st.session_state.user_region = region
                    st.session_state.user_soil_type = soil_type if soil_type != "请选择" else ""
                    st.session_state.user_farm_size = farm_size
                    st.session_state.user_experience = experience if experience != "请选择" else ""
                    st.session_state.user_goals = goals
                    st.success("✅ 信息已更新！")
                    st.rerun()

        st.markdown("---")

        # 地块管理面板（简化版 - 仅列表和操作按钮）
        st.header("📍 我的地块")
        try:
            map_manager = MapManager()
            fields = map_manager.get_all_fields()

            # 显示已有地块列表
            if fields:
                for field in fields:
                    with st.container():
                        cols = st.columns([3, 1])
                        with cols[0]:
                            crop_icon = "🌾" if field.current_crop else "📍"
                            st.markdown(f"**{crop_icon} {field.name}** ({field.area_mu:.2f}亩)")
                            if field.current_crop:
                                st.caption(f"作物: {field.current_crop}")
                        with cols[1]:
                            if st.button("🗑️", key=f"del_field_{field.id}", help="删除此地块"):
                                map_manager.delete_field(field.id)
                                st.rerun()
                        st.markdown("---")

                # 显示总面积
                total_area = map_manager.get_total_area()
                st.caption(f"📊 总计: {len(fields)}个地块, 共{total_area:.2f}亩")
            else:
                st.info("暂无地块记录")

            # 添加新地块按钮 - 点击后在主区域显示大地图
            if st.button("➕ 添加新地块", key="add_field_btn"):
                st.session_state.show_add_field = True
                st.rerun()

        except Exception as e:
            st.error(f"加载地块失败: {e}")

        st.markdown("---")

        # 种植进度卡片展示
        st.header(" 种植进度")
        try:
            tracker = PlantingTracker()
            progress_cards = tracker.get_progress_cards(limit=3)

            if progress_cards:
                for card in progress_cards:
                    with st.container():
                        # 标题行 - 作物和阶段
                        title_cols = st.columns([3, 1])
                        with title_cols[0]:
                            status_color = {"进行中": "🟢", "已完成": "✅", "待开始": "⚪"}
                            status_icon = status_color.get(card['status'], "🟡")
                            st.markdown(f"**{status_icon} {card['crop']}** - {card['stage']}")
                        with title_cols[1]:
                            # 删除按钮
                            if st.button("🗑️", key=f"del_prog_{card['id']}", help="删除此进度"):
                                tracker.delete_progress(card['id'])
                                st.rerun()

                        # 进度条
                        progress_color = "normal"
                        if card['progress'] >= 80:
                            progress_color = "green"
                        elif card['progress'] >= 50:
                            progress_color = "orange"
                        st.progress(card['progress'] / 100, text=f"{card['progress']}%")

                        # 阶段信息和操作按钮
                        info_cols = st.columns([2, 2, 2])
                        with info_cols[0]:
                            st.caption(f"📊 阶段: {card['stage_number']}/{card['total_stages']}")
                        with info_cols[1]:
                            st.caption(f"📅 {card['status']}")
                        with info_cols[2]:
                            # 完成/更新进度按钮
                            if card['status'] != "已完成":
                                if st.button("✅ 完成阶段", key=f"complete_prog_{card['id']}"):
                                    result = tracker.advance_to_next_stage(card['id'])
                                    if result["success"]:
                                        if result["is_completed"]:
                                            st.success(result["message"])
                                        else:
                                            st.info(result["message"])
                                    st.rerun()

                        # 自动计算进度按钮（始终显示）
                        if st.button("🔄 自动计算进度", key=f"auto_calc_{card['id']}", use_container_width=True):
                            result = tracker.auto_calculate_progress(card['id'])
                            if result["success"]:
                                st.success(result["message"])
                            else:
                                st.warning(result["message"])
                            st.rerun()

                        st.markdown("---")
            else:
                st.info("暂无种植进度记录")

            # 添加新进度按钮
            if st.button("+ 添加种植进度", key="add_progress"):
                st.session_state.show_add_progress = True

            # 显示添加进度表单
            if st.session_state.get("show_add_progress", False):
                with st.container():
                    st.markdown("**添加新种植进度**")
                    new_crop = st.text_input("作物名称", key="new_crop_name")
                    new_stage = st.text_input("当前阶段", key="new_stage_name")
                    total_stages = st.number_input("总阶段数", min_value=1, max_value=20, value=5, key="new_total_stages")

                    cols = st.columns(2)
                    with cols[0]:
                        if st.button("保存进度", key="save_progress"):
                            if new_crop and new_stage:
                                try:
                                    tracker = PlantingTracker()
                                    tracker.create_progress({
                                        "crop": new_crop,
                                        "stage": new_stage,
                                        "stage_number": 1,
                                        "total_stages": total_stages,
                                        "start_date": datetime.now().strftime("%Y-%m-%d"),
                                        "expected_end_date": "",
                                        "progress_percent": 0,
                                        "status": "进行中",
                                        "tasks": [],
                                        "notes": ""
                                    })
                                    st.success(f"已添加 {new_crop} 的种植进度")
                                    st.session_state.show_add_progress = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"添加失败: {e}")
                            else:
                                st.warning("请填写作物名称和当前阶段")
                    with cols[1]:
                        if st.button("取消", key="cancel_progress"):
                            st.session_state.show_add_progress = False
                            st.rerun()

        except Exception as e:
            st.error(f"加载进度失败: {e}")

        st.markdown("---")

        # 农事任务卡片展示
        st.header(" 农事任务")
        try:
            tracker = PlantingTracker()
            task_cards = tracker.get_task_cards(limit=5)

            if task_cards:
                for card in task_cards:
                    with st.container():
                        # 任务标题和状态 - 标题行
                        title_cols = st.columns([3, 1])
                        with title_cols[0]:
                            status_emoji = {
                                "待办": "📝",
                                "进行中": "🌱",
                                "已完成": "✅",
                                "已逾期": "⚠️"
                            }.get(card['status'], "📋")
                            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                            priority_icon = priority_color.get(card['priority'], "⚪")
                            st.markdown(f"**{status_emoji} {card['title']}** {priority_icon}")
                        with title_cols[1]:
                            # 删除按钮
                            if st.button("🗑️", key=f"del_task_{card['id']}", help="删除此任务"):
                                tracker.delete_task(card['id'])
                                st.rerun()

                        # 作物和描述
                        st.caption(f"🌾 作物: {card['crop']} | {card['description'][:40]}...")

                        # 进度条
                        st.progress(card['progress'] / 100)

                        # 截止日期和剩余天数
                        if card['days_left'] is not None:
                            if card['days_left'] == 0:
                                st.caption("🔥 今天截止!")
                            elif card['days_left'] > 0:
                                st.caption(f"📅 截止: {card['end_date']} (还剩{card['days_left']}天)")
                            else:
                                st.caption(f"⏰ 已逾期 {abs(card['days_left'])} 天")

                        # 操作按钮
                        if card['status'] != "已完成":
                            if st.button("✅ 标记完成", key=f"complete_{card['id']}"):
                                tracker.update_task_status(card['id'], "已完成", 100)
                                st.rerun()

                        st.markdown("---")
            else:
                st.info("暂无农事任务")

            # 添加新任务按钮
            if st.button("+ 添加任务", key="add_task"):
                st.session_state.show_add_task = True

            # 显示添加任务表单
            if st.session_state.get("show_add_task", False):
                with st.container():
                    st.markdown("**添加新农事任务**")
                    task_crop = st.text_input("作物", key="task_crop_name")
                    task_title = st.text_input("任务标题", key="task_title_input")
                    task_type = st.selectbox("任务类型", ["浇水", "施肥", "除草", "病虫害防治", "修剪", "播种", "收获", "其他"], key="task_type_select")
                    task_priority = st.selectbox("优先级", ["high", "medium", "low"], format_func=lambda x: {"high": "高", "medium": "中", "low": "低"}[x], key="task_priority_select")

                    cols = st.columns(2)
                    with cols[0]:
                        if st.button("保存任务", key="save_task"):
                            if task_crop and task_title:
                                try:
                                    tracker = PlantingTracker()
                                    from datetime import timedelta
                                    end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                                    tracker.create_task({
                                        "crop": task_crop,
                                        "task_type": task_type,
                                        "title": task_title,
                                        "description": f"{task_type}任务",
                                        "status": "待办",
                                        "priority": task_priority,
                                        "end_date": end_date,
                                        "progress_percent": 0
                                    })
                                    st.success(f"已添加任务: {task_title}")
                                    st.session_state.show_add_task = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"添加失败: {e}")
                            else:
                                st.warning("请填写作物和任务标题")
                    with cols[1]:
                        if st.button("取消", key="cancel_task"):
                            st.session_state.show_add_task = False
                            st.rerun()

        except Exception as e:
            st.error(f"加载任务失败: {e}")

        st.markdown("---")

        # 天气查询面板（新增）
        st.header("🌤️ 天气服务")
        try:
            weather_service = WeatherService()
            location = st.session_state.get("user_region", "北京")

            if st.button("🔍 查询天气", key="query_weather"):
                with st.spinner("正在获取天气信息..."):
                    # 获取当前天气
                    current = weather_service.get_current_weather(location)
                    if current:
                        st.markdown(f"**📍 {current.location} 当前天气**")
                        st.markdown(f"🌡️ {current.temperature}℃ ({current.temperature_low}℃~{current.temperature_high}℃)")
                        st.markdown(f"☁️ {current.weather_desc}")
                        st.markdown(f"💧 湿度: {current.humidity}%")

                    # 获取预警
                    alerts = weather_service.check_weather_alerts(location)
                    if alerts:
                        st.warning("⚠️ 有气象预警，请注意防护！")

            # 显示未来3天预报
            forecast = weather_service.get_forecast(location, 3)
            if forecast:
                st.markdown("**📅 未来3天预报**")
                for w in forecast:
                    st.caption(f"{w.date}: {w.weather_desc} {w.temperature_low}~{w.temperature_high}℃")

        except Exception as e:
            st.info("天气服务暂未配置")

        st.markdown("---")

        # 财务管理面板（新增）
        st.header("💰 财务管理")
        try:
            finance_manager = FinanceManager()

            with st.expander("📝 快速记账", expanded=False):
                with st.form("quick_finance_form"):
                    record_type = st.selectbox("类型", ["成本支出", "销售收入"])
                    crop = st.text_input("作物", value=st.session_state.get("user_crop", ""))
                    amount = st.number_input("金额(元)", min_value=0.0, step=10.0)

                    if record_type == "成本支出":
                        cost_type = st.selectbox("成本类型", ["种子", "肥料", "农药", "人工", "农机", "其他"])
                        item_name = st.text_input("项目说明")
                    else:
                        quantity = st.number_input("产量(kg)", min_value=0.0, step=10.0)

                    submitted = st.form_submit_button("💾 保存记录")
                    if submitted:
                        try:
                            if record_type == "成本支出":
                                finance_manager.add_cost({
                                    "crop": crop,
                                    "cost_type": cost_type,
                                    "item_name": item_name,
                                    "quantity": 1,
                                    "unit": "项",
                                    "unit_price": amount
                                })
                                st.success(f"已记录{cost_type}成本 ¥{amount}")
                            else:
                                unit_price = amount / quantity if quantity > 0 else 0
                                finance_manager.add_income({
                                    "crop": crop,
                                    "quantity": quantity,
                                    "unit_price": unit_price
                                })
                                st.success(f"已记录销售收入 ¥{amount}")
                        except Exception as e:
                            st.error(f"保存失败: {e}")

            # 显示财务概览
            with st.expander("📊 财务概览", expanded=False):
                report = finance_manager.get_annual_report()
                if report['crop_reports']:
                    for crop_report in report['crop_reports'][:3]:
                        profit = crop_report['net_profit']
                        emoji = "🟢" if profit >= 0 else "🔴"
                        st.markdown(f"{emoji} **{crop_report['crop']}**: ¥{profit:.2f}")
                else:
                    st.info("暂无财务记录")

            # 导入导出功能
            with st.expander("📁 导入/导出", expanded=False):
                st.markdown("**导入CSV数据**")
                uploaded_file = st.file_uploader("选择CSV文件", type=['csv'], key="finance_csv")
                if uploaded_file:
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        data_type = st.selectbox("数据类型", ["成本", "收入"])
                        if st.button("导入", key="import_csv"):
                            result = finance_manager.import_from_csv(tmp_path, "cost" if data_type == "成本" else "income")
                            if result['success']:
                                st.success(f"成功导入 {result['imported']} 条记录")
                            else:
                                st.error(f"导入失败: {result.get('error', '')}")

                        import os
                        os.unlink(tmp_path)
                    except Exception as e:
                        st.error(f"处理文件失败: {e}")

                st.markdown("**导出数据**")
                if st.button("导出为CSV", key="export_csv"):
                    try:
                        import tempfile
                        export_path = tempfile.mktemp(suffix='.csv')
                        if finance_manager.export_to_csv(export_path):
                            with open(export_path, 'rb') as f:
                                st.download_button(
                                    label="下载CSV文件",
                                    data=f.read(),
                                    file_name=f"finance_export_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv"
                                )
                        os.unlink(export_path)
                    except Exception as e:
                        st.error(f"导出失败: {e}")

        except Exception as e:
            st.info("财务模块加载中...")

        st.markdown("---")

        # 图片分析提示（新增）
        st.markdown("### 📷 图片诊断功能")
        st.info("上传农作物图片，AI可自动识别：\n- 病虫害症状\n- 生长阶段\n- 营养问题")
        st.markdown("---")

        st.markdown("### 使用说明：")
        st.markdown('1. **作物选择** - 询问"华北地区适合种什么？"')
        st.markdown('2. **种植时间** - 询问"小麦什么时候播种？"')
        st.markdown('3. **农事提醒** - 说"为玉米设置浇水提醒"')
        st.markdown('4. **病虫害防治** - 描述症状或上传图片获取建议')
        st.markdown('5. **进度跟踪** - 询问"我的番茄现在该做什么？"')
        st.markdown('6. **天气查询** - 询问"明天适合喷药吗？"')
        st.markdown('7. **财务管理** - 询问"今年小麦赚了多少钱？"')
        st.markdown('8. **地块管理** - 在侧边栏"我的地块"中添加和管理农田地块')
        st.markdown('9. **图片诊断** - 点击上方"📷 上传农作物图片"进行分析')

# =========================
# 兼容CLI模式（保留原有功能）
# =========================
def interactive_chat(agent):
    """原有的CLI交互模式，备用"""
    print("="*60)
    print("      智能种植规划助手（输入 'exit' 退出）")
    print("="*60)

    current_state = AgentState(messages=[])

    while True:
        user_input = input("\n-> 请输入您的问题：").strip()

        if user_input.lower() in ["exit", "quit", "退出", "结束"]:
            print("\n 感谢使用，再见！")
            break

        if not user_input:
            print("[!]   请输入有效的问题！")
            continue

        current_state.messages.append(HumanMessage(content=user_input))

        try:
            result = agent.invoke(current_state)
            if isinstance(result, dict):
                current_state = AgentState(**result)
            else:
                current_state = result

            print("\n 回答：")
            print(current_state.final_answer)

        except Exception as e:
            print(f"\n[X]  回答生成出错：{e}")
            import traceback
            traceback.print_exc()
            current_state.messages.pop()

# =========================
# 主程序入口
# =========================
if __name__ == "__main__":
    # 默认启动Streamlit Web界面
    try:
        streamlit_chat_interface()
    # 如果环境不支持Streamlit（如无Web环境），自动降级到CLI模式
    except Exception as e:
        print(f"启动Web界面失败，切换到CLI模式：{e}")
        # 加载农业知识库
        print("正在加载农业知识库...")
        try:
            # 使用简化版RAG（无需Embeddings）
            rag_system = SimpleAgricultureRAG()
            print("[OK]  农业知识库加载成功！")
        except Exception as e:
            raise RuntimeError(f"加载农业知识库失败：{e}")

        # 构建Agent并启动CLI
        policy_agent = build_agricultural_policy_agent(rag_system)
        interactive_chat(policy_agent)