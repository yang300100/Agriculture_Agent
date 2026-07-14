"""Agent 配置：环境变量 + 意图关键词 + 提示词模板"""

import os
import dotenv

dotenv.load_dotenv()

# LLM对话模型配置
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = LLM_API_KEY
OPENAI_BASE_URL = LLM_BASE_URL
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

if not LLM_API_KEY:
    raise EnvironmentError("未检测到 LLM_API_KEY 环境变量！")

# Vision视觉模型配置
VISION_API_KEY = os.getenv("VISION_API_KEY") or LLM_API_KEY
VISION_BASE_URL = os.getenv("VISION_BASE_URL") or LLM_BASE_URL
VISION_MODEL = os.getenv("VISION_MODEL") or LLM_MODEL
VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", "0.3"))

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

# Agent 自主权级别: low(全部确认) / medium(规则边界内自动) / high(完全自主跳过确认)

def get_autonomy_level() -> str:
    """获取当前自主权级别，优先从 session_state 读取，否则用环境变量"""
    try:
        import streamlit as st
        level = st.session_state.get("autonomy_level")
        if level and level in ("low", "medium", "high"):
            return level
    except Exception:
        pass
    return os.getenv("AUTONOMY_LEVEL", "medium").lower()

# 模块级缓存（非 Streamlit 环境使用）
AUTONOMY_LEVEL = os.getenv("AUTONOMY_LEVEL", "medium").lower()

# UI 行为配置
WEATHER_CACHE_TTL = int(os.getenv("WEATHER_CACHE_TTL", "1800"))  # 天气缓存秒数，默认30分钟
TASK_DEFAULT_DAYS = int(os.getenv("TASK_DEFAULT_DAYS", "7"))  # 新建任务默认截止天数
VOICE_LANG = os.getenv("VOICE_LANG", "zh-CN")  # 语音识别语言
MOBILE_BREAKPOINT = int(os.getenv("MOBILE_BREAKPOINT", "768"))  # 手机/桌面分界像素

# ---- 关键词常量 ----

GREETING_KEYWORDS = ["你好", "您好", "嗨", "哈喽", "早上好", "下午好", "晚上好"]
THANKS_KEYWORDS = ["谢谢", "感谢", "多谢", "辛苦了"]
FAREWELL_KEYWORDS = ["再见", "拜拜", "下次见", "回见"]
IDENTITY_KEYWORDS = ["你是谁", "你叫什么", "名字", "身份"]
FUNCTION_KEYWORDS = ["你能做什么", "功能", "能干什么", "帮助", "作用"]
GENERAL_KEYWORDS = GREETING_KEYWORDS + THANKS_KEYWORDS + FAREWELL_KEYWORDS + IDENTITY_KEYWORDS + FUNCTION_KEYWORDS

# 种植规划意图关键词
CROP_SELECTION_KEYWORDS = ["种什么", "适合种", "推荐作物", "种哪种", "作物选择", "种植品种", "适合种什么"]
PLANTING_SCHEDULE_KEYWORDS = ["什么时候种", "种植时间", "播种时间", "季节", "月份", "时机", "几月份种"]
PLANTING_METHOD_KEYWORDS = ["怎么种", "种植方法", "技术", "栽培", "管理", "步骤", "如何种植", "如何栽培"]
REMINDER_KEYWORDS = ["提醒", "通知", "浇水", "施肥", "除草", "打药", "设置提醒", "添加提醒"]
PROGRESS_KEYWORDS = ["进度", "记录", "生长", "阶段", "里程碑", "跟踪", "现在该做什么", "进展情况"]
DISEASE_KEYWORDS = ["病虫害", "病害", "虫害", "防治", "治疗", "预防", "生病了", "叶子发黄"]
HARVEST_KEYWORDS = ["收获", "收割", "采摘", "成熟", "产量", "收获期", "什么时候收"]

# 天气相关意图关键词
WEATHER_KEYWORDS = ["天气", "气温", "下雨", "预报", "气象", "霜冻", "台风", "干旱", "降水"]

# 财务相关意图关键词
FINANCE_KEYWORDS = ["成本", "收入", "花费", "赚钱", "盈亏", "财务", "记账", "支出", "收益", "利润", "报表", "价格", "行情", "卖多少钱", "市场价", "值多少钱"]

# 政策补贴意图关键词
POLICY_KEYWORDS = ["补贴", "政策", "补助", "扶持", "惠农", "地力保护", "耕地补贴",
                   "农业保险", "最低收购价", "收购政策", "补贴标准", "补贴对象",
                   "补贴范围", "优惠政策", "农业政策", "农机补贴"]

# 地块管理意图关键词
FIELD_KEYWORDS = ["地块", "农田", "位置", "面积", "边界", "定位", "地图", "田地", "土地", "测量"]

# 设备控制意图关键词
DEVICE_KEYWORDS = [
    "浇水", "灌溉", "施肥", "通风", "开窗", "遮阳",
    "补光", "开灯", "关灯", "加热", "降温", "喷雾",
    "自动控制", "手动控制", "设备状态", "打开", "关闭",
    "启动", "停止", "控制", "设备",
]

# 作物监测（摄像头拍照分析）
CROP_MONITOR_KEYWORDS = [
    "监控作物", "拍照分析", "查看长势", "生长情况", "监测",
    "自动巡检", "定时拍照", "摄像", "摄像头",
]

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

# ── 自主决策配置 ──────────────────────────
AUTO_DECISION_INTERVAL = int(os.getenv("AUTO_DECISION_INTERVAL", "30"))          # 巡检间隔（分钟）
AUTO_DECISION_MODEL = os.getenv("AUTO_DECISION_MODEL") or LLM_MODEL              # 决策LLM模型
AUTO_DECISION_REGIONS = os.getenv("AUTO_DECISION_REGIONS", "")                   # 限定区域，逗号分隔
AUTO_DECISION_NIGHT_MODE = os.getenv("AUTO_DECISION_NIGHT_MODE", "silent")      # silent|full
AUTO_DECISION_TIMEOUT = int(os.getenv("AUTO_DECISION_TIMEOUT", "120"))            # LLM决策超时秒数
AUTO_DECISION_MIN_INTERVAL = int(os.getenv("AUTO_DECISION_MIN_INTERVAL", "10"))  # 同区域最小间隔分钟
AUTO_DECISION_MAX_ACTIONS = int(os.getenv("AUTO_DECISION_MAX_ACTIONS", "5"))     # 单次最大操作数
AUTO_DECISION_TEMPERATURE = float(os.getenv("AUTO_DECISION_TEMPERATURE", "0.2")) # 决策LLM温度
