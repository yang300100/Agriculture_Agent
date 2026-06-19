"""任务提取节点 - 从LLM回答中提取建议并自动创建农事任务"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from ..state import AgentState
from ..config import LLM_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, DEBUG_MODE

from core.planting_tracker import PlantingTracker

logger = logging.getLogger(__name__)


def extract_and_create_tasks_node(state: AgentState) -> AgentState:
    """
    从LLM回答中提取建议并自动创建农事任务
    只针对病虫害防治、种植方法、收获规划等会产生可操作建议的意图
    """
    # 只处理会产生建议的意图类型
    actionable_intents = [
        "disease_prevention",   # 病虫害防治
        "planting_method",      # 种植方法
        "harvest_planning",     # 收获规划
        "reminder_setup",       # 提醒设置
        "image_analysis",       # 图片分析
        "device_control",       # 设备控制（提取设备操作参数到任务）
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
                    task_data = {
                        "crop": suggestion.get("crop", crop or "未指定作物"),
                        "task_type": suggestion.get("task_type", "其他"),
                        "title": suggestion.get("title", "农事任务"),
                        "description": suggestion.get("description", ""),
                        "status": "待办",
                        "priority": suggestion.get("priority", "medium"),
                        "end_date": suggestion.get("end_date", ""),
                        "progress_percent": 0,
                    }

                    # 处理设备操作字段：将 LLM 提取的 device_action 解析为设备参数
                    device_action = suggestion.get("device_action")
                    if device_action and isinstance(device_action, dict):
                        action_type = device_action.get("action", "")
                        params = device_action.get("params", {})
                        from ..agents.device_agent import ACTION_TO_DEFAULT_DEVICE
                        device_id = ACTION_TO_DEFAULT_DEVICE.get(action_type)
                        if device_id:
                            task_data["device_id"] = device_id
                            task_data["device_command"] = "start"
                            task_data["device_params"] = params

                    task = tracker.create_task(task_data)
                    created_tasks.append(task)
                except Exception as e:
                    if DEBUG_MODE:
                        logger.debug(f"创建任务失败: {e}")

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
            logger.debug(f"提取建议并创建任务时出错: {e}")

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
4. **重要**: 如果文本中提到可通过智能设备自动执行的操作（浇水、施肥、通风、补光、加热、降温、遮阳），必须提取 device_action 字段：
   - action 映射: 浇水→irrigate, 施肥→fertigate, 通风→ventilate, 补光→light, 加热→heat, 降温→cool, 遮阳→shade
   - 提取具体用量参数: 如"30分钟"→{{"duration":30}}, "5kg"→{{"amount_kg":5}}, "25度"→{{"target_temp":25}}
5. 如果文本中没有可执行的具体任务，返回空数组 []

【当前时间】: {datetime.now().strftime("%Y-%m-%d")}

请以下面JSON格式返回，只返回JSON，不要其他说明:
[
  {{
    "crop": "作物名称",
    "task_type": "任务类型",
    "title": "任务标题",
    "description": "任务描述",
    "priority": "high/medium/low",
    "timeframe": "时间描述",
    "device_action": {{
      "action": "irrigate|fertigate|ventilate|light|heat|cool|shade",
      "params": {{"duration": 数字分钟, "amount_kg": 数字kg, "target_temp": 数字°C}}
    }}
  }}
]

注意: device_action 字段仅在任务可通过设备自动执行时才需要，否则省略该字段。"""

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.2,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )

        response = llm.invoke([HumanMessage(content=extract_prompt)])
        content = response.content.strip()

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
            logger.debug(f"提取建议失败: {e}")
        return []


def calculate_end_date(timeframe: str) -> str:
    """
    根据时间描述计算具体的截止日期
    """
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
