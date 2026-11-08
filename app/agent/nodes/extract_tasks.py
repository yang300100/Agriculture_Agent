"""任务提取节点 - 从LLM回答中提取建议并自动创建农事任务"""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from ..state import AgentState
from ..config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, OPENAI_BASE_URL, DEBUG_MODE

from core.planting_tracker import PlantingTracker
from core.storage_paths import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)


def task_device_fields(device_action: Dict[str, Any], username: str) -> Dict[str, Any]:
    """哥哥，只发现并关联设备，创建任务时不会执行任何设备指令。"""
    if not isinstance(device_action, dict):
        return {}
    from ..agents.device_agent import ACTION_TO_CAPABILITY, _discover_device_for_capability

    action = device_action.get("action", "")
    capability = device_action.get("capability") or ACTION_TO_CAPABILITY.get(action)
    if not capability:
        return {}
    try:
        hint = device_action.get("device_id") or device_action.get("device_hint", "")
        if hint:
            device_id = _discover_device_for_capability(capability, username, hint)
        else:
            device_id = _discover_device_for_capability(capability, username)
    except Exception as exc:
        logger.warning("任务设备发现失败，保留普通任务: %s", exc)
        return {}
    if not device_id:
        return {}
    command = device_action.get("command") or ("stop" if action in ("stop", "cool", "shade") else "start")
    return {
        "device_id": device_id,
        "device_command": command,
        "device_params": device_action.get("params") or {},
    }


def extract_and_create_tasks_node(state: AgentState) -> AgentState:
    """
    从LLM回答中提取建议并自动创建农事任务
    针对所有会产生可操作建议的意图类型（与 graph.py 路由保持一致）
    """
    state.progress_message = "正在提取待办任务..."
    if state.error_code or state.skip_task_extraction or state.intent_type == "device_control":
        return state
    # 哥哥，提醒节点已经直接建好了任务，这里再提取会生成重复卡片。
    if state.intent_type == "reminder_setup" and state.reminders:
        return state
    # 与 graph.py route_after_agent 的 task_intents 保持同步
    actionable_intents = [
        "crop_selection",       # 作物选择 → 创建种植计划任务
        "planting_schedule",    # 种植时间 → 创建播种提醒任务
        "planting_method",      # 种植方法 → 创建农事操作任务
        "disease_prevention",   # 病虫害防治 → 创建防治任务
        "harvest_planning",     # 收获规划 → 创建收获提醒任务
        "reminder_setup",       # 提醒设置 → 创建提醒任务
        "image_analysis",       # 图片分析 → 创建防治任务
        "device_control",       # 设备控制 → 创建设备执行任务
        "crop_monitoring",      # 作物监测 → 创建巡检任务
        "field_management",     # 地块管理 → 创建地块相关任务
    ]

    logger.info("extract_tasks 开始: intent=%s, answer_len=%d",
                state.intent_type, len(state.final_answer or ""))

    if state.intent_type not in actionable_intents:
        logger.info("extract_tasks 跳过: intent=%s 不在可操作范围内", state.intent_type)
        return state

    if not state.final_answer:
        logger.warning("extract_tasks 跳过: final_answer 为空")
        return state

    try:
        # 获取作物名称
        crop = state.short_term_facts.get("crop") or state.user_profile.get("crop", "")
        logger.info("extract_tasks 作物: %s", crop or "未识别")

        # 使用LLM提取建议（传入用户原始问题，帮助LLM理解上下文）
        suggestions = extract_suggestions_from_answer(
            state.final_answer, crop,
            user_question=state.user_question or "",
        )
        logger.info("extract_tasks LLM提取结果: %d 条建议", len(suggestions))

        if suggestions:
            # 创建任务（使用用户专属目录，与 API 读取路径一致）
            username = getattr(state, 'username', 'default')
            tracker = PlantingTracker(os.path.join(DEFAULT_DATA_DIR, username))
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
                    if device_action:
                        task_data.update(task_device_fields(device_action, username))

                    task = tracker.create_task(task_data)
                    created_tasks.append(task)
                    logger.info("extract_tasks 创建任务: %s (类型=%s, 优先级=%s)",
                                task.title, task.task_type, task.priority)
                except Exception as e:
                    logger.warning("extract_tasks 创建任务失败: %s", e)

            # suggestions 非空但所有任务创建都失败了 → 记录错误
            if not created_tasks:
                logger.error("extract_tasks: LLM 返回了 %d 条建议但全部创建失败", len(suggestions))

            # 如果有成功创建的任务，在回答中添加提示
            if created_tasks:
                task_notice = "\n\n---\n📋 **已为您自动生成农事任务**:\n"
                for i, task in enumerate(created_tasks[:3], 1):  # 最多显示3个
                    task_notice += f"{i}. {task.title}\n"
                if len(created_tasks) > 3:
                    task_notice += f"... 还有 {len(created_tasks) - 3} 个任务已添加到任务列表\n"
                task_notice += "\n💡 哥哥，可以在「农事日历」里查看和管理任务。"

                # 追加到回答中
                state.final_answer += task_notice
                # 更新最后一条消息
                if state.messages and isinstance(state.messages[-1], AIMessage):
                    state.messages[-1] = AIMessage(content=state.final_answer)

                logger.info("extract_tasks 完成: 共创建 %d 个任务", len(created_tasks))
            else:
                logger.info("extract_tasks 无任务创建: LLM 未从回答中提取到可执行建议")
        else:
            logger.info("extract_tasks 无建议: LLM 返回空列表或提取失败")

    except Exception as e:
        logger.warning("extract_tasks 整体异常: %s", e, exc_info=True)

    return state


def extract_suggestions_from_answer(answer: str, crop: str = "",
                                     user_question: str = "") -> List[Dict[str, Any]]:
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
                "end_date": "截止日期(YYYY-MM-DD格式)",
                "device_action": {...}  # 可选
            }
        ]
    """
    # 截断过长回答，避免超出 LLM token 限制
    answer_trimmed = answer[:3000] if len(answer) > 3000 else answer
    crop_hint = crop if crop else "从文本中识别"

    # ── 构建提取提示词（对用户输入转义 { } 后用 .format()，避免 KeyError）──
    _safe_question = (user_question or "(无)").replace("{", "{{").replace("}", "}}")
    _safe_answer = answer_trimmed.replace("{", "{{").replace("}", "}}")
    _safe_crop = crop_hint.replace("{", "{{").replace("}", "}}")

    extract_prompt = """请从以下农业建议文本中提取可执行的具体农事任务。

【用户原始问题】: {question}

【作物名称】: {crop}

【建议文本】:
{answer}

【提取要求】:
1. 只提取具体的、可操作的农事任务（如浇水、施肥、喷药、除草等）
2. 忽略一般性建议、解释说明、警告提示等非操作性内容
3. 结合用户原始问题，理解用户真正想要执行的操作，不要漏掉用户明确要求的任务
4. 每个任务需要明确：
   - 任务类型（浇水、施肥、病虫害防治、除草、修剪、收获等）
   - 任务标题（简短明确，如"喷施叶面肥"、"浇灌透水"）
   - 任务描述（具体操作步骤）
   - 优先级（high-紧急重要/medium-一般/low-可延后）
   - 建议完成时间（如"3天内"、"1周内"、"立即"等）
5. **重要**: 如果文本中提到可通过智能设备自动执行的操作（浇水、施肥、通风、补光、加热、降温、遮阳），必须提取 device_action 字段：
   - action 映射: 浇水→irrigate, 施肥→fertigate, 通风→ventilate, 补光→light, 加热→heat, 降温→cool, 遮阳→shade
   - 提取具体用量参数: 如"30分钟"→{{"duration":30}}, "5kg"→{{"amount_kg":5}}, "25度"→{{"target_temp":25}}
6. 如果文本中没有可执行的具体任务，返回空数组 []

【当前时间】: {now}

请严格只返回JSON数组，不要包裹在```json```代码块中，不要添加任何说明文字:
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

注意: device_action 字段仅在任务可通过设备自动执行时才需要，否则省略该字段。""".format(
        question=_safe_question,
        crop=_safe_crop,
        answer=_safe_answer,
        now=datetime.now().strftime("%Y-%m-%d"),
    )

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            request_timeout=60,  # 任务提取超时 60 秒，超时则跳过
        )

        logger.info("extract_tasks LLM 调用开始 (model=%s, answer_len=%d)", LLM_MODEL, len(answer_trimmed))
        try:
            response = llm.invoke([HumanMessage(content=extract_prompt)])
            content = response.content.strip()
            logger.info("extract_tasks LLM 返回: len=%d", len(content))
        except Exception as e:
            logger.warning("extract_tasks LLM 调用超时或失败，跳过任务提取: %s", e)
            return []

        # ── 智能提取 JSON 数组 ──
        suggestions = _parse_json_response(content)

        logger.info("extract_tasks JSON 解析成功: %d 条建议", len(suggestions))

        # 处理时间描述，转换为具体日期
        processed_suggestions = []
        for suggestion in suggestions:
            if not isinstance(suggestion, dict):
                continue
            timeframe = suggestion.get("end_date") or suggestion.get("timeframe", "")
            end_date = calculate_end_date(timeframe)
            suggestion["end_date"] = end_date
            processed_suggestions.append(suggestion)

        return processed_suggestions

    except Exception as e:
        logger.warning("extract_tasks LLM 提取失败: %s", e)
        return []


def _parse_json_response(content: str) -> list:
    """从 LLM 返回内容中鲁棒提取 JSON 数组

    处理场景：
    1. 纯 JSON 数组: [{"crop": ...}]
    2. markdown 代码块: ```json [...] ```
    3. 带说明文字: 以下为结果：[...]
    """
    # 策略1: 先剥离 markdown 代码块
    code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
    if code_block_match:
        return json.loads(code_block_match.group(1))

    # 策略2: 使用平衡括号匹配提取最外层 JSON 数组（鲁棒性更好）
    start = content.find('[')
    end = content.rfind(']')
    if start != -1 and end != -1 and end > start:
        json_str = content[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 策略3: 非贪婪匹配（fallback）
    json_match = re.search(r'\[.*?\]', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # 策略4: 尝试直接解析整个内容
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    logger.warning("extract_tasks 无法从LLM返回中提取JSON: len=%d", len(content))
    return []


def calculate_end_date(timeframe: str, default_days: int = 3) -> str:
    """
    根据时间描述计算具体的截止日期。
    支持：立即/马上/N天内/N周内/N月内 等格式
    """
    timeframe = timeframe.strip() if timeframe else ""
    now = datetime.now()

    # 哥哥，明确的日历日期优先保留，不能被默认天数覆盖。
    date_match = re.search(r'(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})日?', timeframe)
    if date_match:
        return datetime(*(int(part) for part in date_match.groups())).strftime("%Y-%m-%d")

    for word, days in [("大后天", 3), ("后天", 2), ("明天", 1)]:
        if word in timeframe:
            return (now + timedelta(days=days)).strftime("%Y-%m-%d")

    # 立即/马上/今天
    if any(word in timeframe for word in ["立即", "马上", "即刻", "今天"]):
        return now.strftime("%Y-%m-%d")

    # 显式天数提取: "7天", "14天", "3天内", "5个工作日" 等
    day_match = re.search(r'(\d+)\s*天', timeframe)
    if day_match:
        days = int(day_match.group(1))
        return (now + timedelta(days=days)).strftime("%Y-%m-%d")

    # 小时转换: "24小时", "48小时" 等
    hour_match = re.search(r'(\d+)\s*小时(?:内|后)', timeframe) or re.fullmatch(r'(\d+)\s*小时', timeframe)
    if hour_match:
        hours = int(hour_match.group(1))
        days = max(1, hours // 24)
        return (now + timedelta(days=days)).strftime("%Y-%m-%d")

    # 周提取: "1周", "2周内", "两周" 等
    week_cn = {"一": 1, "两": 2, "三": 3, "四": 4}
    for cn, val in week_cn.items():
        if f"{cn}周" in timeframe:
            return (now + timedelta(weeks=val)).strftime("%Y-%m-%d")
    week_match = re.search(r'(\d+)\s*周', timeframe)
    if week_match:
        weeks = int(week_match.group(1))
        return (now + timedelta(weeks=weeks)).strftime("%Y-%m-%d")

    # 半月
    if "半月" in timeframe:
        return (now + timedelta(days=15)).strftime("%Y-%m-%d")

    # 月提取: "1月", "一个月", "2个月内" 等
    month_match = re.search(r'(\d+)\s*个?\s*月', timeframe)
    if month_match:
        months = int(month_match.group(1))
        return (now + timedelta(days=months * 30)).strftime("%Y-%m-%d")

    # 本周/本月
    if "本周" in timeframe or "周内" in timeframe:
        return (now + timedelta(days=5)).strftime("%Y-%m-%d")
    if "本月" in timeframe:
        return (now + timedelta(days=20)).strftime("%Y-%m-%d")

    # 哥哥，日常口语中的中文天数也要按约定日期落库。
    for word, days in {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}.items():
        if f"{word}天" in timeframe:
            return (now + timedelta(days=days)).strftime("%Y-%m-%d")
    if any(word in timeframe for word in ["几天内", "近日"]):
        return (now + timedelta(days=3)).strftime("%Y-%m-%d")

    return (now + timedelta(days=default_days)).strftime("%Y-%m-%d")
