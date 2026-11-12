"""提醒管理节点 - 创建和管理农事提醒"""

import logging
import os
import re
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage

from ..state import AgentState
from ..utils import extract_facts_from_conversation
from .extract_tasks import calculate_end_date, task_device_fields

from core.reminder_system import ReminderSystem
from core.planting_tracker import PlantingTracker
from core.storage_paths import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)


def _reminder_crop(state: AgentState) -> str:
    """哥哥，优先保留本句明确的作物全名，别让动作或快捷指令残片混进来。"""
    def valid_name(value: str) -> bool:
        excluded = (
            "提醒", "任务", "今天", "明天", "后天", "上午", "下午", "晚上", "中午",
            "每天", "每周", "每月", "现在", "我", "水泵", "风扇", "风机", "喷头",
            "阀门", "传感器", "设备", "开关", "启动", "关闭", "开启", "打开", "停止",
            "浇水", "灌水", "灌溉", "施肥", "追肥", "除草", "修剪", "收获",
        )
        return bool(
            re.fullmatch(r"[一-鿿A-Za-z0-9·_-]{1,40}", value)
            and value not in {"浇", "灌", "施", "追", "除", "剪", "收", "采", "启", "关"}
            and not any(word in value for word in excluded)
            and not re.search(r"\d+(?:点|时|分)", value)
        )

    question = state.user_question or ""
    explicit = re.search(
        r"(?:给|为|替)\s*([一-鿿A-Za-z0-9·_-]{1,40}?)\s*"
        r"(?:浇水|灌水|灌溉|施肥|追肥|除草|病虫害防治|防治|修剪|整枝|收获|收割|采摘)",
        question,
    )
    if explicit and valid_name(explicit.group(1)):
        return explicit.group(1)
    if not explicit:
        known = extract_facts_from_conversation(state).get("crop", "")
        if known:
            return known
    for candidate in (state.short_term_facts.get("crop", ""), state.user_profile.get("crop", "")):
        candidate = (candidate or "").strip()
        if valid_name(candidate):
            return candidate
    return ""


def reminder_management_node(state: AgentState) -> AgentState:
    """提醒管理节点 - 创建和管理农事提醒"""
    if state.intent_type == "reminder_setup":
        user_question = state.user_question or ""

        # 从问题中提取关键信息
        crop = _reminder_crop(state)
        if crop:
            state.short_term_facts["crop"] = crop
        else:
            state.short_term_facts.pop("crop", None)
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
        time_match = re.search(r'(\d{1,2})[:点](半|\d{1,2})?', user_question)
        if time_match:
            hour = int(time_match.group(1))
            minute_raw = time_match.group(2) or "00"
            if minute_raw == "半":
                minute = "30"
            else:
                minute = minute_raw
                if len(minute) < 2:
                    minute = "0" + minute  # left-pad: "5" → "05"
            # 哥哥，下午三点要按十五点保存，半点也要完整保留。
            period = user_question[max(0, time_match.start() - 3):time_match.start()]
            if any(word in period for word in ("下午", "晚上", "傍晚", "中午")) and hour < 12:
                hour += 12
            elif "凌晨" in period and hour == 12:
                hour = 0
            time_of_day = f"{hour:02d}:{minute}"

        try:
            datetime.strptime(time_of_day, "%H:%M")
            start_date = calculate_end_date(user_question, default_days=0)
            if start_date == datetime.now().strftime("%Y-%m-%d") and not any(
                word in user_question for word in ("今天", "立即", "马上", "即刻")
            ):
                planned = datetime.strptime(f"{start_date} {time_of_day}", "%Y-%m-%d %H:%M")
                if planned < datetime.now():
                    start_date = (planned + timedelta(days=1)).strftime("%Y-%m-%d")

            # 检测是否启用短信通知
            channels = ["app"]
            want_sms = any(w in user_question for w in ["短信", "手机", "电话通知"])
            user_phone = state.user_profile.get("phone", "")
            if want_sms and user_phone:
                channels.append("sms")

            # 创建提醒
            username = state.username
            sd = os.path.join(DEFAULT_DATA_DIR, username)
            system = ReminderSystem(sd, username=username)
            reminder = system.create_reminder({
                "user_id": username,
                "crop": crop or "未指定作物",
                "reminder_type": reminder_type,
                "task_description": f"给{crop or '作物'}{reminder_type}",
                "frequency": frequency,
                "start_date": start_date,
                "time_of_day": time_of_day,
                "channels": channels
            })

            # 同时创建任务卡片（用于前端展示）
            try:
                tracker = PlantingTracker(sd)
                task_data = {
                    "crop": crop or "未指定作物",
                    "task_type": reminder_type,
                    "title": f"{reminder_type} - {crop or '作物'}",
                    "description": user_question,
                    "status": "待办",
                    "priority": "medium",
                    "end_date": reminder.next_trigger[:10],
                    "progress_percent": 0
                }
                from ..agents.device_agent import DeviceAgent
                # 哥哥，“先别执行”约束即时动作，仍要把预约中的设备和参数记清楚。
                task_question = re.sub(
                    r"(?:现在|当前)?(?:先|暂时)?(?:不要|别|不)(?:立刻|立即|马上|现在)?执行(?:设备)?",
                    "", user_question,
                )
                parsed = DeviceAgent()._keyword_parse(task_question, allow_deferred=True)
                if parsed:
                    parsed["device_hint"] = user_question
                    task_data.update(task_device_fields(parsed, username))
                task = tracker.create_task(task_data)
                state.chat_task_ids.append(str(task.id))
                system.storage.update_reminder(reminder.id, {"task_id": task.id})
                reminder.task_id = task.id
            except Exception:
                # 哥哥，卡片保存失败时撤回本次提醒，避免只完成了一半。
                system.delete_reminder(reminder.id)
                raise

            state.reminders.append({
                "id": reminder.id,
                "crop": reminder.crop,
                "type": reminder.reminder_type,
                "next_trigger": reminder.next_trigger,
                "task_id": task.id,
            })

            # 生成确认回答
            confirmation = f"[OK] 已为您设置农事提醒\n\n"
            confirmation += f"**作物**: {reminder.crop}\n"
            confirmation += f"**任务**: {reminder.reminder_type}\n"
            confirmation += f"**提醒时间**: {reminder.next_trigger}\n"
            confirmation += f"**频率**: {reminder.frequency}\n"
            if frequency != "单次":
                confirmation += "循环提醒会继续按周期通知；日历任务仅代表这一次安排。\n"
            confirmation += f"**已创建任务**: {task.title}\n"
            confirmation += f"**通知方式**: {', '.join(channels)}\n\n"
            if "sms" in channels:
                confirmation += f"📱 短信将发送到: {user_phone}\n\n"
            confirmation += "您可以在「农事日历」里查看和管理任务。"
            if state.interactive_cards:
                confirmation = (
                    f"已为 **{reminder.crop}** 创建 **{reminder.reminder_type}** 预约。\n\n"
                    f"- **提醒时间**：{reminder.next_trigger}\n"
                    f"- **频率**：{reminder.frequency}\n\n"
                    "请在下方任务卡片查看预约详情。"
                )

            state.final_answer = confirmation
            state.messages.append(AIMessage(content=confirmation))

        except Exception:
            logger.exception("对话任务创建失败")
            state.error_code = "TASK_CREATION_FAILED"
            state.final_answer = "任务和提醒未能保存，请稍后重试。"
            state.messages.append(AIMessage(content=state.final_answer))

    return state
