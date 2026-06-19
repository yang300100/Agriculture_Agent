"""提醒管理节点 - 创建和管理农事提醒"""

import logging
import re
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage

from ..state import AgentState

from core.reminder_system import ReminderSystem
from core.planting_tracker import PlantingTracker

logger = logging.getLogger(__name__)


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
        time_match = re.search(r'(\d{1,2})[:点](\d{0,2})', user_question)
        if time_match:
            hour = int(time_match.group(1))
            minute = time_match.group(2) or "00"
            if len(minute) < 2:
                minute += "0"
            time_of_day = f"{hour:02d}:{minute}"

        try:
            # 检测是否启用短信通知
            channels = ["app"]
            want_sms = any(w in user_question for w in ["短信", "手机", "电话通知"])
            user_phone = state.user_profile.get("phone", "")
            if want_sms and user_phone:
                channels.append("sms")

            # 创建提醒
            system = ReminderSystem()
            reminder = system.create_reminder({
                "crop": crop or "未指定作物",
                "reminder_type": reminder_type,
                "task_description": f"给{crop or '作物'}{reminder_type}",
                "frequency": frequency,
                "time_of_day": time_of_day,
                "channels": channels
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
                logger.warning(f"创建任务卡片失败: {e}")

            # 生成确认回答
            confirmation = f"[OK] 已为您设置农事提醒\n\n"
            confirmation += f"**作物**: {reminder.crop}\n"
            confirmation += f"**任务**: {reminder.reminder_type}\n"
            confirmation += f"**提醒时间**: {reminder.next_trigger}\n"
            confirmation += f"**频率**: {reminder.frequency}\n"
            confirmation += f"**通知方式**: {', '.join(channels)}\n\n"
            if "sms" in channels:
                confirmation += f"📱 短信将发送到: {user_phone}\n\n"
            confirmation += "您可以在侧边栏的提醒管理中查看和管理所有提醒。"

            state.final_answer = confirmation
            state.messages.append(AIMessage(content=confirmation))

        except Exception as e:
            state.final_answer = f"设置提醒时出现错误：{str(e)}。请稍后重试。"
            state.messages.append(AIMessage(content=state.final_answer))

    return state
