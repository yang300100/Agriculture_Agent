"""后台提醒调度器 — 定时检查到期提醒并触发通知

针对 Streamlit 请求-响应模型设计：
- 每次侧边栏渲染时机会性检查
- session_state 节流（默认间隔 5 分钟）
- 到期提醒展示在侧边栏 + 尝试 SMS 推送
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from core.reminder_system import ReminderSystem, ReminderStorage

logger = logging.getLogger(__name__)

DEFAULT_CHECK_INTERVAL_MINUTES = 5
DEFAULT_LOOKAHEAD_MINUTES = 30  # 提前多久的提醒视为"即将到期"


class ReminderScheduler:
    """提醒调度器：检查到期/即将到期的提醒并触发通知"""

    def __init__(self, storage_dir: str = None, username: str = "default"):
        self.system = ReminderSystem(storage_dir, username=username)
        self.storage = self.system.storage

    def get_due_reminders(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """获取当前已到期且未处理的提醒（next_trigger <= now）"""
        now = datetime.now()
        active = self.system.get_active_reminders(user_id)
        due = []
        for r in active:
            trigger_str = r.get("next_trigger", "")
            if not trigger_str:
                continue
            try:
                trigger_time = datetime.strptime(trigger_str, "%Y-%m-%d %H:%M")
                if trigger_time <= now:
                    due.append(r)
            except ValueError:
                continue
        due.sort(key=lambda x: x.get("next_trigger", ""))
        return due

    def check_and_fire(self, user_id: str = "default",
                       phone: str = "") -> List[Dict[str, Any]]:
        """
        检查并触发到期提醒

        - 将到期提醒的 next_trigger 更新为下一周期
        - 如果配置了手机号，尝试发送 SMS

        Returns:
            本次触发的提醒列表
        """
        due = self.get_due_reminders(user_id)
        if not due:
            return []

        fired = []
        for r in due:
            reminder_id = r.get("id", "")
            logger.info("触发提醒: %s - %s - %s", r.get("crop"), r.get("reminder_type"), reminder_id)

            # 更新 last_triggered 和 next_trigger
            now = datetime.now()
            r["last_triggered"] = now.strftime("%Y-%m-%d %H:%M:%S")
            r["completed_count"] = r.get("completed_count", 0) + 1

            if r.get("frequency") == "单次":
                r["status"] = "completed"
            else:
                r["next_trigger"] = self.system._calculate_next_trigger(
                    r.get("start_date", ""),
                    r.get("time_of_day", "09:00"),
                    r.get("frequency", "每天"),
                    r.get("interval_days", 1),
                    r.get("specific_days", []),
                )

            self.storage.update_reminder(reminder_id, r)

            # 尝试短信通知
            sms_result = None
            if phone:
                try:
                    sms_result = self.system.send_sms_notification(reminder_id, phone)
                except Exception as e:
                    logger.warning("SMS 发送失败: %s", e)
                    sms_result = {"success": False, "error": str(e)}

            fired.append({
                **r,
                "sms_result": sms_result,
            })

        return fired

    def get_upcoming(self, user_id: str = "default",
                     lookahead_minutes: int = None) -> List[Dict[str, Any]]:
        """获取未来 N 分钟内即将到期的提醒"""
        minutes = lookahead_minutes or DEFAULT_LOOKAHEAD_MINUTES
        now = datetime.now()
        deadline = now + timedelta(minutes=minutes)
        active = self.system.get_active_reminders(user_id)
        upcoming = []
        for r in active:
            trigger_str = r.get("next_trigger", "")
            if not trigger_str:
                continue
            try:
                trigger_time = datetime.strptime(trigger_str, "%Y-%m-%d %H:%M")
                if now <= trigger_time <= deadline:
                    upcoming.append(r)
            except ValueError:
                continue
        upcoming.sort(key=lambda x: x.get("next_trigger", ""))
        return upcoming

    def format_due_banner(self, due_reminders: List[Dict[str, Any]]) -> str:
        """格式化到期提醒为通知文本"""
        if not due_reminders:
            return ""
        lines = [f"⏰ **您有 {len(due_reminders)} 条农事提醒已到期**\n"]
        for r in due_reminders[:5]:
            crop = r.get("crop", "")
            rtype = r.get("reminder_type", "")
            desc = r.get("task_description", "")
            lines.append(f"- 🌾 {crop} · {rtype}: {desc}")
        return "\n".join(lines)

    def format_upcoming_banner(self, upcoming: List[Dict[str, Any]]) -> str:
        """格式化即将到期提醒为通知文本"""
        if not upcoming:
            return ""
        lines = [f"📋 **未来 {DEFAULT_LOOKAHEAD_MINUTES} 分钟内有 {len(upcoming)} 条农事提醒**\n"]
        for r in upcoming[:5]:
            crop = r.get("crop", "")
            rtype = r.get("reminder_type", "")
            trigger = r.get("next_trigger", "")
            lines.append(f"- 🌾 {crop} · {rtype} @ {trigger}")
        return "\n".join(lines)


def _get_check_key() -> str:
    return "_reminder_last_check"


def should_check_now(check_interval_minutes: int = None,
                     _st_module=None) -> bool:
    """判断是否应该执行检查（基于 session_state 节流）"""
    interval = check_interval_minutes or DEFAULT_CHECK_INTERVAL_MINUTES
    try:
        import streamlit as st
    except ImportError:
        return True

    last_check = st.session_state.get(_get_check_key())
    if last_check is None:
        return True
    try:
        last_dt = datetime.strptime(last_check, "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - last_dt) >= timedelta(minutes=interval)
    except (ValueError, TypeError):
        return True


def mark_checked(_st_module=None):
    """记录本次检查时间"""
    try:
        import streamlit as st
    except ImportError:
        return
    st.session_state[_get_check_key()] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def render_reminder_notifications(user_id: str = "default",
                                  phone: str = "",
                                  check_interval_minutes: int = None):
    """
    在 Streamlit 侧边栏渲染提醒通知

    这是供 sidebar.py 调用的主入口。
    节流检查 → 触发到期提醒 → 展示通知横幅 → 展示即将到期列表。
    """
    try:
        import streamlit as st
    except ImportError:
        return

    if not should_check_now(check_interval_minutes):
        return

    mark_checked()

    try:
        scheduler = ReminderScheduler()
        fired = scheduler.check_and_fire(user_id, phone)
        upcoming = scheduler.get_upcoming(user_id)

        if fired:
            st.toast("⏰ 农事提醒已到期！", icon="🌾")
            for r in fired:
                crop = r.get("crop", "")
                rtype = r.get("reminder_type", "")
                desc = r.get("task_description", "")
                st.warning(f"🌾 **{crop}** · {rtype}\n\n{desc}")
                sms = r.get("sms_result")
                if sms:
                    if sms.get("success"):
                        st.caption("📱 短信已发送")
                    else:
                        st.caption(f"📱 短信发送失败: {sms.get('error', '')}")

        if upcoming:
            with st.expander(f"📋 即将到期 ({len(upcoming)})", expanded=False):
                for r in upcoming:
                    trigger = r.get("next_trigger", "")
                    crop = r.get("crop", "")
                    rtype = r.get("reminder_type", "")
                    desc = r.get("task_description", "")
                    st.markdown(
                        f"- 🌾 **{crop}** · {rtype} · ⏰ {trigger}\n  {desc}"
                    )

    except Exception as e:
        logger.warning("提醒调度器运行出错: %s", e)
