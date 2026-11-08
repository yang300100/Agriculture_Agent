"""
提醒管理系统模块
功能：
- 创建和管理种植提醒
- 支持多种提醒类型（浇水、施肥、除草、病虫害防治等）
- 多种频率设置（单次、每日、每周、自定义）
- 提醒历史记录和统计分析
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import dotenv

from core.storage_paths import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# 加载环境变量
dotenv.load_dotenv()
DEFAULT_STORAGE_DIR = DEFAULT_DATA_DIR


class ReminderType(Enum):
    """提醒类型枚举"""
    WATERING = "浇水"
    FERTILIZING = "施肥"
    WEEDING = "除草"
    PEST_CONTROL = "病虫害防治"
    PRUNING = "修剪"
    HARVEST = "收获"
    OTHER = "其他"


class ReminderFrequency(Enum):
    """提醒频率枚举"""
    ONCE = "单次"
    DAILY = "每天"
    WEEKLY = "每周"
    BIWEEKLY = "每两周"
    MONTHLY = "每月"
    CUSTOM = "自定义"


@dataclass
class Reminder:
    """提醒数据类"""
    id: str
    user_id: str
    crop: str
    reminder_type: str
    task_description: str
    growth_stage: str
    start_date: str
    frequency: str
    interval_days: int
    specific_days: List[int]  # 一周中的哪几天，1=周一
    time_of_day: str  # "09:00"
    advance_hours: int
    channels: List[str]
    status: str  # active, paused, completed
    created_at: str
    last_triggered: Optional[str]
    next_trigger: Optional[str]
    completed_count: int
    skipped_count: int
    task_id: Optional[str] = None
    progress_id: Optional[str] = None


def _decode_channels(value: Any) -> List[str]:
    """把数据库中的 JSON 文本安全还原为渠道列表。"""
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            return [value]
    return []


class ReminderStorage:
    """提醒数据存储"""

    def __init__(self, storage_dir: str = None, username: str = "default"):
        from core.database.engine import init_db
        init_db()
        from core.database.repository.users import UserRepository
        repo = UserRepository()
        self._user = repo.get_by_username(username)
        if not self._user:
            self._user = repo.create(username=username, password_hash="")

    @property
    def _uid(self) -> int:
        return self._user.id

    def load_reminders(self) -> List[Dict[str, Any]]:
        """从DB加载所有提醒"""
        from core.database.repository.reminders import ReminderRepository
        repo = ReminderRepository()
        reminders = repo.find_by(user_id=self._uid)
        return [{
            "id": str(r.id), "user_id": self._user.username,
            "crop": r.crop, "reminder_type": r.reminder_type,
            "task_description": r.task_description, "growth_stage": r.growth_stage,
            "frequency": r.frequency, "interval_days": r.interval_days,
            "time_of_day": r.time_of_day, "advance_hours": r.advance_hours or 0,
            "channels": _decode_channels(r.channels), "status": r.status,
            "status_label": {"active": "待通知", "paused": "已暂停", "completed": "已通知", "cancelled": "已取消"}.get(r.status, r.status),
            "start_date": r.start_date.isoformat() if r.start_date else (r.next_trigger.strftime("%Y-%m-%d") if r.next_trigger else ""),
            "task_id": str(r.task_id) if r.task_id is not None else None,
            "progress_id": str(r.progress_id) if r.progress_id is not None else None,
            "last_triggered": (
                r.last_triggered.strftime("%Y-%m-%d %H:%M")
                if r.last_triggered else ""
            ),
            "next_trigger": (
                r.next_trigger.strftime("%Y-%m-%d %H:%M")
                if r.next_trigger else ""
            ),
            "created_at": (
                r.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if r.created_at else ""
            ),
        } for r in reminders]

    @staticmethod
    def _reminder_values(reminder: Dict[str, Any]) -> Dict[str, Any]:
        """哥哥，只保存模型支持的字段，渠道统一按 JSON 写入。"""
        fields = (
            "crop", "reminder_type", "task_description", "growth_stage", "frequency",
            "interval_days", "time_of_day", "advance_hours", "channels", "status",
            "next_trigger", "last_triggered", "created_at",
            "start_date", "task_id", "progress_id",
        )
        values = {key: reminder[key] for key in fields if key in reminder}
        if "channels" in values and not isinstance(values["channels"], str):
            values["channels"] = json.dumps(values["channels"], ensure_ascii=False)
        return values

    def _owned_reminder(self, repo, reminder_id):
        """哥哥，整型和字符串 ID 都只查当前用户自己的提醒。"""
        try:
            numeric_id = int(reminder_id)
        except (ValueError, TypeError):
            return None
        return repo.find_one(id=numeric_id, user_id=self._uid)

    def save_reminders(self, reminders: List[Dict[str, Any]]):
        """哥哥，同步当前用户的提醒列表，保留既有 ID 和触发时间。"""
        from core.database.models import Reminder as ReminderModel
        from core.database.repository.reminders import ReminderRepository
        repo = ReminderRepository()
        with repo.session_scope() as session:
            existing = {str(row.id): row for row in repo.find_by(user_id=self._uid)}
            retained = set()
            for reminder in reminders:
                key = str(reminder.get("id", ""))
                values = repo._coerce_dates(self._reminder_values(reminder))
                row = existing.get(key)
                if row is None:
                    row = ReminderModel(user_id=self._uid, **values)
                    session.add(row)
                else:
                    values.pop("created_at", None)
                    for field, value in values.items():
                        setattr(row, field, value)
                    retained.add(key)
            for key, row in existing.items():
                if key not in retained:
                    session.delete(row)
            session.commit()

    def add_reminder(self, reminder: Dict[str, Any]) -> str:
        """哥哥，只插入这条提醒，并返回数据库的真实 ID。"""
        from core.database.repository.reminders import ReminderRepository
        row = ReminderRepository().create(user_id=self._uid, **self._reminder_values(reminder))
        return str(row.id)

    def update_reminder(self, reminder_id: str, updates: Dict[str, Any]):
        """更新提醒"""
        from core.database.repository.reminders import ReminderRepository
        repo = ReminderRepository()
        row = self._owned_reminder(repo, reminder_id)
        if row is not None:
            values = self._reminder_values(updates)
            values.pop("created_at", None)
            repo.update(row.id, **values)

    def delete_reminder(self, reminder_id: str):
        """删除提醒"""
        from core.database.repository.reminders import ReminderRepository
        repo = ReminderRepository()
        row = self._owned_reminder(repo, reminder_id)
        if row is not None:
            repo.delete(row.id)


class ReminderSystem:
    """提醒系统主类"""

    def __init__(self, storage_dir: str = None, username: str = "default"):
        storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        self.storage = ReminderStorage(storage_dir, username=username)

    def create_reminder(self, reminder_data: Dict[str, Any]) -> Reminder:
        """
        创建提醒

        Args:
            reminder_data: 包含以下字段的字典
                - user_id: 用户ID
                - crop: 作物名称
                - reminder_type: 提醒类型（浇水、施肥等）
                - task_description: 任务描述
                - growth_stage: 生长阶段（可选）
                - start_date: 开始日期（YYYY-MM-DD）
                - frequency: 频率（单次、每天、每周、自定义）
                - interval_days: 间隔天数（自定义频率时用）
                - specific_days: 具体星期几[1,3,5]表示周一三五
                - time_of_day: 提醒时间（HH:MM）
                - advance_hours: 提前提醒小时数
                - channels: 通知渠道列表 ["app"]

        Returns:
            Reminder: 创建的提醒对象
        """
        now = datetime.now()
        from core.database.repository.planting import PlantingPlanRepository, PlantingTaskRepository

        # 哥哥，只有同一用户明确指定的记录才能建立关联。
        for key, repository, label in (("task_id", PlantingTaskRepository, "任务"), ("progress_id", PlantingPlanRepository, "种植进度")):
            value = reminder_data.get(key)
            if value is not None:
                row = repository().get_by_id(int(value)) if str(value).isdigit() else None
                if row is None or row.user_id != self.storage._uid:
                    raise ValueError(f"关联{label}不存在或不属于当前用户")

        # 计算下次触发时间
        next_trigger = self._calculate_next_trigger(
            reminder_data.get("start_date", now.strftime("%Y-%m-%d")),
            reminder_data.get("time_of_day", "09:00"),
            reminder_data.get("frequency", "单次"),
            reminder_data.get("interval_days", 1),
            reminder_data.get("specific_days", [])
        )

        reminder = Reminder(
            id="",
            user_id=self.storage._user.username,
            crop=reminder_data.get("crop", ""),
            reminder_type=reminder_data.get("reminder_type", "其他"),
            task_description=reminder_data.get("task_description", ""),
            growth_stage=reminder_data.get("growth_stage", ""),
            start_date=reminder_data.get("start_date", now.strftime("%Y-%m-%d")),
            frequency=reminder_data.get("frequency", "单次"),
            interval_days=reminder_data.get("interval_days", 0),
            specific_days=reminder_data.get("specific_days", []),
            time_of_day=reminder_data.get("time_of_day", "09:00"),
            advance_hours=reminder_data.get("advance_hours", 0),
            channels=reminder_data.get("channels", ["app"]),
            status="active",
            created_at=now.strftime("%Y-%m-%d %H:%M:%S"),
            last_triggered=None,
            next_trigger=next_trigger,
            completed_count=0,
            skipped_count=0,
            task_id=str(reminder_data["task_id"]) if reminder_data.get("task_id") is not None else None,
            progress_id=str(reminder_data["progress_id"]) if reminder_data.get("progress_id") is not None else None,
        )

        # 保存到存储
        reminder.id = self.storage.add_reminder(asdict(reminder))

        return reminder

    def sync_task_reminders(self, task_id: str, status: str = None,
                            start_date: str = None, deleted: bool = False, session=None) -> None:
        """哥哥，只同步明确关联且尚未通知的单次提醒，循环系列保持独立。"""
        from core.database.engine import get_session
        from core.database.models import Reminder as ReminderModel

        if session is None:
            with get_session() as owned_session:
                self.sync_task_reminders(task_id, status, start_date, deleted, session=owned_session)
                owned_session.commit()
            return
        reminders = session.query(ReminderModel).filter(
            ReminderModel.user_id == self.storage._uid,
            ReminderModel.task_id == int(task_id),
            ReminderModel.frequency.in_(["单次", "once"]),
            ReminderModel.status.in_(["active", "paused"]),
            ReminderModel.last_triggered.is_(None),
        ).all()
        for reminder in reminders:
            if deleted or status in {"已完成", "completed"}:
                reminder.status = "cancelled"
                reminder.next_trigger = None
            elif start_date:
                reminder.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                reminder.next_trigger = datetime.strptime(
                    f"{start_date} {reminder.time_of_day or '09:00'}", "%Y-%m-%d %H:%M"
                )

    def _calculate_next_trigger(self, start_date: str, time_of_day: str,
                                frequency: str, interval_days: int,
                                specific_days: List[int]) -> str:
        """计算下次触发时间"""
        try:
            # 解析开始日期和时间
            base_datetime = datetime.strptime(
                f"{start_date} {time_of_day}",
                "%Y-%m-%d %H:%M"
            )
        except (ValueError, TypeError):
            base_datetime = datetime.now()

        now = datetime.now()

        # 如果开始时间已过，根据频率计算下次
        if base_datetime < now:
            if frequency == "单次":
                return base_datetime.strftime("%Y-%m-%d %H:%M")
            elif frequency == "每天":
                # 明天的同一时间
                next_date = now + timedelta(days=1)
                next_trigger = next_date.replace(
                    hour=base_datetime.hour,
                    minute=base_datetime.minute
                )
                return next_trigger.strftime("%Y-%m-%d %H:%M")
            elif frequency == "每周":
                if specific_days:
                    # 有指定具体星期几，找下一个匹配的日期（今天匹配则今天触发）
                    current_weekday = now.weekday() + 1  # 1=周一, 7=周日
                    future_days = [d for d in specific_days if d >= current_weekday]
                    if future_days:
                        days_ahead = future_days[0] - current_weekday
                    else:
                        # 本周已无匹配，取下周第一个
                        days_ahead = 7 - current_weekday + specific_days[0]
                    next_date = now + timedelta(days=days_ahead)
                else:
                    # 没有指定具体星期几，下周同一时间
                    next_date = now + timedelta(weeks=1)
                next_trigger = next_date.replace(
                    hour=base_datetime.hour,
                    minute=base_datetime.minute
                )
                return next_trigger.strftime("%Y-%m-%d %H:%M")
            elif frequency == "自定义":
                if interval_days <= 0:
                    # interval_days 必须大于 0，否则无法计算周期，回退到当天
                    logger.warning("自定义频率 interval_days=%d 无效，必须 > 0，回退到单次触发", interval_days)
                    return base_datetime.strftime("%Y-%m-%d %H:%M")
                # 计算从start_date到现在经过了多少个周期
                days_diff = (now - base_datetime).days
                periods_passed = days_diff // interval_days
                next_period = periods_passed + 1
                next_date = base_datetime + timedelta(days=next_period * interval_days)
                return next_date.strftime("%Y-%m-%d %H:%M")

        return base_datetime.strftime("%Y-%m-%d %H:%M")

    def get_active_reminders(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """获取用户的活跃提醒"""
        reminders = self.storage.load_reminders()
        return [
            r for r in reminders
            if r.get("user_id") == user_id and r.get("status") == "active"
        ]

    def get_upcoming_reminders(self, user_id: str = "default",
                               hours: int = 24) -> List[Dict[str, Any]]:
        """获取即将到期的提醒"""
        now = datetime.now()
        deadline = now + timedelta(hours=hours)

        active_reminders = self.get_active_reminders(user_id)
        upcoming = []

        for reminder in active_reminders:
            next_trigger_str = reminder.get("next_trigger")
            if next_trigger_str:
                try:
                    next_trigger = datetime.strptime(
                        next_trigger_str, "%Y-%m-%d %H:%M"
                    )
                    if now <= next_trigger <= deadline:
                        upcoming.append(reminder)
                except:
                    continue

        # 按触发时间排序
        upcoming.sort(key=lambda x: x.get("next_trigger", ""))
        return upcoming

    def complete_reminder(self, reminder_id: str):
        """标记提醒为已完成"""
        reminders = self.storage.load_reminders()
        for reminder in reminders:
            if str(reminder.get("id")) == str(reminder_id):
                reminder["completed_count"] = reminder.get("completed_count", 0) + 1
                reminder["last_triggered"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 如果是单次提醒，标记为完成
                if reminder.get("frequency") == "单次":
                    reminder["status"] = "completed"
                else:
                    # 计算下次触发时间（使用提醒自身的实际频率）
                    freq = reminder.get("frequency", "单次")
                    reminder["next_trigger"] = self._calculate_next_trigger(
                        reminder.get("start_date", ""),
                        reminder.get("time_of_day", "09:00"),
                        freq,
                        reminder.get("interval_days", 0),
                        reminder.get("specific_days", [])
                    )

                self.storage.update_reminder(reminder_id, reminder)
                break

    def skip_reminder(self, reminder_id: str):
        """跳过本次提醒"""
        reminders = self.storage.load_reminders()
        for reminder in reminders:
            if str(reminder.get("id")) == str(reminder_id):
                reminder["skipped_count"] = reminder.get("skipped_count", 0) + 1
                reminder["last_triggered"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 计算下次触发时间（使用提醒自身的实际频率）
                freq = reminder.get("frequency", "单次")
                reminder["next_trigger"] = self._calculate_next_trigger(
                    reminder.get("start_date", ""),
                    reminder.get("time_of_day", "09:00"),
                    freq,
                    reminder.get("interval_days", 0),
                    reminder.get("specific_days", [])
                )
                self.storage.update_reminder(reminder_id, reminder)
                break

    def delete_reminder(self, reminder_id: str):
        """删除提醒"""
        self.storage.delete_reminder(reminder_id)

    def pause_reminder(self, reminder_id: str):
        """暂停提醒"""
        self.storage.update_reminder(reminder_id, {"status": "paused"})

    def resume_reminder(self, reminder_id: str):
        """恢复提醒"""
        self.storage.update_reminder(reminder_id, {"status": "active"})

    def get_reminder_statistics(self, user_id: str = "default") -> Dict[str, Any]:
        """获取提醒统计信息"""
        reminders = self.storage.load_reminders()
        user_reminders = [r for r in reminders if r.get("user_id") == user_id]

        stats = {
            "total": len(user_reminders),
            "active": len([r for r in user_reminders if r.get("status") == "active"]),
            "completed": len([r for r in user_reminders if r.get("status") == "completed"]),
            "paused": len([r for r in user_reminders if r.get("status") == "paused"]),
            "total_completed_tasks": sum(r.get("completed_count", 0) for r in user_reminders),
            "total_skipped_tasks": sum(r.get("skipped_count", 0) for r in user_reminders),
            "by_crop": {},
            "by_type": {}
        }

        # 按作物统计
        for reminder in user_reminders:
            crop = reminder.get("crop", "未知")
            if crop not in stats["by_crop"]:
                stats["by_crop"][crop] = 0
            stats["by_crop"][crop] += 1

            # 按类型统计
            r_type = reminder.get("reminder_type", "其他")
            if r_type not in stats["by_type"]:
                stats["by_type"][r_type] = 0
            stats["by_type"][r_type] += 1

        return stats

    def format_reminder_list(self, reminders: List[Dict[str, Any]]) -> str:
        """格式化提醒列表为文本"""
        if not reminders:
            return "暂无提醒任务。"

        text = "📋 **您的农事提醒**\n\n"

        for i, reminder in enumerate(reminders, 1):
            crop = reminder.get("crop", "未知作物")
            r_type = reminder.get("reminder_type", "")
            desc = reminder.get("task_description", "")
            next_trigger = reminder.get("next_trigger", "未设置")
            frequency = reminder.get("frequency", "单次")

            text += f"{i}. **{crop} - {r_type}**\n"
            text += f"   📝 {desc}\n"
            text += f"   ⏰ 下次提醒: {next_trigger} ({frequency})\n"

            if reminder.get("growth_stage"):
                text += f"   🌱 生长阶段: {reminder['growth_stage']}\n"

            text += "\n"

        return text

    def format_upcoming_reminders(self, user_id: str = "default",
                                   hours: int = 24) -> str:
        """格式化即将到期的提醒"""
        upcoming = self.get_upcoming_reminders(user_id, hours)

        if not upcoming:
            return f"未来{hours}小时内没有待办农事。"

        text = f"⏰ **未来{hours}小时农事提醒**\n\n"

        for reminder in upcoming:
            crop = reminder.get("crop", "未知作物")
            r_type = reminder.get("reminder_type", "")
            desc = reminder.get("task_description", "")
            next_trigger = reminder.get("next_trigger", "")

            # 计算距离现在还有多久
            try:
                trigger_time = datetime.strptime(next_trigger, "%Y-%m-%d %H:%M")
                time_diff = trigger_time - datetime.now()
                hours_left = int(time_diff.total_seconds() / 3600)
                minutes_left = int((time_diff.total_seconds() % 3600) / 60)

                if hours_left > 0:
                    time_str = f"还有{hours_left}小时{minutes_left}分钟"
                else:
                    time_str = f"还有{minutes_left}分钟"
            except:
                time_str = "时间未知"

            text += f"🌾 **{crop}** - {r_type}\n"
            text += f"   📝 {desc}\n"
            text += f"   ⏰ {next_trigger} ({time_str})\n\n"

        return text

    def send_sms_notification(self, reminder_id: str, phone: str) -> Dict:
        """为指定提醒发送短信通知"""
        reminders = self.storage.load_reminders()
        reminder = next((r for r in reminders if str(r.get("id")) == str(reminder_id)), None)
        if not reminder:
            return {"success": False, "error": "提醒不存在"}

        try:
            from core.sms_service import SMSService
            sms = SMSService()
            result = sms.send_reminder(
                phone=phone,
                crop=reminder.get("crop", ""),
                task_type=reminder.get("reminder_type", ""),
                task_desc=reminder.get("task_description", ""),
                time_info=reminder.get("next_trigger", ""),
            )
            if result["success"]:
                self.storage.update_reminder(reminder_id, {
                    "last_sms_sent": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}


# 便捷函数
def create_watering_reminder(crop: str, frequency: str = "每天",
                             time_of_day: str = "08:00") -> str:
    """快速创建浇水提醒"""
    system = ReminderSystem()
    reminder = system.create_reminder({
        "crop": crop,
        "reminder_type": "浇水",
        "task_description": f"给{crop}浇水，注意浇透但不要积水",
        "frequency": frequency,
        "time_of_day": time_of_day
    })
    return f"已为{crop}设置{frequency}{time_of_day}的浇水提醒（ID: {reminder.id}）"


def create_fertilizing_reminder(crop: str, fertilizing_time: str,
                                time_of_day: str = "09:00") -> str:
    """快速创建施肥提醒"""
    system = ReminderSystem()
    reminder = system.create_reminder({
        "crop": crop,
        "reminder_type": "施肥",
        "task_description": f"给{crop}施肥: {fertilizing_time}",
        "frequency": "单次",
        "time_of_day": time_of_day
    })
    return f"已为{crop}设置{fertilizing_time}的施肥提醒（ID: {reminder.id}）"


if __name__ == "__main__":
    # 测试
    system = ReminderSystem()

    # 创建测试提醒
    reminder1 = system.create_reminder({
        "crop": "小麦",
        "reminder_type": "浇水",
        "task_description": "给小麦浇水，注意浇透",
        "frequency": "每周",
        "specific_days": [1, 4],  # 周一、周四
        "time_of_day": "08:00"
    })

    print(f"创建提醒: {reminder1}")

    # 获取即将到期的提醒
    upcoming = system.get_upcoming_reminders(hours=168)  # 一周
    print("\n即将到期的提醒:")
    for r in upcoming:
        print(f"  - {r['crop']} {r['reminder_type']} @ {r['next_trigger']}")

    # 统计信息
    stats = system.get_reminder_statistics()
    print(f"\n统计: 总共{stats['total']}个提醒，活跃{stats['active']}个")
