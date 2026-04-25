"""
提醒管理系统模块
功能：
- 创建和管理种植提醒
- 支持多种提醒类型（浇水、施肥、除草、病虫害防治等）
- 多种频率设置（单次、每日、每周、自定义）
- 提醒历史记录和统计分析
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import dotenv

# 加载环境变量
dotenv.load_dotenv()
DEFAULT_STORAGE_DIR = os.getenv("DATA_STORAGE_DIR", "data")


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


class ReminderStorage:
    """提醒数据存储"""

    def __init__(self, storage_dir: str = None):
        storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        self.storage_dir = storage_dir
        self.reminders_file = os.path.join(storage_dir, "reminders.json")
        self._ensure_storage()

    def _ensure_storage(self):
        """确保存储目录和文件存在"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        if not os.path.exists(self.reminders_file):
            with open(self.reminders_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def load_reminders(self) -> List[Dict[str, Any]]:
        """加载所有提醒"""
        try:
            with open(self.reminders_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载提醒失败: {e}")
            return []

    def save_reminders(self, reminders: List[Dict[str, Any]]):
        """保存所有提醒"""
        try:
            with open(self.reminders_file, 'w', encoding='utf-8') as f:
                json.dump(reminders, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存提醒失败: {e}")

    def add_reminder(self, reminder: Dict[str, Any]):
        """添加单个提醒"""
        reminders = self.load_reminders()
        reminders.append(reminder)
        self.save_reminders(reminders)

    def update_reminder(self, reminder_id: str, updates: Dict[str, Any]):
        """更新提醒"""
        reminders = self.load_reminders()
        for i, r in enumerate(reminders):
            if r.get("id") == reminder_id:
                reminders[i].update(updates)
                break
        self.save_reminders(reminders)

    def delete_reminder(self, reminder_id: str):
        """删除提醒"""
        reminders = self.load_reminders()
        reminders = [r for r in reminders if r.get("id") != reminder_id]
        self.save_reminders(reminders)


class ReminderSystem:
    """提醒系统主类"""

    def __init__(self, storage_dir: str = None):
        storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        self.storage = ReminderStorage(storage_dir)

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

        # 计算下次触发时间
        next_trigger = self._calculate_next_trigger(
            reminder_data.get("start_date", now.strftime("%Y-%m-%d")),
            reminder_data.get("time_of_day", "09:00"),
            reminder_data.get("frequency", "单次"),
            reminder_data.get("interval_days", 1),
            reminder_data.get("specific_days", [])
        )

        reminder = Reminder(
            id=str(uuid.uuid4())[:8],  # 简化的8位ID
            user_id=reminder_data.get("user_id", "default"),
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
            skipped_count=0
        )

        # 保存到存储
        self.storage.add_reminder(asdict(reminder))

        return reminder

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
        except:
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
                # 下周的同一时间
                next_date = now + timedelta(weeks=1)
                next_trigger = next_date.replace(
                    hour=base_datetime.hour,
                    minute=base_datetime.minute
                )
                return next_trigger.strftime("%Y-%m-%d %H:%M")
            elif frequency == "自定义" and interval_days > 0:
                # 计算从start_date到现在经过了多少个周期
                days_diff = (now - base_datetime).days
                periods_passed = days_diff // interval_days
                next_period = periods_passed + 1
                next_date = base_datetime + timedelta(days=next_period * interval_days)
                return next_date.strftime("%Y-%m-%d %H:%M")
            elif frequency == "每周" and specific_days:
                # 找到下一个指定的星期几
                current_weekday = now.weekday() + 1  # 1-7
                future_days = [d for d in specific_days if d > current_weekday]
                if future_days:
                    days_ahead = future_days[0] - current_weekday
                else:
                    days_ahead = 7 - current_weekday + specific_days[0]
                next_date = now + timedelta(days=days_ahead)
                next_trigger = next_date.replace(
                    hour=base_datetime.hour,
                    minute=base_datetime.minute
                )
                return next_trigger.strftime("%Y-%m-%d %H:%M")

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
            if reminder.get("id") == reminder_id:
                reminder["completed_count"] = reminder.get("completed_count", 0) + 1
                reminder["last_triggered"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 如果是单次提醒，标记为完成
                if reminder.get("frequency") == "单次":
                    reminder["status"] = "completed"
                else:
                    # 计算下次触发时间
                    reminder["next_trigger"] = self._calculate_next_trigger(
                        reminder.get("start_date", ""),
                        reminder.get("time_of_day", "09:00"),
                        reminder.get("frequency", "每天"),
                        reminder.get("interval_days", 1),
                        reminder.get("specific_days", [])
                    )

                break

        self.storage.save_reminders(reminders)

    def skip_reminder(self, reminder_id: str):
        """跳过本次提醒"""
        reminders = self.storage.load_reminders()
        for reminder in reminders:
            if reminder.get("id") == reminder_id:
                reminder["skipped_count"] = reminder.get("skipped_count", 0) + 1
                reminder["last_triggered"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 计算下次触发时间
                reminder["next_trigger"] = self._calculate_next_trigger(
                    reminder.get("start_date", ""),
                    reminder.get("time_of_day", "09:00"),
                    reminder.get("frequency", "每天"),
                    reminder.get("interval_days", 1),
                    reminder.get("specific_days", [])
                )
                break

        self.storage.save_reminders(reminders)

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
