"""
种植进度跟踪与卡片展示模块
功能：
- 记录种植进度和农事提醒
- 生成卡片数据供前端展示
- 支持进度更新和历史记录
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import dotenv

# 加载环境变量
dotenv.load_dotenv()
DEFAULT_STORAGE_DIR = os.getenv("DATA_STORAGE_DIR", "data")


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "待办"
    IN_PROGRESS = "进行中"
    COMPLETED = "已完成"
    OVERDUE = "已逾期"


class TaskType(Enum):
    """任务类型"""
    WATERING = "浇水"
    FERTILIZING = "施肥"
    WEEDING = "除草"
    PEST_CONTROL = "病虫害防治"
    PRUNING = "修剪"
    SOWING = "播种"
    HARVEST = "收获"
    OTHER = "其他"


@dataclass
class PlantingTask:
    """种植任务卡片数据"""
    id: str
    crop: str
    task_type: str
    title: str
    description: str
    status: str
    priority: str  # high, medium, low
    start_date: str
    end_date: str
    completed_date: Optional[str]
    progress_percent: int  # 0-100
    notes: str
    created_at: str
    updated_at: str


@dataclass
class PlantingProgress:
    """种植进度卡片数据"""
    id: str
    crop: str
    stage: str  # 当前生长阶段
    stage_number: int  # 第几个阶段
    total_stages: int
    start_date: str
    expected_end_date: str
    actual_end_date: Optional[str]
    progress_percent: int
    status: str  # 进行中, 已完成, 待开始
    tasks: List[Dict[str, Any]]  # 该阶段的任务列表
    notes: str
    created_at: str
    updated_at: str


class PlantingTracker:
    """种植跟踪管理器"""

    def __init__(self, storage_dir: str = None):
        storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        self.storage_dir = storage_dir
        self.tasks_file = os.path.join(storage_dir, "planting_tasks.json")
        self.progress_file = os.path.join(storage_dir, "planting_progress.json")
        self._ensure_storage()

    def _ensure_storage(self):
        """确保存储目录和文件存在"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        for file_path in [self.tasks_file, self.progress_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([], f)

    def _generate_id(self) -> str:
        """生成唯一ID"""
        return f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"

    # ========== 任务管理 ==========

    def create_task(self, task_data: Dict[str, Any]) -> PlantingTask:
        """创建新任务"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        task = PlantingTask(
            id=self._generate_id(),
            crop=task_data.get("crop", ""),
            task_type=task_data.get("task_type", "其他"),
            title=task_data.get("title", ""),
            description=task_data.get("description", ""),
            status=task_data.get("status", "待办"),
            priority=task_data.get("priority", "medium"),
            start_date=task_data.get("start_date", datetime.now().strftime("%Y-%m-%d")),
            end_date=task_data.get("end_date", ""),
            completed_date=None,
            progress_percent=task_data.get("progress_percent", 0),
            notes=task_data.get("notes", ""),
            created_at=now,
            updated_at=now
        )

        self._save_task(task)
        return task

    def _save_task(self, task: PlantingTask):
        """保存任务到文件"""
        tasks = self._load_tasks()
        tasks.append(asdict(task))
        self._save_tasks(tasks)

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """加载所有任务"""
        try:
            with open(self.tasks_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载任务失败: {e}")
            return []

    def _save_tasks(self, tasks: List[Dict[str, Any]]):
        """保存任务列表"""
        with open(self.tasks_file, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

    def get_tasks(self, crop: Optional[str] = None, status: Optional[str] = None) -> List[PlantingTask]:
        """获取任务列表，支持筛选"""
        tasks_data = self._load_tasks()
        tasks = [PlantingTask(**data) for data in tasks_data]

        if crop:
            tasks = [t for t in tasks if t.crop == crop]
        if status:
            tasks = [t for t in tasks if t.status == status]

        # 按优先级和时间排序
        priority_order = {"high": 0, "medium": 1, "low": 2}
        tasks.sort(key=lambda x: (priority_order.get(x.priority, 1), x.end_date))

        return tasks

    def update_task_status(self, task_id: str, status: str, progress: Optional[int] = None):
        """更新任务状态"""
        tasks = self._load_tasks()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for task in tasks:
            if task["id"] == task_id:
                task["status"] = status
                task["updated_at"] = now
                if progress is not None:
                    task["progress_percent"] = progress
                if status == "已完成":
                    task["completed_date"] = datetime.now().strftime("%Y-%m-%d")
                break

        self._save_tasks(tasks)

    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        tasks = self._load_tasks()
        original_count = len(tasks)
        tasks = [t for t in tasks if t["id"] != task_id]
        if len(tasks) < original_count:
            self._save_tasks(tasks)
            return True
        return False

    # ========== 进度管理 ==========

    def create_progress(self, progress_data: Dict[str, Any]) -> PlantingProgress:
        """创建新的进度记录"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        progress = PlantingProgress(
            id=self._generate_id(),
            crop=progress_data.get("crop", ""),
            stage=progress_data.get("stage", ""),
            stage_number=progress_data.get("stage_number", 1),
            total_stages=progress_data.get("total_stages", 1),
            start_date=progress_data.get("start_date", datetime.now().strftime("%Y-%m-%d")),
            expected_end_date=progress_data.get("expected_end_date", ""),
            actual_end_date=None,
            progress_percent=progress_data.get("progress_percent", 0),
            status=progress_data.get("status", "进行中"),
            tasks=progress_data.get("tasks", []),
            notes=progress_data.get("notes", ""),
            created_at=now,
            updated_at=now
        )

        self._save_progress(progress)
        return progress

    def _save_progress(self, progress: PlantingProgress):
        """保存进度到文件"""
        progresses = self._load_progresses()
        progresses.append(asdict(progress))
        self._save_progresses(progresses)

    def _load_progresses(self) -> List[Dict[str, Any]]:
        """加载所有进度"""
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载进度失败: {e}")
            return []

    def _save_progresses(self, progresses: List[Dict[str, Any]]):
        """保存进度列表"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progresses, f, ensure_ascii=False, indent=2)

    def get_progress(self, crop: Optional[str] = None) -> List[PlantingProgress]:
        """获取进度列表"""
        progresses_data = self._load_progresses()
        progresses = [PlantingProgress(**data) for data in progresses_data]

        if crop:
            progresses = [p for p in progresses if p.crop == crop]

        # 按时间倒序排列
        progresses.sort(key=lambda x: x.created_at, reverse=True)
        return progresses

    def update_progress(self, progress_id: str, updates: Dict[str, Any]):
        """更新进度"""
        progresses = self._load_progresses()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for progress in progresses:
            if progress["id"] == progress_id:
                progress.update(updates)
                progress["updated_at"] = now
                break

        self._save_progresses(progresses)

    def delete_progress(self, progress_id: str) -> bool:
        """删除进度记录"""
        progresses = self._load_progresses()
        original_count = len(progresses)
        progresses = [p for p in progresses if p["id"] != progress_id]
        if len(progresses) < original_count:
            self._save_progresses(progresses)
            return True
        return False

    def advance_to_next_stage(self, progress_id: str) -> Dict[str, Any]:
        """
        推进到下一阶段

        返回:
            {
                "success": True/False,
                "message": "操作结果消息",
                "new_stage": "新阶段名称",
                "stage_number": 新阶段编号,
                "is_completed": 是否全部完成
            }
        """
        progresses = self._load_progresses()

        for progress in progresses:
            if progress["id"] == progress_id:
                current_stage = progress.get("stage_number", 0)
                total_stages = progress.get("total_stages", 1)
                crop = progress.get("crop", "")

                if current_stage < total_stages:
                    # 推进到下一阶段
                    next_stage = current_stage + 1
                    progress["stage_number"] = next_stage
                    progress["progress_percent"] = 0
                    progress["status"] = "进行中"
                    progress["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # 尝试从作物知识库获取阶段名称
                    stage_name = self._get_stage_name(crop, next_stage, total_stages)
                    progress["stage"] = stage_name

                    self._save_progresses(progresses)
                    return {
                        "success": True,
                        "message": f"已进入下一阶段：{stage_name}",
                        "new_stage": stage_name,
                        "stage_number": next_stage,
                        "is_completed": False
                    }
                else:
                    # 所有阶段已完成
                    progress["status"] = "已完成"
                    progress["progress_percent"] = 100
                    progress["actual_end_date"] = datetime.now().strftime("%Y-%m-%d")
                    progress["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self._save_progresses(progresses)
                    return {
                        "success": True,
                        "message": "恭喜！所有阶段已完成！",
                        "new_stage": progress.get("stage", "完成"),
                        "stage_number": current_stage,
                        "is_completed": True
                    }

        return {"success": False, "message": "未找到进度记录"}

    def _get_stage_name(self, crop: str, stage_number: int, total_stages: int) -> str:
        """根据作物和阶段编号获取阶段名称"""
        try:
            # 尝试从作物知识库获取
            from .planting_planner import CropDatabase
            crop_db = CropDatabase()
            crop_info = crop_db.get_crop(crop)

            if crop_info and crop_info.growth_stages:
                stages = crop_info.growth_stages
                if stage_number <= len(stages):
                    return stages[stage_number - 1].get("stage", f"第{stage_number}阶段")
        except Exception:
            pass

        # 默认阶段名称
        default_stages = ["准备期", "播种期", "苗期", "生长期", "开花期", "结果期", "成熟期", "收获期"]
        if stage_number <= len(default_stages):
            return default_stages[stage_number - 1]
        return f"第{stage_number}阶段"

    def auto_calculate_progress(self, progress_id: str) -> Dict[str, Any]:
        """
        根据日期和作物种类自动计算进度

        返回:
            {
                "success": True/False,
                "stage_number": 当前阶段编号,
                "stage_name": 当前阶段名称,
                "progress_percent": 当前阶段进度百分比,
                "total_stages": 总阶段数,
                "days_elapsed": 总已进行天数,
                "days_in_stage": 在当前阶段已进行天数,
                "status": 状态,
                "message": 说明信息
            }
        """
        progresses = self._load_progresses()

        for progress in progresses:
            if progress["id"] == progress_id:
                crop = progress.get("crop", "")
                start_date_str = progress.get("start_date", "")

                if not start_date_str:
                    return {"success": False, "message": "未设置开始日期，无法计算进度"}

                try:
                    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                    current_date = datetime.now()
                    days_elapsed = (current_date - start_date).days

                    if days_elapsed < 0:
                        return {"success": False, "message": "开始日期在未来，无法计算进度"}

                    # 获取作物生长阶段信息
                    from .planting_planner import CropDatabase
                    crop_db = CropDatabase()
                    crop_info = crop_db.get_crop(crop)

                    if crop_info and crop_info.growth_stages:
                        # 使用作物实际生长阶段计算
                        result = self._calculate_with_crop_stages(days_elapsed, crop_info.growth_stages)
                    else:
                        # 使用默认阶段计算
                        result = self._calculate_with_default_stages(days_elapsed)

                    # 更新进度记录
                    progress["stage_number"] = result["stage_number"]
                    progress["stage"] = result["stage_name"]
                    progress["progress_percent"] = result["progress_percent"]
                    progress["total_stages"] = result["total_stages"]
                    progress["status"] = result["status"]
                    progress["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self._save_progresses(progresses)

                    return {
                        "success": True,
                        "stage_number": result["stage_number"],
                        "stage_name": result["stage_name"],
                        "progress_percent": result["progress_percent"],
                        "total_stages": result["total_stages"],
                        "days_elapsed": days_elapsed,
                        "days_in_stage": result["days_in_stage"],
                        "status": result["status"],
                        "message": f"已自动计算进度：{result['stage_name']} ({result['progress_percent']}%)"
                    }

                except Exception as e:
                    return {"success": False, "message": f"计算进度失败: {str(e)}"}

        return {"success": False, "message": "未找到进度记录"}

    def _calculate_with_crop_stages(self, days_elapsed: int, growth_stages: List[Dict]) -> Dict[str, Any]:
        """根据作物实际生长阶段计算进度"""
        cumulative_days = 0
        total_stages = len(growth_stages)

        for i, stage in enumerate(growth_stages):
            stage_duration = stage.get("duration_days", 30)
            stage_start = cumulative_days
            stage_end = cumulative_days + stage_duration

            if days_elapsed < stage_end:
                # 当前处于这个阶段
                days_in_stage = days_elapsed - stage_start
                progress_percent = min(100, int((days_in_stage / stage_duration) * 100))

                return {
                    "stage_number": i + 1,
                    "stage_name": stage.get("stage", f"第{i+1}阶段"),
                    "progress_percent": progress_percent,
                    "total_stages": total_stages,
                    "days_in_stage": days_in_stage,
                    "status": "进行中"
                }

            cumulative_days += stage_duration

        # 所有阶段已完成
        return {
            "stage_number": total_stages,
            "stage_name": growth_stages[-1].get("stage", "成熟收获期") if growth_stages else "完成",
            "progress_percent": 100,
            "total_stages": total_stages,
            "days_in_stage": 0,
            "status": "已完成"
        }

    def _calculate_with_default_stages(self, days_elapsed: int) -> Dict[str, Any]:
        """使用默认阶段计算进度（当没有作物信息时）"""
        default_stages = [
            {"stage": "准备期", "duration_days": 7},
            {"stage": "播种期", "duration_days": 10},
            {"stage": "苗期", "duration_days": 30},
            {"stage": "生长期", "duration_days": 45},
            {"stage": "开花期", "duration_days": 20},
            {"stage": "结果期", "duration_days": 30},
            {"stage": "成熟期", "duration_days": 15},
            {"stage": "收获期", "duration_days": 7}
        ]
        return self._calculate_with_crop_stages(days_elapsed, default_stages)

    # ========== 卡片数据生成 ==========

    def get_task_cards(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取任务卡片数据（供前端展示）"""
        tasks = self.get_tasks()

        cards = []
        for task in tasks[:limit]:
            # 计算状态颜色
            status_color = {
                "待办": "gray",
                "进行中": "blue",
                "已完成": "green",
                "已逾期": "red"
            }.get(task.status, "gray")

            # 计算优先级颜色
            priority_color = {
                "high": "red",
                "medium": "orange",
                "low": "green"
            }.get(task.priority, "gray")

            # 计算剩余天数
            days_left = None
            if task.end_date:
                try:
                    end = datetime.strptime(task.end_date, "%Y-%m-%d")
                    delta = (end - datetime.now()).days
                    days_left = delta if delta >= 0 else 0
                except:
                    pass

            cards.append({
                "id": task.id,
                "type": "task",
                "crop": task.crop,
                "title": task.title,
                "description": task.description,
                "status": task.status,
                "status_color": status_color,
                "priority": task.priority,
                "priority_color": priority_color,
                "progress": task.progress_percent,
                "start_date": task.start_date,
                "end_date": task.end_date,
                "days_left": days_left,
                "created_at": task.created_at
            })

        return cards

    def get_progress_cards(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取进度卡片数据（供前端展示）"""
        progresses = self.get_progress()

        cards = []
        for progress in progresses[:limit]:
            # 计算状态颜色
            status_color = {
                "进行中": "blue",
                "已完成": "green",
                "待开始": "gray"
            }.get(progress.status, "gray")

            cards.append({
                "id": progress.id,
                "type": "progress",
                "crop": progress.crop,
                "stage": progress.stage,
                "stage_number": progress.stage_number,
                "total_stages": progress.total_stages,
                "progress": progress.progress_percent,
                "status": progress.status,
                "status_color": status_color,
                "start_date": progress.start_date,
                "expected_end_date": progress.expected_end_date,
                "tasks_count": len(progress.tasks),
                "notes": progress.notes
            })

        return cards

    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表盘数据"""
        tasks = self.get_tasks()
        progresses = self.get_progress()

        # 统计任务
        task_stats = {
            "total": len(tasks),
            "pending": len([t for t in tasks if t.status == "待办"]),
            "in_progress": len([t for t in tasks if t.status == "进行中"]),
            "completed": len([t for t in tasks if t.status == "已完成"]),
            "overdue": len([t for t in tasks if t.status == "已逾期"])
        }

        # 统计进度
        progress_stats = {
            "total": len(progresses),
            "active": len([p for p in progresses if p.status == "进行中"]),
            "completed": len([p for p in progresses if p.status == "已完成"])
        }

        # 最近活动
        recent_tasks = [{
            "crop": t.crop,
            "title": t.title,
            "status": t.status,
            "updated_at": t.updated_at
        } for t in sorted(tasks, key=lambda x: x.updated_at, reverse=True)[:5]]

        return {
            "task_stats": task_stats,
            "progress_stats": progress_stats,
            "recent_tasks": recent_tasks,
            "task_cards": self.get_task_cards(),
            "progress_cards": self.get_progress_cards()
        }


# 便捷函数
def create_planting_task(crop: str, task_type: str, title: str,
                         description: str = "", end_date: str = "") -> str:
    """快速创建种植任务"""
    tracker = PlantingTracker()
    task = tracker.create_task({
        "crop": crop,
        "task_type": task_type,
        "title": title,
        "description": description,
        "end_date": end_date
    })
    return task.id


def create_planting_progress(crop: str, stage: str, stage_number: int,
                            total_stages: int, expected_end_date: str) -> str:
    """快速创建进度记录"""
    tracker = PlantingTracker()
    progress = tracker.create_progress({
        "crop": crop,
        "stage": stage,
        "stage_number": stage_number,
        "total_stages": total_stages,
        "expected_end_date": expected_end_date
    })
    return progress.id


if __name__ == "__main__":
    # 测试
    tracker = PlantingTracker()

    # 创建测试任务
    task_id = create_planting_task(
        crop="小麦",
        task_type="浇水",
        title="给小麦浇水",
        description="注意浇透但不要积水",
        end_date=(datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    )
    print(f"创建任务: {task_id}")

    # 创建测试进度
    progress_id = create_planting_progress(
        crop="小麦",
        stage="拔节期",
        stage_number=3,
        total_stages=8,
        expected_end_date=(datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d")
    )
    print(f"创建进度: {progress_id}")

    # 获取仪表盘数据
    dashboard = tracker.get_dashboard_data()
    print(f"\n仪表盘数据:")
    print(f"  任务统计: {dashboard['task_stats']}")
    print(f"  进度统计: {dashboard['progress_stats']}")