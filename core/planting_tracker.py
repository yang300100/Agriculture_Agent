"""
种植进度跟踪与卡片展示模块
功能：
- 记录种植进度和农事提醒
- 生成卡片数据供前端展示
- 支持进度更新和历史记录
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)
from dataclasses import dataclass, asdict
from enum import Enum

import dotenv

from core.storage_paths import DEFAULT_DATA_DIR

# 加载环境变量
dotenv.load_dotenv()
DEFAULT_STORAGE_DIR = DEFAULT_DATA_DIR


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "待办"
    IN_PROGRESS = "进行中"
    COMPLETED = "已完成"
    OVERDUE = "已逾期"
    RUNNING = "运行中"
    ABORTED = "已中止"
    FAILED = "失败"


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
    # ── 设备控制集成（可选）──
    device_id: Optional[str] = None
    device_command: Optional[str] = None
    device_params: Optional[Dict[str, Any]] = None
    execution_state: Optional[Dict[str, Any]] = None


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
    plot_id: Optional[str] = None


class PlantingTracker:
    """种植跟踪管理器"""

    def __init__(self, storage_dir: str = None):
        from core.database.engine import init_db
        init_db()
        storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        self._username = os.path.basename(storage_dir.rstrip("/\\"))
        from core.database.repository.users import UserRepository
        user_repo = UserRepository()
        self._user = user_repo.get_by_username(self._username)
        if not self._user:
            self._user = user_repo.create(username=self._username, password_hash="")

    @property
    def _uid(self) -> int:
        return self._user.id

    def _find_owned_record(self, repo, record_id: str):
        """只查哥哥自己的记录，避免误操作其他用户的数据。"""
        try:
            numeric_id = int(record_id)
        except (ValueError, TypeError):
            return None
        return repo.find_one(id=numeric_id, user_id=self._uid)

    @staticmethod
    def _decode_device_params(value) -> Optional[Dict[str, Any]]:
        """把旧记录中的 JSON 安全还原成设备参数。"""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                return None
        return value if isinstance(value, dict) else None

    # ========== 任务管理 ==========

    def create_task(self, task_data: Dict[str, Any]) -> PlantingTask:
        """创建新任务"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        task = PlantingTask(
            id="",
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
            updated_at=now,
            device_id=task_data.get("device_id"),
            device_command=task_data.get("device_command"),
            device_params=task_data.get("device_params"),
        )

        self._save_task(task)
        return task

    def _save_task(self, task: PlantingTask):
        """只插入新任务，保留哥哥已有任务的 ID 和时间。"""
        from core.database.repository.planting import PlantingTaskRepository

        values = asdict(task)
        for key in ("id", "created_at", "updated_at"):
            values.pop(key)
        values["device_params"] = (
            json.dumps(task.device_params, ensure_ascii=False)
            if task.device_params is not None else None
        )
        values["execution_state"] = json.dumps(task.execution_state) if task.execution_state else None
        row = PlantingTaskRepository().create(user_id=self._uid, **values)
        task.id = str(row.id)
        task.created_at = row.created_at.isoformat()
        task.updated_at = row.updated_at.isoformat()

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """从DB加载所有任务"""
        from core.database.repository.planting import PlantingTaskRepository
        repo = PlantingTaskRepository()
        tasks = repo.find_by(user_id=self._uid)
        return [{
            "id": str(t.id), "crop": t.crop or "", "task_type": t.task_type or "",
            "title": t.title or "", "description": t.description or "",
            "status": t.status or "待办", "priority": t.priority or "medium",
            "start_date": t.start_date.isoformat() if t.start_date else "",
            "end_date": t.end_date.isoformat() if t.end_date else "",
            "completed_date": t.completed_date.isoformat() if t.completed_date else "",
            "progress_percent": 100 if t.status == "已完成" else (t.progress_percent or 0),
            "notes": t.notes or "",
            "created_at": t.created_at.isoformat() if t.created_at else "",
            "updated_at": t.updated_at.isoformat() if t.updated_at else "",
            "device_id": t.device_id, "device_command": t.device_command,
            "device_params": self._decode_device_params(t.device_params),
            "execution_state": self._decode_device_params(t.execution_state),
        } for t in tasks]

    def get_tasks(self, crop: Optional[str] = None, status: Optional[str] = None) -> List[PlantingTask]:
        """获取任务列表，支持筛选"""
        self.sync_running_device_tasks()
        tasks_data = self._load_tasks()
        tasks = [PlantingTask(**data) for data in tasks_data]
        today = datetime.now().date()
        for task in tasks:
            if task.status in ("待办", "进行中") and task.end_date:
                try:
                    if datetime.strptime(task.end_date, "%Y-%m-%d").date() < today:
                        task.status = "已逾期"
                except (ValueError, TypeError):
                    logger.warning("无法解析任务截止日期: %s", task.end_date)

        if crop:
            tasks = [t for t in tasks if t.crop == crop]
        if status:
            tasks = [t for t in tasks if t.status == status]

        # 按优先级和时间排序
        priority_order = {"high": 0, "medium": 1, "low": 2}
        tasks.sort(key=lambda x: (priority_order.get(x.priority, 1), x.end_date))

        return tasks

    def update_task_status(self, task_id: str, status: str, progress: Optional[int] = None) -> bool:
        """更新任务状态，返回 True 表示找到并更新，False 表示未找到"""
        from core.database.repository.planting import PlantingTaskRepository

        repo = PlantingTaskRepository()
        task = self._find_owned_record(repo, task_id)
        if task is None:
            return False
        updates = {"status": status, "updated_at": datetime.now()}
        if progress is not None:
            updates["progress_percent"] = progress
        if status == "已完成":
            updates["progress_percent"] = 100
            updates["completed_date"] = datetime.now().date()
        from core.reminder_system import ReminderSystem
        reminders = ReminderSystem(username=self._username)
        with repo.session_scope() as session:
            row = session.merge(task)
            for key, value in updates.items():
                setattr(row, key, value)
            reminders.sync_task_reminders(task_id, status=status, session=session)
            session.commit()
        return True

    def update_task(self, task_id: str, changes: Dict[str, Any]) -> bool:
        """哥哥，只编辑安排和说明，运行中的设备参数与状态原样保留。"""
        from core.database.repository.planting import PlantingTaskRepository

        allowed = {"title", "crop", "task_type", "description", "priority", "start_date", "end_date"}
        if set(changes) - allowed:
            raise ValueError("任务编辑包含不可修改的字段")
        repo = PlantingTaskRepository()
        task = self._find_owned_record(repo, task_id)
        if task is None:
            return False
        dates = {}
        for key in ("start_date", "end_date"):
            value = changes.get(key, getattr(task, key))
            if isinstance(value, str):
                if value:
                    try:
                        parsed = datetime.strptime(value, "%Y-%m-%d").date()
                        if parsed.isoformat() != value:
                            raise ValueError()
                    except ValueError as exc:
                        raise ValueError("日期必须为有效的 YYYY-MM-DD") from exc
                    dates[key] = parsed
                else:
                    dates[key] = None
            else:
                dates[key] = value
        if dates["start_date"] and dates["end_date"] and dates["start_date"] > dates["end_date"]:
            raise ValueError("截止日期不能早于开始日期")
        if "priority" in changes and changes["priority"] not in {"high", "medium", "low"}:
            raise ValueError("优先级必须为 high、medium 或 low")
        updates = {**changes, "updated_at": datetime.now()}
        from core.reminder_system import ReminderSystem
        reminders = ReminderSystem(username=self._username)
        with repo.session_scope() as session:
            row = session.merge(task)
            for key, value in repo._coerce_dates(updates).items():
                setattr(row, key, value)
            if "start_date" in changes or "end_date" in changes:
                new_date = changes.get("end_date") or changes.get("start_date")
                reminders.sync_task_reminders(task_id, start_date=new_date, session=session)
            session.commit()
        return True

    def record_device_task_result(self, task_id: str, result: Dict[str, Any], registry=None, loop=None) -> None:
        """哥哥，下发成功只代表开始；持续作业要等设备的真实结束证据。"""
        from core.database.repository.planting import PlantingTaskRepository

        repo = PlantingTaskRepository()
        task = self._find_owned_record(repo, task_id)
        if task is None:
            return
        if not result.get("success"):
            pending = result.get("decision") in {"need_confirm", "rejected", "notify_only"}
            self.update_task_status(task_id, "待办" if pending else "失败", 0)
            return
        if task.device_command != "start":
            self.update_task_status(task_id, "已完成", 100)
            self.sync_running_device_tasks(registry, loop)
            return
        device_result = result.get("result")
        state = getattr(device_result, "actual_params", None) or {}
        if not state.get("run_id") and registry is not None and loop is not None:
            state = loop.run_until_complete(registry.read_state(task.device_id))
        runtime = {
            "run_id": state.get("run_id"),
            "started_at": state.get("run_started_at") or datetime.now().isoformat(),
            "run_until": state.get("run_until"),
        }
        repo.update(task.id, status="运行中", progress_percent=10,
                    execution_state=json.dumps(runtime, ensure_ascii=False), updated_at=datetime.now())

    def sync_running_device_tasks(self, registry=None, loop=None) -> None:
        """哥哥，跨请求恢复设备状态；未知和离线不能当作完成。"""
        from core.database.repository.planting import PlantingTaskRepository
        repo = PlantingTaskRepository()
        # 哥哥，保留数据库原始执行快照，提交结果时才知道有没有被新一轮领取替换。
        tasks = [row for row in repo.find_by(user_id=self._uid) if (
            row.status == "运行中" or (
                row.status == "进行中" and row.device_command == "start"
                and (self._decode_device_params(row.execution_state) or {}).get("run_id")
            )
        )]
        if not tasks:
            return
        if registry is None or loop is None:
            from core.device_registry_factory import RegistrySession
            try:
                with RegistrySession(self._username) as (owned_registry, owned_loop):
                    owned_loop.run_until_complete(owned_registry.discover_all())
                    self.sync_running_device_tasks(owned_registry, owned_loop)
            except Exception:
                logger.warning("读取运行中任务设备状态失败，保留原状态", exc_info=True)
            return
        from core.database.models import PlantingTask as TaskRecord
        from core.reminder_system import ReminderSystem
        reminders = ReminderSystem(username=self._username)
        states = {}
        for task in tasks:
            if task.device_id not in states:
                states[task.device_id] = loop.run_until_complete(registry.read_state(task.device_id))
            state = states[task.device_id]
            if not isinstance(state, dict) or (
                state.get("error") and state.get("status") not in {"error", "fault", "failed"}
            ):
                continue
            runtime = self._decode_device_params(task.execution_state) or {}
            run_id = runtime.get("run_id")
            history = state.get("run_history") or {}
            ended = history.get(run_id) or {} if isinstance(history, dict) else {}
            if not ended and run_id and state.get("last_ended_run_id") == run_id:
                ended = {"reason": state.get("last_run_end_reason"), "ended_at": state.get("last_run_ended_at")}
            if task.status == "进行中":
                if not run_id or (run_id != state.get("run_id") and not ended):
                    continue
                source = ended or {
                    "started_at": state.get("run_started_at"), "run_until": state.get("run_until"),
                }
                runtime.update(
                    started_at=source.get("started_at") or runtime.get("started_at"),
                    run_until=source.get("run_until", runtime.get("run_until")),
                )
            status = None
            if ended:
                reason = ended.get("reason")
                if reason == "completed":
                    status = "已完成"
                elif reason == "interrupted":
                    status = "已中止"
                elif reason == "failed":
                    status = "失败"
            if not status and state.get("status") in {"error", "fault", "failed"}:
                status = "失败"
            if not status and run_id and state.get("run_id") and state["run_id"] != run_id:
                status = "已中止"
            if status:
                runtime.update(ended_at=ended.get("ended_at") or datetime.now().isoformat(),
                               end_reason=status)
            if not status and task.status != "进行中":
                continue
            updates = {
                "status": status or "运行中",
                "execution_state": json.dumps(runtime, ensure_ascii=False),
                "updated_at": datetime.now(),
            }
            if status == "已完成":
                updates.update(progress_percent=100, completed_date=datetime.now().date())
            # 哥哥，状态、执行编号和关联提醒一起提交；旧读取绝不能盖掉新启动的回执。
            with repo.session_scope() as session:
                changed = session.query(TaskRecord).filter(
                    TaskRecord.id == task.id, TaskRecord.user_id == self._uid,
                    TaskRecord.status == task.status,
                    TaskRecord.execution_state == task.execution_state,
                ).update(updates, synchronize_session=False)
                if changed and status:
                    reminders.sync_task_reminders(str(task.id), status=status, session=session)
                session.commit()

    def claim_device_task(self, task_id: str) -> bool:
        """哥哥，先原子领取设备任务，避免手动操作和后台同时下发。"""
        from sqlalchemy import or_
        from core.database.engine import get_session
        from core.database.models import PlantingTask as TaskRecord

        try:
            numeric_id = int(task_id)
        except (ValueError, TypeError):
            return False
        with get_session() as session:
            query = session.query(TaskRecord).filter(
                TaskRecord.id == numeric_id,
                TaskRecord.user_id == self._uid,
                or_(
                    TaskRecord.status.in_(["待办", "已逾期", "失败", "已中止", ""]),
                    TaskRecord.status.is_(None),
                ),
                TaskRecord.device_id.isnot(None),
                TaskRecord.device_id != "",
                TaskRecord.device_command.isnot(None),
                TaskRecord.device_command != "",
            )
            task = query.first()
            if task is None:
                return False
            runtime = self._decode_device_params(task.execution_state) or {}
            if not runtime.get("run_id") or runtime.get("started_at") or task.status == "失败":
                import uuid
                runtime = {"run_id": uuid.uuid4().hex}
            claimed = query.update({
                "status": "进行中", "progress_percent": 10,
                "execution_state": json.dumps(runtime, ensure_ascii=False),
                "updated_at": datetime.now(),
            }, synchronize_session=False)
            session.commit()
            return bool(claimed)

    def device_task_execution_lock(self, task_id: str):
        """哥哥，同一任务在API、确认与后台之间共用一个系统执行锁。"""
        from core.device_executor import _PendingFileLock
        from core.storage_paths import DEFAULT_DATA_DIR
        numeric_id = int(task_id)
        return _PendingFileLock(os.path.join(DEFAULT_DATA_DIR, self._username, f"task_{numeric_id}.executing.lock"))

    def recover_unrecorded_device_task(self, task_id: str, registry, loop) -> bool:
        """哥哥，仅在持有任务锁的显式重试中恢复虚拟设备的孤立领取，保留执行编号。"""
        from core.device_registry_factory import load_custom_devices
        from core.database.engine import get_session
        from core.database.models import PlantingTask as TaskRecord

        self.sync_running_device_tasks(registry, loop)
        task = next((row for row in self._load_tasks() if row["id"] == str(task_id)), None)
        if not task or task["status"] != "进行中":
            return False
        config = next((row for row in load_custom_devices(self._username) if row["device_id"] == task["device_id"]), {})
        # 真实协议未声明执行编号幂等时，不能根据缺失回执猜测设备没有启动。
        if config.get("driver") != "simulator":
            return False
        state = loop.run_until_complete(registry.read_state(task["device_id"]))
        if not isinstance(state, dict) or state.get("error") or state.get("status") not in {"powered_off", "standby", "running"}:
            return False
        runtime = task.get("execution_state") or {}
        run_id = runtime.get("run_id")
        if not run_id or state.get("run_id") == run_id or state.get("last_ended_run_id") == run_id or run_id in (state.get("run_history") or {}):
            return False
        with get_session() as session:
            recovered = session.query(TaskRecord).filter(
                TaskRecord.id == int(task_id), TaskRecord.user_id == self._uid,
                TaskRecord.status == "进行中",
            ).update({"status": "待办", "progress_percent": 0, "updated_at": datetime.now()}, synchronize_session=False)
            session.commit()
            return bool(recovered)

    def recover_interrupted_device_task(self, task_id: str, interrupted_at: str) -> bool:
        """哥哥，只恢复旧执行留下的任务，不能覆盖后来已经领取的操作。"""
        self.sync_running_device_tasks()
        from core.database.engine import get_session
        from core.database.models import PlantingTask as TaskRecord

        try:
            numeric_id = int(task_id)
            interrupted_time = datetime.fromisoformat(interrupted_at)
        except (ValueError, TypeError):
            return False
        if interrupted_time.tzinfo is not None:
            interrupted_time = interrupted_time.astimezone().replace(tzinfo=None)
        with get_session() as session:
            recovered = session.query(TaskRecord).filter(
                TaskRecord.id == numeric_id,
                TaskRecord.user_id == self._uid,
                TaskRecord.status == "进行中",
                TaskRecord.updated_at <= interrupted_time,
            ).update({
                "status": "待办", "progress_percent": 0, "updated_at": datetime.now(),
            }, synchronize_session=False)
            session.commit()
            return bool(recovered)

    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        from core.database.repository.planting import PlantingTaskRepository

        repo = PlantingTaskRepository()
        task = self._find_owned_record(repo, task_id)
        if task is None:
            return False
        from core.reminder_system import ReminderSystem
        reminders = ReminderSystem(username=self._username)
        with repo.session_scope() as session:
            reminders.sync_task_reminders(task_id, deleted=True, session=session)
            session.delete(session.merge(task))
            session.commit()
        return True

    def get_task_by_id(self, task_id: str) -> Optional[PlantingTask]:
        """按 ID 获取单个任务"""
        self.sync_running_device_tasks()
        tasks = self._load_tasks()
        for task_data in tasks:
            if task_data.get("id") == task_id:
                return PlantingTask(**task_data)
        return None

    def get_pending_device_tasks(self) -> List[PlantingTask]:
        """获取所有待执行的设备任务（状态为待办且有 device_id）"""
        tasks = self.get_tasks(status="待办")
        return [t for t in tasks if t.device_id and t.device_command]

    # ========== 进度管理 ==========

    def validate_plot_id(self, plot_id):
        if plot_id in (None, ""):
            return None
        if isinstance(plot_id, bool) or not isinstance(plot_id, (int, str)):
            raise ValueError("地块编号必须是整数")
        from core.database.repository.fields import FieldRepository
        try:
            numeric_id = int(plot_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("地块编号无效") from exc
        if not FieldRepository().find_one(id=numeric_id, user_id=self._uid):
            raise ValueError("地块不存在或不属于当前账号")
        return str(numeric_id)

    def create_progress(self, progress_data: Dict[str, Any]) -> PlantingProgress:
        """创建新的进度记录"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        progress = PlantingProgress(
            id="",
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
            updated_at=now,
            plot_id=self.validate_plot_id(progress_data.get("plot_id")),
        )

        if progress.start_date:
            start = datetime.strptime(progress.start_date, "%Y-%m-%d").date()
            if start > datetime.now().date():
                progress.status = "待开始"
                progress.progress_percent = 0

        self._save_progress(progress)
        return progress

    def _save_progress(self, progress: PlantingProgress):
        """只插入新进度，把数据库中的真实 ID 返回给哥哥。"""
        from core.database.repository.planting import PlantingPlanRepository

        values = asdict(progress)
        for key in ("id", "tasks", "created_at", "updated_at"):
            values.pop(key)
        row = PlantingPlanRepository().create(user_id=self._uid, **values)
        progress.id = str(row.id)
        progress.created_at = row.created_at.isoformat()
        progress.updated_at = row.updated_at.isoformat()

    def _load_progresses(self) -> List[Dict[str, Any]]:
        """从DB加载所有进度"""
        from core.database.repository.planting import PlantingPlanRepository
        repo = PlantingPlanRepository()
        plans = repo.find_by(user_id=self._uid)
        return [{
            "id": str(p.id), "crop": p.crop or "", "stage": p.stage or "",
            "plot_id": str(p.plot_id) if p.plot_id is not None else None,
            "stage_number": p.stage_number or 1, "total_stages": p.total_stages or 4,
            "start_date": p.start_date.isoformat() if p.start_date else "",
            "expected_end_date": p.expected_end_date.isoformat() if p.expected_end_date else "",
            "actual_end_date": p.actual_end_date.isoformat() if p.actual_end_date else "",
            "progress_percent": p.progress_percent or 0,
            "status": p.status or "进行中",
            "tasks": [],  # DB模型无此字段，默认空
            "notes": p.notes or "",
            "created_at": p.created_at.isoformat() if p.created_at else "",
            "updated_at": p.updated_at.isoformat() if p.updated_at else "",
        } for p in plans]

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
        from core.database.repository.planting import PlantingPlanRepository

        repo = PlantingPlanRepository()
        progress = self._find_owned_record(repo, progress_id)
        if progress is None:
            return False
        allowed = {
            "crop", "stage", "stage_number", "total_stages", "start_date",
            "expected_end_date", "actual_end_date", "progress_percent", "status", "notes", "plot_id",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        if "plot_id" in values:
            values["plot_id"] = self.validate_plot_id(values["plot_id"])
        values["updated_at"] = datetime.now()
        repo.update(progress.id, **values)
        return True

    def delete_progress(self, progress_id: str) -> bool:
        """删除进度记录"""
        from core.database.repository.planting import PlantingPlanRepository

        repo = PlantingPlanRepository()
        progress = self._find_owned_record(repo, progress_id)
        return repo.delete(progress.id) if progress is not None else False

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
        from core.database.repository.planting import PlantingPlanRepository

        repo = PlantingPlanRepository()
        row = repo.get_by_id(int(progress_id)) if str(progress_id).isdigit() else None
        if not row or row.user_id != self._uid:
            return {"success": False, "message": "未找到进度记录"}
        if row.start_date and row.start_date > datetime.now().date():
            return {"success": False, "message": "哥哥，计划尚未到开始日期，不能提前推进阶段。"}

        progress = self._effective_progress(self._load_progress_by_id(str(row.id)))
        current_stage = progress.stage_number
        total_stages = progress.total_stages
        crop = row.crop or ""

        if current_stage < total_stages:
            next_stage = current_stage + 1
            stage_name = self._get_stage_name(crop, next_stage, total_stages)
            repo.update(
                row.id,
                stage_number=next_stage,
                total_stages=total_stages,
                progress_percent=0,
                status="进行中",
                stage=stage_name,
                updated_at=datetime.now(),
            )
            return {
                "success": True,
                "message": f"已进入下一阶段：{stage_name}",
                "new_stage": stage_name,
                "stage_number": next_stage,
                "is_completed": False,
            }

        repo.update(
            row.id,
            stage_number=current_stage,
            stage=progress.stage,
            total_stages=total_stages,
            status="已完成",
            progress_percent=100,
            actual_end_date=datetime.now().date(),
            updated_at=datetime.now(),
        )
        return {
            "success": True,
            "message": "恭喜！所有阶段已完成！",
            "new_stage": progress.stage or "完成",
            "stage_number": current_stage,
            "is_completed": True,
        }

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
                        self.update_progress(progress_id, {"status": "待开始", "progress_percent": 0})
                        return {"success": True, "status": "待开始", "progress_percent": 0, "stage_progress": 0, "overall_progress": 0, "message": "计划尚未开始"}

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

                    # 哥哥，保存值和返回值都尊重已经人工确认的阶段与完成状态。
                    effective = PlantingProgress(**progress)
                    if effective.status != "已完成":
                        effective = self._merge_display_data(effective, result)
                    self.update_progress(progress_id, asdict(effective))

                    return {
                        "success": True,
                        "stage_number": effective.stage_number,
                        "stage_name": effective.stage,
                        "progress_percent": effective.progress_percent,
                        "stage_progress": effective.progress_percent,
                        "overall_progress": self._overall_progress(effective),
                        "total_stages": effective.total_stages,
                        "days_elapsed": days_elapsed,
                        "days_in_stage": result["days_in_stage"] if effective.stage_number == result["stage_number"] else 0,
                        "status": effective.status,
                        "message": f"当前阶段：{effective.stage} ({effective.progress_percent}%)"
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

    def _calc_progress_display(self, progress) -> Optional[Dict[str, Any]]:
        """计算进度展示数据（纯计算，不持久化）"""
        crop = getattr(progress, "crop", "")
        start_date_str = getattr(progress, "start_date", "")
        if not start_date_str:
            return None
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            current_date = datetime.now()
            days_elapsed = (current_date - start_date).days
            if days_elapsed < 0:
                return {"stage_number": progress.stage_number, "stage_name": progress.stage, "progress_percent": 0, "total_stages": progress.total_stages, "status": "待开始"}
            from .planting_planner import CropDatabase
            crop_db = CropDatabase()
            crop_info = crop_db.get_crop(crop)
            if crop_info and crop_info.growth_stages:
                return self._calculate_with_crop_stages(days_elapsed, crop_info.growth_stages)
            else:
                return self._calculate_with_default_stages(days_elapsed)
        except Exception:
            return None

    def _merge_display_data(self, progress, calc_result: Dict[str, Any]):
        """将计算出的进度数据合并到 progress 对象上（内存操作，不持久化）"""
        from dataclasses import replace
        current_stage = getattr(progress, "stage_number", 0)
        calc_stage = calc_result["stage_number"]
        if calc_stage > current_stage:
            updates = {
                "stage_number": calc_stage,
                "stage": calc_result["stage_name"],
                "total_stages": calc_result["total_stages"],
                "status": calc_result["status"],
                "progress_percent": calc_result["progress_percent"],
            }
        elif calc_stage == current_stage:
            updates = {
                "stage": calc_result["stage_name"],
                "progress_percent": calc_result["progress_percent"],
                "total_stages": calc_result["total_stages"],
                "status": calc_result["status"],
            }
        else:
            return progress  # 用户手动推进过了，保持不动
        return replace(progress, **updates)

    def _overall_progress(self, progress: PlantingProgress) -> int:
        """哥哥，整季进度只按起点和总天数计算，不随阶段清零。"""
        if progress.status == "已完成":
            return 100
        try:
            start = datetime.strptime(progress.start_date, "%Y-%m-%d").date()
            elapsed = max(0, (datetime.now().date() - start).days)
            if progress.expected_end_date:
                end = datetime.strptime(progress.expected_end_date, "%Y-%m-%d").date()
                total = (end - start).days
            else:
                from .planting_planner import CropDatabase
                info = CropDatabase().get_crop(progress.crop)
                total = sum(stage.get("duration_days", 30) for stage in info.growth_stages) if info and info.growth_stages else 164
            return min(100, int(elapsed * 100 / total)) if total > 0 else 0
        except (ValueError, TypeError):
            return 0

    def _effective_progress(self, progress: PlantingProgress) -> PlantingProgress:
        """哥哥，卡片和人工推进共用有效阶段，显示值仍不偷偷写入数据库。"""
        if progress.status != "已完成":
            result = self._calc_progress_display(progress)
            if result is not None:
                return self._merge_display_data(progress, result)
        return progress

    # ========== 卡片数据生成 ==========

    def get_task_cards(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取任务卡片数据（供前端展示）"""
        tasks = self.get_tasks()

        cards = []
        for task in tasks[:limit]:
            retry_available = False
            if task.status == "进行中":
                from core.device_registry_factory import load_custom_devices
                config = next((item for item in load_custom_devices(self._username) if item["device_id"] == task.device_id), {})
                if config.get("driver") == "simulator":
                    lock = self.device_task_execution_lock(task.id)
                    if lock.acquire():
                        retry_available = True
                        lock.release()
            # 计算状态颜色
            status_color = {
                "待办": "gray",
                "进行中": "blue",
                "运行中": "blue",
                "已中止": "orange",
                "失败": "red",
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
                    delta = (end.date() - datetime.now().date()).days
                    days_left = delta if delta >= 0 else 0
                except (ValueError, TypeError):
                    logger.warning("无法解析任务截止日期: %s", task.end_date)

            cards.append({
                "id": task.id,
                "type": "task",
                "crop": task.crop,
                "task_type": task.task_type,
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
                "created_at": task.created_at,
                # 设备控制字段
                "device_id": task.device_id,
                "device_command": task.device_command,
                "device_params": task.device_params,
                "execution_state": task.execution_state,
                "started_at": (task.execution_state or {}).get("started_at"),
                "run_until": (task.execution_state or {}).get("run_until"),
                "ended_at": (task.execution_state or {}).get("ended_at"),
                "retry_available": retry_available,
            })

        return cards

    def get_progress_cards(self, limit: Optional[int] = 5) -> List[Dict[str, Any]]:
        """获取进度卡片数据；哥哥可传 None 获取全部，自动计算不持久化。"""
        progresses = self.get_progress()
        cards = []
        for progress in progresses[:limit]:
            refreshed = self._effective_progress(progress)

            status_color = {
                "进行中": "blue",
                "已完成": "green",
                "待开始": "gray"
            }.get(refreshed.status, "gray")

            stage_label = refreshed.stage
            # 标记自动检测
            if refreshed.status == "进行中" and refreshed.stage_number > 0:
                try:
                    from .planting_planner import CropDatabase
                    crop_db = CropDatabase()
                    crop_info = crop_db.get_crop(refreshed.crop)
                    if crop_info and crop_info.growth_stages:
                        stage_label = f"{refreshed.stage}（预计阶段）"
                except Exception:
                    pass

            cards.append({
                "id": refreshed.id,
                "type": "progress",
                "crop": refreshed.crop,
                "stage": stage_label,
                "stage_number": refreshed.stage_number,
                "total_stages": refreshed.total_stages,
                "progress": refreshed.progress_percent,
                "stage_progress": refreshed.progress_percent,
                "overall_progress": self._overall_progress(refreshed),
                "status": refreshed.status,
                "status_color": status_color,
                "start_date": refreshed.start_date,
                "expected_end_date": refreshed.expected_end_date,
                "tasks_count": len(refreshed.tasks),
                "plot_id": refreshed.plot_id,
                "notes": refreshed.notes,
            })

        return cards

    def _load_progress_by_id(self, progress_id: str) -> Optional[Any]:
        """按 ID 加载单条进度记录"""
        progresses = self._load_progresses()
        for p in progresses:
            if p["id"] == progress_id:
                return PlantingProgress(**p)
        return None

    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表盘数据"""
        tasks = self.get_tasks()
        progresses = self.get_progress()

        # 统计任务
        task_stats = {
            "total": len(tasks),
            "pending": len([t for t in tasks if t.status == "待办"]),
            "in_progress": len([t for t in tasks if t.status in {"进行中", "运行中"}]),
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
