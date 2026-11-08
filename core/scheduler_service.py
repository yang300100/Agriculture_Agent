"""APScheduler 生命周期管理，避免导入模块时产生隐式副作用。"""

import logging
import os
import signal
import threading
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

_scheduler = None
_stop_event = threading.Event()


class SchedulerInstanceLock:
    """持有进程级文件锁，防止同一主机重复启动硬件调度器。"""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        default_path = Path(__file__).resolve().parents[1] / "data" / ".scheduler.lock"
        self.path = Path(path or os.getenv("SCHEDULER_LOCK_FILE", default_path))
        self._handle = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            handle.close()
            return False

        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()).encode("ascii"))
        handle.flush()
        self._handle = handle
        return True

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("已有另一个调度器进程正在运行")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()


def scheduler_enabled() -> bool:
    return os.getenv("ENABLE_SCHEDULER", "false").lower() in (
        "1", "true", "yes", "on"
    )


def _record_heartbeat(status="running", detail=""):
    try:
        from core.autonomy_runtime import record_heartbeat
        record_heartbeat(status=status, detail=detail)
    except Exception:
        logger.exception("记录调度心跳失败")


def _autonomy_maintenance():
    """哥哥，分别尝试维护任务和告警，管道异常才记为维护失败。"""
    from core.autonomy_health import record_maintenance_result
    failures = []
    try:
        from core.autonomy_runtime import process_due_jobs, dispatch_alerts
        for label, operation in (("任务", process_due_jobs), ("告警", dispatch_alerts)):
            try:
                operation()
            except Exception as exc:
                logger.exception("自主维护%s管道失败", label)
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
    except Exception as exc:
        logger.exception("自主维护管道初始化失败")
        failures.append(f"初始化: {type(exc).__name__}: {exc}")
    try:
        record_maintenance_result(not failures, "；".join(failures))
    except Exception:
        logger.exception("记录自主维护健康失败")


def create_scheduler() -> BackgroundScheduler:
    """创建但不启动调度器，方便测试核对任务配置。"""
    from app.scheduler_jobs import (
        check_autonomous_cycle_job,
        check_device_rules_job,
        check_disease_job,
        check_reminders_job,
        check_task_execution_job,
        check_weather_job,
    )

    scheduler = BackgroundScheduler(job_defaults={
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 300,
    })
    scheduler.add_job(_record_heartbeat, "interval", seconds=20, id="autonomy_heartbeat")
    scheduler.add_job(_autonomy_maintenance, "interval", seconds=5, id="autonomy_maintenance")
    scheduler.add_job(check_reminders_job, "interval", minutes=5, id="reminders")
    scheduler.add_job(check_weather_job, "interval", minutes=30, id="weather")
    scheduler.add_job(check_disease_job, "interval", hours=6, id="disease")
    scheduler.add_job(check_device_rules_job, "interval", minutes=5, id="device_rules")
    scheduler.add_job(check_task_execution_job, "interval", minutes=3, id="task_execution")
    scheduler.add_job(
        check_autonomous_cycle_job,
        "interval",
        minutes=int(os.getenv("AUTO_DECISION_INTERVAL", "30")),
        id="autonomous_cycle",
    )
    return scheduler


def start_scheduler():
    """显式启动唯一调度器实例。"""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return _scheduler
    # 启动前在主线程内先初始化数据库，避免调度任务首次运行时在
    # 各自线程中并发导入 core.database 包而触发跨线程模块锁死锁。
    try:
        from core.database.engine import init_db
        init_db()
    except Exception:
        logger.exception("调度进程数据库初始化失败")
        raise
    _scheduler = create_scheduler()
    _scheduler.start()
    _record_heartbeat("running", "调度器已启动")
    logger.info("独立 APScheduler 已启动")
    return _scheduler


def stop_scheduler():
    """安全停止调度器。"""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        # 哥哥，旧设备任务结束后再释放实例锁，避免重启时两边同时下发。
        _scheduler.shutdown(wait=True)
        _record_heartbeat("stopped", "调度器已停止")
    _scheduler = None


def run_scheduler_forever():
    """以前台独立进程运行调度器。"""
    if not scheduler_enabled():
        logger.info("ENABLE_SCHEDULER=false，调度器未启动")
        return 0

    instance_lock = SchedulerInstanceLock()
    if not instance_lock.acquire():
        logger.error("已有另一个调度器进程正在运行，本进程拒绝启动")
        return 2

    _stop_event.clear()

    def _request_stop(signum, frame):
        _stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _request_stop)
        except (ValueError, OSError):
            pass

    try:
        start_scheduler()
        _stop_event.wait()
    finally:
        stop_scheduler()
        instance_lock.release()
    return 0
