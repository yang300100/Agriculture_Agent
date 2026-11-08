"""独立调度器生命周期和单实例保护测试。"""

import threading
from datetime import datetime, timedelta

from core.scheduler_service import SchedulerInstanceLock, run_scheduler_forever


def test_调度器文件锁会拒绝同机重复实例(tmp_path):
    lock_path = tmp_path / "scheduler.lock"
    first = SchedulerInstanceLock(lock_path)
    second = SchedulerInstanceLock(lock_path)

    assert first.acquire() is True
    try:
        assert second.acquire() is False
    finally:
        first.release()

    assert second.acquire() is True
    second.release()


def test_关闭调度配置时不创建锁文件(tmp_path, monkeypatch):
    lock_path = tmp_path / "scheduler.lock"
    monkeypatch.setenv("ENABLE_SCHEDULER", "false")
    monkeypatch.setenv("SCHEDULER_LOCK_FILE", str(lock_path))

    assert run_scheduler_forever() == 0
    assert not lock_path.exists()


def test_停止调度器会等待正在执行的设备任务结束(monkeypatch):
    """哥哥，旧任务结束后才能释放调度器锁，避免新旧进程同时下发。"""
    from core import scheduler_service

    running = threading.Event()
    release_job = threading.Event()
    stopped = threading.Event()
    scheduler = scheduler_service.create_scheduler()
    scheduler.remove_all_jobs()

    def active_device_job():
        running.set()
        release_job.wait(5)

    scheduler.add_job(
        active_device_job, "date", run_date=datetime.now() + timedelta(milliseconds=50)
    )
    monkeypatch.setattr(scheduler_service, "create_scheduler", lambda: scheduler)
    scheduler_service.start_scheduler()

    def stop():
        scheduler_service.stop_scheduler()
        stopped.set()

    stopping = threading.Thread(target=stop)
    try:
        assert running.wait(5)
        stopping.start()
        assert not stopped.wait(0.15)
        release_job.set()
        assert stopped.wait(5)
    finally:
        release_job.set()
        if stopping.ident is not None:
            stopping.join(5)
        scheduler_service.stop_scheduler()
