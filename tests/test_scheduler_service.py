"""独立调度器生命周期和单实例保护测试。"""

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
