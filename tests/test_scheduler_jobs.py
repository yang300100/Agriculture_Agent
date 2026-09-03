"""测试自主巡检调度的跨实例限流。"""
from datetime import datetime, timedelta
from pathlib import Path

from app import scheduler_jobs


def test_camera_capture_reuses_driver_saved_file(tmp_path, monkeypatch):
    """驱动已落盘时，调度器不能再保存一份重复照片。"""
    monkeypatch.setattr(scheduler_jobs, "DEFAULT_DATA_DIR", str(tmp_path))
    saved_path = tmp_path / "camera_user" / "photos" / "cam_01" / "capture_driver.jpg"
    saved_path.parent.mkdir(parents=True)
    saved_path.write_bytes(b"driver-image")

    photo_dir, photo_path = scheduler_jobs._persist_capture_if_needed(
        {"saved_path": str(saved_path)},
        b"driver-image",
        "camera_user",
        "cam_01",
        "20260822_220000",
    )

    assert Path(photo_dir) == saved_path.parent
    assert Path(photo_path) == saved_path
    assert list(saved_path.parent.glob("capture_*.jpg")) == [saved_path]


def setup_function():
    """每条测试前清空进程级领取记录。"""
    scheduler_jobs._autonomous_last_runs.clear()
    from core.database.engine import get_session, init_db
    from core.database.models import AutonomousCycleLease

    init_db()
    session = get_session()
    try:
        session.query(AutonomousCycleLease).delete()
        session.commit()
    finally:
        session.close()


def test_claim_cycle_blocks_same_user_region_inside_interval():
    now = datetime(2026, 7, 29, 10, 0, 0)
    assert scheduler_jobs._claim_autonomous_cycle("u1", "plot1", 10, now)
    assert not scheduler_jobs._claim_autonomous_cycle(
        "u1", "plot1", 10, now + timedelta(minutes=5))


def test_claim_cycle_allows_after_interval():
    now = datetime(2026, 7, 29, 10, 0, 0)
    assert scheduler_jobs._claim_autonomous_cycle("u1", "plot1", 10, now)
    assert scheduler_jobs._claim_autonomous_cycle(
        "u1", "plot1", 10, now + timedelta(minutes=10))


def test_claim_cycle_isolated_by_user_and_region():
    now = datetime(2026, 7, 29, 10, 0, 0)
    assert scheduler_jobs._claim_autonomous_cycle("u1", "plot1", 10, now)
    assert scheduler_jobs._claim_autonomous_cycle("u2", "plot1", 10, now)
    assert scheduler_jobs._claim_autonomous_cycle("u1", "plot2", 10, now)


def test_claim_cycle_database_lease_survives_process_cache_reset():
    now = datetime(2026, 7, 29, 10, 0, 0)
    assert scheduler_jobs._claim_autonomous_cycle("u1", "plot1", 10, now)
    scheduler_jobs._autonomous_last_runs.clear()
    assert not scheduler_jobs._claim_autonomous_cycle(
        "u1", "plot1", 10, now + timedelta(minutes=1)
    )


def test_active_users_are_discovered_from_database(monkeypatch):
    """数据库中有设备或地块的用户不能因缺少旧 JSON 文件而被漏掉。"""
    class FakeQuery:
        def outerjoin(self, *args, **kwargs):
            return self

        def filter(self, *args, **kwargs):
            return self

        def distinct(self):
            return self

        def all(self):
            return [("123",), ("default",)]

    class FakeSession:
        closed = False

        def query(self, *args):
            return FakeQuery()

        def close(self):
            self.closed = True

    fake_session = FakeSession()
    from core.database import engine
    monkeypatch.setattr(engine, "get_session", lambda: fake_session)

    usernames = scheduler_jobs._get_active_usernames()
    assert {"123", "default"} <= set(usernames)
    assert fake_session.closed is True
