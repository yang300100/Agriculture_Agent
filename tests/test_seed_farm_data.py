"""帮哥哥验证农场初始化只增不改，且完整保留现有资料。"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from core.database.models import (
    Base, DeviceConfig, DeviceRule, Field, FieldZone, FinanceRecord,
    PlantingPlan, PlantingTask, Reminder, User, UserProfile,
)


AS_OF = date(2026, 11, 5)
COUNTS = {
    "fields": 3, "zones": 6, "plans": 3, "tasks": 8, "finance": 24,
    "devices": 8, "reminders": 3, "rules": 2, "profiles": 1,
}


@pytest.fixture
def farm_session(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'farm.db').as_posix()}")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([
            User(username="123", password_hash="unchanged-hash"),
            User(username="other", password_hash="other-hash"),
        ])
        session.commit()
        yield session
    engine.dispose()


def snapshot(session):
    """完整比较实际记录，避免哥哥的旧资料被悄悄改掉。"""
    return {
        table.name: list(session.execute(table.select().order_by(table.c.id)).tuples())
        for table in Base.metadata.sorted_tables
    }


def test_seed_builds_consistent_farm_relationships_amounts_and_dates(farm_session):
    from core.map_manager import MapManager
    from scripts.seed_farm_data import seed_farm_data

    result = seed_farm_data(farm_session, "123", as_of=AS_OF)

    assert result["username"] == "123"
    assert result["added"] == COUNTS
    assert {key: len(ids) for key, ids in result["created_ids"].items()} == COUNTS
    assert all(not ids for ids in result["existing_ids"].values())
    fields = farm_session.query(Field).all()
    assert {field.name for field in fields} == {"东侧麦田", "北区番茄温室", "油菜试验田"}
    target_area = {"东侧麦田": 5, "北区番茄温室": 2, "油菜试验田": 6}
    boxes = []
    for field in fields:
        assert field.area_mu == pytest.approx(target_area[field.name], abs=0.02)
        assert field.area_m2 == pytest.approx(field.area_mu * 666.67, rel=1e-6)
        coords = json.loads(field.coordinates)
        measured_m2, measured_mu = MapManager.calculate_area(coords)
        assert measured_mu == pytest.approx(target_area[field.name], abs=0.001)
        assert measured_m2 == pytest.approx(field.area_m2, rel=1e-6)
        assert all(114 < lon < 115 and 37 < lat < 39 for lon, lat in coords)
        boxes.append((min(x for x, _ in coords), min(y for _, y in coords),
                      max(x for x, _ in coords), max(y for _, y in coords)))
        assert farm_session.query(FieldZone).filter_by(field_id=field.id).count() == 2
    for index, (left, bottom, right, top) in enumerate(boxes):
        for x1, y1, x2, y2 in boxes[index + 1:]:
            assert right <= x1 or x2 <= left or top <= y1 or y2 <= bottom
    field_ids = {field.id for field in fields}
    plans = farm_session.query(PlantingPlan).all()
    assert {plan.plot_id for plan in plans} == field_ids
    assert {plan.crop: (AS_OF - plan.start_date).days for plan in plans} == {
        "小麦": 35, "番茄": 75, "油菜": 30,
    }
    assert all(plan.status == "进行中" and plan.expected_end_date > AS_OF for plan in plans)
    tasks = farm_session.query(PlantingTask).all()
    assert {task.status for task in tasks} == {"待办", "进行中", "已完成"}
    assert all(task.plan_id in {plan.id for plan in plans} for task in tasks)
    assert all(task.device_id is None for task in tasks)
    assert all(task.progress_percent == 100 and task.completed_date <= AS_OF
               for task in tasks if task.status == "已完成")
    records = farm_session.query(FinanceRecord).all()
    months = Counter((row.date.year, row.date.month, row.record_type) for row in records)
    assert len({row.date.strftime("%Y-%m") for row in records}) == 6
    assert min(row.date for row in records) == date(2026, 6, 1)
    assert max(row.date for row in records) == date(2026, 11, 1)
    assert set(months.values()) == {1, 3}
    assert all(count == (3 if kind == "cost" else 1) for (_, _, kind), count in months.items())
    assert all(row.date <= AS_OF and row.quantity >= 0 and row.unit_price >= 0
               and row.total_amount == round(row.quantity * row.unit_price, 2) for row in records)
    devices = farm_session.query(DeviceConfig).all()
    device_ids = {device.device_id for device in devices}
    assert len(device_ids) == 8
    assert sum(json.loads(device.capabilities) == ["read_sensor"] for device in devices) == 3
    for device in devices:
        assert device.driver == "simulator"
        assert json.loads(device.connection) == {}
        assert "capture" not in json.loads(device.capabilities)
        assert device.plot_id in field_ids
        assert farm_session.query(FieldZone).filter_by(
            field_id=device.plot_id, zone_id=device.zone_id,
        ).one()
        assert json.loads(device.initial_state)["status"] == "standby"
    for reminder in farm_session.query(Reminder).all():
        assert reminder.status == "paused" and json.loads(reminder.channels) == []
    for rule in farm_session.query(DeviceRule).all():
        assert rule.enabled == 0
        assert json.loads(rule.actions)["device_id"] in device_ids
    profile = farm_session.query(UserProfile).one()
    assert profile.farm_size == 13
    assert profile.experience_level == "中级（3-5年）"
    assert json.loads(profile.goals) == ["高产", "节水", "经济效益"]
    for model in (Field, FieldZone, PlantingPlan, PlantingTask, FinanceRecord, DeviceConfig, Reminder, DeviceRule):
        for row in farm_session.query(model).all():
            strings = [getattr(row, column.name) for column in row.__table__.columns]
            assert all("模拟" not in value and "demo" not in value.lower()
                       for value in strings if isinstance(value, str))
    assert farm_session.query(User).filter_by(username="123").one().password_hash == "unchanged-hash"


def test_seed_is_idempotent_and_preserves_existing_and_other_users_data(farm_session):
    from scripts.seed_farm_data import seed_farm_data

    own_user = farm_session.query(User).filter_by(username="123").one()
    other_user = farm_session.query(User).filter_by(username="other").one()
    existing = Field(user_id=own_user.id, name="hebei", coordinates="[]", current_crop="原有作物")
    other = Field(user_id=other_user.id, name="东侧麦田", coordinates="[]", current_crop="他人的作物")
    profile = UserProfile(user_id=own_user.id, region="原有地区", farm_size=99, goals='["原有目标"]')
    device = DeviceConfig(user_id=own_user.id, device_id="irrigation_pump_01", name="原有水泵",
                          driver="http", connection='{"base_url":"http://127.0.0.1:5000"}')
    farm_session.add_all([existing, other, profile, device])
    farm_session.commit()
    original = snapshot(farm_session)

    first = seed_farm_data(farm_session, "123", as_of=AS_OF)
    farm_session.commit()
    after_first = snapshot(farm_session)
    second = seed_farm_data(farm_session, "123", as_of=AS_OF)
    farm_session.commit()

    assert first["added"]["profiles"] == 0
    assert all(count == 0 for count in second["added"].values())
    assert snapshot(farm_session) == after_first
    for table_name, rows in original.items():
        assert all(row in after_first[table_name] for row in rows)
    assert second["existing_ids"]["fields"] == first["created_ids"]["fields"]


def test_same_name_and_device_id_are_skipped_without_overwriting(farm_session):
    from scripts.seed_farm_data import seed_farm_data

    user = farm_session.query(User).filter_by(username="123").one()
    field = Field(user_id=user.id, name="东侧麦田", coordinates="[]", area_mu=99, current_crop="大豆")
    device = DeviceConfig(user_id=user.id, device_id="wheat_soil_sensor", name="已有外部传感器",
                          driver="modbus", connection='{"host":"192.168.1.2"}')
    farm_session.add_all([field, device])
    farm_session.commit()

    result = seed_farm_data(farm_session, "123", as_of=AS_OF)
    farm_session.expire_all()

    assert result["added"]["fields"] == 2
    assert result["added"]["devices"] == 7
    assert farm_session.get(Field, field.id).area_mu == 99
    assert farm_session.get(Field, field.id).current_crop == "大豆"
    assert farm_session.get(DeviceConfig, device.id).driver == "modbus"
    for rule in farm_session.query(DeviceRule).all():
        assert "wheat_soil_sensor" not in rule.conditions
        assert "wheat_soil_sensor" not in rule.actions


def test_missing_account_creates_nothing(farm_session):
    from scripts.seed_farm_data import seed_farm_data

    original = snapshot(farm_session)

    with pytest.raises(ValueError, match="账号不存在"):
        seed_farm_data(farm_session, "missing", as_of=AS_OF)

    assert snapshot(farm_session) == original


def test_seed_leaves_transaction_commit_to_caller(farm_session):
    from scripts.seed_farm_data import seed_farm_data

    seed_farm_data(farm_session, "123", as_of=AS_OF)
    farm_session.rollback()

    assert farm_session.query(Field).count() == 0
    assert farm_session.query(DeviceConfig).count() == 0


def test_new_profile_includes_existing_field_area(farm_session):
    from scripts.seed_farm_data import seed_farm_data

    user = farm_session.query(User).filter_by(username="123").one()
    farm_session.add(Field(user_id=user.id, name="hebei", coordinates="[]", area_mu=555))
    farm_session.commit()

    seed_farm_data(farm_session, "123", as_of=AS_OF)

    assert farm_session.query(UserProfile).filter_by(user_id=user.id).one().farm_size == 568


def test_sqlite_backup_includes_wal_records_and_restores_original_database(tmp_path):
    from scripts.seed_farm_data import backup_sqlite_database

    database = tmp_path / "farm.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE originals(id INTEGER PRIMARY KEY, note TEXT)")
        connection.execute("INSERT INTO originals VALUES (7, '哥哥的原记录')")
        connection.commit()

        backup = backup_sqlite_database(database, tmp_path / "backups")
        assert backup.is_file()
        connection.execute("DELETE FROM originals")
        connection.commit()
        with sqlite3.connect(backup) as restored:
            assert restored.execute("SELECT * FROM originals").fetchall() == [(7, "哥哥的原记录")]
            assert restored.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            restored.backup(connection)
        assert connection.execute("SELECT * FROM originals").fetchall() == [(7, "哥哥的原记录")]


def test_cli_uses_dotenv_database_and_backs_up_before_migrations(tmp_path):
    """在临时项目副本运行CLI，哥哥的项目.env与真实数据库都不会被碰到。"""
    project = Path(__file__).resolve().parents[1]
    isolated_project = tmp_path / "project"
    (isolated_project / "core").mkdir(parents=True)
    (isolated_project / "scripts").mkdir()
    shutil.copytree(project / "core" / "database", isolated_project / "core" / "database",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(project / "core" / "storage_paths.py", isolated_project / "core" / "storage_paths.py")
    script = isolated_project / "scripts" / "seed_farm_data.py"
    shutil.copy2(project / "scripts" / "seed_farm_data.py", script)
    database = tmp_path / "configured.db"
    data_dir = tmp_path / "configured-data"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, "
                           "password_hash TEXT NOT NULL, created_at DATETIME)")
        connection.execute("INSERT INTO users VALUES (7, '123', 'original-hash', '2025-01-01')")
    (isolated_project / ".env").write_text(
        f"DATABASE_URL=sqlite:///{database.as_posix()}\nDATA_STORAGE_DIR={data_dir.as_posix()}\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    for key in ("PYTHON_DOTENV_DISABLED", "DATABASE_URL", "DATA_STORAGE_DIR"):
        environment.pop(key, None)
    environment["PYTHONIOENCODING"] = "utf-8"

    command = [sys.executable, str(script), "--username", "123", "--as-of", "2026-11-05"]
    completed = subprocess.run(command, cwd=isolated_project, env=environment,
                               capture_output=True, text=True, encoding="utf-8", timeout=30)

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["added"] == COUNTS
    manifest = json.loads((data_dir / "seeded_farms" / "123.json").read_text(encoding="utf-8"))
    assert Path(manifest["database"]) == database
    with sqlite3.connect(manifest["backup"]) as backup:
        assert backup.execute("SELECT password_hash FROM users WHERE username='123'").fetchone() == ("original-hash",)
        assert backup.execute("SELECT name FROM sqlite_master WHERE name='schema_migrations'").fetchone() is None
        assert backup.execute("SELECT name FROM sqlite_master WHERE name='fields'").fetchone() is None
    with sqlite3.connect(database) as seeded:
        assert seeded.execute("SELECT COUNT(*) FROM fields").fetchone() == (3,)
        from core.database.migrations import LATEST_SCHEMA_VERSION
        assert seeded.execute("SELECT MAX(version) FROM schema_migrations").fetchone() == (LATEST_SCHEMA_VERSION,)
    assert not (isolated_project / "data" / "agriculture.db").exists()
