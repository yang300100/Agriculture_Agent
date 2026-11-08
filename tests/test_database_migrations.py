"""数据库版本迁移与分区模型测试。"""

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from core.database.migrations import LATEST_SCHEMA_VERSION, apply_migrations


def test_旧设备表会补充zone_id且迁移可重复执行(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE device_configs (id INTEGER PRIMARY KEY, device_id VARCHAR(100))"
        ))

    apply_migrations(engine)
    apply_migrations(engine)

    columns = {item["name"] for item in inspect(engine).get_columns("device_configs")}
    with engine.connect() as connection:
        versions = connection.execute(
            text("SELECT version FROM schema_migrations ORDER BY version")
        ).scalars().all()
    assert "zone_id" in columns
    assert versions == list(range(1, LATEST_SCHEMA_VERSION + 1))


def test_旧种植数据迁移保留记录并补齐备注和任务进度(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'planting-legacy.db'}")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE planting_plans (id INTEGER PRIMARY KEY, crop VARCHAR(50))"
        ))
        connection.execute(text(
            "CREATE TABLE planting_tasks (id INTEGER PRIMARY KEY, title VARCHAR(200))"
        ))
        connection.execute(text("INSERT INTO planting_plans VALUES (7, '小麦')"))
        connection.execute(text("INSERT INTO planting_tasks VALUES (9, '巡田')"))
    apply_migrations(engine)
    apply_migrations(engine)
    with engine.connect() as connection:
        assert connection.execute(text(
            "SELECT id, crop, notes FROM planting_plans"
        )).one() == (7, "小麦", "")
        assert connection.execute(text(
            "SELECT id, title, progress_percent FROM planting_tasks"
        )).one() == (9, "巡田", 0)
    engine.dispose()


def test_旧分区表会增加数据库唯一约束(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'zones.db'}")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE field_zones ("
            "id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, "
            "field_id INTEGER NOT NULL, zone_id VARCHAR(100) NOT NULL)"
        ))

    apply_migrations(engine)
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO field_zones(user_id, field_id, zone_id) "
            "VALUES (1, 2, 'north')"
        ))
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO field_zones(user_id, field_id, zone_id) "
                "VALUES (1, 2, 'north')"
            ))


def test_get_session会自动初始化空数据库(tmp_path, monkeypatch):
    import importlib
    import core.database.engine as db_engine

    original_url = db_engine.DB_URL
    db_engine._engine.dispose()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'session.db'}")
    importlib.reload(db_engine)
    try:
        session = db_engine.get_session()
        session.close()
        assert "users" in inspect(db_engine._engine).get_table_names()
        assert "schema_migrations" in inspect(db_engine._engine).get_table_names()
    finally:
        db_engine._engine.dispose()
        monkeypatch.setenv("DATABASE_URL", original_url)
        importlib.reload(db_engine)
