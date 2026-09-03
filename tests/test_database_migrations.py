"""数据库版本迁移与分区模型测试。"""

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from core.database.migrations import apply_migrations


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
    assert versions == [1, 2]


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
