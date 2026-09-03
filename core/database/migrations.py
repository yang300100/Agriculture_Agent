"""轻量、可重复执行的数据库结构迁移。"""

import logging

from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)

LATEST_SCHEMA_VERSION = 2


def apply_migrations(engine) -> None:
    """按版本升级已有数据库；新数据库执行后也会登记当前版本。"""
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, "
            "applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        ))

    applied = set()
    with engine.connect() as connection:
        applied = {
            int(row[0])
            for row in connection.execute(text("SELECT version FROM schema_migrations"))
        }

    if 1 not in applied:
        _migration_001_device_zones(engine)
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO schema_migrations(version) VALUES (:version)"),
                {"version": 1},
            )

    if 2 not in applied:
        _migration_002_field_zone_uniqueness(engine)
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO schema_migrations(version) VALUES (:version)"),
                {"version": 2},
            )


def _migration_001_device_zones(engine) -> None:
    """为旧设备配置补作业分区字段。"""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "device_configs" not in tables:
        return
    columns = {item["name"] for item in inspector.get_columns("device_configs")}
    if "zone_id" not in columns:
        with engine.begin() as connection:
            connection.execute(text(
                "ALTER TABLE device_configs ADD COLUMN zone_id VARCHAR(100)"
            ))
        logger.info("数据库迁移 001：device_configs.zone_id 已添加")


def _migration_002_field_zone_uniqueness(engine) -> None:
    """从数据库层阻止同一用户、地块下出现重复分区 ID。"""
    inspector = inspect(engine)
    if "field_zones" not in set(inspector.get_table_names()):
        return

    target_columns = {"user_id", "field_id", "zone_id"}
    constraints = inspector.get_unique_constraints("field_zones")
    indexes = inspector.get_indexes("field_zones")
    already_unique = any(
        set(item.get("column_names") or []) == target_columns
        for item in [*constraints, *indexes]
        if item.get("unique", item in constraints)
    )
    if already_unique:
        return

    with engine.begin() as connection:
        connection.execute(text(
            "CREATE UNIQUE INDEX uq_field_zone_user_field_zone "
            "ON field_zones (user_id, field_id, zone_id)"
        ))
    logger.info("数据库迁移 002：地块分区唯一约束已添加")
