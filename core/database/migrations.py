"""轻量、可重复执行的数据库结构迁移。"""

import logging

from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)

LATEST_SCHEMA_VERSION = 7


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

    if 3 not in applied:
        _migration_003_planting_persistence(engine)
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO schema_migrations(version) VALUES (:version)"),
                {"version": 3},
            )

    if 4 not in applied:
        _migration_004_profile_autonomy(engine)
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO schema_migrations(version) VALUES (:version)"),
                {"version": 4},
            )

    if 5 not in applied:
        _migration_005_task_execution(engine)
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO schema_migrations(version) VALUES (:version)"),
                {"version": 5},
            )

    if 6 not in applied:
        _migration_006_reminder_links(engine)
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO schema_migrations(version) VALUES (:version)"),
                {"version": 6},
            )

    if 7 not in applied:
        inspector = inspect(engine)
        if "planting_plans" in inspector.get_table_names() and "plot_id" not in {c["name"] for c in inspector.get_columns("planting_plans")}:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE planting_plans ADD COLUMN plot_id INTEGER"))
        from core.database.models import AutonomyJob, AutonomyAlert, SchedulerHeartbeat
        for table in (AutonomyJob.__table__, AutonomyAlert.__table__, SchedulerHeartbeat.__table__):
            table.create(bind=engine, checkfirst=True)
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO schema_migrations(version) VALUES (:version)"), {"version": 7})


def _migration_006_reminder_links(engine) -> None:
    """哥哥，旧提醒不猜测关联，只为后续明确绑定留出可空字段。"""
    inspector = inspect(engine)
    if "reminders" not in set(inspector.get_table_names()):
        return
    columns = {item["name"] for item in inspector.get_columns("reminders")}
    for column, definition in (("start_date", "DATE"), ("task_id", "INTEGER"), ("progress_id", "INTEGER")):
        if column not in columns:
            with engine.begin() as connection:
                connection.execute(text(f"ALTER TABLE reminders ADD COLUMN {column} {definition}"))


def _migration_005_task_execution(engine) -> None:
    """哥哥，为旧任务补可恢复的设备运行记录，不改变原任务内容。"""
    inspector = inspect(engine)
    if "planting_tasks" not in set(inspector.get_table_names()):
        return
    columns = {item["name"] for item in inspector.get_columns("planting_tasks")}
    if "execution_state" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE planting_tasks ADD COLUMN execution_state TEXT"))


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


def _migration_004_profile_autonomy(engine) -> None:
    """哥哥，旧档案保留空值，继续使用部署环境的默认自主权。"""
    inspector = inspect(engine)
    if "user_profiles" not in set(inspector.get_table_names()):
        return
    columns = {item["name"] for item in inspector.get_columns("user_profiles")}
    if "autonomy_level" not in columns:
        with engine.begin() as connection:
            connection.execute(text(
                "ALTER TABLE user_profiles ADD COLUMN autonomy_level VARCHAR(10)"
            ))


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


def _migration_003_planting_persistence(engine) -> None:
    """哥哥，补齐备注和任务进度，保留旧记录及其 ID。"""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    additions = [
        ("planting_plans", "notes", "TEXT DEFAULT ''"),
        ("planting_tasks", "progress_percent", "INTEGER DEFAULT 0"),
    ]
    for table, column, definition in additions:
        if table not in tables:
            continue
        columns = {item["name"] for item in inspector.get_columns(table)}
        if column not in columns:
            with engine.begin() as connection:
                connection.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                ))
