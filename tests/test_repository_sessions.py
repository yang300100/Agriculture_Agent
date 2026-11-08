"""哥哥，仓储操作结束后要归还连接，也要保住返回记录和调用方事务。"""

from datetime import date

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from core.database.models import Base, ChatMessage, ChatSession, User
from core.database.repository import base
from core.database.repository.chat import ChatSessionRepository
from core.database.repository.devices import DeviceLogRepository, DeviceRuleRepository
from core.database.repository.finance import FinanceRepository
from core.database.repository.reminders import ReminderRepository
from core.database.repository.users import UserRepository
from core.reminder_system import ReminderStorage


@pytest.fixture
def repository_database(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'repository.db').as_posix()}",
        pool_size=1, max_overflow=0, pool_timeout=0.01,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    sessions = []

    def get_test_session():
        session = factory()
        sessions.append(session)
        return session

    monkeypatch.setattr(base, "get_session", get_test_session)
    with factory() as session:
        session.add_all([
            User(id=1, username="哥哥", password_hash=""),
            User(id=2, username="已有用户名", password_hash=""),
        ])
        session.commit()
    yield engine, factory
    for session in sessions:
        session.close()
    engine.dispose()


@pytest.mark.parametrize("method, arguments", [
    ("get_by_id", {"id": 1}),
    ("get_all", {}),
    ("find_by", {"username": "哥哥"}),
    ("find_one", {"username": "哥哥"}),
    ("count", {}),
])
def test_repeated_reads_do_not_exhaust_a_single_connection_pool(repository_database, method, arguments):
    engine, _ = repository_database
    repositories = [UserRepository() for _ in range(8)]

    results = [getattr(repo, method)(**arguments) for repo in repositories]

    assert len(results) == 8
    assert engine.pool.checkedout() == 0
    if method == "count":
        assert results == [2] * 8
    elif method in {"get_all", "find_by"}:
        assert all(rows[0].username == "哥哥" for rows in results)
    else:
        assert all(row.username == "哥哥" for row in results)


def test_write_results_remain_readable_after_returning_the_connection(repository_database):
    engine, _ = repository_database
    repo = UserRepository()

    created = repo.create(username="新用户", password_hash="")
    created_id = created.id
    assert created.username == "新用户"
    assert engine.pool.checkedout() == 0
    updated = repo.update(created_id, username="已更新")
    assert updated.id == created_id
    assert updated.username == "已更新"
    assert engine.pool.checkedout() == 0
    rows = repo.bulk_create([
        {"username": "批量甲", "password_hash": ""},
        {"username": "批量乙", "password_hash": ""},
    ])
    assert {row.username for row in rows} == {"批量甲", "批量乙"}
    assert all(row.id for row in rows)
    assert engine.pool.checkedout() == 0
    assert repo.delete(created_id) is True
    assert repo.get_by_id(created_id) is None
    assert engine.pool.checkedout() == 0


@pytest.mark.parametrize("external_session", [False, True], ids=["owned", "external"])
def test_failed_update_rolls_back_and_repository_can_be_reused(repository_database, external_session):
    engine, factory = repository_database
    with factory() as session:
        repo = UserRepository(session if external_session else None)

        with pytest.raises(IntegrityError):
            repo.update(1, username="已有用户名")

        assert repo.get_by_id(1).username == "哥哥"
        assert repo.update(1, username="恢复后更新").username == "恢复后更新"
        if not external_session:
            assert engine.pool.checkedout() == 0
    assert engine.pool.checkedout() == 0


def test_external_session_is_not_closed_or_committed_by_reads(repository_database):
    _, factory = repository_database
    with factory() as session:
        pending = User(username="待提交", password_hash="")
        session.add(pending)
        repo = UserRepository(session)

        assert repo.find_one(username="待提交") is pending
        assert inspect(pending).session is session
        assert session.in_transaction()
        session.rollback()

        assert repo.count() == 2
        assert repo.create(username="外部会话仍可写入", password_hash="").id
        assert repo.session is session


def test_chat_messages_are_available_after_the_read_connection_is_returned(repository_database):
    engine, factory = repository_database
    with factory() as session:
        chat = ChatSession(user_id=1, title="农场问答")
        chat.messages = [ChatMessage(role="user", content="查看麦田")]
        session.add(chat)
        session.commit()

    rows = ChatSessionRepository().find_by(user_id=1)

    assert engine.pool.checkedout() == 0
    assert [(message.role, message.content) for message in rows[0].messages] == [
        ("user", "查看麦田"),
    ]
    assert engine.pool.checkedout() == 0


@pytest.mark.parametrize("read", [
    lambda: FinanceRepository().get_by_date_range(1, date(2026, 1, 1), date(2026, 12, 31)),
    lambda: DeviceLogRepository().get_recent(1),
], ids=["finance_date_range", "device_recent_logs"])
def test_custom_repository_reads_return_their_connection(repository_database, read):
    engine, _ = repository_database

    assert read() == []

    assert engine.pool.checkedout() == 0


def test_rule_sync_keeps_existing_rows_attached_until_the_write_finishes(repository_database):
    engine, _ = repository_database
    repo = DeviceRuleRepository()
    original = repo.create(user_id=1, name="原规则", conditions="{}", actions="{}")

    rows = repo.sync_for_user(1, [
        {"id": original.id, "name": "更新规则", "conditions": "{}", "actions": "{}"},
        {"name": "新规则", "conditions": "{}", "actions": "{}"},
    ])

    assert [(row.id, row.name) for row in rows] == [
        (original.id, "更新规则"), (original.id + 1, "新规则"),
    ]
    assert all(row.created_at is not None for row in rows)
    assert engine.pool.checkedout() == 0
    assert [row.name for row in repo.find_by(user_id=1)] == ["更新规则", "新规则"]


def test_reminder_sync_persists_changes_to_existing_rows(repository_database):
    engine, _ = repository_database
    repo = ReminderRepository()
    original = repo.create(user_id=1, reminder_type="浇水", status="active")
    deleted = repo.create(user_id=1, reminder_type="施肥", status="active")
    storage = ReminderStorage(username="哥哥")

    storage.save_reminders([
        {"id": str(original.id), "reminder_type": "浇水", "status": "paused"},
        {"reminder_type": "巡田", "status": "active"},
    ])

    rows = repo.find_by(user_id=1)
    assert [(row.reminder_type, row.status) for row in rows] == [
        ("浇水", "paused"), ("巡田", "active"),
    ]
    assert rows[0].id == original.id
    assert rows[0].created_at == original.created_at
    assert repo.get_by_id(deleted.id) is None
    assert engine.pool.checkedout() == 0


def test_replace_all_failure_preserves_original_records_and_releases_connection(repository_database):
    engine, _ = repository_database
    repo = ReminderRepository()
    original = repo.create(user_id=1, reminder_type="浇水")

    with pytest.raises(IntegrityError):
        repo.replace_all_for_user(1, [{"reminder_type": None}])

    rows = repo.find_by(user_id=1)
    assert [(row.id, row.reminder_type) for row in rows] == [(original.id, "浇水")]
    assert engine.pool.checkedout() == 0
