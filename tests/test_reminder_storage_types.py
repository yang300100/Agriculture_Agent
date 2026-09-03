import datetime
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.database.models import Base, Reminder as ReminderModel, User
from core.database.repository.base import BaseRepository
from core.database.repository.reminders import ReminderRepository
from core.reminder_system import ReminderStorage


def test_datetime_columns_accept_reminder_strings():
    repo = BaseRepository(ReminderModel)
    values = repo._coerce_dates({"next_trigger": "2026-08-02 09:00"})

    assert values["next_trigger"] == datetime.datetime(2026, 8, 2, 9, 0)


def test_reminder_channels_are_serialized_before_database_write():
    storage = object.__new__(ReminderStorage)
    storage._user = SimpleNamespace(id=7)

    with patch.object(
        ReminderRepository,
        "replace_all_for_user",
    ) as replace_all:
        storage.save_reminders(
            [
                {
                    "crop": "玉米",
                    "reminder_type": "浇水",
                    "channels": ["app", "sms"],
                    "next_trigger": "2026-08-02 09:00",
                }
            ]
        )

    user_id, items = replace_all.call_args.args
    assert user_id == 7
    assert items[0]["channels"] == '["app", "sms"]'
    assert items[0]["next_trigger"] == "2026-08-02 09:00"


def test_reminder_repository_writes_datetime_to_sqlite():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        user = User(username="reminder_test", password_hash="")
        session.add(user)
        session.commit()
        session.refresh(user)

        repo = ReminderRepository(session)
        repo.replace_all_for_user(
            user.id,
            [
                {
                    "crop": "玉米",
                    "reminder_type": "浇水",
                    "channels": '["app"]',
                    "next_trigger": "2026-08-02 09:00",
                }
            ],
        )

        saved = repo.find_one(user_id=user.id)
        assert saved is not None
        assert saved.next_trigger == datetime.datetime(2026, 8, 2, 9, 0)
