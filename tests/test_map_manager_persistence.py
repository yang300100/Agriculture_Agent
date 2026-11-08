"""帮哥哥守住地图地块的身份、种植历史和用户边界。"""

import json
from datetime import datetime
from uuid import uuid4

import pytest

from core.database.repository.fields import FieldRepository
from core.database.repository.users import UserRepository
from core.map_manager import MapManager


COORDINATES = [[116.3, 39.9], [116.31, 39.9], [116.3, 39.91]]
HISTORY = [{"crop": "小麦", "season": "2025", "yield_amount": 1200}]


@pytest.fixture
def stored_field():
    username = f"map_persistence_{uuid4().hex}"
    manager = MapManager(username)
    user = UserRepository().get_by_username(username)
    row = FieldRepository().create(
        user_id=user.id,
        name="北地",
        coordinates=json.dumps(COORDINATES),
        current_crop="玉米",
        planting_history=json.dumps(HISTORY, ensure_ascii=False),
        created_at=datetime(2025, 1, 2, 3, 4, 5, 123456),
        updated_at=datetime(2025, 6, 7, 8, 9, 10, 654321),
    )
    return username, row.id


def test_load_preserves_database_id_history_and_timestamps(stored_field):
    username, field_id = stored_field

    field = MapManager(username).get_all_fields()[0]

    assert field.id == str(field_id)
    assert field.history == HISTORY
    assert datetime.fromisoformat(field.created_at) == datetime(2025, 1, 2, 3, 4, 5, 123456)
    assert datetime.fromisoformat(field.updated_at) == datetime(2025, 6, 7, 8, 9, 10, 654321)


def test_update_preserves_field_identity_history_and_related_plan(stored_field):
    from core.database.models import PlantingPlan
    from core.database.repository.base import BaseRepository

    username, field_id = stored_field
    manager = MapManager(username)
    user = UserRepository().get_by_username(username)
    plans = BaseRepository(PlantingPlan)
    plan = plans.create(user_id=user.id, crop="玉米", plot_id=field_id)
    other = MapManager(f"map_other_{uuid4().hex}")
    other.create_field("他人的地块", COORDINATES)

    manager.update_field(manager.fields[0].id, current_crop="大豆")

    row = FieldRepository().find_one(user_id=user.id, id=field_id)
    assert row is not None
    assert row.current_crop == "大豆"
    assert json.loads(row.planting_history) == HISTORY
    assert row.created_at == datetime(2025, 1, 2, 3, 4, 5, 123456)
    assert plans.get_by_id(plan.id).plot_id == field_id
    assert MapManager(other._username).get_all_fields()[0].name == "他人的地块"


def test_new_field_id_and_appended_history_survive_reload():
    username = f"map_history_{uuid4().hex}"
    manager = MapManager(username)
    field = manager.create_field("南地", COORDINATES)
    field_id = field.id

    assert manager.add_planting_history(field_id, "小麦", season="2026", yield_amount=1200)
    reloaded = MapManager(username).get_field(field_id)

    assert reloaded is not None
    assert reloaded.history[0]["crop"] == "小麦"
    assert reloaded.history[0]["yield_amount"] == 1200


def test_update_does_not_delete_fields_added_since_manager_loaded(stored_field):
    username, _ = stored_field
    stale_manager = MapManager(username)
    new_field = MapManager(username).create_field("刚新增的地块", COORDINATES)

    stale_manager.update_field(stale_manager.fields[0].id, soil_type="壤土")

    assert MapManager(username).get_field(new_field.id) is not None


def test_create_does_not_overwrite_another_managers_changes(stored_field):
    username, field_id = stored_field
    stale_manager = MapManager(username)
    current_manager = MapManager(username)
    current_manager.update_field(current_manager.fields[0].id, current_crop="大豆")

    stale_manager.create_field("东地", COORDINATES)

    row = FieldRepository().get_by_id(field_id)
    assert row.current_crop == "大豆"


def test_delete_preserves_unrelated_fields_added_later(stored_field):
    username, _ = stored_field
    stale_manager = MapManager(username)
    new_field = MapManager(username).create_field("保留的地块", COORDINATES)

    assert stale_manager.delete_field(stale_manager.fields[0].id)

    remaining = MapManager(username).get_all_fields()
    assert [(field.id, field.name) for field in remaining] == [(new_field.id, "保留的地块")]
