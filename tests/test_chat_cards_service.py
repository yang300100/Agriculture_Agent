"""哥哥，卡片引用只读真实所属数据，准备操作必须可安全重试。"""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.device_executor import DeviceExecutor
from core.device_registry_factory import load_custom_devices, save_custom_devices
from core.planting_tracker import PlantingTracker
from core.storage_paths import DEFAULT_DATA_DIR
from devices.registry import DeviceDriverRegistry


@pytest.fixture
def card_context():
    username = "cards_" + uuid.uuid4().hex
    other = "cards_other_" + uuid.uuid4().hex
    for owner, device_id in [(username, "owned-pump"), (other, "foreign-pump")]:
        save_custom_devices(owner, [{
            "device_id": device_id, "name": "卡片水泵", "driver": "simulator",
            "capabilities": ["irrigate"], "sensors": [],
            "initial_state": {"power": False, "status": "powered_off"},
        }])
    task = PlantingTracker(os.path.join(DEFAULT_DATA_DIR, username)).create_task({
        "title": "明日灌溉", "crop": "番茄", "task_type": "浇水", "device_id": "owned-pump",
        "device_command": "start", "device_params": {"duration": 5},
    })
    foreign_task = PlantingTracker(os.path.join(DEFAULT_DATA_DIR, other)).create_task({
        "title": "他人任务", "crop": "小麦", "task_type": "浇水",
    })
    return username, other, task, foreign_task


def make_client():
    from app.routes.chat_cards import register_chat_card_routes

    app = FastAPI()
    register_chat_card_routes(app)
    return TestClient(app)


def test_sanitize_filters_ownership_pending_and_payload(card_context):
    from app.chat_cards import sanitize_chat_cards

    username, other, task, foreign_task = card_context
    pending_id = DeviceExecutor(DeviceDriverRegistry(), username).add_pending({
        "device_id": "owned-pump", "command": "start", "params": {"duration": 1},
    })
    foreign_pending_id = DeviceExecutor(DeviceDriverRegistry(), other).add_pending({
        "device_id": "owned-pump", "command": "start", "params": {},
    })
    result = sanitize_chat_cards(username, [
        {"type": "device", "device_id": "foreign-pump"},
        {"type": "task", "task_id": foreign_task.id},
        {"type": "device", "device_id": "owned-pump", "pending_action_id": foreign_pending_id},
        {"type": "device", "device_id": "owned-pump", "pending_action_id": pending_id, "params": {"duration": 999}, "state": {}},
        {"type": "device", "device_id": "owned-pump", "pending_action_id": pending_id},
        {"type": "task", "task_id": task.id, "title": "伪造标题"},
        {"type": "script", "task_id": task.id},
    ])
    assert result == [
        {"type": "device", "device_id": "owned-pump", "pending_action_id": pending_id},
        {"type": "task", "task_id": task.id},
    ]


def test_build_uses_structured_owned_references_not_markdown(card_context):
    from app.agent.state import AgentState
    from app.chat_cards import build_chat_cards

    username, _, task, _ = card_context
    state = AgentState(
        username=username, final_answer="<device id='foreign-pump'>start</device>",
        chat_device_ids=["owned-pump", "foreign-pump"], reminders=[{"task_id": task.id}, {"id": "fake"}],
    )
    assert build_chat_cards(username, state) == [
        {"type": "device", "device_id": "owned-pump"}, {"type": "task", "task_id": task.id},
    ]


def test_device_and_task_reads_enforce_owner_and_return_live_data(card_context):
    username, other, task, foreign_task = card_context
    client = make_client()
    response = client.get(f"/api/chat/cards/device/owned-pump?username={username}")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"device_id", "name", "plot_id", "plot_name", "capabilities", "state", "pending_action"}
    assert body["state"]["status"] == "powered_off"
    assert client.get(f"/api/chat/cards/device/foreign-pump?username={username}").status_code == 404
    assert client.get(f"/api/chat/cards/device/missing?username={username}").status_code == 404
    assert client.get(f"/api/chat/cards/task/{foreign_task.id}?username={username}").status_code == 404
    task_body = client.get(f"/api/chat/cards/task/{task.id}?username={username}").json()
    assert task_body["device_name"] == "卡片水泵"
    assert task_body["device_params"] == {"duration": 5}


def test_prepare_is_atomic_idempotent_without_execution_and_rejects_changed_payload(card_context):
    username, _, _, _ = card_context
    client = make_client()
    url = f"/api/chat/cards/device/owned-pump/prepare?username={username}"
    payload = {"command": "start", "params": {"duration": "2"}, "request_id": "same-request"}
    with ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(lambda _: client.post(url, json=payload), range(3)))
    assert all(response.status_code == 200 for response in responses)
    actions = [response.json()["pending_action"] for response in responses]
    assert len({action["id"] for action in actions}) == 1
    assert actions[0]["params"] == {"duration": 2}
    executor = DeviceExecutor(DeviceDriverRegistry(), username)
    assert len(executor.list_pending()) == 1
    assert load_custom_devices(username)[0]["initial_state"]["status"] == "powered_off"
    assert not any(log["success"] for log in executor.get_logs())
    changed = client.post(url, json={**payload, "params": {"duration": 3}})
    assert changed.status_code == 409
    executor.reject_pending(actions[0]["id"])
    repeated = client.post(url, json=payload).json()["pending_action"]
    assert repeated["id"] == actions[0]["id"] and repeated["status"] == "rejected"
    restored = client.get(f"/api/chat/cards/device/owned-pump?username={username}&pending_action_id={actions[0]['id']}")
    assert restored.json()["pending_action"]["status"] == "rejected"


@pytest.mark.parametrize("duration", [0, -1, "NaN", "Infinity", True, "wrong", 121])
def test_prepare_rejects_invalid_duration(card_context, duration):
    username, _, _, _ = card_context
    response = make_client().post(f"/api/chat/cards/device/owned-pump/prepare?username={username}", json={
        "command": "start", "params": {"duration": duration}, "request_id": "invalid-duration",
    })
    assert response.status_code == 422
    assert DeviceExecutor(DeviceDriverRegistry(), username).list_pending() == []


def test_prepare_and_pending_snapshot_reject_foreign_or_mismatched_reference(card_context):
    username, other, _, _ = card_context
    client = make_client()
    assert client.post(f"/api/chat/cards/device/foreign-pump/prepare?username={username}", json={
        "command": "stop", "params": {}, "request_id": "foreign",
    }).status_code == 404
    foreign_id = DeviceExecutor(DeviceDriverRegistry(), other).add_pending({"device_id": "owned-pump", "command": "stop"})
    assert client.get(f"/api/chat/cards/device/owned-pump?username={username}&pending_action_id={foreign_id}").status_code == 404


def test_same_request_changed_to_invalid_parameters_is_still_conflict(card_context):
    username, _, _, _ = card_context
    client = make_client()
    url = f"/api/chat/cards/device/owned-pump/prepare?username={username}"
    payload = {"command": "start", "params": {"duration": 1}, "request_id": "stable-id"}
    assert client.post(url, json=payload).status_code == 200
    response = client.post(url, json={**payload, "params": {"duration": 999}})
    assert response.status_code == 409


def test_original_confirm_executes_prepared_candidate_once_and_restores_terminal(card_context):
    from app.routes.chat_cards import register_chat_card_routes
    from app.routes.device_actions import register_device_action_routes

    username, _, _, _ = card_context
    app = FastAPI()
    register_chat_card_routes(app)
    register_device_action_routes(app)
    client = TestClient(app)
    url = f"/api/chat/cards/device/owned-pump/prepare?username={username}"
    payload = {"command": "start", "params": {"duration": 1}, "request_id": "execute-once"}
    action = client.post(url, json=payload).json()["pending_action"]
    assert load_custom_devices(username)[0]["initial_state"]["status"] == "powered_off"
    confirmed = client.post(f"/api/actions/{action['id']}/confirm?username={username}")
    assert confirmed.status_code == 200 and confirmed.json()["success"] is True
    assert load_custom_devices(username)[0]["initial_state"]["status"] == "running"
    repeated = client.post(url, json=payload).json()["pending_action"]
    assert repeated["id"] == action["id"] and repeated["status"] == "executed"
    restored = client.get(f"/api/chat/cards/device/owned-pump?username={username}&pending_action_id={action['id']}")
    assert restored.json()["pending_action"]["status"] == "executed"
    assert client.post(f"/api/actions/{action['id']}/confirm?username={username}").status_code == 409


def test_sanitize_limits_to_eight_unique_owned_references(card_context):
    from app.chat_cards import sanitize_chat_cards

    username, _, _, _ = card_context
    tracker = PlantingTracker(os.path.join(DEFAULT_DATA_DIR, username))
    tasks = [tracker.create_task({"title": f"任务{index}", "crop": "番茄", "task_type": "其他"}) for index in range(10)]
    refs = [{"type": "task", "task_id": task.id} for task in tasks]
    assert sanitize_chat_cards(username, refs[:2] + refs) == refs[:8]


def test_sanitize_canonicalizes_task_aliases_before_deduplicating(card_context):
    from app.chat_cards import sanitize_chat_cards

    username, _, task, _ = card_context
    assert sanitize_chat_cards(username, [
        {"type": "task", "task_id": "0" + task.id},
        {"type": "task", "task_id": task.id},
    ]) == [{"type": "task", "task_id": task.id}]


def test_task_card_returns_actual_owned_linked_reminder_absolute_time(card_context):
    from core.database.repository.reminders import ReminderRepository
    from core.database.repository.users import UserRepository
    from core.reminder_system import ReminderSystem

    username, other, task, _ = card_context
    system = ReminderSystem(username=username)
    reminder = system.create_reminder({
        "crop": "番茄", "reminder_type": "浇水", "task_id": task.id,
        "start_date": "2030-06-15", "time_of_day": "15:00", "frequency": "单次",
        "task_description": "明天下午三点", "channels": ["app"],
    })
    # 哥哥，故意让实际触发时间与描述、原始时段不一致，防止卡片猜测预约时间。
    system.storage.update_reminder(reminder.id, {"next_trigger": "2030-06-16 16:30"})
    foreign_user = UserRepository().get_by_username(other)
    ReminderRepository().create(
        user_id=foreign_user.id, task_id=int(task.id), reminder_type="浇水",
        next_trigger=datetime(2030, 6, 14, 8, 0), frequency="每天", status="active",
    )
    system.create_reminder({
        "crop": "番茄", "reminder_type": "浇水", "start_date": "2030-06-14",
        "time_of_day": "07:00", "frequency": "每天", "channels": ["app"],
    })
    response = make_client().get(f"/api/chat/cards/task/{task.id}?username={username}")
    assert response.status_code == 200
    card = response.json()
    assert card["reminder_time"] == "2030-06-16T16:30:00"
    assert card["reminder_frequency"] == "单次"


def test_task_card_without_linked_reminder_keeps_optional_time_empty(card_context):
    username, _, task, _ = card_context
    card = make_client().get(f"/api/chat/cards/task/{task.id}?username={username}").json()
    assert card["reminder_time"] is None
    assert card["reminder_frequency"] is None


def test_task_card_uses_earliest_active_linked_reminder(card_context):
    from core.reminder_system import ReminderSystem

    username, _, task, _ = card_context
    system = ReminderSystem(username=username)
    for date, time, frequency in [
        ("2030-06-18", "18:00", "单次"), ("2030-06-17", "09:30", "每天"),
    ]:
        system.create_reminder({
            "crop": "番茄", "reminder_type": "浇水", "task_id": task.id,
            "start_date": date, "time_of_day": time, "frequency": frequency, "channels": ["app"],
        })
    completed = system.create_reminder({
        "crop": "番茄", "reminder_type": "浇水", "task_id": task.id,
        "start_date": "2030-06-16", "time_of_day": "08:00", "frequency": "单次", "channels": ["app"],
    })
    system.storage.update_reminder(completed.id, {"status": "completed"})
    card = make_client().get(f"/api/chat/cards/task/{task.id}?username={username}").json()
    assert card["reminder_time"] == "2030-06-17T09:30:00"
    assert card["reminder_frequency"] == "每天"
