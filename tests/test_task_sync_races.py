"""哥哥，旧轮同步不能覆盖新执行，结束记录中的时间也必须属于原作业。"""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.device_safety_policy as safety_module
import devices.simulator_driver as simulator_module
from app.api_routes import register_routes
from core.device_registry_factory import RegistrySession, invalidate_registry_cache
from core.planting_tracker import PlantingTracker
from core.storage_paths import DEFAULT_DATA_DIR
from devices.base import DeviceCommand


@pytest.fixture
def task_setup(monkeypatch):
    monkeypatch.setenv("AUTONOMY_LEVEL", "medium")
    current = [2_000_000_000.0]

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(current[0], tz=tz)

    monkeypatch.setattr(simulator_module, "datetime", FixedDatetime)
    monkeypatch.setattr(safety_module, "datetime", FixedDatetime)
    username = f"task_sync_{uuid.uuid4().hex}"
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        assert client.post(f"/api/devices?username={username}", json={
            "device_id": "pump", "name": "麦田水泵", "driver": "simulator",
            "capabilities": ["irrigate"],
            "initial_state": {"power": False, "status": "powered_off"},
        }).json()["success"]
        assert client.post(f"/api/tasks?username={username}", json={
            "crop": "小麦", "title": "麦田灌溉", "device_id": "pump",
            "device_command": "start", "device_params": {"duration": 5},
        }).json()["success"]
        task_id = client.get(f"/api/tasks?username={username}").json()[0]["id"]
        tracker = PlantingTracker(str(Path(DEFAULT_DATA_DIR) / username))
        yield client, username, task_id, tracker, current
    invalidate_registry_cache(username)


def test_旧同步快照不能把已重试的新一轮覆盖回已中止(task_setup):
    client, username, task_id, tracker, clock = task_setup
    assert client.post(f"/api/tasks/{task_id}/execute?username={username}").json()["success"]
    old_run = client.get(f"/api/devices/pump/state?username={username}").json()["run_id"]
    ready = threading.Event()
    release = threading.Event()

    def stale_sync():
        with RegistrySession(username) as (registry, loop):
            loop.run_until_complete(registry.discover_all())

            class PausedRegistry:
                async def read_state(self, device_id):
                    ready.set()
                    assert release.wait(timeout=8)
                    return await registry.read_state(device_id)

            tracker.sync_running_device_tasks(PausedRegistry(), loop)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(stale_sync)
        try:
            assert ready.wait(timeout=8)
            assert client.post(f"/api/devices/pump/command?username={username}", json={
                "command": "stop", "params": {},
            }).json()["success"]
            clock[0] += 120
            restarted = client.post(f"/api/tasks/{task_id}/execute?username={username}").json()
            assert restarted["success"], restarted
            new_run = client.get(f"/api/devices/pump/state?username={username}").json()["run_id"]
            assert new_run != old_run
        finally:
            release.set()
        pending.result(timeout=8)

    task = client.get(f"/api/tasks?username={username}").json()[0]
    assert task["status"] == "运行中"
    assert task["execution_state"]["run_id"] == new_run
    repeated = client.post(f"/api/tasks/{task_id}/execute?username={username}").json()
    assert not repeated["success"]
    assert client.get(f"/api/devices/pump/state?username={username}").json()["run_id"] == new_run


def test_下发后未落库的旧轮结束时间与启动时间均来自自身历史(task_setup):
    client, username, task_id, tracker, clock = task_setup
    assert tracker.claim_device_task(task_id)
    run_id = tracker._load_tasks()[0]["execution_state"]["run_id"]
    with RegistrySession(username) as (registry, loop):
        loop.run_until_complete(registry.discover_all())
        started = loop.run_until_complete(registry.execute(
            "pump", DeviceCommand("start", {"duration": 1}, execution_id=run_id),
        ))
        assert started.success
    # 哥哥，模拟设备启动后进程退出，任务只有领取编号，没有保存启动回执。
    clock[0] += 60
    assert client.get(f"/api/devices/pump/state?username={username}").json()["last_run_end_reason"] == "completed"
    assert client.post(f"/api/devices/pump/command?username={username}", json={
        "command": "start", "params": {"duration": 2},
    }).json()["success"]
    clock[0] += 120
    assert client.get(f"/api/devices/pump/state?username={username}").json()["last_run_end_reason"] == "completed"

    task = client.get(f"/api/tasks?username={username}").json()[0]

    assert task["status"] == "已完成"
    assert task["execution_state"]["run_id"] == run_id
    assert task["started_at"] == datetime.fromtimestamp(2_000_000_000).isoformat()
    assert task["run_until"] == 2_000_000_060
    assert task["ended_at"] == datetime.fromtimestamp(2_000_000_060).isoformat()
