"""农事任务 API 测试。"""

from fastapi import FastAPI

import app.api_routes as api_routes
import core.planting_tracker as planting_tracker


def _route_endpoint(app: FastAPI, path: str, method: str):
    for route in app.routes:
        if route.path == path and method in route.methods:
            return route.endpoint
    raise AssertionError(f"未找到路由: {method} {path}")


def test_task_list_does_not_hide_new_items_after_ten(monkeypatch):
    requested_limits = []
    rows = [{"id": str(index), "title": f"任务 {index}"} for index in range(14)]

    class FakeTracker:
        def __init__(self, storage_dir):
            self.storage_dir = storage_dir

        def get_task_cards(self, limit):
            requested_limits.append(limit)
            return rows[:limit]

    monkeypatch.setattr(planting_tracker, "PlantingTracker", FakeTracker)

    app = FastAPI()
    api_routes.register_routes(app)
    list_tasks = _route_endpoint(app, "/api/tasks", "GET")

    result = list_tasks(username="farmer", limit=200)

    assert len(result) == 14
    assert result[-1]["title"] == "任务 13"
    assert requested_limits == [200]


def test_task_list_limit_is_bounded(monkeypatch):
    requested_limits = []

    class FakeTracker:
        def __init__(self, storage_dir):
            self.storage_dir = storage_dir

        def get_task_cards(self, limit):
            requested_limits.append(limit)
            return []

    monkeypatch.setattr(planting_tracker, "PlantingTracker", FakeTracker)

    app = FastAPI()
    api_routes.register_routes(app)
    list_tasks = _route_endpoint(app, "/api/tasks", "GET")

    list_tasks(username="farmer", limit=5000)

    assert requested_limits == [1000]
