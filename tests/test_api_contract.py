"""前后端部署边界所需的 API 契约测试。"""

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api_routes import register_routes


def test_部署所需接口均已注册():
    app = FastAPI()
    register_routes(app)
    routes = {(route.path, method) for route in app.routes for method in route.methods}

    expected = {
        ("/api/auth/login", "POST"),
        ("/api/auth/register", "POST"),
        ("/api/chat/sessions", "GET"),
        ("/api/chat/sessions", "POST"),
        ("/api/chat/sessions/{session_id}", "GET"),
        ("/api/chat/sessions/{session_id}", "DELETE"),
        ("/api/fields", "GET"),
        ("/api/fields", "POST"),
        ("/api/fields/{fid}/history", "POST"),
        ("/api/fields/{fid}/zones", "GET"),
        ("/api/fields/{fid}/zones", "POST"),
        ("/api/fields/{fid}/zones/{zone_id}", "DELETE"),
        ("/api/profile", "GET"),
        ("/api/profile", "POST"),
        ("/api/alerts/proactive", "GET"),
        ("/api/reminders/check", "POST"),
        ("/api/sms/test", "POST"),
        ("/api/devices/{device_id}", "PUT"),
        ("/api/devices/{device_id}/config", "POST"),
        ("/api/device-actions/catalog", "GET"),
        ("/api/actions/{action_id}", "PUT"),
        ("/api/safety-policies", "GET"),
        ("/api/safety-policies", "POST"),
        ("/api/safety-policies/{policy_id}", "PUT"),
        ("/api/safety-policies/{policy_id}", "DELETE"),
    }

    assert expected <= routes


def test_待确认读取异常返回服务错误而不是空列表(monkeypatch):
    import core.device_executor as executor_module

    class BrokenExecutor:
        def __init__(self, *args, **kwargs):
            raise OSError("pending storage unavailable")

    monkeypatch.setattr(executor_module, "DeviceExecutor", BrokenExecutor)
    app = FastAPI()
    register_routes(app)

    response = TestClient(app).get("/api/actions/pending?username=default")

    assert response.status_code == 500
    assert response.json()["detail"] == "待确认操作暂时无法读取"


def test_财务导出使用当前用户而不是默认用户(tmp_path):
    from core.finance_manager import FinanceManager

    username = "finance_export_contract_user"
    FinanceManager(str(tmp_path / username)).add_cost({
        "crop": "小麦",
        "cost_type": "肥料",
        "item_name": "复合肥",
        "quantity": 1,
        "unit": "袋",
        "unit_price": 500,
    })
    app = FastAPI()
    register_routes(app)

    response = TestClient(app).get(f"/api/finance/export?username={username}")

    assert response.status_code == 200
    assert "复合肥" in response.json()["csv"]
    assert "500" in response.json()["csv"]


def test_待确认参数可通过api保存并重新读取():
    from core.device_executor import DeviceExecutor
    from devices.registry import DeviceDriverRegistry

    username = "api_pending_user"
    executor = DeviceExecutor(DeviceDriverRegistry(), username=username)
    action_id = executor.add_pending({
        "device_id": "virtual_irrigation_01",
        "command": "start",
        "params": {"duration": 20},
        "capability": "irrigate",
    })
    app = FastAPI()
    register_routes(app)
    client = TestClient(app)

    updated = client.put(
        f"/api/actions/{action_id}?username={username}",
        json={"params": {"duration": 35}},
    )

    assert updated.status_code == 200
    assert updated.json()["action"]["params"] == {"duration": 35}
    pending = client.get(
        f"/api/actions/pending?username={username}"
    ).json()
    assert pending[0]["params"] == {"duration": 35}
