"""设备配置更新接口测试。"""

from copy import deepcopy

from fastapi import FastAPI

import app.api_routes as api_routes
import core.plot_manager as plot_manager


def _route_endpoint(app: FastAPI, path: str, method: str):
    for route in app.routes:
        if route.path == path and method in route.methods:
            return route.endpoint
    raise AssertionError(f"未找到路由: {method} {path}")


def test_update_device_preserves_identity_and_reconnects(monkeypatch):
    devices = [
        {
            "device_id": "soil_01",
            "name": "一号土壤传感器",
            "driver": "mqtt",
            "capabilities": ["read_sensor"],
            "sensors": ["soil_moisture", "temperature"],
            "location": "旧位置",
            "plot_id": "plot_old",
            "zone_id": "north_bed",
            "connection": {"host": "old-broker", "port": 1883},
            "initial_state": {"power": True, "status": "online"},
        }
    ]
    saved = []
    invalidated = []
    monkeypatch.setattr(
        api_routes,
        "load_custom_devices",
        lambda username: deepcopy(devices),
    )
    monkeypatch.setattr(
        api_routes,
        "save_custom_devices",
        lambda username, rows: saved.extend(deepcopy(rows)),
    )
    monkeypatch.setattr(
        api_routes,
        "invalidate_registry_cache",
        lambda username: invalidated.append(username),
    )

    app = FastAPI()
    api_routes.register_routes(app)
    update = _route_endpoint(app, "/api/devices/{device_id}", "PUT")
    result = update(
        "soil_01",
        {
            "location": "东侧温室",
            "plot_id": "plot_east",
            "connection": {"host": "new-broker", "port": 2883},
        },
        "farmer",
    )

    assert result["success"] is True
    assert saved[0]["device_id"] == "soil_01"
    assert saved[0]["name"] == "一号土壤传感器"
    assert saved[0]["location"] == "东侧温室"
    assert saved[0]["zone_id"] == "north_bed"
    assert saved[0]["connection"]["host"] == "new-broker"
    assert saved[0]["sensors"] == ["soil_moisture", "temperature"]
    assert invalidated == ["farmer"]


def test_update_device_rejects_invalid_connection(monkeypatch):
    devices = [
        {
            "device_id": "http_01",
            "name": "HTTP 灌溉器",
            "driver": "http",
            "capabilities": ["irrigate"],
            "sensors": [],
            "connection": {"base_url": "http://192.168.1.10"},
            "initial_state": {},
        }
    ]
    monkeypatch.setattr(
        api_routes,
        "load_custom_devices",
        lambda username: deepcopy(devices),
    )
    monkeypatch.setattr(
        api_routes,
        "save_custom_devices",
        lambda username, rows: (_ for _ in ()).throw(
            AssertionError("非法配置不应写入")
        ),
    )

    app = FastAPI()
    api_routes.register_routes(app)
    update = _route_endpoint(app, "/api/devices/{device_id}", "PUT")
    result = update(
        "http_01",
        {"connection": {"base_url": "not-a-url"}},
        "farmer",
    )

    assert result == {"success": False, "error": "HTTP 设备必须提供有效的 base_url"}


def test_update_device_supports_post_config_route():
    app = FastAPI()
    api_routes.register_routes(app)

    put_update = _route_endpoint(app, "/api/devices/{device_id}", "PUT")
    post_update = _route_endpoint(
        app, "/api/devices/{device_id}/config", "POST"
    )

    assert post_update is put_update


def test_list_devices_keeps_unreachable_config_visible(monkeypatch):
    configured = {
        "device_id": "offline_sensor",
        "name": "离线土壤传感器",
        "driver": "mqtt",
        "capabilities": ["read_sensor"],
        "sensors": ["soil_moisture"],
        "location": "北侧棚室",
        "plot_id": "plot_north",
        "zone_id": "seedling_bed",
        "connection": {"host": "unreachable", "port": 1883},
        "initial_state": {"soil_moisture": 42},
    }
    monkeypatch.setattr(
        api_routes,
        "load_custom_devices",
        lambda username: [deepcopy(configured)],
    )

    class FakeRegistry:
        def discover_all(self):
            return "discover"

    class FakeLoop:
        def run_until_complete(self, operation):
            assert operation == "discover"
            return []

    class FakeRegistrySession:
        def __init__(self, username):
            self.username = username

        def __enter__(self):
            return FakeRegistry(), FakeLoop()

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakePlotManager:
        def __init__(self, username):
            self.username = username

        def list_plots(self):
            return [{"plot_id": "plot_north", "name": "北侧地块"}]

    monkeypatch.setattr(api_routes, "RegistrySession", FakeRegistrySession)
    monkeypatch.setattr(plot_manager, "PlotManager", FakePlotManager)

    app = FastAPI()
    api_routes.register_routes(app)
    list_devices = _route_endpoint(app, "/api/devices", "GET")
    rows = list_devices("farmer")

    assert len(rows) == 1
    assert rows[0]["device_id"] == "offline_sensor"
    assert rows[0]["status"] == "offline"
    assert rows[0]["plot_name"] == "北侧地块"
    assert rows[0]["zone_id"] == "seedling_bed"
    assert rows[0]["connection"]["host"] == "unreachable"
    assert rows[0]["state"]["soil_moisture"] == 42
    assert rows[0]["editable"] is True
