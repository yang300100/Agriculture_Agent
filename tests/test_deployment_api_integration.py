"""使用临时数据库验证可分离部署的关键数据链路。"""

import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api_routes import register_routes
from core.storage_paths import DEFAULT_DATA_DIR


@pytest.fixture
def deployment_client(tmp_path, monkeypatch):
    import core.database.engine as db_engine

    original_url = db_engine.DB_URL
    db_engine._engine.dispose()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'deployment.db'}")
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv(
        "AUTH_SECRET_KEY", "integration-test-secret-at-least-32-chars"
    )
    importlib.reload(db_engine)

    app = FastAPI()
    register_routes(app)
    try:
        yield TestClient(app)
    finally:
        db_engine._engine.dispose()
        monkeypatch.setenv("DATABASE_URL", original_url)
        importlib.reload(db_engine)


def test_认证档案地块和会话均可只经api完成(deployment_client):
    client = deployment_client

    registered = client.post(
        "/api/auth/register",
        json={"username": "deploy_user", "password": "secret"},
    )
    assert registered.status_code == 200
    client.headers["Authorization"] = f"Bearer {registered.json()['token']}"
    assert client.post(
        "/api/auth/login",
        json={"username": "deploy_user", "password": "secret"},
    ).json()["success"] is True

    profile = {
        "user_region": "北京",
        "user_soil_type": "壤土",
        "user_farm_size": 12.5,
        "user_experience": "中级",
        "user_goals": ["高产"],
        "user_phone": "13800138000",
    }
    assert client.post(
        "/api/profile?username=deploy_user", json=profile
    ).json()["success"] is True
    assert client.get(
        "/api/profile?username=deploy_user"
    ).json()["user_region"] == "北京"

    field_result = client.post(
        "/api/fields?username=deploy_user",
        json={
            "name": "东地块",
            "coordinates": [[116.0, 39.0], [116.01, 39.0], [116.01, 39.01]],
            "soil_type": "壤土",
            "current_crop": "小麦",
        },
    ).json()
    assert field_result["success"] is True
    field_id = field_result["field"]["id"]
    assert client.post(
        f"/api/fields/{field_id}/history?username=deploy_user",
        json={"crop": "玉米", "season": "2025秋", "yield_amount": 500},
    ).json()["success"] is True
    assert client.get(
        "/api/fields?username=deploy_user"
    ).json()[0]["history"][0]["crop"] == "玉米"

    zone = client.post(
        f"/api/fields/{field_id}/zones?username=deploy_user",
        json={
            "zone_id": "north_bed",
            "name": "北侧苗床",
            "zone_type": "observation",
            "coordinates": [[116.0, 39.0], [116.005, 39.0]],
        },
    )
    assert zone.status_code == 200
    zones = client.get(
        f"/api/fields/{field_id}/zones?username=deploy_user"
    ).json()
    assert zones[0]["zone_id"] == "north_bed"
    assert zones[0]["name"] == "北侧苗床"
    assert client.post(
        f"/api/fields/{field_id}/zones?username=deploy_user",
        json={"zone_id": "north_bed", "name": "重复分区"},
    ).status_code == 409

    created_device = client.post(
        "/api/devices?username=deploy_user",
        json={
            "device_id": "zone_sensor_01",
            "name": "分区传感器",
            "driver": "simulator",
            "capabilities": ["read_sensor"],
            "sensors": ["soil_moisture"],
            "plot_id": field_id,
            "zone_id": "north_bed",
        },
    )
    assert created_device.json()["success"] is True
    assigned_delete = client.delete(
        f"/api/fields/{field_id}/zones/north_bed?username=deploy_user"
    )
    assert assigned_delete.status_code == 409
    assert client.put(
        "/api/devices/zone_sensor_01?username=deploy_user",
        json={"zone_id": ""},
    ).json()["success"] is True
    assert client.delete(
        f"/api/fields/{field_id}/zones/north_bed?username=deploy_user"
    ).json()["success"] is True

    saved = client.post(
        "/api/chat/sessions?username=deploy_user",
        json={
            "session_id": "new-session",
            "messages": [{"role": "user", "content": "你好"}],
        },
    ).json()
    assert saved["success"] is True
    session_id = saved["id"]
    assert client.get(
        f"/api/chat/sessions/{session_id}?username=deploy_user"
    ).json()["messages"] == [{"role": "user", "content": "你好"}]

    # 当前令牌不能冒充另一个用户。
    assert client.get("/api/fields?username=other_user").status_code == 403
    other = client.post(
        "/api/auth/register",
        json={"username": "other_user", "password": "secret"},
    ).json()
    client.headers["Authorization"] = f"Bearer {other['token']}"
    assert client.get("/api/fields?username=other_user").json() == []
    assert client.get("/api/chat/sessions?username=other_user").json() == []

    # 请求体里的用户名也不能绕过令牌主体校验。
    assert client.post(
        "/api/chat",
        json={"username": "deploy_user", "user_question": "越权测试"},
    ).status_code == 403

    # 部署后任何带用户名的数据路径都必须拒绝目录穿越。
    assert client.get("/api/fields?username=../../escape").status_code == 400


def test_开启认证后健康检查仍可公开访问(deployment_client):
    response = deployment_client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["service"] == "agriculture-agent"
    assert response.json()["status"] == "ok"


def test_省略查询用户名时自动使用令牌主体(deployment_client):
    client = deployment_client
    registered = client.post(
        "/api/auth/register",
        json={"username": "token_owner", "password": "secret"},
    ).json()
    client.headers["Authorization"] = f"Bearer {registered['token']}"

    created = client.post(
        "/api/fields",
        json={
            "name": "令牌地块",
            "coordinates": [[116, 39], [116.01, 39], [116.01, 39.01]],
        },
    )

    assert created.status_code == 200
    assert client.get("/api/fields").json()[0]["name"] == "令牌地块"


def test_认证后文件接口拒绝路径穿越(deployment_client):
    client = deployment_client
    registered = client.post(
        "/api/auth/register",
        json={"username": "path_owner", "password": "secret"},
    ).json()
    client.headers["Authorization"] = f"Bearer {registered['token']}"

    victim_dir = Path(DEFAULT_DATA_DIR, "victim", "autonomous_reports")
    victim_dir.mkdir(parents=True, exist_ok=True)
    (victim_dir / "secret.json").write_text(
        '{"marker":"不应被读取"}', encoding="utf-8"
    )

    report_response = client.get(
        "/api/autonomous/reports/..%5C..%5Cvictim%5Cautonomous_reports%5Csecret",
        params={"username": "path_owner"},
    )
    camera_response = client.get(
        "/api/camera/analysis/..%5C..%5Cvictim",
        params={"username": "path_owner"},
    )
    encyclopedia_response = client.get(
        "/api/encyclopedia/..%5C..%5Cdata%5Cvictim",
        params={"username": "path_owner"},
    )

    assert report_response.status_code == 400
    assert camera_response.status_code == 400
    assert encyclopedia_response.status_code == 400


def test_并发注册同名用户只会成功一次且不会返回500(deployment_client):
    def register_once(_):
        return deployment_client.post(
            "/api/auth/register",
            json={"username": "same_user", "password": "secret"},
        ).status_code

    with ThreadPoolExecutor(max_workers=4) as executor:
        statuses = list(executor.map(register_once, range(4)))

    assert statuses.count(200) == 1
    assert statuses.count(409) == 3
