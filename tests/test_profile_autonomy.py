"""哥哥，自主权必须跟随用户持久保存，并在最终设备门禁生效。"""

import asyncio
import importlib
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.api_routes import register_routes
from core.database.migrations import apply_migrations
from core.database.repository.users import UserProfileRepository, UserRepository
from core.device_executor import DeviceExecutor
from devices.base import DeviceCommand, DeviceResult


@pytest.fixture
def profile_client():
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        yield client


def _username():
    return f"autonomy_{uuid.uuid4().hex}"


def test_用户自主权隔离并在重新加载配置后保留(profile_client, monkeypatch):
    from app.agent import config

    monkeypatch.setenv("AUTONOMY_LEVEL", "medium")
    alice, bob = _username(), _username()
    assert profile_client.post(f"/api/profile?username={alice}", json={"autonomy_level": "low"}).status_code == 200
    assert profile_client.post(f"/api/profile?username={bob}", json={"autonomy_level": "high"}).status_code == 200

    importlib.reload(config)

    assert profile_client.get(f"/api/profile?username={alice}").json()["autonomy_level"] == "low"
    assert profile_client.get(f"/api/profile?username={bob}").json()["autonomy_level"] == "high"
    assert config.get_autonomy_level(alice) == "low"
    assert config.get_autonomy_level(bob) == "high"


def test_旧客户端省略自主权不覆盖已保存值(profile_client):
    username = _username()
    profile_client.post(f"/api/profile?username={username}", json={"autonomy_level": "low"})
    assert profile_client.post(f"/api/profile?username={username}", json={"user_region": "河北"}).json()["success"] is True
    profile = profile_client.get(f"/api/profile?username={username}").json()
    assert profile["user_region"] == "河北"
    assert profile["autonomy_level"] == "low"


@pytest.mark.parametrize("invalid", ["invalid", "HIGH", "", 1, []])
def test_无效自主权被服务端拒绝且不覆盖原值(profile_client, invalid):
    username = _username()
    profile_client.post(f"/api/profile?username={username}", json={"autonomy_level": "low"})
    response = profile_client.post(f"/api/profile?username={username}", json={"autonomy_level": invalid})
    assert response.status_code == 422
    assert profile_client.get(f"/api/profile?username={username}").json()["autonomy_level"] == "low"


def test_没有档案或旧档案未设置时返回环境默认值(profile_client, monkeypatch):
    from app.agent.config import get_autonomy_level

    monkeypatch.setenv("AUTONOMY_LEVEL", "high")
    username = _username()
    assert profile_client.get(f"/api/profile?username={username}").json()["autonomy_level"] == "high"
    profile_client.post(f"/api/profile?username={username}", json={"user_region": "河北"})
    assert get_autonomy_level(username) == "high"
    user = UserRepository().get_by_username(username)
    assert UserProfileRepository().get_by_user_id(user.id).autonomy_level is None
    monkeypatch.setenv("AUTONOMY_LEVEL", "not-a-level")
    assert get_autonomy_level(username) == "medium"


def test_旧数据库补充可空自主权且迁移可重入(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old-profiles.db'}")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE user_profiles (id INTEGER PRIMARY KEY, region VARCHAR(100))"))
            connection.execute(text("INSERT INTO user_profiles VALUES (7, '河北')"))
        apply_migrations(engine)
        apply_migrations(engine)
        with engine.connect() as connection:
            assert connection.execute(text("SELECT id, region, autonomy_level FROM user_profiles")).one() == (7, "河北", None)
    finally:
        engine.dispose()


def test_最终执行门禁读取对应用户自主权并保留停止和确认路径(profile_client, tmp_path, monkeypatch):
    import core.device_executor as executor_module

    monkeypatch.setattr(executor_module, "DEFAULT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AUTONOMY_LEVEL", "high")
    low_user, high_user = _username(), _username()
    profile_client.post(f"/api/profile?username={low_user}", json={"autonomy_level": "low"})
    profile_client.post(f"/api/profile?username={high_user}", json={"autonomy_level": "high"})

    class Registry:
        async def execute(self, device_id, command):
            return DeviceResult(success=True, device_id=device_id, executed_command=command.command, actual_params=command.params)

    registry = Registry()
    low = DeviceExecutor(registry, low_user)
    high = DeviceExecutor(registry, high_user)
    command = DeviceCommand("start", {"duration": 1})
    blocked = asyncio.run(low.execute("pump", command, capability="irrigate"))
    assert blocked["success"] is False
    assert blocked["result"].error_code == "POLICY_CONFIRM_REQUIRED"
    assert blocked["pending_id"]
    assert asyncio.run(high.execute("pump", command, capability="irrigate"))["success"] is True
    assert asyncio.run(low.execute("stop-pump", DeviceCommand("stop", {}), capability="irrigate"))["success"] is True
    assert low.confirm_pending(blocked["pending_id"])["success"] is True
