"""验证演示设备注册与旧摄像头配置的定向修复。"""

import json
import uuid

import pytest

from core.database.engine import Session, init_db
from core.database.models import DeviceConfig, User
from scripts.register_simulated_devices import register


def make_user_camera(connection=None):
    init_db()
    username = f"camera_registration_{uuid.uuid4().hex}"
    with Session() as session:
        user = User(username=username, password_hash="")
        session.add(user)
        session.flush()
        if connection is not None:
            session.add(DeviceConfig(
                user_id=user.id, device_id="greenhouse_camera_01",
                name="北棚摄像头", driver="modbus", capabilities='["capture"]',
                sensors="[]", location="北棚入口", plot_id=42, zone_id="north",
                initial_state='{"power":true}', connection=json.dumps(connection),
            ))
        session.commit()
    return username


def camera_for(username):
    with Session() as session:
        return session.query(DeviceConfig).join(User).filter(
            User.username == username,
            DeviceConfig.device_id == "greenhouse_camera_01",
        ).one()


def test_new_camera_registration_uses_image_capable_http_driver():
    username = make_user_camera()
    register(username)
    camera = camera_for(username)
    assert camera.driver == "http"
    assert json.loads(camera.connection) == {"base_url": "http://127.0.0.1:5000"}


def test_explicit_repair_changes_only_legacy_local_camera_transport():
    username = make_user_camera({
        "mode": "tcp", "host": "127.0.0.1", "port": "5020", "slave_id": 2,
    })
    before = camera_for(username)
    register(username, repair_legacy_camera=True)
    register(username, repair_legacy_camera=True)
    after = camera_for(username)
    assert after.driver == "http"
    assert json.loads(after.connection) == {"base_url": "http://127.0.0.1:5000"}
    for key in ("id", "user_id", "name", "capabilities", "sensors", "location",
                "plot_id", "zone_id", "initial_state"):
        assert getattr(after, key) == getattr(before, key)
    with Session() as session:
        assert session.query(DeviceConfig).filter_by(user_id=after.user_id).count() == 7


@pytest.mark.parametrize("connection,repair", [
    ({"mode": "tcp", "host": "192.168.1.10", "port": "5020", "slave_id": 2}, True),
    ({"mode": "tcp", "host": "127.0.0.1", "port": "5021", "slave_id": 2}, True),
    ({"mode": "tcp", "host": "127.0.0.1", "port": "5020", "slave_id": 7}, True),
    ({"mode": "tcp", "host": "127.0.0.1", "port": "5020", "slave_id": 2}, False),
])
def test_registration_preserves_other_camera_connections(connection, repair):
    username = make_user_camera(connection)
    before = camera_for(username)
    register(username, repair_legacy_camera=repair)
    after = camera_for(username)
    assert after.driver == before.driver
    assert after.connection == before.connection
