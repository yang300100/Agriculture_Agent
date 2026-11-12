"""摄像头实时预览的传输、账号隔离与连接回收。"""

import asyncio
import threading
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api_routes import register_routes
from core.device_registry_factory import save_custom_devices


@pytest.fixture
def camera_api(monkeypatch):
    import devices.camera_driver as module
    username = f"stream_{uuid.uuid4().hex}"
    save_custom_devices(username, [{
        "device_id": "camera", "name": "温室摄像头", "driver": "camera",
        "capabilities": ["capture"], "connection": {"camera_type": "usb", "source": "0"},
    }, {
        "device_id": "http_camera", "name": "HTTP摄像头", "driver": "http",
        "capabilities": ["capture"], "connection": {"base_url": "http://127.0.0.1:1"},
    }])
    instances = []

    class Camera:
        fail = False

        def __init__(self, username):
            self.username = username
            self.closed = False
            self.registration = None
            instances.append(self)

        def register_device(self, **config):
            self.registration = config

        def stream_frames(self, device_id):
            try:
                if self.fail:
                    raise RuntimeError("摄像头不可用")
                yield b"first-jpeg"
                yield b"second-jpeg"
            finally:
                self.closed = True

    monkeypatch.setattr(module, "CameraDriver", Camera)
    app = FastAPI()
    register_routes(app)
    return app, username, instances, Camera


def test_camera_stream_returns_frames_and_releases_source(camera_api):
    app, username, instances, _ = camera_api
    with TestClient(app) as client:
        response = client.get(f"/api/devices/camera/stream?username={username}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace; boundary=frame")
    assert response.headers["cache-control"] == "no-store"
    assert response.content.count(b"Content-Type: image/jpeg") == 2
    assert b"Content-Length: 10\r\n" in response.content
    assert b"first-jpeg" in response.content and b"second-jpeg" in response.content
    assert len(instances) == 1 and instances[0].closed
    assert instances[0].username == username
    assert instances[0].registration["source"] == "0"


def test_unreadable_camera_returns_failure_before_video_headers(camera_api):
    app, username, instances, Camera = camera_api
    Camera.fail = True
    with TestClient(app) as client:
        response = client.get(f"/api/devices/camera/stream?username={username}")
    assert response.status_code == 503
    assert "摄像头" in response.json()["detail"]
    assert instances[0].closed


def test_stream_cannot_open_missing_or_unsupported_camera(camera_api):
    app, username, instances, _ = camera_api
    with TestClient(app) as client:
        assert client.get(f"/api/devices/missing/stream?username={username}").status_code == 404
        assert client.get(f"/api/devices/http_camera/stream?username={username}").status_code == 409
        assert client.get(f"/api/devices/camera/stream?username=other_{uuid.uuid4().hex}").status_code == 404
    assert not instances


def test_camera_stream_respects_bearer_account_and_duplicate_query(camera_api, monkeypatch):
    from core.auth_token import create_token
    app, username, instances, _ = camera_api
    secret = "camera-stream-test-authentication-secret"
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("AUTH_SECRET_KEY", secret)
    with TestClient(app) as client:
        assert client.get(f"/api/devices/camera/stream?username={username}").status_code == 401
        headers = {"Authorization": "Bearer " + create_token("someone_else", secret)}
        assert client.get(f"/api/devices/camera/stream?username={username}", headers=headers).status_code == 403
        assert client.get(f"/api/devices/camera/stream?username=someone_else&username={username}",
                          headers=headers).status_code == 400
        assert not instances
        headers = {"Authorization": "Bearer " + create_token(username, secret)}
        assert client.get("/api/devices/camera/stream", headers=headers).status_code == 200
    assert instances[0].username == username and instances[0].closed


def test_client_disconnect_closes_primed_camera_iterator(camera_api):
    app, username, instances, _ = camera_api
    route = next((route for route in app.routes if route.path == "/api/devices/{device_id}/stream"), None)
    assert route is not None
    response = route.endpoint("camera", username)

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("浏览器已断开")

    async def receive():
        return {"type": "http.disconnect"}

    async def run():
        from starlette.requests import ClientDisconnect
        try:
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        except ClientDisconnect:
            pass
    asyncio.run(run())
    assert instances[0].closed


def test_cancel_while_next_frame_is_pending_releases_camera(camera_api, monkeypatch):
    app, username, instances, Camera = camera_api
    reading = threading.Event()
    finish_read = threading.Event()

    def slow_frames(self, device_id):
        try:
            yield b"first-jpeg"
            reading.set()
            assert finish_read.wait(timeout=3)
            yield b"second-jpeg"
        finally:
            self.closed = True

    monkeypatch.setattr(Camera, "stream_frames", slow_frames)
    route = next(route for route in app.routes if route.path == "/api/devices/{device_id}/stream")
    response = route.endpoint("camera", username)

    async def send(message):
        pass

    async def receive():
        return {"type": "http.disconnect"}

    async def run():
        task = asyncio.create_task(response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send))
        assert await asyncio.to_thread(reading.wait, 2)
        task.cancel()
        # 一次调度确认取消已进入清理；缓慢帧仍在另一线程等待。
        await asyncio.sleep(.05)
        finish_read.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
    try:
        asyncio.run(run())
        assert instances[0].closed
    finally:
        finish_read.set()
