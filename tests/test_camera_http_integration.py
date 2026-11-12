"""通过实际 HTTP 协议验证模拟器到摄像头抓拍 API 的链路。"""

import base64
import io
import threading
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from werkzeug.serving import make_server

from app.api_routes import register_routes
from core.device_registry_factory import invalidate_registry_cache, save_custom_devices


def test_camera_snapshot_crosses_http_driver_and_user_registry():
    from hardware_examples.all_hardware_simulator import UnifiedDeviceManager, create_http_app

    manager = UnifiedDeviceManager()
    manager.init_all()
    server = make_server("127.0.0.1", 0, create_http_app(manager))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    username = f"camera_http_{uuid.uuid4().hex}"
    save_custom_devices(username, [{
        "device_id": "greenhouse_camera_01", "name": "温室摄像头", "driver": "http",
        "capabilities": ["capture"], "connection": {
            "base_url": f"http://127.0.0.1:{server.server_port}",
        },
    }])
    app = FastAPI()
    register_routes(app)
    try:
        with TestClient(app) as client:
            devices = client.get("/api/devices", params={"username": username}).json()
            assert len(devices) == 1
            assert devices[0]["status"] == "online"
            response = client.get("/api/devices/greenhouse_camera_01/snapshot",
                                  params={"username": username})
            result = response.json()
            assert response.status_code == 200
            assert result["success"] is True, result
            assert result["mime_type"] == "image/jpeg"
            picture = Image.open(io.BytesIO(base64.b64decode(result["image_base64"], validate=True)))
            assert picture.width >= 320 and picture.height >= 240
            assert picture.size == (result["metadata"]["width"], result["metadata"]["height"])
            picture.verify()
            # 另一账号不能借用同名摄像头配置。
            denied = client.get("/api/devices/greenhouse_camera_01/snapshot",
                                params={"username": f"missing_{uuid.uuid4().hex}"}).json()
            assert denied["success"] is False
    finally:
        invalidate_registry_cache(username)
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
