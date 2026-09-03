"""CameraDriver 单元测试 — 无需真实摄像头即可运行"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from devices.base import (
    DeviceCapability, DeviceStatus, DeviceInfo,
    DeviceCommand, DeviceResult,
)


# ── 模拟 cv2 和 numpy（避免依赖真实安装）──

class FakeFrame:
    """模拟 numpy ndarray 帧，支持 .shape 属性"""
    def __init__(self, h=480, w=640, c=3):
        self._h, self._w, self._c = h, w, c
    @property
    def shape(self):
        return (self._h, self._w, self._c)
    def tobytes(self):
        return b"fake_frame" * 1000


class FakeVideoCapture:
    """模拟 cv2.VideoCapture"""
    def __init__(self, *args, **kwargs):
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        return True, FakeFrame()

    def release(self):
        self._opened = False

    def get(self, prop):
        return 30  # fps

    def set(self, prop, value):
        pass


class FakeNumpyArray:
    """模拟 numpy array，支持 tobytes()"""
    def __init__(self, data):
        self._data = data
    def tobytes(self):
        return self._data


class FakeCv2Module:
    """完整的 cv2 mock 模块"""
    VideoCapture = FakeVideoCapture
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_FPS = 5
    CAP_PROP_OPEN_TIMEOUT_MSEC = 0
    IMWRITE_JPEG_QUALITY = 1

    @staticmethod
    def imencode(ext, img, params=None):
        return True, FakeNumpyArray(b"fake_jpeg_bytes" * 100)

    @staticmethod
    def imdecode(*args, **kwargs):
        return [[[100, 150, 200]]]


@pytest.fixture(autouse=True)
def _mock_cv2():
    """全局注入 mock cv2 + numpy 模块"""
    fake_cv2 = FakeCv2Module()
    sys.modules["cv2"] = fake_cv2
    sys.modules["numpy"] = MagicMock()
    # 直接修改已导入的 camera_driver 模块的属性
    import devices.camera_driver as cd
    cd.cv2 = fake_cv2
    cd.np = MagicMock()
    cd.HAS_OPENCV = True
    yield
    # 恢复（可选）


# ── 测试用例 ──────────────────────────────────


class TestCameraDriverRegistration:
    """摄像头驱动注册和基础功能测试"""

    def test_driver_name_is_camera(self):
        from devices.camera_driver import CameraDriver
        assert CameraDriver.driver_name == "camera"

    def test_import_error_when_opencv_missing(self):
        from devices.camera_driver import HAS_OPENCV as original_has
        try:
            import devices.camera_driver as cd
            cd.HAS_OPENCV = False
            with pytest.raises(ImportError, match="opencv-python"):
                cd.CameraDriver()
        finally:
            import devices.camera_driver as cd
            cd.HAS_OPENCV = original_has

    def test_register_device_stores_config(self):
        from devices.camera_driver import CameraDriver
        driver = CameraDriver()
        driver.register_device(
            "cam_test", "测试摄像头",
            capabilities=[DeviceCapability.CAPTURE],
            sensors=[], location="大棚A",
            camera_type="usb", source="0",
            username="admin", password="123",
        )
        assert "cam_test" in driver._devices
        dev = driver._devices["cam_test"]
        assert dev["info"]["name"] == "测试摄像头"
        assert dev["info"]["camera_type"] == "usb"
        assert dev["info"]["source"] == "0"
        assert dev["info"]["username"] == "admin"

    def test_discover_returns_device_info(self):
        from devices.camera_driver import CameraDriver
        driver = CameraDriver()
        driver._connected = True
        driver.register_device(
            "cam_01", "摄像头1",
            capabilities=[DeviceCapability.CAPTURE],
            location="大棚",
        )
        loop = asyncio.new_event_loop()
        devices = loop.run_until_complete(driver.discover())
        loop.close()

        assert len(devices) == 1
        d = devices[0]
        assert d.device_id == "cam_01"
        assert d.driver_name == "camera"
        assert DeviceCapability.CAPTURE in d.capabilities

    def test_read_state_returns_expected_keys(self):
        from devices.camera_driver import CameraDriver
        driver = CameraDriver()
        driver.register_device(
            "cam_01", "摄像头1",
            capabilities=[DeviceCapability.CAPTURE],
            camera_type="usb", source="0",
        )
        driver._connected = True
        loop = asyncio.new_event_loop()
        state = loop.run_until_complete(driver.read_state("cam_01"))
        loop.close()

        assert state["camera_type"] == "usb"
        assert state["source"] == "0"
        assert "_driver" in state
        assert state["_driver"] == "camera"

    def test_execute_unknown_command_returns_error(self):
        from devices.camera_driver import CameraDriver
        driver = CameraDriver()
        driver.register_device(
            "cam_01", "摄像头1",
            capabilities=[DeviceCapability.CAPTURE],
        )
        loop = asyncio.new_event_loop()
        cmd = DeviceCommand(command="unknown_cmd", params={})
        result = loop.run_until_complete(driver.execute("cam_01", cmd))
        loop.close()

        assert result.success is False
        assert result.error_code == "UNSUPPORTED_COMMAND"

    def test_health_check_reflects_connected_state(self):
        from devices.camera_driver import CameraDriver
        driver = CameraDriver()
        loop = asyncio.new_event_loop()
        ok = loop.run_until_complete(driver.health_check())
        loop.close()
        assert ok is False  # connect() 未调用

    def test_connect_verifies_cameras(self):
        from devices.camera_driver import CameraDriver
        driver = CameraDriver()
        driver.register_device(
            "cam_01", "摄像头1",
            capabilities=[DeviceCapability.CAPTURE],
            camera_type="usb", source="0",
        )
        loop = asyncio.new_event_loop()
        ok = loop.run_until_complete(driver.connect())
        loop.close()
        assert ok is True
        assert driver._connected is True

    def test_capture_succeeds_with_mock_camera(self):
        from devices.camera_driver import CameraDriver
        driver = CameraDriver(username="camera_user")
        driver.register_device(
            "cam_01", "摄像头1",
            capabilities=[DeviceCapability.CAPTURE],
            camera_type="usb", source="0",
        )
        driver._connected = True
        loop = asyncio.new_event_loop()
        cmd = DeviceCommand(command="capture", params={})
        result = loop.run_until_complete(driver.execute("cam_01", cmd))
        loop.close()

        assert result.success is True
        assert result.device_id == "cam_01"
        assert result.executed_command == "capture"
        assert "image_bytes" in result.raw_response
        assert len(result.raw_response["image_bytes"]) > 0
        saved_path = Path(result.raw_response["saved_path"])
        expected_dir = (
            Path(os.environ["DATA_STORAGE_DIR"])
            / "camera_user" / "photos" / "cam_01"
        ).resolve()
        assert saved_path.parent.resolve() == expected_dir
        assert saved_path.read_bytes() == result.raw_response["image_bytes"]


class TestCameraDriverInRegistry:
    """CameraDriver 在注册中心中的集成测试"""

    def test_camera_driver_registry_integration(self):
        from devices.registry import DeviceDriverRegistry
        from devices.camera_driver import CameraDriver

        registry = DeviceDriverRegistry()
        driver = CameraDriver()
        driver.register_device(
            "cam_01", "摄像头1",
            capabilities=[DeviceCapability.CAPTURE],
        )
        registry.register("camera", driver)

        assert "camera" in registry.driver_names
        assert registry.get_driver("cam_01") is None  # 需先 discover

        loop = asyncio.new_event_loop()
        devices = loop.run_until_complete(registry.discover_all())
        loop.close()

        assert any(d.driver_name == "camera" for d in devices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
