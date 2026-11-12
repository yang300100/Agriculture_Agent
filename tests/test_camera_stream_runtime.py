"""哥哥，用可控摄像头验证连续帧共享、错误和资源释放，不连接真实硬件。"""

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

import devices.camera_driver as module
from devices.base import DeviceCapability, DeviceCommand, DeviceStatus


class Frame:
    shape = (240, 320, 3)


class Encoded:
    def tobytes(self):
        return b"jpeg-frame"


@pytest.fixture
def fake_cv(monkeypatch):
    class FakeCv:
        CAP_PROP_BUFFERSIZE = 38
        CAP_PROP_FPS = 5
        CAP_DSHOW = 700
        IMWRITE_JPEG_QUALITY = 1
        opened = []
        fail_sources = set()
        read_fails = False
        encode_fails = False
        read_raises = False
        lock = threading.Lock()
        active = 0
        max_active = 0
        operations = []

        @classmethod
        def VideoCapture(cls, source, *args):
            cls.operations.append(("open", threading.get_ident()))
            class Capture:
                released = False
                def isOpened(self):
                    return source not in cls.fail_sources
                def read(self):
                    cls.operations.append(("read", threading.get_ident()))
                    if cls.read_raises:
                        raise RuntimeError("设备读取异常")
                    return (False, None) if cls.read_fails else (True, Frame())
                def get(self, prop):
                    return 30
                def set(self, *args):
                    pass
                def release(self):
                    cls.operations.append(("release", threading.get_ident()))
                    with cls.lock:
                        if not self.released:
                            self.released = True
                            cls.active -= 1
            capture = Capture()
            with cls.lock:
                cls.opened.append((source, args, capture))
                cls.active += 1
                cls.max_active = max(cls.max_active, cls.active)
            return capture

        @classmethod
        def imencode(cls, *args):
            cls.operations.append(("encode", threading.get_ident()))
            return (False, None) if cls.encode_fails else (True, Encoded())

    monkeypatch.setattr(module, "cv2", FakeCv)
    monkeypatch.setattr(module, "HAS_OPENCV", True)
    yield FakeCv
    assert FakeCv.active == 0, "测试结束后不能占用摄像头"


def driver(tmp_path, source="0"):
    value = module.CameraDriver(image_storage_dir=str(tmp_path))
    value.register_device("cam", "监控摄像头", [DeviceCapability.CAPTURE], source=source)
    return value


def frames(value):
    factory = getattr(value, "stream_frames", None)
    assert callable(factory), "摄像头应提供同步 JPEG 连续帧迭代器"
    return factory("cam")


def test_two_previews_capture_and_registry_probe_share_one_capture(fake_cv, tmp_path):
    first, second = driver(tmp_path / "one"), driver(tmp_path / "two", source=0)
    preview_a, preview_b = frames(first), frames(second)
    try:
        assert next(preview_a) == b"jpeg-frame"
        assert next(preview_b) == b"jpeg-frame"
        assert asyncio.run(second.connect())
        result = asyncio.run(second.execute("cam", DeviceCommand("capture", {"save": False})))
        assert result.success, result
        assert result.raw_response["saved_path"] is None
        assert result.raw_response["metadata"]["width"] == 320
        assert result.raw_response["metadata"]["height"] == 240
        assert len(fake_cv.opened) == 1
        assert fake_cv.max_active == 1
        preview_a.close()
        assert fake_cv.active == 1
        assert next(preview_b) == b"jpeg-frame"
    finally:
        preview_a.close()
        preview_b.close()
        asyncio.run(first.disconnect())
        asyncio.run(second.disconnect())
    assert fake_cv.active == 0
    assert not list(tmp_path.rglob("*.jpg"))


def test_usb_ignores_unused_credentials_when_sharing_capture(fake_cv, tmp_path):
    first, second = driver(tmp_path / "one"), driver(tmp_path / "two", source=0)
    second.register_device("cam", "监控摄像头", [DeviceCapability.CAPTURE],
                           source=0, username="admin", password="unused")
    preview_a, preview_b = frames(first), frames(second)
    try:
        assert next(preview_a) == b"jpeg-frame"
        assert next(preview_b) == b"jpeg-frame"
        assert len(fake_cv.opened) == 1
        assert fake_cv.max_active == 1
    finally:
        preview_a.close()
        preview_b.close()
    assert fake_cv.active == 0


def test_capture_save_false_does_not_persist_and_default_still_saves(fake_cv, tmp_path):
    value = driver(tmp_path)
    transient = asyncio.run(value.execute("cam", DeviceCommand("capture", {"save": False})))
    assert transient.success, transient
    assert not list(tmp_path.rglob("*.jpg"))
    saved = asyncio.run(value.execute("cam", DeviceCommand("capture")))
    assert saved.success, saved
    assert Path(saved.raw_response["saved_path"]).read_bytes() == b"jpeg-frame"
    assert fake_cv.active == 0


def test_per_device_observation_detects_failure_and_recovery(fake_cv, tmp_path):
    value = driver(tmp_path)
    value.register_device("offline", "离线摄像头", [DeviceCapability.CAPTURE], source="1")
    fake_cv.fail_sources.add(1)
    assert asyncio.run(value.connect())
    state = asyncio.run(value.read_state("cam"))
    assert state["power"] is True and state["_quality"] == "good"
    assert state["_observed_at"]
    found = {d.device_id: d.status for d in asyncio.run(value.discover())}
    assert found == {"cam": DeviceStatus.ONLINE, "offline": DeviceStatus.ERROR}
    failed = asyncio.run(value.read_state("offline"))
    assert failed["power"] is False and failed["_quality"] == "error"
    fake_cv.fail_sources.clear()
    recovered = asyncio.run(value.read_state("offline"))
    assert recovered["power"] is True and recovered["_quality"] == "good"
    assert "error_reason" not in recovered


@pytest.mark.parametrize("fault,expected_code", [
    ("read_fails", "FRAME_READ_FAILED"), ("read_raises", "FRAME_READ_FAILED"),
    ("encode_fails", "ENCODE_FAILED"),
])
def test_stream_failure_raises_and_releases_without_simulated_fallback(fake_cv, tmp_path, fault, expected_code):
    value = driver(tmp_path)
    preview = frames(value)
    try:
        assert next(preview) == b"jpeg-frame"
        observed = value._devices["cam"]["state"]["_observed_at"]
        setattr(fake_cv, fault, True)
        time.sleep(.15)
        with pytest.raises(RuntimeError) as error:
            next(preview)
        assert error.value.error_code == expected_code
        assert value._devices["cam"]["state"]["_quality"] == "error"
        assert value._devices["cam"]["state"]["_observed_at"] == observed
    finally:
        preview.close()
        asyncio.run(value.disconnect())
    assert fake_cv.active == 0


def test_disconnect_releases_only_own_consumers(fake_cv, tmp_path):
    first, second = driver(tmp_path / "one"), driver(tmp_path / "two")
    preview_a, preview_b = frames(first), frames(second)
    try:
        next(preview_a)
        next(preview_b)
        asyncio.run(first.disconnect())
        assert fake_cv.active == 1
        assert next(preview_b) == b"jpeg-frame"
        asyncio.run(second.disconnect())
        assert fake_cv.active == 0
    finally:
        preview_a.close()
        preview_b.close()


def test_usb_accepts_integer_source_and_prefers_directshow_on_windows(fake_cv, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    value = driver(tmp_path, source=0)
    assert asyncio.run(value.connect())
    assert fake_cv.opened[0][0:2] == (0, (700,))
    assert fake_cv.active == 0


def test_capture_open_read_encode_release_stay_on_one_worker(fake_cv, tmp_path):
    value = driver(tmp_path)
    result = asyncio.run(value.execute("cam", DeviceCommand("capture", {"save": False})))
    assert result.success, result
    assert {name for name, _ in fake_cv.operations} == {"open", "read", "encode", "release"}
    assert len({thread_id for _, thread_id in fake_cv.operations}) == 1
    assert fake_cv.operations[0][1] != threading.get_ident()


@pytest.mark.parametrize("source", [-1, "-1", True, None, ""])
def test_invalid_camera_source_is_rejected_before_open(fake_cv, tmp_path, source):
    with pytest.raises(ValueError):
        driver(tmp_path, source=source)
    assert fake_cv.opened == []
