"""摄像头驱动：USB、IP、ESP32-CAM 的抓拍、连续 JPEG 和实际观测状态。"""

import asyncio
import logging
import os
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List

from core.storage_paths import DEFAULT_DATA_DIR
from .base import BaseDeviceDriver, DeviceCapability, DeviceStatus, DeviceInfo, DeviceCommand, DeviceResult
from .camera_runtime import CAPTURE_POOL, CameraRuntimeError, camera_key

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    cv2 = np = None
    HAS_OPENCV = False


class CameraType:
    USB = "usb"
    IP = "ip"
    ESP32_CAM = "esp32cam"


class CameraDriver(BaseDeviceDriver):
    """同一进程内按图源共享采集；没有消费者时释放摄像头。"""

    driver_name = "camera"

    def __init__(self, image_storage_dir: str = None, username: str = None):
        if not HAS_OPENCV:
            raise ImportError("opencv-python 未安装。请运行: pip install opencv-python")
        self._connected = False
        self._closed = False
        self._devices = {}
        self._streaming = {}
        self._last_capture = {}
        self._leases = {}
        self._streams = {}
        self._lease_lock = threading.RLock()
        self._image_storage_dir = image_storage_dir or os.path.join(DEFAULT_DATA_DIR, username or "default", "photos")
        os.makedirs(self._image_storage_dir, exist_ok=True)

    def register_device(self, device_id: str, name: str,
                        capabilities: List[DeviceCapability], sensors: List[str] = None,
                        location: str = "", camera_type: str = "usb", source="0",
                        username: str = "", password: str = "") -> None:
        if camera_type not in (CameraType.USB, CameraType.IP, CameraType.ESP32_CAM):
            raise ValueError(f"未知摄像头类型: {camera_type}")
        camera_key(camera_type, source, username, password)
        self._devices[device_id] = {
            "info": {"device_id": device_id, "name": name, "capabilities": capabilities,
                     "sensors": sensors or [], "location": location, "camera_type": camera_type,
                     "source": source, "username": username, "password": password},
            "state": {"power": False, "status": "idle", "last_capture_time": None,
                      "resolution": "unknown", "fps": 0, "camera_type": camera_type,
                      "source": source, "_quality": "unknown", "_observed_at": None},
        }
        self._streaming[device_id] = False
        self._leases[device_id] = set()
        self._streams[device_id] = set()
        logger.info("摄像头设备已注册: %s (%s)", device_id, camera_type)

    def _acquire(self, device_id, streaming=False):
        with self._lease_lock:
            if self._closed:
                raise CameraRuntimeError("摄像头驱动已断开", "STREAM_CLOSED")
            if device_id not in self._devices:
                raise CameraRuntimeError(f"设备 '{device_id}' 未注册", "DEVICE_NOT_FOUND")
            info = self._devices[device_id]["info"]
            key = camera_key(info["camera_type"], info["source"], info["username"], info["password"])

            def encode(frame):
                ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok or jpeg is None:
                    raise CameraRuntimeError("JPEG 编码失败", "ENCODE_FAILED")
                return jpeg.tobytes()

            lease = CAPTURE_POOL.acquire(key, lambda: self._open_camera(*key), encode)
            self._leases[device_id].add(lease)
            if streaming:
                self._streams[device_id].add(lease)
                self._streaming[device_id] = True
            return lease

    def _release(self, device_id, lease):
        with self._lease_lock:
            self._leases.get(device_id, set()).discard(lease)
            self._streams.get(device_id, set()).discard(lease)
            self._streaming[device_id] = bool(self._streams.get(device_id))
        lease.close()

    def _observe(self, device_id, frame):
        with self._lease_lock:
            state = self._devices[device_id]["state"]
            state.update(power=True, status="idle", resolution=f"{frame.width}x{frame.height}",
                         fps=frame.fps, _quality="good", _observed_at=frame.observed_at)
            state.pop("error_reason", None)
            self._connected = True

    def _mark_error(self, device_id, error):
        with self._lease_lock:
            if device_id not in self._devices:
                return
            self._devices[device_id]["state"].update(
                power=False, status="error", _quality="error", error_reason=str(error)[:200])
            self._connected = any(d["state"].get("_quality") == "good" for d in self._devices.values())

    def _read_frame(self, device_id):
        lease = None
        try:
            lease = self._acquire(device_id)
            frame = lease.next_frame()
            self._observe(device_id, frame)
            return frame
        except CameraRuntimeError as exc:
            if exc.error_code != "STREAM_CLOSED":
                self._mark_error(device_id, exc)
            raise
        finally:
            if lease is not None:
                self._release(device_id, lease)

    def stream_frames(self, device_id):
        """同步迭代 JPEG bytes；关闭迭代器或 disconnect 后释放本消费者。

        首帧和后续帧失败均抛 CameraRuntimeError（带 error_code），不回退图片。
        """
        lease = None
        try:
            lease = self._acquire(device_id, streaming=True)
            sequence = 0
            while not lease.closed:
                frame = lease.next_frame(after_sequence=sequence)
                sequence = frame.sequence
                self._observe(device_id, frame)
                yield frame.image_bytes
        except CameraRuntimeError as exc:
            if exc.error_code != "STREAM_CLOSED":
                self._mark_error(device_id, exc)
                raise
        finally:
            if lease is not None:
                self._release(device_id, lease)

    async def _probe(self, device_id):
        try:
            await asyncio.to_thread(self._read_frame, device_id)
            return True
        except CameraRuntimeError:
            return False

    async def connect(self) -> bool:
        self._closed = False
        results = await asyncio.gather(*(self._probe(device_id) for device_id in self._devices))
        self._connected = any(results)
        return self._connected

    async def disconnect(self) -> None:
        with self._lease_lock:
            self._closed = True
            leases = [(device_id, lease) for device_id, values in self._leases.items() for lease in values]
        for device_id, lease in leases:
            await asyncio.to_thread(self._release, device_id, lease)
        self._connected = False

    async def health_check(self) -> bool:
        return self._connected

    async def discover(self) -> List[DeviceInfo]:
        if not self._closed:
            await asyncio.gather(*(self._probe(device_id) for device_id in self._devices))
        result = []
        for dev in self._devices.values():
            info, state = dev["info"], dev["state"]
            status = DeviceStatus.ONLINE if state.get("_quality") == "good" and not self._closed else DeviceStatus.OFFLINE
            if state.get("_quality") == "error":
                status = DeviceStatus.ERROR
            result.append(DeviceInfo(
                device_id=info["device_id"], name=info["name"], driver_name=self.driver_name,
                capabilities=info["capabilities"], sensors=info["sensors"], status=status,
                location=info["location"], metadata={"camera_type": info["camera_type"], "source": info["source"]}))
        return result

    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        if device_id not in self._devices:
            return DeviceResult(False, device_id, command.command, message=f"设备 '{device_id}' 未注册", error_code="DEVICE_NOT_FOUND")
        if command.command == "capture":
            return await self._handle_capture(device_id, command)
        if command.command == "stop_stream":
            with self._lease_lock:
                streams = list(self._streams[device_id])
            for lease in streams:
                await asyncio.to_thread(self._release, device_id, lease)
            return DeviceResult(True, device_id, command.command, message="推流已停止")
        if command.command == "start_stream":
            return DeviceResult(True, device_id, command.command, message="请连接实时预览接口，画面将按需开始传输")
        return DeviceResult(False, device_id, command.command,
                            message=f"摄像头不支持指令: {command.command}", error_code="UNSUPPORTED_COMMAND")

    async def _handle_capture(self, device_id, command):
        try:
            frame = await asyncio.to_thread(self._read_frame, device_id)
            info = self._devices[device_id]["info"]
            saved_path = None
            if command.params.get("save", True) is not False:
                saved_path = await asyncio.to_thread(self._save_image, device_id, frame.image_bytes)
            with self._lease_lock:
                self._devices[device_id]["state"]["last_capture_time"] = frame.observed_at
                self._last_capture[device_id] = frame.observed_at
            return DeviceResult(True, device_id, "capture",
                                message=f"抓拍成功 ({frame.width}x{frame.height}, {len(frame.image_bytes)} bytes)",
                                raw_response={"image_bytes": frame.image_bytes, "saved_path": saved_path,
                                              "metadata": {"width": frame.width, "height": frame.height,
                                                           "size_bytes": len(frame.image_bytes), "timestamp": frame.observed_at,
                                                           "camera_type": info["camera_type"], "source": info["source"]}})
        except Exception as exc:
            self._mark_error(device_id, exc)
            return DeviceResult(False, device_id, "capture", message=f"抓拍失败: {exc}",
                                error_code=getattr(exc, "error_code", "CAPTURE_ERROR"))

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        if device_id not in self._devices:
            return {"error": f"设备 '{device_id}' 不存在"}
        if not self._closed:
            await self._probe(device_id)
        with self._lease_lock:
            state = dict(self._devices[device_id]["state"])
            state.update(_read_at=datetime.now().isoformat(), _driver="camera",
                         _streaming=self._streaming.get(device_id, False))
        return state

    def _open_camera(self, camera_type, source, username="", password=""):
        """采集线程独占打开与释放；Windows USB 优先使用 DirectShow。"""
        cap = None
        try:
            if camera_type == CameraType.USB:
                if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
                    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(source)
            else:
                url = source
                if username and password and "://" in url:
                    proto, rest = url.split("://", 1)
                    url = f"{proto}://{username}:{password}@{rest}"
                cap = cv2.VideoCapture(url)
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        except Exception as exc:
            if cap is not None:
                cap.release()
            raise CameraRuntimeError("无法打开摄像头", "CAMERA_OPEN_FAILED") from exc

    def _save_image(self, device_id, image_bytes):
        photo_dir = os.path.join(self._image_storage_dir, device_id)
        os.makedirs(photo_dir, exist_ok=True)
        filepath = os.path.join(photo_dir, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
        with open(filepath, "wb") as output:
            output.write(image_bytes)
        return filepath
