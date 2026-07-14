"""摄像头设备驱动 — 支持 USB / IP(RTSP) / ESP32-CAM，实现定时拍照与状态监测

依赖: pip install opencv-python

使用方式:
    driver = CameraDriver()
    driver.register_device("cam_01", "大棚摄像头#1", [...], camera_type="usb", source="0")
    await driver.connect()
    result = await driver.execute("cam_01", DeviceCommand("capture"))
"""

import asyncio
import base64
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base import (
    BaseDeviceDriver, DeviceCapability, DeviceStatus,
    DeviceInfo, DeviceCommand, DeviceResult,
)

logger = logging.getLogger(__name__)

# opencv-python 是可选依赖
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None
    np = None
    logger.warning("opencv-python 未安装，摄像头驱动不可用。安装: pip install opencv-python")

# requests 用于 ESP32-CAM MJPEG 流拉取
try:
    import requests as req_lib
except ImportError:
    req_lib = None


class CameraType:
    """摄像头类型常量"""
    USB = "usb"
    IP = "ip"
    ESP32_CAM = "esp32cam"


class CameraDriver(BaseDeviceDriver):
    """摄像头设备驱动 — 支持 USB / IP(RTSP) / ESP32-CAM

    三种摄像头类型:
    - USB:  直接通过 OpenCV 读取本地摄像头（索引 0/1/2 或路径 /dev/video0）
    - IP:    通过 RTSP/HTTP 流地址连接网络摄像头
    - ESP32-CAM: 通过 HTTP MJPEG 流读取低成本 WiFi 摄像头

    核心操作:
    - capture:  抓取单帧 JPEG 图像，存入指定目录
    """

    driver_name = "camera"

    def __init__(self, image_storage_dir: str = None):
        if not HAS_OPENCV:
            raise ImportError("opencv-python 未安装。请运行: pip install opencv-python")

        self._connected = False
        self._streaming: Dict[str, bool] = {}       # device_id → 是否正在推流
        self._caps: Dict[str, Any] = {}             # device_id → cv2.VideoCapture（仅拍摄时临时持有）
        self._devices: Dict[str, Dict] = {}          # device_id → {info, state}
        self._last_capture: Dict[str, str] = {}      # device_id → ISO timestamp

        # 照片存储目录
        self._image_storage_dir = image_storage_dir or os.path.join("data", "photos")
        os.makedirs(self._image_storage_dir, exist_ok=True)

    # ── 设备注册 ──────────────────────────────

    def register_device(self, device_id: str, name: str,
                        capabilities: List[DeviceCapability],
                        sensors: List[str] = None,
                        location: str = "",
                        camera_type: str = "usb",
                        source: str = "0",
                        username: str = "",
                        password: str = "") -> None:
        """注册一个摄像头设备

        Args:
            device_id:   设备唯一标识
            name:        设备名称
            capabilities: 设备能力列表（应包含 DeviceCapability.CAPTURE）
            sensors:     传感器字段列表（摄像头通常无传感器）
            location:    物理位置
            camera_type: 摄像头类型 "usb" | "ip" | "esp32cam"
            source:      设备源（USB: 索引号或路径，IP: RTSP URL，ESP32-CAM: HTTP URL）
            username:    认证用户名（可选）
            password:    认证密码（可选）
        """
        self._devices[device_id] = {
            "info": {
                "device_id": device_id,
                "name": name,
                "capabilities": capabilities,
                "sensors": sensors or [],
                "location": location,
                "camera_type": camera_type,
                "source": source,
                "username": username,
                "password": password,
            },
            "state": {
                "power": False,
                "status": "idle",
                "last_capture_time": None,
                "resolution": "unknown",
                "fps": 0,
                "camera_type": camera_type,
                "source": source,
            },
        }
        self._streaming[device_id] = False
        # 掩码密码后再记录日志
        masked_source = source
        if password:
            masked_source = source.replace(password, "***")
        logger.info("摄像头设备已注册: %s (%s → %s)", device_id, camera_type, masked_source)

    # ── 生命周期 ──────────────────────────────

    async def connect(self) -> bool:
        """打开每个摄像头验证可达性，验证后立即释放

        注：不保持长连接，避免占用摄像头资源。
        每次 capture 时临时打开→拍摄→释放。
        """
        if self._connected:
            return True

        async def _verify_one(dev_id: str, dev: dict) -> bool:
            """验证单个摄像头的协程"""
            info = dev["info"]
            try:
                cap = await asyncio.to_thread(
                    self._open_camera, info["camera_type"], info["source"],
                    info.get("username", ""), info.get("password", "")
                )
                if cap is not None and cap.isOpened():
                    # 读取一帧确认可用
                    ret, frame = await asyncio.to_thread(cap.read)
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        dev["state"]["resolution"] = f"{w}x{h}"
                        dev["state"]["fps"] = int(cap.get(cv2.CAP_PROP_FPS) or 0)
                        dev["state"]["status"] = "idle"
                        logger.info("摄像头 %s 验证成功: %s %sx%s",
                                     dev_id, info["camera_type"], w, h)
                        cap.release()
                        return True
                    else:
                        dev["state"]["status"] = "error"
                        dev["state"]["error_reason"] = "无法读取画面"
                        logger.warning("摄像头 %s 无法读取画面", dev_id)
                    cap.release()
                else:
                    dev["state"]["status"] = "error"
                    dev["state"]["error_reason"] = f"无法打开: {info['source']}"
                    logger.warning("摄像头 %s 无法打开: %s", dev_id, info["source"])
            except Exception as e:
                dev["state"]["status"] = "error"
                dev["state"]["error_reason"] = str(e)[:200]
                logger.warning("摄像头 %s 验证异常: %s", dev_id, e)
            return False

        # 并行验证所有摄像头
        results = await asyncio.gather(*[
            _verify_one(dev_id, dev) for dev_id, dev in self._devices.items()
        ])
        success_count = sum(1 for r in results if r)

        if success_count == 0:
            self._connected = False
            logger.warning("CameraDriver: 验证完成，无可用摄像头 (%d/%d)", success_count, len(self._devices))
            return False

        self._connected = True
        logger.info("CameraDriver: 验证完成 (%d/%d 可用)", success_count, len(self._devices))
        return True

    async def disconnect(self) -> None:
        """释放所有打开的摄像头"""
        for dev_id in list(self._caps.keys()):
            cap = self._caps.pop(dev_id, None)
            if cap is not None:
                try:
                    await asyncio.to_thread(cap.release)
                except Exception:
                    pass
        self._connected = False
        self._streaming = {k: False for k in self._streaming}
        logger.info("CameraDriver: 已断开")

    async def health_check(self) -> bool:
        return self._connected

    # ── 设备发现 ──────────────────────────────

    async def discover(self) -> List[DeviceInfo]:
        """返回所有注册的摄像头设备"""
        result = []
        for dev_id, dev in self._devices.items():
            info = dev["info"]
            state = dev["state"]
            status = DeviceStatus.ONLINE if self._connected else DeviceStatus.OFFLINE
            if state.get("status") == "error":
                status = DeviceStatus.ERROR

            result.append(DeviceInfo(
                device_id=info["device_id"],
                name=info["name"],
                driver_name=self.driver_name,
                capabilities=info["capabilities"],
                sensors=info["sensors"],
                status=status,
                location=info.get("location", ""),
                metadata={
                    "camera_type": info.get("camera_type"),
                    "source": info.get("source"),
                },
            ))
        return result

    # ── 指令执行 ──────────────────────────────

    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        """执行摄像头指令

        支持子命令:
        - capture:      抓取单帧 JPEG → 返回 image_bytes 在 raw_response 中
        - start_stream: 开启持续推流（暂未实现，预留）
        - stop_stream:  停止推流
        """
        if device_id not in self._devices:
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message=f"设备 '{device_id}' 未注册",
                error_code="DEVICE_NOT_FOUND",
            )

        if command.command == "capture":
            return await self._handle_capture(device_id, command)
        elif command.command == "start_stream":
            return await self._handle_start_stream(device_id)
        elif command.command == "stop_stream":
            return await self._handle_stop_stream(device_id)
        else:
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message=f"摄像头不支持指令: {command.command}（支持: capture, start_stream, stop_stream）",
                error_code="UNSUPPORTED_COMMAND",
            )

    async def _handle_capture(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        """抓取单帧 JPEG 图像"""
        dev = self._devices[device_id]
        info = dev["info"]

        try:
            # 打开摄像头 → 读取一帧 → 编码 JPEG → 释放
            cap = await asyncio.to_thread(
                self._open_camera, info["camera_type"], info["source"],
                info.get("username", ""), info.get("password", "")
            )
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                return DeviceResult(
                    success=False, device_id=device_id,
                    executed_command="capture",
                    message=f"无法打开摄像头: {info['source']}",
                    error_code="CAMERA_OPEN_FAILED",
                )

            ret, frame = await asyncio.to_thread(cap.read)
            if not ret or frame is None:
                cap.release()
                return DeviceResult(
                    success=False, device_id=device_id,
                    executed_command="capture",
                    message="无法从摄像头读取画面",
                    error_code="FRAME_READ_FAILED",
                )

            # 编码为 JPEG
            retval, jpeg_bytes = await asyncio.to_thread(
                cv2.imencode, ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            cap.release()

            if not retval or jpeg_bytes is None:
                return DeviceResult(
                    success=False, device_id=device_id,
                    executed_command="capture",
                    message="JPEG 编码失败",
                    error_code="ENCODE_FAILED",
                )

            image_bytes = jpeg_bytes.tobytes()
            h, w = frame.shape[:2]
            timestamp = datetime.now().isoformat()

            # 更新状态
            dev["state"]["last_capture_time"] = timestamp
            dev["state"]["resolution"] = f"{w}x{h}"
            dev["state"]["status"] = "idle"
            self._last_capture[device_id] = timestamp

            logger.info("摄像头 %s 抓拍成功: %sx%s, %d bytes",
                         device_id, w, h, len(image_bytes))

            # 持久化图片到磁盘
            saved_path = self._save_image(device_id, image_bytes)
            logger.info("摄像头 %s 图片已保存: %s", device_id, saved_path)

            return DeviceResult(
                success=True,
                device_id=device_id,
                executed_command="capture",
                message=f"抓拍成功 ({w}x{h}, {len(image_bytes)} bytes)",
                raw_response={
                    "image_bytes": image_bytes,
                    "metadata": {
                        "width": w, "height": h,
                        "size_bytes": len(image_bytes),
                        "timestamp": timestamp,
                        "camera_type": info["camera_type"],
                        "source": info["source"],
                    },
                },
            )

        except Exception as e:
            dev["state"]["status"] = "error"
            dev["state"]["error_reason"] = str(e)[:200]
            logger.error("摄像头 %s 抓拍失败: %s", device_id, e)
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command="capture",
                message=f"抓拍失败: {e}",
                error_code="CAPTURE_ERROR",
            )

    async def _handle_start_stream(self, device_id: str) -> DeviceResult:
        """开启持续推流（预留接口）"""
        return DeviceResult(
            success=False, device_id=device_id,
            executed_command="start_stream",
            message="持续推流功能暂未实现，请使用 capture 单帧抓拍",
            error_code="NOT_IMPLEMENTED",
        )

    async def _handle_stop_stream(self, device_id: str) -> DeviceResult:
        """停止推流"""
        if device_id not in self._devices:
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command="stop_stream",
                message=f"设备 '{device_id}' 未注册",
                error_code="DEVICE_NOT_FOUND",
            )
        self._streaming[device_id] = False
        return DeviceResult(
            success=True, device_id=device_id,
            executed_command="stop_stream",
            message="推流已停止",
        )

    # ── 状态读取 ──────────────────────────────

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        """读取摄像头当前状态"""
        if device_id not in self._devices:
            return {"error": f"设备 '{device_id}' 不存在"}

        state = dict(self._devices[device_id]["state"])
        state["_read_at"] = datetime.now().isoformat()
        state["_driver"] = "camera"
        state["_streaming"] = self._streaming.get(device_id, False)
        return state

    # ── 内部方法 ──────────────────────────────

    def _open_camera(self, camera_type: str, source: str,
                     username: str = "", password: str = "") -> Optional[Any]:
        """根据摄像头类型打开 cv2.VideoCapture

        Returns:
            cv2.VideoCapture 实例，失败返回 None
        """
        try:
            if camera_type == CameraType.USB:
                # USB 摄像头: source 是索引号(0/1/2) 或路径(/dev/video0)
                idx = int(source) if source.isdigit() else source
                cap = cv2.VideoCapture(idx)
                # 设置缓冲区大小，减少延迟
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap

            elif camera_type == CameraType.IP:
                # IP 摄像头: source 是 RTSP/HTTP 流地址
                # 如果有认证信息，构建带认证的 URL
                url = source
                if username and password:
                    # 尝试将认证嵌入 URL: rtsp://user:pass@host/path
                    if "://" in url:
                        proto, rest = url.split("://", 1)
                        url = f"{proto}://{username}:{password}@{rest}"
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # 设置超时（OpenCV 4.x+）
                if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                return cap

            elif camera_type == CameraType.ESP32_CAM:
                # ESP32-CAM: source 是 HTTP MJPEG 流 URL
                # 优先尝试 OpenCV 直接打开（支持 MJPEG 流）
                url = source
                if username and password:
                    if "://" in url:
                        proto, rest = url.split("://", 1)
                        url = f"{proto}://{username}:{password}@{rest}"
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap

            else:
                logger.error("未知摄像头类型: %s", camera_type)
                return None

        except Exception as e:
            logger.error("打开摄像头失败 (%s, %s): %s", camera_type, source, e)
            return None

    def _save_image(self, device_id: str, image_bytes: bytes,
                    username: str = "default") -> str:
        """保存照片到磁盘

        Returns:
            保存的文件路径
        """
        photo_dir = os.path.join(self._image_storage_dir, username, device_id)
        os.makedirs(photo_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(photo_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        logger.info("照片已保存: %s (%d bytes)", filepath, len(image_bytes))
        return filepath
