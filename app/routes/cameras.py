"""按账号读取摄像头配置，提供持续 JPEG 视频预览。"""

import asyncio
import logging

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from core.device_registry_factory import load_custom_devices

logger = logging.getLogger(__name__)


def _multipart_frame(jpeg: bytes) -> bytes:
    return (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n"
    )


class CameraStreamResponse(StreamingResponse):
    """连接中断或发送失败时也回收已经打开的摄像头迭代器。"""

    def __init__(self, frames, first_frame):
        self.frames = frames
        self._pending_read = None
        self._cleanup_task = None

        async def content():
            yield _multipart_frame(first_frame)
            while True:
                # 读取任务由响应持有；取消 HTTP 请求不会遗弃仍在运行的 next()。
                self._pending_read = asyncio.create_task(asyncio.to_thread(next, frames, None))
                jpeg = await asyncio.shield(self._pending_read)
                self._pending_read = None
                if jpeg is None:
                    return
                yield _multipart_frame(jpeg)

        super().__init__(content(), media_type="multipart/x-mixed-replace; boundary=frame",
                         headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # 哥哥，先等有超时保护的读帧返回，再关闭生成器，避免跨线程 close/next 竞态。
            async def close_when_idle():
                if self._pending_read is not None:
                    try:
                        await self._pending_read
                    except Exception:
                        pass
                    self._pending_read = None
                await asyncio.to_thread(self.frames.close)

            self._cleanup_task = asyncio.create_task(close_when_idle())
            with anyio.CancelScope(shield=True):
                await asyncio.shield(self._cleanup_task)


def register_camera_routes(app: FastAPI):
    @app.get("/api/devices/{device_id}/stream")
    def stream_camera(device_id: str, username: str = "default"):
        config = next((row for row in load_custom_devices(username)
                       if row.get("device_id") == device_id), None)
        if config is None:
            raise HTTPException(404, "摄像头不存在")
        if config.get("driver") != "camera" or "capture" not in config.get("capabilities", []):
            raise HTTPException(409, "此设备未配置连续视频源，请使用 camera 驱动并填写摄像头来源")
        connection = config.get("connection", {})
        if connection.get("source") in (None, ""):
            raise HTTPException(409, "请先配置摄像头来源")
        try:
            from devices.base import DeviceCapability
            from devices.camera_driver import CameraDriver
            driver = CameraDriver(username=username)
            driver.register_device(
                device_id=device_id, name=config.get("name", device_id),
                capabilities=[DeviceCapability.CAPTURE],
                camera_type=connection.get("camera_type", "usb"),
                source=connection["source"],
                username=connection.get("username", ""),
                password=connection.get("password", ""),
            )
            frames = driver.stream_frames(device_id)
        except (ImportError, ValueError):
            raise HTTPException(503, "摄像头驱动不可用，请检查依赖与连接配置")
        try:
            first_frame = next(frames)
        except Exception:
            frames.close()
            logger.warning("摄像头视频源不可用: %s", device_id)
            raise HTTPException(503, "无法读取摄像头，请检查设备连接、系统权限及是否被其他程序占用")
        return CameraStreamResponse(frames, first_frame)
