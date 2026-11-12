"""哥哥，同一进程内的摄像头消费者共享一个采集线程和最新 JPEG。"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime


class CameraRuntimeError(RuntimeError):
    """携带稳定错误码，供抓拍结果和实时响应统一处理。"""

    def __init__(self, message, error_code="FRAME_READ_FAILED"):
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class CameraFrame:
    image_bytes: bytes
    width: int
    height: int
    observed_at: str
    fps: float
    sequence: int


def camera_key(camera_type, source, username="", password=""):
    if isinstance(source, bool) or not isinstance(source, (str, int)):
        raise ValueError("摄像头 source 必须是字符串或非负整数")
    if isinstance(source, int) and source < 0:
        raise ValueError("摄像头索引不能为负数")
    if isinstance(source, str):
        source = source.strip()
        if not source:
            raise ValueError("摄像头 source 不能为空")
        if camera_type == "usb" and source.lstrip("-").isdigit():
            source = int(source)
            if source < 0:
                raise ValueError("摄像头索引不能为负数")
    # 哥哥，USB 不使用认证，残留凭据不能把同一物理摄像头拆成两个句柄。
    if camera_type == "usb":
        return camera_type, source, "", ""
    return camera_type, source, username or "", password or ""


class _CaptureWorker:
    def __init__(self, opener, encoder):
        self.opener = opener
        self.encoder = encoder
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.frame = None
        self.error = None
        self.consumers = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera-capture")

    def _run(self):
        cap = None
        error_code = "CAMERA_OPEN_FAILED"
        try:
            cap = self.opener()
            if cap is None or not cap.isOpened():
                raise CameraRuntimeError("无法打开摄像头", error_code)
            sequence = 0
            while not self.stop_event.is_set():
                started = time.monotonic()
                error_code = "FRAME_READ_FAILED"
                ok, image = cap.read()
                if not ok or image is None:
                    raise CameraRuntimeError("无法从摄像头读取画面", error_code)
                height, width = image.shape[:2]
                error_code = "ENCODE_FAILED"
                data = self.encoder(image)
                if not data:
                    raise CameraRuntimeError("JPEG 编码失败", error_code)
                sequence += 1
                frame = CameraFrame(data, int(width), int(height), datetime.now().isoformat(), 10, sequence)
                with self.condition:
                    self.frame = frame
                    self.condition.notify_all()
                self.stop_event.wait(max(0, .1 - (time.monotonic() - started)))
        except Exception as exc:
            self.error = exc if isinstance(exc, CameraRuntimeError) else CameraRuntimeError(str(exc), error_code)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            with self.condition:
                self.condition.notify_all()


class CameraSubscription:
    def __init__(self, pool, key, worker):
        self._pool = pool
        self._key = key
        self._worker = worker
        self.closed = False

    def next_frame(self, after_sequence=0, timeout=5):
        worker = self._worker
        deadline = time.monotonic() + timeout
        with worker.condition:
            while True:
                if self.closed:
                    raise CameraRuntimeError("摄像头消费者已断开", "STREAM_CLOSED")
                if worker.error is not None:
                    raise CameraRuntimeError(str(worker.error), worker.error.error_code)
                if worker.frame is not None and worker.frame.sequence > after_sequence:
                    return worker.frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraRuntimeError("等待摄像头画面超时", "FRAME_TIMEOUT")
                worker.condition.wait(remaining)

    def close(self):
        self._pool.release(self)


class _CapturePool:
    def __init__(self):
        self._lock = threading.Lock()
        self._workers = {}

    def acquire(self, key, opener, encoder):
        with self._lock:
            worker = self._workers.get(key)
            if worker is not None and worker.stop_event.is_set():
                if worker.thread.is_alive():
                    raise CameraRuntimeError("摄像头正在释放，请稍后重试", "CAMERA_BUSY")
                self._workers.pop(key)
                worker = None
            if worker is None:
                worker = _CaptureWorker(opener, encoder)
                self._workers[key] = worker
                worker.thread.start()
            worker.consumers += 1
            return CameraSubscription(self, key, worker)

    def release(self, subscription):
        with self._lock:
            worker = subscription._worker
            with worker.condition:
                if subscription.closed:
                    return
                subscription.closed = True
                worker.consumers -= 1
                worker.condition.notify_all()
            if worker.consumers == 0:
                worker.stop_event.set()
                # 哥哥，关闭完成前保留池内所有权，防止另一请求抢先重复打开同一 USB。
                worker.thread.join(timeout=5)
                if not worker.thread.is_alive() and self._workers.get(subscription._key) is worker:
                    self._workers.pop(subscription._key)


CAPTURE_POOL = _CapturePool()
