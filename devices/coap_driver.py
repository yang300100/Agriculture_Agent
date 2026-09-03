"""CoAP 设备驱动——适用于低功耗传感器、边缘节点和受限网络设备。"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import (
    BaseDeviceDriver,
    DeviceCapability,
    DeviceCommand,
    DeviceInfo,
    DeviceResult,
    DeviceStatus,
)

logger = logging.getLogger(__name__)

try:
    from aiocoap import Context, GET, POST, Message

    HAS_AIOCOAP = True
except ImportError:
    HAS_AIOCOAP = False
    Context = GET = POST = Message = None
    logger.warning("aiocoap 未安装，CoAP 驱动不可用。安装: pip install aiocoap")


class CoAPDriver(BaseDeviceDriver):
    """通过 CoAP GET/POST 读取状态并下发 JSON 指令。"""

    driver_name = "coap"

    def __init__(self, request_timeout: float = 5.0):
        if not HAS_AIOCOAP:
            raise ImportError("aiocoap 未安装。请运行: pip install aiocoap")
        self._timeout = float(request_timeout)
        self._connected = False
        self._devices: Dict[str, Dict[str, Any]] = {}

    def register_device(
        self,
        device_id: str,
        name: str,
        capabilities: List[DeviceCapability],
        sensors: Optional[List[str]] = None,
        location: str = "",
        base_uri: str = "",
        command_path: str = "/command",
        state_path: str = "/state",
        auth_token: Optional[str] = None,
    ) -> None:
        base_uri = str(base_uri).rstrip("/")
        if not base_uri.startswith(("coap://", "coaps://")):
            raise ValueError("CoAP base_uri 必须以 coap:// 或 coaps:// 开头")
        self._devices[device_id] = {
            "info": {
                "device_id": device_id,
                "name": name,
                "capabilities": capabilities,
                "sensors": sensors or [],
                "location": location,
                "base_uri": base_uri,
                "command_path": self._normalize_path(command_path),
                "state_path": self._normalize_path(state_path),
                "auth_token": auth_token,
            },
            "state": {"power": False, "status": "powered_off"},
            "reachable": False,
        }

    async def connect(self) -> bool:
        if not self._devices:
            self._connected = True
            return True
        results = await asyncio.gather(
            *(self._probe(device_id) for device_id in self._devices),
            return_exceptions=True,
        )
        self._connected = any(result is True for result in results)
        return self._connected

    async def disconnect(self) -> None:
        # 每次请求都在同一事件循环内创建并关闭 Context，不保留跨线程资源。
        self._connected = False

    async def health_check(self) -> bool:
        return self._connected

    async def discover(self) -> List[DeviceInfo]:
        devices = []
        for dev in self._devices.values():
            info = dev["info"]
            status = DeviceStatus.ONLINE if dev["reachable"] else DeviceStatus.OFFLINE
            if dev["state"].get("status") == "error":
                status = DeviceStatus.ERROR
            devices.append(
                DeviceInfo(
                    device_id=info["device_id"],
                    name=info["name"],
                    driver_name=self.driver_name,
                    capabilities=info["capabilities"],
                    sensors=info["sensors"],
                    status=status,
                    location=info["location"],
                    metadata={
                        "protocol": "coap",
                        "base_uri": info["base_uri"],
                        "state_path": info["state_path"],
                    },
                )
            )
        return devices

    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        dev = self._devices.get(device_id)
        if dev is None:
            return self._failure(device_id, command.command, "设备未注册", "DEVICE_NOT_FOUND")

        info = dev["info"]
        payload = {
            "device_id": device_id,
            "command": command.command,
            "params": command.params,
            "timestamp": datetime.now().isoformat(),
        }
        if info.get("auth_token"):
            payload["auth_token"] = info["auth_token"]
        timeout = max(float(command.timeout_ms) / 1000.0, 0.001)
        try:
            response = await self._request(
                POST,
                self._uri(info["base_uri"], info["command_path"]),
                payload,
                timeout,
            )
            if not self._is_success(response):
                dev["reachable"] = False
                return self._failure(
                    device_id,
                    command.command,
                    f"CoAP 设备拒绝指令: {response.code}",
                    "COAP_ERROR",
                )
            body = self._decode_json(response.payload)
            if not isinstance(body, dict):
                return self._failure(
                    device_id, command.command,
                    "CoAP 响应必须是 JSON 对象", "INVALID_RESPONSE",
                )
            dev["reachable"] = True
            success = bool(body.get("success", True))
            if success and isinstance(body.get("state"), dict):
                dev["state"].update(body["state"])
            return DeviceResult(
                success=success,
                device_id=device_id,
                executed_command=command.command,
                actual_params=command.params,
                message=body.get("message", "CoAP 指令已执行"),
                error_code=None if success else body.get("error_code", "DEVICE_REJECTED"),
                raw_response=body,
            )
        except asyncio.TimeoutError:
            dev["reachable"] = False
            return self._failure(device_id, command.command, "CoAP 请求超时", "TIMEOUT")
        except Exception as exc:
            dev["reachable"] = False
            logger.warning("CoAP 指令失败 %s: %s", device_id, exc)
            return self._failure(device_id, command.command, str(exc), "COAP_ERROR")

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        dev = self._devices.get(device_id)
        if dev is None:
            return {"error": f"设备 '{device_id}' 不存在"}
        info = dev["info"]
        try:
            response = await self._request(
                GET,
                self._uri(info["base_uri"], info["state_path"]),
                None,
                self._timeout,
            )
            if not self._is_success(response):
                raise RuntimeError(f"CoAP 状态码: {response.code}")
            state = self._decode_json(response.payload)
            if not isinstance(state, dict):
                raise ValueError("CoAP 状态响应必须是 JSON 对象")
            dev["state"].update(state)
            dev["reachable"] = True
            return self._state_view(dev)
        except Exception as exc:
            dev["reachable"] = False
            return {**self._state_view(dev), "_error": str(exc)}

    async def _probe(self, device_id: str) -> bool:
        state = await self.read_state(device_id)
        return "_error" not in state and "error" not in state

    async def _request(self, code, uri: str, payload: Optional[Dict], timeout: float):
        context = await Context.create_client_context()
        try:
            data = b"" if payload is None else json.dumps(
                payload, ensure_ascii=False
            ).encode("utf-8")
            request = Message(code=code, uri=uri, payload=data, content_format=50)
            return await asyncio.wait_for(context.request(request).response, timeout=timeout)
        finally:
            await context.shutdown()

    def _state_view(self, dev: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **dev["state"],
            "_driver": self.driver_name,
            "_read_at": datetime.now().isoformat(),
        }

    @staticmethod
    def _normalize_path(path: str) -> str:
        return "/" + str(path or "").lstrip("/")

    @staticmethod
    def _uri(base_uri: str, path: str) -> str:
        return f"{base_uri}{path}"

    @staticmethod
    def _decode_json(payload: bytes):
        if not payload:
            return {}
        return json.loads(payload.decode("utf-8"))

    @staticmethod
    def _is_success(response) -> bool:
        checker = getattr(response.code, "is_successful", None)
        return bool(checker()) if callable(checker) else str(response.code).startswith("2.")

    @staticmethod
    def _failure(device_id: str, command: str, message: str, code: str) -> DeviceResult:
        return DeviceResult(
            success=False,
            device_id=device_id,
            executed_command=command,
            message=message,
            error_code=code,
        )
