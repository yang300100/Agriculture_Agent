"""HTTP REST 设备驱动 — 通过 HTTP API 控制智能设备

适用场景:
- 智能插座 (如 Tasmota/ESPHome)
- REST API 设备 (如树莓派 GPIO 控制器)
- 任何支持 HTTP 控制的 IoT 设备

设备端要求:
- POST {base_url}/command  接收: {"command": "start", "params": {...}}
                             返回: {"success": true, "message": "..."}
- GET  {base_url}/state    返回: {"power": false, "temperature": 22.5, ...}
"""

import asyncio
import functools
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import requests

from .base import (
    BaseDeviceDriver, DeviceCapability, DeviceStatus,
    DeviceInfo, DeviceCommand, DeviceResult,
)

logger = logging.getLogger(__name__)


class HTTPDriver(BaseDeviceDriver):
    """HTTP REST 设备驱动 — 通过 HTTP API 控制设备

    使用方式:
        driver = HTTPDriver()
        driver.register_device("smart_plug_01", "智能插座#1", [...],
                              base_url="http://192.168.1.101:8080")
        await driver.connect()
        result = await driver.execute("smart_plug_01", DeviceCommand("start", {"duration": 30}))
    """

    driver_name = "http"

    def __init__(self, request_timeout: int = 10):
        self._timeout = request_timeout
        self._connected = False
        self._devices: Dict[str, Dict] = {}

    # ── 设备注册 ──────────────────────────────

    def register_device(self, device_id: str, name: str,
                        capabilities: List[DeviceCapability],
                        sensors: List[str] = None,
                        location: str = "",
                        base_url: str = "",
                        api_key: str = None) -> None:
        """注册一个 HTTP 设备

        Args:
            device_id: 设备唯一标识
            name: 设备名称
            capabilities: 设备能力列表
            sensors: 传感器字段列表
            location: 物理位置
            base_url: 设备 HTTP 地址 (如 http://192.168.1.101:8080)
            api_key: API 密钥（可选，会作为 Bearer token 或 X-API-Key 发送）
        """
        self._devices[device_id] = {
            "info": {
                "device_id": device_id,
                "name": name,
                "capabilities": capabilities,
                "sensors": sensors or [],
                "location": location,
                "base_url": base_url.rstrip("/") if base_url else "",
                "api_key": api_key,
            },
            "state": {"power": False, "status": "idle"},
        }
        logger.info("HTTP 设备已注册: %s → %s", device_id, base_url)

    # ── 生命周期 ──────────────────────────────

    async def connect(self) -> bool:
        """验证所有设备可达"""
        self._connected = True
        online_count = 0
        for dev_id, dev in self._devices.items():
            base_url = dev["info"].get("base_url")
            if not base_url:
                continue
            try:
                headers = self._make_headers(dev["info"])
                resp = await self._async_request("GET", f"{base_url}/state",
                                                 headers=headers, timeout=self._timeout)
                if resp.status_code == 200:
                    online_count += 1
                    dev["state"] = resp.json()
                else:
                    logger.warning("HTTP 设备 %s 返回 %d", dev_id, resp.status_code)
            except requests.ConnectionError:
                logger.warning("HTTP 设备 %s 不可达: %s", dev_id, base_url)
            except Exception as e:
                logger.warning("HTTP 设备 %s 连接异常: %s", dev_id, e)

        logger.info("HTTPDriver: 已连接 (%d/%d 设备在线)", online_count, len(self._devices))
        return True

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("HTTPDriver: 已断开")

    async def health_check(self) -> bool:
        return self._connected

    # ── 设备发现 ──────────────────────────────

    async def discover(self) -> List[DeviceInfo]:
        """返回所有注册的设备"""
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
                    "protocol": "http",
                    "base_url": info.get("base_url"),
                },
            ))
        return result

    # ── 指令执行 ──────────────────────────────

    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        """通过 HTTP POST 发送控制指令"""
        if device_id not in self._devices:
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message=f"设备 '{device_id}' 未注册",
                error_code="DEVICE_NOT_FOUND",
            )

        dev = self._devices[device_id]
        base_url = dev["info"].get("base_url")
        if not base_url:
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message="设备未配置 base_url",
                error_code="NO_URL",
            )

        try:
            headers = self._make_headers(dev["info"])
            payload = {
                "command": command.command,
                "params": command.params,
                "timestamp": datetime.now().isoformat(),
                "device_id": device_id,
            }

            resp = await self._async_request("POST", f"{base_url}/command",
                                              json=payload, headers=headers, timeout=self._timeout)

            if resp.status_code == 200:
                resp_data = resp.json() if resp.text else {}
                if command.command == "start":
                    dev["state"]["power"] = True
                    dev["state"]["status"] = "running"
                elif command.command == "stop":
                    dev["state"]["power"] = False
                    dev["state"]["status"] = "idle"

                return DeviceResult(
                    success=resp_data.get("success", True),
                    device_id=device_id,
                    executed_command=command.command,
                    actual_params=command.params,
                    message=resp_data.get("message", f"[HTTP] 指令已发送到 {base_url}"),
                    raw_response=resp_data,
                )
            else:
                dev["state"]["status"] = "error"
                return DeviceResult(
                    success=False, device_id=device_id,
                    executed_command=command.command,
                    message=f"HTTP {resp.status_code}: {resp.text[:100]}",
                    error_code=f"HTTP_{resp.status_code}",
                )

        except requests.ConnectionError:
            dev["state"]["status"] = "error"
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message=f"设备不可达: {base_url}",
                error_code="CONNECTION_REFUSED",
            )
        except requests.Timeout:
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message=f"设备超时: {base_url}",
                error_code="TIMEOUT",
            )
        except Exception as e:
            logger.error("HTTP 执行失败: %s → %s", device_id, e)
            dev["state"]["status"] = "error"
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message=str(e),
                error_code="HTTP_ERROR",
            )

    # ── 状态读取 ──────────────────────────────

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        """通过 HTTP GET 读取设备状态"""
        if device_id not in self._devices:
            return {"error": f"设备 '{device_id}' 不存在"}

        dev = self._devices[device_id]
        base_url = dev["info"].get("base_url")
        if not base_url:
            return {**dev["state"], "_read_at": datetime.now().isoformat()}

        try:
            headers = self._make_headers(dev["info"])
            resp = await self._async_request("GET", f"{base_url}/state",
                                            headers=headers, timeout=self._timeout)
            if resp.status_code == 200:
                state = resp.json()
                # 合并远程状态
                dev["state"].update(state)
                state["_read_at"] = datetime.now().isoformat()
                state["_driver"] = "http"
                return state
            else:
                return {**dev["state"], "_error": f"HTTP {resp.status_code}",
                        "_read_at": datetime.now().isoformat()}
        except Exception as e:
            return {**dev["state"], "_error": str(e),
                    "_read_at": datetime.now().isoformat()}

    # ── 内部方法 ──────────────────────────────

    async def _async_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """在线程池中执行同步 HTTP 请求，避免阻塞事件循环"""
        method_upper = method.upper()
        req_func = getattr(requests, method_upper.lower(), None)
        if req_func is None:
            raise ValueError(f"不支持的 HTTP 方法: {method}")
        return await asyncio.to_thread(functools.partial(req_func, url, **kwargs))

    def _make_headers(self, info: Dict) -> Dict:
        """构建 HTTP 请求头"""
        headers = {"Content-Type": "application/json"}
        api_key = info.get("api_key")
        if api_key:
            # 尝试 Bearer token 格式，如果看起来不像则用 X-API-Key
            if api_key.startswith("sk-") or api_key.startswith("bearer"):
                headers["Authorization"] = f"Bearer {api_key}"
            else:
                headers["X-API-Key"] = api_key
        return headers
