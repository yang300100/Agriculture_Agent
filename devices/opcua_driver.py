"""OPC UA 设备驱动——用于 PLC、SCADA、工业网关和标准化节点数据。"""

import asyncio
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
    from asyncua import Client, ua

    HAS_ASYNCUA = True
except ImportError:
    HAS_ASYNCUA = False
    Client = ua = None
    logger.warning("asyncua 未安装，OPC UA 驱动不可用。安装: pip install asyncua")


class OPCUADriver(BaseDeviceDriver):
    """按配置的 OPC UA 节点读取状态并写入控制值。"""

    driver_name = "opcua"

    def __init__(self, request_timeout: float = 5.0):
        if not HAS_ASYNCUA:
            raise ImportError("asyncua 未安装。请运行: pip install asyncua")
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
        endpoint: str = "",
        command_nodes: Optional[Dict[str, Any]] = None,
        state_nodes: Optional[Dict[str, str]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        security_string: Optional[str] = None,
    ) -> None:
        endpoint = str(endpoint).strip()
        if not endpoint.startswith("opc.tcp://"):
            raise ValueError("OPC UA endpoint 必须以 opc.tcp:// 开头")
        if not isinstance(command_nodes or {}, dict) or not isinstance(state_nodes or {}, dict):
            raise TypeError("command_nodes 和 state_nodes 必须是 JSON 对象")
        self._devices[device_id] = {
            "info": {
                "device_id": device_id,
                "name": name,
                "capabilities": capabilities,
                "sensors": sensors or [],
                "location": location,
                "endpoint": endpoint,
                "command_nodes": command_nodes or {},
                "state_nodes": state_nodes or {},
                "username": username,
                "password": password,
                "security_string": security_string,
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
        # 客户端按请求创建和释放，避免缓存 Registry 跨事件循环复用连接。
        self._connected = False

    async def health_check(self) -> bool:
        return self._connected

    async def discover(self) -> List[DeviceInfo]:
        result = []
        for dev in self._devices.values():
            info = dev["info"]
            status = DeviceStatus.ONLINE if dev["reachable"] else DeviceStatus.OFFLINE
            if dev["state"].get("status") == "error":
                status = DeviceStatus.ERROR
            result.append(
                DeviceInfo(
                    device_id=info["device_id"],
                    name=info["name"],
                    driver_name=self.driver_name,
                    capabilities=info["capabilities"],
                    sensors=info["sensors"],
                    status=status,
                    location=info["location"],
                    metadata={
                        "protocol": "opcua",
                        "endpoint": info["endpoint"],
                        "state_nodes": list(info["state_nodes"].keys()),
                    },
                )
            )
        return result

    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        dev = self._devices.get(device_id)
        if dev is None:
            return self._failure(device_id, command.command, "设备未注册", "DEVICE_NOT_FOUND")
        info = dev["info"]
        node_config = info["command_nodes"].get(command.command)
        if node_config is None:
            return self._failure(
                device_id,
                command.command,
                f"未配置指令 '{command.command}' 对应的 OPC UA 节点",
                "UNSUPPORTED_COMMAND",
            )
        if isinstance(node_config, str):
            node_id = node_config
            configured_value = True
            variant_type = None
        elif isinstance(node_config, dict):
            node_id = node_config.get("node_id")
            configured_value = node_config.get("value", True)
            variant_type = node_config.get("variant_type")
        else:
            return self._failure(
                device_id, command.command,
                "OPC UA 指令节点配置格式错误", "INVALID_CONFIG",
            )
        if not node_id:
            return self._failure(
                device_id, command.command,
                "OPC UA 指令节点缺少 node_id", "INVALID_CONFIG",
            )
        value = command.params.get("value", configured_value)
        client = await self._make_client(info)
        try:
            await asyncio.wait_for(client.connect(), timeout=self._command_timeout(command))
            node = client.get_node(node_id)
            write_value = self._to_variant(value, variant_type)
            await asyncio.wait_for(
                node.write_value(write_value), timeout=self._command_timeout(command)
            )
            dev["reachable"] = True
            return DeviceResult(
                success=True,
                device_id=device_id,
                executed_command=command.command,
                actual_params={**command.params, "value": value},
                message=f"OPC UA 节点 {node_id} 写入成功",
                raw_response={"node_id": node_id, "value": value},
            )
        except asyncio.TimeoutError:
            dev["reachable"] = False
            return self._failure(device_id, command.command, "OPC UA 请求超时", "TIMEOUT")
        except Exception as exc:
            dev["reachable"] = False
            logger.warning("OPC UA 指令失败 %s: %s", device_id, exc)
            return self._failure(device_id, command.command, str(exc), "OPCUA_ERROR")
        finally:
            await self._safe_disconnect(client)

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        dev = self._devices.get(device_id)
        if dev is None:
            return {"error": f"设备 '{device_id}' 不存在"}
        info = dev["info"]
        client = await self._make_client(info)
        try:
            await asyncio.wait_for(client.connect(), timeout=self._timeout)
            state = {}
            for field, node_id in info["state_nodes"].items():
                node = client.get_node(node_id)
                state[field] = await asyncio.wait_for(
                    node.read_value(), timeout=self._timeout
                )
            dev["state"].update(state)
            dev["reachable"] = True
            return self._state_view(dev)
        except Exception as exc:
            dev["reachable"] = False
            return {**self._state_view(dev), "_error": str(exc)}
        finally:
            await self._safe_disconnect(client)

    async def _probe(self, device_id: str) -> bool:
        dev = self._devices[device_id]
        client = await self._make_client(dev["info"])
        try:
            await asyncio.wait_for(client.connect(), timeout=self._timeout)
            dev["reachable"] = True
            return True
        except Exception:
            dev["reachable"] = False
            return False
        finally:
            await self._safe_disconnect(client)

    async def _make_client(self, info: Dict[str, Any]):
        client = Client(url=info["endpoint"], timeout=self._timeout)
        if info.get("security_string"):
            await client.set_security_string(info["security_string"])
        if info.get("username"):
            client.set_user(info["username"])
            client.set_password(info.get("password") or "")
        return client

    @staticmethod
    async def _safe_disconnect(client) -> None:
        try:
            await client.disconnect()
        except Exception:
            pass

    @staticmethod
    def _to_variant(value: Any, variant_type: Optional[str]):
        if not variant_type:
            return value
        try:
            ua_type = getattr(ua.VariantType, variant_type)
        except AttributeError as exc:
            raise ValueError(f"未知 OPC UA VariantType: {variant_type}") from exc
        return ua.Variant(value, ua_type)

    def _state_view(self, dev: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **dev["state"],
            "_driver": self.driver_name,
            "_read_at": datetime.now().isoformat(),
        }

    @staticmethod
    def _command_timeout(command: DeviceCommand) -> float:
        return max(float(command.timeout_ms) / 1000.0, 0.001)

    @staticmethod
    def _failure(device_id: str, command: str, message: str, code: str) -> DeviceResult:
        return DeviceResult(
            success=False,
            device_id=device_id,
            executed_command=command,
            message=message,
            error_code=code,
        )
