"""Modbus 设备驱动 — 通过 Modbus RTU/TCP 协议与工业设备通信

依赖: pip install pymodbus

使用方式:
    driver = ModbusDriver(mode="rtu", port="/dev/ttyUSB0", baudrate=9600)
    driver.register_device("plc_01", "PLC控制器#1", [...], slave_id=1)
    await driver.connect()
    result = await driver.execute("plc_01", DeviceCommand("start", {"duration": 30}))
"""

import asyncio
import inspect
import logging
from datetime import datetime
from typing import Any, Dict, List

from .base import (
    BaseDeviceDriver, DeviceCapability, DeviceStatus,
    DeviceInfo, DeviceCommand, DeviceResult,
)

logger = logging.getLogger(__name__)

# pymodbus 是可选依赖
try:
    from pymodbus.client import ModbusSerialClient, ModbusTcpClient
    HAS_PYMODBUS = True
except ImportError:
    HAS_PYMODBUS = False
    logger.warning("pymodbus 未安装，Modbus 驱动不可用。安装: pip install pymodbus")


class ModbusDriver(BaseDeviceDriver):
    """Modbus 设备驱动 — 通过 Modbus RTU 或 TCP 控制工业设备

    连接方式:
    - RTU 模式: 通过串口 (如 /dev/ttyUSB0, COM3) 连接 RS-485 总线
    - TCP 模式: 通过以太网连接 Modbus TCP 网关

    设备端要求:
    - 支持 Modbus RTU 或 TCP 协议
    - 保持寄存器 (Holding Registers) 用于控制和状态读写
      HR[0] = 电源开关 (写 1 启动, 写 0 停止)
      HR[1] = 运行状态 (0=空闲, 1=运行中, 2=故障)
    """

    driver_name = "modbus"

    def __init__(self, mode: str = "rtu", port: str = "/dev/ttyUSB0",
                 baudrate: int = 9600, timeout: float = 2.0):
        if not HAS_PYMODBUS:
            raise ImportError("pymodbus 未安装。请运行: pip install pymodbus")

        mode = str(mode).lower()
        if mode not in ("rtu", "tcp"):
            raise ValueError(f"Modbus mode 必须为 'rtu' 或 'tcp'，收到: {mode!r}")
        self._mode = mode
        self._port = str(port)
        self._baudrate = baudrate
        self._timeout = timeout
        self._client = None
        self._connected = False
        self._devices: Dict[str, Dict] = {}

    # ── 设备注册 ──────────────────────────────

    def register_device(self, device_id: str, name: str,
                        capabilities: List[DeviceCapability],
                        sensors: List[str] = None,
                        location: str = "",
                        slave_id: int = 1) -> None:
        """注册一个 Modbus 从设备

        Args:
            device_id: 设备唯一标识
            name: 设备名称
            capabilities: 设备能力列表
            sensors: 传感器字段列表
            location: 物理位置
            slave_id: Modbus 从站地址 (1-247)
        """
        # 校验 slave_id 范围：Modbus 标准规定 1-247
        slave_id = int(slave_id)
        if not (1 <= slave_id <= 247):
            raise ValueError(f"Modbus slave_id 必须在 1-247 范围内，收到: {slave_id}")

        self._devices[device_id] = {
            "info": {
                "device_id": device_id,
                "name": name,
                "capabilities": capabilities,
                "sensors": sensors or [],
                "location": location,
                "slave_id": slave_id,
            },
            "state": {"power": False, "status": "idle"},
        }
        logger.info("Modbus 设备已注册: %s (从站 %d)", device_id, slave_id)

    # ── 生命周期 ──────────────────────────────

    async def connect(self) -> bool:
        """连接 Modbus 总线"""
        # 关闭旧客户端，避免资源泄漏
        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception:
                pass
            self._client = None
        self._connected = False

        try:
            if self._mode == "tcp":
                # 解析 host:port，正确处理 IPv6 地址: [::1]:502, ::1:502, 127.0.0.1:502
                if ":" in self._port:
                    # 检测方括号包裹的 IPv6（如 [::1]:5020）
                    if self._port.startswith("[") and "]" in self._port:
                        bracket_end = self._port.index("]")
                        host = self._port[1:bracket_end]
                        port_str = self._port[bracket_end + 1:].lstrip(":") or "502"
                    else:
                        # 尝试判断是 IPv6 还是 IPv4:port
                        # IPv6 含多个冒号，IPv4 仅一个冒号分隔 host:port
                        colon_count = self._port.count(":")
                        if colon_count > 1:
                            # IPv6 地址（无方括号），从最后一个冒号分割
                            host, port_str = self._port.rsplit(":", 1)
                        else:
                            # IPv4:port 或 host:port
                            host, port_str = self._port.rsplit(":", 1)
                else:
                    host, port_str = self._port, "502"
                self._client = ModbusTcpClient(host=host, port=int(port_str), timeout=self._timeout)
            else:
                self._client = ModbusSerialClient(
                    port=self._port,
                    baudrate=self._baudrate,
                    timeout=self._timeout,
                )
            # 用 try/except 包裹 connect() 避免未定义的 _connected 状态
            try:
                self._connected = await asyncio.to_thread(self._client.connect)
            except Exception as conn_err:
                logger.error("Modbus connect() 调用失败: %s", conn_err)
                self._connected = False
            if self._connected:
                logger.info("ModbusDriver: 已连接 %s (%s)", self._port, self._mode)
            else:
                logger.warning("ModbusDriver: 连接失败 %s", self._port)
            return self._connected
        except Exception as e:
            logger.error("Modbus 连接异常: %s", e)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        if self._client:
            await asyncio.to_thread(self._client.close)
            self._client = None
        self._connected = False
        logger.info("ModbusDriver: 已断开")

    async def health_check(self) -> bool:
        return self._connected and self._client is not None

    # ── 设备发现 ──────────────────────────────

    async def discover(self) -> List[DeviceInfo]:
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
                    "protocol": "modbus",
                    "mode": self._mode,
                    "port": self._port,
                    "slave_id": info.get("slave_id"),
                },
            ))
        return result

    # ── 指令执行 ──────────────────────────────

    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        """通过 Modbus 写寄存器控制设备"""
        if device_id not in self._devices:
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message=f"设备 '{device_id}' 未注册",
                error_code="DEVICE_NOT_FOUND",
            )

        if not self._connected:
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message="Modbus 总线未连接",
                error_code="NOT_CONNECTED",
            )

        try:
            dev = self._devices[device_id]
            slave_id = dev["info"]["slave_id"]

            # 命令码映射（与模拟器 HR[2] 对齐）
            CMD_MAP = {
                "power_on": 1, "boot": 1,
                "power_off": 2, "shutdown": 2,
                "start": 3, "stop": 4, "reset": 5,
            }
            cmd_val = CMD_MAP.get(command.command)
            if cmd_val is None:
                return DeviceResult(
                    success=False, device_id=device_id,
                    executed_command=command.command,
                    message=f"Modbus 驱动不支持命令 '{command.command}'",
                    error_code="UNSUPPORTED_COMMAND",
                )

            # 写 HR[2]=命令码, HR[3]=duration(秒), 一次性写入确保原子性
            duration = int(command.params.get("duration") or 0) * 60  # 分钟转秒
            if not (0 <= duration <= 65535):
                return DeviceResult(
                    success=False, device_id=device_id,
                    executed_command=command.command,
                    message="Modbus duration 超出单寄存器范围（0-1092 分钟）",
                    error_code="INVALID_PARAMS",
                )
            unit_kwargs = self._unit_kwargs(self._client.write_registers, slave_id)
            result = await asyncio.to_thread(
                self._client.write_registers,
                2,
                [cmd_val, duration],
                **unit_kwargs,
            )
            if result.isError():
                return DeviceResult(
                    success=False, device_id=device_id,
                    executed_command=command.command,
                    message=f"[Modbus] 指令失败 (从站 {slave_id}): 写入错误",
                    error_code="MODBUS_EXCEPTION",
                )

            # 乐观更新本地状态（实际状态由 read_state 刷新）
            if command.command == "start":
                dev["state"]["power"] = True
                dev["state"]["status"] = "running"
            elif command.command in ("stop",):
                dev["state"]["status"] = "standby"
            elif command.command in ("power_on", "boot"):
                dev["state"]["power"] = True
                dev["state"]["status"] = "standby"
            elif command.command in ("power_off", "shutdown"):
                dev["state"]["power"] = False
                dev["state"]["status"] = "powered_off"
            elif command.command == "reset":
                dev["state"]["power"] = True
                dev["state"]["status"] = "standby"

            return DeviceResult(
                success=True, device_id=device_id,
                executed_command=command.command,
                actual_params=command.params,
                message=f"[Modbus] 从站 {slave_id} 已执行 {command.command}",
            )

        except Exception as e:
            logger.error("Modbus 执行失败: %s → %s", device_id, e)
            dev["state"]["status"] = "error"
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message=str(e),
                error_code="MODBUS_EXCEPTION",
            )

    # ── 状态读取 ──────────────────────────────

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        """读取 Modbus 从设备状态

        寄存器布局（与模拟器 ModbusTcpServer 对齐）:
          HR[0]: 设备状态 (0=关机, 1=待机, 2=工作中, 3=故障)
          HR[1]: 电源 (0=关, 1=开)
          HR[10]: 温度 × 10
          HR[11]: 湿度 × 10
          HR[12]: 土壤湿度 × 10
          HR[13]: pH × 10
          HR[14]: 光照 / 100
        """
        if device_id not in self._devices:
            return {"error": f"设备 '{device_id}' 不存在"}

        dev = self._devices[device_id]
        if not self._connected:
            return {**dev["state"], "_driver": "modbus", "_read_at": datetime.now().isoformat()}

        STATUS_RMAP = {0: "powered_off", 1: "standby", 2: "running", 3: "error"}
        try:
            slave_id = dev["info"]["slave_id"]
            # 读 HR[0-14] 覆盖控制寄存器 + 传感器数据
            unit_kwargs = self._unit_kwargs(
                self._client.read_holding_registers, slave_id
            )
            result = await asyncio.to_thread(
                self._client.read_holding_registers,
                0,
                count=15,
                **unit_kwargs,
            )
            if not result.isError():
                registers = result.registers
                new_state = {
                    "status": STATUS_RMAP.get(registers[0], "powered_off"),
                    "power": registers[1] == 1,
                }
                # 传感器数据（如果有值）
                if len(registers) > 10:
                    if registers[10] > 0:
                        new_state["temperature"] = registers[10] / 10.0
                    if registers[11] > 0:
                        new_state["humidity"] = registers[11] / 10.0
                    if registers[12] > 0:
                        new_state["soil_moisture"] = registers[12] / 10.0
                    if registers[13] > 0:
                        new_state["ph"] = registers[13] / 10.0
                    if registers[14] > 0:
                        new_state["light_lux"] = registers[14] * 100
                dev["state"].update(new_state)
        except Exception as e:
            logger.debug("Modbus 读状态失败: %s", e)
            return {
                **dev["state"],
                "status": "error",
                "_driver": "modbus",
                "_read_at": datetime.now().isoformat(),
                "_error": str(e),
            }

        return {**dev["state"], "_driver": "modbus", "_read_at": datetime.now().isoformat()}

    @staticmethod
    def _unit_kwargs(method, slave_id: int) -> Dict[str, int]:
        """兼容 pymodbus 3.5-3.14 的从站参数改名。"""
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "device_id" in parameters:
            return {"device_id": slave_id}
        if "slave" in parameters:
            return {"slave": slave_id}
        # 新版本默认使用 device_id；无法反射签名时优先采用新接口。
        return {"device_id": slave_id}
