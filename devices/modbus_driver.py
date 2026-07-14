"""Modbus 设备驱动 — 通过 Modbus RTU/TCP 协议与工业设备通信

依赖: pip install pymodbus

使用方式:
    driver = ModbusDriver(mode="rtu", port="/dev/ttyUSB0", baudrate=9600)
    driver.register_device("plc_01", "PLC控制器#1", [...], slave_id=1)
    await driver.connect()
    result = await driver.execute("plc_01", DeviceCommand("start", {"duration": 30}))
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

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

        self._mode = mode
        self._port = port
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
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._connected = False

        try:
            if self._mode == "tcp":
                # 使用 rsplit 从右侧分割一次，正确处理 IPv6 地址（如 [::1]:502 或 ::1:502）
                if ":" in self._port:
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
                self._connected = self._client.connect()
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
            self._client.close()
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

            if command.command == "start":
                # 写保持寄存器 HR[0] = 1 (启动)
                result = self._client.write_register(0, 1, slave=slave_id)
                if not result.isError():
                    dev["state"]["power"] = True
                    dev["state"]["status"] = "running"
                    return DeviceResult(
                        success=True, device_id=device_id,
                        executed_command="start",
                        actual_params=command.params,
                        message=f"[Modbus] 从站 {slave_id} 已启动",
                    )
                else:
                    return DeviceResult(
                        success=False, device_id=device_id,
                        executed_command="start",
                        message=f"[Modbus] 启动失败 (从站 {slave_id}): Modbus 写入错误",
                        error_code="MODBUS_EXCEPTION",
                    )

            elif command.command == "stop":
                # 写保持寄存器 HR[0] = 0 (停止)
                result = self._client.write_register(0, 0, slave=slave_id)
                if not result.isError():
                    dev["state"]["power"] = False
                    dev["state"]["status"] = "idle"
                    return DeviceResult(
                        success=True, device_id=device_id,
                        executed_command="stop",
                        message=f"[Modbus] 从站 {slave_id} 已停止",
                    )
                else:
                    return DeviceResult(
                        success=False, device_id=device_id,
                        executed_command="stop",
                        message=f"[Modbus] 停止失败 (从站 {slave_id}): Modbus 写入错误",
                        error_code="MODBUS_EXCEPTION",
                    )

            # 无法识别的命令：使用更明确的错误码和消息
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message=f"Modbus 驱动不支持命令 '{command.command}'，仅支持 start / stop",
                error_code="UNSUPPORTED_COMMAND",
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
        """读取 Modbus 从设备状态"""
        if device_id not in self._devices:
            return {"error": f"设备 '{device_id}' 不存在"}

        dev = self._devices[device_id]
        if not self._connected:
            return {**dev["state"], "_driver": "modbus", "_read_at": datetime.now().isoformat()}

        try:
            slave_id = dev["info"]["slave_id"]
            # 读保持寄存器 HR[0-3] (电源, 状态, 设定值, 当前值)
            result = self._client.read_holding_registers(0, 4, slave=slave_id)
            if not result.isError():
                registers = result.registers
                dev["state"].update({
                    "power": registers[0] == 1,
                    "status": {0: "idle", 1: "running", 2: "error"}.get(registers[1], "unknown"),
                    "setpoint": registers[2],
                    "current_value": registers[3],
                })
        except Exception as e:
            logger.debug("Modbus 读状态失败: %s", e)
            # 异常时也要标记 driver 和 read_at，并设置 error 状态
            # 不返回静默的旧数据，而是明确告知调用方读取出错了
            return {
                **dev["state"],
                "status": "error",
                "_driver": "modbus",
                "_read_at": datetime.now().isoformat(),
                "_error": str(e),
            }

        return {**dev["state"], "_driver": "modbus", "_read_at": datetime.now().isoformat()}
