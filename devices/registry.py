"""设备驱动注册中心 — 统一管理所有驱动，对外提供设备操作接口"""

import logging
from typing import Dict, List, Optional

from .base import BaseDeviceDriver, DeviceInfo, DeviceCommand, DeviceResult

logger = logging.getLogger(__name__)


class DeviceDriverRegistry:
    """设备驱动注册中心

    用法:
        registry = DeviceDriverRegistry()
        registry.register("simulator", SimulatorDriver())

        # 发现所有设备
        devices = await registry.discover_all()

        # 执行指令（自动路由到对应驱动）
        result = await registry.execute("irrigation_valve_01", cmd)
    """

    def __init__(self):
        self._drivers: Dict[str, BaseDeviceDriver] = {}
        # device_id → driver_name 映射表（discover 后自动填充）
        self._device_map: Dict[str, str] = {}

    def register(self, name: str, driver: BaseDeviceDriver) -> None:
        """注册一个驱动"""
        self._drivers[name] = driver
        logger.info("驱动已注册: %s (%s)", name, driver.driver_name)

    def unregister(self, name: str) -> None:
        """注销一个驱动"""
        self._drivers.pop(name, None)
        self._device_map = {k: v for k, v in self._device_map.items() if v != name}

    def get_driver(self, device_id: str) -> Optional[BaseDeviceDriver]:
        """根据 device_id 找到对应的驱动"""
        driver_name = self._device_map.get(device_id)
        if driver_name:
            return self._drivers.get(driver_name)
        return None

    async def discover_all(self) -> List[DeviceInfo]:
        """发现所有驱动下的设备，返回完整设备列表"""
        all_devices: List[DeviceInfo] = []
        self._device_map.clear()

        for name, driver in self._drivers.items():
            try:
                devices = await driver.discover()
                for d in devices:
                    self._device_map[d.device_id] = name
                all_devices.extend(devices)
                logger.info("驱动 %s: 发现 %d 个设备", name, len(devices))
            except Exception as e:
                logger.warning("驱动 %s 设备发现失败: %s", name, e)

        return all_devices

    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        """向指定设备发送指令"""
        driver = self.get_driver(device_id)
        if driver is None:
            return DeviceResult(
                success=False,
                device_id=device_id,
                executed_command=command.command,
                message=f"设备 '{device_id}' 未找到对应驱动（请先执行 discover_all）",
                error_code="DEVICE_NOT_FOUND",
            )

        try:
            return await driver.execute(device_id, command)
        except Exception as e:
            logger.error("设备 %s 执行失败: %s", device_id, e)
            return DeviceResult(
                success=False,
                device_id=device_id,
                executed_command=command.command,
                message=str(e),
                error_code="EXECUTION_ERROR",
            )

    async def read_state(self, device_id: str) -> Dict:
        """读取设备当前状态"""
        driver = self.get_driver(device_id)
        if driver is None:
            return {"error": f"设备 '{device_id}' 未找到"}
        try:
            return await driver.read_state(device_id)
        except Exception as e:
            logger.error("设备 %s 状态读取失败: %s", device_id, e)
            return {"error": str(e)}

    async def read_all_states(self) -> Dict[str, Dict]:
        """读取所有设备状态"""
        states = {}
        for device_id in self._device_map:
            states[device_id] = await self.read_state(device_id)
        return states

    async def connect_all(self) -> None:
        """连接所有驱动"""
        for name, driver in self._drivers.items():
            try:
                ok = await driver.connect()
                logger.info("驱动 %s 连接: %s", name, "成功" if ok else "失败")
            except Exception as e:
                logger.warning("驱动 %s 连接异常: %s", name, e)

    async def disconnect_all(self) -> None:
        """断开所有驱动"""
        for name, driver in self._drivers.items():
            try:
                await driver.disconnect()
            except Exception as e:
                logger.warning("驱动 %s 断开异常: %s", name, e)

    @property
    def driver_names(self) -> List[str]:
        return list(self._drivers.keys())

    @property
    def device_count(self) -> int:
        return len(self._device_map)
