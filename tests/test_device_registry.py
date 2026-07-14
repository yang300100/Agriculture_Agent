"""测试设备注册中心"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from devices.registry import DeviceDriverRegistry
from devices.simulator_driver import SimulatorDriver
from devices.base import DeviceCommand, DeviceCapability


class TestDeviceDriverRegistry:
    def setup_method(self):
        self.registry = DeviceDriverRegistry()
        self.sim = SimulatorDriver(simulated_latency_ms=0)
        # 注册测试需要的虚拟设备
        self.sim.add_virtual_device(
            "virtual_irrigation_01", "虚拟灌溉器#1",
            [DeviceCapability.IRRIGATE], ["flow_rate"], "默认区域")
        self.sim.add_virtual_device(
            "virtual_soil_sensor_01", "虚拟土壤传感器#1",
            [DeviceCapability.READ_SENSOR], ["temperature", "humidity", "soil_moisture"], "默认区域")
        for i in range(2, 6):
            self.sim.add_virtual_device(
                f"virtual_device_{i:02d}", f"虚拟设备#{i}",
                [DeviceCapability.IRRIGATE], [], "默认区域")

    def test_register(self):
        self.registry.register("sim", self.sim)
        assert "sim" in self.registry.driver_names

    def test_unregister(self):
        self.registry.register("sim", self.sim)
        self.registry.unregister("sim")
        assert "sim" not in self.registry.driver_names

    def test_discover_all(self):
        import asyncio
        async def _async_body():
            self.registry.register("sim", self.sim)
            await self.sim.connect()
            devices = await self.registry.discover_all()
            assert len(devices) == 6
            assert self.registry.device_count == 6

        asyncio.run(_async_body())
    def test_execute_routes_correctly(self):
        import asyncio
        async def _async_body():
            self.registry.register("sim", self.sim)
            await self.sim.connect()
            await self.registry.discover_all()
            cmd = DeviceCommand(command="start", params={"duration": 20})
            result = await self.registry.execute("virtual_irrigation_01", cmd)
            assert result.success

        asyncio.run(_async_body())
    def test_execute_unknown_device(self):
        import asyncio
        async def _async_body():
            self.registry.register("sim", self.sim)
            cmd = DeviceCommand(command="start")
            result = await self.registry.execute("unknown_device", cmd)
            assert not result.success
            assert result.error_code == "DEVICE_NOT_FOUND"

        asyncio.run(_async_body())