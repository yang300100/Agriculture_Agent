"""测试设备注册中心"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from devices.registry import DeviceDriverRegistry
from devices.simulator_driver import SimulatorDriver
from devices.base import DeviceCommand


class TestDeviceDriverRegistry:
    def setup_method(self):
        self.registry = DeviceDriverRegistry()
        self.sim = SimulatorDriver(simulated_latency_ms=0)

    def test_register(self):
        self.registry.register("sim", self.sim)
        assert "sim" in self.registry.driver_names

    def test_unregister(self):
        self.registry.register("sim", self.sim)
        self.registry.unregister("sim")
        assert "sim" not in self.registry.driver_names

    @pytest.mark.asyncio
    async def test_discover_all(self):
        self.registry.register("sim", self.sim)
        await self.sim.connect()
        devices = await self.registry.discover_all()
        assert len(devices) == 6
        assert self.registry.device_count == 6

    @pytest.mark.asyncio
    async def test_execute_routes_correctly(self):
        self.registry.register("sim", self.sim)
        await self.sim.connect()
        await self.registry.discover_all()
        cmd = DeviceCommand(command="start", params={"duration": 20})
        result = await self.registry.execute("virtual_irrigation_01", cmd)
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_unknown_device(self):
        self.registry.register("sim", self.sim)
        cmd = DeviceCommand(command="start")
        result = await self.registry.execute("unknown_device", cmd)
        assert not result.success
        assert result.error_code == "DEVICE_NOT_FOUND"
