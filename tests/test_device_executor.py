"""测试指令执行器"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from devices.registry import DeviceDriverRegistry
from devices.simulator_driver import SimulatorDriver
from devices.base import DeviceCommand, DeviceCapability
from core.device_executor import DeviceExecutor


class TestDeviceExecutor:
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
        self.registry.register("sim", self.sim)
        self.executor = DeviceExecutor(self.registry, username="test_user")

    def test_execute_success(self):
        import asyncio
        async def _async_body():
            await self.sim.connect()
            await self.registry.discover_all()
            cmd = DeviceCommand(command="start", params={"duration": 20})
            result = await self.executor.execute("virtual_irrigation_01", cmd)
            assert result["success"]
            assert result["attempts"] == 1

        asyncio.run(_async_body())
    def test_execute_device_not_found(self):
        import asyncio
        async def _async_body():
            await self.sim.connect()
            cmd = DeviceCommand(command="start")
            result = await self.executor.execute("nonexistent", cmd)
            assert not result["success"]

        asyncio.run(_async_body())
    def test_pending_actions(self):
        action = {
            "device_id": "virtual_irrigation_01",
            "command": "start",
            "params": {"duration": 45},
            "reason": "测试待确认操作",
        }
        aid = self.executor.add_pending(action)
        assert aid.startswith("pending_")
        pending = self.executor.list_pending()
        assert len(pending) == 1
        self.executor.reject_pending(aid)
        assert len(self.executor.list_pending()) == 0

    def test_audit_log(self):
        import asyncio
        async def _async_body():
            await self.sim.connect()
            await self.registry.discover_all()
            cmd = DeviceCommand(command="start", params={"duration": 10})
            await self.executor.execute("virtual_irrigation_01", cmd, trigger="rule", rule_id="rule_test")
            logs = self.executor.get_logs()
            assert len(logs) >= 1
            assert logs[0]["device_id"] == "virtual_irrigation_01"
            assert logs[0]["trigger"] == "rule"

        asyncio.run(_async_body())