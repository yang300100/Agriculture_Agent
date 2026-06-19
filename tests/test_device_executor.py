"""测试指令执行器"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from devices.registry import DeviceDriverRegistry
from devices.simulator_driver import SimulatorDriver
from devices.base import DeviceCommand
from core.device_executor import DeviceExecutor


class TestDeviceExecutor:
    def setup_method(self):
        self.registry = DeviceDriverRegistry()
        self.sim = SimulatorDriver(simulated_latency_ms=0)
        self.registry.register("sim", self.sim)
        self.executor = DeviceExecutor(self.registry, username="test_user")

    @pytest.mark.asyncio
    async def test_execute_success(self):
        await self.sim.connect()
        await self.registry.discover_all()
        cmd = DeviceCommand(command="start", params={"duration": 20})
        result = await self.executor.execute("virtual_irrigation_01", cmd)
        assert result["success"]
        assert result["attempts"] == 1

    @pytest.mark.asyncio
    async def test_execute_device_not_found(self):
        await self.sim.connect()
        cmd = DeviceCommand(command="start")
        result = await self.executor.execute("nonexistent", cmd)
        assert not result["success"]

    @pytest.mark.asyncio
    async def test_pending_actions(self):
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

    @pytest.mark.asyncio
    async def test_audit_log(self):
        await self.sim.connect()
        await self.registry.discover_all()
        cmd = DeviceCommand(command="start", params={"duration": 10})
        await self.executor.execute("virtual_irrigation_01", cmd, trigger="rule", rule_id="rule_test")
        logs = self.executor.get_logs()
        assert len(logs) >= 1
        assert logs[-1]["device_id"] == "virtual_irrigation_01"
        assert logs[-1]["trigger"] == "rule"
