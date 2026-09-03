"""测试指令执行器"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from devices.registry import DeviceDriverRegistry
from devices.simulator_driver import SimulatorDriver
from devices.base import DeviceCommand, DeviceCapability
from core.device_executor import DeviceExecutor
from core.device_safety_policy import (
    AUTO_EXECUTE,
    NEED_CONFIRM,
    REJECTED,
    PolicyEvaluation,
)


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

    def test_待确认参数可修改且确认时重新执行安全链(self, monkeypatch):
        import asyncio
        import core.device_safety_policy as safety_module

        evaluated = {}

        class FakeSafetyPolicyService:
            def __init__(self, username):
                self.username = username

            def evaluate(self, **kwargs):
                evaluated.update(kwargs)
                return PolicyEvaluation(
                    AUTO_EXECUTE,
                    "参数安全",
                    kwargs["params"],
                    kwargs["capability"],
                )

        monkeypatch.setattr(
            safety_module, "SafetyPolicyService", FakeSafetyPolicyService
        )

        asyncio.run(self.sim.connect())
        asyncio.run(self.registry.discover_all())
        action_id = self.executor.add_pending({
            "device_id": "virtual_irrigation_01",
            "command": "start",
            "params": {"duration": 20},
            "capability": "irrigate",
            "reason": "需要人工确认",
        })

        updated = self.executor.update_pending(action_id, {"duration": 30})
        assert updated["success"] is True
        assert updated["action"]["params"] == {"duration": 30}

        result = self.executor.confirm_pending(action_id)

        assert result["success"] is True, result
        assert result["action_status"] == "executed"
        assert evaluated["params"] == {"duration": 30}
        stored = next(
            action for action in self.executor.pending_actions
            if action["id"] == action_id
        )
        assert stored["status"] == "executed"
        assert self.executor.list_pending() == []

    def test_待确认执行失败后保留为可重试状态(self):
        action_id = self.executor.add_pending({
            "device_id": "missing_irrigation_device",
            "command": "start",
            "params": {"duration": 10},
            "capability": "irrigate",
        })

        result = self.executor.confirm_pending(action_id)

        assert result["success"] is False
        assert result["action_status"] == "failed"
        pending_action = next(
            action for action in self.executor.list_pending()
            if action["id"] == action_id
        )
        assert pending_action["status"] == "failed"
        assert pending_action["last_error"]

        retriable = self.executor.update_pending(action_id, {"duration": 15})
        assert retriable["success"] is True
        assert retriable["action"]["status"] == "pending"

    def test_待确认列表返回副本(self):
        action_id = self.executor.add_pending({
            "device_id": "virtual_irrigation_01",
            "command": "start",
            "params": {"duration": 20},
        })

        listed = self.executor.list_pending()
        listed[0]["params"]["duration"] = 999

        stored = next(
            action for action in self.executor.pending_actions
            if action["id"] == action_id
        )
        assert stored["params"]["duration"] == 20

    def test_进程中断后的执行中操作恢复为失败可重试(self):
        action_id = self.executor.add_pending({
            "device_id": "virtual_irrigation_01",
            "command": "start",
            "params": {"duration": 20},
        })
        action = next(
            item for item in self.executor.pending_actions
            if item["id"] == action_id
        )
        action["status"] = "executing"
        self.executor._save_pending()

        recovered_executor = DeviceExecutor(self.registry, username="test_user")
        recovered = next(
            item for item in recovered_executor.list_pending()
            if item["id"] == action_id
        )

        assert recovered["status"] == "failed"
        assert "执行被中断" in recovered["last_error"]

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

    def test_旧调用未传能力时仍应用物理上限(self, monkeypatch):
        import asyncio

        monkeypatch.setattr(self.executor, "_write_log", lambda *args: None)

        async def _async_body():
            await self.sim.connect()
            await self.registry.discover_all()
            cmd = DeviceCommand(command="start", params={"duration": 121})
            result = await self.executor.execute("virtual_irrigation_01", cmd)
            assert not result["success"]
            assert result["decision"] == REJECTED
            assert "物理上限" in result["result"].message

        asyncio.run(_async_body())

    def test_high自主模式不能跳过安全策略确认(self, monkeypatch):
        import asyncio
        import core.device_safety_policy as safety_module

        class FakeSafetyPolicyService:
            def __init__(self, username):
                self.username = username

            def evaluate(self, **kwargs):
                return PolicyEvaluation(
                    NEED_CONFIRM,
                    "用户策略要求确认",
                    kwargs["params"],
                    kwargs["capability"],
                )

        monkeypatch.setattr(
            safety_module, "SafetyPolicyService", FakeSafetyPolicyService
        )
        monkeypatch.setenv("AUTONOMY_LEVEL", "high")
        monkeypatch.setattr(self.executor, "_write_log", lambda *args: None)
        monkeypatch.setattr(
            self.executor, "add_pending", lambda action: "pending_policy_test"
        )

        cmd = DeviceCommand(command="start", params={"duration": 20})
        result = asyncio.run(self.executor.execute(
            "virtual_irrigation_01", cmd, capability="irrigate"
        ))

        assert not result["success"]
        assert result["decision"] == NEED_CONFIRM
        assert result["pending_id"] == "pending_policy_test"

    def test_人工确认不能跳过物理拒绝(self, monkeypatch):
        import asyncio
        import core.device_safety_policy as safety_module

        class FakeSafetyPolicyService:
            def __init__(self, username):
                self.username = username

            def evaluate(self, **kwargs):
                return PolicyEvaluation(
                    REJECTED,
                    "超过物理绝对上限",
                    kwargs["params"],
                    kwargs["capability"],
                )

        monkeypatch.setattr(
            safety_module, "SafetyPolicyService", FakeSafetyPolicyService
        )
        monkeypatch.setattr(self.executor, "_write_log", lambda *args: None)

        cmd = DeviceCommand(command="start", params={"duration": 999})
        result = asyncio.run(self.executor.execute(
            "virtual_irrigation_01",
            cmd,
            trigger="confirmed",
            capability="irrigate",
        ))

        assert not result["success"]
        assert result["decision"] == REJECTED
