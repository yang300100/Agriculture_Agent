"""测试设备抽象基类"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from devices.base import DeviceCapability, DeviceInfo, DeviceCommand, DeviceResult, DeviceStatus, CommandPriority


class TestDeviceCapability:
    def test_all_capabilities_exist(self):
        caps = [c.value for c in DeviceCapability]
        assert "irrigate" in caps
        assert "fertigate" in caps
        assert "read_sensor" in caps
        assert "capture" in caps
        assert len(caps) == 9


class TestDeviceInfo:
    def test_create_minimal(self):
        info = DeviceInfo(device_id="test_01", name="测试设备", driver_name="simulator", capabilities=[])
        assert info.device_id == "test_01"
        assert info.status == DeviceStatus.ONLINE
        assert info.sensors == []


class TestDeviceCommand:
    def test_defaults(self):
        cmd = DeviceCommand(command="start")
        assert cmd.timeout_ms == 30000
        assert cmd.priority == CommandPriority.NORMAL
        assert cmd.params == {}


class TestDeviceResult:
    def test_success_result(self):
        r = DeviceResult(success=True, device_id="d1", executed_command="start", message="OK")
        assert r.success
        assert r.error_code is None

    def test_failure_result(self):
        r = DeviceResult(success=False, device_id="d1", executed_command="start", message="timeout", error_code="TIMEOUT")
        assert not r.success
        assert r.error_code == "TIMEOUT"
