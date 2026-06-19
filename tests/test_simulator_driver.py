"""测试虚拟设备驱动"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import pytest
from devices.simulator_driver import SimulatorDriver
from devices.base import DeviceCommand, DeviceCapability


class TestSimulatorDriver:
    def setup_method(self):
        self.driver = SimulatorDriver(simulated_latency_ms=0)

    def test_init_has_devices(self):
        assert len(self.driver._devices) == 6

    @pytest.mark.asyncio
    async def test_connect_discover(self):
        await self.driver.connect()
        devices = await self.driver.discover()
        assert len(devices) == 6
        ids = [d.device_id for d in devices]
        assert "virtual_irrigation_01" in ids
        assert "virtual_soil_sensor_01" in ids

    @pytest.mark.asyncio
    async def test_execute_start_stop(self):
        await self.driver.connect()
        cmd = DeviceCommand(command="start", params={"duration": 30})
        result = await self.driver.execute("virtual_irrigation_01", cmd)
        assert result.success
        assert "已启动" in result.message

        state = await self.driver.read_state("virtual_irrigation_01")
        assert state["power"] is True

        stop_cmd = DeviceCommand(command="stop")
        result2 = await self.driver.execute("virtual_irrigation_01", stop_cmd)
        assert result2.success
        state2 = await self.driver.read_state("virtual_irrigation_01")
        assert state2["power"] is False

    @pytest.mark.asyncio
    async def test_read_sensor_with_fluctuation(self):
        await self.driver.connect()
        state1 = await self.driver.read_state("virtual_soil_sensor_01")
        state2 = await self.driver.read_state("virtual_soil_sensor_01")
        assert state1["temperature"] != state2["temperature"] or state1["humidity"] != state2["humidity"]

    @pytest.mark.asyncio
    async def test_simulated_failure(self):
        driver = SimulatorDriver(simulated_latency_ms=0, simulated_failure_rate=1.0)
        await driver.connect()
        cmd = DeviceCommand(command="start")
        result = await driver.execute("virtual_irrigation_01", cmd)
        assert not result.success
        assert result.error_code == "SIMULATED_FAILURE"

    @pytest.mark.asyncio
    async def test_execute_unknown_device(self):
        await self.driver.connect()
        cmd = DeviceCommand(command="start")
        result = await self.driver.execute("nonexistent", cmd)
        assert not result.success
        assert result.error_code == "DEVICE_NOT_FOUND"

    def test_add_custom_device(self):
        self.driver.add_virtual_device("custom_pump_01", "自定义水泵", [DeviceCapability.IRRIGATE], sensors=["pressure"])
        assert "custom_pump_01" in self.driver._devices

    def test_set_sensor_value(self):
        self.driver.set_sensor_value("virtual_soil_sensor_01", "soil_moisture", 25.0)
        assert self.driver._devices["virtual_soil_sensor_01"]["state"]["soil_moisture"] == 25.0

    def test_get_history(self):
        self.driver._history.append({"test": True})
        assert len(self.driver.get_history()) == 1
