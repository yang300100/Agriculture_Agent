"""测试虚拟设备驱动"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from devices.base import DeviceCommand, DeviceCapability
from devices.simulator_driver import SimulatorDriver


class TestSimulatorDriver:
    def setup_method(self):
        self.driver = SimulatorDriver(simulated_latency_ms=0)
        # 注册测试需要的虚拟设备
        self.driver.add_virtual_device(
            device_id="virtual_irrigation_01", name="虚拟灌溉器#1",
            capabilities=[DeviceCapability.IRRIGATE],
            sensors=["flow_rate"],
            location="默认区域",
        )
        self.driver.add_virtual_device(
            device_id="virtual_soil_sensor_01", name="虚拟土壤传感器#1",
            capabilities=[DeviceCapability.READ_SENSOR],
            sensors=["temperature", "humidity", "soil_moisture"],
            location="默认区域",
        )
        self.driver.add_virtual_device(
            device_id="virtual_ventilation_01", name="虚拟通风设备#1",
            capabilities=[DeviceCapability.VENTILATE],
            sensors=[],
            location="默认区域",
        )
        self.driver.add_virtual_device(
            device_id="virtual_light_01", name="虚拟补光灯#1",
            capabilities=[DeviceCapability.LIGHT],
            sensors=[],
            location="默认区域",
        )
        self.driver.add_virtual_device(
            device_id="virtual_fertigator_01", name="虚拟施肥器#1",
            capabilities=[DeviceCapability.FERTIGATE],
            sensors=[],
            location="默认区域",
        )
        self.driver.add_virtual_device(
            device_id="virtual_heater_01", name="虚拟加热器#1",
            capabilities=[DeviceCapability.HEAT],
            sensors=["temperature"],
            location="默认区域",
        )

    def test_init_has_devices(self):
        assert len(self.driver._devices) == 6

    def test_connect_discover(self):
        async def _async_body():
            await self.driver.connect()
            devices = await self.driver.discover()
            assert len(devices) == 6
            ids = [d.device_id for d in devices]
            assert "virtual_irrigation_01" in ids
            assert "virtual_soil_sensor_01" in ids

        asyncio.run(_async_body())

    def test_execute_start_stop(self):
        async def _async_body():
            await self.driver.connect()
            cmd = DeviceCommand(command="start", params={"duration": 30})
            result = await self.driver.execute("virtual_irrigation_01", cmd)
            assert result.success
            assert "通电并启动" in result.message

            state = await self.driver.read_state("virtual_irrigation_01")
            assert state["power"] is True

            stop_cmd = DeviceCommand(command="stop")
            result2 = await self.driver.execute("virtual_irrigation_01", stop_cmd)
            assert result2.success
            state2 = await self.driver.read_state("virtual_irrigation_01")
            assert state2["power"] is True  # stop不改变power，只改status
            assert state2["status"] == "standby"

        asyncio.run(_async_body())
    def test_read_sensor_with_fluctuation(self):
        async def _async_body():
            await self.driver.connect()
            state1 = await self.driver.read_state("virtual_soil_sensor_01")
            state2 = await self.driver.read_state("virtual_soil_sensor_01")
            assert state1["temperature"] != state2["temperature"] or state1["humidity"] != state2["humidity"]

        asyncio.run(_async_body())
    def test_simulated_failure(self):
        async def _async_body():
            driver = SimulatorDriver(simulated_latency_ms=0, simulated_failure_rate=1.0)
            await driver.connect()
            cmd = DeviceCommand(command="start")
            result = await driver.execute("virtual_irrigation_01", cmd)
            assert not result.success
            assert result.error_code == "SIMULATED_FAILURE"

        asyncio.run(_async_body())
    def test_execute_unknown_device(self):
        async def _async_body():
            await self.driver.connect()
            cmd = DeviceCommand(command="start")
            result = await self.driver.execute("nonexistent", cmd)
            assert not result.success
            assert result.error_code == "DEVICE_NOT_FOUND"

        asyncio.run(_async_body())
    def test_add_custom_device(self):
        self.driver.add_virtual_device("custom_pump_01", "自定义水泵", [DeviceCapability.IRRIGATE], sensors=["pressure"])
        assert "custom_pump_01" in self.driver._devices

    def test_set_sensor_value(self):
        self.driver.set_sensor_value("virtual_soil_sensor_01", "soil_moisture", 25.0)
        assert self.driver._devices["virtual_soil_sensor_01"]["state"]["soil_moisture"] == 25.0

    def test_get_history(self):
        self.driver._history.append({"test": True})
        assert len(self.driver.get_history()) == 1


@pytest.mark.parametrize(
    "device_id,capability,commands,expected",
    [
        ("irrigation_01", DeviceCapability.IRRIGATE, ["power_on"], 44.8),
        ("irrigation_01", DeviceCapability.IRRIGATE, ["start", "stop"], 44.8),
        ("pump_01", DeviceCapability.IRRIGATE, ["start"], 46.0),
        ("irrigation_fan", DeviceCapability.VENTILATE, ["start"], 44.8),
    ],
)
def test_土壤水分只响应正在工作的灌溉能力设备(
    monkeypatch, device_id, capability, commands, expected
):
    monkeypatch.setattr(
        "devices.simulator_driver.random.uniform", lambda lo, hi: (lo + hi) / 2
    )

    async def run():
        driver = SimulatorDriver(simulated_latency_ms=0)
        await driver.connect()
        driver.add_virtual_device(device_id, "测试执行器", [capability])
        driver.add_virtual_device(
            "soil_01", "测试土壤传感器", [DeviceCapability.READ_SENSOR],
            sensors=["soil_moisture"], initial_state={"soil_moisture": 45.0},
        )
        for command in commands:
            assert (await driver.execute(device_id, DeviceCommand(command=command))).success
        assert (await driver.read_state("soil_01"))["soil_moisture"] == expected

    asyncio.run(run())


def test_不同地块的灌溉反馈不会串到另一块土壤(monkeypatch):
    monkeypatch.setattr(
        "devices.simulator_driver.random.uniform", lambda lo, hi: (lo + hi) / 2
    )

    async def run():
        driver = SimulatorDriver(simulated_latency_ms=0)
        await driver.connect()
        for suffix, location in [("a", "东侧地块"), ("b", "西侧地块")]:
            driver.add_virtual_device(
                f"pump_{suffix}", "测试水泵", [DeviceCapability.IRRIGATE],
                location=location,
            )
            driver.add_virtual_device(
                f"soil_{suffix}", "测试土壤传感器", [DeviceCapability.READ_SENSOR],
                sensors=["soil_moisture"], location=location,
                initial_state={"soil_moisture": 45.0},
            )
        assert (await driver.execute("pump_a", DeviceCommand(command="start"))).success
        assert (await driver.read_state("soil_a"))["soil_moisture"] == 46.0
        assert (await driver.read_state("soil_b"))["soil_moisture"] == 44.8

    asyncio.run(run())


@pytest.mark.parametrize(
    "sensor_location,pump_location", [("", ""), ("东侧地块", ""), ("", "东侧地块")]
)
def test_缺少位置信息时保持旧模拟设备的灌溉反馈(
    monkeypatch, sensor_location, pump_location
):
    monkeypatch.setattr(
        "devices.simulator_driver.random.uniform", lambda lo, hi: (lo + hi) / 2
    )

    async def run():
        driver = SimulatorDriver(simulated_latency_ms=0)
        await driver.connect()
        driver.add_virtual_device(
            "pump_01", "测试水泵", [DeviceCapability.IRRIGATE], location=pump_location,
        )
        driver.add_virtual_device(
            "soil_01", "测试土壤传感器", [DeviceCapability.READ_SENSOR],
            sensors=["soil_moisture"], location=sensor_location,
            initial_state={"soil_moisture": 45.0},
        )
        assert (await driver.execute("pump_01", DeviceCommand(command="start"))).success
        assert (await driver.read_state("soil_01"))["soil_moisture"] == 46.0

    asyncio.run(run())
