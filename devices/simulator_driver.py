"""虚拟设备模拟器 — 无需真实硬件即可跑通全链路测试"""

import asyncio
import copy
import logging
import random
from datetime import datetime
from typing import Dict, List, Any

from .base import (
    BaseDeviceDriver, DeviceCapability, DeviceStatus,
    DeviceInfo, DeviceCommand, DeviceResult,
)

logger = logging.getLogger(__name__)

# 内置虚拟设备模板
_VIRTUAL_DEVICE_TEMPLATES = [
    {
        "device_id": "virtual_irrigation_01",
        "name": "虚拟灌溉阀#1",
        "capabilities": [DeviceCapability.IRRIGATE],
        "sensors": ["flow_rate", "total_water_liters"],
        "location": "大棚A区",
        "initial_state": {"power": False, "flow_rate": 0, "total_water_liters": 0,
                          "last_duration": 0, "status": "idle"},
    },
    {
        "device_id": "virtual_soil_sensor_01",
        "name": "虚拟土壤传感器#1",
        "capabilities": [DeviceCapability.READ_SENSOR],
        "sensors": ["temperature", "humidity", "soil_moisture", "ph"],
        "location": "大棚A区",
        "initial_state": {"temperature": 22.5, "humidity": 65.0, "soil_moisture": 45.0, "ph": 6.8},
    },
    {
        "device_id": "virtual_ventilation_01",
        "name": "虚拟通风风机#1",
        "capabilities": [DeviceCapability.VENTILATE],
        "sensors": ["rpm", "power"],
        "location": "大棚A区",
        "initial_state": {"power": False, "rpm": 0, "status": "idle"},
    },
    {
        "device_id": "virtual_light_01",
        "name": "虚拟补光灯#1",
        "capabilities": [DeviceCapability.LIGHT],
        "sensors": ["power", "brightness_percent"],
        "location": "大棚A区",
        "initial_state": {"power": False, "brightness_percent": 0, "status": "idle"},
    },
    {
        "device_id": "virtual_fertigator_01",
        "name": "虚拟施肥一体机#1",
        "capabilities": [DeviceCapability.FERTIGATE],
        "sensors": ["flow_rate", "total_fertilizer_kg"],
        "location": "大棚A区",
        "initial_state": {"power": False, "flow_rate": 0, "total_fertilizer_kg": 0,
                          "last_amount_kg": 0, "status": "idle"},
    },
    {
        "device_id": "virtual_heater_01",
        "name": "虚拟加热器#1",
        "capabilities": [DeviceCapability.HEAT],
        "sensors": ["power", "target_temp", "current_temp"],
        "location": "大棚A区",
        "initial_state": {"power": False, "target_temp": 20.0, "current_temp": 18.0, "status": "idle"},
    },
]


class SimulatorDriver(BaseDeviceDriver):
    """虚拟设备驱动 — 在内存中模拟设备行为"""

    driver_name = "simulator"

    def __init__(self,
                 simulated_latency_ms: int = 100,
                 simulated_failure_rate: float = 0.0):
        self._latency = simulated_latency_ms
        self._failure_rate = simulated_failure_rate
        self._connected = False
        self._devices: Dict[str, Dict] = {}
        self._history: List[Dict] = []

        # 初始化内置虚拟设备
        for template in _VIRTUAL_DEVICE_TEMPLATES:
            self._devices[template["device_id"]] = {
                "info": {
                    "device_id": template["device_id"],
                    "name": template["name"],
                    "capabilities": template["capabilities"],
                    "sensors": template["sensors"],
                    "location": template.get("location", ""),
                },
                "state": copy.deepcopy(template["initial_state"]),
            }

    # ── 生命周期 ──────────────────────────────

    async def connect(self) -> bool:
        self._connected = True
        logger.info("SimulatorDriver: 已连接（%d 个虚拟设备）", len(self._devices))
        return True

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("SimulatorDriver: 已断开")

    async def health_check(self) -> bool:
        return self._connected

    # ── 设备发现 ──────────────────────────────

    async def discover(self) -> List[DeviceInfo]:
        if not self._connected:
            await self.connect()

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
                metadata={"simulated": True},
            ))
        return result

    # ── 指令执行 ──────────────────────────────

    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        await self._simulate_latency()

        # 故障模拟
        if random.random() < self._failure_rate:
            return DeviceResult(
                success=False,
                device_id=device_id,
                executed_command=command.command,
                message=f"[模拟] 设备 {device_id} 执行失败（模拟故障）",
                error_code="SIMULATED_FAILURE",
            )

        if device_id not in self._devices:
            return DeviceResult(
                success=False,
                device_id=device_id,
                executed_command=command.command,
                message=f"设备 '{device_id}' 不存在",
                error_code="DEVICE_NOT_FOUND",
            )

        dev = self._devices[device_id]

        if command.command == "start":
            dev["state"]["power"] = True
            dev["state"]["status"] = "running"
            msg = f"[模拟] {dev['info']['name']} 已启动"

            if "duration" in command.params:
                dev["state"]["last_duration"] = command.params["duration"]
            if "flow_rate" in command.params:
                dev["state"]["flow_rate"] = command.params["flow_rate"]
            if "target_temp" in command.params:
                dev["state"]["target_temp"] = command.params["target_temp"]
            if "brightness_percent" in command.params:
                dev["state"]["brightness_percent"] = command.params["brightness_percent"]
            if "amount_kg" in command.params:
                dev["state"]["last_amount_kg"] = command.params["amount_kg"]

        elif command.command == "stop":
            dev["state"]["power"] = False
            dev["state"]["status"] = "idle"
            if "flow_rate" in dev["state"]:
                dev["state"]["flow_rate"] = 0
            if "rpm" in dev["state"]:
                dev["state"]["rpm"] = 0
            msg = f"[模拟] {dev['info']['name']} 已停止"

        elif command.command == "set_param":
            for key, val in command.params.items():
                if key in dev["state"]:
                    dev["state"][key] = val
            msg = f"[模拟] {dev['info']['name']} 参数已更新: {command.params}"
        else:
            msg = f"[模拟] {dev['info']['name']} 收到未知指令: {command.command}"

        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "command": command.command,
            "params": command.params,
            "success": True,
        })

        return DeviceResult(
            success=True,
            device_id=device_id,
            executed_command=command.command,
            actual_params=command.params,
            message=msg,
        )

    # ── 状态读取 ──────────────────────────────

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        await self._simulate_latency(latency_ms=max(20, self._latency // 2))

        if device_id not in self._devices:
            return {"error": f"设备 '{device_id}' 不存在"}

        state = copy.deepcopy(self._devices[device_id]["state"])

        info = self._devices[device_id]["info"]
        if DeviceCapability.READ_SENSOR in info.get("capabilities", []):
            if "temperature" in state:
                state["temperature"] = round(state["temperature"] + random.uniform(-0.5, 0.5), 1)
            if "humidity" in state:
                state["humidity"] = round(state["humidity"] + random.uniform(-2, 2), 1)
            if "soil_moisture" in state:
                irrigation = self._devices.get("virtual_irrigation_01", {})
                if irrigation.get("state", {}).get("power"):
                    state["soil_moisture"] = round(state["soil_moisture"] + random.uniform(0.5, 1.5), 1)
                else:
                    state["soil_moisture"] = round(state["soil_moisture"] - random.uniform(0.1, 0.3), 1)
                state["soil_moisture"] = max(0, min(100, state["soil_moisture"]))

        state["_read_at"] = datetime.now().isoformat()
        return state

    # ── 自定义虚拟设备管理 ────────────────────

    def add_virtual_device(self, device_id: str, name: str,
                           capabilities: List[DeviceCapability],
                           sensors: List[str] = None,
                           location: str = "",
                           initial_state: Dict = None) -> None:
        self._devices[device_id] = {
            "info": {
                "device_id": device_id,
                "name": name,
                "capabilities": capabilities,
                "sensors": sensors or [],
                "location": location,
            },
            "state": initial_state or {"power": False, "status": "idle"},
        }
        logger.info("SimulatorDriver: 已添加虚拟设备 %s (%s)", device_id, name)

    def remove_virtual_device(self, device_id: str) -> None:
        self._devices.pop(device_id, None)

    def set_sensor_value(self, device_id: str, field: str, value: Any) -> None:
        if device_id in self._devices:
            self._devices[device_id]["state"][field] = value

    def get_history(self, limit: int = 50) -> List[Dict]:
        return self._history[-limit:]

    async def _simulate_latency(self, latency_ms: int = None):
        ms = latency_ms if latency_ms is not None else self._latency
        if ms > 0:
            await asyncio.sleep(ms / 1000.0)
