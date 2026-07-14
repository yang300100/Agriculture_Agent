"""虚拟设备模拟器 — 无需真实硬件即可跑通全链路测试

设备生命周期状态机:
  powered_off(关机) ──[power_on]──▶ standby(待机) ──[start]──▶ running(工作中)
      ▲                                  ▲                       │
      │                                  │ [stop]                │
      │                                  ◀───────────────────────┘
      │
      └──────────[power_off]─────────────┘
"""

import asyncio
import copy
import logging
import math
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

from .base import (
    BaseDeviceDriver, DeviceCapability, DeviceStatus,
    DeviceInfo, DeviceCommand, DeviceResult,
)

logger = logging.getLogger(__name__)


class SimulatorDriver(BaseDeviceDriver):
    """虚拟设备驱动 — 在内存中模拟设备行为。

    真实场景模式下，SimulatorDriver 初始为空，设备需通过 add_virtual_device() 显式添加。
    """

    driver_name = "simulator"

    def __init__(self,
                 simulated_latency_ms: int = 100,
                 simulated_failure_rate: float = 0.0):
        self._latency = simulated_latency_ms
        self._failure_rate = simulated_failure_rate
        self._connected = False
        self._devices: Dict[str, Dict] = {}
        self._history: List[Dict] = []

    # ── 生命周期 ──────────────────────────────

    async def connect(self) -> bool:
        self._connected = True
        logger.debug("SimulatorDriver: 已连接（%d 个虚拟设备）", len(self._devices))
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

        # 连接状态检查
        if not self._connected:
            return DeviceResult(
                success=False,
                device_id=device_id,
                executed_command=command.command,
                message="驱动未连接，无法执行指令",
                error_code="NOT_CONNECTED",
            )

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

        current = dev["state"].get("status", "powered_off")
        name = dev["info"]["name"]

        # ── 通电启动 ──
        if command.command in ("power_on", "boot"):
            if current == "powered_off":
                dev["state"]["power"] = True
                dev["state"]["status"] = "standby"
                msg = f"[模拟] {name} 通电启动，进入待机"
            elif current == "standby":
                msg = f"[模拟] {name} 已在待机状态"
            elif current == "running":
                msg = f"[模拟] {name} 正在工作中，无需重复通电"
            elif current == "error":
                return DeviceResult(success=False, device_id=device_id,
                                   executed_command=command.command,
                                   message=f"[模拟] {name} 处于故障状态，请先复位(reset)",
                                   error_code="DEVICE_ERROR")

        # ── 关机断电 ──
        elif command.command in ("power_off", "shutdown"):
            if current in ("standby", "running"):
                dev["state"]["power"] = False
                dev["state"]["status"] = "powered_off"
                msg = f"[模拟] {name} 关机断电"
            elif current == "powered_off":
                msg = f"[模拟] {name} 已在关机状态"
            elif current == "error":
                dev["state"]["power"] = False
                dev["state"]["status"] = "powered_off"
                msg = f"[模拟] {name} 故障状态下强制关机"

        # ── 开始工作 ──
        elif command.command == "start":
            if current == "powered_off":
                dev["state"]["power"] = True
                dev["state"]["status"] = "running"
                msg = f"[模拟] {name} 通电并启动"
            elif current == "standby":
                dev["state"]["status"] = "running"
                msg = f"[模拟] {name} 已开始工作"
            elif current == "running":
                msg = f"[模拟] {name} 已在工作中，更新参数"
            elif current == "error":
                return DeviceResult(success=False, device_id=device_id,
                                   executed_command=command.command,
                                   message=f"[模拟] {name} 处于故障状态，请先复位(reset)",
                                   error_code="DEVICE_ERROR")

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

        # ── 停止工作（回到待机，保持通电）──
        elif command.command == "stop":
            if current == "running":
                dev["state"]["status"] = "standby"
                # 关键：power 保持 True，不断电！
                if "flow_rate" in dev["state"]:
                    dev["state"]["flow_rate"] = 0
                if "rpm" in dev["state"]:
                    dev["state"]["rpm"] = 0
                msg = f"[模拟] {name} 已停止工作（保持通电待机）"
            elif current == "standby":
                msg = f"[模拟] {name} 当前未在工作（待机中）"
            elif current == "powered_off":
                msg = f"[模拟] {name} 处于关机状态"
            else:
                msg = f"[模拟] {name} 已停止"

        # ── 故障复位 ──
        elif command.command == "reset":
            if current == "error":
                dev["state"]["power"] = True
                dev["state"]["status"] = "standby"
                msg = f"[模拟] {name} 故障复位，恢复到待机"
            else:
                msg = f"[模拟] {name} 未处于故障状态"

        elif command.command == "set_param":
            for key, val in command.params.items():
                if key in dev["state"]:
                    dev["state"][key] = val
            msg = f"[模拟] {name} 参数已更新: {command.params}"

        else:
            # 未知指令直接返回失败
            return DeviceResult(
                success=False,
                device_id=device_id,
                executed_command=command.command,
                message=f"[模拟] {name} 不支持指令: {command.command}",
                error_code="UNSUPPORTED_COMMAND",
            )

        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "command": command.command,
            "params": command.params,
            "success": True,
        })
        # 历史记录上限防止无界增长
        self._history = self._history[-1000:]

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

        internal = self._devices[device_id]["state"]
        state = copy.deepcopy(internal)

        info = self._devices[device_id]["info"]
        if DeviceCapability.READ_SENSOR in info.get("capabilities", []):
            if "temperature" in state:
                state["temperature"] = round(state["temperature"] + random.uniform(-0.5, 0.5), 1)
                state["temperature"] = max(-50.0, min(60.0, state["temperature"]))  # 温度范围 -50~60°C
            if "humidity" in state:
                state["humidity"] = round(state["humidity"] + random.uniform(-2, 2), 1)
                state["humidity"] = max(0.0, min(100.0, state["humidity"]))  # 湿度范围 0~100%
            if "soil_moisture" in state:
                # 通用化：遍历所有设备，检查名称含 "irrigat" 且已开启的灌溉设备
                irrigation_on = False
                for did, dev in self._devices.items():
                    if "irrigat" in did.lower():
                        if dev.get("state", {}).get("power"):
                            irrigation_on = True
                            break
                if irrigation_on:
                    state["soil_moisture"] = round(state["soil_moisture"] + random.uniform(0.5, 1.5), 1)
                else:
                    state["soil_moisture"] = round(state["soil_moisture"] - random.uniform(0.1, 0.3), 1)
                state["soil_moisture"] = max(0, min(100, state["soil_moisture"]))
            if "ph" in state:
                # pH 漂移模拟 + 范围夹持（正常土壤 pH 范围 3.5~9.5）
                state["ph"] = round(state["ph"] + random.uniform(-0.1, 0.1), 1)
                state["ph"] = max(3.5, min(9.5, state["ph"]))

        state["_read_at"] = datetime.now().isoformat()

        # 把所有漂移后的传感器值写回内部状态，确保漂移跨调用累积
        # 遍历 internal 中的所有 key，凡是 state 中也存在的都写回
        for key in list(internal.keys()):
            if key in state and key not in ("_read_at",):
                internal[key] = state[key]

        return state

    # ── 自定义虚拟设备管理 ────────────────────

    def add_virtual_device(self, device_id: str, name: str,
                           capabilities: List[DeviceCapability],
                           sensors: List[str] = None,
                           location: str = "",
                           initial_state: Optional[Dict] = None) -> None:
        if initial_state is None:
            initial_state = {"power": False, "status": "powered_off"}
        self._devices[device_id] = {
            "info": {
                "device_id": device_id,
                "name": name,
                "capabilities": capabilities,
                "sensors": sensors or [],
                "location": location,
            },
            "state": copy.deepcopy(initial_state),
        }
        logger.debug("SimulatorDriver: 已添加虚拟设备 %s (%s)", device_id, name)

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
