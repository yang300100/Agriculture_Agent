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
import threading
import uuid
from datetime import datetime
from typing import Callable, Dict, List, Any, Optional

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
                 simulated_failure_rate: float = 0.0,
                 state_loader: Optional[Callable[[], Dict[str, Dict]]] = None,
                 state_saver: Optional[Callable[[str, Dict], None]] = None,
                 state_expirer: Optional[Callable[[str, Dict, float], bool]] = None):
        self._latency = simulated_latency_ms
        self._failure_rate = simulated_failure_rate
        self._connected = False
        self._devices: Dict[str, Dict] = {}
        self._history: List[Dict] = []
        self._state_loader = state_loader
        self._state_saver = state_saver
        self._state_expirer = state_expirer
        self._state_lock = threading.RLock()

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

        with self._state_lock:
            self._restore_states()

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
                metadata={**info.get("metadata", {}), "simulated": True},
            ))
        return result

    # ── 指令执行 ──────────────────────────────

    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        await self._simulate_latency()
        with self._state_lock:
            self._restore_states()
            result = self._execute_command(device_id, command)
            if result.success:
                result.execution_id = result.actual_params.get("run_id") if command.command == "start" else command.execution_id
                result.acknowledged = True
                result.retry_safe = True
            return result

    def _execute_command(self, device_id: str, command: DeviceCommand) -> DeviceResult:

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
        previous_state = copy.deepcopy(dev["state"])

        current = dev["state"].get("status", "powered_off")
        name = dev["info"]["name"]

        # ── 通电启动 ──
        if command.command in ("power_on", "boot"):
            if current == "powered_off":
                dev["state"]["power"] = True
                dev["state"]["status"] = "standby"
                dev["state"]["run_until"] = None
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
            if current == "running":
                self._end_run(dev["state"], "interrupted")
            dev["state"]["run_until"] = None
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
            history = dev["state"].get("run_history") or {}
            ended = history.get(command.execution_id) if isinstance(history, dict) else None
            if command.execution_id and isinstance(ended, dict):
                # 哥哥，旧编号的重试只返回原作业信息，不能中断后来启动的作业。
                return DeviceResult(
                    success=True, device_id=device_id, executed_command="start",
                    actual_params={
                        **command.params, "run_id": command.execution_id,
                        "run_started_at": ended.get("started_at"),
                        "run_until": ended.get("run_until"),
                    }, message=f"[模拟] {name} 本轮作业已经结束，保留当前状态",
                )
            if command.execution_id and dev["state"].get("run_id") == command.execution_id:
                return DeviceResult(
                    success=True, device_id=device_id, executed_command="start",
                    actual_params={**command.params, **{key: dev["state"].get(key) for key in (
                        "run_id", "run_started_at", "run_until",
                    )}}, message=f"[模拟] {name} 本轮作业已经执行，保留当前状态",
                )
            duration = command.params.get("duration", command.params.get("duration_minutes"))
            run_until = None
            if duration is not None:
                try:
                    minutes = float(duration)
                    run_until = datetime.now().timestamp() + minutes * 60
                except (TypeError, ValueError, OverflowError):
                    minutes = -1
                if minutes < 0 or run_until is None or not math.isfinite(run_until):
                    return DeviceResult(
                        success=False, device_id=device_id,
                        executed_command=command.command,
                        message="运行时长必须是有限的非负分钟数",
                        error_code="INVALID_PARAMS",
                    )
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

            # 哥哥，截止时间随设备状态一起保存，无时长启动会取消旧计时。
            if current == "running":
                self._end_run(dev["state"], "interrupted")
            dev["state"]["run_id"] = command.execution_id or uuid.uuid4().hex
            dev["state"]["run_started_at"] = datetime.now().isoformat()
            if run_until is not None or "run_until" in dev["state"]:
                dev["state"]["run_until"] = run_until
            if duration is not None:
                dev["state"]["last_duration"] = duration
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
            if command.execution_id and current == "running" and dev["state"].get("run_id") != command.execution_id:
                return DeviceResult(success=False, device_id=device_id, executed_command=command.command,
                                    message="运行编号已变化，不能停止其他作业", error_code="RUN_MISMATCH")
            if current == "running":
                self._end_run(dev["state"], "interrupted")
            dev["state"]["run_until"] = None
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
                dev["state"]["run_until"] = None
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

        self._persist_state(device_id, previous_state)
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
            actual_params={
                **command.params,
                **({key: dev["state"].get(key) for key in (
                    "run_id", "run_started_at", "run_until",
                )} if command.command == "start" else {}),
            },
            message=msg,
        )

    # ── 状态读取 ──────────────────────────────

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        await self._simulate_latency(latency_ms=max(20, self._latency // 2))
        with self._state_lock:
            self._restore_states()
            return self._read_state(device_id)

    def _read_state(self, device_id: str) -> Dict[str, Any]:

        if device_id not in self._devices:
            return {"error": f"设备 '{device_id}' 不存在"}

        internal = self._devices[device_id]["state"]
        previous_state = copy.deepcopy(internal)
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
                # 哥哥，待机只代表通电；实际工作的灌溉设备才会增加水分。
                sensor_location = str(info.get("location") or "").strip()
                sensor_scope = info.get("metadata", {})
                sensor_plot = str(sensor_scope.get("plot_id") or "").strip()
                sensor_zone = str(sensor_scope.get("zone_id") or "").strip()
                irrigation_on = False
                for dev in self._devices.values():
                    if (DeviceCapability.IRRIGATE not in dev["info"]["capabilities"]
                            or dev["state"].get("status") != "running"
                            or dev["state"].get("power") is not True):
                        continue
                    pump_location = str(dev["info"].get("location") or "").strip()
                    pump_scope = dev["info"].get("metadata", {})
                    pump_plot = str(pump_scope.get("plot_id") or "").strip()
                    pump_zone = str(pump_scope.get("zone_id") or "").strip()
                    # 哥哥，明确的地块和分区归属优先，位置名称只兼容旧配置。
                    if sensor_plot and pump_plot:
                        if sensor_plot != pump_plot:
                            continue
                        if sensor_zone and pump_zone and sensor_zone != pump_zone:
                            continue
                    elif sensor_location and pump_location and sensor_location != pump_location:
                        continue
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

        state["_observed_at"] = datetime.now().isoformat()
        state["_quality"] = "good"
        state["_read_at"] = state["_observed_at"]

        # 把所有漂移后的传感器值写回内部状态，确保漂移跨调用累积
        # 遍历 internal 中的所有 key，凡是 state 中也存在的都写回
        for key in list(internal.keys()):
            if key in state and key not in ("_read_at",):
                internal[key] = state[key]

        self._persist_state(device_id, previous_state)

        return state

    # ── 自定义虚拟设备管理 ────────────────────

    def add_virtual_device(self, device_id: str, name: str,
                           capabilities: List[DeviceCapability],
                           sensors: List[str] = None,
                           location: str = "",
                           initial_state: Optional[Dict] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        if initial_state is None:
            initial_state = {"power": False, "status": "powered_off"}
        # 初始化传感器默认值
        sensor_defaults = {
            "temperature": 22.0, "humidity": 65.0, "soil_moisture": 45.0,
            "ph": 7.0, "light_lux": 5000.0, "co2_ppm": 400.0,
            "wind_speed": 1.5, "flow_rate": 0.0, "pressure": 101.3,
        }
        for s in (sensors or []):
            if s in sensor_defaults and s not in initial_state:
                initial_state[s] = sensor_defaults[s]
        self._devices[device_id] = {
            "info": {
                "device_id": device_id,
                "name": name,
                "capabilities": capabilities,
                "sensors": sensors or [],
                "location": location,
                "metadata": copy.deepcopy(metadata or {}),
            },
            "state": copy.deepcopy(initial_state),
        }
        logger.debug("SimulatorDriver: 已添加虚拟设备 %s (%s)", device_id, name)

    def remove_virtual_device(self, device_id: str) -> None:
        self._devices.pop(device_id, None)

    def set_sensor_value(self, device_id: str, field: str, value: Any) -> None:
        with self._state_lock:
            self._restore_states()
            if device_id in self._devices:
                previous_state = copy.deepcopy(self._devices[device_id]["state"])
                self._devices[device_id]["state"][field] = value
                self._persist_state(device_id, previous_state)

    def _restore_states(self) -> None:
        """哥哥，不同请求创建的驱动也要看见已保存的最新模拟状态。"""
        if self._state_loader is not None:
            for device_id, state in self._state_loader().items():
                if device_id in self._devices:
                    self._devices[device_id]["state"].update(copy.deepcopy(state))
        # 哥哥，先校正所有过期执行器，土壤读数才不会继续收到灌溉反馈。
        now = datetime.now().timestamp()
        for device_id, device in self._devices.items():
            state = device["state"]
            deadline = state.get("run_until")
            if (state.get("status") != "running"
                    or not isinstance(deadline, (int, float))
                    or not math.isfinite(deadline) or now < deadline):
                continue
            previous_state = copy.deepcopy(state)
            self._end_run(state, "completed")
            state["status"] = "standby"
            state["run_until"] = None
            for field in ("flow_rate", "rpm"):
                if field in state:
                    state[field] = 0
            if not self._persist_state(device_id, previous_state, expired_deadline=deadline):
                # 哥哥，另一个请求已更新计时，读取它的新状态，不能覆盖新启动。
                if self._state_loader is not None:
                    state.update(copy.deepcopy(self._state_loader().get(device_id, {})))

    @staticmethod
    def _end_run(state: Dict, reason: str) -> None:
        """哥哥，保存最近百轮结束证据，后续运行不能抹掉旧任务的结果。"""
        run_id = state.get("run_id")
        ended_at = datetime.now().isoformat()
        history = state.get("run_history")
        history = dict(history) if isinstance(history, dict) else {}
        if run_id:
            history[str(run_id)] = {
                "reason": reason, "ended_at": ended_at,
                "started_at": state.get("run_started_at"),
                "run_until": state.get("run_until"),
            }
            state["run_history"] = dict(list(history.items())[-100:])
        state["last_ended_run_id"] = run_id
        state["last_run_end_reason"] = reason
        state["last_run_ended_at"] = ended_at

    def _persist_state(self, device_id: str, previous_state: Dict,
                       expired_deadline: Optional[float] = None) -> bool:
        """只保存本次改变的字段，哥哥的读取请求不能覆盖并发执行结果。"""
        if self._state_saver is not None or self._state_expirer is not None:
            changes = {
                key: copy.deepcopy(value)
                for key, value in self._devices[device_id]["state"].items()
                if key not in previous_state or previous_state[key] != value
            }
            if changes:
                if expired_deadline is not None and self._state_expirer is not None:
                    return self._state_expirer(device_id, changes, expired_deadline)
                if self._state_saver is not None:
                    self._state_saver(device_id, changes)
        return True

    def get_history(self, limit: int = 50) -> List[Dict]:
        return self._history[-limit:]

    async def _simulate_latency(self, latency_ms: int = None):
        ms = latency_ms if latency_ms is not None else self._latency
        if ms > 0:
            await asyncio.sleep(ms / 1000.0)
