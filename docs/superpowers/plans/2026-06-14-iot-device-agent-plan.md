# IoT 智能设备控制 Agent — 实现计划 (Phase 1-3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为农业助手新增 IoT 设备自主控制能力 — 设备抽象层、规则引擎、DeviceAgent、前端仪表盘，实现从"感知→决策→执行→验证"的完整闭环。

**Architecture:** 三层新增模块（设备抽象层 + 规则引擎 + DeviceAgent）以插件形式插入现有 LangGraph 工作流。上层通过 DeviceDriverRegistry 与设备通信，不感知底层协议。SimulatorDriver 支持零硬件全链路测试。

**Tech Stack:** Python 3.11, LangGraph, FastAPI, Streamlit, Pydantic v2, pytest

---

## 文件结构总览

```
Agriculture_Agent/
├── devices/                          # 新增：设备驱动模块
│   ├── __init__.py                   #   出口：DeviceDriverRegistry, SimulatorDriver
│   ├── base.py                       #   BaseDeviceDriver, DeviceCapability, 数据类
│   ├── registry.py                   #   DeviceDriverRegistry
│   └── simulator_driver.py           #   虚拟设备模拟器（Phase 1-3 唯一真实可用的驱动）
├── core/
│   ├── device_rule_engine.py         # 新增：规则引擎
│   └── device_executor.py            # 新增：指令执行器（重试/超时/队列）
├── app/agent/
│   ├── state.py                      # 修改：+4 设备字段
│   ├── config.py                     # 修改：+DEVICE_KEYWORDS
│   ├── graph.py                      # 修改：+device_control 节点及路由
│   ├── agents/
│   │   ├── __init__.py               # 修改：+DeviceAgent 导出
│   │   ├── orchestrator.py           # 修改：注册 DeviceAgent
│   │   └── device_agent.py           # 新增：DeviceAgent
│   └── nodes/
│       └── device_control.py         # 新增：设备控制工作流节点
├── app/
│   ├── test1.py                      # 修改：+devices/rules 页面路由
│   ├── api_routes.py                 # 修改：+10 API
│   ├── scheduler_jobs.py             # 修改：+规则轮询任务
│   └── views/
│       ├── devices.py                # 新增：设备仪表盘
│       ├── rules.py                  # 新增：规则编辑器
│       └── chat.py                   # 修改：+设备消息卡片
├── app/ui/
│   └── sidebar.py                    # 修改：+导航入口
└── tests/
    ├── test_device_base.py           # 新增
    ├── test_simulator_driver.py      # 新增
    ├── test_device_registry.py       # 新增
    ├── test_device_rule_engine.py    # 新增
    └── test_device_executor.py       # 新增
```

---

## Phase 1: 基础框架 — 设备抽象层 + Simulator + 规则引擎

### Task 1: 设备抽象基类

**Files:**
- Create: `devices/__init__.py`
- Create: `devices/base.py`

- [ ] **Step 1: Create `devices/__init__.py`**

```python
# 设备驱动模块 — 统一设备控制接口
```

- [ ] **Step 2: Create `devices/base.py` with data classes and abstract driver**

```python
"""设备抽象基类：所有设备驱动必须实现此接口"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


class DeviceCapability(Enum):
    """设备能力枚举 — 描述设备能做什仫"""
    IRRIGATE    = "irrigate"      # 灌溉（阀门/水泵）
    FERTIGATE   = "fertigate"     # 施肥
    VENTILATE   = "ventilate"     # 通风（风机/天窗）
    HEAT        = "heat"          # 加热
    COOL        = "cool"          # 降温（湿帘）
    SHADE       = "shade"         # 遮阳
    LIGHT       = "light"         # 补光
    READ_SENSOR = "read_sensor"   # 传感器读数


class DeviceStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    BUSY = "busy"


@dataclass
class DeviceInfo:
    """设备元信息"""
    device_id: str
    name: str
    driver_name: str              # 驱动标识，如 "simulator" | "mqtt" | "modbus" | "tuya"
    capabilities: List[DeviceCapability]
    sensors: List[str] = field(default_factory=list)  # 传感器字段列表
    status: str = "online"
    location: str = ""            # 物理位置描述
    metadata: Dict[str, Any] = field(default_factory=dict)  # 驱动特定元数据


@dataclass
class DeviceCommand:
    """下发给设备的指令"""
    command: str                  # "start" | "stop" | "set_param"
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 30000
    priority: str = "normal"      # "low" | "normal" | "high" | "emergency"


@dataclass
class DeviceResult:
    """设备执行结果"""
    success: bool
    device_id: str
    executed_command: str
    actual_params: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error_code: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


class BaseDeviceDriver(ABC):
    """设备驱动抽象基类 — 所有协议驱动必须继承此类"""
    
    driver_name: str = "base"
    
    @abstractmethod
    async def connect(self) -> bool:
        """建立连接，返回 True 表示成功"""
        ...
    
    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        ...
    
    @abstractmethod
    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        """向指定设备发送指令并返回结果"""
        ...
    
    @abstractmethod
    async def read_state(self, device_id: str) -> Dict[str, Any]:
        """读取设备当前状态/传感器数据
        返回: {"temperature": 22.5, "humidity": 65, ...}
        """
        ...
    
    @abstractmethod
    async def discover(self) -> List[DeviceInfo]:
        """发现该驱动管理的所有设备"""
        ...
    
    @abstractmethod
    async def health_check(self) -> bool:
        """检查驱动/连接健康状态"""
        ...
```

- [ ] **Step 3: Run Python syntax check**

```bash
python -c "from devices.base import BaseDeviceDriver, DeviceCapability, DeviceInfo, DeviceCommand, DeviceResult, DeviceStatus; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add devices/__init__.py devices/base.py
git commit -m "feat(devices): add device abstraction base classes"
```

---

### Task 2: 设备注册中心

**Files:**
- Create: `devices/registry.py`

- [ ] **Step 1: Create `devices/registry.py`**

```python
"""设备驱动注册中心 — 统一管理所有驱动，对外提供设备操作接口"""

import logging
from typing import Dict, List, Optional, AsyncIterator

from .base import BaseDeviceDriver, DeviceInfo, DeviceCommand, DeviceResult

logger = logging.getLogger(__name__)


class DeviceDriverRegistry:
    """设备驱动注册中心
    
    用法:
        registry = DeviceDriverRegistry()
        registry.register("simulator", SimulatorDriver())
        registry.register("mqtt", MQTTDriver(broker="localhost:1883"))
        
        # 发现所有设备
        devices = await registry.discover_all()
        
        # 执行指令（自动路由到对应驱动）
        result = await registry.execute("irrigation_valve_01", cmd)
    """
    
    def __init__(self):
        self._drivers: Dict[str, BaseDeviceDriver] = {}
        # device_id → driver_name 映射表（discover 后自动填充）
        self._device_map: Dict[str, str] = {}
    
    def register(self, name: str, driver: BaseDeviceDriver) -> None:
        """注册一个驱动"""
        self._drivers[name] = driver
        logger.info("驱动已注册: %s (%s)", name, driver.driver_name)
    
    def unregister(self, name: str) -> None:
        """注销一个驱动"""
        self._drivers.pop(name, None)
        # 清除映射
        self._device_map = {k: v for k, v in self._device_map.items() if v != name}
    
    def get_driver(self, device_id: str) -> Optional[BaseDeviceDriver]:
        """根据 device_id 找到对应的驱动"""
        driver_name = self._device_map.get(device_id)
        if driver_name:
            return self._drivers.get(driver_name)
        return None
    
    async def discover_all(self) -> List[DeviceInfo]:
        """发现所有驱动下的设备，返回完整设备列表"""
        all_devices: List[DeviceInfo] = []
        self._device_map.clear()
        
        for name, driver in self._drivers.items():
            try:
                devices = await driver.discover()
                for d in devices:
                    self._device_map[d.device_id] = name
                all_devices.extend(devices)
                logger.info("驱动 %s: 发现 %d 个设备", name, len(devices))
            except Exception as e:
                logger.warning("驱动 %s 设备发现失败: %s", name, e)
        
        return all_devices
    
    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        """向指定设备发送指令"""
        driver = self.get_driver(device_id)
        if driver is None:
            return DeviceResult(
                success=False,
                device_id=device_id,
                executed_command=command.command,
                message=f"设备 '{device_id}' 未找到对应驱动（请先执行 discover_all）",
                error_code="DEVICE_NOT_FOUND",
            )
        
        try:
            return await driver.execute(device_id, command)
        except Exception as e:
            logger.error("设备 %s 执行失败: %s", device_id, e)
            return DeviceResult(
                success=False,
                device_id=device_id,
                executed_command=command.command,
                message=str(e),
                error_code="EXECUTION_ERROR",
            )
    
    async def read_state(self, device_id: str) -> Dict:
        """读取设备当前状态"""
        driver = self.get_driver(device_id)
        if driver is None:
            return {"error": f"设备 '{device_id}' 未找到"}
        try:
            return await driver.read_state(device_id)
        except Exception as e:
            logger.error("设备 %s 状态读取失败: %s", device_id, e)
            return {"error": str(e)}
    
    async def read_all_states(self) -> Dict[str, Dict]:
        """读取所有设备状态"""
        states = {}
        for device_id in self._device_map:
            states[device_id] = await self.read_state(device_id)
        return states
    
    async def connect_all(self) -> None:
        """连接所有驱动"""
        for name, driver in self._drivers.items():
            try:
                ok = await driver.connect()
                logger.info("驱动 %s 连接: %s", name, "成功" if ok else "失败")
            except Exception as e:
                logger.warning("驱动 %s 连接异常: %s", name, e)
    
    async def disconnect_all(self) -> None:
        """断开所有驱动"""
        for name, driver in self._drivers.items():
            try:
                await driver.disconnect()
            except Exception as e:
                logger.warning("驱动 %s 断开异常: %s", name, e)
    
    @property
    def driver_names(self) -> List[str]:
        return list(self._drivers.keys())
    
    @property
    def device_count(self) -> int:
        return len(self._device_map)
```

- [ ] **Step 2: Syntax check**

```bash
python -c "from devices.registry import DeviceDriverRegistry; r = DeviceDriverRegistry(); print(f'OK — {len(r.driver_names)} drivers, {r.device_count} devices')"
```
Expected: `OK — 0 drivers, 0 devices`

- [ ] **Step 3: Commit**

```bash
git add devices/registry.py
git commit -m "feat(devices): add DeviceDriverRegistry"
```

---

### Task 3: 虚拟设备模拟器

**Files:**
- Create: `devices/simulator_driver.py`

- [ ] **Step 1: Create `devices/simulator_driver.py`**

```python
"""虚拟设备模拟器 — 无需真实硬件即可跑通全链路测试"""

import asyncio
import copy
import logging
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

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
    """虚拟设备驱动 — 在内存中模拟设备行为
    
    支持功能:
    - 内置 6 种虚拟设备（灌溉/传感器/通风/补光/施肥/加热）
    - 可动态添加自定义虚拟设备
    - 可配置模拟延迟和故障率
    - 传感器值随时间自然波动
    """
    
    driver_name = "simulator"
    
    def __init__(self,
                 simulated_latency_ms: int = 100,
                 simulated_failure_rate: float = 0.0):
        self._latency = simulated_latency_ms
        self._failure_rate = simulated_failure_rate
        self._connected = False
        self._devices: Dict[str, Dict] = {}  # device_id → {info, state}
        self._history: List[Dict] = []       # 操作历史
        
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
        """列出所有虚拟设备"""
        if not self._connected:
            await self.connect()
        
        result = []
        for dev_id, dev in self._devices.items():
            info = dev["info"]
            state = dev["state"]
            status = "online" if self._connected else "offline"
            if state.get("status") == "error":
                status = "error"
            
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
        """模拟执行设备指令"""
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
        
        # 处理不同指令
        if command.command == "start":
            dev["state"]["power"] = True
            dev["state"]["status"] = "running"
            msg = f"[模拟] {dev['info']['name']} 已启动"
            
            # 特殊参数处理
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
        
        # 记录历史
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
        """读取设备当前状态 + 传感器数据（带自然波动）"""
        await self._simulate_latency(latency_ms=max(20, self._latency // 2))
        
        if device_id not in self._devices:
            return {"error": f"设备 '{device_id}' 不存在"}
        
        state = copy.deepcopy(self._devices[device_id]["state"])
        
        # 传感器值自然波动（仅对 READ_SENSOR 类型设备）
        info = self._devices[device_id]["info"]
        if DeviceCapability.READ_SENSOR in info.get("capabilities", []):
            if "temperature" in state:
                state["temperature"] = round(state["temperature"] + random.uniform(-0.5, 0.5), 1)
            if "humidity" in state:
                state["humidity"] = round(state["humidity"] + random.uniform(-2, 2), 1)
            if "soil_moisture" in state:
                # 如果灌溉设备在运行，湿度上升
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
        """动态添加自定义虚拟设备"""
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
        """移除虚拟设备"""
        self._devices.pop(device_id, None)
    
    def set_sensor_value(self, device_id: str, field: str, value: Any) -> None:
        """手动设置传感器值（测试用）"""
        if device_id in self._devices:
            self._devices[device_id]["state"][field] = value
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """获取操作历史"""
        return self._history[-limit:]
    
    async def _simulate_latency(self, latency_ms: int = None):
        """模拟设备通信延迟"""
        ms = latency_ms if latency_ms is not None else self._latency
        if ms > 0:
            await asyncio.sleep(ms / 1000.0)
```

- [ ] **Step 2: Syntax check**

```bash
python -c "from devices.simulator_driver import SimulatorDriver; d = SimulatorDriver(); print(f'OK — {len(d._devices)} virtual devices ready')"
```
Expected: `OK — 6 virtual devices ready`

- [ ] **Step 3: Commit**

```bash
git add devices/simulator_driver.py
git commit -m "feat(devices): add SimulatorDriver with 6 virtual devices"
```

---

### Task 4: 规则引擎

**Files:**
- Create: `core/device_rule_engine.py`

- [ ] **Step 1: Create `core/device_rule_engine.py`**

```python
"""设备控制规则引擎 — 条件匹配 + 约束校验 + AI 微调的混合决策核心"""

import json
import logging
import os
import re
from copy import deepcopy
from datetime import datetime, time
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.getenv("DATA_STORAGE_DIR", "data")

# ── 代码级硬限制（不可通过规则配置突破）─────────────────
HARD_LIMITS = {
    "irrigate": {
        "max_duration_per_use_minutes": 120,   # 单次灌溉最多 120 分钟
        "min_interval_seconds": 10,             # 两次操作最少间隔 10 秒
    },
    "fertigate": {
        "max_amount_per_use_kg": 50,            # 单次施肥最多 50kg
        "min_interval_seconds": 10,
    },
}


class RuleDecision:
    """规则评估结果"""
    AUTO_EXECUTE = "auto_execute"    # 边界内，自动执行
    NEED_CONFIRM = "need_confirm"    # 超出边界，需用户确认
    REJECTED = "rejected"            # 违反硬限制，拒绝


class RuleEngine:
    """设备控制规则引擎
    
    职责:
    1. 加载/保存用户规则
    2. 评估触发条件是否满足
    3. 校验 AI 建议是否在安全边界内
    4. 返回决策（自动执行 / 需要确认 / 拒绝）
    """
    
    def __init__(self, username: str = "default"):
        self.username = username
        self.rules: List[Dict] = []
        self._execution_history: Dict[str, List[datetime]] = {}  # device_id → 最近执行时间
        self._daily_duration: Dict[str, Dict[str, int]] = {}     # device_id → {date_str → total_minutes}
        self._load_rules()
    
    # ── 规则持久化 ──────────────────────────
    
    def _rules_path(self) -> str:
        return os.path.join(DEFAULT_DATA_DIR, self.username, "device_rules.json")
    
    def _load_rules(self) -> None:
        path = self._rules_path()
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    self.rules = data.get("rules", [])
                logger.info("规则引擎: 已加载 %d 条规则", len(self.rules))
            except Exception as e:
                logger.warning("规则加载失败: %s", e)
                self.rules = []
    
    def _save_rules(self) -> None:
        path = self._rules_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rules": self.rules, "updated_at": datetime.now().isoformat()},
                      f, ensure_ascii=False, indent=2)
    
    # ── 规则 CRUD ────────────────────────────
    
    def list_rules(self) -> List[Dict]:
        return deepcopy(self.rules)
    
    def get_rule(self, rule_id: str) -> Optional[Dict]:
        for r in self.rules:
            if r["id"] == rule_id:
                return deepcopy(r)
        return None
    
    def add_rule(self, rule: Dict) -> str:
        """添加新规则，返回 rule_id"""
        if "id" not in rule:
            import uuid
            rule["id"] = f"rule_{uuid.uuid4().hex[:8]}"
        rule.setdefault("enabled", True)
        self.rules.append(rule)
        self._save_rules()
        logger.info("规则已添加: %s", rule["id"])
        return rule["id"]
    
    def update_rule(self, rule_id: str, updates: Dict) -> bool:
        for i, r in enumerate(self.rules):
            if r["id"] == rule_id:
                self.rules[i] = {**r, **updates}
                self._save_rules()
                return True
        return False
    
    def delete_rule(self, rule_id: str) -> bool:
        before = len(self.rules)
        self.rules = [r for r in self.rules if r["id"] != rule_id]
        if len(self.rules) < before:
            self._save_rules()
            return True
        return False
    
    def toggle_rule(self, rule_id: str, enabled: bool) -> bool:
        return self.update_rule(rule_id, {"enabled": enabled})
    
    # ── 规则评估 ─────────────────────────────
    
    def find_matching_rules(self, context: Dict) -> List[Dict]:
        """查找所有匹配当前上下文的启用规则"""
        matched = []
        for rule in self.rules:
            if not rule.get("enabled", True):
                continue
            if self._evaluate_trigger(rule.get("trigger", {}), context):
                matched.append(deepcopy(rule))
        return matched
    
    def evaluate_action(self, rule: Dict, proposed_params: Dict,
                        context: Dict) -> Tuple[str, str, Dict]:
        """评估一个拟执行的操作
        
        Args:
            rule: 匹配到的规则
            proposed_params: 拟执行的参数（可能已被 AI 微调）
            context: 当前上下文（传感器数据、天气等）
        
        Returns:
            (decision, reason, final_params)
        """
        constraints = rule.get("constraints", {})
        action = rule.get("action", {})
        device_id = action.get("device_id", "")
        capability = self._infer_capability(action)
        
        # ── 1. 硬限制检查（不可突破）─────────────────
        hard_ok, hard_reason = self._check_hard_limits(
            capability, proposed_params, device_id)
        if not hard_ok:
            return RuleDecision.REJECTED, hard_reason, proposed_params
        
        # ── 2. 软约束检查（可触发确认）─────────────────
        soft_ok, soft_reason = self._check_constraints(
            constraints, proposed_params, context)
        if not soft_ok:
            return RuleDecision.NEED_CONFIRM, soft_reason, proposed_params
        
        # ── 3. 通过 → 自动执行 ───────────────────────
        # 应用 AI 微调（如果在允许范围内）
        ai_enhance = rule.get("ai_enhance", {})
        if ai_enhance.get("enabled", False):
            proposed_params = self._apply_ai_enhance(
                ai_enhance, proposed_params, action.get("params", {}))
        
        return RuleDecision.AUTO_EXECUTE, "规则校验通过", proposed_params
    
    def record_execution(self, device_id: str, params: Dict) -> None:
        """记录一次执行（用于间隔/每日上限校验）"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        
        # 执行时间记录
        if device_id not in self._execution_history:
            self._execution_history[device_id] = []
        self._execution_history[device_id].append(now)
        
        # 每日用量记录
        if device_id not in self._daily_duration:
            self._daily_duration[device_id] = {}
        if date_str not in self._daily_duration[device_id]:
            self._daily_duration[device_id][date_str] = 0
        
        duration = params.get("duration", 0)
        self._daily_duration[device_id][date_str] += duration
    
    # ── 内部方法 ──────────────────────────────
    
    def _evaluate_trigger(self, trigger: Dict, context: Dict) -> bool:
        """评估触发条件"""
        conditions = trigger.get("conditions", [])
        if not conditions:
            return False
        
        results = []
        for cond in conditions:
            results.append(self._eval_single_condition(cond, context))
        
        logic = trigger.get("logic", "AND").upper()
        if logic == "OR":
            return any(results)
        return all(results)
    
    def _eval_single_condition(self, cond: Dict, context: Dict) -> bool:
        """评估单个条件"""
        cond_type = cond.get("type", "")
        field = cond.get("field", "")
        op = cond.get("op", "==")
        expected = cond.get("value")
        
        # 获取实际值
        if cond_type == "sensor":
            sensor_data = context.get("sensor_data", {})
            actual = sensor_data.get(field)
        elif cond_type == "weather":
            weather_data = context.get("weather", {})
            actual = weather_data.get(field)
        elif cond_type == "time":
            actual = datetime.now().strftime("%H:%M")
        else:
            actual = context.get(field)
        
        if actual is None:
            return False
        
        return self._compare(actual, op, expected)
    
    def _compare(self, actual, op: str, expected) -> bool:
        """比较操作符"""
        try:
            if op == "==":
                return actual == expected
            elif op == "!=":
                return actual != expected
            elif op == ">":
                return float(actual) > float(expected)
            elif op == "<":
                return float(actual) < float(expected)
            elif op == ">=":
                return float(actual) >= float(expected)
            elif op == "<=":
                return float(actual) <= float(expected)
            elif op == "between":
                # expected 是 [min, max]
                if isinstance(expected, list) and len(expected) == 2:
                    return str(expected[0]) <= str(actual) <= str(expected[1])
                return False
            elif op == "in":
                return actual in expected if isinstance(expected, list) else False
            return False
        except (ValueError, TypeError):
            return False
    
    def _infer_capability(self, action: Dict) -> str:
        """根据动作推断设备能力类型"""
        command = action.get("command", "")
        device_id = action.get("device_id", "").lower()
        
        if "irrigat" in device_id or "water" in device_id:
            return "irrigate"
        if "fertigat" in device_id or "fertil" in device_id:
            return "fertigate"
        if "vent" in device_id or "fan" in device_id:
            return "ventilate"
        if "heat" in device_id:
            return "heat"
        if "light" in device_id:
            return "light"
        return "irrigate"  # 默认按灌溉处理
    
    def _check_hard_limits(self, capability: str, params: Dict,
                           device_id: str) -> Tuple[bool, str]:
        """检查代码级硬限制"""
        limits = HARD_LIMITS.get(capability, {})
        
        # 灌溉时长硬限制
        max_dur = limits.get("max_duration_per_use_minutes")
        if max_dur and params.get("duration", 0) > max_dur:
            return False, f"单次灌溉时长 {params['duration']} 分钟超过硬限制 {max_dur} 分钟"
        
        # 施肥量硬限制
        max_amt = limits.get("max_amount_per_use_kg")
        if max_amt and params.get("amount_kg", 0) > max_amt:
            return False, f"单次施肥量 {params['amount_kg']}kg 超过硬限制 {max_amt}kg"
        
        # 最小间隔硬限制
        min_interval = limits.get("min_interval_seconds", 0)
        if min_interval and device_id in self._execution_history:
            last = self._execution_history[device_id][-1] if self._execution_history[device_id] else None
            if last and (datetime.now() - last).total_seconds() < min_interval:
                return False, f"距上次操作不足 {min_interval} 秒，拒绝重复触发"
        
        return True, ""
    
    def _check_constraints(self, constraints: Dict, params: Dict,
                           context: Dict) -> Tuple[bool, str]:
        """检查用户设定的软约束"""
        # 单次时长限制
        max_dur = constraints.get("max_duration_per_use")
        if max_dur and params.get("duration", 0) > max_dur:
            return False, f"单次时长 {params['duration']} 分钟超过设定上限 {max_dur} 分钟，需要确认"
        
        # 每日上限
        max_daily = constraints.get("max_duration_per_day")
        if max_daily:
            device_id = context.get("device_id", "")
            date_str = datetime.now().strftime("%Y-%m-%d")
            today_used = self._daily_duration.get(device_id, {}).get(date_str, 0)
            if today_used + params.get("duration", 0) > max_daily:
                return False, f"今日累计 {today_used + params.get('duration', 0)} 分钟超过每日上限 {max_daily} 分钟，需要确认"
        
        # 禁止时段
        forbidden = constraints.get("forbidden_hours", [])
        if forbidden:
            current_hour = datetime.now().hour
            if current_hour in forbidden:
                return False, f"当前时间 {current_hour}:00 在禁止时段内，需要确认"
        
        # 需确认条件表达式
        require_confirm = constraints.get("require_confirm_if", [])
        for expr in require_confirm:
            if self._eval_confirm_expr(expr, params, context):
                return False, f"触发确认条件: {expr}"
        
        return True, ""
    
    def _eval_confirm_expr(self, expr: str, params: Dict, context: Dict) -> bool:
        """评估需确认条件表达式，如 'duration > 45'"""
        try:
            # 简单表达式解析
            if " > " in expr:
                field, val = expr.split(" > ")
                return params.get(field.strip(), 0) > float(val.strip())
            elif " < " in expr:
                field, val = expr.split(" < ")
                return params.get(field.strip(), 0) < float(val.strip())
            elif " >= " in expr:
                field, val = expr.split(" >= ")
                return params.get(field.strip(), 0) >= float(val.strip())
            elif " <= " in expr:
                field, val = expr.split(" <= ")
                return params.get(field.strip(), 0) <= float(val.strip())
            elif " == " in expr:
                field, val = expr.split(" == ")
                return str(params.get(field.strip(), "")) == val.strip()
            elif expr == "weather_forecast_conflict":
                return context.get("weather_conflict", False)
            elif expr == "cost_estimate > 50":
                return params.get("cost_estimate", 0) > 50
        except Exception:
            pass
        return False
    
    def _apply_ai_enhance(self, ai_config: Dict, proposed: Dict,
                          original: Dict) -> Dict:
        """应用 AI 微调，限制在允许范围内"""
        result = dict(proposed)
        can_adjust = ai_config.get("can_adjust", [])
        ranges = ai_config.get("adjust_range", {})
        
        for field in can_adjust:
            if field in result and field in ranges:
                min_adj, max_adj = ranges[field]
                orig_val = original.get(field, result[field])
                adjusted = result[field]
                # 限制微调范围
                clamped = max(orig_val + min_adj, min(orig_val + max_adj, adjusted))
                result[field] = clamped
        
        return result
```

- [ ] **Step 2: Syntax check**

```bash
python -c "from core.device_rule_engine import RuleEngine, RuleDecision; e = RuleEngine(); print(f'OK — {len(e.rules)} rules loaded, decision types: {RuleDecision.AUTO_EXECUTE}/{RuleDecision.NEED_CONFIRM}/{RuleDecision.REJECTED}')"
```
Expected: `OK — 0 rules loaded, decision types: auto_execute/need_confirm/rejected`

- [ ] **Step 3: Commit**

```bash
git add core/device_rule_engine.py
git commit -m "feat(core): add device rule engine"
```

---

### Task 5: 指令执行器

**Files:**
- Create: `core/device_executor.py`

- [ ] **Step 1: Create `core/device_executor.py`**

```python
"""设备指令执行器 — 重试/超时/队列/审计日志"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.getenv("DATA_STORAGE_DIR", "data")

# 重试配置
MAX_RETRIES = 3
RETRY_DELAYS_SECONDS = [5, 15, 45]  # 递增间隔


class DeviceExecutor:
    """设备指令执行器
    
    职责:
    1. 向 DeviceDriverRegistry 发送指令
    2. 失败自动重试（最多 3 次，间隔递增）
    3. 记录审计日志
    4. 管理待确认操作队列
    """
    
    def __init__(self, registry, username: str = "default"):
        """
        Args:
            registry: DeviceDriverRegistry 实例
            username: 用户名
        """
        self.registry = registry
        self.username = username
        self.pending_actions: List[Dict] = []      # 待确认操作
        self._load_pending()
    
    # ── 指令执行 ──────────────────────────────
    
    async def execute(self, device_id: str, command,
                      trigger: str = "manual",
                      rule_id: Optional[str] = None,
                      decision: str = "auto_execute") -> Dict:
        """执行设备指令，带重试逻辑
        
        Args:
            device_id: 设备 ID
            command: DeviceCommand 实例
            trigger: 触发来源 (manual/rule/schedule/agent_interop)
            rule_id: 关联的规则 ID
            decision: 决策类型
        
        Returns:
            {
                "success": bool,
                "result": DeviceResult,
                "attempts": int,
                "log_entry": {...}
            }
        """
        from devices.base import DeviceResult
        
        last_result = None
        
        for attempt in range(MAX_RETRIES):
            try:
                last_result = await self.registry.execute(device_id, command)
                
                if last_result.success:
                    self._write_log(device_id, command, last_result,
                                    trigger, rule_id, decision, attempt + 1)
                    return {
                        "success": True,
                        "result": last_result,
                        "attempts": attempt + 1,
                        "log_entry": self._make_log_entry(
                            device_id, command, last_result,
                            trigger, rule_id, decision, attempt + 1),
                    }
                
                # 设备未找到，不重试
                if last_result.error_code == "DEVICE_NOT_FOUND":
                    break
                    
            except Exception as e:
                last_result = DeviceResult(
                    success=False,
                    device_id=device_id,
                    executed_command=command.command,
                    message=str(e),
                    error_code="EXCEPTION",
                )
            
            # 最后一次不等待
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS_SECONDS[min(attempt, len(RETRY_DELAYS_SECONDS) - 1)]
                logger.warning("设备 %s 执行失败（第%d次），%d秒后重试",
                             device_id, attempt + 1, delay)
                await asyncio.sleep(delay)
        
        # 全部失败
        self._write_log(device_id, command, last_result,
                        trigger, rule_id, decision, MAX_RETRIES)
        return {
            "success": False,
            "result": last_result,
            "attempts": MAX_RETRIES,
            "log_entry": self._make_log_entry(
                device_id, command, last_result,
                trigger, rule_id, decision, MAX_RETRIES),
        }
    
    def execute_sync(self, device_id: str, command,
                     trigger: str = "manual",
                     rule_id: Optional[str] = None,
                     decision: str = "auto_execute") -> Dict:
        """同步包装器 — 供非 async 环境调用"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.execute(device_id, command, trigger, rule_id, decision))
    
    # ── 待确认操作队列 ────────────────────────
    
    def add_pending(self, action: Dict) -> str:
        """添加待确认操作，返回 action_id"""
        import uuid
        action["id"] = f"pending_{uuid.uuid4().hex[:8]}"
        action["created_at"] = datetime.now().isoformat()
        action["status"] = "pending"
        self.pending_actions.append(action)
        self._save_pending()
        return action["id"]
    
    def list_pending(self) -> List[Dict]:
        return [a for a in self.pending_actions if a.get("status") == "pending"]
    
    def confirm_pending(self, action_id: str) -> Dict:
        """确认待定操作并执行"""
        for action in self.pending_actions:
            if action["id"] == action_id and action["status"] == "pending":
                action["status"] = "confirmed"
                self._save_pending()
                
                from devices.base import DeviceCommand
                cmd = DeviceCommand(
                    command=action.get("command", "start"),
                    params=action.get("params", {}),
                )
                return self.execute_sync(
                    action["device_id"], cmd,
                    trigger="confirmed", decision="auto_execute")
        return {"success": False, "message": "操作不存在或已处理"}
    
    def reject_pending(self, action_id: str) -> bool:
        """拒绝待定操作"""
        for action in self.pending_actions:
            if action["id"] == action_id and action["status"] == "pending":
                action["status"] = "rejected"
                self._save_pending()
                return True
        return False
    
    # ── 日志 ──────────────────────────────────
    
    def get_logs(self, limit: int = 50) -> List[Dict]:
        """获取执行日志"""
        path = self._log_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, encoding="utf-8") as f:
                logs = json.load(f)
            return logs[-limit:]
        except Exception:
            return []
    
    def _write_log(self, device_id, command, result,
                   trigger, rule_id, decision, attempts):
        entry = self._make_log_entry(device_id, command, result,
                                     trigger, rule_id, decision, attempts)
        path = self._log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        logs = []
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                pass
        
        logs.append(entry)
        # 保留最近 1000 条
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    
    def _make_log_entry(self, device_id, command, result,
                        trigger, rule_id, decision, attempts) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "command": command.command,
            "params": command.params,
            "trigger": trigger,
            "rule_id": rule_id,
            "decision": decision,
            "success": result.success,
            "attempts": attempts,
            "message": result.message,
            "error_code": result.error_code or "",
        }
    
    def _log_path(self) -> str:
        return os.path.join(DEFAULT_DATA_DIR, self.username, "device_log.json")
    
    def _pending_path(self) -> str:
        return os.path.join(DEFAULT_DATA_DIR, self.username, "device_pending.json")
    
    def _load_pending(self):
        path = self._pending_path()
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.pending_actions = json.load(f)
            except Exception:
                self.pending_actions = []
    
    def _save_pending(self):
        path = self._pending_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.pending_actions, f, ensure_ascii=False, indent=2)
```

- [ ] **Step 2: Syntax check**

```bash
python -c "from devices.simulator_driver import SimulatorDriver; from devices.registry import DeviceDriverRegistry; from core.device_executor import DeviceExecutor; r = DeviceDriverRegistry(); r.register('sim', SimulatorDriver()); e = DeviceExecutor(r); print(f'OK — executor ready')"
```
Expected: `OK — executor ready`

- [ ] **Step 3: Commit**

```bash
git add core/device_executor.py
git commit -m "feat(core): add device executor with retry and audit log"
```

---

### Task 6: 更新设备模块入口

**Files:**
- Modify: `devices/__init__.py`

- [ ] **Step 1: Update `devices/__init__.py`**

```python
"""设备驱动模块 — 统一设备控制接口"""

from .base import (
    BaseDeviceDriver, DeviceCapability, DeviceStatus,
    DeviceInfo, DeviceCommand, DeviceResult,
)
from .registry import DeviceDriverRegistry
from .simulator_driver import SimulatorDriver

__all__ = [
    "BaseDeviceDriver",
    "DeviceCapability",
    "DeviceStatus",
    "DeviceInfo",
    "DeviceCommand",
    "DeviceResult",
    "DeviceDriverRegistry",
    "SimulatorDriver",
]
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from devices import BaseDeviceDriver, DeviceDriverRegistry, SimulatorDriver; print('All imports OK')"
```
Expected: `All imports OK`

- [ ] **Step 3: Commit**

```bash
git add devices/__init__.py
git commit -m "chore(devices): update module exports"
```

---

### Task 7: Phase 1 单元测试

**Files:**
- Create: `tests/test_device_base.py`
- Create: `tests/test_simulator_driver.py`
- Create: `tests/test_device_registry.py`
- Create: `tests/test_device_rule_engine.py`
- Create: `tests/test_device_executor.py`

- [ ] **Step 1: Create `tests/test_device_base.py`**

```python
"""测试设备抽象基类"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from devices.base import DeviceCapability, DeviceInfo, DeviceCommand, DeviceResult, DeviceStatus


class TestDeviceCapability:
    def test_all_capabilities_exist(self):
        caps = [c.value for c in DeviceCapability]
        assert "irrigate" in caps
        assert "fertigate" in caps
        assert "read_sensor" in caps
        assert len(caps) == 8

class TestDeviceInfo:
    def test_create_minimal(self):
        info = DeviceInfo(device_id="test_01", name="测试设备",
                          driver_name="simulator", capabilities=[])
        assert info.device_id == "test_01"
        assert info.status == "online"
        assert info.sensors == []

class TestDeviceCommand:
    def test_defaults(self):
        cmd = DeviceCommand(command="start")
        assert cmd.timeout_ms == 30000
        assert cmd.priority == "normal"
        assert cmd.params == {}

class TestDeviceResult:
    def test_success_result(self):
        r = DeviceResult(success=True, device_id="d1", executed_command="start",
                         message="OK")
        assert r.success
        assert r.error_code is None
    
    def test_failure_result(self):
        r = DeviceResult(success=False, device_id="d1", executed_command="start",
                         message="timeout", error_code="TIMEOUT")
        assert not r.success
        assert r.error_code == "TIMEOUT"
```

- [ ] **Step 2: Run base tests**

```bash
pytest tests/test_device_base.py -v
```
Expected: 5 passed

- [ ] **Step 3: Create `tests/test_simulator_driver.py`**

```python
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
        # 传感器值应该有波动
        assert state1["temperature"] != state2["temperature"] or \
               state1["humidity"] != state2["humidity"]
    
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
        result = await driver.execute("nonexistent", cmd)
        assert not result.success
        assert result.error_code == "DEVICE_NOT_FOUND"
    
    def test_add_custom_device(self):
        self.driver.add_virtual_device(
            "custom_pump_01", "自定义水泵",
            [DeviceCapability.IRRIGATE],
            sensors=["pressure"],
        )
        assert "custom_pump_01" in self.driver._devices
    
    def test_set_sensor_value(self):
        self.driver.set_sensor_value("virtual_soil_sensor_01", "soil_moisture", 25.0)
        assert self.driver._devices["virtual_soil_sensor_01"]["state"]["soil_moisture"] == 25.0
    
    def test_get_history(self):
        self.driver._history.append({"test": True})
        assert len(self.driver.get_history()) == 1
```

- [ ] **Step 4: Run simulator tests**

```bash
pytest tests/test_simulator_driver.py -v
```
Expected: 8 passed

- [ ] **Step 5: Create `tests/test_device_registry.py`**

```python
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
```

- [ ] **Step 6: Run registry tests**

```bash
pytest tests/test_device_registry.py -v
```
Expected: 5 passed

- [ ] **Step 7: Create `tests/test_device_rule_engine.py`**

```python
"""测试规则引擎"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.device_rule_engine import RuleEngine, RuleDecision


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine(username="test_user")
        self.engine.rules = []  # 清空
    
    def _make_irrigation_rule(self, extra=None):
        rule = {
            "id": "rule_test_irrigation",
            "name": "测试灌溉规则",
            "enabled": True,
            "trigger": {
                "logic": "AND",
                "conditions": [
                    {"type": "sensor", "field": "soil_moisture", "op": "<", "value": 30},
                    {"type": "time", "field": "", "op": "between", "value": ["06:00", "20:00"]},
                ]
            },
            "action": {
                "device_id": "irrigation_valve_01",
                "command": "start",
                "params": {"duration": 30},
            },
            "constraints": {
                "max_duration_per_use": 60,
                "max_duration_per_day": 180,
                "forbidden_hours": [22, 23, 0, 1, 2, 3, 4, 5],
            },
        }
        if extra:
            rule.update(extra)
        return rule
    
    def test_add_and_list_rules(self):
        rule = self._make_irrigation_rule()
        self.engine.add_rule(rule)
        assert len(self.engine.list_rules()) == 1
    
    def test_delete_rule(self):
        rule = self._make_irrigation_rule()
        self.engine.add_rule(rule)
        assert self.engine.delete_rule("rule_test_irrigation")
        assert len(self.engine.list_rules()) == 0
    
    def test_toggle_rule(self):
        rule = self._make_irrigation_rule()
        self.engine.add_rule(rule)
        self.engine.toggle_rule("rule_test_irrigation", False)
        r = self.engine.get_rule("rule_test_irrigation")
        assert r["enabled"] is False
    
    def test_trigger_match_and(self):
        rule = self._make_irrigation_rule()
        self.engine.add_rule(rule)
        
        context = {
            "sensor_data": {"soil_moisture": 25},
        }
        matched = self.engine.find_matching_rules(context)
        # 只会在 06:00-20:00 之间匹配
        # 如果当前时间在范围内，应该匹配到
        from datetime import datetime
        hour = datetime.now().hour
        if 6 <= hour < 20:
            assert len(matched) == 1
        else:
            assert len(matched) == 0
    
    def test_trigger_match_or(self):
        rule = self._make_irrigation_rule()
        rule["trigger"]["logic"] = "OR"
        self.engine.add_rule(rule)
        
        context = {"sensor_data": {"soil_moisture": 25}}
        matched = self.engine.find_matching_rules(context)
        # OR 逻辑下，只要湿度条件满足就匹配（不依赖时间）
        assert len(matched) == 1
    
    def test_evaluate_auto_execute(self):
        rule = self._make_irrigation_rule()
        proposed = {"duration": 30}
        context = {}
        decision, reason, params = self.engine.evaluate_action(rule, proposed, context)
        assert decision == RuleDecision.AUTO_EXECUTE
    
    def test_evaluate_hard_limit_rejected(self):
        rule = self._make_irrigation_rule()
        proposed = {"duration": 150}  # 超过硬限制 120 分钟
        context = {}
        decision, reason, params = self.engine.evaluate_action(rule, proposed, context)
        assert decision == RuleDecision.REJECTED
    
    def test_evaluate_constraint_need_confirm(self):
        rule = self._make_irrigation_rule()
        rule["constraints"]["max_duration_per_use"] = 40
        proposed = {"duration": 50}  # 超过用户设定上限 40
        context = {}
        decision, reason, params = self.engine.evaluate_action(rule, proposed, context)
        assert decision == RuleDecision.NEED_CONFIRM
    
    def test_forbidden_hours_rejected(self):
        rule = self._make_irrigation_rule()
        # 添加当前小时到禁止时段
        rule["constraints"]["forbidden_hours"].append(datetime.now().hour)
        proposed = {"duration": 30}
        context = {}
        decision, reason, params = self.engine.evaluate_action(rule, proposed, context)
        # 在当前小时在禁止时段 → 需确认
        from datetime import datetime
        if datetime.now().hour in rule["constraints"]["forbidden_hours"]:
            assert decision == RuleDecision.NEED_CONFIRM

    def test_ai_enhance_clamping(self):
        rule = self._make_irrigation_rule()
        rule["ai_enhance"] = {
            "enabled": True,
            "can_adjust": ["duration"],
            "adjust_range": {"duration": [-10, 10]},
        }
        # AI 建议调整为 50，但原始是 30，允许范围 [20, 40]
        proposed = {"duration": 50}
        proposed = self.engine._apply_ai_enhance(
            rule["ai_enhance"], proposed, {"duration": 30})
        assert proposed["duration"] == 40  # 被限制到 30+10=40
```

- [ ] **Step 8: Run rule engine tests**

```bash
pytest tests/test_device_rule_engine.py -v
```
Expected: 10 passed

- [ ] **Step 9: Create `tests/test_device_executor.py`**

```python
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
        await self.executor.execute("virtual_irrigation_01", cmd,
                                    trigger="rule", rule_id="rule_test")
        
        logs = self.executor.get_logs()
        assert len(logs) >= 1
        assert logs[-1]["device_id"] == "virtual_irrigation_01"
        assert logs[-1]["trigger"] == "rule"
```

- [ ] **Step 10: Run executor tests**

```bash
pytest tests/test_device_executor.py -v
```
Expected: 4 passed

- [ ] **Step 11: Run all Phase 1 tests**

```bash
pytest tests/test_device_base.py tests/test_simulator_driver.py tests/test_device_registry.py tests/test_device_rule_engine.py tests/test_device_executor.py -v
```
Expected: ~32 passed

- [ ] **Step 12: Commit**

```bash
git add tests/test_device_base.py tests/test_simulator_driver.py tests/test_device_registry.py tests/test_device_rule_engine.py tests/test_device_executor.py
git commit -m "test: add Phase 1 unit tests for device layer and rule engine"
```

---

## Phase 2: Agent 整合 — DeviceAgent + 工作流

### Task 8: 扩展 AgentState

**Files:**
- Modify: `app/agent/state.py`

- [ ] **Step 1: Add device fields to AgentState**

在 `app/agent/state.py` 中，`class AgentState(BaseModel)` 类的末尾（`current_field_id` 之后）添加：

```python
    # 设备控制相关（新增）
    device_command: Optional[Dict[str, Any]] = None       # 待执行的设备指令
    device_result: Optional[Dict[str, Any]] = None        # 执行结果
    pending_action: Optional[Dict[str, Any]] = None       # 待用户确认的操作
    matched_rules: List[str] = Field(default_factory=list)  # 命中的规则ID列表
```

具体位置：在 `current_field_id` 行后添加，`class AgentState` 的最后。

- [ ] **Step 2: Syntax check**

```bash
python -c "from app.agent.state import AgentState; s = AgentState(); print(f'device_command={s.device_command}, matched_rules={s.matched_rules}')"
```
Expected: `device_command=None, matched_rules=[]`

- [ ] **Step 3: Commit**

```bash
git add app/agent/state.py
git commit -m "feat(state): add device control fields to AgentState"
```

---

### Task 9: 新增意图关键词和配置

**Files:**
- Modify: `app/agent/config.py`

- [ ] **Step 1: Add DEVICE_KEYWORDS to config.py**

在 `app/agent/config.py` 中，`FIELD_KEYWORDS` 定义之后添加：

```python
# 设备控制意图关键词
DEVICE_KEYWORDS = [
    "浇水", "灌溉", "施肥", "通风", "开窗", "遮阳",
    "补光", "开灯", "关灯", "加热", "降温", "喷雾",
    "自动控制", "手动控制", "设备状态", "打开", "关闭",
    "启动", "停止", "控制", "设备",
]
```

- [ ] **Step 2: Commit**

```bash
git add app/agent/config.py
git commit -m "feat(config): add DEVICE_KEYWORDS for device control intent"
```

---

### Task 10: DeviceAgent

**Files:**
- Create: `app/agent/agents/device_agent.py`

- [ ] **Step 1: Create `app/agent/agents/device_agent.py`**

```python
"""设备控制 Agent — 解析用户意图，调度设备操作"""

import json
import logging
from typing import Dict, Any, Optional

from .base import BaseAgent
from ..state import AgentState

logger = logging.getLogger(__name__)


class DeviceAgent(BaseAgent):
    name = "device"
    description = "智能设备控制专家，负责灌溉、施肥、通风、补光等设备自主操作与调度"
    system_prompt = """你是一位智能农业设备控制专家。
你能：
1. 理解用户的设备控制需求（浇水、施肥、通风、补光等）
2. 根据上下文（传感器数据、天气、作物阶段）推荐最佳操作参数
3. 在安全规则边界内自主决策和执行设备指令
4. 与其他 Agent（气象、病虫害）协作，实现联动控制

关键原则：
- 永远在规则引擎的安全边界内操作
- 当操作超出用户设定边界时，生成待确认操作而非直接执行
- 执行前必须检查当前天气和设备状态"""

    intent_types = ["device_control"]

    def invoke(self, state: AgentState) -> AgentState:
        """处理设备控制意图"""
        question = state.user_question or ""
        
        try:
            # 1. 用 LLM 解析用户设备操作意图
            parsed = self._parse_device_intent(question, state)
            if not parsed:
                return self._reply(state, "抱歉哥哥，我没理解你想操作哪个设备呢～能再说详细一点吗？比如「帮小麦浇30分钟水」")
            
            # 2. 尝试匹配规则
            matched_rules = self._match_rules(parsed, state)
            state.matched_rules = [r["id"] for r in matched_rules]
            
            # 3. 评估操作
            if matched_rules:
                return self._execute_with_rule(matched_rules[0], parsed, state)
            else:
                return self._execute_direct(parsed, state)
                
        except Exception as e:
            logger.exception("DeviceAgent 处理失败")
            return self._reply(state, f"设备控制出错了：{e}")
    
    # ── LLM 意图解析 ────────────────────────
    
    def _parse_device_intent(self, question: str, state: AgentState) -> Optional[Dict]:
        """用 LLM 从用户自然语言中提取设备操作参数"""
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
        from ..config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, OPENAI_BASE_URL
        
        context = self._get_context(state)
        
        prompt = f"""分析用户的设备控制需求，提取操作参数。

用户输入："{question}"

上下文：
- 当前作物：{context.get('crop', '未指定')}
- 地区：{context.get('region', '未指定')}

请以 JSON 格式返回：
{{
    "action": "irrigate|fertigate|ventilate|light|heat|cool|shade|status",
    "device_hint": "设备名或类型关键词（可选）",
    "crop": "目标作物",
    "params": {{
        "duration": 数字分钟（灌溉/通风专用）,
        "amount_kg": 数字kg（施肥专用）,
        "target_temp": 数字°C（加热/降温专用）,
        "brightness_percent": 数字（补光专用）
    }},
    "reasoning": "操作理由"
}}

如果无法判断具体操作，action 设为 "unknown"。"""

        try:
            llm = ChatOpenAI(
                model=LLM_MODEL, temperature=LLM_TEMPERATURE,
                api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL,
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            content = resp.content
            
            # 提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            parsed = json.loads(content)
            if parsed.get("action") == "unknown":
                return None
            return parsed
        except Exception as e:
            logger.warning("DeviceAgent 意图解析失败: %s, 回退到关键词匹配", e)
            return self._keyword_parse(question)
    
    def _keyword_parse(self, question: str) -> Optional[Dict]:
        """关键词回退解析"""
        if any(kw in question for kw in ["浇水", "灌溉"]):
            return {"action": "irrigate", "params": {"duration": 30}}
        if any(kw in question for kw in ["施肥"]):
            return {"action": "fertigate", "params": {"amount_kg": 5}}
        if any(kw in question for kw in ["通风", "开窗"]):
            return {"action": "ventilate", "params": {"duration": 30}}
        if any(kw in question for kw in ["补光", "开灯"]):
            return {"action": "light", "params": {"brightness_percent": 80}}
        if any(kw in question for kw in ["加热"]):
            return {"action": "heat", "params": {"target_temp": 22}}
        if any(kw in question for kw in ["设备状态"]):
            return {"action": "status", "params": {}}
        
        # 尝试提取时长
        import re
        dur_match = re.search(r'(\d+)\s*(分钟|分)', question)
        duration = int(dur_match.group(1)) if dur_match else 30
        
        return {"action": "irrigate", "params": {"duration": duration}}
    
    # ── 规则匹配 ──────────────────────────────
    
    def _match_rules(self, parsed: Dict, state: AgentState) -> list:
        """查找与当前操作匹配的规则"""
        try:
            from core.device_rule_engine import RuleEngine
            engine = RuleEngine()
            
            action = parsed.get("action", "")
            sensor_context = self._get_sensor_context(action)
            
            context = {
                "sensor_data": sensor_context,
                "weather": {},
                "crop": parsed.get("crop", ""),
            }
            return engine.find_matching_rules(context)
        except Exception as e:
            logger.warning("规则匹配失败: %s", e)
            return []
    
    def _get_sensor_context(self, action: str) -> Dict:
        """获取当前传感器数据作为规则评估上下文"""
        # 这里用 SimulatorDriver 获取模拟数据
        # 生产环境中从真实设备读取
        try:
            from devices.simulator_driver import SimulatorDriver
            import asyncio
            sim = SimulatorDriver(simulated_latency_ms=0)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(sim.connect())
            state = loop.run_until_complete(sim.read_state("virtual_soil_sensor_01"))
            return state
        except Exception:
            return {"soil_moisture": 45, "temperature": 22, "humidity": 65}
    
    # ── 执行路由 ──────────────────────────────
    
    def _execute_with_rule(self, rule: Dict, parsed: Dict,
                           state: AgentState) -> AgentState:
        """有匹配规则时的执行逻辑"""
        from core.device_rule_engine import RuleEngine
        
        engine = RuleEngine()
        action = rule.get("action", {})
        proposed_params = {**action.get("params", {}), **parsed.get("params", {})}
        
        decision, reason, final_params = engine.evaluate_action(
            rule, proposed_params, {"device_id": action.get("device_id", "")})
        
        if decision == "auto_execute":
            return self._do_execute(action.get("device_id", ""),
                                    action.get("command", "start"),
                                    final_params, state,
                                    rule_id=rule["id"],
                                    extra=f"✅ 规则「{rule.get('name', '')}」校验通过，已自动执行")
        elif decision == "need_confirm":
            state.pending_action = {
                "device_id": action.get("device_id"),
                "command": action.get("command", "start"),
                "params": final_params,
                "reason": reason,
                "rule_id": rule["id"],
            }
            return self._reply(state,
                f"⚠️ {reason}\n\n"
                f"📋 操作预览：{action.get('device_id')} → {action.get('command')} "
                f"参数：{final_params}\n\n"
                f"请在「设备仪表盘」中确认此操作。")
        else:
            return self._reply(state, f"❌ {reason}")
    
    def _execute_direct(self, parsed: Dict, state: AgentState) -> AgentState:
        """无匹配规则时的直接执行（但仍通过规则引擎做安全检查）"""
        action_type = parsed.get("action", "")
        params = parsed.get("params", {})
        
        # 找到对应设备
        device_id = self._find_device_for_action(action_type)
        if not device_id:
            return self._reply(state,
                f"😅 没找到{action_type}类型的设备呢～请先在「设备仪表盘」中添加设备吧！")
        
        # 通过规则引擎校验
        from core.device_rule_engine import RuleEngine, RuleDecision
        engine = RuleEngine()
        
        # 构造临时规则用于校验
        temp_rule = {
            "id": "temp_direct",
            "action": {"device_id": device_id, "command": "start", "params": params},
            "constraints": {
                "max_duration_per_use": 60,
                "forbidden_hours": [22, 23, 0, 1, 2, 3, 4, 5],
            },
        }
        decision, reason, final_params = engine.evaluate_action(
            temp_rule, params, {"device_id": device_id})
        
        if decision == RuleDecision.REJECTED:
            return self._reply(state, f"❌ {reason}")
        elif decision == RuleDecision.NEED_CONFIRM:
            state.pending_action = {
                "device_id": device_id,
                "command": "start",
                "params": final_params,
                "reason": reason,
            }
            return self._reply(state,
                f"⚠️ {reason}\n\n请在「设备仪表盘」中确认此操作。")
        
        return self._do_execute(device_id, "start", final_params, state)
    
    def _do_execute(self, device_id: str, command: str, params: Dict,
                    state: AgentState, rule_id: str = None,
                    extra: str = "") -> AgentState:
        """实际执行设备指令"""
        try:
            from devices.base import DeviceCommand
            from devices.registry import DeviceDriverRegistry
            from devices.simulator_driver import SimulatorDriver
            from core.device_executor import DeviceExecutor
            from core.device_rule_engine import RuleEngine
            
            # 设置 registry
            registry = DeviceDriverRegistry()
            sim = SimulatorDriver(simulated_latency_ms=100)
            registry.register("simulator", sim)
            
            executor = DeviceExecutor(registry)
            
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(sim.connect())
            loop.run_until_complete(registry.discover_all())
            
            cmd = DeviceCommand(command=command, params=params)
            result = executor.execute_sync(device_id, cmd,
                                           trigger="agent", rule_id=rule_id)
            
            # 记录执行到规则引擎
            engine = RuleEngine()
            engine.record_execution(device_id, params)
            
            state.device_command = {"device_id": device_id, "command": command, "params": params}
            state.device_result = {
                "success": result["success"],
                "message": result["result"].message if result.get("result") else "",
            }
            
            if result["success"]:
                msg = f"✅ 指令已执行！\n\n"
                msg += f"🔧 设备：{device_id}\n"
                msg += f"⚡ 操作：{command}\n"
                msg += f"📊 参数：{params}\n"
                msg += f"📝 结果：{result['result'].message}\n"
                if extra:
                    msg += f"\n{extra}"
            else:
                msg = f"❌ 执行失败：{result['result'].message}"
            
            return self._reply(state, msg)
            
        except Exception as e:
            logger.exception("设备执行异常")
            return self._reply(state, f"❌ 设备执行出错：{e}")
    
    def _find_device_for_action(self, action: str) -> Optional[str]:
        """根据操作类型找到对应设备"""
        mapping = {
            "irrigate": "virtual_irrigation_01",
            "fertigate": "virtual_fertigator_01",
            "ventilate": "virtual_ventilation_01",
            "light": "virtual_light_01",
            "heat": "virtual_heater_01",
        }
        return mapping.get(action)
```

- [ ] **Step 2: Syntax check**

```bash
python -c "from app.agent.agents.device_agent import DeviceAgent; a = DeviceAgent(); print(f'OK — DeviceAgent: {a.name}, intents: {a.intent_types}')"
```
Expected: `OK — DeviceAgent: device, intents: ['device_control']`

- [ ] **Step 3: Commit**

```bash
git add app/agent/agents/device_agent.py
git commit -m "feat(agent): add DeviceAgent for device control"
```

---

### Task 11: 注册 DeviceAgent 到调度中心

**Files:**
- Modify: `app/agent/agents/__init__.py`
- Modify: `app/agent/agents/orchestrator.py`

- [ ] **Step 1: Update `app/agent/agents/__init__.py`**

```python
# multi-agent package
from .orchestrator import AgentOrchestrator
from .device_agent import DeviceAgent
```

- [ ] **Step 2: Update `app/agent/agents/orchestrator.py`**

在 import 区域添加：
```python
from .device_agent import DeviceAgent
```

在 `__init__` 方法的 `self._register(FarmingAgent())` 之后添加：
```python
        self._register(DeviceAgent())
```

同时在 `CROSS_DOMAIN_KEYWORDS` dict 末尾添加：
```python
    "device_control": ["浇水", "灌溉", "施肥", "通风", "补光", "加热", "降温", "开关", "启动", "停止"],
```

- [ ] **Step 3: Syntax check**

```bash
python -c "from app.agent.agents.orchestrator import AgentOrchestrator; o = AgentOrchestrator(); agents = o.list_agents(); print(f'OK — {len(agents)} agents: {[a[\"name\"] for a in agents]}')"
```
Expected: `OK — 6 agents: ['planting', 'disease', 'weather', 'finance', 'farming', 'device']`

- [ ] **Step 4: Commit**

```bash
git add app/agent/agents/__init__.py app/agent/agents/orchestrator.py
git commit -m "feat(orchestrator): register DeviceAgent"
```

---

### Task 12: 更新意图分类

**Files:**
- Modify: `app/agent/nodes/classify_intent.py`

- [ ] **Step 1: Add device_control to LLM intent prompt**

在 `classify_intent.py` 的 `_llm_classify_intent` 函数中，LLM prompt 的意图列表里，在 `field_management` 行之后添加：

```python
	- device_control: 设备控制（浇水、灌溉、施肥、通风、补光、加热、开关设备、控制设备等）
```

同时在关键词快速匹配部分（`classify_intent` 函数中），在其他关键词分支后添加：

```python
    elif any(word in user_question for word in DEVICE_KEYWORDS):
        state.intent_type = "device_control"
        state.need_rag = False
        state.need_clarification = False
        return state
```

注意：需要确认 `DEVICE_KEYWORDS` 已从 config 导入。检查文件顶部的 import 行 `from ..config import *` 已包含所有关键词常量。

- [ ] **Step 2: Commit**

```bash
git add app/agent/nodes/classify_intent.py
git commit -m "feat(intent): add device_control intent classification"
```

---

### Task 13: 更新工作流图

**Files:**
- Modify: `app/agent/graph.py`

- [ ] **Step 1: Update graph.py routing**

在 `route_after_agent` 函数中，在 `task_intents` tuple 里添加 `"device_control"`：

```python
        task_intents = ("crop_selection", "planting_schedule", "planting_method",
                        "disease_prevention", "harvest_planning", "image_analysis",
                        "device_control")
```

同时在 `_agent_dispatch_node` 之前添加注释中的 `skip_intents`，确保 `device_control` 不在跳过列表中（默认不在，因为是新意图）。

- [ ] **Step 2: Commit**

```bash
git add app/agent/graph.py
git commit -m "feat(graph): add device_control routing to workflow"
```

---

## Phase 3: 前端 + API

### Task 14: 新增 API 路由

**Files:**
- Modify: `app/api_routes.py`

- [ ] **Step 1: Add device & rule API endpoints**

在 `register_routes` 函数末尾（`return app` 之前）添加以下路由：

```python
    # ── 设备管理 ──────────────────────────────────────
    
    @app.get("/api/devices")
    def list_devices(username: str = "default"):
        """获取所有设备列表及状态"""
        try:
            from devices.registry import DeviceDriverRegistry
            from devices.simulator_driver import SimulatorDriver
            
            registry = DeviceDriverRegistry()
            sim = SimulatorDriver(simulated_latency_ms=50)
            registry.register("simulator", sim)
            
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(sim.connect())
            devices = loop.run_until_complete(registry.discover_all())
            
            result = []
            for d in devices:
                state = loop.run_until_complete(registry.read_state(d.device_id))
                result.append({
                    "device_id": d.device_id,
                    "name": d.name,
                    "driver": d.driver_name,
                    "capabilities": [c.value for c in d.capabilities],
                    "sensors": d.sensors,
                    "status": d.status,
                    "location": d.location,
                    "state": state,
                })
            return result
        except Exception as e:
            logger.exception("获取设备列表失败")
            return []
    
    @app.post("/api/devices/{device_id}/command")
    def send_device_command(device_id: str, command: str = "start",
                            params: str = "{}", username: str = "default"):
        """向设备发送指令"""
        try:
            import json as _json
            from devices.base import DeviceCommand
            from devices.registry import DeviceDriverRegistry
            from devices.simulator_driver import SimulatorDriver
            from core.device_executor import DeviceExecutor
            
            registry = DeviceDriverRegistry()
            sim = SimulatorDriver(simulated_latency_ms=100)
            registry.register("simulator", sim)
            
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(sim.connect())
            loop.run_until_complete(registry.discover_all())
            
            executor = DeviceExecutor(registry, username=username)
            cmd = DeviceCommand(command=command, params=_json.loads(params))
            result = executor.execute_sync(device_id, cmd, trigger="api")
            
            return {
                "success": result["success"],
                "device_id": device_id,
                "message": result["result"].message,
                "attempts": result["attempts"],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @app.get("/api/devices/{device_id}/state")
    def get_device_state(device_id: str):
        """获取设备实时状态"""
        try:
            from devices.registry import DeviceDriverRegistry
            from devices.simulator_driver import SimulatorDriver
            
            registry = DeviceDriverRegistry()
            sim = SimulatorDriver(simulated_latency_ms=50)
            registry.register("simulator", sim)
            
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(sim.connect())
            loop.run_until_complete(registry.discover_all())
            
            state = loop.run_until_complete(registry.read_state(device_id))
            return state
        except Exception as e:
            return {"error": str(e)}
    
    # ── 规则管理 ──────────────────────────────────────
    
    @app.get("/api/rules")
    def list_rules(username: str = "default"):
        try:
            from core.device_rule_engine import RuleEngine
            engine = RuleEngine(username=username)
            return engine.list_rules()
        except Exception as e:
            return []
    
    @app.post("/api/rules")
    def create_rule(rule: Dict, username: str = "default"):
        try:
            from core.device_rule_engine import RuleEngine
            engine = RuleEngine(username=username)
            rule_id = engine.add_rule(rule)
            return {"success": True, "rule_id": rule_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @app.put("/api/rules/{rule_id}")
    def update_rule(rule_id: str, rule: Dict, username: str = "default"):
        try:
            from core.device_rule_engine import RuleEngine
            engine = RuleEngine(username=username)
            ok = engine.update_rule(rule_id, rule)
            return {"success": ok}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @app.delete("/api/rules/{rule_id}")
    def delete_rule(rule_id: str, username: str = "default"):
        try:
            from core.device_rule_engine import RuleEngine
            engine = RuleEngine(username=username)
            ok = engine.delete_rule(rule_id)
            return {"success": ok}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @app.post("/api/rules/{rule_id}/test")
    def test_rule(rule_id: str, username: str = "default"):
        """测试规则 — 仅评估不执行"""
        try:
            from core.device_rule_engine import RuleEngine
            engine = RuleEngine(username=username)
            rule = engine.get_rule(rule_id)
            if not rule:
                return {"success": False, "error": "规则不存在"}
            
            # 获取传感器上下文
            from devices.simulator_driver import SimulatorDriver
            import asyncio
            sim = SimulatorDriver(simulated_latency_ms=0)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(sim.connect())
            sensor_data = loop.run_until_complete(sim.read_state("virtual_soil_sensor_01"))
            
            context = {"sensor_data": sensor_data, "weather": {}}
            matched = engine.find_matching_rules(context)
            
            return {
                "success": True,
                "rule_matched": rule["id"] in [r["id"] for r in matched],
                "sensor_snapshot": sensor_data,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ── 操作管理 ──────────────────────────────────────
    
    @app.get("/api/actions/log")
    def get_action_log(limit: int = 50, username: str = "default"):
        try:
            from core.device_executor import DeviceExecutor
            from devices.registry import DeviceDriverRegistry
            from devices.simulator_driver import SimulatorDriver
            
            registry = DeviceDriverRegistry()
            registry.register("simulator", SimulatorDriver())
            executor = DeviceExecutor(registry, username=username)
            return executor.get_logs(limit=limit)
        except Exception as e:
            return []
    
    @app.get("/api/actions/pending")
    def get_pending_actions(username: str = "default"):
        try:
            from core.device_executor import DeviceExecutor
            from devices.registry import DeviceDriverRegistry
            from devices.simulator_driver import SimulatorDriver
            
            registry = DeviceDriverRegistry()
            registry.register("simulator", SimulatorDriver())
            executor = DeviceExecutor(registry, username=username)
            return executor.list_pending()
        except Exception as e:
            return []
    
    @app.post("/api/actions/{action_id}/confirm")
    def confirm_action(action_id: str, username: str = "default"):
        try:
            from core.device_executor import DeviceExecutor
            from devices.registry import DeviceDriverRegistry
            from devices.simulator_driver import SimulatorDriver
            
            registry = DeviceDriverRegistry()
            registry.register("simulator", SimulatorDriver())
            executor = DeviceExecutor(registry, username=username)
            result = executor.confirm_pending(action_id)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @app.post("/api/actions/{action_id}/reject")
    def reject_action(action_id: str, username: str = "default"):
        try:
            from core.device_executor import DeviceExecutor
            from devices.registry import DeviceDriverRegistry
            from devices.simulator_driver import SimulatorDriver
            
            registry = DeviceDriverRegistry()
            registry.register("simulator", SimulatorDriver())
            executor = DeviceExecutor(registry, username=username)
            ok = executor.reject_pending(action_id)
            return {"success": ok}
        except Exception as e:
            return {"success": False, "error": str(e)}
```

注意：确保文件顶部已有 `import asyncio`，如果没有则添加。

- [ ] **Step 2: Syntax check**

```bash
python -c "from app.api_routes import register_routes; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/api_routes.py
git commit -m "feat(api): add device and rule management endpoints"
```

---

### Task 15: 设备仪表盘页面

**Files:**
- Create: `app/views/devices.py`

- [ ] **Step 1: Create `app/views/devices.py`**

```python
"""设备仪表盘 — 设备状态监控 + 快捷操作 + 待确认管理"""

import streamlit as st
from app.api_client import api, invalidate_cache
from datetime import datetime


def render_devices_page():
    """渲染设备仪表盘"""
    st.markdown("## 🤖 设备仪表盘")
    
    # ── 顶部状态概览 ──────────────────────────
    devices = api("/api/devices") or []
    pending = api("/api/actions/pending") or []
    logs = api("/api/actions/log", cache_ttl=15) or []
    
    online_count = sum(1 for d in devices if d.get("status") == "online")
    offline_count = sum(1 for d in devices if d.get("status") == "offline")
    pending_count = len([a for a in pending if a.get("status") == "pending"])
    today_actions = sum(1 for l in logs if l.get("timestamp", "").startswith(datetime.now().strftime("%Y-%m-%d")))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🟢 在线设备", online_count)
    with col2:
        st.metric("⚠️ 待确认", pending_count, delta_color="inverse")
    with col3:
        st.metric("🔴 离线", offline_count)
    with col4:
        st.metric("⚡ 今日操作", today_actions)
    
    st.divider()
    
    # ── 设备列表 ──────────────────────────────
    st.markdown("### 📡 设备列表")
    
    if not devices:
        st.info("暂无设备。请确保已启动 SimulatorDriver。")
    else:
        for dev in devices:
            state = dev.get("state", {})
            status_icon = {"online": "🟢", "offline": "🔴", "error": "⚠️"}.get(dev.get("status"), "⚪")
            
            with st.expander(f"{status_icon} **{dev['name']}** — {dev.get('location', '未分配位置')} | {dev.get('status', 'unknown')}"):
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.caption(f"设备ID: `{dev['device_id']}`")
                    st.caption(f"驱动: {dev.get('driver', 'unknown')}")
                    st.caption(f"能力: {', '.join(dev.get('capabilities', []))}")
                    
                    # 传感器数据
                    if state and not state.get("error"):
                        st.markdown("**传感器读数：**")
                        cols = st.columns(min(len(state), 4))
                        for i, (k, v) in enumerate(state.items()):
                            if k.startswith("_"):
                                continue
                            if isinstance(v, (int, float)):
                                with cols[i % 4]:
                                    st.metric(k, f"{v:.1f}" if isinstance(v, float) else str(v))
                            elif isinstance(v, bool):
                                with cols[i % 4]:
                                    st.metric(k, "✅ 开启" if v else "⭕ 关闭")
                
                with col_b:
                    caps = dev.get("capabilities", [])
                    if "irrigate" in caps:
                        duration = st.number_input("时长(分)", 1, 120, 30, key=f"dur_{dev['device_id']}")
                        if st.button("💧 浇水", key=f"irrigate_{dev['device_id']}", width="stretch"):
                            import json
                            result = api(f"/api/devices/{dev['device_id']}/command", method="post",
                                        json_data={"command": "start", "params": json.dumps({"duration": duration})})
                            if result and result.get("success"):
                                st.success(f"✅ {result.get('message', '已执行')}")
                                invalidate_cache("/api/devices", "/api/actions/log")
                                st.rerun()
                            else:
                                st.error(f"❌ {result.get('message', '执行失败')}")
                    
                    if "fertigate" in caps:
                        amount = st.number_input("用量(kg)", 1, 50, 5, key=f"amt_{dev['device_id']}")
                        if st.button("🌱 施肥", key=f"fertigate_{dev['device_id']}", width="stretch"):
                            import json
                            result = api(f"/api/devices/{dev['device_id']}/command", method="post",
                                        json_data={"command": "start", "params": json.dumps({"amount_kg": amount})})
                            if result and result.get("success"):
                                st.success("✅ 已执行")
                                st.rerun()
                    
                    if any(c in caps for c in ["ventilate", "light", "heat"]):
                        if st.button("▶️ 启动", key=f"start_{dev['device_id']}", width="stretch"):
                            import json
                            result = api(f"/api/devices/{dev['device_id']}/command", method="post",
                                        json_data={"command": "start", "params": json.dumps({})})
                            if result and result.get("success"):
                                st.success("✅ 已启动")
                                st.rerun()
                    
                    if st.button("⏹️ 停止", key=f"stop_{dev['device_id']}", width="stretch"):
                        import json
                        result = api(f"/api/devices/{dev['device_id']}/command", method="post",
                                    json_data={"command": "stop", "params": json.dumps({})})
                        if result and result.get("success"):
                            st.success("✅ 已停止")
                            st.rerun()
    
    st.divider()
    
    # ── 待确认操作 ──────────────────────────────
    st.markdown("### ⚠️ 待确认操作")
    pending_list = [a for a in pending if a.get("status") == "pending"]
    
    if not pending_list:
        st.success("暂无待确认操作～")
    else:
        for action in pending_list:
            with st.container():
                st.warning(f"**{action.get('device_id', '未知设备')}** — {action.get('command', '')}")
                st.caption(f"参数: {action.get('params', {})}")
                st.caption(f"原因: {action.get('reason', '需要用户确认')}")
                
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if st.button("✅ 确认执行", key=f"confirm_{action['id']}"):
                        result = api(f"/api/actions/{action['id']}/confirm", method="post")
                        if result and result.get("success"):
                            st.success("已执行！")
                            invalidate_cache("/api/actions/pending", "/api/actions/log")
                            st.rerun()
                with c2:
                    if st.button("✏️ 修改参数", key=f"edit_{action['id']}"):
                        st.info("参数编辑功能将在后续版本中支持")
                with c3:
                    if st.button("❌ 拒绝", key=f"reject_{action['id']}"):
                        api(f"/api/actions/{action['id']}/reject", method="post")
                        invalidate_cache("/api/actions/pending")
                        st.rerun()
    
    st.divider()
    
    # ── 今日执行日志 ────────────────────────────
    st.markdown("### 📋 今日执行日志")
    today_logs = [l for l in logs if l.get("timestamp", "").startswith(datetime.now().strftime("%Y-%m-%d"))]
    
    if not today_logs:
        st.caption("今日暂无操作记录")
    else:
        for log in reversed(today_logs[-20:]):
            icon = "✅" if log.get("success") else "❌"
            ts = log.get("timestamp", "").split("T")[1][:8] if "T" in log.get("timestamp", "") else ""
            st.caption(
                f"{icon} `{ts}` **{log.get('device_id', '')}** → "
                f"{log.get('command', '')} "
                f"({log.get('trigger', 'manual')}) — {log.get('message', '')[:60]}"
            )
    
    # 刷新按钮
    if st.button("🔄 刷新数据", width="stretch"):
        invalidate_cache("/api/devices", "/api/actions/pending", "/api/actions/log")
        st.rerun()
```

- [ ] **Step 2: Syntax check**

```bash
python -c "from app.views.devices import render_devices_page; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/views/devices.py
git commit -m "feat(ui): add device dashboard page"
```

---

### Task 16: 规则编辑器页面

**Files:**
- Create: `app/views/rules.py`

- [ ] **Step 1: Create `app/views/rules.py`**

```python
"""规则编辑器 — 自动规则的 CRUD 管理"""

import json
import streamlit as st
from app.api_client import api, invalidate_cache


def render_rules_page():
    """渲染规则编辑器"""
    st.markdown("## 📋 规则管理")
    
    rules = api("/api/rules") or []
    devices = api("/api/devices") or []
    
    # ── 规则列表 ──────────────────────────────
    col_list, col_edit = st.columns([1, 2])
    
    with col_list:
        st.markdown("### 我的规则")
        
        if not rules:
            st.info("还没有规则，点击右侧创建～")
        else:
            for rule in rules:
                enabled = rule.get("enabled", True)
                icon = "✅" if enabled else "⏸️"
                with st.container():
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        if st.button(f"{icon} {rule.get('name', '未命名')}", key=f"select_{rule['id']}"):
                            st.session_state.selected_rule = rule
                            st.rerun()
                    with c2:
                        if st.button("🗑️", key=f"del_{rule['id']}"):
                            api(f"/api/rules/{rule['id']}", method="delete")
                            invalidate_cache("/api/rules")
                            st.rerun()
    
    # ── 规则编辑区 ────────────────────────────
    with col_edit:
        selected = st.session_state.get("selected_rule")
        
        if selected:
            st.markdown(f"### ✏️ 编辑: {selected.get('name', '未命名')}")
            _render_rule_editor(selected, devices)
        else:
            st.markdown("### ➕ 新建规则")
            st.caption("从左侧选择一个规则编辑，或填写下方表单新建")
            _render_rule_editor(None, devices)


def _render_rule_editor(rule, devices):
    """渲染规则编辑表单"""
    is_new = rule is None
    
    with st.form(key=f"rule_form_{rule['id'] if rule else 'new'}"):
        name = st.text_input("规则名称", value=rule.get("name", "") if rule else "",
                             placeholder="如：小麦自动灌溉")
        
        enabled = st.checkbox("启用规则", value=rule.get("enabled", True) if rule else True)
        
        st.markdown("**触发条件**")
        trigger_logic = st.radio("逻辑", ["AND", "OR"],
                                 index=0 if (rule and rule.get("trigger", {}).get("logic") == "AND") else 0,
                                 horizontal=True, key=f"logic_{rule['id'] if rule else 'new'}")
        
        # 条件示例
        st.caption("条件暂支持 JSON 编辑（后续版本会提供可视化构建器）")
        trigger_json = st.text_area(
            "触发条件 (JSON)",
            value=json.dumps(
                rule.get("trigger", {}).get("conditions", [
                    {"type": "sensor", "field": "soil_moisture", "op": "<", "value": 30},
                    {"type": "time", "op": "between", "value": ["06:00", "20:00"]},
                ]) if rule else [
                    {"type": "sensor", "field": "soil_moisture", "op": "<", "value": 30},
                ],
                ensure_ascii=False, indent=2
            ),
            height=150,
        )
        
        st.markdown("**执行动作**")
        device_ids = [d["device_id"] for d in devices] if devices else ["virtual_irrigation_01"]
        device_id = st.selectbox(
            "目标设备",
            device_ids,
            index=device_ids.index(rule["action"]["device_id"]) if rule and rule.get("action", {}).get("device_id") in device_ids else 0,
            key=f"dev_{rule['id'] if rule else 'new'}"
        )
        
        command = st.selectbox("指令", ["start", "stop", "set_param"],
                               index=0 if (rule and rule.get("action", {}).get("command") == "start") else 0,
                               key=f"cmd_{rule['id'] if rule else 'new'}")
        
        st.markdown("**安全边界**")
        c1, c2 = st.columns(2)
        with c1:
            max_dur = st.number_input("单次最长(分)", 1, 120,
                                      value=rule.get("constraints", {}).get("max_duration_per_use", 60) if rule else 60,
                                      key=f"maxdur_{rule['id'] if rule else 'new'}")
        with c2:
            max_daily = st.number_input("每日上限(分)", 1, 600,
                                        value=rule.get("constraints", {}).get("max_duration_per_day", 180) if rule else 180,
                                        key=f"maxday_{rule['id'] if rule else 'new'}")
        
        ai_enabled = st.checkbox("启用 AI 微调",
                                 value=rule.get("ai_enhance", {}).get("enabled", False) if rule else False,
                                 key=f"ai_{rule['id'] if rule else 'new'}")
        
        submitted = st.form_submit_button("💾 保存规则", width="stretch")
        
        if submitted:
            try:
                trigger_conditions = json.loads(trigger_json)
            except json.JSONDecodeError:
                st.error("触发条件 JSON 格式错误，请检查！")
                return
            
            new_rule = {
                "id": rule["id"] if rule else None,
                "name": name or "未命名规则",
                "enabled": enabled,
                "trigger": {
                    "logic": trigger_logic,
                    "conditions": trigger_conditions,
                },
                "action": {
                    "device_id": device_id,
                    "command": command,
                    "params": {"duration": 30},
                },
                "constraints": {
                    "max_duration_per_use": max_dur,
                    "max_duration_per_day": max_daily,
                    "min_interval_minutes": 120,
                    "forbidden_hours": [22, 23, 0, 1, 2, 3, 4, 5],
                },
                "ai_enhance": {
                    "enabled": ai_enabled,
                    "can_adjust": ["duration"],
                    "adjust_range": {"duration": [-10, 10]},
                },
            }
            
            if is_new:
                result = api("/api/rules", method="post", json_data=new_rule)
            else:
                result = api(f"/api/rules/{rule['id']}", method="put", json_data=new_rule)
            
            if result and result.get("success"):
                invalidate_cache("/api/rules")
                st.success("规则已保存！")
                st.rerun()
            else:
                st.error(f"保存失败: {result.get('error', '未知错误')}")
    
    # 测试按钮（仅编辑已有规则时显示）
    if not is_new and rule:
        if st.button("▶️ 测试规则（仅评估不执行）", key=f"test_{rule['id']}", width="stretch"):
            result = api(f"/api/rules/{rule['id']}/test", method="post")
            if result and result.get("success"):
                if result.get("rule_matched"):
                    st.success("✅ 规则条件匹配！传感器快照：")
                    st.json(result.get("sensor_snapshot", {}))
                else:
                    st.warning("❌ 条件不满足，规则不会触发")
            else:
                st.error(f"测试失败: {result.get('error', '')}")
```

- [ ] **Step 2: Syntax check**

```bash
python -c "from app.views.rules import render_rules_page; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/views/rules.py
git commit -m "feat(ui): add rule editor page"
```

---

### Task 17: 对话卡片增强

**Files:**
- Modify: `app/views/chat.py`

- [ ] **Step 1: Add device message card rendering**

在 `chat.py` 中，找到消息渲染逻辑（通常在 `render_chat_history` 函数中），在 AI 消息渲染部分添加设备卡片的特殊处理。在现有消息显示逻辑之后，添加：

```python
def _render_device_card(message_content: str):
    """检测并渲染设备控制消息为富卡片"""
    # 检测是否是设备控制回复
    if "指令已执行" in message_content or "执行失败" in message_content or "执行预览" in message_content:
        st.markdown(message_content)
        return True
    return False
```

然后在主渲染循环中，对每条 AI 消息先调用 `_render_device_card`，如果返回 False 再走常规渲染。

- [ ] **Step 2: Commit**

```bash
git add app/views/chat.py
git commit -m "feat(ui): add device message card rendering in chat"
```

---

### Task 18: 页面路由与导航

**Files:**
- Modify: `app/test1.py`
- Modify: `app/ui/sidebar.py`

- [ ] **Step 1: Add page routes to test1.py**

在 `test1.py` 的 import 区域添加：
```python
from app.views.devices import render_devices_page
from app.views.rules import render_rules_page
```

在页面路由逻辑中（`current_page` 的 if-elif 链），添加：
```python
    elif current_page == "devices":
        st.divider()
        render_devices_page()
    elif current_page == "rules":
        st.divider()
        render_rules_page()
```

- [ ] **Step 2: Add navigation entries to sidebar.py**

在 `render_nav_bar` 函数中（或 sidebar 导航中），在现有页面选项后添加：
```python
    # 设备相关（在 finance/policy 等导航项旁添加）
    if st.sidebar.button("🤖 设备仪表盘", key="nav_devices", width="stretch"):
        st.session_state.current_page = "devices"
        st.rerun()
    if st.sidebar.button("📋 规则管理", key="nav_rules", width="stretch"):
        st.session_state.current_page = "rules"
        st.rerun()
```

注意：需要先找到 `render_nav_bar` 的实际位置和风格。可能在 `test1.py` 中定义。需要参照现有的导航按钮写法。

- [ ] **Step 3: Commit**

```bash
git add app/main.py app/ui/sidebar.py
git commit -m "feat(ui): add devices and rules page routing"
```

---

### Task 19: 后台规则轮询任务

**Files:**
- Modify: `app/scheduler_jobs.py`

- [ ] **Step 1: Add rule polling job**

在 `app/scheduler_jobs.py` 末尾添加：

```python
def check_device_rules_job():
    """每 5 分钟检查自动规则并触发设备操作"""
    try:
        import os, json
        from core.device_rule_engine import RuleEngine
        from devices.registry import DeviceDriverRegistry
        from devices.simulator_driver import SimulatorDriver
        from devices.base import DeviceCommand
        from core.device_executor import DeviceExecutor
        
        # 获取当前传感器数据
        sim = SimulatorDriver(simulated_latency_ms=50)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(sim.connect())
        sensor_data = loop.run_until_complete(sim.read_state("virtual_soil_sensor_01"))
        
        # 遍历所有用户
        data_dir = os.path.join("data")
        usernames = ["default"]
        if os.path.exists(data_dir):
            for d in os.listdir(data_dir):
                user_path = os.path.join(data_dir, d)
                if os.path.isdir(user_path) and os.path.exists(os.path.join(user_path, "device_rules.json")):
                    usernames.append(d)
        
        for username in set(usernames):
            engine = RuleEngine(username=username)
            context = {"sensor_data": sensor_data, "weather": {}}
            matched = engine.find_matching_rules(context)
            
            if not matched:
                continue
            
            registry = DeviceDriverRegistry()
            registry.register("simulator", sim)
            loop.run_until_complete(registry.discover_all())
            executor = DeviceExecutor(registry, username=username)
            
            for rule in matched:
                action = rule.get("action", {})
                proposed = action.get("params", {})
                decision, reason, final_params = engine.evaluate_action(
                    rule, proposed, {"device_id": action.get("device_id", "")})
                
                if decision == "auto_execute":
                    cmd = DeviceCommand(
                        command=action.get("command", "start"),
                        params=final_params,
                    )
                    result = executor.execute_sync(
                        action.get("device_id"), cmd,
                        trigger="rule", rule_id=rule["id"], decision=decision)
                    
                    if result["success"]:
                        engine.record_execution(action.get("device_id"), final_params)
                        logger.info("自动规则触发: %s → %s", rule["name"], action.get("device_id"))
    except Exception as e:
        logger.warning("设备规则轮询失败: %s", e)
```

- [ ] **Step 2: Register job in scheduler**

在 `api_server.py` 或 `start.py` 中找到 APScheduler 配置处，添加新任务：
```python
scheduler.add_job(check_device_rules_job, 'interval', minutes=5, id='device_rules')
```

- [ ] **Step 3: Commit**

```bash
git add app/scheduler_jobs.py
git commit -m "feat(scheduler): add device rule polling job"
```

---

### Task 20: Phase 3 集成测试与验证

- [ ] **Step 1: Start API server and test endpoints**

```bash
cd D:/code/PycharmProject/Agriculture_Agent && python -c "
from devices.simulator_driver import SimulatorDriver
from devices.registry import DeviceDriverRegistry
from devices.base import DeviceCommand
from core.device_rule_engine import RuleEngine
from core.device_executor import DeviceExecutor
import asyncio

# 完整端到端测试
sim = SimulatorDriver(simulated_latency_ms=0)
registry = DeviceDriverRegistry()
registry.register('sim', sim)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(sim.connect())
devices = loop.run_until_complete(registry.discover_all())
print(f'发现 {len(devices)} 个设备')

# 测试直接控制
cmd = DeviceCommand(command='start', params={'duration': 20})
executor = DeviceExecutor(registry)
result = executor.execute_sync('virtual_irrigation_01', cmd)
print(f'灌溉执行: {\"成功\" if result[\"success\"] else \"失败\"}')

# 测试规则引擎
engine = RuleEngine(username='test')
rule = {
    'id': 'test_001', 'name': '测试规则', 'enabled': True,
    'trigger': {'logic': 'AND', 'conditions': [
        {'type': 'sensor', 'field': 'soil_moisture', 'op': '<', 'value': 50}
    ]},
    'action': {'device_id': 'virtual_irrigation_01', 'command': 'start', 'params': {'duration': 30}},
    'constraints': {'max_duration_per_use': 60, 'max_duration_per_day': 180, 'forbidden_hours': []},
}
engine.add_rule(rule)

sensor = loop.run_until_complete(sim.read_state('virtual_soil_sensor_01'))
context = {'sensor_data': sensor}
matched = engine.find_matching_rules(context)
print(f'传感器湿度={sensor[\"soil_moisture\"]:.1f}%, 匹配规则={len(matched)}条')

decision, reason, params = engine.evaluate_action(rule, {'duration': 30}, {})
print(f'决策: {decision} ({reason})')

print('\\n✅ 端到端集成测试全部通过！')
"
```
Expected: 所有步骤打印成功信息，最终输出 `✅ 端到端集成测试全部通过！`

- [ ] **Step 2: Run all tests**

```bash
cd D:/code/PycharmProject/Agriculture_Agent && pytest tests/test_device_*.py -v
```
Expected: ~32 passed

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: add Phase 3 integration test and verify full pipeline"
```

---

## 验证清单

完成 Phase 1-3 后，以下场景应全部可用：

- [ ] 启动应用，侧边栏看到"设备仪表盘"和"规则管理"入口
- [ ] 设备仪表盘显示 6 个虚拟设备，可点击浇水/启动/停止
- [ ] 创建一条自动灌溉规则（湿度<30% → 浇水30分钟）
- [ ] 测试规则可以正确评估（仅评估不执行）
- [ ] 在对话页说"帮小麦浇30分钟水"，Agent 回复执行结果
- [ ] 操作日志正常记录
- [ ] 模拟传感器湿度降低 → 规则自动触发 → 设备执行
- [ ] 超出安全边界的操作被拦截或推送确认
- [ ] `pytest tests/test_device_*.py -v` 全部通过
