# IoT 智能设备控制 Agent — 设计方案

> 日期：2026-06-14
> 状态：待评审
> 目标：将农业 Agent 从"顾问"升级为"执行者"，支持自主控制 IoT 设备

---

## 1. 需求概述

### 1.1 目标

在现有"智能种植规划助手"基础上，新增 IoT 设备自主控制能力。让 Agent 不止生成方案，而是能在用户设定的安全边界内，自动执行灌溉、施肥、通风、补光等操作。

### 1.2 核心需求

| 维度 | 选择 |
|------|------|
| 核心方向 | IoT 智能设备控制 |
| 自主权级别 | 规则驱动 — 边界内自动执行，边界外询问用户 |
| 适用场景 | 通用框架，不绑定特定硬件 |
| 设备类型 | 全类型：灌溉/施肥/环控/光照等 |
| 对接方式 | IoT 平台 API（涂鸦等）+ 自建协议（MQTT/Modbus）双通道 |

### 1.3 非功能需求

- 安全第一：任何操作需经过规则引擎校验，禁止夜间灌溉等危险行为
- 可扩展：添加新设备类型/协议只需实现 Driver 接口
- 可测试：内置虚拟设备模拟器，无需真实硬件即可全链路测试
- 低耦合：设备层变更不影响 Agent 层

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit 前端                         │
│  设备仪表盘 │ 规则编辑器 │ 对话控制 │ 执行日志             │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/WebSocket
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI 后端                             │
│  /api/chat │ /api/devices │ /api/rules │ /api/actions    │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              LangGraph Agent 工作流                       │
│                                                          │
│  parse_input → classify_intent → agent_dispatch          │
│                                     │                    │
│                          ┌──────────┼──────────┐         │
│                          │          │          │         │
│                    DeviceAgent  WeatherAgent  Disease..  │
│                          │                               │
│                    ┌─────┴─────┐                         │
│                    │  规则引擎   │ ← 混合决策核心          │
│                    └─────┬─────┘                         │
└──────────────────────────┼──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  设备抽象层 (DAL)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │
│  │ 涂鸦驱动  │  │ MQTT驱动  │  │Modbus驱动│  │ 虚拟设备 │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘  │
│         ↑           ↑            ↑              ↑        │
│    涂鸦IoT云    MQTT Broker   RS-485总线    本地模拟器    │
└──────────────────────────────────────────────────────────┘
```

三层新增模块（DeviceAgent + 规则引擎 + 设备抽象层）以插件形式插入现有架构，不破坏原有 Agent 工作流。

---

## 3. 规则引擎 — 混合决策核心

### 3.1 设计原则

- **用户定边界，AI 在边界内优化**
- 三层校验：硬限制（不可突破）→ 约束边界（可触发确认）→ AI 微调（边界内自由）

### 3.2 决策流程

```
触发源（传感器/定时/用户对话/Agent互调）
    │
    ▼
┌─────────────────────────────┐
│  条件匹配                     │
│  检查规则触发条件是否满足      │
└───────────┬─────────────────┘
            │ 命中
            ▼
┌─────────────────────────────┐
│  约束校验                     │
│  ├─ 硬限制检查（夜间禁止等）   │ → 违反 → ❌ 拒绝执行
│  ├─ 边界检查（时长/水量等）    │ → 超越 → ⚠️ 推送确认
│  └─ AI 微调范围检查           │ → 范围内 → ✅ 自动执行
└───────────┬─────────────────┘
            │
    自动执行 / 推送确认 / 拒绝
```

### 3.3 规则数据结构

```python
{
    "id": "rule_wheat_irrigation",
    "name": "小麦自动灌溉",
    "enabled": True,

    # 触发条件（支持 AND/OR）
    "trigger": {
        "logic": "AND",
        "conditions": [
            {"type": "sensor",  "device": "soil_sensor_01", "field": "humidity", "op": "<", "value": 30},
            {"type": "weather", "field": "rain_24h",        "op": "==", "value": False},
            {"type": "time",    "op": "between",             "value": ["06:00", "20:00"]}
        ]
    },

    # 执行动作
    "action": {
        "device_id": "irrigation_valve_01",
        "command": "start",
        "params": {"duration": 30, "flow_rate": "medium"}
    },

    # 安全约束
    "constraints": {
        "max_duration_per_use": 60,
        "max_duration_per_day": 180,
        "min_interval_minutes": 120,
        "forbidden_hours": [22, 23, 0, 1, 2, 3, 4, 5],
        "require_confirm_if": [
            "duration > 45",
            "cost_estimate > 50",
            "weather_forecast_conflict"
        ]
    },

    # AI 增强（可选）
    "ai_enhance": {
        "enabled": True,
        "can_adjust": ["duration", "flow_rate"],
        "adjust_range": {"duration": [-15, 15]}
    }
}
```

### 3.4 存储

规则存储在 `data/{username}/device_rules.json`，规则引擎启动时加载到内存，变更后实时写回。

---

## 4. 设备抽象层 — Driver 插件架构

### 4.1 核心接口

```python
class DeviceCapability(Enum):
    IRRIGATE    = "irrigate"
    FERTIGATE   = "fertigate"
    VENTILATE   = "ventilate"
    HEAT        = "heat"
    COOL        = "cool"
    SHADE       = "shade"
    LIGHT       = "light"
    READ_SENSOR = "read_sensor"


class BaseDeviceDriver(ABC):
    driver_name: str = "base"

    @abstractmethod
    async def connect(self) -> bool: ...
    @abstractmethod
    async def disconnect(self) -> None: ...
    @abstractmethod
    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult: ...
    @abstractmethod
    async def read_state(self, device_id: str) -> Dict[str, Any]: ...
    @abstractmethod
    async def discover(self) -> List[DeviceInfo]: ...
    @abstractmethod
    async def health_check(self) -> bool: ...
```

### 4.2 四个内置 Driver

| Driver | 协议/平台 | 场景 |
|--------|----------|------|
| `TuyaDriver` | 涂鸦 OpenAPI (OAuth) | 对接涂鸦生态设备（主流智能家居） |
| `MQTTDriver` | MQTT 3.1.1/5.0 | 通用 IoT 设备，自建协议 |
| `ModbusDriver` | Modbus RTU/TCP | 工业传感器、PLC 控制器 |
| `SimulatorDriver` | 本地内存 | 开发测试，无需真实硬件 |

### 4.3 注册中心

```python
class DeviceDriverRegistry:
    def register(self, name: str, driver: BaseDeviceDriver): ...
    async def discover_all(self) -> List[DeviceInfo]: ...
    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult: ...
    async def read_state(self, device_id: str) -> Dict[str, Any]: ...
```

上层代码只与 `DeviceDriverRegistry` 和 `BaseDeviceDriver` 交互，不感知底层协议差异。

### 4.4 设备注册表

`data/{username}/device_registry.json` — 持久化设备信息：

```json
{
  "devices": [
    {
      "device_id": "irrigation_valve_01",
      "name": "灌溉阀#1",
      "driver": "mqtt",
      "capabilities": ["irrigate"],
      "sensors": ["flow_rate"],
      "location": "大棚A区",
      "metadata": {"mqtt_topic": "devices/valve01/control"}
    }
  ]
}
```

---

## 5. DeviceAgent + 工作流整合

### 5.1 Agent 注册

`DeviceAgent` 作为第 6 个专业 Agent 注册到 `AgentOrchestrator`，处理 `device_control` 意图。

### 5.2 两条执行通路

**通路 1 — 对话驱动（用户主动）**

```
用户: "帮小麦浇30分钟水"
    → classify_intent → "device_control"
    → DeviceAgent.invoke()
    → LLM 解析意图 → 匹配设备/规则
    → 规则引擎评估
    → 执行 / 确认 / 拒绝
```

**通路 2 — 事件驱动（Agent 自主）**

```
后台调度器(APScheduler) 每 5 分钟轮询：
    → 遍历所有启用的规则
    → 检查触发条件（传感器值/时间/天气）
    → 命中 → 规则引擎评估
    → 自主执行 → 写日志 + 推送通知
```

### 5.3 Agent 间互调

DeviceAgent 可被其他 Agent 通过 `interop_call()` 调用：

- **DiseaseAgent 检测到虫害** → 调 DeviceAgent 检查喷药条件 → 获取气象窗口 → 自动执行喷药
- **WeatherAgent 检测到霜冻预警** → 调 DeviceAgent 检查加热设备 → 自动开启防霜

### 5.4 State 扩展

```python
class AgentState(BaseModel):
    # ... 原有字段 ...
    device_command: Optional[DeviceCommand] = None
    device_result: Optional[DeviceResult] = None
    pending_action: Optional[Dict] = None       # 待确认操作
    matched_rules: List[str] = []
```

### 5.5 新增意图关键词

```python
DEVICE_KEYWORDS = [
    "浇水", "灌溉", "施肥", "通风", "开窗", "遮阳",
    "补光", "开灯", "关灯", "加热", "降温", "喷雾",
    "自动控制", "手动控制", "设备状态"
]
```

---

## 6. 前端设计

### 6.1 导航栏新增

在现有侧边栏/导航栏增加两个入口：
- **设备仪表盘** (`devices`) — 设备状态监控 + 快捷操作
- **规则管理** (`rules`) — 自动规则的 CRUD

### 6.2 设备仪表盘 (`views/devices.py`)

- 状态概览（在线/离线/待确认计数）
- 设备列表（名称、状态、最新传感器读数、快捷操作按钮）
- 待确认操作区（AI 建议但需要用户确认的操作，支持一键确认/修改/拒绝）
- 今日执行日志（时间线展示）

### 6.3 规则编辑器 (`views/rules.py`)

- 规则列表（启用/暂停开关、名称、触发条件摘要）
- 规则编辑表单：
  - 触发条件构建器（传感器阈值、天气条件、时间范围，支持 AND/OR 组合）
  - 执行动作选择（设备下拉框、命令选择、参数设置）
  - 安全边界设定（硬限制数值、需确认条件勾选）
  - AI 微调开关 + 调节范围
- 保存 / 测试 / 删除按钮

### 6.4 对话页增强 (`views/chat.py`)

DeviceAgent 的消息渲染为富交互卡片：
- 执行预览（设备名、操作参数、当前环境数据）
- 实时状态更新（执行中 / 已完成 / 失败）
- 需要确认时渲染确认/拒绝按钮

---

## 7. API 路由新增

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/devices` | 获取设备列表及状态 |
| POST | `/api/devices/{id}/command` | 向设备发送指令 |
| GET | `/api/devices/{id}/state` | 获取设备实时状态 |
| GET | `/api/rules` | 获取规则列表 |
| POST | `/api/rules` | 创建新规则 |
| PUT | `/api/rules/{id}` | 更新规则 |
| DELETE | `/api/rules/{id}` | 删除规则 |
| POST | `/api/rules/{id}/test` | 测试规则（仅评估不执行） |
| GET | `/api/actions/log` | 获取执行日志 |
| POST | `/api/actions/{id}/confirm` | 确认待定操作 |
| POST | `/api/actions/{id}/reject` | 拒绝待定操作 |

---

## 8. 错误处理与安全

### 8.1 设备层错误

| 场景 | 处理策略 |
|------|---------|
| 设备离线 | 立即通知用户，命令入队等待设备上线后重试（最多保留 24h） |
| 指令超时 | 最多重试 3 次，间隔递增（5s/15s/45s），全失败则告警 |
| 执行失败 | 记录完整错误日志，通过对话/推送通知用户 |
| 传感器数据异常 | 标记为 "unreliable"，暂停依赖该传感器的自动规则 |

### 8.2 安全硬限制（代码级，不可配置）

- 灌溉单次不超过 120 分钟
- 施肥单次不超过设备额定最大流量
- 同一设备两次操作间隔不少于 10 秒（防重复触发）
- 紧急停止命令跳过所有校验直接执行

### 8.3 审计日志

所有设备操作记录到 `data/{username}/device_log.json`：

```json
{
  "timestamp": "2026-06-14T10:00:00",
  "device_id": "irrigation_valve_01",
  "command": "start",
  "params": {"duration": 35},
  "trigger": "rule_wheat_irrigation",
  "decision": "auto_execute",
  "result": "success",
  "rule_evaluation": {...}
}
```

---

## 9. 项目文件变更清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `app/agent/agents/device_agent.py` | DeviceAgent 实现 |
| `app/agent/nodes/device_control.py` | 设备控制工作流节点 |
| `core/device_rule_engine.py` | 规则引擎 |
| `core/device_abstraction.py` | 设备抽象基类 + 注册中心 |
| `core/device_executor.py` | 指令执行器（重试/超时/队列） |
| `devices/__init__.py` | 设备模块入口 |
| `devices/tuya_driver.py` | 涂鸦 IoT 平台驱动 |
| `devices/mqtt_driver.py` | MQTT 通用驱动 |
| `devices/modbus_driver.py` | Modbus 驱动 |
| `devices/simulator_driver.py` | 虚拟设备模拟器 |
| `app/views/devices.py` | 设备仪表盘页面 |
| `app/views/rules.py` | 规则编辑器页面 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `app/agent/state.py` | 新增 4 个设备相关字段 |
| `app/agent/config.py` | 新增 `DEVICE_KEYWORDS` |
| `app/agent/graph.py` | 新增 `device_control` 节点及路由 |
| `app/agent/agents/orchestrator.py` | 注册 DeviceAgent |
| `app/test1.py` | 新增 `devices`/`rules` 页面路由 |
| `app/api_routes.py` | 新增 10 个设备/规则 API |
| `app/ui/sidebar.py` | 导航栏新增入口 |
| `app/views/chat.py` | 设备消息交互卡片渲染 |
| `app/scheduler_jobs.py` | 新增规则轮询任务 |
| `requirements.txt` | 新增 `paho-mqtt`、`pymodbus` 等依赖 |

---

## 10. 依赖新增

```
paho-mqtt>=2.0.0        # MQTT 客户端
pymodbus>=3.6.0         # Modbus 协议
```

涂鸦驱动使用 HTTP API（`requests` 已有），无需额外依赖。

---

## 11. 测试策略

### 11.1 单元测试

- `test_device_rule_engine.py` — 规则匹配/约束校验/AI 微调
- `test_device_abstraction.py` — Driver 注册/发现/执行
- `test_device_agent.py` — 意图解析/规则匹配
- `test_simulator_driver.py` — 虚拟设备读写/故障模拟

### 11.2 集成测试

- SimulatorDriver + DeviceAgent + RuleEngine 全链路
- 模拟传感器数据变化 → 规则触发 → 自动执行 → 状态更新
- 模拟设备离线/超时 → 重试 → 失败降级

### 11.3 端到端测试

- 启动全套服务（FastAPI + SimulatorDriver）
- 通过 API 创建规则 → 模拟传感器数据 → 验证自动执行
- 通过对话接口发送设备控制请求 → 验证回复卡片

---

## 12. 分期规划

| 阶段 | 内容 | 预计 |
|------|------|------|
| **Phase 1** | 设备抽象层 + SimulatorDriver + 规则引擎核心 | 基础框架 |
| **Phase 2** | DeviceAgent + 工作流整合 + Agent 互调 | AI 决策 |
| **Phase 3** | 前端（设备仪表盘 + 规则编辑器 + 对话卡片） | 用户界面 |
| **Phase 4** | MQTT Driver + Modbus Driver | 真实协议 |
| **Phase 5** | Tuya Driver + 实际硬件联调 | 生态对接 |
