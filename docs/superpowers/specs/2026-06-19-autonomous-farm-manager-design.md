# 农田自主决策闭环 — 设计文档

> 日期：2026-06-19 | 状态：设计完成 | 关联计划：待创建

## 1. 概述

### 1.1 目标

将现有的独立模块（摄像头巡检、Vision 分析、传感器采集、天气服务、规则引擎、设备控制）整合为一个完整的**感知→分析→决策→执行**智能闭环。

### 1.2 核心流程

```
定时触发 → 按区域分组 → 多源数据并行采集 → 状态聚合 → LLM 综合决策 → 安全校验 → 设备执行
```

### 1.3 关键决策

| 决策项 | 选择 |
|--------|------|
| 决策粒度 | 按区域（location）分组，每区域独立决策 |
| 决策自由度 | 半自主型：可超越预设规则，但受硬限制约束 |
| 架构方案 | 新建 `core/autonomous_farm_manager.py` 作为专用编排器 |

---

## 2. 架构

### 2.1 新增/修改文件

```
core/autonomous_farm_manager.py    # 新增：自主决策编排器 ★核心
app/scheduler_jobs.py              # 修改：替换摄像头巡检为轻量触发
app/agent/config.py                # 修改：新增自主决策配置项
app/api_routes.py                  # 修改：新增手动触发接口 + 报告查询
```

### 2.2 模块交互

```
┌─────────────────────────────────────────────────────────────┐
│                  AutonomousFarmManager (新增)                │
│                                                             │
│  run_cycle(username, region) → CycleReport                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ① collect_farm_state()                   ~15s       │   │
│  │    ├─ 拍照+Vision   ──▶ Camera + CropMonitorAgent  │   │
│  │    ├─ 传感器读数     ──▶ DeviceRegistry             │   │
│  │    ├─ 天气预报+历史  ──▶ WeatherService + history   │   │
│  │    ├─ 作物阶段       ──▶ PlantingTracker            │   │
│  │    └─ 病虫害风险     ──▶ disease_risk               │   │
│  │                                                      │   │
│  │ ② build_decision_prompt() → str          ~0.1s      │   │
│  │                                                      │   │
│  │ ③ request_decision(prompt) → DecisionPlan  ~5-10s   │   │
│  │    └─ LLM API (非Vision模型)                          │   │
│  │    └─ parse_decision() → validate_plan()             │   │
│  │                                                      │   │
│  │ ④ execute_plan(plan) → List[ActionResult]  ~varies  │   │
│  │    └─ RuleEngine.evaluate_action()                   │   │
│  │    └─ DeviceExecutor.execute_sync()                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
     │                    │                    │
     ▼                    ▼                    ▼
CropMonitorAgent    DeviceRegistry      DeviceExecutor
WeatherService      RuleEngine          PlantingTracker
```

### 2.3 与现有模块的关系

- **不侵入** CropMonitorAgent、DeviceAgent、RuleEngine、WeatherService
- **复用** 现有所有 core/ 和 devices/ 模块
- **替换** scheduler_jobs.py 中的 `check_camera_capture_job` 为 `check_autonomous_cycle_job`
- **共存** 与其他定时任务（提醒/天气/病害）并行运行，互不干扰

---

## 3. 数据结构

### 3.1 FarmState（聚合状态）

```python
@dataclass
class CameraView:
    device_id: str
    location: str
    image_base64: Optional[str]       # 拍照成功时有值
    vision_analysis: Optional[Dict]   # Vision AI 分析结果
    error: Optional[str]              # 拍照/分析失败时的错误

@dataclass
class FarmState:
    region: str
    username: str
    timestamp: str
    # 多源数据
    camera_views: List[CameraView]
    sensor_readings: Dict[str, Any]    # {"soil_moisture": 28.5, "temperature": 22.3, ...}
    current_weather: Optional[Dict]    # WeatherInfo 字典
    weather_forecast: List[Dict]       # 未来3天预报
    weather_persistence: List[Dict]    # 持续异常检测结果
    active_crops: List[Dict]           # 进行中的种植进度
    disease_risks: List[Dict]          # 病虫害风险评估
    recent_actions: List[Dict]         # 最近10条设备操作日志
```

### 3.2 DecisionPlan（LLM 决策输出）

```json
{
  "region": "大棚A区",
  "overall_assessment": "该区域番茄处于结果期，土壤湿度偏低(28%)...",
  "actions": [
    {
      "action": "irrigate",
      "device_hint": "灌溉",
      "params": {"duration": 25},
      "urgency": "today",
      "reason": "土壤湿度28%低于适宜范围，且未来无雨"
    }
  ],
  "follow_up": "建议3天后再次巡检"
}
```

### 3.3 CycleReport（巡检报告）

```python
@dataclass
class ActionResult:
    action: str
    device_id: str
    success: bool
    message: str
    rule_matched: Optional[str]    # 匹配到的规则ID
    executed_params: Dict

@dataclass
class CycleReport:
    cycle_id: str
    username: str
    region: str
    timestamp: str
    farm_state: FarmState
    decision_plan: Optional[DecisionPlan]  # None 表示决策失败
    execution_results: List[ActionResult]
    fallback_used: bool                    # 是否使用了规则引擎 fallback
    summary: str
    duration_ms: int
```

---

## 4. LLM 决策引擎

### 4.1 提示词结构

```
[系统指令]
你是农业自主决策专家。根据农田综合状态数据，生成结构化的操作计划。
决策原则：
1. 优先解决紧急问题（干旱 > 病虫害 > 缺肥 > 其他）
2. 操作参数在安全范围内尽可能精确（看数据定量，不要拍脑袋）
3. 如果一切正常，actions 为空数组即可
4. 考虑未来天气：如果预报有雨，推迟灌溉

[硬限制 - 不可违反]
- 单次灌溉 ≤ 120分钟
- 单次施肥 ≤ 50kg
- 夜间时段(22:00-06:00)禁止噪音操作（灌溉/施肥/通风）
- 同一设备10分钟内不可重复执行相同操作

[当前农场状态]
区域: {region}
作物: {crops_summary}
天气: 当前{temp}°C 湿度{hum}% | 预报: {forecast_text}
持续异常: {persistence_text}
传感器: {sensors_text}
摄像头分析: {cameras_text}
病虫害风险: {disease_text}
近期操作: {recent_actions_text}

[输出要求]
严格按以下JSON格式输出，不要包含markdown代码块标记：
{"region":"","overall_assessment":"","actions":[...],"follow_up":""}
```

### 4.2 决策校验层

在 LLM 输出 → 执行之间，进行以下校验：

| 校验项 | 规则 | 失败处理 |
|--------|------|---------|
| JSON 解析 | 合法 JSON，包含必填字段 | 尝试修复截断 → 失败则 fallback |
| action 白名单 | irrigate/fertigate/ventilate/light/heat/cool/alert | 跳过非法 action |
| 参数硬上限 | duration≤120, amount≤50 | 裁剪到上限值 |
| 设备存在性 | device_id 在注册表中且在线 | 跳过该 action |
| 时间窗口 | 夜间禁止灌溉/施肥/通风（除非 autonomy=high） | 降级为 alert |
| 重复操作 | 10min 内相同 device+action 不重复 | 跳过 |
| 互斥检查 | 同一设备不能同时 heat+cool | 按 urgency 优先 |

### 4.3 Fallback 链路

```
LLM 不可用 (超时/报错/返回非法JSON)
    │
    ▼
规则引擎兜底:
  遍历用户启用规则 → 匹配传感器+天气条件 → 触发匹配规则
    │
    ▼
  无匹配规则 → 仅生成告警，不执行操作
```

### 4.4 配置项

```env
# 自主决策
AUTO_DECISION_INTERVAL=30          # 巡检间隔（分钟）
AUTO_DECISION_MODEL=               # 决策LLM模型（空=复用LLM_MODEL）
AUTO_DECISION_REGIONS=             # 限定区域，逗号分隔（空=所有区域）
AUTO_DECISION_NIGHT_MODE=silent    # silent=只告警 / full=照常执行
AUTO_DECISION_TIMEOUT=30           # LLM 决策超时（秒）
AUTO_DECISION_MIN_INTERVAL=10      # 同区域最小间隔（分钟）
AUTO_DECISION_MAX_ACTIONS=5        # 单次决策最大操作数
```

---

## 5. 定时调度

### 5.1 注册

```python
# api_server.py
scheduler.add_job(
    check_autonomous_cycle_job, "interval",
    minutes=int(os.getenv("AUTO_DECISION_INTERVAL", "30")),
    id="autonomous_cycle",
)
```

### 5.2 调度入口

```python
def check_autonomous_cycle_job():
    """遍历用户 → 发现区域 → 触发自主决策"""
    for username in get_active_users():
        regions = discover_regions(username)
        for region in regions:
            # 检查最小间隔
            if should_skip(region): continue
            manager = AutonomousFarmManager()
            report = manager.run_cycle(username, region)
            save_report(report)
```

### 5.3 区域发现

按设备的 `location` 字段自动分组：

```python
def discover_regions(username) -> List[str]:
    devices = discover_all_devices(username)
    locations = set(d.location for d in devices if d.location)
    # 如果配置了 AUTO_DECISION_REGIONS，取交集
    if configured_regions:
        locations = locations & set(configured_regions)
    return sorted(locations)
```

### 5.4 与现有定时任务的关系

| 任务 | 间隔 | 职责 | 变更 |
|------|------|------|------|
| `autonomous_cycle` | 30min | **新增** 自主决策闭环 | 全新 |
| `reminders` | 5min | 提醒检查+SMS | 不变 |
| `weather` | 30min | 天气预警缓存 | 不变 |
| `disease` | 6h | 病虫害风险评估 | 不变 |
| `device_rules` | 5min | 设备规则轮询 | 不变 |
| `task_execution` | 3min | 任务自动执行 | 不变 |
| `camera_capture` | 30min | **移除**，由 autonomous_cycle 替代 | 删除 |

---

## 6. 执行流程（单次巡检）

```
run_cycle("user", "大棚A")

① 收集阶段 (并行, ~15s)
  ├─ 📷 拍照×N + Vision分析  ──────────┐
  ├─ 🌡 传感器读数×M        ──────────┤ asyncio.gather
  ├─ 🌤 天气API + 历史异常  ──────────┤
  ├─ 🌱 作物阶段            ──────────┘
  └─ 🦠 病害风险            ──────────┘

② 聚合阶段
  FarmState → build_decision_prompt() → 文本

③ 决策阶段 (~5-10s)
  LLM API → parse_decision() → validate_plan()
  ├─ 成功 → DecisionPlan
  └─ 失败 → fallback 规则引擎

④ 执行阶段 (逐个，按urgency排序)
  for action in plan.actions:
    ├─ RuleEngine.evaluate_action(action)
    ├─ apply_autonomy(decision, autonomy_level)
    ├─ auto_execute → DeviceExecutor.execute()
    ├─ need_confirm → 写入待确认队列
    └─ rejected → 跳过

⑤ 保存报告
  CycleReport → data/{username}/autonomous_reports/{cycle_id}.json
```

---

## 7. 错误处理

### 7.1 分级策略

| 级别 | 场景 | 处理 |
|------|------|------|
| 跳过继续 | 单个摄像头离线、单个传感器故障 | 该数据标记 unavailable，其余照常 |
| 降级运行 | 天气API超时 | 用 weather_history 最近3天估算 |
| 降级运行 | LLM 决策超时 | fallback 到规则引擎 |
| 安全停止 | 所有摄像头离线 | 跳过该区域本轮，写告警日志 |
| 立即中止 | 执行时硬件故障 | 停止该 action，后续 action 继续 |

### 7.2 具体异常处理

```
拍照失败    → CameraView.error = str(e)，继续下一个摄像头
Vision超时  → 重试1次，仍失败标记 "analysis_unavailable"
传感器离线  → sensor_readings[device_id] = None
天气API失败 → 用 weather_history 估算；如果历史也为空，标记 unavailable
LLM超时     → 重试1次 → fallback到规则引擎
JSON解析失败 → 尝试修复截断JSON → fallback
校验全部驳回 → 记录 "本轮无有效操作"，不执行
设备执行失败 → 重试3次(5s/15s/45s) → 跳过
```

---

## 8. API 接口

### 8.1 新增接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/autonomous/trigger` | 手动触发一次完整巡检（指定区域） |
| `GET` | `/api/autonomous/reports` | 查询历史巡检报告列表 |
| `GET` | `/api/autonomous/reports/{id}` | 查看单次巡检详情 |
| `GET` | `/api/autonomous/status` | 当前自主决策运行状态 |

### 8.2 手动触发请求

```json
POST /api/autonomous/trigger
{
  "region": "大棚A区",
  "username": "default"
}

Response:
{
  "success": true,
  "cycle_id": "cycle_20260619_143000_a1b2c3",
  "message": "巡检完成，执行了2项操作"
}
```

---

## 9. 测试策略

### 9.1 单元测试

| 测试对象 | 内容 |
|----------|------|
| `FarmState` | 序列化/反序列化正确性 |
| `validate_plan()` | 白名单校验、参数裁剪、互斥检测、重复检测 |
| `build_decision_prompt()` | 各字段正确拼入提示文本 |
| `parse_decision()` | 正常JSON解析、截断修复、非法输入处理 |
| `discover_regions()` | 按 location 正确分组 |

### 9.2 集成测试（使用 SimulatorDriver）

| 场景 | 验证点 |
|------|--------|
| 正常闭环 | Mock LLM 返回 irrigate → 规则匹配 → 设备执行成功 |
| LLM 超时 fallback | Mock LLM 超时 → 规则引擎接管 → 匹配规则自动执行 |
| 全部校验驳回 | Mock LLM 返回非法操作 → 日志记录，不执行 |
| 传感器全离线 | sensor_readings 全部为 None → 决策仍可进行 |
| 零摄像头区域 | camera_views 为空 → 仅基于传感器+天气决策 |

### 9.3 边界测试

- 区域无任何设备
- LLM 返回空 actions 数组
- LLM 返回超过 MAX_ACTIONS 个操作（应裁剪）
- 两个 action 指向同一设备（互斥检测）
- 夜间时段触发巡检

---

## 10. 报告存储

巡检报告保存为 JSON 文件，按用户分目录：

```
data/{username}/autonomous_reports/
├── cycle_20260619_140000_a1b2c3.json
├── cycle_20260619_143000_d4e5f6.json
└── ...
```

每个报告文件包含完整的 FarmState + DecisionPlan + 执行结果，支持历史回溯和分析。

---

## 11. 风险与限制

| 风险 | 缓解措施 |
|------|---------|
| LLM 幻觉导致不合理操作 | 硬限制校验层 + 规则引擎兜底 |
| Vision API 成本高 | 可配置仅巡检部分摄像头 |
| 夜间执行打扰 | NIGHT_MODE=silent 只告警不执行 |
| 多用户并发 | 串行处理每用户，asyncio 并行采集数据 |
| LLM 决策不稳定 | temperature 设为 0.2，降低随机性 |

---

## 12. 实施范围

### 包含
- `core/autonomous_farm_manager.py` 完整实现
- `scheduler_jobs.py` 替换摄像头巡检
- `app/agent/config.py` 新增配置
- `app/api_routes.py` 新增手动触发+报告查询
- `tests/test_autonomous_farm_manager.py` 单元+集成测试

### 不包含
- 现有 Agent 体系改造（保持不变）
- Streamlit UI 改造（后续迭代）
- 新硬件驱动（复用现有 4 种协议）
- 远程云端决策（本地 LLM API 调用）
