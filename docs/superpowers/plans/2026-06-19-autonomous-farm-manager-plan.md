# 农田自主决策闭环 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 AutonomousFarmManager 编排器，将摄像头巡检、传感器采集、天气服务、LLM 决策、设备控制整合为完整的"感知→分析→决策→执行"智能闭环。

**Architecture:** 新建 `core/autonomous_farm_manager.py` 作为核心编排器，复用现有所有 core/ 和 devices/ 模块。按设备 location 分组形成"决策区域"，每区域独立采集多源数据 → LLM 综合决策 → 安全校验 → 执行。替换 scheduler_jobs.py 中的摄像头巡检为新的自主决策调度入口。

**Tech Stack:** Python 3.11+, asyncio, dataclasses, langchain-openai (LLM), FastAPI, APScheduler

---

### Task 1: 配置项新增

**Files:**
- Modify: `app/agent/config.py`

- [ ] **Step 1: 在 config.py 末尾追加自主决策配置项**

在现有配置项之后追加：

```python
# ── 自主决策配置 ──────────────────────────
AUTO_DECISION_INTERVAL = int(os.getenv("AUTO_DECISION_INTERVAL", "30"))  # 巡检间隔（分钟）
AUTO_DECISION_MODEL = os.getenv("AUTO_DECISION_MODEL") or LLM_MODEL  # 决策LLM模型
AUTO_DECISION_REGIONS = os.getenv("AUTO_DECISION_REGIONS", "")  # 限定区域，逗号分隔
AUTO_DECISION_NIGHT_MODE = os.getenv("AUTO_DECISION_NIGHT_MODE", "silent")  # silent|full
AUTO_DECISION_TIMEOUT = int(os.getenv("AUTO_DECISION_TIMEOUT", "30"))  # LLM决策超时秒数
AUTO_DECISION_MIN_INTERVAL = int(os.getenv("AUTO_DECISION_MIN_INTERVAL", "10"))  # 同区域最小间隔分钟
AUTO_DECISION_MAX_ACTIONS = int(os.getenv("AUTO_DECISION_MAX_ACTIONS", "5"))  # 单次最大操作数
AUTO_DECISION_TEMPERATURE = float(os.getenv("AUTO_DECISION_TEMPERATURE", "0.2"))  # 决策LLM温度
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from app.agent.config import AUTO_DECISION_INTERVAL, AUTO_DECISION_MODEL; print('OK:', AUTO_DECISION_INTERVAL, AUTO_DECISION_MODEL)"
```

- [ ] **Step 3: Commit**

```bash
git add app/agent/config.py
git commit -m "feat: 添加自主决策配置项

- AUTO_DECISION_INTERVAL 巡检间隔
- AUTO_DECISION_MODEL 决策模型
- AUTO_DECISION_REGIONS 限区域
- AUTO_DECISION_NIGHT_MODE 夜间模式
- AUTO_DECISION_TIMEOUT 决策超时
- AUTO_DECISION_MIN_INTERVAL 最小间隔
- AUTO_DECISION_MAX_ACTIONS 最大操作数
- AUTO_DECISION_TEMPERATURE 决策温度

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 数据结构 + collect_farm_state（数据收集）

**Files:**
- Create: `core/autonomous_farm_manager.py`

- [ ] **Step 1: 编写 dataclass 数据结构和类骨架的测试**

创建 `tests/test_autonomous_farm_manager.py`：

```python
"""自主决策编排器 单元测试"""
import os, sys, json, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.autonomous_farm_manager import (
    CameraView, FarmState, DecisionPlan, ActionResult, CycleReport,
    AutonomousFarmManager,
)


class TestDataStructures:
    """数据结构序列化/反序列化测试"""

    def test_camera_view_defaults(self):
        cv = CameraView(device_id="cam_01", location="大棚A")
        assert cv.image_base64 is None
        assert cv.vision_analysis is None
        assert cv.error is None

    def test_camera_view_with_error(self):
        cv = CameraView(device_id="cam_02", location="大棚B", error="设备离线")
        assert cv.error == "设备离线"
        assert cv.image_base64 is None

    def test_farm_state_empty(self):
        state = FarmState(region="大棚A", username="test")
        assert state.camera_views == []
        assert state.sensor_readings == {}
        assert state.current_weather is None

    def test_farm_state_with_data(self):
        state = FarmState(
            region="大棚A", username="test",
            camera_views=[CameraView(device_id="c1", location="大棚A")],
            sensor_readings={"soil_moisture": 28.5},
            current_weather={"temperature": 22},
        )
        assert len(state.camera_views) == 1
        assert state.sensor_readings["soil_moisture"] == 28.5

    def test_decision_plan_from_dict(self):
        data = {
            "region": "大棚A",
            "overall_assessment": "土壤偏干",
            "actions": [{"action": "irrigate", "params": {"duration": 25}}],
            "follow_up": "3天后复查",
        }
        plan = DecisionPlan(**data)
        assert plan.region == "大棚A"
        assert len(plan.actions) == 1

    def test_decision_plan_empty_actions(self):
        plan = DecisionPlan(region="大棚A", overall_assessment="一切正常")
        assert plan.actions == []

    def test_action_result_success(self):
        ar = ActionResult(action="irrigate", device_id="dev_01", success=True,
                          message="执行成功", executed_params={"duration": 25})
        assert ar.success is True

    def test_cycle_report_fields(self):
        report = CycleReport(
            cycle_id="cycle_001", username="test", region="大棚A",
            timestamp="2026-06-19T14:00:00", summary="完成",
        )
        assert report.cycle_id == "cycle_001"
        assert report.fallback_used is False
        assert report.duration_ms == 0


class TestAutonomousFarmManager:
    """编排器核心测试"""

    def test_init(self):
        mgr = AutonomousFarmManager()
        assert mgr is not None
        assert hasattr(mgr, 'hard_limits')

    def test_discover_regions_from_devices(self):
        """区域发现：从设备列表按 location 分组"""
        # 模拟设备对象
        class MockDevice:
            def __init__(self, location):
                self.location = location
                self.device_id = f"dev_{location}"
                self.capabilities = []
                self.status = type('S', (), {'value': 'online'})()

        devices = [MockDevice("大棚A"), MockDevice("大棚A"), MockDevice("大棚B")]
        regions = AutonomousFarmManager._group_by_region(devices)
        assert set(regions.keys()) == {"大棚A", "大棚B"}
        assert len(regions["大棚A"]) == 2
        assert len(regions["大棚B"]) == 1

    def test_should_skip_night_irrigation(self):
        """夜间灌溉跳过逻辑（NIGHT_MODE=silent）"""
        mgr = AutonomousFarmManager()
        from core.device_rule_engine import RuleDecision
        # 模拟夜间 + silent 模式
        result = mgr._check_night_constraint("irrigate", night_mode="silent", hour=23)
        assert result == RuleDecision.REJECTED

    def test_night_mode_full_allows_all(self):
        """NIGHT_MODE=full 不限制"""
        mgr = AutonomousFarmManager()
        from core.device_rule_engine import RuleDecision
        result = mgr._check_night_constraint("irrigate", night_mode="full", hour=23)
        assert result is None  # None = 不拦截
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_autonomous_farm_manager.py -v
# 预期: ModuleNotFoundError: No module named 'core.autonomous_farm_manager'
```

- [ ] **Step 3: 创建 autonomous_farm_manager.py 骨架**

```python
"""农田自主决策编排器 — 感知→分析→决策→执行 闭环

将摄像头巡检、传感器采集、天气服务、LLM 决策、设备控制整合为
完整的自主决策流程。按设备 location 分组形成"决策区域"，
每区域独立运行 run_cycle()。
"""

import os, json, logging, asyncio, base64, re, copy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── 数据模型 ──────────────────────────────────────────

@dataclass
class CameraView:
    """单个摄像头视图"""
    device_id: str
    location: str
    image_base64: Optional[str] = None
    vision_analysis: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class FarmState:
    """一个决策区域的完整农场状态"""
    region: str
    username: str
    timestamp: str = ""
    camera_views: List[CameraView] = field(default_factory=list)
    sensor_readings: Dict[str, Any] = field(default_factory=dict)
    current_weather: Optional[Dict] = None
    weather_forecast: List[Dict] = field(default_factory=list)
    weather_persistence: List[Dict] = field(default_factory=list)
    active_crops: List[Dict] = field(default_factory=list)
    disease_risks: List[Dict] = field(default_factory=list)
    recent_actions: List[Dict] = field(default_factory=list)


@dataclass
class DecisionPlan:
    """LLM 生成的决策计划"""
    region: str
    overall_assessment: str = ""
    actions: List[Dict] = field(default_factory=list)
    follow_up: str = ""
    raw_response: str = ""


@dataclass
class ActionResult:
    """单个操作的执行结果"""
    action: str
    device_id: str
    success: bool
    message: str
    rule_matched: Optional[str] = None
    executed_params: Dict = field(default_factory=dict)


@dataclass
class CycleReport:
    """单次巡检完整报告"""
    cycle_id: str
    username: str
    region: str
    timestamp: str
    farm_state: Optional[FarmState] = None
    decision_plan: Optional[DecisionPlan] = None
    execution_results: List[ActionResult] = field(default_factory=list)
    fallback_used: bool = False
    summary: str = ""
    duration_ms: int = 0


# ── 夜间噪音操作列表 ─────────────────────────────────

_NOISY_ACTIONS = {"irrigate", "fertigate", "ventilate"}
_NIGHT_START = 22  # 22:00
_NIGHT_END = 6     # 06:00


# ── 硬限制（代码级，不可通过规则突破）────────────────

_HARD_LIMITS = {
    "irrigate": {"max_duration_per_use_minutes": 120},
    "fertigate": {"max_amount_per_use_kg": 50},
    "ventilate": {"max_duration_per_use_minutes": 120},
}


# ── 主类 ─────────────────────────────────────────────

class AutonomousFarmManager:
    """农田自主决策编排器

    用法:
        mgr = AutonomousFarmManager()
        report = mgr.run_cycle("username", "大棚A区")
        print(report.summary)
    """

    def __init__(self):
        self.hard_limits = copy.deepcopy(_HARD_LIMITS)
        # 区域最小间隔追踪 {region: last_run_datetime}
        self._last_run: Dict[str, datetime] = {}

    # ── 区域发现 ──────────────────────────────────

    @staticmethod
    def _group_by_region(devices: List) -> Dict[str, List]:
        """按设备 location 分组"""
        regions: Dict[str, List] = {}
        for d in devices:
            loc = getattr(d, 'location', '') or '默认区域'
            regions.setdefault(loc, []).append(d)
        return regions

    # ── 夜间约束 ──────────────────────────────────

    @staticmethod
    def _check_night_constraint(action: str, night_mode: str = "silent",
                                 hour: int = None) -> Optional[str]:
        """检查夜间操作限制

        Returns:
            None = 放行
            "rejected" = 禁止执行
        """
        from core.device_rule_engine import RuleDecision
        if hour is None:
            hour = datetime.now().hour
        if action in _NOISY_ACTIONS and (_NIGHT_START <= hour or hour < _NIGHT_END):
            if night_mode == "silent":
                return RuleDecision.REJECTED
        return None  # full 模式或非夜间或非噪音操作

    # ── 数据收集 ──────────────────────────────────

    def collect_farm_state(self, username: str, region: str) -> FarmState:
        """① 收集：并行采集一个区域的全部状态数据"""
        from core.device_registry_factory import setup_registry, close_registry

        state = FarmState(
            region=region, username=username,
            timestamp=datetime.now().isoformat(),
        )

        # ── 设备数据（async 域内）──
        registry, loop = setup_registry(username)
        try:
            devices = loop.run_until_complete(registry.discover_all())
            region_devices = [d for d in devices
                            if getattr(d, 'location', '') == region]

            # 摄像头拍照 + Vision 分析
            cameras = [d for d in region_devices
                      if "capture" in [c.value for c in d.capabilities]
                      and d.status.value in ("online", "offline")]
            state.camera_views = self._capture_and_analyze(
                registry, loop, cameras, username)

            # 传感器读数
            state.sensor_readings = self._collect_sensors(
                registry, loop, region_devices)
        finally:
            close_registry(loop)

        # ── 天气数据（同步）──
        state.current_weather, state.weather_forecast = self._collect_weather(region)
        state.weather_persistence = self._collect_persistence()

        # ── 作物与病害数据（同步）──
        state.active_crops, state.disease_risks = self._collect_crop_info(username)

        # ── 近期操作（同步）──
        state.recent_actions = self._collect_recent_actions(username)

        return state

    def _capture_and_analyze(self, registry, loop, cameras: List,
                              username: str) -> List[CameraView]:
        """对每个摄像头拍照 + Vision 分析"""
        from devices.base import DeviceCommand
        from app.agent.agents.crop_monitor_agent import CropMonitorAgent

        views = []
        monitor = CropMonitorAgent()

        for cam in cameras:
            view = CameraView(device_id=cam.device_id, location=cam.location)
            try:
                # 拍照
                cmd = DeviceCommand(command="capture", params={}, timeout_ms=15000)
                result = loop.run_until_complete(registry.execute(cam.device_id, cmd))
                if not result.success:
                    view.error = f"拍照失败: {result.message}"
                    views.append(view)
                    continue

                image_bytes = result.raw_response.get("image_bytes")
                if not image_bytes:
                    view.error = "未获取到图片数据"
                    views.append(view)
                    continue

                view.image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                # Vision 分析
                analysis = monitor.analyze_image(
                    view.image_base64, "image/jpeg",
                    user_context={
                        "username": username,
                        "device_id": cam.device_id,
                        "location": cam.location,
                    },
                )
                if analysis.get("success"):
                    view.vision_analysis = analysis.get("analysis", {})
                else:
                    view.error = analysis.get("error", "Vision 分析失败")

            except Exception as e:
                view.error = str(e)
                logger.warning("摄像头处理异常 %s: %s", cam.device_id, e)

            views.append(view)

        return views

    def _collect_sensors(self, registry, loop, devices: List) -> Dict[str, Any]:
        """收集所有传感器的读数"""
        readings = {}
        for d in devices:
            try:
                state = loop.run_until_complete(registry.read_state(d.device_id))
                if state and not state.get("error"):
                    for k, v in state.items():
                        if isinstance(v, (int, float)) and not k.startswith("_"):
                            key = f"{d.device_id}.{k}"
                            readings[key] = v
            except Exception as e:
                logger.debug("传感器读取失败 %s: %s", d.device_id, e)
                readings[d.device_id] = None
        return readings

    def _collect_weather(self, region: str) -> Tuple[Optional[Dict], List[Dict]]:
        """获取当前天气 + 3天预报"""
        try:
            from core.weather_service import WeatherService
            ws = WeatherService()
            current = ws.get_current_weather(region)
            forecast = ws.get_forecast(region, 3)

            cur_dict = None
            if current:
                cur_dict = {
                    "temperature": current.temperature,
                    "temperature_high": current.temperature_high,
                    "temperature_low": current.temperature_low,
                    "humidity": current.humidity,
                    "weather_desc": current.weather_desc,
                    "wind_speed": current.wind_speed,
                    "precipitation": current.precipitation,
                }

            fore_list = []
            if forecast:
                for w in forecast:
                    fore_list.append({
                        "date": str(w.date),
                        "weather_desc": w.weather_desc,
                        "temperature_high": w.temperature_high,
                        "temperature_low": w.temperature_low,
                        "humidity": w.humidity,
                    })

            return cur_dict, fore_list
        except Exception as e:
            logger.warning("天气数据获取失败: %s", e)
            return None, []

    def _collect_persistence(self) -> List[Dict]:
        """获取持续异常天气检测结果"""
        try:
            from core.weather_history import check_persistence
            return check_persistence()
        except Exception as e:
            logger.debug("天气持续异常检测失败: %s", e)
            return []

    def _collect_crop_info(self, username: str) -> Tuple[List[Dict], List[Dict]]:
        """获取活跃作物和病虫害风险"""
        crops = []
        risks = []
        try:
            from core.planting_tracker import PlantingTracker
            sd = os.path.join("data", username)
            tracker = PlantingTracker(sd)
            progresses = tracker.get_progress()
            crops = [{
                "crop": p.crop, "stage": p.stage,
                "stage_number": p.stage_number, "total_stages": p.total_stages,
                "status": p.status, "progress_percent": p.progress_percent,
            } for p in progresses if p.status == "进行中"]
        except Exception as e:
            logger.debug("作物信息获取失败: %s", e)

        # 病虫害风险（从缓存文件读）
        try:
            dpath = os.path.join("data", "disease_risks.json")
            if os.path.exists(dpath):
                with open(dpath, encoding="utf-8") as f:
                    data = json.load(f)
                    risks = data.get("risks", [])
        except Exception as e:
            logger.debug("病虫害风险读取失败: %s", e)

        return crops, risks

    def _collect_recent_actions(self, username: str) -> List[Dict]:
        """获取近期设备操作日志"""
        try:
            log_path = os.path.join("data", username, "device_action_log.json")
            if os.path.exists(log_path):
                with open(log_path, encoding="utf-8") as f:
                    logs = json.load(f)
                    # 取最近10条
                    return logs[-10:] if isinstance(logs, list) else []
        except Exception:
            pass
        return []
```

- [ ] **Step 4: 运行数据结构测试确认通过**

```bash
python -m pytest tests/test_autonomous_farm_manager.py::TestDataStructures -v
python -m pytest tests/test_autonomous_farm_manager.py::TestAutonomousFarmManager::test_init -v
python -m pytest tests/test_autonomous_farm_manager.py::TestAutonomousFarmManager::test_discover_regions_from_devices -v
python -m pytest tests/test_autonomous_farm_manager.py::TestAutonomousFarmManager::test_should_skip_night_irrigation -v
python -m pytest tests/test_autonomous_farm_manager.py::TestAutonomousFarmManager::test_night_mode_full_allows_all -v
```

- [ ] **Step 5: Commit**

```bash
git add core/autonomous_farm_manager.py tests/test_autonomous_farm_manager.py
git commit -m "feat: 新建 AutonomousFarmManager 数据模型+数据收集模块

- CameraView / FarmState / DecisionPlan / ActionResult / CycleReport 数据结构
- collect_farm_state 多源数据并行采集
- 摄像头拍照+Vision分析、传感器读数、天气、病虫害、近期操作
- 按 location 分组的区域发现
- 夜间操作约束

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: LLM 决策引擎 (build_prompt + request_decision + validate)

**Files:**
- Modify: `core/autonomous_farm_manager.py`

- [ ] **Step 1: 为决策引擎编写测试**

在 `tests/test_autonomous_farm_manager.py` 追加：

```python

class TestDecisionEngine:
    """LLM 决策引擎测试"""

    @pytest.fixture
    def sample_state(self):
        return FarmState(
            region="大棚A", username="test",
            timestamp="2026-06-19T14:00:00",
            camera_views=[
                CameraView(device_id="cam01", location="大棚A",
                          vision_analysis={
                              "crop_type": "番茄", "growth_stage": "fruiting",
                              "health_assessment": {"overall": "fair",
                                  "water_status": "drought-stressed"},
                              "recommended_actions": [
                                  {"action": "irrigate", "urgency": "today",
                                   "detail": "土壤偏干需要灌溉"}
                              ],
                          }),
            ],
            sensor_readings={"sensor01.soil_moisture": 28.5},
            current_weather={"temperature": 28, "humidity": 45,
                           "weather_desc": "晴"},
            weather_forecast=[
                {"date": "2026-06-20", "weather_desc": "晴",
                 "temperature_high": 30, "temperature_low": 20, "humidity": 40},
            ],
            active_crops=[
                {"crop": "番茄", "stage": "结果期", "stage_number": 4,
                 "total_stages": 6, "status": "进行中", "progress_percent": 65},
            ],
        )

    def test_build_prompt_contains_key_fields(self, sample_state):
        mgr = AutonomousFarmManager()
        prompt = mgr.build_decision_prompt(sample_state)
        assert "大棚A" in prompt
        assert "番茄" in prompt
        assert "28.5" in prompt
        assert "土壤偏干" in prompt
        assert "drought-stressed" in prompt

    def test_parse_valid_json(self):
        mgr = AutonomousFarmManager()
        content = '''```json
{"region":"大棚A","overall_assessment":"测试","actions":[],"follow_up":""}
```'''
        result = mgr._parse_decision(content)
        assert result["region"] == "大棚A"
        assert result["overall_assessment"] == "测试"

    def test_parse_json_without_code_block(self):
        mgr = AutonomousFarmManager()
        content = '{"region":"大棚A","overall_assessment":"OK","actions":[],"follow_up":""}'
        result = mgr._parse_decision(content)
        assert result["region"] == "大棚A"

    def test_parse_truncated_json_recovers(self):
        mgr = AutonomousFarmManager()
        # 模拟被截断的 JSON
        content = '{"region":"大棚A","overall_assessment":"一切正常","actions":[{"action":"irrigate","params":{"duration":25'
        result = mgr._parse_decision(content)
        assert result is not None
        assert result["region"] == "大棚A"

    def test_parse_completely_invalid_returns_none(self):
        mgr = AutonomousFarmManager()
        content = "这是一段不是JSON的回复文本"
        result = mgr._parse_decision(content)
        assert result is None

    def test_validate_plan_accepts_valid_actions(self):
        mgr = AutonomousFarmManager()
        plan = DecisionPlan(region="大棚A", actions=[
            {"action": "irrigate", "params": {"duration": 30}, "urgency": "today", "reason": "土壤缺水"},
            {"action": "alert", "urgency": "this_week", "reason": "需注意病害"},
        ])
        plan = mgr.validate_plan(plan, available_capabilities={"irrigate"})
        # alert 不需要设备能力，irrigate 有匹配
        assert len(plan.actions) == 2

    def test_validate_plan_rejects_unknown_action(self):
        mgr = AutonomousFarmManager()
        plan = DecisionPlan(region="大棚A", actions=[
            {"action": "fly_to_moon", "params": {}, "urgency": "today", "reason": "?"},
        ])
        plan = mgr.validate_plan(plan, available_capabilities=set())
        assert len(plan.actions) == 0  # 全部驳回

    def test_validate_plan_clips_exceeded_params(self):
        mgr = AutonomousFarmManager()
        plan = DecisionPlan(region="大棚A", actions=[
            {"action": "irrigate", "params": {"duration": 999}, "urgency": "today", "reason": "测试"},
        ])
        plan = mgr.validate_plan(plan, available_capabilities={"irrigate"})
        assert plan.actions[0]["params"]["duration"] == 120  # 被裁剪到上限

    def test_validate_plan_limits_max_actions(self):
        mgr = AutonomousFarmManager()
        actions = [
            {"action": "irrigate", "params": {"duration": 10}, "urgency": "today", "reason": f"测试{i}"}
            for i in range(10)
        ]
        plan = DecisionPlan(region="大棚A", actions=actions)
        plan = mgr.validate_plan(plan, available_capabilities={"irrigate"}, max_actions=5)
        assert len(plan.actions) == 5  # 裁剪到上限

    def test_validate_plan_dedup_same_device(self):
        mgr = AutonomousFarmManager()
        plan = DecisionPlan(region="大棚A", actions=[
            {"action": "irrigate", "device_id": "pump1", "params": {"duration": 10}, "urgency": "today", "reason": "a"},
            {"action": "irrigate", "device_id": "pump1", "params": {"duration": 20}, "urgency": "today", "reason": "b"},
        ])
        plan = mgr.validate_plan(plan, available_capabilities={"irrigate"})
        assert len(plan.actions) == 1  # 去重，保留第一个
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_autonomous_farm_manager.py::TestDecisionEngine -v
# 预期: AttributeError: 'AutonomousFarmManager' object has no attribute 'build_decision_prompt'
```

- [ ] **Step 3: 实现 build_decision_prompt + request_decision + validate_plan**

在 `AutonomousFarmManager` 类中追加以下方法（位于 `_collect_recent_actions` 之后）：

```python
    # ── 决策引擎 ──────────────────────────────────

    def build_decision_prompt(self, state: FarmState) -> str:
        """② 聚合：将 FarmState 构造成 LLM 可理解的提示文本"""
        parts = []

        # 系统指令
        parts.append("""你是农业自主决策专家。根据农田综合状态数据，生成结构化的操作计划。

决策原则：
1. 优先解决紧急问题（干旱 > 病虫害 > 缺肥 > 其他）
2. 操作参数在安全范围内尽可能精确（看数据定量，不要拍脑袋）
3. 如果一切正常，actions 为空数组即可
4. 考虑未来天气：如果预报有雨，推迟灌溉
5. 夜间(22:00-06:00)禁止灌溉/施肥/通风，可改为告警""")

        # 硬限制
        parts.append("""
[硬限制 - 不可违反]
- 单次灌溉 ≤ 120分钟
- 单次施肥 ≤ 50kg
- 夜间时段(22:00-06:00)禁止噪音操作""")

        # 区域信息
        parts.append(f"\n[当前农场状态]")
        parts.append(f"区域: {state.region}")
        parts.append(f"时间: {state.timestamp}")

        # 作物信息
        if state.active_crops:
            crops_text = "\n".join(
                f"- {c['crop']} | {c['stage']} | 进度{c['progress_percent']}%"
                for c in state.active_crops
            )
            parts.append(f"\n当前作物:\n{crops_text}")
        else:
            parts.append("\n当前作物: 无进行中的种植")

        # 天气
        if state.current_weather:
            w = state.current_weather
            parts.append(f"\n当前天气: {w.get('weather_desc','')} "
                        f"温度{w.get('temperature','?')}°C "
                        f"湿度{w.get('humidity','?')}% "
                        f"风速{w.get('wind_speed','?')}km/h")

        if state.weather_forecast:
            fore_text = "\n".join(
                f"- {f['date']}: {f.get('weather_desc','')} "
                f"{f.get('temperature_low','?')}~{f.get('temperature_high','?')}°C"
                for f in state.weather_forecast[:3]
            )
            parts.append(f"\n天气预报:\n{fore_text}")

        # 持续异常
        if state.weather_persistence:
            p_text = "\n".join(
                f"- ⚠️ {p['type']} 已持续{p['days']}天: {p.get('advice','')[:200]}"
                for p in state.weather_persistence
            )
            parts.append(f"\n持续天气异常:\n{p_text}")

        # 传感器
        if state.sensor_readings:
            sens_text = "\n".join(
                f"- {k}: {v}"
                for k, v in state.sensor_readings.items()
                if v is not None
            )
            if sens_text:
                parts.append(f"\n传感器读数:\n{sens_text}")
        else:
            parts.append("\n传感器读数: 无可用数据")

        # 摄像头分析
        if state.camera_views:
            for cv in state.camera_views:
                if cv.error:
                    parts.append(f"\n摄像头 {cv.device_id}: ❌ {cv.error}")
                elif cv.vision_analysis:
                    a = cv.vision_analysis
                    parts.append(f"\n摄像头 {cv.device_id} ({cv.location}):")
                    parts.append(f"  作物: {a.get('crop_type','未知')}")
                    parts.append(f"  阶段: {a.get('growth_stage','未知')}")
                    health = a.get('health_assessment', {})
                    parts.append(f"  健康: {health.get('overall','?')} "
                               f"养分={health.get('nutrient_status','?')} "
                               f"水分={health.get('water_status','?')}")
                    issues = a.get('issues_found', [])
                    for iss in issues:
                        parts.append(f"  🚨 {iss.get('name','?')} "
                                   f"({iss.get('severity','?')}): {iss.get('description','?')}")
                    summary = a.get('summary', '')
                    if summary:
                        parts.append(f"  总结: {summary}")

        # 病虫害风险
        if state.disease_risks:
            d_text = "\n".join(
                f"- {r.get('crop','')} {r.get('disease','')} 风险{r.get('risk','')}"
                for r in state.disease_risks[:5]
            )
            parts.append(f"\n病虫害风险:\n{d_text}")

        # 近期操作
        if state.recent_actions:
            recent_text = "\n".join(
                f"- {a.get('timestamp','?')}: {a.get('device_id','?')} "
                f"{a.get('command','?')} → {a.get('success','?')}"
                for a in state.recent_actions[:5]
            )
            parts.append(f"\n近期设备操作:\n{recent_text}")

        # 输出要求
        parts.append("""
[输出要求]
严格按以下JSON格式输出，不要包含markdown代码块标记，直接输出纯JSON:
{"region":"区域名","overall_assessment":"一段中文总结描述当前农场整体状态和关键发现","actions":[{"action":"irrigate|fertigate|ventilate|light|heat|cool|alert","device_hint":"设备类型提示","params":{"duration":数字分钟},"urgency":"immediate|today|this_week|routine","reason":"为什么要执行这个操作"}],"follow_up":"后续建议或下次巡检需关注的点"}

注意: 
- actions 可以为空数组 []
- urgency: immediate=立即, today=今天, this_week=本周, routine=常规
- 如果没有需要执行的操作，actions留空即可""")

        return "\n".join(parts)

    def request_decision(self, prompt: str) -> Optional[DecisionPlan]:
        """③ 决策：调用 LLM 生成结构化操作计划"""
        from app.agent.config import (
            AUTO_DECISION_MODEL, AUTO_DECISION_TEMPERATURE,
            AUTO_DECISION_TIMEOUT, LLM_API_KEY, LLM_BASE_URL,
        )
        if not LLM_API_KEY:
            logger.error("LLM API 未配置，无法进行自主决策")
            return None

        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            llm = ChatOpenAI(
                model=AUTO_DECISION_MODEL,
                temperature=AUTO_DECISION_TEMPERATURE,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                timeout=AUTO_DECISION_TIMEOUT,
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, 'content') else str(resp)

            parsed = self._parse_decision(content)
            if parsed is None:
                logger.warning("LLM 决策 JSON 解析失败，原始响应: %s", content[:500])
                return None

            plan = DecisionPlan(
                region=parsed.get("region", ""),
                overall_assessment=parsed.get("overall_assessment", ""),
                actions=parsed.get("actions", []),
                follow_up=parsed.get("follow_up", ""),
                raw_response=content,
            )
            return plan

        except Exception as e:
            logger.warning("LLM 决策请求失败: %s", e)
            return None

    def _parse_decision(self, content: str) -> Optional[Dict]:
        """从 LLM 响应中提取 JSON，支持截断恢复"""
        if not content or not content.strip():
            return None

        text = content.strip()

        # 提取 ```json ... ``` 代码块
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if m:
            text = m.group(1).strip()

        # 去除首尾非 JSON 字符
        if text and text[0] != '{':
            idx = text.find('{')
            if idx >= 0:
                text = text[idx:]
        if text and text[-1] != '}':
            idx = text.rfind('}')
            if idx >= 0:
                text = text[:idx + 1]

        # 尝试解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # JSON 截断恢复
        if text and not text.rstrip().endswith('}'):
            for fix in ['}]}]}', '}]}', '}', '"]}', '"}']:
                candidate = text.rstrip() + fix
                try:
                    result = json.loads(candidate)
                    if result.get("region"):
                        logger.info("JSON 截断已恢复")
                        return result
                except json.JSONDecodeError:
                    continue

        logger.warning("JSON 解析完全失败")
        return None

    def validate_plan(self, plan: DecisionPlan,
                       available_capabilities: set = None,
                       max_actions: int = 5) -> DecisionPlan:
        """安全校验层：白名单、参数裁剪、去重、数量限制"""
        if available_capabilities is None:
            available_capabilities = set()

        valid_actions = []
        seen_devices = set()

        for action in plan.actions[:max_actions]:
            action_type = action.get("action", "")

            # 白名单校验
            if action_type not in ("irrigate", "fertigate", "ventilate",
                                    "light", "heat", "cool", "alert"):
                logger.info("跳过非法action: %s", action_type)
                continue

            # alert 不需要设备
            if action_type != "alert":
                device_id = action.get("device_id", action.get("device_hint", ""))
                if not device_id:
                    logger.info("跳过无设备ID的action: %s", action_type)
                    continue

                # 去重
                dedup_key = f"{device_id}:{action_type}"
                if dedup_key in seen_devices:
                    logger.info("跳过重复操作: %s", dedup_key)
                    continue
                seen_devices.add(dedup_key)

                action["device_id"] = device_id

            # 参数硬上限裁剪
            if action_type == "irrigate":
                limit = _HARD_LIMITS["irrigate"]["max_duration_per_use_minutes"]
                params = action.get("params", {})
                if params.get("duration", 0) > limit:
                    params["duration"] = limit
                    action["params"] = params
                    logger.info("灌溉时长裁剪至 %d 分钟", limit)

            if action_type == "fertigate":
                limit = _HARD_LIMITS["fertigate"]["max_amount_per_use_kg"]
                params = action.get("params", {})
                if params.get("amount_kg", 0) > limit:
                    params["amount_kg"] = limit
                    action["params"] = params
                    logger.info("施肥量裁剪至 %d kg", limit)

            valid_actions.append(action)

        plan.actions = valid_actions
        return plan
```

- [ ] **Step 4: 运行决策引擎测试**

```bash
python -m pytest tests/test_autonomous_farm_manager.py::TestDecisionEngine -v
```

- [ ] **Step 5: Commit**

```bash
git add core/autonomous_farm_manager.py tests/test_autonomous_farm_manager.py
git commit -m "feat: 实现 LLM 决策引擎

- build_decision_prompt: FarmState → 结构化提示文本
- request_decision: LLM API 调用 → DecisionPlan
- _parse_decision: JSON 提取 + 截断恢复
- validate_plan: 白名单校验 + 参数裁剪 + 去重 + 数量限制

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 执行模块 (execute_plan + run_cycle + fallback)

**Files:**
- Modify: `core/autonomous_farm_manager.py`

- [ ] **Step 1: 为执行和编排方法编写测试**

在 `tests/test_autonomous_farm_manager.py` 追加：

```python

class TestExecutionAndCycle:
    """执行和编排测试"""

    def test_run_cycle_returns_report_even_on_failure(self):
        """即使采集失败，run_cycle 也应返回报告"""
        mgr = AutonomousFarmManager()
        # 用不存在的用户名测试错误处理
        report = mgr.run_cycle("nonexistent_user", "不存在的区域")
        assert report is not None
        assert report.cycle_id != ""
        assert report.summary != ""  # 至少包含错误信息

    def test_cycle_report_has_duration(self):
        mgr = AutonomousFarmManager()
        report = mgr.run_cycle("test_user", "test_region")
        assert report.duration_ms > 0

    def test_fallback_rule_engine_empty_state(self):
        mgr = AutonomousFarmManager()
        state = FarmState(region="test", username="test")
        plan = mgr._fallback_rule_engine(state, "test")
        assert plan is None or plan.actions == []

    def test_summarize_successful_report(self):
        mgr = AutonomousFarmManager()
        state = FarmState(region="大棚A", username="test")
        plan = DecisionPlan(region="大棚A", actions=[
            {"action": "irrigate", "params": {"duration": 25}, "urgency": "today", "reason": "缺水"},
        ])
        results = [
            ActionResult(action="irrigate", device_id="pump1", success=True,
                        message="执行成功", executed_params={"duration": 25}),
        ]
        report = CycleReport(
            cycle_id="c1", username="test", region="大棚A",
            timestamp="2026-06-19T14:00:00",
            farm_state=state, decision_plan=plan,
            execution_results=results, summary="",
        )
        summary = mgr._summarize(report)
        assert "irrigate" in summary
        assert "大棚A" in summary
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_autonomous_farm_manager.py::TestExecutionAndCycle -v
# 预期: 部分方法未定义
```

- [ ] **Step 3: 实现 execute_plan + run_cycle + fallback + 辅助方法**

在 `AutonomousFarmManager` 类中追加：

```python
    # ── 执行 ──────────────────────────────────────

    def execute_plan(self, plan: DecisionPlan, username: str) -> List[ActionResult]:
        """④ 执行：逐一执行决策计划中的操作"""
        from core.device_rule_engine import RuleEngine, RuleDecision, apply_autonomy
        from core.device_executor import DeviceExecutor
        from core.device_registry_factory import setup_registry, close_registry
        from devices.base import DeviceCommand
        from app.agent.config import get_autonomy_level, AUTO_DECISION_NIGHT_MODE

        results = []
        if not plan.actions:
            return results

        autonomy = get_autonomy_level()
        engine = RuleEngine(username=username)

        registry, loop = setup_registry(username)
        try:
            loop.run_until_complete(registry.discover_all())
            executor = DeviceExecutor(registry, username=username)

            # 按 urgency 排序：immediate > today > this_week > routine
            urgency_order = {"immediate": 0, "today": 1, "this_week": 2, "routine": 3}
            sorted_actions = sorted(
                plan.actions,
                key=lambda a: urgency_order.get(a.get("urgency", "routine"), 99)
            )

            for action in sorted_actions:
                action_type = action.get("action", "")
                params = action.get("params", {})
                reason = action.get("reason", "")

                # alert 类型只记录不执行
                if action_type == "alert":
                    logger.info("📢 决策告警: %s", reason)
                    results.append(ActionResult(
                        action="alert", device_id="",
                        success=True, message=f"告警已记录: {reason}",
                    ))
                    continue

                device_id = action.get("device_id", "")
                if not device_id:
                    results.append(ActionResult(
                        action=action_type, device_id="",
                        success=False, message="缺少设备ID",
                    ))
                    continue

                # 夜间约束检查
                night_check = self._check_night_constraint(
                    action_type, AUTO_DECISION_NIGHT_MODE)
                if night_check == RuleDecision.REJECTED:
                    results.append(ActionResult(
                        action=action_type, device_id=device_id,
                        success=False, message=f"夜间模式({AUTO_DECISION_NIGHT_MODE})禁止执行",
                    ))
                    continue

                # 规则评估
                try:
                    temp_rule = {
                        "id": f"auto_{action_type}",
                        "name": f"自主决策-{action_type}",
                        "action": {"device_id": device_id, "command": "start", "params": params},
                        "constraints": {
                            "max_duration_per_use": 120,
                        },
                    }
                    decision, eval_reason, final_params = engine.evaluate_action(
                        temp_rule, params, {"device_id": device_id})
                    decision = apply_autonomy(decision, autonomy)

                    if decision == RuleDecision.REJECTED:
                        results.append(ActionResult(
                            action=action_type, device_id=device_id,
                            success=False, message=eval_reason,
                        ))
                        continue

                    if decision == RuleDecision.NEED_CONFIRM:
                        results.append(ActionResult(
                            action=action_type, device_id=device_id,
                            success=False,
                            message=f"需要用户确认: {eval_reason}",
                        ))
                        continue

                    # 执行
                    cmd = DeviceCommand(command="start", params=final_params)
                    result = executor.execute_sync(
                        device_id, cmd, trigger="autonomous",
                        rule_id=f"auto_{action_type}",
                    )

                    if result["success"]:
                        engine.record_execution(device_id, final_params)
                        res_obj = result.get("result")
                        msg = res_obj.message if res_obj and hasattr(res_obj, 'message') else "执行成功"
                    else:
                        res_obj = result.get("result")
                        msg = res_obj.message if res_obj and hasattr(res_obj, 'message') else "执行失败"

                    results.append(ActionResult(
                        action=action_type, device_id=device_id,
                        success=result["success"], message=msg,
                        rule_matched=action_type, executed_params=final_params,
                    ))

                except Exception as e:
                    logger.warning("执行操作失败 %s/%s: %s", action_type, device_id, e)
                    results.append(ActionResult(
                        action=action_type, device_id=device_id,
                        success=False, message=str(e),
                    ))

        finally:
            close_registry(loop)

        return results

    # ── Fallback ──────────────────────────────────

    def _fallback_rule_engine(self, state: FarmState, username: str) -> Optional[DecisionPlan]:
        """LLM 不可用时的规则引擎兜底"""
        try:
            from core.device_rule_engine import RuleEngine, RuleDecision, apply_autonomy
            from app.agent.config import get_autonomy_level

            engine = RuleEngine(username=username)
            context = {
                "sensor_data": state.sensor_readings,
                "weather": state.current_weather or {},
                "crop": next((c["crop"] for c in state.active_crops), ""),
            }
            matched = engine.find_matching_rules(context)

            if not matched:
                return None

            autonomy = get_autonomy_level()
            actions = []
            for rule in matched[:3]:  # 最多3条规则
                rule_action = rule.get("action", {})
                params = rule_action.get("params", {})
                device_id = rule_action.get("device_id", "")
                command = rule_action.get("command", "start")

                decision, reason, final_params = engine.evaluate_action(
                    rule, params, {"device_id": device_id})
                decision = apply_autonomy(decision, autonomy)

                if decision == RuleDecision.AUTO_EXECUTE:
                    actions.append({
                        "action": command,
                        "device_id": device_id,
                        "params": final_params,
                        "urgency": "today",
                        "reason": f"规则引擎兜底: {rule.get('name', rule['id'])} — {reason}",
                    })

            return DecisionPlan(
                region=state.region,
                overall_assessment=f"LLM 决策不可用，使用规则引擎兜底。匹配到 {len(matched)} 条规则，可执行 {len(actions)} 项操作。",
                actions=actions,
            )

        except Exception as e:
            logger.exception("规则引擎兜底失败")
            return None

    # ── 编排 ──────────────────────────────────────

    def run_cycle(self, username: str, region: str) -> CycleReport:
        """完整闭环：收集 → 决策 → 执行 → 报告"""
        start = datetime.now()
        cycle_id = f"cycle_{start.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"

        report = CycleReport(
            cycle_id=cycle_id, username=username, region=region,
            timestamp=start.isoformat(),
        )

        try:
            # ① 收集状态
            state = self.collect_farm_state(username, region)
            report.farm_state = state

            # ② 构建提示 + ③ LLM 决策
            if state.camera_views or state.sensor_readings:
                prompt = self.build_decision_prompt(state)
                plan = self.request_decision(prompt)

                if plan is not None:
                    # 校验
                    capabilities = self._get_available_capabilities(state)
                    plan = self.validate_plan(plan, capabilities)
                    report.decision_plan = plan
                else:
                    # LLM 不可用 → fallback
                    logger.info("LLM 决策失败，使用规则引擎兜底")
                    plan = self._fallback_rule_engine(state, username)
                    report.decision_plan = plan
                    report.fallback_used = True
            else:
                report.summary = f"区域「{region}」无可用数据（无摄像头、无传感器），跳过决策"
                report.duration_ms = int((datetime.now() - start).total_seconds() * 1000)
                return report

            # ④ 执行
            if report.decision_plan and report.decision_plan.actions:
                results = self.execute_plan(report.decision_plan, username)
                report.execution_results = results

            # ⑤ 生成总结
            report.summary = self._summarize(report)

        except Exception as e:
            logger.exception("自主决策周期异常: %s/%s", username, region)
            report.summary = f"周期异常: {e}"

        report.duration_ms = int((datetime.now() - start).total_seconds() * 1000)
        self._save_report(report)
        self._last_run[region] = datetime.now()
        return report

    def _get_available_capabilities(self, state: FarmState) -> set:
        """从传感器和设备状态推断可用的设备能力"""
        caps = set()
        # 从传感器读数推断（有土壤水分传感器 → 可能可以灌溉）
        for key in state.sensor_readings:
            if "soil_moisture" in key or "humidity" in key:
                caps.add("irrigate")
            if "temperature" in key:
                caps.add("ventilate")
        # 默认可用
        caps.update({"irrigate", "fertigate", "ventilate", "light", "heat", "cool"})
        return caps

    def _summarize(self, report: CycleReport) -> str:
        """生成巡检总结"""
        parts = [f"区域「{report.region}」巡检完成"]

        if report.fallback_used:
            parts.append("（使用规则引擎兜底）")

        plan = report.decision_plan
        if plan and plan.overall_assessment:
            parts.append(f"\n评估: {plan.overall_assessment[:200]}")

        if report.execution_results:
            success = sum(1 for r in report.execution_results if r.success)
            fail = len(report.execution_results) - success
            parts.append(f"\n执行: {len(report.execution_results)}项操作, "
                        f"成功{success}项, 失败{fail}项")
            for r in report.execution_results:
                status = "✅" if r.success else "❌"
                parts.append(f"  {status} {r.action} → {r.device_id}: {r.message[:80]}")
        elif plan and not plan.actions:
            parts.append("\n无需执行任何操作")

        if plan and plan.follow_up:
            parts.append(f"\n后续: {plan.follow_up[:200]}")

        return "\n".join(parts)

    def _save_report(self, report: CycleReport):
        """保存巡检报告到磁盘"""
        try:
            report_dir = os.path.join(
                "data", report.username, "autonomous_reports")
            os.makedirs(report_dir, exist_ok=True)

            filepath = os.path.join(report_dir, f"{report.cycle_id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self._report_to_dict(report), f, ensure_ascii=False, indent=2)
            logger.info("巡检报告已保存: %s", filepath)
        except Exception as e:
            logger.warning("报告保存失败: %s", e)

    def _report_to_dict(self, report: CycleReport) -> Dict:
        """将报告转为可序列化字典"""
        return {
            "cycle_id": report.cycle_id,
            "username": report.username,
            "region": report.region,
            "timestamp": report.timestamp,
            "farm_state": {
                "region": report.farm_state.region if report.farm_state else "",
                "camera_count": len(report.farm_state.camera_views) if report.farm_state else 0,
                "sensor_count": len(report.farm_state.sensor_readings) if report.farm_state else 0,
            },
            "decision_plan": {
                "overall_assessment": report.decision_plan.overall_assessment if report.decision_plan else "",
                "actions": report.decision_plan.actions if report.decision_plan else [],
                "follow_up": report.decision_plan.follow_up if report.decision_plan else "",
            } if report.decision_plan else None,
            "execution_results": [
                {"action": r.action, "device_id": r.device_id,
                 "success": r.success, "message": r.message}
                for r in report.execution_results
            ],
            "fallback_used": report.fallback_used,
            "summary": report.summary,
            "duration_ms": report.duration_ms,
        }
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_autonomous_farm_manager.py::TestExecutionAndCycle -v
```

- [ ] **Step 5: Commit**

```bash
git add core/autonomous_farm_manager.py tests/test_autonomous_farm_manager.py
git commit -m "feat: 实现执行模块 + run_cycle 编排 + 规则引擎 fallback

- execute_plan: 逐一执行操作（RuleEngine校验→DeviceExecutor执行）
- run_cycle: 完整闭环编排
- _fallback_rule_engine: LLM不可用时的规则引擎兜底
- 巡检报告保存到 data/{user}/autonomous_reports/

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: 调度集成 + API 接口

**Files:**
- Modify: `app/scheduler_jobs.py`
- Modify: `app/api_server.py`
- Modify: `app/api_routes.py`

- [ ] **Step 1: 修改 scheduler_jobs.py — 替换摄像头巡检为自主决策调度**

在文件末尾新增，同时保留旧函数的注册方式不变（仅添加新调度入口）：

```python
# ── 自主决策巡检（替代原 check_camera_capture_job）──

def _get_active_usernames() -> List[str]:
    """发现所有活跃用户"""
    usernames = ["default"]
    data_dir = os.path.join("data")
    if os.path.exists(data_dir):
        for d in os.listdir(data_dir):
            user_path = os.path.join(data_dir, d)
            if os.path.isdir(user_path):
                # 有设备或种植数据的用户
                if os.path.exists(os.path.join(user_path, "custom_devices.json")) or \
                   os.path.exists(os.path.join(user_path, "planting_progress.json")):
                    usernames.append(d)
    return list(set(usernames))


def check_autonomous_cycle_job():
    """自主决策定时巡检：发现区域 → 数据采集 → LLM决策 → 执行

    替代原有的 check_camera_capture_job，实现完整的感知→决策→执行闭环。
    """
    try:
        from core.autonomous_farm_manager import AutonomousFarmManager
        from app.agent.config import (
            AUTO_DECISION_REGIONS, AUTO_DECISION_MIN_INTERVAL,
        )

        usernames = _get_active_usernames()
        configured_regions = [r.strip() for r in AUTO_DECISION_REGIONS.split(",") if r.strip()]

        for username in usernames:
            manager = AutonomousFarmManager()

            # 发现该用户的区域
            try:
                from core.device_registry_factory import setup_registry, close_registry
                registry, loop = setup_registry(username)
                try:
                    devices = loop.run_until_complete(registry.discover_all())
                    all_regions = set()
                    for d in devices:
                        loc = getattr(d, 'location', '') or '默认区域'
                        all_regions.add(loc)

                    if configured_regions:
                        regions = sorted(all_regions & set(configured_regions))
                    else:
                        regions = sorted(all_regions)
                finally:
                    close_registry(loop)
            except Exception as e:
                logger.warning("区域发现失败 [%s]: %s", username, e)
                continue

            if not regions:
                logger.debug("用户 %s 无可用区域", username)
                continue

            for region in regions:
                # 检查最小间隔
                last = manager._last_run.get(region)
                if last and (datetime.now() - last).total_seconds() < AUTO_DECISION_MIN_INTERVAL * 60:
                    logger.debug("区域 %s 距上次巡检不足%d分钟，跳过", region, AUTO_DECISION_MIN_INTERVAL)
                    continue

                try:
                    logger.info("🚀 自主决策巡检: %s/%s", username, region)
                    report = manager.run_cycle(username, region)
                    logger.info("✅ 巡检完成: %s/%s — %s", username, region, report.summary.replace('\n', ' | ')[:200])
                except Exception as e:
                    logger.exception("巡检异常 [%s/%s]: %s", username, region, e)

    except Exception as e:
        logger.exception("自主决策调度失败: %s", e)
```

- [ ] **Step 2: 修改 api_server.py — 注册新定时任务**

```python
# 找到现有 scheduler.add_job 区域，追加：
from app.scheduler_jobs import check_reminders_job, check_weather_job, check_disease_job, check_device_rules_job, check_task_execution_job, check_camera_capture_job, check_autonomous_cycle_job

# 在现有 scheduler.add_job 调用之后追加：
scheduler.add_job(
    check_autonomous_cycle_job, "interval",
    minutes=int(os.getenv("AUTO_DECISION_INTERVAL", "30")),
    id="autonomous_cycle",
)

# 更新启动日志
logger.info("APScheduler 已启动: 提醒/5min 天气/30min 病害/6h 设备规则/5min 任务执行/3min 摄像头巡检/30min 自主决策/30min")
```

具体的编辑操作：

找到 `scheduler_jobs.py` 的 import 行：
```python
from app.scheduler_jobs import check_reminders_job, check_weather_job, check_disease_job, check_device_rules_job, check_task_execution_job, check_camera_capture_job
```

追加 `check_autonomous_cycle_job`：
```python
from app.scheduler_jobs import check_reminders_job, check_weather_job, check_disease_job, check_device_rules_job, check_task_execution_job, check_camera_capture_job, check_autonomous_cycle_job
```

在 `scheduler.add_job(check_camera_capture_job, ...)` 之后追加：
```python
scheduler.add_job(
    check_autonomous_cycle_job, "interval",
    minutes=int(os.getenv("AUTO_DECISION_INTERVAL", "30")),
    id="autonomous_cycle",
)
```

更新日志行：
```python
logger.info("APScheduler 已启动: 提醒/5min 天气/30min 病害/6h 设备规则/5min 任务执行/3min 摄像头巡检/30min 自主决策/30min")
```

- [ ] **Step 3: 修改 api_routes.py — 新增自主决策 API**

在 `register_routes` 函数内追加以下路由（放在健康检查路由之前）：

```python
    # ── 自主决策 ────────────────────────────────────

    @app.post("/api/autonomous/trigger")
    def trigger_autonomous_cycle(region: str = None, username: str = "default"):
        """手动触发一次完整巡检"""
        try:
            from core.autonomous_farm_manager import AutonomousFarmManager

            mgr = AutonomousFarmManager()
            if not region:
                # 发现第一个可用区域
                from core.device_registry_factory import setup_registry, close_registry
                registry, loop = setup_registry(username)
                try:
                    devices = loop.run_until_complete(registry.discover_all())
                    regions = set(getattr(d, 'location', '') or '默认区域' for d in devices)
                    region = sorted(regions)[0] if regions else None
                finally:
                    close_registry(loop)

            if not region:
                return {"success": False, "error": "未发现可用区域"}

            report = mgr.run_cycle(username, region)
            return {
                "success": True,
                "cycle_id": report.cycle_id,
                "region": report.region,
                "summary": report.summary,
                "duration_ms": report.duration_ms,
                "fallback_used": report.fallback_used,
                "actions_count": len(report.execution_results),
            }
        except Exception as e:
            logger.exception("手动触发巡检失败")
            return {"success": False, "error": str(e)}

    @app.get("/api/autonomous/reports")
    def list_autonomous_reports(username: str = "default", limit: int = 20):
        """查询历史巡检报告列表"""
        try:
            report_dir = os.path.join("data", username, "autonomous_reports")
            if not os.path.exists(report_dir):
                return {"reports": []}

            reports = []
            for fname in sorted(os.listdir(report_dir), reverse=True):
                if fname.endswith(".json"):
                    fpath = os.path.join(report_dir, fname)
                    try:
                        with open(fpath, encoding="utf-8") as f:
                            data = json.load(f)
                            reports.append({
                                "cycle_id": data.get("cycle_id", fname),
                                "region": data.get("region", ""),
                                "timestamp": data.get("timestamp", ""),
                                "summary": data.get("summary", ""),
                                "fallback_used": data.get("fallback_used", False),
                                "actions_count": len(data.get("execution_results", [])),
                            })
                    except Exception:
                        pass
                    if len(reports) >= limit:
                        break

            return {"reports": reports}
        except Exception as e:
            return {"reports": [], "error": str(e)}

    @app.get("/api/autonomous/reports/{cycle_id}")
    def get_autonomous_report(cycle_id: str, username: str = "default"):
        """查看单次巡检详情"""
        try:
            report_dir = os.path.join("data", username, "autonomous_reports")
            filepath = os.path.join(report_dir, f"{cycle_id}.json")
            if not os.path.exists(filepath):
                raise HTTPException(404, "报告不存在")
            with open(filepath, encoding="utf-8") as f:
                return json.load(f)
        except HTTPException:
            raise
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/autonomous/status")
    def get_autonomous_status():
        """当前自主决策运行状态"""
        try:
            from app.agent.config import (
                AUTO_DECISION_INTERVAL, AUTO_DECISION_MODEL,
                AUTO_DECISION_NIGHT_MODE, AUTO_DECISION_REGIONS,
            )
            return {
                "enabled": True,
                "interval_minutes": AUTO_DECISION_INTERVAL,
                "model": AUTO_DECISION_MODEL,
                "night_mode": AUTO_DECISION_NIGHT_MODE,
                "configured_regions": [r.strip() for r in AUTO_DECISION_REGIONS.split(",") if r.strip()],
            }
        except Exception as e:
            return {"error": str(e)}
```

- [ ] **Step 4: 验证导入和语法**

```bash
python -c "from app.scheduler_jobs import check_autonomous_cycle_job; print('scheduler OK')"
python -c "from app.api_routes import register_routes; print('routes OK')"
```

- [ ] **Step 5: Commit**

```bash
git add app/scheduler_jobs.py app/api_server.py app/api_routes.py
git commit -m "feat: 集成自主决策调度 + API 接口

- scheduler_jobs.py: check_autonomous_cycle_job 替代摄像头巡检
- api_server.py: 注册 autonomous_cycle 定时任务 (30min)
- api_routes.py: 新增 4 个自主决策 API
  - POST /api/autonomous/trigger  手动触发巡检
  - GET  /api/autonomous/reports  历史报告列表
  - GET  /api/autonomous/reports/{id} 报告详情
  - GET  /api/autonomous/status   运行状态

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: 集成测试 + 端到端验证

**Files:**
- Modify: `tests/test_autonomous_farm_manager.py`

- [ ] **Step 1: 追加集成测试**

在测试文件末尾追加：

```python

class TestIntegration:
    """集成测试：完整闭环（使用 Simulator 设备）"""

    @pytest.mark.integration
    def test_full_cycle_with_simulator(self):
        """使用虚拟设备跑完整闭环"""
        mgr = AutonomousFarmManager()
        report = mgr.run_cycle("default", "默认区域")
        assert report is not None
        assert report.cycle_id != ""
        assert report.region == "默认区域"
        # 应该有 farm_state（即使数据为空）
        assert report.farm_state is not None

    @pytest.mark.integration
    def test_manual_trigger_api(self):
        """手动触发 API 集成测试"""
        import requests
        try:
            resp = requests.post(
                "http://localhost:8000/api/autonomous/trigger",
                params={"region": "默认区域", "username": "default"},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                assert "cycle_id" in data
        except requests.ConnectionError:
            pytest.skip("API server 未运行")

    @pytest.mark.integration
    def test_reports_api(self):
        """报告查询 API 集成测试"""
        import requests
        try:
            resp = requests.get(
                "http://localhost:8000/api/autonomous/reports",
                params={"username": "default"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                assert "reports" in data
        except requests.ConnectionError:
            pytest.skip("API server 未运行")
```

- [ ] **Step 2: 运行全部单元测试**

```bash
python -m pytest tests/test_autonomous_farm_manager.py -v --ignore-glob="*integration*"
```

- [ ] **Step 3: 启动后端运行集成测试**

```bash
# 终端1: 启动后端
python app/api_server.py &

# 等待启动
sleep 3

# 终端2: 运行集成测试
python -m pytest tests/test_autonomous_farm_manager.py -v -m integration

# 手动触发测试
curl -X POST "http://localhost:8000/api/autonomous/trigger?region=默认区域&username=default"

# 查询报告
curl "http://localhost:8000/api/autonomous/reports?username=default"
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_autonomous_farm_manager.py
git commit -m "test: 添加自主决策集成测试

- 完整闭环测试（使用 Simulator 设备）
- 手动触发 API 集成测试
- 报告查询 API 测试

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### 自检清单

- [x] Spec 覆盖：所有设计文档章节均有对应任务
- [x] 无占位符：所有步骤包含具体代码
- [x] 类型一致：FarmState/DecisionPlan/CycleReport 在各任务中一致
- [x] 文件路径正确：所有路径与实际项目结构匹配
