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
_NIGHT_START = 22
_NIGHT_END = 6


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
        return None

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
                    return logs[-10:] if isinstance(logs, list) else []
        except Exception:
            pass
        return []
