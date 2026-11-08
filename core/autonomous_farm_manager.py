"""农田自主决策编排器 — 感知→分析→决策→执行 闭环

将摄像头巡检、传感器采集、天气服务、LLM 决策、设备控制整合为
完整的自主决策流程。

按设备绑定的地块(plot_id)分组形成"决策区域"，
每区域使用地块的精确坐标获取天气，避免 location 字符串 geocode。
"""

import os, json, logging, asyncio, base64, re, copy, math, uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from core.storage_paths import DEFAULT_DATA_DIR

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
    sensor_quality: Dict[str, Any] = field(default_factory=dict)
    current_weather: Optional[Dict] = None
    weather_quality: Dict[str, Any] = field(default_factory=dict)
    weather_forecast: List[Dict] = field(default_factory=list)
    weather_persistence: List[Dict] = field(default_factory=list)
    available_devices: List[Dict] = field(default_factory=list)
    active_crops: List[Dict] = field(default_factory=list)
    disease_risks: List[Dict] = field(default_factory=list)
    disease_risk_quality: Dict[str, Any] = field(default_factory=dict)
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
    execution_id: Optional[str] = None
    acknowledged: bool = False


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


# ── 硬限制（代码级，不可通过规则突破）— 从 device_rule_engine 导入权威定义 ──

from core.device_rule_engine import HARD_LIMITS as _GLOBAL_HARD_LIMITS


# ── 主类 ─────────────────────────────────────────────

class AutonomousFarmManager:
    """农田自主决策编排器

    用法:
        mgr = AutonomousFarmManager()
        report = mgr.run_cycle("username", "大棚A区")
        print(report.summary)
    """

    def __init__(self):
        self.hard_limits = copy.deepcopy(_GLOBAL_HARD_LIMITS)

    # ── 区域发现 ──────────────────────────────────

    def _group_by_region(self, devices: List, username: str = "default") -> Dict[str, List]:
        """按设备绑定的地块(plot_id)分组。

        从 custom_devices.json 读取每台设备的 plot_id，
        未绑定地块的设备按 location 分组。
        """
        # 加载设备→地块映射
        from core.device_registry_factory import load_custom_devices
        device_configs = load_custom_devices(username)
        device_to_plot = {}
        for dc in device_configs:
            pid = dc.get("plot_id", "")
            if pid:
                device_to_plot[dc["device_id"]] = pid

        regions: Dict[str, List] = {}
        for d in devices:
            did = d.device_id
            # 优先使用 plot_id 配置
            plot_id = device_to_plot.get(did, "")
            if not plot_id:
                plot_id = getattr(d, 'location', '') or '默认区域'
            regions.setdefault(plot_id, []).append(d)
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

    def collect_farm_state(self, username: str, region: str,
                           plot_lat: float = None, plot_lon: float = None) -> FarmState:
        """① 收集：并行采集一个区域的全部状态数据。

        Args:
            plot_lat, plot_lon: 地块精确坐标，优先用于天气查询
        """
        from core.device_registry_factory import setup_registry, close_registry

        state = FarmState(
            region=region, username=username,
            timestamp=datetime.now().isoformat(),
        )

        state.active_crops, state.disease_risks = self._collect_crop_info(username, region, state.disease_risk_quality)

        # ── 设备数据（async 域内）──
        registry, loop = setup_registry(username)
        try:
            devices = loop.run_until_complete(registry.discover_all())
            region_devices = self._select_region_devices(
                devices, username, region)
            state.available_devices = self._build_available_devices(
                region_devices)

            # 摄像头拍照 + Vision 分析
            cameras = [d for d in region_devices
                      if "capture" in [c.value for c in d.capabilities]
                      and d.status.value in ("online", "offline")]
            state.camera_views = self._capture_and_analyze(
                registry, loop, cameras, username, region, state.active_crops)

            # 传感器读数
            state.sensor_readings = self._collect_sensors(
                registry, loop, region_devices, state.sensor_quality)
        finally:
            close_registry(loop, registry)

        # ── 天气数据（使用地块精确坐标）──
        # 哥哥，调用方给出的坐标不能替代当前用户的地块归属。
        try:
            from core.plot_manager import PlotManager
            plot = PlotManager(username).get_plot(str(region))
        except Exception as exc:
            logger.debug("地块天气坐标读取失败: %s", exc)
            plot = None
        state.current_weather, state.weather_forecast = self._collect_weather(
            region, lat=plot.get("lat") if plot else None, lon=plot.get("lon") if plot else None)
        state.weather_quality = {
            "current": "good" if state.current_weather else "unavailable",
            "forecast": "unavailable", "persistence": "unavailable",
        }
        state.weather_persistence = self._collect_persistence()

        # ── 作物与病害数据（同步）──

        # ── 近期操作（同步）──
        state.recent_actions = self._collect_recent_actions(username)

        return state

    def _select_region_devices(self, devices: List, username: str,
                               region: str) -> List:
        """按地块 ID、地块名称或设备 location 选择巡检设备。

        数据库中的地块运行时 ID 可能是数字，而旧设备配置保存的是地块名称；
        这里同时接受两者，避免设备绑定后反而无法被巡检发现。
        """
        from core.device_registry_factory import load_custom_devices
        from core.plot_manager import PlotManager

        aliases = {str(region)}
        try:
            plot = PlotManager(username).get_plot(str(region))
            if plot:
                aliases.add(str(plot.get("name", "")))
                aliases.add(str(plot.get("plot_id", "")))
        except Exception as e:
            logger.debug("地块别名读取失败 %s/%s: %s", username, region, e)

        configs = {
            item.get("device_id"): item
            for item in load_custom_devices(username)
        }
        selected = []
        for device in devices:
            config = configs.get(device.device_id, {})
            configured_plot = str(config.get("plot_id") or "")
            if configured_plot:
                if configured_plot in aliases:
                    selected.append(device)
                continue
            candidates = {
                str(getattr(device, "location", "") or ""),
                str(config.get("location", "") or ""),
                str(config.get("plot_id", "") or ""),
            }
            if aliases & candidates:
                selected.append(device)
        return selected

    @staticmethod
    def _build_available_devices(devices: List) -> List[Dict]:
        """构造供决策器使用的精确设备清单。"""
        result = []
        for device in devices:
            capabilities = [
                capability.value for capability in device.capabilities
                if capability.value not in ("read_sensor", "capture")
            ]
            if not capabilities:
                continue
            result.append({
                "device_id": device.device_id,
                "name": device.name,
                "capabilities": capabilities,
                "status": device.status.value,
                "location": device.location,
            })
        return result

    def _capture_and_analyze(self, registry, loop, cameras: List,
                              username: str, region: str = "", crops=None) -> List[CameraView]:
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

                raw = result.raw_response or {}
                observed = raw.get("_observed_at") or (raw.get("metadata") or {}).get("timestamp")
                from core.observation_quality import validate_sensor_state
                _, image_issues = validate_sensor_state({"_observed_at": observed, "_quality": raw.get("_quality", "good"), "_error": raw.get("_error")})
                if image_issues:
                    view.error = "图片时间或质量无效: " + "; ".join(image_issues)
                    views.append(view)
                    continue
                image_bytes = raw.get("image_bytes")
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
                        "plot_id": region,
                        "crop": (crops or [{}])[0].get("crop", ""),
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

    def _collect_sensors(self, registry, loop, devices: List, quality=None) -> Dict[str, Any]:
        """哥哥，保留质量和采样时间，只有合格读数进入决策。"""
        from core.observation_quality import validate_sensor_state
        readings = {}
        quality = quality if quality is not None else {}
        for device in devices:
            try:
                raw = loop.run_until_complete(registry.read_state(device.device_id))
                valid, issues = validate_sensor_state(raw)
                quality[device.device_id] = {"state": raw, "issues": issues}
                readings.update({f"{device.device_id}.{key}": value for key, value in valid.items()})
            except Exception as exc:
                quality[device.device_id] = {"issues": [str(exc)]}
        return readings

    def _collect_weather(self, region: str, lat: float = None, lon: float = None) -> Tuple[Optional[Dict], List[Dict]]:
        """哥哥，只用地块坐标的真实、新鲜观测；预报缺少可信生成时间。"""
        try:
            if isinstance(lat, bool) or isinstance(lon, bool):
                return None, []
            lat, lon = float(lat), float(lon)
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
                return None, []
        except (TypeError, ValueError, OverflowError):
            return None, []
        try:
            from core.weather_service import WeatherService
            raw = WeatherService().get_grid_weather(lon, lat)
            if not isinstance(raw, dict) or raw.get("error") or raw.get("_error"):
                return None, []
            if raw.get("_quality", "good") != "good":
                return None, []
            observed_text = raw.get("obsTime")
            observed = datetime.fromisoformat(str(observed_text).replace("Z", "+00:00"))
            if observed.tzinfo is None:
                observed = observed.astimezone()
            if not 0 <= (datetime.now().astimezone() - observed).total_seconds() <= 3600:
                return None, []
            values = {}
            for key, source, bounds in (("temperature", "temp", (-60,100)),
                                         ("humidity", "humidity", (0,100)),
                                         ("precipitation", "precip", (0,1000))):
                value = raw.get(source)
                if value is None or isinstance(value, bool):
                    continue
                try:
                    number = float(value)
                    if math.isfinite(number) and bounds[0] <= number <= bounds[1]:
                        values[key] = number
                except (TypeError, ValueError, OverflowError):
                    continue
            if not values:
                return None, []
            values.update({"_observed_at": observed_text, "_quality": "good"})
            return values, []
        except (TypeError, ValueError, OverflowError) as exc:
            logger.debug("天气观测无效 %s: %s", region, exc)
        except Exception as exc:
            logger.warning("天气观测获取失败 %s: %s", region, exc)
        return None, []

    def _collect_persistence(self) -> List[Dict]:
        """哥哥，共享历史没有地块归属，持续天气暂不评估。"""
        return []

    def _collect_crop_info(self, username: str, region: str = "", quality: Optional[Dict] = None) -> Tuple[List[Dict], List[Dict]]:
        """哥哥，只读取明确绑定本地块的作物卡片和新鲜风险。"""
        crops, risks = [], []
        if quality is not None:
            quality.update({"quality": "unavailable", "reason": "病害风险尚未评估或缓存不可用"})
        try:
            from core.plot_manager import PlotManager
            from core.planting_tracker import PlantingTracker
            plot = PlotManager(username).get_plot(str(region))
            if not plot:
                return crops, risks
            try:
                cards = PlantingTracker(os.path.join(DEFAULT_DATA_DIR, username)).get_progress_cards(limit=None)
            except Exception as exc:
                logger.debug("地块进度读取失败: %s", exc)
                cards = []
            crops = [dict(card, progress_percent=card.get("progress", 0)) for card in cards
                     if card.get("plot_id") is not None and str(card["plot_id"]) == str(region)
                     and card.get("status") == "进行中"]
            if not crops and plot.get("crop"):
                crops = [{"crop": plot["crop"], "stage": "未知", "progress_percent": 0}]
            path = os.path.join(DEFAULT_DATA_DIR, username, "disease_risks", f"{region}.json")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                updated = datetime.fromisoformat(data["updated"].replace("Z", "+00:00"))
                from datetime import timezone
                now = datetime.now(timezone.utc) if updated.tzinfo else datetime.now()
                age = (now - updated).total_seconds()
                if str(data.get("plot_id")) == str(region) and 0 <= age <= 86400 and isinstance(data.get("risks"), list):
                    risk_quality = data.get("quality", "unavailable")
                    if quality is not None:
                        quality.update({"quality": risk_quality if risk_quality == "good" else "unavailable", "updated": data["updated"],
                                        "reason": "" if risk_quality == "good" else "病害风险未评估"})
                    if risk_quality == "good":
                        risks = data["risks"]
        except Exception as exc:
            logger.debug("地块作物或风险读取失败: %s", exc)
        return crops, risks

    def _collect_recent_actions(self, username: str) -> List[Dict]:
        """获取近期设备操作日志（与 DeviceExecutor 共用 device_log.json）"""
        try:
            log_path = os.path.join(DEFAULT_DATA_DIR, username, "device_log.json")
            if os.path.exists(log_path):
                with open(log_path, encoding="utf-8") as f:
                    logs = json.load(f)
                    return logs[-10:] if isinstance(logs, list) else []
        except Exception as e:
            logger.debug("近期操作日志读取失败: %s", e)
        return []

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
        if not state.current_weather:
            parts.append("\n当前天气: 未评估或观测不可用")
        if state.weather_quality.get("forecast") == "unavailable":
            parts.append("天气预报和持续异常: 暂未评估")
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

        # 可执行设备必须给出精确 ID，避免 LLM 生成无法路由的设备别名
        if state.available_devices:
            device_text = "\n".join(
                f"- device_id={d['device_id']} | 名称={d['name']} | "
                f"能力={','.join(d['capabilities'])} | 状态={d['status']}"
                for d in state.available_devices
            )
            parts.append(f"\n可执行设备（操作时只能使用以下 device_id）:\n{device_text}")
        else:
            parts.append("\n可执行设备: 无；只能生成 alert，不能生成设备操作")

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
        if state.disease_risk_quality.get("quality") != "good":
            parts.append("\n病虫害风险: 尚未评估或数据不可用")
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
{"region":"区域名","overall_assessment":"一段中文总结描述当前农场整体状态和关键发现","actions":[{"action":"irrigate|fertigate|ventilate|light|heat|cool|alert","device_id":"必须从可执行设备清单原样选择；alert时为空","params":{"duration":数字分钟},"urgency":"immediate|today|this_week|routine","scheduled_at":"非立即动作的 ISO 8601 时间，未知则留空","id":"操作唯一标识","depends_on":"前置操作id（无则留空）","reason":"为什么要执行这个操作"}],"follow_up":"后续建议或下次巡检需关注的点"}

注意:
- actions 可以为空数组 []
- 不得编造 device_id，不得用设备名称或设备类型代替 device_id
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
                logger.warning("LLM 决策 JSON 解析失败，响应长度: %d", len(content))
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
            for fix in ['}]}]}', '}}]}', '}]}', '}}', '}', '"]}', '"}']:
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
                       available_devices: List[Dict] = None,
                       max_actions: int = 5) -> DecisionPlan:
        """安全校验层：白名单、参数裁剪、去重、数量限制"""
        if available_capabilities is None:
            available_capabilities = set()
        device_map = {
            device.get("device_id"): device
            for device in (available_devices or [])
            if device.get("device_id")
        }

        valid_actions = []
        seen_devices = set()
        temperature_action = None

        for action in plan.actions[:max_actions]:
            action_type = action.get("action", "")

            # 白名单校验
            if action_type not in ("irrigate", "fertigate", "ventilate",
                                    "light", "heat", "cool", "alert"):
                logger.info("跳过非法action: %s", action_type)
                continue

            if action_type in ("heat", "cool"):
                if temperature_action and temperature_action != action_type:
                    continue

            # alert 不需要设备
            if action_type != "alert":
                device_id = action.get("device_id", "")
                if not device_id:
                    logger.info("跳过无设备ID的action: %s", action_type)
                    continue
                if available_devices is not None:
                    device = device_map.get(device_id)
                    if device is None:
                        logger.warning("跳过未注册设备操作: %s → %s",
                                       action_type, device_id)
                        continue
                    if action_type not in device.get("capabilities", []):
                        logger.warning("跳过能力不匹配操作: %s 不支持 %s",
                                       device_id, action_type)
                        continue
                    if device.get("status") != "online":
                        logger.warning("跳过非在线设备操作: %s (%s)",
                                       device_id, device.get("status"))
                        continue
                elif available_capabilities and action_type not in available_capabilities:
                    logger.warning("跳过不可用能力: %s", action_type)
                    continue

                # 去重
                dedup_key = f"{device_id}:{action_type}"
                if dedup_key in seen_devices:
                    logger.info("跳过重复操作: %s", dedup_key)
                    continue
                seen_devices.add(dedup_key)

                action["device_id"] = device_id

            # 参数硬上限裁剪（使用 HARD_LIMITS 权威定义）
            caps = self.hard_limits.get(action_type, {})
            if action_type == "irrigate":
                limit = caps.get("max_duration_per_use_minutes", 120)
                params = action.get("params", {})
                if params.get("duration", 0) > limit:
                    params["duration"] = limit
                    action["params"] = params
                    logger.info("灌溉时长裁剪至 %d 分钟", limit)

            if action_type == "fertigate":
                limit = caps.get("max_amount_per_use_kg", 50)
                params = action.get("params", {})
                if params.get("amount_kg", 0) > limit:
                    params["amount_kg"] = limit
                    action["params"] = params
                    logger.info("施肥量裁剪至 %d kg", limit)

            if action_type == "ventilate":
                limit = caps.get("max_duration_per_use_minutes", 120)
                params = action.get("params", {})
                if params.get("duration", 0) > limit:
                    params["duration"] = limit
                    action["params"] = params
                    logger.info("通风时长裁剪至 %d 分钟", limit)

            if action_type in ("heat", "cool"):
                temperature_action = action_type
            valid_actions.append(action)

        plan.actions = valid_actions
        return plan

    # ── 执行 ──────────────────────────────────────

    def execute_plan(self, plan: DecisionPlan, username: str,
                     policy_context: Dict[str, Any] = None) -> List[ActionResult]:
        """④ 执行：逐一执行决策计划中的操作"""
        from core.device_rule_engine import RuleEngine, RuleDecision, apply_autonomy
        from core.device_executor import DeviceExecutor
        from core.device_registry_factory import setup_registry, close_registry
        from devices.base import DeviceCommand
        from app.agent.config import get_autonomy_level, AUTO_DECISION_NIGHT_MODE

        results = []
        from core.autonomy_runtime import schedule_action, queue_effect_check, raise_alert
        if plan.follow_up:
            schedule_action(username, plan.region, {"action": "review", "reason": plan.follow_up}, due_at=None)
        if not plan.actions:
            return results

        autonomy = get_autonomy_level(username)
        engine = RuleEngine(username=username)
        if policy_context is None:
            policy_context = getattr(self, "_execution_policy_context", {})
        policy_context = dict(policy_context or {})

        registry, loop = setup_registry(username)
        try:
            discovered_devices = loop.run_until_complete(registry.discover_all())
            executor = DeviceExecutor(registry, username=username)
            runtime_devices = {
                device.device_id: device for device in discovered_devices
            }

            # 按 urgency 排序：immediate > today > this_week > routine
            urgency_order = {"immediate": 0, "today": 1, "this_week": 2, "routine": 3}
            # 哥哥，原计划顺序决定依赖，不能按紧急度重排。
            sorted_actions = plan.actions
            completed = {}
            previous_actions = {}
            previous_id = None
            previous_action = None
            for action_item in sorted_actions:
                if previous_id is not None:
                    completed[str(previous_id)] = results[-1]
                    previous_actions[str(previous_id)] = previous_action
                previous_id = action_item.get("id")
                previous_action = action_item
                if action_item.get("action") != "alert" and not action_item.get("execution_id"):
                    action_item["execution_id"] = uuid.uuid4().hex
                dependency = action_item.get("depends_on")
                dependencies = dependency if isinstance(dependency, list) else ([dependency] if dependency is not None else [])
                if dependencies:
                    # 哥哥，启动 ACK 不是作业完成；持久等待运行状态确认。
                    runs, durations = [], []
                    ready = True
                    for dep in dependencies:
                        key = str(dep)
                        outcome = completed.get(key)
                        prior = previous_actions.get(key) or {}
                        execution_id = outcome.execution_id if outcome else None
                        if not outcome or not outcome.success or not outcome.acknowledged or not execution_id or not prior.get("device_id"):
                            ready = False
                        else:
                            runs.append({"device_id": prior["device_id"], "execution_id": execution_id})
                        try:
                            duration = float((outcome.executed_params if outcome else {}).get("duration", 0))
                            if not math.isfinite(duration) or duration <= 0:
                                ready = False
                            else:
                                durations.append(duration)
                        except (TypeError, ValueError, OverflowError):
                            ready = False
                    delayed = dict(action_item, dependency_runs=runs)
                    due_at = datetime.now() + timedelta(minutes=max(durations)) if ready and durations else None
                    scheduled_at = action_item.get("scheduled_at")
                    if action_item.get("urgency", "routine") != "immediate" and not scheduled_at:
                        due_at = None
                    elif scheduled_at and due_at:
                        try:
                            requested = datetime.fromisoformat(str(scheduled_at).replace("Z", "+00:00"))
                            if requested.tzinfo is not None:
                                requested = requested.astimezone().replace(tzinfo=None)
                            due_at = max(due_at, requested)
                        except (TypeError, ValueError, OverflowError):
                            due_at = None
                    schedule_action(username, plan.region, delayed, due_at=due_at)
                    results.append(ActionResult(action=action_item.get("action", ""), device_id=action_item.get("device_id", ""),
                                                success=False, message="等待依赖作业实际完成" if due_at else "依赖状态不明确，已持久保存待安排"))
                    continue
                action_type = action_item.get("action", "")
                params = action_item.get("params", {})
                reason = action_item.get("reason", "")
                if action_type != "alert" and action_item.get("urgency", "routine") != "immediate":
                    due_at = action_item.get("scheduled_at")
                    try:
                        if due_at:
                            parsed = datetime.fromisoformat(str(due_at).replace("Z", "+00:00"))
                            due_at = parsed
                    except (TypeError, ValueError):
                        due_at = None
                    schedule_action(username, plan.region, action_item, due_at=due_at)
                    results.append(ActionResult(action=action_type, device_id=action_item.get("device_id", ""),
                                                success=False, message="已排期" if due_at else "已持久保存，待安排时间"))
                    continue
                rule_id = action_item.get("rule_id")
                source_rule = engine.get_rule(rule_id) if rule_id is not None else None
                if rule_id is not None and (
                    not source_rule or not source_rule.get("enabled", True)
                    or source_rule.get("action", {}).get("device_id") != action_item.get("device_id")
                ):
                    results.append(ActionResult(
                        action=action_type, device_id=action_item.get("device_id", ""),
                        success=False, message="原规则已删除、停用或变更目标设备",
                    ))
                    continue
                source_action = source_rule.get("action", {}) if source_rule else {}
                command = source_action.get("command", "start")
                execution_mode = source_rule.get("execution_mode", "auto") if source_rule else "auto"
                if source_rule:
                    params = source_action.get("params", {})
                if execution_mode == "notify":
                    reason = f"规则提醒: {source_rule.get('name', rule_id)} — {reason}"
                    action_type = "alert"

                # alert 类型只记录不执行
                if action_type == "alert":
                    raise_alert(username, plan.region, reason)
                    results.append(ActionResult(
                        action="alert", device_id="",
                        success=True, message=f"告警已进入待送达队列: {reason}",
                    ))
                    continue

                device_id = action_item.get("device_id", "")
                if not device_id:
                    results.append(ActionResult(
                        action=action_type, device_id="",
                        success=False, message="缺少设备ID",
                    ))
                    continue

                # 执行前再次核验真实设备和能力，防止使用过期或伪造的决策结果
                runtime_device = runtime_devices.get(device_id)
                if runtime_device is None:
                    results.append(ActionResult(
                        action=action_type, device_id=device_id,
                        success=False, message="设备不存在或当前驱动未加载",
                    ))
                    continue
                runtime_capabilities = {
                    capability.value for capability in runtime_device.capabilities
                }
                if action_type not in runtime_capabilities:
                    results.append(ActionResult(
                        action=action_type, device_id=device_id,
                        success=False, message=f"设备不支持操作 {action_type}",
                    ))
                    continue
                if runtime_device.status.value != "online":
                    results.append(ActionResult(
                        action=action_type, device_id=device_id,
                        success=False,
                        message=f"设备当前状态不可执行: {runtime_device.status.value}",
                    ))
                    continue

                # 夜间约束检查
                night_check = self._check_night_constraint(
                    action_type, AUTO_DECISION_NIGHT_MODE)
                if command == "start" and night_check == RuleDecision.REJECTED:
                    results.append(ActionResult(
                        action=action_type, device_id=device_id,
                        success=False, message=f"夜间模式({AUTO_DECISION_NIGHT_MODE})禁止执行",
                    ))
                    continue

                # 规则评估
                try:
                    temp_rule = source_rule or {
                        "id": f"auto_{action_type}",
                        "name": f"自主决策-{action_type}",
                        "action": {
                            "device_id": device_id,
                            "capability": action_type,
                            "command": command,
                            "params": params,
                        },
                        "constraints": {},
                    }
                    action_context = engine.context_for_device(device_id, policy_context)
                    decision, eval_reason, final_params = engine.evaluate_action(
                        temp_rule, params, action_context)
                    decision = apply_autonomy(decision, autonomy)
                    if execution_mode == "confirm" and decision != RuleDecision.REJECTED:
                        decision = RuleDecision.NEED_CONFIRM
                        eval_reason = f"规则「{source_rule.get('name', rule_id)}」设置为执行前确认"
                    execution_rule_id = rule_id if rule_id is not None else f"auto_{action_type}"
                    cmd = DeviceCommand(command=command, params=final_params, execution_id=action_item.get("execution_id"))

                    if decision == RuleDecision.REJECTED:
                        executor.record_decision(
                            device_id, cmd, decision, eval_reason,
                            trigger="autonomous", rule_id=execution_rule_id,
                            capability=action_type,
                            policy_context=action_context,
                        )
                        results.append(ActionResult(
                            action=action_type, device_id=device_id,
                            success=False, message=eval_reason,
                        ))
                        continue

                    if decision == RuleDecision.NEED_CONFIRM:
                        pending = executor.record_decision(
                            device_id, cmd, decision, eval_reason,
                            trigger="autonomous", rule_id=execution_rule_id,
                            add_pending=True, capability=action_type,
                            policy_context=action_context,
                        )
                        pending_note = (
                            f"，待确认ID={pending['pending_id']}"
                            if pending.get("pending_id") else ""
                        )
                        results.append(ActionResult(
                            action=action_type, device_id=device_id,
                            success=False,
                            message=f"需要用户确认: {eval_reason}{pending_note}",
                        ))
                        continue

                    # 执行
                    result = executor.execute_sync(
                        device_id, cmd, trigger="autonomous",
                        rule_id=execution_rule_id,
                        loop=loop, capability=action_type,
                        policy_context=action_context,
                    )

                    if result["success"]:
                        if command == "start":
                            engine.record_execution(device_id, final_params)
                        res_obj = result.get("result")
                        msg = res_obj.message if res_obj and hasattr(res_obj, 'message') else "执行成功"
                    else:
                        res_obj = result.get("result")
                        msg = res_obj.message if res_obj and hasattr(res_obj, 'message') else "执行失败"

                    action_result = ActionResult(
                        action=action_type, device_id=device_id,
                        success=result["success"], message=msg,
                        rule_matched=action_type, executed_params=final_params,
                        execution_id=getattr(res_obj, "execution_id", None) or getattr(cmd, "execution_id", None),
                        acknowledged=bool(getattr(res_obj, "acknowledged", False)),
                    )
                    results.append(action_result)
                    if result["success"] and command == "start":
                        queue_effect_check(username, plan.region, action_item, action_result,
                                           getattr(self, "_cycle_baseline", None))

                except Exception as e:
                    logger.warning("执行操作失败 %s/%s: %s", action_type, device_id, e)
                    results.append(ActionResult(
                        action=action_type, device_id=device_id,
                        success=False, message=str(e),
                    ))

        finally:
            close_registry(loop, registry)

        return results

    # ── Fallback ──────────────────────────────────

    @staticmethod
    def _build_rule_context(state: FarmState, username: str) -> Dict:
        """哥哥，保留每个读数的设备归属，让规则按地块选择传感器。"""
        from core.device_registry_factory import load_custom_devices

        device_states = {}
        sensor_data = {}
        for key, value in state.sensor_readings.items():
            source_id, _, field_name = key.rpartition(".")
            device_states.setdefault(source_id, {})[field_name or key] = value
            sensor_data[key] = value
            sensor_data.setdefault(field_name or key, value)
        return {
            "device_configs": load_custom_devices(username),
            "device_states": device_states,
            "sensor_data": sensor_data,
            "weather": state.current_weather or {},
            "crop": next((crop["crop"] for crop in state.active_crops), ""),
        }

    def _fallback_rule_engine(self, state: FarmState, username: str) -> Optional[DecisionPlan]:
        """LLM 不可用时的规则引擎兜底"""
        try:
            from core.device_rule_engine import RuleEngine

            engine = RuleEngine(username=username)
            context = self._build_rule_context(state, username)
            matched = engine.find_matching_rules(context)

            if not matched:
                return None

            actions = []
            for rule in matched[:3]:
                rule_action = rule.get("action", {})
                params = rule_action.get("params", {})
                device_id = rule_action.get("device_id", "")
                capability = engine._infer_capability(rule_action)

                # 哥哥，原规则由执行端重新读取并评估一次，避免丢指令或重复 AI 微调。
                actions.append({
                    "action": "alert" if rule.get("execution_mode") == "notify" else capability,
                    "device_id": device_id,
                    "rule_id": rule["id"],
                    "params": params,
                    "urgency": "immediate",
                    "reason": f"规则引擎兜底: {rule.get('name', rule['id'])}",
                })

            if actions:
                return DecisionPlan(
                    region=state.region,
                    overall_assessment=f"LLM 决策不可用，使用规则引擎兜底。匹配到 {len(matched)} 条规则，可执行 {len(actions)} 项操作。",
                    actions=actions,
                )
            return None

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
            # ① 解析地块坐标
            from core.plot_manager import PlotManager
            pm = PlotManager(username)
            plot = pm.get_plot(region)
            plot_lat = plot["lat"] if plot else None
            plot_lon = plot["lon"] if plot else None

            # ② 收集状态（传入地块坐标）
            state = self.collect_farm_state(username, region,
                                            plot_lat=plot_lat, plot_lon=plot_lon)
            report.farm_state = state

            # ② 构建提示 + ③ LLM 决策
            if (state.camera_views or state.sensor_readings
                    or state.current_weather or state.weather_forecast):
                prompt = self.build_decision_prompt(state)
                plan = self.request_decision(prompt)

                if plan is not None:
                    # 校验
                    capabilities = self._get_available_capabilities(state)
                    from app.agent.config import AUTO_DECISION_MAX_ACTIONS
                    plan = self.validate_plan(
                        plan,
                        available_capabilities=capabilities,
                        available_devices=state.available_devices,
                        max_actions=AUTO_DECISION_MAX_ACTIONS,
                    )
                    report.decision_plan = plan
                else:
                    # LLM 不可用 → fallback
                    logger.info("LLM 决策失败，使用规则引擎兜底")
                    plan = self._fallback_rule_engine(state, username)
                    if plan is not None:
                        from app.agent.config import AUTO_DECISION_MAX_ACTIONS
                        report.decision_plan = self.validate_plan(
                            plan,
                            available_capabilities=self._get_available_capabilities(state),
                            available_devices=state.available_devices,
                            max_actions=AUTO_DECISION_MAX_ACTIONS,
                        )
                        report.fallback_used = True
                    else:
                        report.summary = f"区域「{region}」LLM 和规则引擎均不可用，无法做出决策"
                        return report
            else:
                report.summary = f"区域「{region}」无可用数据，跳过决策"
                return report

            # ④ 执行
            if report.decision_plan and (report.decision_plan.actions or report.decision_plan.follow_up):
                self._cycle_baseline = state
                self._execution_policy_context = {
                    **self._build_rule_context(state, username),
                    "plot_id": region,
                }
                try:
                    # 保留原有二参数调用契约，便于旧扩展和测试继续替换。
                    results = self.execute_plan(report.decision_plan, username)
                finally:
                    self._cycle_baseline = state
                self._execution_policy_context = {}
                report.execution_results = results

            # ⑤ 生成总结
            report.summary = self._summarize(report)

        except Exception as e:
            logger.exception("自主决策周期异常: %s/%s", username, region)
            report.summary = f"周期异常: {e}"
        finally:
            # 哥哥，提前结束的周期也要记录耗时并保存报告。
            report.duration_ms = int((datetime.now() - start).total_seconds() * 1000)
            self._save_report(report)

        return report

    def _get_available_capabilities(self, state: FarmState) -> set:
        """从传感器和设备状态推断可用的设备能力"""
        caps = set()
        for key in state.sensor_readings:
            if "soil_moisture" in key or "humidity" in key:
                caps.add("irrigate")
            if "temperature" in key:
                caps.add("ventilate")
            if "light" in key or "lux" in key:
                caps.add("light")
        # 如果传感器数据为空，默认开放基础能力
        if not caps:
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
                DEFAULT_DATA_DIR, report.username, "autonomous_reports")
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
                "sensor_readings": report.farm_state.sensor_readings if report.farm_state else {},
                "sensor_quality": report.farm_state.sensor_quality if report.farm_state else {},
                "disease_risk_quality": report.farm_state.disease_risk_quality if report.farm_state else {},
                "weather_quality": report.farm_state.weather_quality if report.farm_state else {},
                "current_weather": report.farm_state.current_weather if report.farm_state else None,
            },
            "decision_plan": {
                "overall_assessment": report.decision_plan.overall_assessment if report.decision_plan else "",
                "actions": report.decision_plan.actions if report.decision_plan else [],
                "follow_up": report.decision_plan.follow_up if report.decision_plan else "",
            } if report.decision_plan else None,
            "execution_results": [
                {"action": r.action, "device_id": r.device_id,
                 "success": r.success, "message": r.message, "execution_id": r.execution_id, "acknowledged": r.acknowledged}
                for r in report.execution_results
            ],
            "fallback_used": report.fallback_used,
            "summary": report.summary,
            "duration_ms": report.duration_ms,
        }
