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
