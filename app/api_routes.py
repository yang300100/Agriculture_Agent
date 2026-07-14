"""FastAPI 路由 — 全部业务逻辑的 HTTP 接口"""

import os, json, sys, base64, asyncio, logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.device_registry_factory import setup_registry, load_custom_devices, save_custom_devices, close_registry, DEFAULT_DATA_DIR, RegistrySession, invalidate_registry_cache


def _user_dir(username: str = "default") -> str:
    """返回用户专属数据目录"""
    path = os.path.join("data", username)
    os.makedirs(path, exist_ok=True)
    return path


def _storage_dir(username: str = "default") -> str:
    u = username or "default"
    return _user_dir(u)


# 服务端天气缓存（减少 API 调用）
_weather_cache = {}
WEATHER_CACHE_SECONDS = 1800


def _cached_weather(location: str):
    now = datetime.now()
    if location in _weather_cache:
        data, ts = _weather_cache[location]
        if (now - ts).total_seconds() < WEATHER_CACHE_SECONDS:
            return data
    return None


def _set_weather_cache(location: str, data):
    _weather_cache[location] = (data, datetime.now())


# ── 请求模型 ───────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: List[Dict] = []
    user_question: str = ""
    user_profile: Dict = {}
    image_data: Optional[str] = None
    image_mime_type: Optional[str] = None
    username: str = "default"

class ProgressData(BaseModel):
    crop: str
    stage: str = ""
    stage_number: int = 1
    total_stages: int = 5
    start_date: str = ""
    expected_end_date: str = ""
    progress_percent: int = 0
    status: str = "进行中"
    notes: str = ""

class TaskData(BaseModel):
    crop: str
    task_type: str = "其他"
    title: str = ""
    description: str = ""
    status: str = "待办"
    priority: str = "medium"
    end_date: str = ""
    progress_percent: int = 0
    # 设备控制字段（可选）
    device_id: Optional[str] = None
    device_command: Optional[str] = None
    device_params: Optional[Dict[str, Any]] = None

class FieldData(BaseModel):
    name: str
    coordinates: List[List[float]] = []
    soil_type: str = ""
    current_crop: str = ""

class CostData(BaseModel):
    crop: str
    cost_type: str = "其他"
    item_name: str = ""
    quantity: float = 1
    unit: str = "项"
    unit_price: float = 0

class IncomeData(BaseModel):
    crop: str
    quantity: float = 1
    unit_price: float = 0

class ProfileData(BaseModel):
    user_region: str = ""
    user_soil_type: str = ""
    user_farm_size: float = 1.0
    user_experience: str = ""
    user_goals: List[str] = []
    user_phone: str = ""

class ReminderData(BaseModel):
    crop: str
    reminder_type: str = "其他"
    task_description: str = ""
    growth_stage: str = ""
    start_date: str = ""
    frequency: str = "单次"
    time_of_day: str = "09:00"

class PlanRequest(BaseModel):
    region: str = ""
    soil_type: str = ""
    farm_size: float = 1.0
    goals: List[str] = []
    experience: str = ""
    crop: str = ""


def register_routes(app: FastAPI):

    # ── 对话 ──────────────────────────────────────────

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        from app.agent.state import AgentState
        from app.agent.graph import build_agricultural_policy_agent
        from knowledge.simple_agriculture_rag import SimpleAgricultureRAG
        from knowledge.faiss_agriculture_rag import FAISSAgricultureRAG

        rag = SimpleAgricultureRAG()
        faiss = FAISSAgricultureRAG() if FAISSAgricultureRAG().is_available else None
        agent = build_agricultural_policy_agent(rag, faiss)

        state = AgentState(
            messages=[],
            user_profile=req.user_profile,
            username=req.username,
        )
        from langchain_core.messages import HumanMessage
        content = req.user_question or "请分析这张农作物图片"
        state.messages.append(HumanMessage(content=content))

        if req.image_data:
            state.image_data = req.image_data
            state.image_mime_type = req.image_mime_type
            state.has_image = True

        try:
            result = agent.invoke(state)
            if isinstance(result, dict):
                state = AgentState(**result)
            else:
                state = result
            answer = str(state.final_answer or "")
        except Exception as e:
            logger.exception("Agent 调用失败")
            answer = f"抱歉，处理您的问题时出现错误：{e}"

        logger.info("API 返回: answer_len=%d\n━━━ 最终回答 ━━━\n%s\n━━━━━━━━━━━━",
                    len(answer),
                    answer[:3000] if len(answer) > 3000 else answer)

        # 安全序列化
        facts = {}
        for k, v in state.short_term_facts.items():
            try:
                facts[k] = v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v)
            except Exception:
                facts[k] = str(v)

        return {"final_answer": answer, "short_term_facts": facts}

    # ── 仪表盘 ────────────────────────────────────────

    @app.get("/api/dashboard")
    def dashboard(username: str = "default"):
        from core.planting_tracker import PlantingTracker
        from core.finance_manager import FinanceManager
        from core.weather_alerts import check_weather_alert_for_region
        from core.lunar_calendar import get_lunar_today

        sd = _storage_dir(username)
        tracker = PlantingTracker(sd)
        progresses = [{
            "id": p.id, "crop": p.crop, "stage": p.stage,
            "stage_number": p.stage_number, "total_stages": p.total_stages,
            "progress_percent": p.progress_percent, "status": p.status,
        } for p in tracker.get_progress()]

        tasks = tracker.get_tasks()
        active = [t for t in tasks if t.status in ("待办", "进行中")]
        overdue = [t for t in tasks if t.status == "已逾期"]

        fm = FinanceManager(sd)
        month = datetime.now().strftime("%Y-%m")
        costs = fm.storage.load_costs()
        income = fm.storage.load_income()
        month_cost = sum(c.get("total_amount", 0) for c in costs if c.get("date", "").startswith(month))
        month_income = sum(i.get("total_amount", 0) for i in income if i.get("date", "").startswith(month))

        alerts = None
        try:
            alerts = check_weather_alert_for_region("北京")
        except Exception:
            pass

        lunar = get_lunar_today()

        return {
            "progress": progresses,
            "tasks": {"active": [{"title": t.title, "crop": t.crop, "priority": t.priority, "status": t.status} for t in active],
                      "overdue": [{"title": t.title, "crop": t.crop} for t in overdue]},
            "finance": {"month_income": month_income, "month_cost": month_cost, "profit": month_income - month_cost},
            "weather_alerts": alerts,
            "lunar": lunar,
        }

    # ── 种植进度 ──────────────────────────────────────

    @app.get("/api/progress")
    def get_progress(username: str = "default"):
        from core.planting_tracker import PlantingTracker
        tracker = PlantingTracker(_storage_dir(username))
        cards = tracker.get_progress_cards()
        return cards

    @app.post("/api/progress")
    def create_progress(data: ProgressData, username: str = "default"):
        from core.planting_tracker import PlantingTracker
        tracker = PlantingTracker(_storage_dir(username))
        tracker.create_progress({
            "crop": data.crop, "stage": data.stage or "准备期",
            "stage_number": data.stage_number, "total_stages": data.total_stages,
            "start_date": data.start_date or datetime.now().strftime("%Y-%m-%d"),
            "expected_end_date": data.expected_end_date,
            "progress_percent": data.progress_percent, "status": data.status,
            "tasks": [], "notes": data.notes,
        })
        return {"success": True}

    @app.post("/api/progress/{pid}/advance")
    def advance_progress(pid: str, username: str = "default"):
        from core.planting_tracker import PlantingTracker
        tracker = PlantingTracker(_storage_dir(username))
        result = tracker.advance_to_next_stage(pid)
        return result

    @app.delete("/api/progress/{pid}")
    def delete_progress(pid: str, username: str = "default"):
        from core.planting_tracker import PlantingTracker
        PlantingTracker(_storage_dir(username)).delete_progress(pid)
        return {"success": True}

    # ── 农事任务 ──────────────────────────────────────

    @app.get("/api/tasks")
    def get_tasks(username: str = "default"):
        from core.planting_tracker import PlantingTracker
        cards = PlantingTracker(_storage_dir(username)).get_task_cards()
        return cards

    @app.post("/api/tasks")
    def create_task(data: TaskData, username: str = "default"):
        from core.planting_tracker import PlantingTracker
        PlantingTracker(_storage_dir(username)).create_task({
            "crop": data.crop, "task_type": data.task_type,
            "title": data.title, "description": data.description,
            "status": data.status, "priority": data.priority,
            "end_date": data.end_date or datetime.now().strftime("%Y-%m-%d"),
            "progress_percent": data.progress_percent,
            "device_id": data.device_id,
            "device_command": data.device_command,
            "device_params": data.device_params,
        })
        return {"success": True}

    @app.post("/api/tasks/{tid}/complete")
    def complete_task(tid: str, username: str = "default"):
        from core.planting_tracker import PlantingTracker
        PlantingTracker(_storage_dir(username)).update_task_status(tid, "已完成", 100)
        return {"success": True}

    @app.post("/api/tasks/{tid}/execute")
    def execute_task(tid: str, username: str = "default"):
        """执行任务关联的设备操作 — 对话→任务→硬件 联动核心"""
        from core.planting_tracker import PlantingTracker
        from devices.base import DeviceCommand
        from core.device_executor import DeviceExecutor
        from core.device_registry_factory import setup_registry, close_registry

        tracker = PlantingTracker(_storage_dir(username))
        task = tracker.get_task_by_id(tid)

        if not task:
            return {"success": False, "error": "任务不存在"}
        if not task.device_id or not task.device_command:
            return {"success": False, "error": "该任务没有关联的设备操作"}
        if task.status == "已完成":
            return {"success": False, "error": "任务已完成，无需再次执行"}

        # 更新状态为"进行中"
        tracker.update_task_status(tid, "进行中", progress=10)

        try:
            with RegistrySession(username) as (registry, loop):
                loop.run_until_complete(registry.discover_all())
                cmd = DeviceCommand(
                    command=task.device_command,
                    params=task.device_params or {},
                )
                executor = DeviceExecutor(registry, username=username)
                result = executor.execute_sync(
                    task.device_id, cmd,
                    trigger="task", rule_id=None, loop=loop,
                )

                if result["success"]:
                    tracker.update_task_status(tid, "已完成", progress=100)
                    msg = result.get("result")
                    msg_text = msg.message if msg and hasattr(msg, 'message') else "执行成功"
                    return {
                        "success": True,
                        "message": f"任务执行成功: {msg_text}",
                        "device_id": task.device_id,
                        "task_id": tid,
                    }
                else:
                    tracker.update_task_status(tid, "待办", progress=0)
                    msg = result.get("result")
                    err_text = msg.message if msg and hasattr(msg, 'message') else str(msg or "未知错误")
                    return {
                        "success": False,
                        "error": f"设备执行失败: {err_text}",
                    }
        except Exception as e:
            logger.exception("任务执行失败")
            tracker.update_task_status(tid, "待办", progress=0)
            return {"success": False, "error": f"执行异常: {str(e)}"}

    @app.delete("/api/tasks/{tid}")
    def delete_task(tid: str, username: str = "default"):
        from core.planting_tracker import PlantingTracker
        PlantingTracker(_storage_dir(username)).delete_task(tid)
        return {"success": True}

    # ── 地块管理 ──────────────────────────────────────

    @app.get("/api/fields")
    def get_fields(username: str = "default"):
        from core.map_manager import MapManager
        fields = MapManager(_storage_dir(username)).get_all_fields()
        return [{
            "id": f.id, "name": f.name, "area_mu": f.area_mu,
            "coordinates": f.coordinates, "center_lat": f.center_lat,
            "center_lon": f.center_lon, "soil_type": f.soil_type,
            "current_crop": f.current_crop,
        } for f in fields]

    @app.post("/api/fields")
    def create_field(data: FieldData, username: str = "default"):
        from core.map_manager import MapManager
        mgr = MapManager(username)
        mgr.add_field({
            "name": data.name, "coordinates": data.coordinates,
            "soil_type": data.soil_type, "current_crop": data.current_crop,
        })
        return {"success": True}

    @app.delete("/api/fields/{fid}")
    def delete_field(fid: str, username: str = "default"):
        from core.map_manager import MapManager
        MapManager(username).delete_field(fid)
        return {"success": True}

    # ── 财务管理 ──────────────────────────────────────

    @app.get("/api/finance/summary")
    def finance_summary(username: str = "default"):
        from core.finance_manager import FinanceManager
        fm = FinanceManager(_storage_dir(username))
        report = fm.get_annual_report()
        return report

    @app.get("/api/finance/costs")
    def get_costs(username: str = "default"):
        from core.finance_manager import FinanceManager
        return FinanceManager(_storage_dir(username)).storage.load_costs()

    @app.post("/api/finance/costs")
    def add_cost(data: CostData, username: str = "default"):
        from core.finance_manager import FinanceManager
        FinanceManager(_storage_dir(username)).add_cost({
            "crop": data.crop, "cost_type": data.cost_type,
            "item_name": data.item_name, "quantity": data.quantity,
            "unit": data.unit, "unit_price": data.unit_price,
        })
        return {"success": True}

    @app.get("/api/finance/income")
    def get_income(username: str = "default"):
        from core.finance_manager import FinanceManager
        return FinanceManager(_storage_dir(username)).storage.load_income()

    @app.post("/api/finance/income")
    def add_income(data: IncomeData, username: str = "default"):
        from core.finance_manager import FinanceManager
        FinanceManager(_storage_dir(username)).add_income({
            "crop": data.crop, "quantity": data.quantity,
            "unit_price": data.unit_price,
        })
        return {"success": True}

    @app.get("/api/finance/export")
    def export_finance():
        from core.finance_manager import FinanceManager
        import tempfile
        path = tempfile.mktemp(suffix=".csv")
        FinanceManager().export_to_csv(path)
        with open(path) as f:
            content = f.read()
        os.unlink(path)
        return {"csv": content}

    # ── 用户档案 ──────────────────────────────────────

    @app.get("/api/profile")
    def get_profile():
        path = os.path.join("data", "user_profile.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    @app.post("/api/profile")
    def save_profile(data: ProfileData):
        os.makedirs("data", exist_ok=True)
        with open(os.path.join("data", "user_profile.json"), "w", encoding="utf-8") as f:
            json.dump(data.model_dump(), f, ensure_ascii=False, indent=2)
        return {"success": True}

    # ── 天气 ──────────────────────────────────────────

    @app.get("/api/weather/{location}")
    def get_weather(location: str):
        cached = _cached_weather(location)
        if cached:
            return cached

        from core.weather_service import WeatherService
        ws = WeatherService()
        current = ws.get_current_weather(location)
        forecast = ws.get_forecast(location, 3)
        alerts = ws.check_weather_alerts(location)
        result = {
            "current": {"temperature": current.temperature, "weather_desc": current.weather_desc,
                        "humidity": current.humidity, "temperature_high": current.temperature_high,
                        "temperature_low": current.temperature_low} if current else None,
            "forecast": [{"date": str(w.date), "weather_desc": w.weather_desc,
                         "temperature_low": w.temperature_low, "temperature_high": w.temperature_high}
                        for w in forecast] if forecast else [],
            "alerts": alerts,
        }
        _set_weather_cache(location, result)
        return result

    @app.get("/api/weather/alerts/{region}")
    def weather_alerts(region: str):
        from core.weather_alerts import check_weather_alert_for_region
        return check_weather_alert_for_region(region)

    # ── 农历节气 ──────────────────────────────────────

    @app.get("/api/solar-terms")
    def solar_terms():
        from core.lunar_calendar import get_lunar_today
        return get_lunar_today()

    # ── 提醒 ──────────────────────────────────────────

    @app.post("/api/reminders")
    def create_reminder(data: ReminderData, username: str = "default"):
        from core.reminder_system import ReminderSystem
        sys = ReminderSystem(_storage_dir(username))
        r = sys.create_reminder({
            "crop": data.crop, "reminder_type": data.reminder_type,
            "task_description": data.task_description,
            "growth_stage": data.growth_stage,
            "start_date": data.start_date or datetime.now().strftime("%Y-%m-%d"),
            "frequency": data.frequency, "time_of_day": data.time_of_day,
            "channels": ["app"],
        })
        return {"success": True, "id": r.id}

    @app.get("/api/reminders/due")
    def due_reminders():
        from core.reminder_scheduler import ReminderScheduler
        sched = ReminderScheduler()
        due = sched.get_due_reminders()
        upcoming = sched.get_upcoming()
        return {"due": due, "upcoming": upcoming}

    # ── 作物百科 ──────────────────────────────────────

    @app.get("/api/encyclopedia")
    def encyclopedia_list():
        crops_dir = os.path.join("agriculture_knowledge", "crops")
        crops = {}
        if os.path.exists(crops_dir):
            for f in sorted(os.listdir(crops_dir)):
                if f.endswith(".json"):
                    with open(os.path.join(crops_dir, f), encoding="utf-8") as fh:
                        d = json.load(fh)
                        crops[d["crop_name"]] = d
        return crops

    @app.get("/api/encyclopedia/{crop_name}")
    def encyclopedia_detail(crop_name: str):
        path = os.path.join("agriculture_knowledge", "crops", f"{crop_name}.json")
        # try fuzzy match
        crops_dir = os.path.join("agriculture_knowledge", "crops")
        if os.path.exists(crops_dir):
            for f in os.listdir(crops_dir):
                if f.endswith(".json"):
                    with open(os.path.join(crops_dir, f), encoding="utf-8") as fh:
                        d = json.load(fh)
                        if d.get("crop_name") == crop_name or crop_name in d.get("aliases", []):
                            return d
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        raise HTTPException(404, "作物未找到")

    # ── 政策搜索 ──────────────────────────────────────

    @app.get("/api/policy/search")
    def policy_search(q: str = ""):
        from knowledge.simple_agriculture_rag import SimpleAgricultureRAG
        rag = SimpleAgricultureRAG()
        results = rag._search_policy(q, k=8) if q else []
        return results

    # ── 种植向导 ──────────────────────────────────────

    @app.post("/api/plan")
    def generate_plan(req: PlanRequest, username: str = "default"):
        from core.planting_planner import PlantingPlanner
        from core.planting_tracker import PlantingTracker
        from core.reminder_system import ReminderSystem

        sd = _storage_dir(username)
        planner = PlantingPlanner()
        plan = planner.generate_plan({
            "region": req.region, "soil_type": req.soil_type,
            "farm_size": req.farm_size, "goals": req.goals,
            "experience": req.experience, "crop": req.crop,
        })

        tracker = PlantingTracker(sd)
        stage_name = plan.schedule.get("stages", [{}])[0].get("stage", "准备期") if plan.schedule else "准备期"
        total_stages = len(plan.schedule.get("stages", [])) if plan.schedule else 1

        tracker.create_progress({
            "crop": plan.crop, "stage": stage_name,
            "stage_number": 1, "total_stages": total_stages,
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "expected_end_date": plan.schedule.get("harvest_time", ""),
            "progress_percent": 0, "status": "进行中",
            "tasks": [], "notes": f"面积: {plan.farm_size}亩",
        })

        task_count = 0
        if plan.tasks:
            for si in plan.tasks:
                for ti in si.get("tasks", [])[:2]:
                    tracker.create_task({
                        "crop": plan.crop,
                        "task_type": ti.get("task", "")[:4],
                        "title": f"{si.get('stage', '')} - {ti.get('task', '')}",
                        "description": f"{plan.crop}的{si.get('stage', '')}阶段",
                        "status": "待办", "priority": "high",
                        "end_date": ti.get("date", ""),
                        "progress_percent": 0,
                    })
                    task_count += 1

        rem_count = 0
        rem_sys = ReminderSystem(sd)
        if plan.tasks:
            for si in plan.tasks:
                for ti in si.get("tasks", [])[:1]:
                    rem_sys.create_reminder({
                        "crop": plan.crop, "reminder_type": ti.get("task", "其他"),
                        "task_description": f"{si.get('stage', '')} - {ti.get('task', '')}",
                        "growth_stage": si.get("stage", ""),
                        "start_date": ti.get("date", datetime.now().strftime("%Y-%m-%d")),
                        "frequency": "单次", "time_of_day": "09:00", "channels": ["app"],
                    })
                    rem_count += 1

        return {
            "plan_text": planner.format_plan_as_text(plan),
            "crop": plan.crop, "task_count": task_count, "reminder_count": rem_count,
            "stage_name": stage_name, "total_stages": total_stages,
        }

    # ── Vision 诊断 ────────────────────────────────────

    @app.get("/api/diagnose/vision")
    def diagnose_vision():
        """诊断 Vision API 配置是否正确"""
        from app.agent.config import VISION_MODEL, VISION_API_KEY, VISION_BASE_URL
        import requests as req

        results = []

        # Step 1: 检查配置
        results.append({"step": "配置检查", "status": "ok" if os.getenv("VISION_MODEL") else "fail",
                       "detail": f"VISION_MODEL={VISION_MODEL} | BASE_URL={VISION_BASE_URL} | "
                       f"KEY={'***' + VISION_API_KEY[-4:] if VISION_API_KEY else '未设置'}"})

        # Step 2: 测试纯文本请求（验证 API Key 和 URL）
        try:
            r = req.post(f"{VISION_BASE_URL}/chat/completions",
                         headers={"Authorization": f"Bearer {VISION_API_KEY}", "Content-Type": "application/json"},
                         json={"model": VISION_MODEL, "messages": [{"role": "user", "content": "回复：OK"}],
                               "max_tokens": 5}, timeout=15)
            if r.status_code == 200:
                results.append({"step": "纯文本连通测试", "status": "ok", "detail": "API Key 和 Base URL 正确"})
            else:
                results.append({"step": "纯文本连通测试", "status": "fail",
                               "detail": f"HTTP {r.status_code}: {r.text[:200]}"})
        except Exception as e:
            results.append({"step": "纯文本连通测试", "status": "fail", "detail": str(e)})

        # Step 3: 测试 1x1 像素图片（验证多模态）
        tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        try:
            r = req.post(f"{VISION_BASE_URL}/chat/completions",
                         headers={"Authorization": f"Bearer {VISION_API_KEY}", "Content-Type": "application/json"},
                         json={"model": VISION_MODEL, "messages": [{"role": "user", "content": [
                             {"type": "text", "text": "这个图片是什么颜色？只说一个词"},
                             {"type": "image_url", "image_url": {"url": "data:image/png;base64," + tiny_png}},
                         ]}], "max_tokens": 10}, timeout=15)
            if r.status_code == 200:
                reply = r.json()["choices"][0]["message"]["content"]
                results.append({"step": "图片识别测试", "status": "ok", "detail": f"模型回复: {reply}"})
            elif r.status_code == 400 and "image_url" in r.text:
                results.append({"step": "图片识别测试", "status": "fail",
                               "detail": f"模型 {VISION_MODEL} 不支持图片输入（非多模态模型）"})
            else:
                results.append({"step": "图片识别测试", "status": "fail",
                               "detail": f"HTTP {r.status_code}: {r.text[:200]}"})
        except Exception as e:
            results.append({"step": "图片识别测试", "status": "fail", "detail": str(e)})

        all_ok = all(r["status"] == "ok" for r in results)
        return {"overall": "pass" if all_ok else "fail", "results": results}

    # ── 自主决策 ────────────────────────────────────

    @app.post("/api/autonomous/trigger")
    def trigger_autonomous_cycle(region: str = None, username: str = "default"):
        """手动触发一次完整巡检"""
        try:
            from core.autonomous_farm_manager import AutonomousFarmManager

            mgr = AutonomousFarmManager()
            if not region:
                from core.device_registry_factory import setup_registry, close_registry
                registry, loop = setup_registry(username)
                try:
                    devices = loop.run_until_complete(registry.discover_all())
                    regions = set(getattr(d, 'location', '') or '默认区域' for d in devices)
                    region = sorted(regions)[0] if regions else None
                finally:
                    close_registry(loop, registry)

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

    # ── 健康检查 ──────────────────────────────────────

    @app.get("/api/health")
    def health():
        return {"status": "ok", "time": datetime.now().isoformat()}

    # ── 设备管理 ──────────────────────────────────────

    @app.get("/api/devices")
    def list_devices(username: str = "default"):
        """获取所有设备列表及状态 — 支持多驱动路由"""
        try:
            # 加载设备→地块映射
            configs = load_custom_devices(username)
            device_to_plot = {d["device_id"]: d.get("plot_id", "") for d in configs}

            # 加载地块信息
            from core.plot_manager import PlotManager
            pm = PlotManager(username)
            plot_map = {p["plot_id"]: p for p in pm.list_plots()}

            with RegistrySession(username) as (registry, loop):
                devices = loop.run_until_complete(registry.discover_all())
                result = []
                for d in devices:
                    state = loop.run_until_complete(registry.read_state(d.device_id))
                    state_clean = {k: v for k, v in state.items() if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                    pid = device_to_plot.get(d.device_id, "")
                    plot_info = plot_map.get(pid, {})
                    result.append({
                        "device_id": d.device_id,
                        "name": d.name,
                        "driver": d.driver_name,
                        "capabilities": [c.value for c in d.capabilities],
                        "sensors": d.sensors,
                        "status": d.status.value if hasattr(d.status, 'value') else str(d.status),
                        "location": d.location,
                        "plot_id": pid,
                        "plot_name": plot_info.get("name", ""),
                        "plot_crop": plot_info.get("crop", ""),
                        "state": state_clean,
                    })
                return result
        except Exception as e:
            logger.exception("获取设备列表失败")
            return []

    @app.post("/api/devices/refresh")
    def refresh_devices(username: str = "default"):
        """清除注册中心缓存，强制重连所有驱动"""
        from core.device_registry_factory import invalidate_registry_cache
        invalidate_registry_cache(username)
        logger.info("设备注册中心缓存已清除 (user=%s)", username)
        return {"success": True, "message": "Registry cache cleared, drivers will reconnect"}

    @app.post("/api/devices")
    def create_device(device_data: Dict, username: str = "default"):
        """添加新的自定义设备"""
        try:
            device_id = device_data.get("device_id", "").strip()
            name = device_data.get("name", "").strip()
            if not device_id or not name:
                return {"success": False, "error": "设备ID和名称不能为空"}

            # 检查 device_id 是否已存在
            custom_devices = load_custom_devices(username)
            existing_ids = {d["device_id"] for d in custom_devices}
            # 也要检查内置设备
            from core.device_registry_factory import BUILTIN_DEVICE_IDS
            if device_id in existing_ids or device_id in BUILTIN_DEVICE_IDS:
                return {"success": False, "error": f"设备ID '{device_id}' 已存在"}

            new_device = {
                "device_id": device_id,
                "name": name,
                "capabilities": device_data.get("capabilities", ["irrigate"]),
                "sensors": device_data.get("sensors", []),
                "location": device_data.get("location", ""),
                "plot_id": device_data.get("plot_id", ""),
                "driver": device_data.get("driver", "mqtt"),
                "initial_state": device_data.get("initial_state", {"power": False, "status": "powered_off"}),
            }
            # 保存驱动连接参数
            conn = device_data.get("connection")
            if conn:
                new_device["connection"] = conn
            custom_devices.append(new_device)
            save_custom_devices(username, custom_devices)
            invalidate_registry_cache(username)
            logger.info("用户 %s 添加了新设备: %s (%s)", username, device_id, name)
            return {"success": True, "device_id": device_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.delete("/api/devices/{device_id}")
    def delete_device(device_id: str, username: str = "default"):
        """删除自定义设备"""
        try:
            custom_devices = load_custom_devices(username)
            before = len(custom_devices)
            custom_devices = [d for d in custom_devices if d["device_id"] != device_id]
            if len(custom_devices) < before:
                save_custom_devices(username, custom_devices)
                invalidate_registry_cache(username)
                return {"success": True}
            return {"success": False, "error": "设备不存在或为内置设备，无法删除"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.post("/api/devices/{device_id}/command")
    def send_device_command(device_id: str, data: Dict = Body({}), username: str = "default"):
        """向设备发送指令 — 自动路由到正确的驱动

        Request body: {"command": "start", "params": {"duration": 30}}
        """
        try:
            from devices.base import DeviceCommand
            from core.device_executor import DeviceExecutor

            command = data.get("command", "start")
            params = data.get("params", {})
            # params 可能是 JSON 字符串(旧前端) 或 dict(新前端)
            if isinstance(params, str):
                params = json.loads(params) if params else {}

            with RegistrySession(username) as (registry, loop):
                loop.run_until_complete(registry.discover_all())

                executor = DeviceExecutor(registry, username=username)
                cmd = DeviceCommand(command=command, params=params)
                result = executor.execute_sync(device_id, cmd, trigger="api", loop=loop)

                msg = result.get("result")
                msg_text = msg.message if msg and hasattr(msg, 'message') else str(msg or "")
                return {
                    "success": result["success"],
                    "device_id": device_id,
                    "message": msg_text,
                    "attempts": result["attempts"],
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/api/devices/{device_id}/state")
    def get_device_state(device_id: str, username: str = "default"):
        """获取设备实时状态"""
        try:
            with RegistrySession(username) as (registry, loop):
                loop.run_until_complete(registry.discover_all())
                state = loop.run_until_complete(registry.read_state(device_id))
                return {k: v for k, v in state.items() if not k.startswith("_")}
        except Exception as e:
            return {"error": str(e)}

    # ── 摄像头拍照 / 分析 ────────────────────────────

    @app.get("/api/devices/{device_id}/snapshot")
    def get_device_snapshot(device_id: str, username: str = "default"):
        """手动触发摄像头拍照，返回 base64 JPEG"""
        try:
            from devices.base import DeviceCommand

            with RegistrySession(username) as (registry, loop):
                loop.run_until_complete(registry.discover_all())

                # 检查设备是否支持拍照
                all_devices = loop.run_until_complete(registry.discover_all())
                device_caps = []
                for d in all_devices:
                    if d.device_id == device_id:
                        device_caps = [c.value for c in d.capabilities]
                        break
                if "capture" not in device_caps:
                    return {"success": False, "error": f"设备 '{device_id}' 不是摄像头，不支持拍照功能"}

                cmd = DeviceCommand(command="capture", params={}, timeout_ms=15000)
                result = loop.run_until_complete(registry.execute(device_id, cmd))

                if not result.success:
                    return {"success": False, "error": result.message}

                # 兼容两种返回格式：CameraDriver 返回 image_bytes，HTTP 设备返回 image_base64
                raw = result.raw_response or {}
                image_b64 = raw.get("image_base64", "")
                if not image_b64:
                    image_bytes = raw.get("image_bytes")
                    if image_bytes:
                        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    else:
                        return {"success": False, "error": "未获取到图片数据"}

                return {
                    "success": True,
                    "device_id": device_id,
                    "image_base64": image_b64,
                    "mime_type": "image/jpeg",
                    "timestamp": datetime.now().isoformat(),
                    "metadata": raw.get("metadata", {}),
                }
        except Exception as e:
            logger.exception("摄像头拍照失败")
            return {"success": False, "error": str(e)}

    @app.get("/api/camera/analysis/{device_id}")
    def get_camera_analysis(device_id: str, username: str = "default", limit: int = 10):
        """获取摄像头最近的分析记录"""
        try:
            photo_dir = os.path.join(DEFAULT_DATA_DIR, username, "photos", device_id)
            if not os.path.exists(photo_dir):
                return {"device_id": device_id, "analyses": []}

            analyses = []
            for fname in sorted(os.listdir(photo_dir), reverse=True):
                if fname.startswith("analysis_") and fname.endswith(".json"):
                    fpath = os.path.join(photo_dir, fname)
                    try:
                        with open(fpath, encoding="utf-8") as f:
                            analyses.append(json.load(f))
                    except Exception:
                        pass
                if len(analyses) >= limit:
                    break

            return {"device_id": device_id, "analyses": analyses}
        except Exception as e:
            return {"device_id": device_id, "analyses": [], "error": str(e)}

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
            from devices.simulator_driver import SimulatorDriver

            engine = RuleEngine(username=username)
            rule = engine.get_rule(rule_id)
            if not rule:
                return {"success": False, "error": "规则不存在"}

            sim = SimulatorDriver(simulated_latency_ms=0)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(sim.connect())
                sensor_data = loop.run_until_complete(sim.read_state("virtual_soil_sensor_01"))

                context = {"sensor_data": sensor_data, "weather": {}}
                matched = engine.find_matching_rules(context)

                return {
                    "success": True,
                    "rule_matched": rule["id"] in [r["id"] for r in matched],
                    "sensor_snapshot": {k: v for k, v in sensor_data.items() if not k.startswith("_")},
                }
            finally:
                close_registry(loop)
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

            registry, loop = setup_registry(username)
            try:
                loop.run_until_complete(registry.discover_all())
                executor = DeviceExecutor(registry, username=username)
                result = executor.confirm_pending(action_id)
                return result
            finally:
                close_registry(loop, registry)
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.post("/api/actions/{action_id}/reject")
    def reject_action(action_id: str, username: str = "default"):
        try:
            from core.device_executor import DeviceExecutor

            registry, loop = setup_registry(username)
            try:
                executor = DeviceExecutor(registry, username=username)
                ok = executor.reject_pending(action_id)
                return {"success": ok}
            finally:
                close_registry(loop, registry)
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── 巡检日志 ──────────────────────────────────────

    @app.get("/api/inspection/log")
    def get_inspection_log(username: str = "default", job_name: str = None, limit: int = 50):
        """获取定时巡检日志"""
        try:
            from app.scheduler_jobs import InspectionLogger
            insp = InspectionLogger.for_user(username)
            logs = insp.get_recent(job_name=job_name, limit=limit)
            return {"username": username, "total": len(logs), "logs": logs}
        except Exception as e:
            return {"error": str(e)}
