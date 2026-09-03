"""FastAPI 路由 — 全部业务逻辑的 HTTP 接口"""

import asyncio
import base64
import json
import logging
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import parse_qsl, urlencode
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field as PydanticField

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.device_registry_factory import setup_registry, load_custom_devices, save_custom_devices, close_registry, DEFAULT_DATA_DIR, RegistrySession, invalidate_registry_cache


def _user_dir(username: str = "default") -> str:
    """返回用户专属数据目录"""
    username = _safe_username(username)
    path = os.path.join(DEFAULT_DATA_DIR, username)
    os.makedirs(path, exist_ok=True)
    return path


def _storage_dir(username: str = "default") -> str:
    u = _safe_username(username)
    return _user_dir(u)


def _safe_username(username: str = "default") -> str:
    """限制用户名字符，防止用户目录发生路径穿越。"""
    value = username or "default"
    if not re.fullmatch(r"[A-Za-z0-9_\-\u4e00-\u9fff]{1,50}", value):
        raise HTTPException(status_code=400, detail="用户名格式无效")
    return value


def _report_dir(username: str = "default") -> Path:
    """返回用户巡检报告目录。"""
    return Path(_storage_dir(username), "autonomous_reports").resolve()


def _safe_report_path(username: str, cycle_id: str) -> Path:
    """生成不会越过用户报告目录的报告路径。"""
    cycle_id = _safe_identifier(cycle_id, "巡检报告编号")
    report_dir = _report_dir(username)
    filepath = (report_dir / f"{cycle_id}.json").resolve()
    if filepath.parent != report_dir:
        raise HTTPException(status_code=400, detail="巡检报告路径无效")
    return filepath


def _safe_identifier(value: str, label: str = "标识符") -> str:
    """限制文件路径中的标识符，拒绝斜杠、反斜杠和点号。"""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value or ""):
        raise HTTPException(status_code=400, detail=f"{label}格式无效")
    return value


def _safe_crop_name(value: str) -> str:
    """允许常用中英文作物名，但不允许任何路径字符。"""
    if not re.fullmatch(r"[A-Za-z0-9_\-\u4e00-\u9fff]{1,50}", value or ""):
        raise HTTPException(status_code=400, detail="作物名称格式无效")
    return value


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


def _public_operation_error(message: str = "操作失败，请稍后重试") -> Dict[str, Any]:
    """向客户端返回稳定错误，不暴露驱动、数据库或第三方 SDK 细节。"""
    return {"success": False, "error": message}


# ── 请求模型 ───────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: List[Dict] = PydanticField(default_factory=list)
    user_question: str = ""
    user_profile: Dict = PydanticField(default_factory=dict)
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
    coordinates: List[List[float]] = PydanticField(default_factory=list)
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
    user_goals: List[str] = PydanticField(default_factory=list)
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
    goals: List[str] = PydanticField(default_factory=list)
    experience: str = ""
    crop: str = ""


class AuthRequest(BaseModel):
    username: str
    password: str


class ChatSessionData(BaseModel):
    session_id: str = ""
    messages: List[Dict[str, Any]] = PydanticField(default_factory=list)
    title: str = ""


class FieldHistoryData(BaseModel):
    crop: str
    season: str = ""
    yield_amount: float = 0
    notes: str = ""


class FieldZoneData(BaseModel):
    zone_id: str
    name: str
    zone_type: str = "operation"
    coordinates: List[List[float]] = PydanticField(default_factory=list)
    notes: str = ""


class PhoneData(BaseModel):
    phone: str = ""


def _get_or_create_user(username: str):
    """由后端统一解析用户，避免前端直接接触数据库。"""
    from core.database.engine import init_db
    from core.database.repository.users import UserRepository
    init_db()
    repo = UserRepository()
    username = _safe_username(username)
    user = repo.get_by_username(username)
    if not user:
        user = repo.create(username=username, password_hash="")
    return user


def _serialize_field(field) -> Dict[str, Any]:
    """将数据库地块对象转换为稳定的前端契约。"""
    try:
        coordinates = json.loads(field.coordinates) if field.coordinates else []
    except (TypeError, json.JSONDecodeError):
        coordinates = []
    try:
        history = json.loads(field.planting_history) if field.planting_history else []
    except (TypeError, json.JSONDecodeError):
        history = []
    return {
        "id": str(field.id), "name": field.name,
        "area_mu": field.area_mu or 0, "area_m2": field.area_m2 or 0,
        "coordinates": coordinates,
        "center_lat": field.center_lat or 0, "center_lon": field.center_lon or 0,
        "soil_type": field.soil_type or "", "current_crop": field.current_crop or "",
        "history": history,
        "created_at": field.created_at.isoformat() if field.created_at else "",
        "updated_at": field.updated_at.isoformat() if field.updated_at else "",
    }


def _calculate_field_geometry(coordinates: List[List[float]]):
    """轻量计算地块面积和中心，避免 API 依赖地图渲染库。"""
    if not coordinates:
        return 0.0, 0.0, 0.0, 0.0
    center_lon = sum(point[0] for point in coordinates) / len(coordinates)
    center_lat = sum(point[1] for point in coordinates) / len(coordinates)
    if len(coordinates) < 3:
        return 0.0, 0.0, center_lat, center_lon

    meters_per_deg_lon = 111320 * math.cos(math.radians(center_lat))
    points_m = [
        ((lon - center_lon) * meters_per_deg_lon, (lat - center_lat) * 111320)
        for lon, lat in coordinates
    ]
    area_m2 = abs(sum(
        points_m[index][0] * points_m[(index + 1) % len(points_m)][1]
        - points_m[(index + 1) % len(points_m)][0] * points_m[index][1]
        for index in range(len(points_m))
    )) / 2.0
    return area_m2, area_m2 / 666.67, center_lat, center_lon


def _issue_auth_token(username: str) -> str:
    from core.auth_token import create_token
    secret = os.getenv("AUTH_SECRET_KEY", "")
    require_auth = os.getenv("REQUIRE_AUTH", "false").lower() in ("1", "true", "yes", "on")
    if require_auth and len(secret) < 32:
        raise HTTPException(
            status_code=503,
            detail="后端 AUTH_SECRET_KEY 未配置或长度不足 32 位",
        )
    return create_token(username, secret) if secret else ""


def _migrate_legacy_profile(user) -> bool:
    """把无归属的旧档案只迁给首个成功登录的既有用户。"""
    from core.database.repository.users import UserProfileRepository
    repo = UserProfileRepository()
    if repo.get_by_user_id(user.id) or repo.count() > 0:
        return False
    legacy_path = os.path.join(DEFAULT_DATA_DIR, "user_profile.json")
    if not os.path.exists(legacy_path):
        return False
    try:
        with open(legacy_path, encoding="utf-8") as file:
            legacy = json.load(file)
        repo.create(
            user_id=user.id,
            region=legacy.get("user_region", ""),
            soil_type=legacy.get("user_soil_type", ""),
            farm_size=legacy.get("user_farm_size", 1.0),
            experience_level=legacy.get("user_experience", ""),
            goals=json.dumps(legacy.get("user_goals", []), ensure_ascii=False),
            phone=legacy.get("user_phone", ""),
        )
        logger.info("旧版用户档案已迁入数据库用户: %s", user.username)
        return True
    except (OSError, json.JSONDecodeError):
        logger.warning("旧版用户档案迁移失败", exc_info=True)
        return False


def _migrate_legacy_chat_history(user) -> None:
    """档案迁移成功时，把旧版 default 会话复制给同一用户。"""
    if user.username == "default":
        return
    from core.chat_history import ChatHistoryStore
    source = ChatHistoryStore(username="default")
    target = ChatHistoryStore(username=user.username)
    if target.list_sessions(limit=1):
        return
    for session in reversed(source.list_sessions(limit=100)):
        messages = source.load_session(session["id"]) or []
        target.save_session("", messages, session.get("title", ""))


def register_routes(app: FastAPI):

    @app.middleware("http")
    async def validate_username_query(request: Request, call_next):
        """统一拦截非法用户名，保护所有按用户拼接的存储路径。"""
        query_items = parse_qsl(
            request.scope.get("query_string", b"").decode("utf-8"),
            keep_blank_values=True,
        )
        username = next(
            (value for key, value in query_items if key == "username"),
            None,
        )
        if username is not None:
            try:
                _safe_username(username)
            except HTTPException as exc:
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        require_auth = os.getenv("REQUIRE_AUTH", "false").lower() in ("1", "true", "yes", "on")
        public_path = (
            request.url.path.startswith("/api/auth/")
            or request.url.path in ("/api/health", "/docs", "/openapi.json", "/redoc")
        )
        if require_auth and not public_path and request.method != "OPTIONS":
            from core.auth_token import verify_token
            secret = os.getenv("AUTH_SECRET_KEY", "")
            if len(secret) < 32:
                return JSONResponse(
                    status_code=503,
                    content={"detail": "后端 AUTH_SECRET_KEY 未配置或长度不足 32 位"},
                )
            authorization = request.headers.get("Authorization", "")
            token = authorization[7:] if authorization.startswith("Bearer ") else ""
            token_username = verify_token(token, secret) if secret and token else None
            if not token_username:
                return JSONResponse(status_code=401, content={"detail": "登录令牌无效或已过期"})
            if username is not None and token_username != username:
                return JSONResponse(status_code=403, content={"detail": "无权访问其他用户的数据"})
            request.state.authenticated_username = token_username
            # 未显式传用户名时，以令牌主体作为唯一身份来源。
            if username is None:
                query_items.append(("username", token_username))
                request.scope["query_string"] = urlencode(query_items).encode("utf-8")
                if hasattr(request, "_query_params"):
                    delattr(request, "_query_params")
        return await call_next(request)

    # ── 用户认证 ──────────────────────────────────────

    @app.post("/api/auth/login")
    def login(data: AuthRequest):
        from core.database.engine import init_db
        from core.database.repository.users import UserRepository
        from core.password_security import hash_password, verify_password
        init_db()
        username = _safe_username(data.username)
        if not data.password or len(data.password) > 128:
            raise HTTPException(status_code=400, detail="密码格式无效")
        repo = UserRepository()
        user = repo.get_by_username(username)
        verified, needs_upgrade = verify_password(
            data.password, user.password_hash
        ) if user else (False, False)
        if not verified:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        if needs_upgrade:
            repo.update(user.id, password_hash=hash_password(data.password))
            if _migrate_legacy_profile(user):
                _migrate_legacy_chat_history(user)
        return {
            "success": True,
            "username": user.username,
            "token": _issue_auth_token(user.username),
        }

    @app.post("/api/auth/register")
    def register(data: AuthRequest):
        from core.database.engine import init_db
        from core.database.repository.users import UserRepository
        from core.password_security import hash_password
        from sqlalchemy.exc import IntegrityError
        init_db()
        username = _safe_username(data.username)
        if not data.password or len(data.password) > 128:
            raise HTTPException(status_code=400, detail="密码格式无效")
        repo = UserRepository()
        if repo.get_by_username(username):
            raise HTTPException(status_code=409, detail="用户名已存在")
        try:
            repo.create(
                username=username,
                password_hash=hash_password(data.password),
            )
        except IntegrityError as exc:
            repo.session.rollback()
            raise HTTPException(status_code=409, detail="用户名已存在") from exc
        _user_dir(username)
        return {
            "success": True,
            "username": username,
            "token": _issue_auth_token(username),
        }

    # ── 对话 ──────────────────────────────────────────

    @app.post("/api/chat")
    def chat(req: ChatRequest, request: Request, username: str = "default"):
        from app.agent.state import AgentState
        from app.chat_service import get_chat_agent

        authenticated_username = getattr(request.state, "authenticated_username", "")
        if (
            authenticated_username
            and req.username not in ("", "default", authenticated_username)
        ):
            raise HTTPException(status_code=403, detail="无权以其他用户身份发起对话")
        resolved_username = _safe_username(
            authenticated_username or username or req.username
        )
        agent = get_chat_agent()

        state = AgentState(
            messages=[],
            user_profile=req.user_profile,
            username=resolved_username,
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
        except Exception:
            logger.exception("Agent 调用失败")
            answer = "抱歉，处理请求时发生内部错误，请稍后重试。"
            error_code = "AGENT_EXECUTION_FAILED"
        else:
            error_code = ""

        logger.info(
            "对话请求完成: user=%s answer_len=%d error_code=%s",
            resolved_username,
            len(answer),
            error_code or "none",
        )

        # 安全序列化
        facts = {}
        for k, v in state.short_term_facts.items():
            try:
                facts[k] = v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v)
            except Exception:
                facts[k] = str(v)

        return {
            "final_answer": answer,
            "short_term_facts": facts,
            "error_code": error_code,
        }

    # ── 对话历史 ──────────────────────────────────────

    @app.get("/api/chat/sessions")
    def list_chat_sessions(username: str = "default", limit: int = 20):
        from core.chat_history import ChatHistoryStore
        return ChatHistoryStore(username=username).list_sessions(limit=max(1, min(limit, 100)))

    @app.get("/api/chat/sessions/{session_id}")
    def load_chat_session(session_id: str, username: str = "default"):
        from core.chat_history import ChatHistoryStore
        messages = ChatHistoryStore(username=username).load_session(session_id)
        if messages is None:
            raise HTTPException(status_code=404, detail="对话不存在")
        return {"id": session_id, "messages": messages}

    @app.post("/api/chat/sessions")
    def save_chat_session(data: ChatSessionData, username: str = "default"):
        from core.chat_history import ChatHistoryStore
        sid = ChatHistoryStore(username=username).save_session(
            data.session_id, data.messages, data.title
        )
        return {"success": True, "id": sid}

    @app.delete("/api/chat/sessions/{session_id}")
    def delete_chat_session(session_id: str, username: str = "default"):
        from core.chat_history import ChatHistoryStore
        deleted = ChatHistoryStore(username=username).delete_session(session_id)
        return {"success": deleted}

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

        persistence = {}
        disease_risks = []
        try:
            with open(os.path.join(DEFAULT_DATA_DIR, "weather_persistence.json"), encoding="utf-8") as f:
                persistence = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        try:
            with open(os.path.join(DEFAULT_DATA_DIR, "disease_risks.json"), encoding="utf-8") as f:
                disease_risks = json.load(f).get("risks", [])
        except (OSError, json.JSONDecodeError):
            pass

        return {
            "progress": progresses,
            "tasks": {"active": [{"title": t.title, "crop": t.crop, "priority": t.priority, "status": t.status} for t in active],
                      "overdue": [{"title": t.title, "crop": t.crop} for t in overdue]},
            "finance": {"month_income": month_income, "month_cost": month_cost, "profit": month_income - month_cost},
            "weather_alerts": alerts,
            "weather_persistence": persistence,
            "disease_risks": disease_risks,
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
    def get_tasks(username: str = "default", limit: int = 200):
        from core.planting_tracker import PlantingTracker
        # 日历需要拿到完整的近期任务集合；旧默认值 10 会让新增任务
        # 在已有高优先级任务较多时被截断，看起来像是保存失败。
        safe_limit = max(1, min(limit, 1000))
        cards = PlantingTracker(_storage_dir(username)).get_task_cards(
            limit=safe_limit
        )
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
        from core.database.repository.fields import FieldRepository
        user = _get_or_create_user(username)
        return [_serialize_field(f) for f in FieldRepository().find_by(user_id=user.id)]

    @app.post("/api/fields")
    def create_field(data: FieldData, username: str = "default"):
        from core.database.repository.fields import FieldRepository
        user = _get_or_create_user(username)
        area_m2, area_mu, center_lat, center_lon = _calculate_field_geometry(data.coordinates)
        field = FieldRepository().create(
            user_id=user.id, name=data.name,
            coordinates=json.dumps(data.coordinates, ensure_ascii=False),
            center_lat=center_lat, center_lon=center_lon,
            area_mu=area_mu, area_m2=area_m2,
            soil_type=data.soil_type, current_crop=data.current_crop,
            planting_history="[]",
        )
        return {"success": True, "field": _serialize_field(field)}

    @app.delete("/api/fields/{fid}")
    def delete_field(fid: str, username: str = "default"):
        from core.database.repository.fields import FieldRepository
        user = _get_or_create_user(username)
        repo = FieldRepository()
        field = repo.get_by_id(int(fid)) if fid.isdigit() else None
        if not field or field.user_id != user.id:
            raise HTTPException(status_code=404, detail="地块不存在")
        repo.delete(field.id)
        return {"success": True}

    @app.post("/api/fields/{fid}/history")
    def add_field_history(fid: str, data: FieldHistoryData, username: str = "default"):
        from core.database.repository.fields import FieldRepository
        user = _get_or_create_user(username)
        repo = FieldRepository()
        field = repo.get_by_id(int(fid)) if fid.isdigit() else None
        if not field or field.user_id != user.id:
            raise HTTPException(status_code=404, detail="地块不存在")
        try:
            history = json.loads(field.planting_history) if field.planting_history else []
        except (TypeError, json.JSONDecodeError):
            history = []
        history.append({
            "crop": data.crop,
            "season": data.season or datetime.now().strftime("%Y"),
            "yield_amount": data.yield_amount,
            "notes": data.notes,
            "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        repo.update(field.id, planting_history=json.dumps(history, ensure_ascii=False))
        return {"success": True}

    @app.get("/api/fields/{fid}/zones")
    def list_field_zones(fid: str, username: str = "default"):
        from core.database.repository.fields import FieldRepository
        from core.database.repository.zones import FieldZoneRepository

        user = _get_or_create_user(username)
        field = FieldRepository().get_by_id(int(fid)) if fid.isdigit() else None
        if not field or field.user_id != user.id:
            raise HTTPException(status_code=404, detail="地块不存在")
        rows = FieldZoneRepository().find_by(user_id=user.id, field_id=field.id)
        return [{
            "id": row.id,
            "field_id": row.field_id,
            "zone_id": row.zone_id,
            "name": row.name,
            "zone_type": row.zone_type,
            "coordinates": json.loads(row.coordinates or "[]"),
            "notes": row.notes or "",
        } for row in rows]

    @app.post("/api/fields/{fid}/zones")
    def create_field_zone(
        fid: str,
        data: FieldZoneData,
        username: str = "default",
    ):
        from core.database.repository.fields import FieldRepository
        from core.database.repository.zones import FieldZoneRepository

        user = _get_or_create_user(username)
        field = FieldRepository().get_by_id(int(fid)) if fid.isdigit() else None
        if not field or field.user_id != user.id:
            raise HTTPException(status_code=404, detail="地块不存在")
        zone_id = str(data.zone_id).strip()
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,100}", zone_id):
            raise HTTPException(status_code=400, detail="分区 ID 格式无效")
        repo = FieldZoneRepository()
        if repo.find_one(user_id=user.id, field_id=field.id, zone_id=zone_id):
            raise HTTPException(status_code=409, detail="分区 ID 已存在")
        row = repo.create(
            user_id=user.id,
            field_id=field.id,
            zone_id=zone_id,
            name=data.name.strip() or zone_id,
            zone_type=data.zone_type.strip() or "operation",
            coordinates=json.dumps(data.coordinates, ensure_ascii=False),
            notes=data.notes,
        )
        return {"success": True, "id": row.id, "zone_id": row.zone_id}

    @app.delete("/api/fields/{fid}/zones/{zone_id}")
    def delete_field_zone(fid: str, zone_id: str, username: str = "default"):
        from core.database.repository.fields import FieldRepository
        from core.database.repository.devices import DeviceConfigRepository
        from core.database.repository.zones import FieldZoneRepository

        user = _get_or_create_user(username)
        field = FieldRepository().get_by_id(int(fid)) if fid.isdigit() else None
        if not field or field.user_id != user.id:
            raise HTTPException(status_code=404, detail="地块不存在")
        repo = FieldZoneRepository()
        row = repo.find_one(user_id=user.id, field_id=field.id, zone_id=zone_id)
        if not row:
            raise HTTPException(status_code=404, detail="分区不存在")
        assigned_device = DeviceConfigRepository().find_one(
            user_id=user.id,
            plot_id=field.id,
            zone_id=zone_id,
        )
        if assigned_device:
            raise HTTPException(
                status_code=409,
                detail="该分区仍有关联设备，请先调整设备分区",
            )
        repo.delete(row.id)
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
    def export_finance(username: str = "default"):
        from core.finance_manager import FinanceManager
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            exported = FinanceManager(_storage_dir(username)).export_to_csv(path)
            if not exported:
                raise HTTPException(status_code=500, detail="财务数据导出失败")
            # 保留 UTF-8 BOM，确保 Excel 直接打开中文 CSV 时不会乱码。
            with open(path, encoding="utf-8") as file:
                content = file.read()
            return {"csv": content}
        finally:
            if os.path.exists(path):
                os.unlink(path)

    # ── 用户档案 ──────────────────────────────────────

    @app.get("/api/profile")
    def get_profile(username: str = "default"):
        from core.database.repository.users import UserProfileRepository
        user = _get_or_create_user(username)
        profile = UserProfileRepository().get_by_user_id(user.id)
        if not profile:
            return {}
        try:
            goals = json.loads(profile.goals) if profile.goals else []
        except (TypeError, json.JSONDecodeError):
            goals = []
        return {
            "user_region": profile.region or "",
            "user_soil_type": profile.soil_type or "",
            "user_farm_size": profile.farm_size or 1.0,
            "user_experience": profile.experience_level or "",
            "user_goals": goals,
            "user_phone": profile.phone or "",
        }

    @app.post("/api/profile")
    def save_profile(data: ProfileData, username: str = "default"):
        from core.database.repository.users import UserProfileRepository
        user = _get_or_create_user(username)
        repo = UserProfileRepository()
        values = {
            "region": data.user_region,
            "soil_type": data.user_soil_type,
            "farm_size": data.user_farm_size,
            "experience_level": data.user_experience,
            "goals": json.dumps(data.user_goals, ensure_ascii=False),
            "phone": data.user_phone,
        }
        existing = repo.get_by_user_id(user.id)
        if existing:
            repo.update(existing.id, **values)
        else:
            repo.create(user_id=user.id, **values)
        return {"success": True}

    # ── 天气 ──────────────────────────────────────────

    @app.get("/api/weather-by-coordinates")
    def get_weather_by_coordinates(lon: float, lat: float):
        from core.weather_service import WeatherService
        return WeatherService().get_grid_weather(lon, lat) or {}

    @app.get("/api/geocode")
    def geocode(address: str):
        from core.map_manager import get_location_from_address
        coords = get_location_from_address(address)
        return {"lat": coords[0], "lon": coords[1]} if coords else {}

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

    @app.get("/api/alerts/proactive")
    def proactive_alerts():
        """集中读取调度器生成的主动预警，前端不再依赖共享磁盘。"""
        alerts = []
        try:
            with open(os.path.join(DEFAULT_DATA_DIR, "weather_alerts_cache.json"), encoding="utf-8") as f:
                weather = json.load(f)
            if weather.get("has_alert"):
                for item in weather.get("alerts", []):
                    alerts.append({"kind": "weather", **item})
        except (OSError, json.JSONDecodeError):
            pass
        try:
            with open(os.path.join(DEFAULT_DATA_DIR, "disease_risks.json"), encoding="utf-8") as f:
                disease = json.load(f)
            for item in disease.get("risks", [])[:3]:
                if item.get("risk") in ("高", "中"):
                    alerts.append({"kind": "disease", **item})
        except (OSError, json.JSONDecodeError):
            pass
        return {"alerts": alerts[:5]}

    # ── 农历节气 ──────────────────────────────────────

    @app.get("/api/solar-terms")
    def solar_terms():
        from core.lunar_calendar import get_lunar_today
        return get_lunar_today()

    # ── 提醒 ──────────────────────────────────────────

    @app.post("/api/reminders")
    def create_reminder(data: ReminderData, username: str = "default"):
        from core.reminder_system import ReminderSystem
        sys = ReminderSystem(_storage_dir(username), username=username)
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
    def due_reminders(username: str = "default"):
        from core.reminder_scheduler import ReminderScheduler
        sched = ReminderScheduler(_storage_dir(username), username=username)
        due = sched.get_due_reminders(username)
        upcoming = sched.get_upcoming(username)
        return {"due": due, "upcoming": upcoming}

    @app.post("/api/reminders/check")
    def check_reminders(data: PhoneData, username: str = "default"):
        from core.reminder_scheduler import ReminderScheduler
        sched = ReminderScheduler(_storage_dir(username), username=username)
        fired = sched.check_and_fire(username, data.phone)
        upcoming = sched.get_upcoming(username)
        return {"fired": fired, "upcoming": upcoming}

    @app.post("/api/sms/test")
    def test_sms(data: PhoneData):
        from core.sms_service import SMSService
        sms = SMSService()
        if not sms.is_configured:
            return {"success": False, "error": "短信服务未配置，请在后端环境变量中设置短信密钥"}
        return sms.send_reminder(
            phone=data.phone, crop="测试", task_type="测试",
            task_desc="这是一条测试短信", time_info="测试时间",
        )

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
        crop_name = _safe_crop_name(crop_name)
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
        keyword = str(q or "").strip()
        if not keyword:
            return []
        from knowledge.simple_agriculture_rag import SimpleAgricultureRAG
        rag = SimpleAgricultureRAG()
        local_results = rag._search_policy(keyword, k=8)
        if local_results:
            return local_results
        try:
            from core.policy_search import search_official_policies
            return search_official_policies(keyword, limit=8)
        except Exception as exc:
            logger.exception("政策检索失败")
            raise HTTPException(
                status_code=502,
                detail="政策数据源暂时不可用，请稍后重试",
            ) from exc

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
        rem_sys = ReminderSystem(sd, username=username)
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
        except Exception:
            logger.exception("手动触发巡检失败")
            return _public_operation_error("巡检启动失败，请稍后重试")

    @app.get("/api/autonomous/reports")
    def list_autonomous_reports(username: str = "default", limit: int = 20):
        """查询历史巡检报告列表"""
        try:
            report_dir = _report_dir(username)
            if not os.path.exists(report_dir):
                return {"reports": []}

            reports = []
            for fname in sorted(os.listdir(report_dir), reverse=True):
                if fname.endswith(".json"):
                    fpath = report_dir / fname
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
            filepath = _safe_report_path(username, cycle_id)
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
        return {
            "service": "agriculture-agent",
            "status": "ok",
            "time": datetime.now().isoformat(),
        }

    # ── 设备管理 ──────────────────────────────────────

    def normalize_device_config(
        device_id: str,
        device_data: Dict,
        current: Optional[Dict] = None,
    ) -> Dict:
        """校验并规范化设备配置，更新时保留未提交的原配置字段。"""
        source = {**(current or {}), **(device_data or {})}
        name = str(source.get("name", "")).strip()
        if not device_id or not name:
            raise ValueError("设备ID和名称不能为空")

        driver = str(source.get("driver", "mqtt")).lower().strip()
        from core.device_registry_factory import SUPPORTED_DEVICE_DRIVERS
        if driver not in SUPPORTED_DEVICE_DRIVERS:
            raise ValueError(f"不支持的设备驱动: {driver}")

        connection = source.get("connection", {})
        if connection is None:
            connection = {}
        if not isinstance(connection, dict):
            raise ValueError("connection 必须是 JSON 对象")
        if driver == "mqtt" and any(char in device_id for char in "+#"):
            raise ValueError("MQTT 设备ID不能包含 + 或 #")
        if driver == "http" and not str(connection.get("base_url", "")).startswith(
            ("http://", "https://")
        ):
            raise ValueError("HTTP 设备必须提供有效的 base_url")
        if driver == "coap" and not str(connection.get("base_uri", "")).startswith(
            ("coap://", "coaps://")
        ):
            raise ValueError("CoAP 设备必须提供有效的 base_uri")
        if driver == "opcua":
            if not str(connection.get("endpoint", "")).startswith("opc.tcp://"):
                raise ValueError("OPC UA 设备必须提供有效的 endpoint")
            if not isinstance(connection.get("command_nodes", {}), dict) or not isinstance(
                connection.get("state_nodes", {}), dict
            ):
                raise ValueError("OPC UA 节点映射必须是 JSON 对象")

        capabilities = source.get("capabilities", ["irrigate"])
        if not isinstance(capabilities, list):
            raise ValueError("capabilities 必须是数组")
        from devices.base import DeviceCapability
        valid_capabilities = {item.value for item in DeviceCapability}
        invalid_capabilities = [
            item for item in capabilities if item not in valid_capabilities
        ]
        if invalid_capabilities:
            raise ValueError(f"不支持的设备能力: {invalid_capabilities[0]}")

        sensors = source.get("sensors", [])
        if not isinstance(sensors, list) or any(
            not isinstance(item, str) for item in sensors
        ):
            raise ValueError("sensors 必须是字符串数组")
        initial_state = source.get(
            "initial_state", {"power": False, "status": "powered_off"}
        )
        if not isinstance(initial_state, dict):
            raise ValueError("initial_state 必须是 JSON 对象")

        return {
            "device_id": device_id,
            "name": name,
            "capabilities": capabilities,
            "sensors": sensors,
            "location": str(source.get("location", "")).strip(),
            "plot_id": source.get("plot_id") or "",
            "zone_id": str(source.get("zone_id", "")).strip(),
            "driver": driver,
            "initial_state": initial_state,
            "connection": connection,
        }

    @app.get("/api/devices")
    def list_devices(username: str = "default"):
        """获取所有设备列表及状态 — 支持多驱动路由"""
        configs = []
        plot_map = {}

        def configured_offline_rows(excluded_ids=None):
            """即使驱动尚未连接，也返回用户已经保存的设备配置。"""
            excluded = excluded_ids or set()
            rows = []
            for config in configs:
                if config.get("device_id") in excluded:
                    continue
                pid = config.get("plot_id", "")
                plot_info = plot_map.get(pid, {})
                rows.append({
                    "device_id": config.get("device_id", ""),
                    "name": config.get("name", config.get("device_id", "")),
                    "driver": config.get("driver", "simulator"),
                    "capabilities": config.get("capabilities", []),
                    "sensors": config.get("sensors", []),
                    "status": "offline",
                    "location": config.get("location", ""),
                    "plot_id": pid,
                    "zone_id": config.get("zone_id", ""),
                    "plot_name": plot_info.get("name", ""),
                    "plot_crop": plot_info.get("crop", ""),
                    "state": config.get("initial_state", {}),
                    "connection": config.get("connection", {}),
                    "initial_state": config.get("initial_state", {}),
                    "editable": True,
                })
            return rows

        try:
            # 加载设备→地块映射
            configs = load_custom_devices(username)
            config_map = {d["device_id"]: d for d in configs}
            device_to_plot = {d["device_id"]: d.get("plot_id", "") for d in configs}

            # 加载地块信息
            from core.plot_manager import PlotManager
            pm = PlotManager(username)
            plot_map = {p["plot_id"]: p for p in pm.list_plots()}

            with RegistrySession(username) as (registry, loop):
                devices = loop.run_until_complete(registry.discover_all())
                result = []
                discovered_ids = set()
                for d in devices:
                    discovered_ids.add(d.device_id)
                    state = loop.run_until_complete(registry.read_state(d.device_id))
                    state_clean = {k: v for k, v in state.items() if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                    pid = device_to_plot.get(d.device_id, "")
                    plot_info = plot_map.get(pid, {})
                    config = config_map.get(d.device_id, {})
                    result.append({
                        "device_id": d.device_id,
                        "name": d.name,
                        "driver": d.driver_name,
                        "capabilities": [c.value for c in d.capabilities],
                        "sensors": d.sensors,
                        "status": d.status.value if hasattr(d.status, 'value') else str(d.status),
                        "location": d.location,
                        "plot_id": pid,
                        "zone_id": config.get("zone_id", ""),
                        "plot_name": plot_info.get("name", ""),
                        "plot_crop": plot_info.get("crop", ""),
                        "state": state_clean,
                        "connection": config.get("connection", {}),
                        "initial_state": config.get("initial_state", {}),
                        "editable": bool(config),
                    })
                # 连接失败或驱动依赖缺失的设备也必须可见，方便用户修正配置。
                result.extend(configured_offline_rows(discovered_ids))
                return result
        except Exception:
            logger.exception("获取设备列表失败")
            return configured_offline_rows()

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
            try:
                new_device = normalize_device_config(device_id, device_data)
            except ValueError as exc:
                return {"success": False, "error": str(exc)}

            # 检查 device_id 是否已存在
            custom_devices = load_custom_devices(username)
            existing_ids = {d["device_id"] for d in custom_devices}
            # 也要检查内置设备
            from core.device_registry_factory import BUILTIN_DEVICE_IDS
            if device_id in existing_ids or device_id in BUILTIN_DEVICE_IDS:
                return {"success": False, "error": f"设备ID '{device_id}' 已存在"}

            custom_devices.append(new_device)
            save_custom_devices(username, custom_devices)
            invalidate_registry_cache(username)
            logger.info("用户 %s 添加了新设备: %s (%s)", username, device_id, new_device["name"])
            return {"success": True, "device_id": device_id}
        except Exception:
            logger.exception("创建设备失败")
            return _public_operation_error("设备保存失败，请检查配置后重试")

    @app.post("/api/devices/{device_id}/config")
    @app.put("/api/devices/{device_id}")
    def update_device(
        device_id: str,
        device_data: Dict = Body({}),
        username: str = "default",
    ):
        """更新已有自定义设备的位置、驱动、能力和连接参数。"""
        try:
            custom_devices = load_custom_devices(username)
            index = next(
                (
                    item_index
                    for item_index, item in enumerate(custom_devices)
                    if item.get("device_id") == device_id
                ),
                None,
            )
            if index is None:
                return {
                    "success": False,
                    "error": "设备不存在或为内置设备，无法修改配置",
                }
            try:
                updated = normalize_device_config(
                    device_id, device_data, current=custom_devices[index]
                )
            except ValueError as exc:
                return {"success": False, "error": str(exc)}

            custom_devices[index] = updated
            save_custom_devices(username, custom_devices)
            invalidate_registry_cache(username)
            logger.info("用户 %s 更新了设备配置: %s", username, device_id)
            return {"success": True, "device": updated}
        except Exception:
            logger.exception("更新设备配置失败: %s", device_id)
            return _public_operation_error("设备配置更新失败，请稍后重试")

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
        except Exception:
            logger.exception("删除设备失败: %s", device_id)
            return _public_operation_error("设备删除失败，请稍后重试")

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
            timeout_ms = int(data.get("timeout_ms", 30000))
            if not (1 <= timeout_ms <= 300000):
                return {"success": False, "error": "timeout_ms 必须在 1-300000 之间"}
            # params 可能是 JSON 字符串(旧前端) 或 dict(新前端)
            if isinstance(params, str):
                params = json.loads(params) if params else {}
            if not isinstance(params, dict):
                return {"success": False, "error": "params 必须是 JSON 对象"}

            with RegistrySession(username) as (registry, loop):
                loop.run_until_complete(registry.discover_all())

                executor = DeviceExecutor(registry, username=username)
                cmd = DeviceCommand(
                    command=command, params=params, timeout_ms=timeout_ms
                )
                result = executor.execute_sync(
                    device_id, cmd, trigger="api", loop=loop,
                    capability=data.get("capability"),
                    policy_context={
                        "plot_id": data.get("plot_id"),
                        "zone_id": data.get("zone_id"),
                        "sensor_data": data.get("sensor_data", {}),
                    },
                )

                msg = result.get("result")
                msg_text = msg.message if msg and hasattr(msg, 'message') else str(msg or "")
                return {
                    "success": result["success"],
                    "device_id": device_id,
                    "message": msg_text,
                    "attempts": result["attempts"],
                    "decision": result.get("decision", "auto_execute"),
                    "pending_id": result.get("pending_id"),
                }
        except Exception:
            logger.exception("发送设备指令失败: %s", device_id)
            return _public_operation_error("设备指令执行失败，请稍后重试")

    @app.get("/api/devices/{device_id}/state")
    def get_device_state(device_id: str, username: str = "default"):
        """获取设备实时状态"""
        try:
            with RegistrySession(username) as (registry, loop):
                loop.run_until_complete(registry.discover_all())
                state = loop.run_until_complete(registry.read_state(device_id))
                return {k: v for k, v in state.items() if not k.startswith("_")}
        except Exception:
            logger.exception("读取设备状态失败: %s", device_id)
            return {"error": "设备状态读取失败，请稍后重试"}

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
        except Exception:
            logger.exception("摄像头拍照失败")
            return _public_operation_error("摄像头拍照失败，请稍后重试")

    @app.get("/api/camera/analysis/{device_id}")
    def get_camera_analysis(device_id: str, username: str = "default", limit: int = 10):
        """获取摄像头最近的分析记录"""
        try:
            device_id = _safe_identifier(device_id, "设备编号")
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
        except HTTPException:
            raise
        except Exception as e:
            return {"device_id": device_id, "analyses": [], "error": str(e)}

    # ── 设备动作参数与安全策略 ────────────────────────

    @app.get("/api/device-actions/catalog")
    def get_device_action_catalog():
        """返回各设备能力支持的动作参数，供规则页面动态渲染。"""
        from core.device_action_schema import get_action_catalog

        return get_action_catalog()

    @app.get("/api/safety-policies/catalog")
    def get_safety_policy_catalog():
        from core.device_safety_policy import get_safety_catalog

        return get_safety_catalog()

    @app.get("/api/safety-policies")
    def list_safety_policies(username: str = "default"):
        from core.device_safety_policy import SafetyPolicyService

        return SafetyPolicyService(username).list_policies()

    @app.post("/api/safety-policies")
    def create_safety_policy(data: Dict = Body({}), username: str = "default"):
        try:
            from core.device_safety_policy import SafetyPolicyService

            policy = SafetyPolicyService(username).create_policy(data)
            return {"success": True, "policy": policy}
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception:
            logger.exception("创建安全策略失败")
            return _public_operation_error("安全策略保存失败，请稍后重试")

    @app.put("/api/safety-policies/{policy_id}")
    def update_safety_policy(
        policy_id: int, data: Dict = Body({}), username: str = "default"
    ):
        try:
            from core.device_safety_policy import SafetyPolicyService

            policy = SafetyPolicyService(username).update_policy(policy_id, data)
            if not policy:
                return {"success": False, "error": "安全策略不存在"}
            return {"success": True, "policy": policy}
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception:
            logger.exception("更新安全策略失败: %s", policy_id)
            return _public_operation_error("安全策略更新失败，请稍后重试")

    @app.delete("/api/safety-policies/{policy_id}")
    def delete_safety_policy(policy_id: int, username: str = "default"):
        try:
            from core.device_safety_policy import SafetyPolicyService

            return {
                "success": SafetyPolicyService(username).delete_policy(policy_id)
            }
        except Exception:
            logger.exception("删除安全策略失败: %s", policy_id)
            return _public_operation_error("安全策略删除失败，请稍后重试")

    # ── 自动化规则管理 ────────────────────────────────

    @app.get("/api/rules")
    def list_rules(username: str = "default"):
        try:
            from core.device_rule_engine import RuleEngine
            engine = RuleEngine(username=username)
            return engine.list_rules()
        except Exception:
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
            from core.device_registry_factory import setup_registry, close_registry

            engine = RuleEngine(username=username)
            rule = engine.get_rule(rule_id)
            if not rule:
                return {"success": False, "error": "规则不存在"}

            registry, loop = setup_registry(username)
            try:
                devices = loop.run_until_complete(registry.discover_all())
                sensor_data = {}
                for device in devices:
                    try:
                        state = loop.run_until_complete(
                            registry.read_state(device.device_id)
                        )
                    except Exception:
                        continue
                    if not state or state.get("error"):
                        continue
                    for key, value in state.items():
                        if isinstance(value, (int, float)) and not key.startswith("_"):
                            sensor_data.setdefault(key, value)
                            sensor_data[f"{device.device_id}.{key}"] = value

                context = {"sensor_data": sensor_data, "weather": {}}
                matched = engine.find_matching_rules(context)

                return {
                    "success": True,
                    "rule_matched": rule["id"] in [r["id"] for r in matched],
                    "sensor_snapshot": {k: v for k, v in sensor_data.items() if not k.startswith("_")},
                }
            finally:
                close_registry(loop, registry)
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── 操作管理：逐步拆出超大路由文件 ─────────────────
    from app.routes.device_actions import register_device_action_routes
    register_device_action_routes(app)

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
