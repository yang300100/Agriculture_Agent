"""APScheduler 定时任务 — 提醒检查 / 天气预警 / 病害风险评估 / 设备巡检"""

import logging
import os
import json
import threading
from datetime import datetime
from typing import List, Dict, Optional

from core.storage_paths import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# 巡检日志 — 每次定时任务执行时写入结构化记录
# ═══════════════════════════════════════════════════

class InspectionLogger:
    """巡检日志管理器 — 记录每次定时任务的执行结果到磁盘"""

    _instances: Dict[str, "InspectionLogger"] = {}
    _lock = threading.Lock()

    def __init__(self, username: str):
        self.username = username
        self.log_path = os.path.join(DEFAULT_DATA_DIR, username, "inspection_log.json")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    @classmethod
    def for_user(cls, username: str) -> "InspectionLogger":
        with cls._lock:
            if username not in cls._instances:
                cls._instances[username] = cls(username)
            return cls._instances[username]

    def log(self, job_name: str, status: str, detail: str = "",
            device_id: str = "", action: str = "", duration_ms: int = 0):
        """写入一条巡检记录"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "job": job_name,
            "status": status,       # started | success | failed | skipped
            "detail": detail,
            "device_id": device_id,
            "action": action,
            "duration_ms": duration_ms,
        }
        try:
            logs = []
            if os.path.exists(self.log_path):
                with open(self.log_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            logs.append(entry)
            # 只保留最近500条
            if len(logs) > 500:
                logs = logs[-500:]
            tmp_path = self.log_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.log_path)
        except Exception as e:
            logger.warning("巡检日志写入失败: %s", e)

    def get_recent(self, job_name: str = None, limit: int = 20) -> list:
        """读取最近的巡检记录"""
        try:
            if os.path.exists(self.log_path):
                with open(self.log_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
                if job_name:
                    logs = [l for l in logs if l.get("job") == job_name]
                return logs[-limit:]
        except Exception:
            pass
        return []


# ── 统一的巡检日志输出 ──

def _log_inspection(username: str, job_name: str, status: str, detail: str = "",
                    device_id: str = "", action: str = "", duration_ms: int = 0):
    """同时输出到控制台和磁盘"""
    icon = {"started": "▶", "success": "✓", "failed": "✗", "skipped": "○"}.get(status, "?")
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{icon}] [{job_name}] {detail}"
    if device_id:
        line += f" | 设备={device_id}"
    if action:
        line += f" | 动作={action}"
    if duration_ms:
        line += f" | 耗时={duration_ms}ms"
    logger.info(line)
    InspectionLogger.for_user(username).log(job_name, status, detail, device_id, action, duration_ms)


def check_reminders_job():
    """每 5 分钟检查到期提醒并发送 SMS"""
    try:
        from core.reminder_scheduler import ReminderScheduler
        import os, json

        sched = ReminderScheduler()
        # 从 user_profile 读取手机号
        phone = ""
        profile_path = os.path.join(DEFAULT_DATA_DIR, "user_profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, encoding="utf-8") as f:
                profile = json.load(f)
                phone = profile.get("user_phone", "")

        fired = sched.check_and_fire(phone=phone)
        if fired:
            logger.info("提醒调度: 触发 %d 条", len(fired))
    except Exception as e:
        logger.warning("提醒检查失败: %s", e)


def _disease_usernames():
    from core.database.repository.users import UserRepository
    return [u.username for u in UserRepository().get_all() if u.username]


def check_weather_job():
    """哥哥，按用户地块获取天气，不写共享缓存或发送短信。"""
    for username in _disease_usernames():
        try:
            _weather_by_plot(username)
        except Exception as exc:
            logger.warning("地块天气检查失败 [%s]: %s", username, exc)


def check_disease_job():
    """哥哥，每个地块独立评估并覆盖旧风险，天气缺失时清空旧结论。"""
    from pathlib import Path
    from core.plot_manager import PlotManager
    from core.disease_risk import assess_disease_risk
    from core.device_registry_factory import _validate_username
    for username in _disease_usernames():
        try:
            _validate_username(username)
            plots = PlotManager(username).list_plots()
            weather = _weather_by_plot(username)
            for plot in plots:
                plot_id = str(plot.get("plot_id", ""))
                if not plot_id.isdecimal():
                    logger.warning("跳过无效地块 ID [%s/%s]", username, plot_id)
                    continue
                current = weather.get(plot_id, {})
                risks = []
                assessed = bool(plot.get("crop") and "temperature" in current and "humidity" in current)
                if assessed:
                    risks = assess_disease_risk(plot["crop"], temperature=current["temperature"], humidity=current["humidity"])
                directory = Path(DEFAULT_DATA_DIR) / username / "disease_risks"
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / f"{plot_id}.json"
                tmp = target.with_suffix(".tmp")
                tmp.write_text(json.dumps({"updated": datetime.now().isoformat(), "plot_id": plot_id, "risks": risks, "quality": "good" if assessed else "unavailable"}, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, target)
        except Exception as exc:
            logger.warning("病害检查失败 [%s]: %s", username, exc)


def _weather_by_plot(username):
    """哥哥，只用用户自己地块的有效坐标和真实天气。"""
    import math
    from core.plot_manager import PlotManager
    from core.weather_service import WeatherService
    service = WeatherService()
    result = {}
    for plot in PlotManager(username).list_plots():
        try:
            lat, lon = float(plot.get("lat")), float(plot.get("lon"))
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
                continue
            raw = service.get_grid_weather(lon, lat)
            if not isinstance(raw, dict):
                continue
            observed = datetime.fromisoformat(str(raw.get("obsTime", "")).replace("Z", "+00:00"))
            current_time = datetime.now().astimezone()
            if observed.tzinfo is None:
                observed = observed.astimezone()
            if not 0 <= (current_time - observed).total_seconds() <= 3600:
                continue
            values = {}
            for key, source in (("temperature", "temp"), ("humidity", "humidity"), ("precipitation", "precip")):
                if source in raw:
                    number = float(raw[source])
                    limits = {"temperature": (-60, 100), "humidity": (0, 100), "precipitation": (0, 1000)}
                    if math.isfinite(number) and limits[key][0] <= number <= limits[key][1]:
                        values[key] = number
            if values:
                result[str(plot["plot_id"])] = values
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            logger.warning("地块天气无效 [%s/%s]: %s", username, plot.get("plot_id"), exc)
    return result


def _collect_device_rule_context(username, registry, loop):
    """哥哥，保留每个设备的读数，交给规则引擎按地块隔离。"""
    from core.device_registry_factory import load_custom_devices
    from core.observation_quality import validate_sensor_state

    configs = load_custom_devices(username)
    states = {}
    devices = loop.run_until_complete(registry.discover_all())
    for device in devices:
        try:
            state = loop.run_until_complete(registry.read_state(device.device_id))
            if isinstance(state, dict):
                valid, issues = validate_sensor_state(state)
                # 哥哥，状态和质量留给执行安全检查，规则只看合格数值。
                retained = {k: v for k, v in state.items() if k.startswith("_") or k in ("status", "power", "error")}
                retained.update(valid)
                if issues:
                    retained["_observation_issues"] = issues
                states[device.device_id] = retained
        except Exception as exc:
            logger.warning("读取设备状态失败 [%s/%s]: %s", username, device.device_id, exc)
    try:
        weather = _weather_by_plot(username)
    except Exception as exc:
        logger.warning("读取地块天气失败 [%s]: %s", username, exc)
        weather = {}
    return {"device_configs": configs, "device_states": states, "weather_by_plot": weather}


def check_device_rules_job():
    """每 5 分钟检查自动规则并触发设备操作"""
    import time as _time
    job_start = _time.time()
    total_triggered = 0
    try:
        import os
        from core.device_rule_engine import (
            RuleDecision, RuleEngine, apply_autonomy,
        )
        from devices.base import DeviceCommand
        from core.device_executor import DeviceExecutor
        from core.device_registry_factory import setup_registry, close_registry
        from app.agent.config import get_autonomy_level

        # 遍历数据库中真正拥有规则的用户，并兼容旧文件目录。
        data_dir = DEFAULT_DATA_DIR
        usernames = {"default"}
        try:
            from core.database.engine import get_session
            from core.database.models import DeviceRule, User

            session = get_session()
            try:
                rows = (
                    session.query(User.username)
                    .join(DeviceRule, DeviceRule.user_id == User.id)
                    .distinct()
                    .all()
                )
                usernames.update(row[0] for row in rows if row[0])
            finally:
                session.close()
        except Exception as exc:
            logger.warning("从数据库发现规则用户失败: %s", exc)
        if os.path.exists(data_dir):
            for d in os.listdir(data_dir):
                user_path = os.path.join(data_dir, d)
                if os.path.isdir(user_path) and os.path.exists(os.path.join(user_path, "device_rules.json")):
                    usernames.add(d)

        for username in usernames:
            engine = RuleEngine(username=username)
            if not any(rule.get("enabled", True) for rule in engine.rules):
                continue

            # 使用 setup_registry 加载用户设备并读取传感器数据
            registry, loop = setup_registry(username)
            try:
                context = _collect_device_rule_context(username, registry, loop)
                matched = engine.find_matching_rules(context)

                if not matched:
                    continue

                executor = DeviceExecutor(registry, username=username)
                autonomy = get_autonomy_level(username)

                for rule in matched:
                    action = rule.get("action", {})
                    proposed = action.get("params", {})
                    rule_context = engine.context_for_device(
                        action.get("device_id", ""), context
                    )
                    decision, reason, final_params = engine.evaluate_action(
                        rule, proposed, rule_context)
                    decision = apply_autonomy(decision, autonomy)
                    execution_mode = rule.get("execution_mode", "auto")

                    if execution_mode == "notify":
                        _log_inspection(
                            username, "设备规则", "skipped",
                            f"规则提醒: {rule.get('name', '未命名')} — {reason}",
                            device_id=action.get("device_id", ""),
                            action=action.get("command", "start"),
                        )
                        continue
                    if (execution_mode == "confirm"
                            and decision != RuleDecision.REJECTED):
                        decision = RuleDecision.NEED_CONFIRM
                        reason = f"规则「{rule.get('name', '未命名')}」设置为执行前确认"

                    cmd = DeviceCommand(
                        command=action.get("command", "start"),
                        params=final_params,
                    )
                    if decision == RuleDecision.AUTO_EXECUTE:
                        if cmd.command == "start":
                            # 哥哥，周期规则不替运行中的设备续期，手动启动仍可重设计时。
                            try:
                                current_state = loop.run_until_complete(
                                    registry.read_state(action.get("device_id"))
                                )
                            except Exception as exc:
                                logger.warning("读取自动启动目标状态失败，跳过本轮: %s", exc)
                                continue
                            if current_state.get("error") or current_state.get("status") == "running":
                                continue
                        result = executor.execute_sync(
                            action.get("device_id"), cmd,
                            trigger="rule", rule_id=rule["id"], decision=decision,
                            loop=loop, capability=action.get("capability"),
                            policy_context=rule_context,
                        )

                        if result["success"]:
                            engine.record_execution(action.get("device_id"), final_params)
                            total_triggered += 1
                            _log_inspection(username, "设备规则", "success",
                                            f"规则触发: {rule.get('name', '未命名')}",
                                            device_id=action.get("device_id", ""),
                                            action=action.get("command", "start"))
                    else:
                        if decision == RuleDecision.NEED_CONFIRM and any(
                            str(pending.get("rule_id")) == str(rule.get("id"))
                            and pending.get("device_id") == action.get("device_id")
                            and pending.get("command") == cmd.command
                            and pending.get("params") == final_params
                            for pending in executor.list_pending()
                        ):
                            continue
                        executor.record_decision(
                            action.get("device_id", ""), cmd,
                            decision=decision, reason=reason,
                            trigger="rule", rule_id=rule.get("id"),
                            add_pending=decision == RuleDecision.NEED_CONFIRM,
                            capability=action.get("capability"),
                            policy_context=rule_context,
                        )
            finally:
                close_registry(loop, registry)

        elapsed = int((_time.time() - job_start) * 1000)
        if total_triggered > 0:
            _log_inspection("system", "设备规则", "success",
                            f"触发 {total_triggered} 条规则", duration_ms=elapsed)

    except Exception as e:
        elapsed = int((_time.time() - job_start) * 1000)
        _log_inspection("system", "设备规则", "failed",
                        f"轮询异常: {e}", duration_ms=elapsed)


def check_task_execution_job():
    """每 3 分钟检查待执行的设备任务，匹配启用规则后自动执行"""
    try:
        from core.planting_tracker import PlantingTracker
        from core.device_rule_engine import RuleDecision, RuleEngine, apply_autonomy
        from devices.base import DeviceCommand
        from core.device_executor import DeviceExecutor
        from core.device_registry_factory import setup_registry, close_registry
        from core.database.engine import get_session
        from core.database.models import PlantingTask, User
        from app.agent.config import get_autonomy_level

        data_dir = DEFAULT_DATA_DIR
        usernames = {"default"}
        with get_session() as session:
            rows = (
                session.query(User.username)
                .join(PlantingTask, PlantingTask.user_id == User.id)
                .filter(PlantingTask.device_id.isnot(None))
                .distinct()
                .all()
            )
            usernames.update(row[0] for row in rows if row[0])
        if os.path.exists(data_dir):
            for d in os.listdir(data_dir):
                user_path = os.path.join(data_dir, d)
                if os.path.isdir(user_path) and os.path.exists(os.path.join(user_path, "planting_tasks.json")):
                    usernames.add(d)

        for username in usernames:
            sd = os.path.join(DEFAULT_DATA_DIR, username)
            tracker = PlantingTracker(sd)
            pending_tasks = tracker.get_pending_device_tasks()

            if not pending_tasks:
                continue

            engine = RuleEngine(username=username)
            registry, loop = setup_registry(username)
            try:
                context = _collect_device_rule_context(username, registry, loop)
                matched = engine.find_matching_rules(context)
                executor = DeviceExecutor(registry, username=username)
                autonomy = get_autonomy_level(username)

                for task in pending_tasks:
                    # 哥哥，尚未到期的任务和已过截止日期的任务都不能提前下发。
                    try:
                        today = datetime.now().date()
                        if task.start_date and datetime.strptime(task.start_date, "%Y-%m-%d").date() > today:
                            continue
                        if task.end_date and datetime.strptime(task.end_date, "%Y-%m-%d").date() < today:
                            tracker.update_task_status(task.id, "已逾期")
                            continue
                    except (ValueError, TypeError):
                        logger.warning("设备任务日期无效，跳过执行: %s", task.id)
                        continue

                    task_context = {
                        **engine.context_for_device(task.device_id, context),
                        "task_id": task.id,
                    }
                    matching_rule = None
                    final_params = task.device_params or {}
                    for rule in matched:
                        rule_action = rule.get("action", {})
                        if (rule.get("execution_mode", "auto") != "auto"
                                or rule_action.get("device_id") != task.device_id
                                or rule_action.get("command", "start") != task.device_command):
                            continue
                        decision, _, evaluated_params = engine.evaluate_action(
                            rule, task.device_params or {}, task_context
                        )
                        if apply_autonomy(decision, autonomy) == RuleDecision.AUTO_EXECUTE:
                            matching_rule = rule
                            final_params = evaluated_params
                            break
                    if not matching_rule:
                        continue
                    task_execution_lock = tracker.device_task_execution_lock(task.id)
                    if not task_execution_lock.acquire():
                        continue
                    try:
                        if not tracker.claim_device_task(task.id):
                            continue
                        # 哥哥，领取后再检查待确认队列，避免与手动执行并发重复下发。
                        if any(
                            str((action.get("policy_context") or {}).get("task_id")) == task.id
                            for action in executor.list_pending()
                        ):
                            tracker.update_task_status(task.id, "待办", progress=0)
                            continue
                        cmd = DeviceCommand(
                            command=task.device_command,
                            params=final_params,
                        )
                        result = executor.execute_sync(
                            task.device_id, cmd,
                            trigger="task_scheduler",
                            rule_id=matching_rule.get("id"),
                            loop=loop,
                            capability=matching_rule.get("action", {}).get("capability"),
                            policy_context=task_context,
                        )
                        if result["success"]:
                            engine.record_execution(task.device_id, final_params)
                            tracker.record_device_task_result(task.id, result, registry, loop)
                            logger.info("自动执行任务成功: %s → %s", task.title, task.device_id)
                        else:
                            tracker.record_device_task_result(task.id, result, registry, loop)
                            # 哥哥，目标设备已移除时终止此任务，不能恢复设备后悄悄自动重发。
                            if not any(c.get("device_id") == task.device_id for c in context["device_configs"]):
                                tracker.update_task_status(task.id, "失败", progress=0)
                            logger.warning("自动执行任务失败: %s → %s", task.title, task.device_id)
                    except Exception as e:
                        logger.warning("自动执行任务异常: %s → %s", task.title, e)
                        tracker.update_task_status(task.id, "失败", progress=0)
                    finally:
                        task_execution_lock.release()
            finally:
                close_registry(loop, registry)

    except Exception as e:
        logger.warning("任务自动执行检查失败: %s", e)


# ── 摄像头定时巡检 ──────────────────────────────────

def _get_int_env(key: str, default: int) -> int:
    """安全读取整数环境变量，解析失败时返回默认值"""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        logger.warning("环境变量 %s 值非法，使用默认值 %d", key, default)
        return default

CHECK_CAMERA_INTERVAL_MINUTES = _get_int_env("CAMERA_CHECK_INTERVAL", 30)


def _persist_capture_if_needed(raw, image_bytes, username, device_id, timestamp):
    """复用驱动已经保存的照片，只有远程驱动未落盘时才补写。"""
    saved_path = raw.get("saved_path")
    if saved_path and os.path.isfile(saved_path):
        return os.path.dirname(saved_path), saved_path

    photo_dir = os.path.join(
        DEFAULT_DATA_DIR, username, "photos", device_id)
    os.makedirs(photo_dir, exist_ok=True)
    photo_path = os.path.join(photo_dir, f"capture_{timestamp}.jpg")
    with open(photo_path, "wb") as file:
        file.write(image_bytes)
    return photo_dir, photo_path


def check_camera_capture_job():
    """每 N 分钟自动拍照 + Vision 分析 + 自主执行

    流程:
    1. 遍历所有用户，发现摄像头设备
    2. 逐个摄像头拍照
    3. 调用 CropMonitorAgent 分析照片
    4. 如有推荐操作，通过 RuleEngine + DeviceExecutor 执行
    """
    import time as _time
    job_start = _time.time()
    try:
        import os
        import json
        import base64
        from datetime import datetime
        from core.device_registry_factory import setup_registry, close_registry
        from devices.base import DeviceCommand

        data_dir = DEFAULT_DATA_DIR
        usernames = ["default"]
        if os.path.exists(data_dir):
            for d in os.listdir(data_dir):
                user_path = os.path.join(data_dir, d)
                if os.path.isdir(user_path) and os.path.exists(
                    os.path.join(user_path, "custom_devices.json")
                ):
                    usernames.append(d)

        total_cameras = 0
        total_captures = 0

        for username in set(usernames):
            registry, loop = setup_registry(username)
            try:
                devices = loop.run_until_complete(registry.discover_all())
                cameras = [
                    d for d in devices
                    if "capture" in [c.value for c in d.capabilities]
                    and d.status.value in ("online", "offline")
                ]
                error_cameras = [
                    d for d in devices
                    if "capture" in [c.value for c in d.capabilities]
                    and d.status.value == "error"
                ]
                for ec in error_cameras:
                    _log_inspection(username, "摄像头巡检", "failed",
                                    f"摄像头故障: {ec.device_id}", device_id=ec.device_id)

                if not cameras:
                    continue

                total_cameras += len(cameras)
                _log_inspection(username, "摄像头巡检", "started",
                                f"发现 {len(cameras)} 个摄像头，开始巡检",
                                device_id=cameras[0].device_id if cameras else "")

                for cam in cameras:
                    try:
                        # 1. 拍摄照片
                        cmd = DeviceCommand(command="capture", params={}, timeout_ms=15000)
                        result = loop.run_until_complete(
                            registry.execute(cam.device_id, cmd)
                        )
                        if not result.success:
                            _log_inspection(username, "摄像头巡检", "failed",
                                            f"拍照失败: {result.message}", device_id=cam.device_id)
                            continue

                        # 兼容 CameraDriver(image_bytes) 和 HTTP 设备(image_base64)
                        raw = result.raw_response or {}
                        image_bytes = raw.get("image_bytes")
                        if not image_bytes:
                            image_b64 = raw.get("image_base64", "")
                            if image_b64:
                                image_bytes = base64.b64decode(image_b64)
                        if not image_bytes:
                            logger.warning("摄像头未返回图片数据: %s", cam.device_id)
                            continue

                        # 2. 驱动已保存时直接复用；远程驱动未保存时才在此落盘。
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        photo_dir, photo_path = _persist_capture_if_needed(
                            raw, image_bytes, username, cam.device_id, ts)
                        total_captures += 1
                        _log_inspection(username, "摄像头巡检", "success",
                                        f"拍照成功 ({len(image_bytes)} bytes)", device_id=cam.device_id)

                        # 3. base64 编码 → Vision 分析
                        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                        from app.agent.agents.crop_monitor_agent import CropMonitorAgent
                        monitor = CropMonitorAgent()
                        analysis_result = monitor.analyze_image(
                            image_b64, "image/jpeg",
                            user_context={
                                "username": username,
                                "device_id": cam.device_id,
                                "location": cam.location,
                            },
                        )

                        # 4. 存储分析结果
                        analysis_path = os.path.join(
                            photo_dir, f"analysis_{ts}.json"
                        )
                        with open(analysis_path, "w", encoding="utf-8") as f:
                            json.dump({
                                "timestamp": datetime.now().isoformat(),
                                "device_id": cam.device_id,
                                "photo_path": photo_path,
                                "analysis": analysis_result.get("analysis", {}),
                                "error": analysis_result.get("error"),
                            }, f, ensure_ascii=False, indent=2)

                        # 5. 如有推荐操作 → 自动执行
                        if analysis_result.get("success"):
                            analysis = analysis_result.get("analysis", {})
                            issues = analysis.get("issues", [])
                            actions = analysis.get("recommended_actions", [])
                            health = analysis.get("health_status", "unknown")
                            _log_inspection(username, "摄像头巡检", "success",
                                            f"分析完成: 健康={health}, 问题={len(issues)}个, 建议操作={len(actions)}个",
                                            device_id=cam.device_id)
                            if actions:
                                _process_camera_actions(
                                    actions, username, registry, loop, cam.device_id
                                )
                        else:
                            _log_inspection(username, "摄像头巡检", "failed",
                                            f"分析失败: {analysis_result.get('error', '未知')}",
                                            device_id=cam.device_id)

                    except Exception as e:
                        _log_inspection(username, "摄像头巡检", "failed",
                                        f"异常: {e}", device_id=cam.device_id)
            finally:
                close_registry(loop, registry)

        elapsed = int((_time.time() - job_start) * 1000)
        _log_inspection("system", "摄像头巡检", "success",
                        f"巡检完成: {total_captures}/{total_cameras} 台摄像头拍照成功",
                        duration_ms=elapsed)

    except Exception as e:
        elapsed = int((_time.time() - job_start) * 1000)
        _log_inspection("system", "摄像头巡检", "failed",
                        f"巡检异常: {e}", duration_ms=elapsed)


def _process_camera_actions(actions: list, username: str, registry, loop, camera_id: str):
    """处理 Vision 分析推荐的设备操作

    通过 RuleEngine 评估，尊重自主权设置，满足条件则自动执行。
    """
    try:
        from core.device_rule_engine import RuleEngine, RuleDecision, apply_autonomy
        from devices.base import DeviceCommand
        from core.device_executor import DeviceExecutor
        from app.agent.config import get_autonomy_level

        engine = RuleEngine(username=username)
        executor = DeviceExecutor(registry, username=username)
        autonomy = get_autonomy_level(username)

        # 哥哥，摄像头和执行设备必须明确绑定同一地块。
        all_devices = loop.run_until_complete(registry.discover_all())
        context = _collect_device_rule_context(username, registry, loop)
        configs = {item.get("device_id"): item for item in context["device_configs"]}
        camera_plot = configs.get(camera_id, {}).get("plot_id")
        if camera_plot in (None, ""):
            logger.warning("摄像头未绑定地块，拒绝自动设备动作: %s", camera_id)
            return

        for action in actions:
            action_type = action.get("action", "")
            urgency = action.get("urgency", "routine")
            detail = action.get("detail", "")

            if action_type in ("none",):
                continue
            if action_type == "alert":
                logger.warning(
                    "摄像头告警 [%s/%s]: %s (紧急度: %s)", username, camera_id, detail, urgency
                )
                continue

            # 映射 action → 能力
            capability_map = {"irrigate": "irrigate", "fertigate": "fertigate"}
            capability = capability_map.get(action_type)
            if not capability:
                continue

            # 找到匹配的执行设备
            target_device = None
            for d in all_devices:
                if str(configs.get(d.device_id, {}).get("plot_id") or "") == str(camera_plot) and capability in [c.value for c in d.capabilities]:
                    target_device = d.device_id
                    break
            if not target_device:
                logger.info("未找到 %s 类型执行设备，跳过", action_type)
                continue

            # 哥哥，沿用按地块过滤、校验新鲜度的采集结果。
            matched = [rule for rule in engine.find_matching_rules(context)
                       if rule.get("action", {}).get("device_id") == target_device]
            target_context = engine.context_for_device(target_device, context)
            duration = 15  # 默认时长

            if matched:
                rule = matched[0]
                proposed = rule.get("action", {}).get("params", {"duration": duration})
                decision, reason, final_params = engine.evaluate_action(
                    rule, proposed, target_context
                )
                decision = apply_autonomy(decision, autonomy)

                if decision == RuleDecision.AUTO_EXECUTE:
                    cmd = DeviceCommand(command="start", params=final_params)
                    result = executor.execute_sync(
                        target_device, cmd, trigger="camera", rule_id=rule["id"], policy_context=target_context
                    )
                    if result["success"]:
                        engine.record_execution(target_device, final_params)
                        logger.info(
                            "摄像头触发操作: %s → %s (%s, 自主权=%s)",
                            target_device, action_type, detail, autonomy,
                        )
                    else:
                        logger.warning("摄像头触发操作失败: %s", target_device)
            else:
                # 无匹配规则 → 仅在 high 自主权下执行
                if autonomy == "high" and urgency in ("immediate", "today"):
                    cmd = DeviceCommand(command="start", params={"duration": duration})
                    result = executor.execute_sync(
                        target_device, cmd, trigger="camera", policy_context=target_context
                    )
                    if result["success"]:
                        logger.info(
                            "摄像头自主操作 (high): %s → %s", target_device, action_type
                        )

    except Exception as e:
        logger.warning("处理摄像头推荐操作失败: %s", e)


# ── 自主决策巡检（替代原 check_camera_capture_job）──

_autonomous_last_runs: Dict[str, datetime] = {}
_autonomous_run_lock = threading.Lock()


def _claim_autonomous_cycle(username: str, region: str,
                            min_interval_minutes: int,
                            now: datetime = None) -> bool:
    """原子领取巡检周期，并通过数据库租约阻止跨进程重复执行。"""
    current = now or datetime.now()
    key = f"{username}:{region}"
    with _autonomous_run_lock:
        last_run = _autonomous_last_runs.get(key)
        if last_run:
            elapsed = (current - last_run).total_seconds()
            if elapsed < max(0, min_interval_minutes) * 60:
                return False
        try:
            from core.autonomous_cycle_lease import claim_cycle_lease
            if not claim_cycle_lease(
                username, region, min_interval_minutes, current
            ):
                return False
        except Exception:
            logger.exception("巡检数据库租约领取失败，按安全原则拒绝执行")
            return False
        # 数据库租约成功后再写进程缓存，减少同进程数据库访问。
        _autonomous_last_runs[key] = current
        return True


def _get_active_usernames() -> List[str]:
    """从数据库发现拥有设备或地块的活跃用户，并兼容旧文件数据。"""
    usernames = {"default"}

    # 当前业务数据以 SQLite 为准，不能再只依赖旧 JSON 文件判断活跃用户
    try:
        from sqlalchemy import or_
        from core.database.engine import get_session
        from core.database.models import User, DeviceConfig, Field

        session = get_session()
        try:
            rows = (
                session.query(User.username)
                .outerjoin(DeviceConfig, DeviceConfig.user_id == User.id)
                .outerjoin(Field, Field.user_id == User.id)
                .filter(or_(DeviceConfig.id.isnot(None), Field.id.isnot(None)))
                .distinct()
                .all()
            )
            usernames.update(row[0] for row in rows if row[0])
        finally:
            session.close()
    except Exception as e:
        logger.warning("从数据库发现活跃用户失败，回退旧文件扫描: %s", e)

    # 兼容尚未迁移的旧数据目录
    data_dir = DEFAULT_DATA_DIR
    if os.path.exists(data_dir):
        for d in os.listdir(data_dir):
            user_path = os.path.join(data_dir, d)
            if os.path.isdir(user_path):
                if os.path.exists(os.path.join(user_path, "custom_devices.json")) or \
                   os.path.exists(os.path.join(user_path, "planting_progress.json")):
                    usernames.add(d)
    return sorted(usernames)


def check_autonomous_cycle_job():
    """自主决策定时巡检：发现区域 → 数据采集 → LLM决策 → 执行

    替代原有的 check_camera_capture_job，实现完整的感知→决策→执行闭环。
    """
    import time as _time
    job_start = _time.time()
    total_cycles = 0
    try:
        from core.autonomous_farm_manager import AutonomousFarmManager
        from app.agent.config import (
            AUTO_DECISION_REGIONS, AUTO_DECISION_MIN_INTERVAL,
        )

        usernames = _get_active_usernames()
        configured_regions = [r.strip() for r in AUTO_DECISION_REGIONS.split(",") if r.strip()]

        _log_inspection("system", "自主决策", "started",
                        f"开始巡检: {len(usernames)} 用户, {len(configured_regions)} 区域")

        for username in usernames:
            manager = AutonomousFarmManager()

            # 发现该用户的巡检区域 — 优先使用地块(独立坐标+天气)
            try:
                from core.plot_manager import PlotManager
                pm = PlotManager(username)
                plots = pm.list_plots()
                if configured_regions:
                    configured = set(configured_regions)
                    plots = [
                        p for p in plots
                        if p["plot_id"] in configured or p.get("name") in configured
                    ]

                if plots:
                    regions = [p["plot_id"] for p in plots]
                    _log_inspection(username, "自主决策", "started",
                                    f"按地块巡检: {len(plots)} 个地块")
                else:
                    # 无地块 → 回退：按设备 location 分组
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
                        close_registry(loop, registry)
            except Exception as e:
                logger.warning("区域发现失败 [%s]: %s", username, e)
                continue

            if not regions:
                logger.debug("用户 %s 无可用区域", username)
                continue

            for region in regions:
                # 检查最小间隔
                if not _claim_autonomous_cycle(
                        username, region, AUTO_DECISION_MIN_INTERVAL):
                    logger.debug("区域 %s 距上次巡检不足%d分钟，跳过", region, AUTO_DECISION_MIN_INTERVAL)
                    continue

                try:
                    logger.info("自主决策巡检: %s/%s", username, region)
                    report = manager.run_cycle(username, region)
                    total_cycles += 1
                    _log_inspection(username, "自主决策", "success",
                                    f"区域={region}, 摘要={report.summary.replace(chr(10), ' | ')[:120]}")
                except Exception as e:
                    _log_inspection(username, "自主决策", "failed",
                                    f"区域={region}, 异常={e}")

        elapsed = int((_time.time() - job_start) * 1000)
        _log_inspection("system", "自主决策", "success",
                        f"巡检完成: {total_cycles} 个周期", duration_ms=elapsed)

    except Exception as e:
        elapsed = int((_time.time() - job_start) * 1000)
        _log_inspection("system", "自主决策", "failed",
                        f"调度异常: {e}", duration_ms=elapsed)
