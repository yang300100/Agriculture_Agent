"""哥哥，让自主维护的计划、复查和通知都有持久、可核对的结果。"""

import copy
import json
import math
import os
import urllib.request
from datetime import datetime, timedelta

from core.autonomy_store import (
    claim_alerts, claim_jobs, enqueue, finish_alert, finish_job, key_for,
    list_rows, plain, save_alert, serialize, timestamp,
)
from core.database.engine import get_session
from core.database.models import AutonomyAlert, AutonomyJob, SchedulerHeartbeat


def schedule_action(username, region, action, due_at=None):
    action = copy.deepcopy(plain(action))
    action.setdefault("execution_id", key_for(username, region, action, str(due_at)))
    return enqueue(username, region, "planned_action", {"action": action}, due_at)


def raise_alert(username, region, message, *, dedup_key=None, severity="warning"):
    return save_alert(username, region, message, dedup_key, severity)


def list_jobs(username, limit=100):
    return list_rows(AutonomyJob, username, limit)


def list_alerts(username, limit=100):
    return list_rows(AutonomyAlert, username, limit)


def arrange_job(username, job_id, due_at):
    """哥哥，只安排尚未下发的动作，不能借排期重试未知作业。"""
    due = timestamp(due_at)
    if due <= datetime.now():
        raise ValueError("请选择未来的执行时间")
    with get_session() as session:
        row = session.query(AutonomyJob).filter_by(id=job_id, username=username).first()
        if row is None:
            raise LookupError("巡检任务不存在")
        if row.kind != "planned_action" or row.status != "awaiting_schedule":
            raise ValueError("只有待安排的动作可以设置执行时间")
        action = json.loads(row.payload).get("action") or {}
        if action.get("action") == "review":
            raise ValueError("复查建议需要先制定明确操作与判据")
        changed = session.query(AutonomyJob).filter_by(id=job_id, username=username,
            status="awaiting_schedule").update({"status": "pending", "due_at": due,
            "updated_at": datetime.now()}, synchronize_session=False)
        if not changed:
            raise ValueError("任务状态已变化，请刷新后重试")
        session.commit()
        session.expire_all()
        return serialize(session.get(AutonomyJob, job_id))


def acknowledge_alert(username, alert_id):
    with get_session() as session:
        updated = session.query(AutonomyAlert).filter_by(username=username, id=alert_id).update({
            "status": "acknowledged", "acknowledged_at": datetime.now(),
            "lease_until": None, "updated_at": datetime.now(),
        }, synchronize_session=False)
        session.commit()
        return bool(updated)


def record_heartbeat(status="running", detail="", now=None):
    now = timestamp(now)
    with get_session() as session:
        row = session.get(SchedulerHeartbeat, "scheduler")
        if row is None:
            row = SchedulerHeartbeat(id="scheduler")
            session.add(row)
        row.status, row.detail = status, str(detail)[:2000]
        row.last_seen_at, row.process_id = now, os.getpid()
        session.commit()


def runtime_status(now=None, max_age_seconds=60):
    from core.autonomy_health import maintenance_status
    maintenance = maintenance_status()
    now = timestamp(now)
    with get_session() as session:
        row = session.get(SchedulerHeartbeat, "scheduler")
        if row is None:
            return {**maintenance, "running": False, "heartbeat_status": "missing", "last_heartbeat": None, "detail": "尚未收到调度进程心跳"}
        age = (now - row.last_seen_at).total_seconds()
        alive = row.status == "running" and 0 <= age <= max_age_seconds
        return {**maintenance, "running": alive, "heartbeat_status": row.status if alive or row.status == "stopped" else "stale",
                "last_heartbeat": row.last_seen_at.isoformat(), "heartbeat_age_seconds": max(0, round(age, 1)),
                "detail": row.detail or ("调度进程运行中" if alive else "调度进程未运行或心跳过期")}


def queue_effect_check(username, region, action, result, baseline):
    action, result, baseline = copy.deepcopy(plain(action)), plain(result), plain(baseline)
    if result.get("success") is not True:
        return None
    from core.device_registry_factory import load_custom_devices
    config = next((item for item in load_custom_devices(username) if item.get("device_id") == action.get("device_id")), {})
    feedback = (config.get("connection") or {}).get("feedback") or {}
    params = result.get("executed_params") or action.get("params") or {}
    try:
        duration = float(params.get("duration", params.get("duration_minutes", 0)) or 0)
        settle = float(feedback.get("settle_seconds", 30))
        if not math.isfinite(duration) or not math.isfinite(settle) or duration < 0 or settle < 0:
            raise ValueError("复查时间无效")
    except (TypeError, ValueError, AttributeError, OverflowError):
        feedback = {"configuration_error": "维护效果复查配置无效，需要人工复核"}
        duration, settle = 0, 0
    execution_id = result.get("execution_id") or action.get("execution_id")
    payload = {"action": action, "execution_id": execution_id, "baseline": baseline,
               "feedback": feedback, "acknowledged": result.get("acknowledged", False)}
    due_at = datetime.now() + timedelta(seconds=min(86400, duration * 60) + min(3600, settle))
    key = key_for("effect", region, action.get("device_id"), execution_id or payload)
    return enqueue(username, region, "effect_check", payload, due_at, key)


def _run_planned_action(job):
    from core.autonomous_farm_manager import AutonomousFarmManager, DecisionPlan
    manager = AutonomousFarmManager()
    action = copy.deepcopy(job["payload"]["action"])
    # 哥哥，依赖已由工作队列核对；单动作执行器不再按本批索引重复判断。
    action.pop("depends_on", None)
    action.pop("dependency_runs", None)
    if action.get("action") == "review":
        raise_alert(job["username"], job["region"], action.get("reason") or "巡检后续工作需要安排", dedup_key=f"review:{job['id']}")
        return {"success": False, "terminal": True, "status": "unverified", "error": "后续复查需要明确判据与时间"}
    action["urgency"] = "immediate"
    state = manager.collect_farm_state(job["username"], job["region"])
    context = {**manager._build_rule_context(state, job["username"]), "plot_id": job["region"]}
    if action.get("rule_id") is not None:
        from core.device_rule_engine import RuleEngine
        matched = RuleEngine(job["username"]).find_matching_rules(context)
        if not any(str(rule.get("id")) == str(action["rule_id"]) for rule in matched):
            return {"success": False, "terminal": True, "error": "排期规则当前条件已不满足或观测不可用，未启动设备"}
    plan = manager.validate_plan(DecisionPlan(region=job["region"], actions=[action]),
                                 available_devices=state.available_devices,
                                 available_capabilities=manager._get_available_capabilities(state))
    if not plan.actions:
        return {"success": False, "terminal": True, "error": "排期动作重新校验未通过"}
    manager._execution_policy_context = context
    manager._cycle_baseline = state
    try:
        results = manager.execute_plan(plan, job["username"])
    finally:
        manager._execution_policy_context = {}
    if results and results[0].success:
        queue_effect_check(job["username"], job["region"], action, results[0], state)
        return {"success": True, "message": results[0].message}
    # 哥哥，执行结果不明不得由队列再次下发，由设备门禁及核对入口处置。
    return {"success": False, "terminal": True, "error": results[0].message if results else "动作未执行"}


def _check_dependencies(job):
    """哥哥，启动回执只代表开始，后续作业必须等同编号的完成记录。"""
    from core.device_registry_factory import RegistrySession, load_custom_devices
    from core.observation_quality import validate_sensor_state
    action = job["payload"].get("action") or {}
    runs = action.get("dependency_runs")
    def blocked(message):
        return {"success": False, "terminal": True, "status": "unverified", "error": message}
    if not runs:
        return blocked("前置作业缺少可核验的执行编号，需人工复核") if action.get("depends_on") else None
    if not isinstance(runs, list) or not all(isinstance(run, dict) and run.get("device_id") and run.get("execution_id") for run in runs):
        return blocked("前置作业的完成记录引用无效")
    configs = {item.get("device_id"): item for item in load_custom_devices(job["username"])}
    if any(str(configs.get(run["device_id"], {}).get("plot_id")) != str(job["region"]) for run in runs):
        return blocked("前置作业设备必须仍绑定本次地块")
    waiting = False
    with RegistrySession(job["username"]) as (registry, loop):
        loop.run_until_complete(registry.discover_all())
        for run in runs:
            state = loop.run_until_complete(registry.read_state(run["device_id"]))
            _, issues = validate_sensor_state(state)
            if issues:
                return blocked("前置作业状态无法核验：" + "；".join(issues))
            execution_id = run["execution_id"]
            history = state.get("run_history") or {}
            ended = history.get(execution_id, {}) if isinstance(history, dict) else {}
            if not ended and state.get("last_ended_run_id") == execution_id:
                ended = {"reason": state.get("last_run_end_reason")}
            if not ended and state.get("execution_id") == execution_id:
                ended = {"reason": state.get("execution_status")}
            if isinstance(ended, dict) and ended.get("reason") in {"completed", "success"}:
                continue
            if state.get("status") == "running" and (state.get("run_id") or state.get("execution_id")) == execution_id:
                waiting = True
                continue
            return blocked("前置作业缺少匹配编号的成功结束证据，需要人工复核")
    if waiting:
        now = datetime.now()
        if (now - timestamp(job["created_at"])).total_seconds() > 86400:
            return blocked("等待前置作业超时，需要检查设备")
        return {"success": False, "defer_until": (now + timedelta(seconds=30)).isoformat(), "error": "前置作业仍在运行，等待实际完成"}
    return None


def process_due_jobs(*, now=None, limit=20, handlers=None, username=None):
    now = timestamp(now)
    if handlers is None:
        from core.autonomy_feedback import run_effect_check
        handlers = {"planned_action": _run_planned_action, "effect_check": run_effect_check}
    outcomes = []
    for job in claim_jobs(now, max(1, min(int(limit), 100)), username):
        if job["status"] == "unverified":
            raise_alert(job["username"], job["region"], job["last_error"], dedup_key=f"job:{job['id']}")
            outcomes.append(job)
            continue
        try:
            handler = handlers.get(job["kind"])
            dependency = _check_dependencies(job) if job["kind"] == "planned_action" else None
            result = dependency or (handler(job) if handler else {"success": False, "terminal": True, "error": "未知维护任务类型"})
        except Exception as exc:
            result = {"success": False, "error": f"处理失败：{type(exc).__name__}"}
            if job["kind"] == "planned_action":
                result.update(terminal=True, status="unverified")
        saved = finish_job(job, result, now)
        outcomes.append(saved)
        if saved["status"] in {"failed", "unverified"}:
            raise_alert(job["username"], job["region"], saved["last_error"], dedup_key=f"job:{job['id']}")
    return outcomes


def _send_alert(alert):
    webhook = os.getenv("AUTONOMY_ALERT_WEBHOOK", "").strip()
    if not webhook:
        return {"success": False, "error": "未配置通知渠道；告警已保留，等待人工处理"}
    if not webhook.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
        return {"success": False, "error": "通知通道须使用HTTPS（本机测试地址除外）"}
    request = urllib.request.Request(webhook, data=json.dumps({
        "alert_id": alert["id"], "region": alert["region"], "severity": alert["severity"],
        "message": alert["message"], "created_at": alert["created_at"],
    }, ensure_ascii=False).encode("utf-8"), headers={
        "Content-Type": "application/json", "Idempotency-Key": f"autonomy-alert-{alert['id']}",
    })
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            if not 200 <= response.status < 300:
                return {"success": False, "error": f"通知通道返回HTTP {response.status}"}
            body = response.read(65537)
            if len(body) > 65536:
                return {"success": False, "error": "通知通道回执过大，无法确认送达"}
            if body:
                try:
                    receipt = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    receipt = None
                if isinstance(receipt, dict) and (receipt.get("success") is False or receipt.get("ok") is False):
                    return {"success": False, "error": "通知通道明确拒绝接收"}
            return {"success": True}
    except Exception as exc:
        return {"success": False, "error": f"通知未送达（{type(exc).__name__}）"}


def dispatch_alerts(*, now=None, sender=None, username=None):
    now = timestamp(now)
    sender = sender or _send_alert
    outcomes = []
    for alert in claim_alerts(now, username=username):
        try:
            result = sender(alert)
        except Exception as exc:
            result = {"success": False, "error": f"通知未送达（{type(exc).__name__}）"}
        outcomes.append(finish_alert(alert, result, now))
    return outcomes
