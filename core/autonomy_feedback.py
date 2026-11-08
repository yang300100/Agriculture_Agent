"""哥哥，维护结束后核对实际效果，未知结果不会被写成成功。"""

import math
from datetime import datetime

from core.autonomy_store import timestamp
from core.observation_quality import validate_sensor_state


def _unverified(message):
    return {"success": False, "terminal": True, "status": "unverified", "error": message}


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def evaluate_effect(payload, device_state, sensor_states, *, now=None):
    now = timestamp(now)
    feedback = payload.get("feedback") or {}
    if feedback.get("configuration_error"):
        return _unverified(feedback["configuration_error"])
    sensor_id, field = feedback.get("sensor_id"), feedback.get("field")
    if not sensor_id or not field:
        return _unverified("未配置维护效果的传感器和判据，需人工复核")
    _, device_issues = validate_sensor_state(device_state, now=now)
    if device_issues:
        return _unverified("设备状态不可核验：" + "；".join(device_issues))
    execution_id = payload.get("execution_id")
    ended = (device_state.get("run_history") or {}).get(execution_id) or {}
    if not ended and execution_id and device_state.get("last_ended_run_id") == execution_id:
        ended = {"reason": device_state.get("last_run_end_reason"), "ended_at": device_state.get("last_run_ended_at")}
    if not ended and execution_id and device_state.get("execution_id") == execution_id:
        ended = {"reason": device_state.get("execution_status"), "ended_at": device_state.get("execution_ended_at")}
    if ended.get("reason") not in {"completed", "success"}:
        return _unverified("本轮设备作业尚无可核对的完成证据")
    state = sensor_states.get(sensor_id) or {}
    values, issues = validate_sensor_state(state, now=now, max_age_seconds=feedback.get("max_age_seconds", 300))
    if field not in values:
        return _unverified("效果观测缺失或无效：" + "；".join(issues))
    baseline = payload.get("baseline") or {}
    try:
        if not ended.get("ended_at") or timestamp(ended["ended_at"]) > now:
            return _unverified("缺少本轮作业实际结束时间，不能确认维护后的效果")
        observed = timestamp(state.get("_observed_at"))
        if observed <= timestamp(ended["ended_at"]):
            return _unverified("效果观测须晚于本轮作业结束，请重新采样")
        if observed <= timestamp(baseline["timestamp"]):
            return _unverified("效果观测早于作业，不能作为复查依据")
    except (ValueError, TypeError, KeyError):
        return _unverified("缺少作业与效果观测的有效时间")
    before = _number((baseline.get("sensor_readings") or {}).get(f"{sensor_id}.{field}"))
    after = values[field]
    checks = []
    if "min_delta" in feedback:
        delta = _number(feedback["min_delta"])
        if before is None or delta is None or delta <= 0:
            return _unverified("效果变化量缺少有效基线或正数判据")
        direction = feedback.get("direction", "increase")
        if direction not in {"increase", "decrease"}:
            return _unverified("效果变化方向配置无效")
        change = before - after if direction == "decrease" else after - before
        checks.append(change >= delta)
    for key, passed in (("target_min", lambda target: after >= target), ("target_max", lambda target: after <= target)):
        if key in feedback:
            target = _number(feedback[key])
            if target is None:
                return _unverified("效果目标值必须是有限数字")
            checks.append(passed(target))
    if not checks:
        return _unverified("未配置维护效果目标或变化量，需人工复核")
    comparison = {"sensor_id": sensor_id, "field": field, "before": before, "after": after}
    return {"success": all(checks), "terminal": True, "comparison": comparison,
            "message": "维护效果达到配置判据" if all(checks) else "维护效果未达到判据，需要停止检查与人工接管",
            "error": "" if all(checks) else "维护效果未达到判据，需要停止检查与人工接管"}


def run_effect_check(job):
    from core.device_registry_factory import RegistrySession, load_custom_devices
    from core.device_executor import DeviceExecutor
    from devices.base import DeviceCommand, CommandPriority

    payload = job["payload"]
    action = payload.get("action") or {}
    device_id = action.get("device_id")
    feedback = payload.get("feedback") or {}
    sensor_id = feedback.get("sensor_id")
    configs = {item["device_id"]: item for item in load_custom_devices(job["username"])}
    device_config, sensor_config = configs.get(device_id, {}), configs.get(sensor_id, {})
    expected_plot = str(job["region"])
    if sensor_id and (str(device_config.get("plot_id")) != expected_plot or str(sensor_config.get("plot_id")) != expected_plot):
        return _unverified("效果传感器与设备必须明确绑定到本次地块")
    with RegistrySession(job["username"]) as (registry, loop):
        loop.run_until_complete(registry.discover_all())
        state = loop.run_until_complete(registry.read_state(device_id))
        sensor_states = {sensor_id: loop.run_until_complete(registry.read_state(sensor_id))} if sensor_id else {}
        outcome = evaluate_effect(payload, state, sensor_states)
        # 哥哥，只停止仍属于这一轮的运行，不能误停后来启动的新作业。
        current_id = state.get("run_id") or state.get("execution_id")
        _, device_issues = validate_sensor_state(state)
        if (not outcome["success"] and not device_issues and current_id and payload.get("execution_id")
            and state.get("status") == "running" and current_id == payload["execution_id"]):
            stop = DeviceExecutor(registry, username=job["username"]).execute_sync(
                device_id, DeviceCommand("stop", priority=CommandPriority.EMERGENCY, execution_id=current_id),
                trigger="effect_check", loop=loop, policy_context={"plot_id": expected_plot},
            )
            outcome["stop_confirmed"] = bool(stop.get("success"))
        return outcome
