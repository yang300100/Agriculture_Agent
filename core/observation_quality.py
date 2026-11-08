"""哥哥，自动决策只使用可追溯且质量合格的传感器观测。"""
import math
from datetime import datetime, timezone

_RANGES = {
    "soil_moisture": (0, 100, {"%", "percent"}),
    "humidity": (0, 100, {"%", "percent"}),
    "temperature": (-60, 100, {"°C", "C", "℃"}),
    "soil_temperature": (-60, 100, {"°C", "C", "℃"}),
    "ph": (0, 14, {"pH", "ph"}),
    "light": (0, 250000, {"lux"}),
    "light_intensity": (0, 250000, {"lux"}),
}


def validate_sensor_state(state: dict, *, now=None, max_age_seconds=300) -> tuple[dict, list[str]]:
    if not isinstance(state, dict):
        return {}, ["观测格式无效"]
    if state.get("error") or state.get("_error") or state.get("_quality") != "good":
        return {}, ["观测存在错误或质量不合格"]
    try:
        observed = state.get("_observed_at")
        observed = datetime.fromisoformat(observed.replace("Z", "+00:00")) if isinstance(observed, str) else observed
        if not isinstance(observed, datetime):
            raise ValueError("缺少观测时间")
        now = now or datetime.now().astimezone()
        if observed.tzinfo is None:
            observed = observed.astimezone()
        if now.tzinfo is None:
            now = now.astimezone()
        age = (now - observed).total_seconds()
        if age < 0 or age > max_age_seconds:
            raise ValueError("观测时间过期或处于未来")
    except (ValueError, TypeError, OverflowError) as exc:
        return {}, [f"观测时间无效: {exc}"]
    units = state.get("_units") or {}
    valid, issues = {}, []
    for name, value in state.items():
        if name.startswith("_") or name == "error" or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            issues.append(f"{name}: 非有限读数")
            continue
        spec = _RANGES.get(name.lower())
        if spec and (not spec[0] <= value <= spec[1] or (name in units and units[name] not in spec[2])):
            issues.append(f"{name}: 范围或单位无效")
            continue
        valid[name] = value
    return valid, issues
