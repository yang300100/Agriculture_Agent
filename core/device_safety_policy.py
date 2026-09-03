"""统一设备安全策略。

所有控制入口都可以通过本模块获得同一套安全判断：代码级物理上限不可
突破，用户策略可在物理上限内按能力、设备、地块或作业分区收紧限制。
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

AUTO_EXECUTE = "auto_execute"
NEED_CONFIRM = "need_confirm"
REJECTED = "rejected"


# 设备物理绝对上限。可通过部署环境变量按真实设备规格调整，但普通用户
# 的安全策略不能突破这些数值。
ABSOLUTE_CEILINGS: Dict[str, Dict[str, float]] = {
    "irrigate": {
        "max_duration_per_use_minutes": float(
            os.getenv("ABSOLUTE_IRRIGATE_MAX_DURATION_MINUTES", "120")
        ),
        "min_interval_seconds": 10,
    },
    "fertigate": {
        "max_duration_per_use_minutes": float(
            os.getenv("ABSOLUTE_FERTIGATE_MAX_DURATION_MINUTES", "120")
        ),
        "max_amount_per_use_kg": float(
            os.getenv("ABSOLUTE_FERTIGATE_MAX_AMOUNT_KG", "50")
        ),
        "min_interval_seconds": 10,
    },
    "ventilate": {
        "max_duration_per_use_minutes": float(
            os.getenv("ABSOLUTE_VENTILATE_MAX_DURATION_MINUTES", "120")
        ),
        "min_interval_seconds": 5,
    },
    "light": {
        "max_duration_per_use_minutes": float(
            os.getenv("ABSOLUTE_LIGHT_MAX_DURATION_MINUTES", "720")
        ),
        "min_interval_seconds": 5,
    },
    "heat": {
        "max_duration_per_use_minutes": float(
            os.getenv("ABSOLUTE_HEAT_MAX_DURATION_MINUTES", "240")
        ),
        "min_interval_seconds": 10,
    },
    "cool": {
        "max_duration_per_use_minutes": float(
            os.getenv("ABSOLUTE_COOL_MAX_DURATION_MINUTES", "240")
        ),
        "min_interval_seconds": 10,
    },
    "shade": {
        "max_duration_per_use_minutes": float(
            os.getenv("ABSOLUTE_SHADE_MAX_DURATION_MINUTES", "240")
        ),
        "min_interval_seconds": 5,
    },
}

SUPPORTED_LIMITS = {
    "max_duration_per_use_minutes",
    "max_duration_per_day_minutes",
    "max_amount_per_use_kg",
    "max_amount_per_day_kg",
    "max_volume_per_use_liters",
    "max_volume_per_day_liters",
    "min_interval_minutes",
    "forbidden_hours",
    "require_sensor_data",
    "rated_flow_lpm",
}

VALID_SCOPE_TYPES = {"global", "capability", "device", "plot", "zone"}
VALID_VIOLATION_ACTIONS = {"reject", "confirm"}


@dataclass
class PolicyEvaluation:
    decision: str
    reason: str
    params: Dict[str, Any]
    capability: str
    matched_policy_ids: List[int] = field(default_factory=list)
    calculated_volume_liters: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "params": self.params,
            "capability": self.capability,
            "matched_policy_ids": self.matched_policy_ids,
            "calculated_volume_liters": self.calculated_volume_liters,
        }


def _number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _duration(params: Dict[str, Any]) -> float:
    return _number(params.get("duration")) or _number(
        params.get("duration_minutes")
    ) or 0.0


def _amount(params: Dict[str, Any]) -> float:
    return _number(params.get("amount_kg")) or 0.0


def _volume(params: Dict[str, Any], rated_flow_lpm: Optional[float] = None) -> Optional[float]:
    direct = _number(params.get("volume_liters"))
    if direct is not None:
        return direct
    flow = (
        _number(params.get("flow_rate_lpm"))
        or _number(params.get("flow_rate"))
        or rated_flow_lpm
    )
    duration = _duration(params)
    if flow is not None and duration > 0:
        return flow * duration
    return None


class SafetyPolicyService:
    """读取、校验并评估用户安全策略。"""

    def __init__(self, username: str = "default", policies: Optional[List[Dict]] = None):
        self.username = username
        self._provided_policies = deepcopy(policies) if policies is not None else None

    # ── CRUD ──────────────────────────────────────

    def _get_user(self, create: bool = False):
        from core.database.engine import init_db
        from core.database.repository.users import UserRepository

        init_db()
        repo = UserRepository()
        user = repo.get_by_username(self.username)
        if not user and create:
            user = repo.create(username=self.username, password_hash="")
        return user

    @staticmethod
    def _serialize(row) -> Dict[str, Any]:
        try:
            limits = json.loads(row.limits) if row.limits else {}
        except (TypeError, json.JSONDecodeError):
            limits = {}
        return {
            "id": row.id,
            "name": row.name,
            "enabled": bool(row.enabled),
            "scope_type": row.scope_type or "capability",
            "capability": row.capability or "",
            "device_id": row.device_id or "",
            "plot_id": row.plot_id,
            "zone_id": row.zone_id or "",
            "limits": limits,
            "violation_action": row.violation_action or "reject",
            "created_at": row.created_at.isoformat() if row.created_at else "",
            "updated_at": row.updated_at.isoformat() if row.updated_at else "",
        }

    def list_policies(self) -> List[Dict[str, Any]]:
        if self._provided_policies is not None:
            return deepcopy(self._provided_policies)
        from core.database.repository.devices import DeviceSafetyPolicyRepository

        user = self._get_user(create=False)
        if not user:
            return []
        rows = DeviceSafetyPolicyRepository().find_by(user_id=user.id)
        return [self._serialize(row) for row in rows]

    def get_policy(self, policy_id: int) -> Optional[Dict[str, Any]]:
        for policy in self.list_policies():
            try:
                if int(policy.get("id")) == int(policy_id):
                    return policy
            except (TypeError, ValueError):
                # 外部注入的临时策略可能没有数据库 ID，不应影响其他策略查询。
                continue
        return None

    def validate_policy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        scope_type = str(data.get("scope_type", "capability")).lower()
        if scope_type not in VALID_SCOPE_TYPES:
            raise ValueError(f"不支持的策略范围: {scope_type}")
        violation_action = str(data.get("violation_action", "reject")).lower()
        if violation_action not in VALID_VIOLATION_ACTIONS:
            raise ValueError("超限处理只能是 reject 或 confirm")
        capability = str(data.get("capability", "")).lower()
        if capability and capability not in ABSOLUTE_CEILINGS:
            raise ValueError(f"不支持的设备能力: {capability}")
        if scope_type == "capability" and not capability:
            raise ValueError("能力级策略必须选择设备能力")
        if scope_type == "device" and not str(data.get("device_id", "")).strip():
            raise ValueError("设备级策略必须选择目标设备")
        if scope_type == "plot" and data.get("plot_id") in (None, ""):
            raise ValueError("地块级策略必须选择地块")
        if scope_type == "zone" and not str(data.get("zone_id", "")).strip():
            raise ValueError("分区级策略必须填写分区ID")

        raw_limits = data.get("limits", {})
        if not isinstance(raw_limits, dict):
            raise ValueError("limits 必须是 JSON 对象")
        limits: Dict[str, Any] = {}
        for key, value in raw_limits.items():
            if key not in SUPPORTED_LIMITS:
                continue
            if key == "forbidden_hours":
                if not isinstance(value, list):
                    raise ValueError("forbidden_hours 必须是小时数组")
                limits[key] = sorted({
                    int(hour) for hour in value
                    if isinstance(hour, (int, float)) and 0 <= int(hour) <= 23
                })
            elif key == "require_sensor_data":
                limits[key] = bool(value)
            else:
                parsed = _number(value)
                if parsed is None or parsed < 0:
                    raise ValueError(f"{key} 必须是非负数字")
                limits[key] = parsed

        # 用户可以收紧或恢复到物理上限，但不能突破设备绝对规格。
        if capability:
            ceiling = ABSOLUTE_CEILINGS.get(capability, {})
            for key in ("max_duration_per_use_minutes", "max_amount_per_use_kg"):
                if key in limits and key in ceiling and limits[key] > ceiling[key]:
                    raise ValueError(
                        f"{key} 不能超过设备物理上限 {ceiling[key]:g}"
                    )

        plot_id = data.get("plot_id")
        if plot_id not in (None, ""):
            try:
                plot_id = int(plot_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("plot_id 必须是整数") from exc
        else:
            plot_id = None

        return {
            "name": str(data.get("name", "未命名安全策略")).strip() or "未命名安全策略",
            "enabled": bool(data.get("enabled", True)),
            "scope_type": scope_type,
            "capability": capability or None,
            "device_id": str(data.get("device_id", "")).strip() or None,
            "plot_id": plot_id,
            "zone_id": str(data.get("zone_id", "")).strip() or None,
            "limits": limits,
            "violation_action": violation_action,
        }

    def create_policy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from core.database.repository.devices import DeviceSafetyPolicyRepository

        clean = self.validate_policy(data)
        user = self._get_user(create=True)
        row = DeviceSafetyPolicyRepository().create(
            user_id=user.id,
            name=clean["name"],
            enabled=1 if clean["enabled"] else 0,
            scope_type=clean["scope_type"],
            capability=clean["capability"],
            device_id=clean["device_id"],
            plot_id=clean["plot_id"],
            zone_id=clean["zone_id"],
            limits=json.dumps(clean["limits"], ensure_ascii=False),
            violation_action=clean["violation_action"],
        )
        return self._serialize(row)

    def update_policy(self, policy_id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        from core.database.repository.devices import DeviceSafetyPolicyRepository

        existing = self.get_policy(policy_id)
        if not existing:
            return None
        clean = self.validate_policy({**existing, **data})
        row = DeviceSafetyPolicyRepository().update(
            int(policy_id),
            name=clean["name"],
            enabled=1 if clean["enabled"] else 0,
            scope_type=clean["scope_type"],
            capability=clean["capability"],
            device_id=clean["device_id"],
            plot_id=clean["plot_id"],
            zone_id=clean["zone_id"],
            limits=json.dumps(clean["limits"], ensure_ascii=False),
            violation_action=clean["violation_action"],
        )
        return self._serialize(row) if row else None

    def delete_policy(self, policy_id: int) -> bool:
        from core.database.repository.devices import DeviceSafetyPolicyRepository

        existing = self.get_policy(policy_id)
        if not existing:
            return False
        return DeviceSafetyPolicyRepository().delete(int(policy_id))

    # ── 评估 ──────────────────────────────────────

    @staticmethod
    def _matches(policy: Dict[str, Any], capability: str, device_id: str,
                 context: Dict[str, Any]) -> bool:
        if not policy.get("enabled", True):
            return False
        scope = policy.get("scope_type", "capability")
        if policy.get("capability") and policy.get("capability") != capability:
            return False
        if scope == "global":
            return True
        if scope == "capability":
            return policy.get("capability") == capability
        if scope == "device":
            return policy.get("device_id") == device_id
        if scope == "plot":
            return str(policy.get("plot_id")) == str(context.get("plot_id"))
        if scope == "zone":
            return str(policy.get("zone_id")) == str(context.get("zone_id"))
        return False

    def _successful_usage_today(self, device_id: str) -> Dict[str, float]:
        """从数据库日志汇总今日真实成功用量，进程重启后仍然有效。"""
        usage = {"duration": 0.0, "amount_kg": 0.0, "volume_liters": 0.0}
        if self._provided_policies is not None:
            return usage
        try:
            from core.database.models import DeviceActionLog, User
            from core.database.engine import get_session

            session = get_session()
            try:
                start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                rows = (
                    session.query(DeviceActionLog)
                    .join(User, User.id == DeviceActionLog.user_id)
                    .filter(
                        User.username == self.username,
                        DeviceActionLog.device_id == device_id,
                        DeviceActionLog.success == 1,
                        DeviceActionLog.created_at >= start,
                    )
                    .all()
                )
                for row in rows:
                    try:
                        params = json.loads(row.params) if row.params else {}
                    except (TypeError, json.JSONDecodeError):
                        params = {}
                    usage["duration"] += _duration(params)
                    usage["amount_kg"] += _amount(params)
                    usage["volume_liters"] += _volume(params) or 0.0
            finally:
                session.close()
        except Exception as exc:
            logger.debug("读取设备当日用量失败: %s", exc)
        return usage

    def _last_success_at(self, device_id: str) -> Optional[datetime]:
        if self._provided_policies is not None:
            return None
        try:
            from core.database.models import DeviceActionLog, User
            from core.database.engine import get_session

            session = get_session()
            try:
                row = (
                    session.query(DeviceActionLog)
                    .join(User, User.id == DeviceActionLog.user_id)
                    .filter(
                        User.username == self.username,
                        DeviceActionLog.device_id == device_id,
                        DeviceActionLog.success == 1,
                    )
                    .order_by(DeviceActionLog.created_at.desc())
                    .first()
                )
                return row.created_at if row else None
            finally:
                session.close()
        except Exception as exc:
            logger.debug("读取设备最近执行时间失败: %s", exc)
            return None

    @staticmethod
    def _decision_for(violations: Iterable[tuple[str, str]]) -> tuple[str, str]:
        items = list(violations)
        if not items:
            return AUTO_EXECUTE, "安全策略校验通过"
        decision = REJECTED if any(action == "reject" for action, _ in items) else NEED_CONFIRM
        return decision, "；".join(reason for _, reason in items)

    def evaluate(self, device_id: str, capability: str, params: Dict[str, Any],
                 command: str = "start", context: Optional[Dict[str, Any]] = None) -> PolicyEvaluation:
        params = deepcopy(params or {})
        context = dict(context or {})
        capability = str(capability or "").lower()

        # 停止设备属于减险操作，不受运行时长和禁用时段限制。
        if command in {"stop", "power_off", "shutdown"}:
            return PolicyEvaluation(AUTO_EXECUTE, "停止操作直接放行", params, capability)

        for key, value in params.items():
            if key in {"duration", "duration_minutes", "amount_kg", "volume_liters", "flow_rate_lpm", "flow_rate"}:
                parsed = _number(value)
                if parsed is not None and parsed < 0:
                    return PolicyEvaluation(REJECTED, f"参数 {key} 不能为负数", params, capability)

        duration = _duration(params)
        amount = _amount(params)
        ceiling = ABSOLUTE_CEILINGS.get(capability, {})
        physical_violations: List[tuple[str, str]] = []
        max_duration = ceiling.get("max_duration_per_use_minutes")
        if max_duration is not None and duration > max_duration:
            physical_violations.append((
                "reject",
                f"单次{capability}时长 {duration:g} 分钟超过物理上限 {max_duration:g} 分钟",
            ))
        max_amount = ceiling.get("max_amount_per_use_kg")
        if max_amount is not None and amount > max_amount:
            physical_violations.append((
                "reject",
                f"单次{capability}用量 {amount:g}kg 超过物理上限 {max_amount:g}kg",
            ))
        if physical_violations:
            decision, reason = self._decision_for(physical_violations)
            return PolicyEvaluation(decision, reason, params, capability)

        last_success = self._last_success_at(device_id)
        physical_interval = _number(ceiling.get("min_interval_seconds")) or 0
        if last_success and physical_interval > 0:
            elapsed = (datetime.now() - last_success).total_seconds()
            if elapsed < physical_interval:
                return PolicyEvaluation(
                    REJECTED,
                    f"距上次成功操作仅 {elapsed:.1f} 秒，低于物理最小间隔 {physical_interval:g} 秒",
                    params,
                    capability,
                )

        matched = [
            policy for policy in self.list_policies()
            if self._matches(policy, capability, device_id, context)
        ]
        usage = self._successful_usage_today(device_id)
        violations: List[tuple[str, str]] = []
        policy_ids: List[int] = []
        calculated_volume: Optional[float] = None

        for policy in matched:
            try:
                policy_ids.append(int(policy.get("id")))
            except (TypeError, ValueError):
                # 允许测试或上层规划器传入尚未持久化的临时策略。
                pass
            limits = policy.get("limits", {}) or {}
            action = policy.get("violation_action", "reject")
            rated_flow = _number(limits.get("rated_flow_lpm"))
            volume = _volume(params, rated_flow)
            if volume is not None:
                calculated_volume = volume

            checks = (
                ("max_duration_per_use_minutes", duration,
                 "单次时长", "分钟"),
                ("max_duration_per_day_minutes", usage["duration"] + duration,
                 "今日累计时长", "分钟"),
                ("max_amount_per_use_kg", amount,
                 "单次用量", "kg"),
                ("max_amount_per_day_kg", usage["amount_kg"] + amount,
                 "今日累计用量", "kg"),
            )
            for key, actual, label, unit in checks:
                limit = _number(limits.get(key))
                if limit is not None and actual > limit:
                    violations.append((
                        action,
                        f"策略「{policy['name']}」：{label} {actual:g}{unit} 超过上限 {limit:g}{unit}",
                    ))

            per_volume = _number(limits.get("max_volume_per_use_liters"))
            daily_volume = _number(limits.get("max_volume_per_day_liters"))
            if per_volume is not None:
                if volume is None:
                    violations.append((
                        action,
                        f"策略「{policy['name']}」要求按水量校验，但设备缺少流量或水量参数",
                    ))
                elif volume > per_volume:
                    violations.append((
                        action,
                        f"策略「{policy['name']}」：单次水量 {volume:g}L 超过上限 {per_volume:g}L",
                    ))
            if daily_volume is not None:
                if volume is None:
                    violations.append((
                        action,
                        f"策略「{policy['name']}」要求累计水量校验，但设备缺少流量或水量参数",
                    ))
                elif usage["volume_liters"] + volume > daily_volume:
                    violations.append((
                        action,
                        f"策略「{policy['name']}」：今日累计水量将超过 {daily_volume:g}L",
                    ))

            forbidden = limits.get("forbidden_hours", [])
            if datetime.now().hour in forbidden:
                violations.append((action, f"策略「{policy['name']}」：当前处于禁止运行时段"))
            if limits.get("require_sensor_data") and not context.get("sensor_data"):
                violations.append((action, f"策略「{policy['name']}」：缺少传感器数据"))
            minimum_interval = _number(limits.get("min_interval_minutes"))
            if minimum_interval and last_success:
                elapsed_minutes = (
                    datetime.now() - last_success
                ).total_seconds() / 60
                if elapsed_minutes < minimum_interval:
                    violations.append((
                        action,
                        f"策略「{policy['name']}」：距上次操作 {elapsed_minutes:.1f} 分钟，"
                        f"低于最小间隔 {minimum_interval:g} 分钟",
                    ))

        decision, reason = self._decision_for(violations)
        return PolicyEvaluation(
            decision=decision,
            reason=reason,
            params=params,
            capability=capability,
            matched_policy_ids=policy_ids,
            calculated_volume_liters=calculated_volume,
        )


def get_safety_catalog() -> Dict[str, Any]:
    """供前端动态构建安全策略表单。"""
    return {
        "absolute_ceilings": deepcopy(ABSOLUTE_CEILINGS),
        "scope_types": sorted(VALID_SCOPE_TYPES),
        "violation_actions": sorted(VALID_VIOLATION_ACTIONS),
        "supported_limits": sorted(SUPPORTED_LIMITS),
    }
