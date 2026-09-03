"""设备控制规则引擎 — 条件匹配 + 约束校验 + AI 微调的混合决策核心"""

import json
import logging
import os
import re
import shutil
import threading
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from core.device_safety_policy import ABSOLUTE_CEILINGS, SafetyPolicyService
from core.storage_paths import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# ── 全局执行历史 + 线程锁，跨 RuleEngine 实例共享 ──
_global_history: Dict[str, Dict] = {}
_history_lock = threading.Lock()

# ── 代码级硬限制（不可通过规则配置突破）— 唯一权威来源 ──
HARD_LIMITS = ABSOLUTE_CEILINGS

# 规则文件 JSON schema 基本要求
_RULE_REQUIRED_FIELDS = {"trigger", "action"}


class RuleDecision:
    """规则评估结果"""
    AUTO_EXECUTE = "auto_execute"
    NEED_CONFIRM = "need_confirm"
    REJECTED = "rejected"


def apply_autonomy(decision: str, autonomy_level: str = "medium") -> str:
    """根据自主权级别调整决策结果

    Args:
        decision: 原始决策 (auto_execute / need_confirm / rejected)
        autonomy_level: 自主权级别
            - low: 全部需要确认（need_confirm 不变，auto_execute 降级为 need_confirm）
            - medium: 规则边界内自动，边界外确认（默认行为，不调整）
            - high: 完全自主，need_confirm 升级为 auto_execute

    Returns:
        调整后的决策
    """
    valid_levels = {"low", "medium", "high"}
    if autonomy_level not in valid_levels:
        logger.warning("未知的自主权级别: %r，按 medium 处理", autonomy_level)
        autonomy_level = "medium"

    if autonomy_level == "low":
        if decision == RuleDecision.AUTO_EXECUTE:
            return RuleDecision.NEED_CONFIRM
    elif autonomy_level == "high":
        if decision == RuleDecision.NEED_CONFIRM:
            return RuleDecision.AUTO_EXECUTE
    return decision


class RuleEngine:
    """设备控制规则引擎"""

    def __init__(self, username: str = "default"):
        self.username = username
        self.rules: List[Dict] = []
        global _global_history
        if username not in _global_history:
            _global_history[username] = {"execution_history": {}, "daily_duration": {}}
        self._execution_history = _global_history[username]["execution_history"]
        self._daily_duration = _global_history[username]["daily_duration"]
        self._load_rules()

    # ── 规则持久化 ──────────────────────────

    def _rules_path(self) -> str:
        return os.path.join(DEFAULT_DATA_DIR, self.username, "device_rules.json")

    def _load_rules(self) -> None:
        """从DB加载规则"""
        from core.database.repository.devices import DeviceRuleRepository
        from core.database.repository.users import UserRepository
        user_repo = UserRepository()
        user = user_repo.get_by_username(self.username)
        if not user:
            self.rules = []
            return
        repo = DeviceRuleRepository()
        db_rules = repo.find_by(user_id=user.id)
        self.rules = []
        for row in db_rules:
            try:
                trigger_payload = json.loads(row.conditions) if row.conditions else []
            except (TypeError, json.JSONDecodeError):
                trigger_payload = []
            if isinstance(trigger_payload, list):
                # 兼容旧数据库：旧版只保存 conditions，逻辑默认为 AND。
                trigger = {"logic": "AND", "conditions": trigger_payload}
                metadata = {}
            elif isinstance(trigger_payload, dict):
                trigger = {
                    "logic": str(trigger_payload.get("logic", "AND")).upper(),
                    "conditions": trigger_payload.get("conditions", []),
                }
                metadata = trigger_payload
            else:
                trigger = {"logic": "AND", "conditions": []}
                metadata = {}
            try:
                action = json.loads(row.actions) if row.actions else {}
            except (TypeError, json.JSONDecodeError):
                action = {}
            self.rules.append({
                "id": row.id,
                "name": row.name,
                "enabled": bool(row.enabled),
                "trigger": trigger,
                "action": action,
                "constraints": json.loads(row.constraints) if row.constraints else {},
                "ai_enhance": metadata.get("ai_enhance", {}),
                "execution_mode": metadata.get("execution_mode", "auto"),
                "created_at": row.created_at.isoformat() if row.created_at else "",
            })
        logger.info("规则引擎(DB): 已加载 %d 条规则", len(self.rules))

    def _save_rules(self) -> None:
        """保存规则到DB"""
        from core.database.repository.devices import DeviceRuleRepository
        from core.database.repository.users import UserRepository
        user_repo = UserRepository()
        user = user_repo.get_by_username(self.username)
        if not user:
            user = user_repo.create(username=self.username, password_hash="")
        repo = DeviceRuleRepository()
        items = []
        for rule in self.rules:
            trigger = rule.get("trigger", {})
            trigger_payload = {
                "logic": str(trigger.get("logic", "AND")).upper(),
                "conditions": trigger.get("conditions", []),
                "ai_enhance": rule.get("ai_enhance", {}),
                "execution_mode": rule.get("execution_mode", "auto"),
            }
            items.append({
                "id": rule.get("id"),
                "name": rule.get("name", ""),
                "enabled": 1 if rule.get("enabled", True) else 0,
                "conditions": json.dumps(trigger_payload, ensure_ascii=False),
                "actions": json.dumps(rule.get("action", {}), ensure_ascii=False),
                "constraints": json.dumps(rule.get("constraints", {}), ensure_ascii=False),
            })
        rows = repo.sync_for_user(user.id, items)
        for rule, row in zip(self.rules, rows):
            rule["id"] = row.id
            if not rule.get("created_at") and row.created_at:
                rule["created_at"] = row.created_at.isoformat()

    def _backup_corrupted(self, path: str):
        """将损坏/异常文件备份，防止数据完全丢失"""
        try:
            corrupted = path + ".corrupted." + datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.exists(path):
                os.rename(path, corrupted)
                logger.info("已保留损坏文件: %s", corrupted)
        except Exception as e:
            logger.warning("备份损坏文件失败: %s", e)

    # ── 规则 CRUD ────────────────────────────

    def list_rules(self) -> List[Dict]:
        return deepcopy(self.rules)

    def get_rule(self, rule_id: str) -> Optional[Dict]:
        for r in self.rules:
            if str(r["id"]) == str(rule_id):
                return deepcopy(r)
        return None

    def add_rule(self, rule: Dict) -> str:
        rule = self._normalize_rule(rule)
        # 数据库 ID 是唯一权威；忽略客户端自带 ID，防止覆盖其他规则。
        rule.pop("id", None)
        rule.setdefault("enabled", True)
        # 基础 schema 校验
        missing = _RULE_REQUIRED_FIELDS - set(rule.keys())
        if missing:
            raise ValueError(f"规则缺少必要字段: {missing}")
        if not isinstance(rule.get("trigger", {}).get("conditions"), list):
            raise ValueError("规则 trigger.conditions 必须是列表")
        self.rules.append(rule)
        self._save_rules()
        logger.info("规则已添加: %s", rule["id"])
        return str(rule["id"])

    def update_rule(self, rule_id: str, updates: Dict) -> bool:
        # 复制 updates 避免修改调用者的数据
        updates = dict(updates)
        # 不允许通过 update 修改规则 ID
        if "id" in updates:
            logger.warning("update_rule 忽略 id 字段 (不允许修改规则ID)")
            del updates["id"]
        for i, r in enumerate(self.rules):
            if str(r["id"]) == str(rule_id):
                self.rules[i] = self._normalize_rule({**r, **updates})
                self._save_rules()
                return True
        return False

    def delete_rule(self, rule_id: str) -> bool:
        before = len(self.rules)
        self.rules = [
            r for r in self.rules if str(r["id"]) != str(rule_id)
        ]
        if len(self.rules) < before:
            self._save_rules()
            return True
        return False

    def toggle_rule(self, rule_id: str, enabled: bool) -> bool:
        return self.update_rule(rule_id, {"enabled": enabled})

    # ── 规则评估 ─────────────────────────────

    def _normalize_rule(self, rule: Dict) -> Dict:
        """规范化自动化规则，同时兼容旧规则结构。"""
        from core.device_action_schema import normalize_action

        normalized = deepcopy(rule)
        trigger = normalized.get("trigger", {})
        conditions = trigger.get("conditions", [])
        if not isinstance(conditions, list):
            raise ValueError("规则 trigger.conditions 必须是列表")
        logic = str(trigger.get("logic", "AND")).upper()
        if logic not in {"AND", "OR"}:
            raise ValueError("规则触发逻辑只能是 AND 或 OR")
        normalized["trigger"] = {"logic": logic, "conditions": conditions}

        action = dict(normalized.get("action", {}))
        if not action.get("device_id"):
            raise ValueError("规则必须选择目标设备")
        capability = action.get("capability") or self._infer_capability(action)
        action["capability"] = capability
        action["command"] = str(action.get("command", "start")).lower()
        action["params"] = normalize_action(
            capability, action["command"], action.get("params", {})
        )
        normalized["action"] = action
        mode = str(normalized.get("execution_mode", "auto")).lower()
        if mode not in {"auto", "confirm", "notify"}:
            raise ValueError("执行方式只能是 auto、confirm 或 notify")
        normalized["execution_mode"] = mode
        normalized.setdefault("constraints", {})
        normalized.setdefault("ai_enhance", {})
        return normalized

    def find_matching_rules(self, context: Dict) -> List[Dict]:
        matched = []
        for rule in self.rules:
            if not rule.get("enabled", True):
                continue
            if self._evaluate_trigger(rule.get("trigger", {}), context):
                matched.append(deepcopy(rule))
        return matched

    def evaluate_action(self, rule: Dict, proposed_params: Dict,
                        context: Dict) -> Tuple[str, str, Dict]:
        constraints = rule.get("constraints", {})
        action = rule.get("action", {})
        device_id = action.get("device_id", "")
        capability = self._infer_capability(action)

        context = {**self._device_scope_context(device_id), **(context or {})}

        # 1. 统一安全策略：先检查物理绝对上限，再检查用户可配置边界。
        policy_result = SafetyPolicyService(self.username).evaluate(
            device_id=device_id,
            capability=capability,
            params=proposed_params,
            command=action.get("command", "start"),
            context=context,
        )
        if policy_result.decision != RuleDecision.AUTO_EXECUTE:
            return policy_result.decision, policy_result.reason, policy_result.params

        # 兼容原有进程级最小秒级间隔，防止同一周期瞬时重复触发。
        hard_ok, hard_reason = self._check_hard_limits(
            capability, proposed_params, device_id
        )
        if not hard_ok:
            return RuleDecision.REJECTED, hard_reason, proposed_params

        # 2. 软约束检查 — 显式传递 device_id，不从 context 取
        soft_ok, soft_reason = self._check_constraints(constraints, proposed_params, device_id, context)
        if not soft_ok:
            return RuleDecision.NEED_CONFIRM, soft_reason, proposed_params

        # 3. AI 微调
        ai_enhance = rule.get("ai_enhance", {})
        if ai_enhance.get("enabled", False):
            proposed_params = self._apply_ai_enhance(ai_enhance, proposed_params, action.get("params", {}))

        return RuleDecision.AUTO_EXECUTE, "规则校验通过", proposed_params

    def _device_scope_context(self, device_id: str) -> Dict:
        """从设备配置补全地块范围，为未来 zone_id 留出兼容入口。"""
        try:
            from core.device_registry_factory import load_custom_devices

            config = next(
                (item for item in load_custom_devices(self.username)
                 if item.get("device_id") == device_id),
                {},
            )
            return {
                "plot_id": config.get("plot_id"),
                "zone_id": config.get("zone_id"),
            }
        except Exception:
            return {}

    def record_execution(self, device_id: str, params: Dict, success: bool = True) -> None:
        """记录设备执行。

        Args:
            device_id: 设备ID
            params: 执行参数
            success: 是否实际执行成功。失败的操作不记录历史，
                     避免阻止后续重试。
        """
        if not success:
            return  # 失败不记录，不消耗 min_interval 配额

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        with _history_lock:
            if device_id not in self._execution_history:
                self._execution_history[device_id] = []
            hist = self._execution_history[device_id]
            hist.append(now)
            if len(hist) > 100:
                self._execution_history[device_id] = hist[-100:]

            if device_id not in self._daily_duration:
                self._daily_duration[device_id] = {}
            if date_str not in self._daily_duration[device_id]:
                self._daily_duration[device_id][date_str] = 0

            duration = params.get("duration") or 0
            self._daily_duration[device_id][date_str] += duration

        # 定期清理旧日期记录（保留最近30天）
        self._prune_old_dates(device_id)

    def _prune_old_dates(self, device_id: str):
        """清理超过30天的 daily_duration 记录，防止内存泄漏"""
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        with _history_lock:
            if device_id in self._daily_duration:
                old_keys = [k for k in self._daily_duration[device_id] if k < cutoff]
                for k in old_keys:
                    del self._daily_duration[device_id][k]

    # ── 内部方法 ──────────────────────────────

    def _evaluate_trigger(self, trigger: Dict, context: Dict) -> bool:
        conditions = trigger.get("conditions", [])
        if not conditions:
            return False
        results = [self._eval_single_condition(cond, context) for cond in conditions]
        logic = trigger.get("logic", "AND").upper()
        return all(results) if logic == "AND" else any(results)

    def _eval_single_condition(self, cond: Dict, context: Dict) -> bool:
        cond_type = cond.get("type", "")
        field = cond.get("field", "")
        op = cond.get("op", "==")
        expected = cond.get("value")

        if cond_type == "sensor":
            actual = context.get("sensor_data", {}).get(field)
        elif cond_type == "weather":
            actual = context.get("weather", {}).get(field)
        elif cond_type == "time":
            actual = datetime.now().strftime("%H:%M")
        else:
            actual = context.get(field)

        if actual is None:
            return False
        return self._compare(actual, op, expected)

    def _compare(self, actual, op: str, expected) -> bool:
        try:
            if op == "==":
                try:
                    return abs(float(actual) - float(expected)) < 1e-6
                except (ValueError, TypeError):
                    return actual == expected
            if op == "!=":
                # 数值容差比较 — 与 == 对称
                try:
                    return abs(float(actual) - float(expected)) >= 1e-6
                except (ValueError, TypeError):
                    return actual != expected
            if op == ">": return float(actual) > float(expected)
            if op == "<": return float(actual) < float(expected)
            if op == ">=": return float(actual) >= float(expected)
            if op == "<=": return float(actual) <= float(expected)
            if op == "between":
                if isinstance(expected, list) and len(expected) == 2:
                    try:
                        a_num, lo_num, hi_num = float(actual), float(expected[0]), float(expected[1])
                        # 数值比较：lo > hi 视为配置错误，按正常区间处理
                        if lo_num > hi_num:
                            lo_num, hi_num = hi_num, lo_num
                        return lo_num <= a_num <= hi_num
                    except (ValueError, TypeError):
                        # 字符串比较：先验证是否像时间格式（HH:MM），不是则拒绝
                        a_str, lo_str, hi_str = str(actual), str(expected[0]), str(expected[1])
                        time_pattern = re.compile(r'^\d{1,2}:\d{2}$')
                        if not (time_pattern.match(a_str) and time_pattern.match(lo_str) and time_pattern.match(hi_str)):
                            logger.warning("between 字符串比较仅支持 HH:MM 时间格式，收到 actual=%r lo=%r hi=%r，返回 False", a_str, lo_str, hi_str)
                            return False
                        # 时间比较（HH:MM）：允许 wraparound
                        if lo_str > hi_str:
                            return a_str >= lo_str or a_str <= hi_str
                        return lo_str <= a_str <= hi_str
                return False
            if op == "in":
                if not isinstance(expected, list):
                    return False
                # 尝试数值容差匹配
                try:
                    a_num = float(actual)
                    for e in expected:
                        try:
                            if abs(a_num - float(e)) < 1e-6:
                                return True
                        except (ValueError, TypeError):
                            continue
                    return False
                except (ValueError, TypeError):
                    return actual in expected
            return False
        except (ValueError, TypeError):
            return False

    def _infer_capability(self, action: Dict) -> str:
        """从设备 ID 或 action 类型推断设备能力类型"""
        explicit = str(action.get("capability", "")).lower()
        if explicit in HARD_LIMITS:
            return explicit
        device_id = action.get("device_id", "").lower()
        action_type = action.get("command", "").lower()

        # 优先从数据库设备能力读取，避免依赖设备 ID 命名。
        try:
            from core.device_registry_factory import load_custom_devices

            config = next(
                (item for item in load_custom_devices(self.username)
                 if str(item.get("device_id", "")).lower() == device_id),
                None,
            )
            if config:
                capabilities = [
                    value for value in config.get("capabilities", [])
                    if value in HARD_LIMITS
                ]
                if capabilities:
                    return capabilities[0]
        except Exception:
            pass

        # 按优先级检查 device_id 关键词
        if "fertigat" in device_id or "fertil" in device_id:
            return "fertigate"
        if "ventilat" in device_id or "fan" in device_id:
            return "ventilate"
        if "light" in device_id or "lamp" in device_id or "shade" in device_id:
            return "light"
        if "heat" in device_id:
            return "heat"
        if "cool" in device_id:
            return "cool"
        if "irrigat" in device_id or "water" in device_id:
            return "irrigate"

        # 从 action_type 推断
        if action_type in HARD_LIMITS:
            return action_type

        # 默认回退
        return "irrigate"

    def _check_hard_limits(self, capability: str, params: Dict, device_id: str) -> Tuple[bool, str]:
        limits = HARD_LIMITS.get(capability, {})
        max_dur = limits.get("max_duration_per_use_minutes")
        if max_dur and (params.get("duration") or 0) > max_dur:
            return False, f"单次{capability}操作时长 {params['duration']} 分钟超过硬限制 {max_dur} 分钟"
        max_amt = limits.get("max_amount_per_use_kg")
        if max_amt and (params.get("amount_kg") or 0) > max_amt:
            return False, f"单次{capability}施肥量 {params['amount_kg']}kg 超过硬限制 {max_amt}kg"
        min_interval = limits.get("min_interval_seconds", 0)
        if min_interval:
            with _history_lock:
                if device_id in self._execution_history:
                    hist = self._execution_history[device_id]
                    last = hist[-1] if hist else None
                    if last and (datetime.now() - last).total_seconds() < min_interval:
                        return False, f"距上次操作不足 {min_interval} 秒，拒绝重复触发"
        return True, ""

    def _check_constraints(self, constraints: Dict, params: Dict,
                           device_id: str, context: Dict) -> Tuple[bool, str]:
        max_dur = constraints.get("max_duration_per_use")
        if max_dur is not None and (params.get("duration") or 0) > max_dur:
            return False, f"单次时长 {params['duration']} 分钟超过设定上限 {max_dur} 分钟，需要确认"

        max_daily = constraints.get("max_duration_per_day")
        if max_daily is not None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            with _history_lock:
                today_used = self._daily_duration.get(device_id, {}).get(date_str, 0)
            if today_used + (params.get("duration") or 0) > max_daily:
                return False, f"今日累计超过每日上限，需要确认"

        # 最小间隔检查
        min_interval = constraints.get("min_interval_minutes")
        if min_interval:
            with _history_lock:
                if device_id in self._execution_history and self._execution_history[device_id]:
                    last = self._execution_history[device_id][-1]
                    elapsed = (datetime.now() - last).total_seconds() / 60
                else:
                    elapsed = float('inf')
            if elapsed < min_interval - 0.001:
                return False, f"距上次操作 {elapsed:.1f} 分钟，不足最小间隔 {min_interval} 分钟，需要确认"

        forbidden = constraints.get("forbidden_hours", [])
        # 过滤无效小时值
        valid_forbidden = [h for h in forbidden if isinstance(h, int) and 0 <= h <= 23]
        if valid_forbidden and datetime.now().hour in valid_forbidden:
            return False, f"当前时间在禁止时段内，需要确认"

        require_confirm = constraints.get("require_confirm_if", [])
        for expr in require_confirm:
            if self._eval_confirm_expr(expr, params, context):
                return False, f"触发确认条件: {expr}"

        return True, ""

    def _eval_confirm_expr(self, expr: str, params: Dict, context: Dict) -> bool:
        try:
            if " >= " in expr:
                field, val = expr.rsplit(" >= ", 1)
                return (params.get(field.strip()) or 0) >= float(val.strip())
            if " <= " in expr:
                field, val = expr.rsplit(" <= ", 1)
                return (params.get(field.strip()) or 0) <= float(val.strip())
            if " > " in expr:
                field, val = expr.rsplit(" > ", 1)
                return (params.get(field.strip()) or 0) > float(val.strip())
            if " < " in expr:
                field, val = expr.rsplit(" < ", 1)
                return (params.get(field.strip()) or 0) < float(val.strip())
            if " == " in expr:
                field, val = expr.rsplit(" == ", 1)
                return str(params.get(field.strip(), "")) == val.strip()
            if expr == "weather_forecast_conflict":
                return context.get("weather_conflict", False)
        except Exception as e:
            logger.warning("confirm_expr 解析失败: expr=%r, error=%s", expr, e)
            pass
        return False

    def _apply_ai_enhance(self, ai_config: Dict, proposed: Dict, original: Dict) -> Dict:
        result = dict(proposed)
        can_adjust = ai_config.get("can_adjust", [])
        ranges = ai_config.get("adjust_range", {})
        absolute_ranges = ai_config.get("absolute_range", {})
        for field in can_adjust:
            if field in result and field in absolute_ranges:
                lo, hi = absolute_ranges[field]
                if lo > hi:
                    lo, hi = hi, lo
                result[field] = max(lo, min(hi, result[field]))
                continue
            if field in result and field in ranges:
                min_adj, max_adj = ranges[field]
                # 验证 min_adj <= max_adj，否则交换并警告
                if min_adj > max_adj:
                    logger.warning("AI微调范围异常: %s 的 adjust_range [%s, %s] min>max，已自动交换",
                                   field, min_adj, max_adj)
                    min_adj, max_adj = max_adj, min_adj
                orig_val = original.get(field, result.get(field, 0))
                adjusted = result.get(field, 0)
                clamped = max(orig_val + min_adj, min(orig_val + max_adj, adjusted))
                result[field] = clamped
        return result
