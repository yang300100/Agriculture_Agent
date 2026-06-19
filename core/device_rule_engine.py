"""设备控制规则引擎 — 条件匹配 + 约束校验 + AI 微调的混合决策核心"""

import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.getenv("DATA_STORAGE_DIR", "data")

# ── 全局执行历史，跨 RuleEngine 实例共享 ──
_global_history: Dict[str, Dict] = {}

# ── 代码级硬限制（不可通过规则配置突破）─────────────────
HARD_LIMITS = {
    "irrigate": {
        "max_duration_per_use_minutes": 120,
        "min_interval_seconds": 10,
    },
    "fertigate": {
        "max_amount_per_use_kg": 50,
        "min_interval_seconds": 10,
    },
}


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
    if autonomy_level == "low":
        # 全部操作都需要确认
        if decision == RuleDecision.AUTO_EXECUTE:
            return RuleDecision.NEED_CONFIRM
    elif autonomy_level == "high":
        # 跳过确认，直接执行（REJECTED 硬限制不可突破）
        if decision == RuleDecision.NEED_CONFIRM:
            return RuleDecision.AUTO_EXECUTE
    # medium: 保持原样
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
        path = self._rules_path()
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    self.rules = data.get("rules", [])
                logger.info("规则引擎: 已加载 %d 条规则", len(self.rules))
            except Exception as e:
                logger.warning("规则加载失败: %s", e)
                self.rules = []

    def _save_rules(self) -> None:
        path = self._rules_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rules": self.rules, "updated_at": datetime.now().isoformat()},
                      f, ensure_ascii=False, indent=2)

    # ── 规则 CRUD ────────────────────────────

    def list_rules(self) -> List[Dict]:
        return deepcopy(self.rules)

    def get_rule(self, rule_id: str) -> Optional[Dict]:
        for r in self.rules:
            if r["id"] == rule_id:
                return deepcopy(r)
        return None

    def add_rule(self, rule: Dict) -> str:
        if "id" not in rule:
            import uuid
            rule["id"] = f"rule_{uuid.uuid4().hex[:8]}"
        rule.setdefault("enabled", True)
        self.rules.append(rule)
        self._save_rules()
        logger.info("规则已添加: %s", rule["id"])
        return rule["id"]

    def update_rule(self, rule_id: str, updates: Dict) -> bool:
        for i, r in enumerate(self.rules):
            if r["id"] == rule_id:
                self.rules[i] = {**r, **updates}
                self._save_rules()
                return True
        return False

    def delete_rule(self, rule_id: str) -> bool:
        before = len(self.rules)
        self.rules = [r for r in self.rules if r["id"] != rule_id]
        if len(self.rules) < before:
            self._save_rules()
            return True
        return False

    def toggle_rule(self, rule_id: str, enabled: bool) -> bool:
        return self.update_rule(rule_id, {"enabled": enabled})

    # ── 规则评估 ─────────────────────────────

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

        # 1. 硬限制检查
        hard_ok, hard_reason = self._check_hard_limits(capability, proposed_params, device_id)
        if not hard_ok:
            return RuleDecision.REJECTED, hard_reason, proposed_params

        # 2. 软约束检查
        soft_ok, soft_reason = self._check_constraints(constraints, proposed_params, context)
        if not soft_ok:
            return RuleDecision.NEED_CONFIRM, soft_reason, proposed_params

        # 3. AI 微调
        ai_enhance = rule.get("ai_enhance", {})
        if ai_enhance.get("enabled", False):
            proposed_params = self._apply_ai_enhance(ai_enhance, proposed_params, action.get("params", {}))

        return RuleDecision.AUTO_EXECUTE, "规则校验通过", proposed_params

    def record_execution(self, device_id: str, params: Dict) -> None:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        if device_id not in self._execution_history:
            self._execution_history[device_id] = []
        hist = self._execution_history[device_id]
        hist.append(now)
        # 只保留最近 100 条执行记录，防止长期运行内存膨胀
        if len(hist) > 100:
            self._execution_history[device_id] = hist[-100:]

        if device_id not in self._daily_duration:
            self._daily_duration[device_id] = {}
        if date_str not in self._daily_duration[device_id]:
            self._daily_duration[device_id][date_str] = 0

        duration = params.get("duration", 0)
        self._daily_duration[device_id][date_str] += duration
        # _execution_history 和 _daily_duration 与 _global_history 共享同一 dict 对象，
        # 无需显式写回（Python 引用语义）

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
            if op == "!=": return actual != expected
            if op == ">": return float(actual) > float(expected)
            if op == "<": return float(actual) < float(expected)
            if op == ">=": return float(actual) >= float(expected)
            if op == "<=": return float(actual) <= float(expected)
            if op == "between":
                if isinstance(expected, list) and len(expected) == 2:
                    # 尝试数值比较，失败则回退到字符串比较（用于时间格式 HH:MM）
                    try:
                        a_num, lo_num, hi_num = float(actual), float(expected[0]), float(expected[1])
                        if lo_num > hi_num:
                            return a_num >= lo_num or a_num <= hi_num
                        return lo_num <= a_num <= hi_num
                    except (ValueError, TypeError):
                        a_str, lo_str, hi_str = str(actual), str(expected[0]), str(expected[1])
                        if lo_str > hi_str:
                            return a_str >= lo_str or a_str <= hi_str
                        return lo_str <= a_str <= hi_str
                return False
            if op == "in":
                return actual in expected if isinstance(expected, list) else False
            return False
        except (ValueError, TypeError):
            return False

    def _infer_capability(self, action: Dict) -> str:
        device_id = action.get("device_id", "").lower()
        if "irrigat" in device_id or "water" in device_id:
            return "irrigate"
        if "fertigat" in device_id or "fertil" in device_id:
            return "fertigate"
        return "irrigate"

    def _check_hard_limits(self, capability: str, params: Dict, device_id: str) -> Tuple[bool, str]:
        limits = HARD_LIMITS.get(capability, {})
        max_dur = limits.get("max_duration_per_use_minutes")
        if max_dur and params.get("duration", 0) > max_dur:
            return False, f"单次灌溉时长 {params['duration']} 分钟超过硬限制 {max_dur} 分钟"
        max_amt = limits.get("max_amount_per_use_kg")
        if max_amt and params.get("amount_kg", 0) > max_amt:
            return False, f"单次施肥量 {params['amount_kg']}kg 超过硬限制 {max_amt}kg"
        min_interval = limits.get("min_interval_seconds", 0)
        if min_interval and device_id in self._execution_history:
            last = self._execution_history[device_id][-1] if self._execution_history[device_id] else None
            if last and (datetime.now() - last).total_seconds() < min_interval:
                return False, f"距上次操作不足 {min_interval} 秒，拒绝重复触发"
        return True, ""

    def _check_constraints(self, constraints: Dict, params: Dict, context: Dict) -> Tuple[bool, str]:
        max_dur = constraints.get("max_duration_per_use")
        if max_dur is not None and params.get("duration", 0) > max_dur:
            return False, f"单次时长 {params['duration']} 分钟超过设定上限 {max_dur} 分钟，需要确认"

        max_daily = constraints.get("max_duration_per_day")
        if max_daily is not None:
            device_id = context.get("device_id", "")
            date_str = datetime.now().strftime("%Y-%m-%d")
            today_used = self._daily_duration.get(device_id, {}).get(date_str, 0)
            if today_used + params.get("duration", 0) > max_daily:
                return False, f"今日累计超过每日上限，需要确认"

        # 最小间隔检查
        min_interval = constraints.get("min_interval_minutes")
        if min_interval:
            device_id = context.get("device_id", "")
            if device_id in self._execution_history and self._execution_history[device_id]:
                last = self._execution_history[device_id][-1]
                elapsed = (datetime.now() - last).total_seconds() / 60
                if elapsed < min_interval:
                    return False, f"距上次操作 {elapsed:.1f} 分钟，不足最小间隔 {min_interval} 分钟，需要确认"

        forbidden = constraints.get("forbidden_hours", [])
        if forbidden and datetime.now().hour in forbidden:
            return False, f"当前时间在禁止时段内，需要确认"

        require_confirm = constraints.get("require_confirm_if", [])
        for expr in require_confirm:
            if self._eval_confirm_expr(expr, params, context):
                return False, f"触发确认条件: {expr}"

        return True, ""

    def _eval_confirm_expr(self, expr: str, params: Dict, context: Dict) -> bool:
        try:
            if " > " in expr:
                field, val = expr.split(" > ")
                return params.get(field.strip(), 0) > float(val.strip())
            if " < " in expr:
                field, val = expr.split(" < ")
                return params.get(field.strip(), 0) < float(val.strip())
            if " >= " in expr:
                field, val = expr.split(" >= ")
                return params.get(field.strip(), 0) >= float(val.strip())
            if " <= " in expr:
                field, val = expr.split(" <= ")
                return params.get(field.strip(), 0) <= float(val.strip())
            if " == " in expr:
                field, val = expr.split(" == ")
                return str(params.get(field.strip(), "")) == val.strip()
            if expr == "weather_forecast_conflict":
                return context.get("weather_conflict", False)
        except Exception as e:
            logger.debug("confirm_expr 解析失败: expr=%r, error=%s", expr, e)
            pass
        return False

    def _apply_ai_enhance(self, ai_config: Dict, proposed: Dict, original: Dict) -> Dict:
        result = dict(proposed)
        can_adjust = ai_config.get("can_adjust", [])
        ranges = ai_config.get("adjust_range", {})
        for field in can_adjust:
            if field in result and field in ranges:
                min_adj, max_adj = ranges[field]
                orig_val = original.get(field, result[field])
                adjusted = result[field]
                clamped = max(orig_val + min_adj, min(orig_val + max_adj, adjusted))
                result[field] = clamped
        return result
