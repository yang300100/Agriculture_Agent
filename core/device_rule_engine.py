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

logger = logging.getLogger(__name__)

_raw_dir = os.getenv("DATA_STORAGE_DIR", "data")
DEFAULT_DATA_DIR = _raw_dir if _raw_dir else "data"

# ── 全局执行历史 + 线程锁，跨 RuleEngine 实例共享 ──
_global_history: Dict[str, Dict] = {}
_history_lock = threading.Lock()

# ── 代码级硬限制（不可通过规则配置突破）— 唯一权威来源 ──
HARD_LIMITS = {
    "irrigate": {
        "max_duration_per_use_minutes": 120,
        "min_interval_seconds": 10,
    },
    "fertigate": {
        "max_amount_per_use_kg": 50,
        "min_interval_seconds": 10,
    },
    "ventilate": {
        "max_duration_per_use_minutes": 120,
        "min_interval_seconds": 5,
    },
    "light": {
        "max_duration_per_use_minutes": 720,
        "min_interval_seconds": 5,
    },
    "heat": {
        "max_duration_per_use_minutes": 240,
        "min_interval_seconds": 10,
    },
    "cool": {
        "max_duration_per_use_minutes": 240,
        "min_interval_seconds": 10,
    },
}

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
        path = self._rules_path()
        if not os.path.exists(path):
            self.rules = []
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                self.rules = data.get("rules", [])
                if not isinstance(self.rules, list):
                    raise ValueError("规则数据格式错误：rules 字段应为列表")
            logger.info("规则引擎: 已加载 %d 条规则", len(self.rules))
        except Exception as e:
            logger.error("规则加载失败: %s，尝试从备份恢复", e)
            bak_path = path + ".bak"
            if os.path.exists(bak_path):
                try:
                    with open(bak_path, encoding="utf-8") as f:
                        data = json.load(f)
                        self.rules = data.get("rules", [])
                        if not isinstance(self.rules, list):
                            self.rules = []
                    logger.warning("规则引擎: 已从备份恢复 %d 条规则", len(self.rules))
                    # 恢复后立即保存到原文件
                    self._save_rules()
                    return
                except Exception as e2:
                    logger.error("备份恢复也失败: %s", e2)
            # 保留损坏文件作为 .corrupted 备份
            self._backup_corrupted(path)
            self.rules = []

    def _save_rules(self) -> None:
        path = self._rules_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 原子写入：先写临时文件，再原子重命名
        tmp_path = path + ".tmp"
        data = {"rules": self.rules, "updated_at": datetime.now().isoformat()}
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            # 保留最近一次成功保存的副本
            try:
                bak_path = path + ".bak"
                if os.path.exists(path):
                    shutil.copy2(path, bak_path)
            except Exception:
                pass
        except Exception as e:
            logger.error("规则保存失败: %s", e)
            raise

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
            if r["id"] == rule_id:
                return deepcopy(r)
        return None

    def add_rule(self, rule: Dict) -> str:
        if "id" not in rule:
            import uuid
            rule["id"] = f"rule_{uuid.uuid4().hex[:8]}"
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
        return rule["id"]

    def update_rule(self, rule_id: str, updates: Dict) -> bool:
        # 复制 updates 避免修改调用者的数据
        updates = dict(updates)
        # 不允许通过 update 修改规则 ID
        if "id" in updates:
            logger.warning("update_rule 忽略 id 字段 (不允许修改规则ID)")
            del updates["id"]
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

        # 2. 软约束检查 — 显式传递 device_id，不从 context 取
        soft_ok, soft_reason = self._check_constraints(constraints, proposed_params, device_id, context)
        if not soft_ok:
            return RuleDecision.NEED_CONFIRM, soft_reason, proposed_params

        # 3. AI 微调
        ai_enhance = rule.get("ai_enhance", {})
        if ai_enhance.get("enabled", False):
            proposed_params = self._apply_ai_enhance(ai_enhance, proposed_params, action.get("params", {}))

        return RuleDecision.AUTO_EXECUTE, "规则校验通过", proposed_params

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
        device_id = action.get("device_id", "").lower()
        action_type = action.get("command", "").lower()

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
        for field in can_adjust:
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
