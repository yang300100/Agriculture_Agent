"""设备指令执行器 — 重试/超时/队列/审计日志"""

import asyncio
import copy
import json
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from core.storage_paths import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# 重试配置
MAX_RETRIES = 3
RETRY_DELAYS_SECONDS = [5, 15, 45]
MAX_LOG_ENTRIES = int(os.getenv("DEVICE_LOG_MAX_ENTRIES", "1000"))


class DeviceExecutor:
    """设备指令执行器 — 重试/超时/队列/审计日志"""

    def __init__(self, registry, username: str = "default"):
        self.registry = registry
        # 验证 username 防止路径遍历
        if not self._validate_username(username):
            raise ValueError(f"无效的用户名: {username}（不允许包含路径分隔符）")
        self.username = username
        self.pending_actions: List[Dict] = []
        self._log_lock = threading.Lock()       # 保护日志文件读写
        self._pending_lock = threading.Lock()    # 保护 pending_actions 读写
        self._load_pending()

    @staticmethod
    def _validate_username(username: str) -> bool:
        """验证用户名不包含路径遍历字符"""
        if not username or not isinstance(username, str):
            return False
        forbidden = {"..", "/", "\\", "\0"}
        return not any(c in username for c in forbidden)

    # ── 指令执行 ──────────────────────────────

    async def execute(self, device_id: str, command,
                      trigger: str = "manual",
                      rule_id: Optional[str] = None,
                      decision: str = "auto_execute",
                      capability: Optional[str] = None,
                      policy_context: Optional[Dict] = None,
                      skip_policy: bool = False) -> Dict:
        from devices.base import DeviceResult

        # 最终执行门禁：即使上游遗漏规则评估，所有实际下发仍统一经过
        # 物理上限和用户安全策略。停止命令由策略服务直接放行。
        if not skip_policy:
            from core.device_rule_engine import RuleDecision, apply_autonomy
            from core.device_safety_policy import SafetyPolicyService

            resolved_capability, resolved_context = self._resolve_policy_scope(
                device_id, capability, policy_context
            )
            if not resolved_capability:
                resolved_capability = await self._resolve_registry_capability(
                    device_id
                )
            if not resolved_capability:
                resolved_capability = self._infer_capability_from_device_id(
                    device_id
                )
            policy_result = SafetyPolicyService(self.username).evaluate(
                device_id=device_id,
                capability=resolved_capability,
                params=getattr(command, "params", {}) or {},
                command=getattr(command, "command", "start"),
                context=resolved_context,
            )
            policy_decision = policy_result.decision
            if policy_decision == RuleDecision.NEED_CONFIRM and trigger == "confirmed":
                policy_decision = RuleDecision.AUTO_EXECUTE
            elif (
                policy_decision == RuleDecision.AUTO_EXECUTE
                and trigger != "confirmed"
            ):
                # 低自主模式可以把原本安全的操作降级为确认；但高自主模式
                # 不能跳过用户在安全策略中明确要求的确认。
                autonomy = os.getenv("AUTONOMY_LEVEL", "medium").lower()
                policy_decision = apply_autonomy(policy_decision, autonomy)

            if policy_decision != RuleDecision.AUTO_EXECUTE:
                error_code = (
                    "POLICY_CONFIRM_REQUIRED"
                    if policy_decision == RuleDecision.NEED_CONFIRM
                    else "POLICY_REJECTED"
                )
                blocked = DeviceResult(
                    success=False,
                    device_id=device_id,
                    executed_command=getattr(command, "command", "unknown"),
                    actual_params=policy_result.params,
                    message=policy_result.reason,
                    error_code=error_code,
                )
                pending_id = None
                if policy_decision == RuleDecision.NEED_CONFIRM:
                    pending_id = self.add_pending({
                        "device_id": device_id,
                        "command": getattr(command, "command", "start"),
                        "params": policy_result.params,
                        "capability": resolved_capability,
                        "policy_context": resolved_context,
                        "reason": policy_result.reason,
                        "rule_id": rule_id,
                    })
                self._write_log(
                    device_id, command, blocked, trigger, rule_id,
                    policy_decision, 0,
                )
                return {
                    "success": False,
                    "result": blocked,
                    "attempts": 0,
                    "pending_id": pending_id,
                    "decision": policy_decision,
                    "policy": policy_result.to_dict(),
                    "log_entry": self._make_log_entry(
                        device_id, command, blocked, trigger, rule_id,
                        policy_decision, 0,
                    ),
                }
            decision = policy_decision

        last_result = None

        for attempt in range(MAX_RETRIES):
            try:
                last_result = await self.registry.execute(device_id, command)

                if last_result.success:
                    # 先写入日志，日志写入失败不影响返回成功状态
                    try:
                        self._write_log(device_id, command, last_result,
                                        trigger, rule_id, decision, attempt + 1)
                    except Exception:
                        logger.warning("日志写入失败，但设备指令已成功执行")
                    return {
                        "success": True,
                        "result": last_result,
                        "attempts": attempt + 1,
                        "log_entry": self._make_log_entry(
                            device_id, command, last_result,
                            trigger, rule_id, decision, attempt + 1),
                    }

                if last_result.error_code == "DEVICE_NOT_FOUND":
                    break

            except asyncio.CancelledError:
                logger.warning("设备 %s 执行被取消 (attempt %d)", device_id, attempt + 1)
                raise
            except Exception as e:
                last_result = DeviceResult(
                    success=False, device_id=device_id,
                    executed_command=command.command,
                    message=str(e), error_code="EXCEPTION",
                )

            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS_SECONDS[min(attempt, len(RETRY_DELAYS_SECONDS) - 1)]
                logger.warning("设备 %s 执行失败（第%d次），%d秒后重试", device_id, attempt + 1, delay)
                await asyncio.sleep(delay)

        # 所有重试失败，写日志（日志写失败不覆盖执行结果）
        if last_result is None:
            from devices.base import DeviceResult
            last_result = DeviceResult(
                success=False, device_id=device_id,
                executed_command=getattr(command, 'command', 'unknown'),
                message="所有重试均失败且无结果",
                error_code="ALL_RETRIES_FAILED",
            )
        try:
            self._write_log(device_id, command, last_result, trigger, rule_id, decision, MAX_RETRIES)
        except Exception:
            logger.warning("最终失败日志写入失败")
        return {
            "success": False,
            "result": last_result,
            "attempts": MAX_RETRIES,
            "log_entry": self._make_log_entry(device_id, command, last_result, trigger, rule_id, decision, MAX_RETRIES),
        }

    def execute_sync(self, device_id: str, command,
                     trigger: str = "manual",
                     rule_id: Optional[str] = None,
                     decision: str = "auto_execute",
                     loop=None,
                     capability: Optional[str] = None,
                     policy_context: Optional[Dict] = None,
                     skip_policy: bool = False) -> Dict:
        """同步执行设备指令。

        Args:
            loop: 可选，外部传入的 event loop。不传入时自动获取或创建。
        """
        created_loop = False
        if loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                created_loop = True
        try:
            return loop.run_until_complete(
                self.execute(
                    device_id, command, trigger, rule_id, decision,
                    capability, policy_context, skip_policy,
                ))
        finally:
            if created_loop and loop is not None:
                try:
                    if not loop.is_closed():
                        loop.close()
                except Exception:
                    pass
                try:
                    asyncio.set_event_loop(None)
                except Exception:
                    pass

    # ── 待确认操作队列 ────────────────────────

    def add_pending(self, action: Dict) -> str:
        action = copy.deepcopy(action)
        action["id"] = f"pending_{uuid.uuid4().hex[:8]}"
        action["created_at"] = datetime.now().isoformat()
        action["status"] = "pending"
        with self._pending_lock:
            self.pending_actions.append(action)
            self._save_pending()
        return action["id"]

    def list_pending(self) -> List[Dict]:
        """返回仍可由用户处理的操作副本，避免调用方意外修改内存状态。"""
        with self._pending_lock:
            return copy.deepcopy([
                action for action in self.pending_actions
                if action.get("status") in {"pending", "failed"}
            ])

    def update_pending(self, action_id: str, params: Dict) -> Dict:
        """更新待确认参数；真正执行时仍会重新经过完整安全策略。"""
        if not isinstance(params, dict):
            return {"success": False, "message": "操作参数必须是 JSON 对象"}
        try:
            serialized = json.dumps(params, ensure_ascii=False)
        except (TypeError, ValueError):
            return {"success": False, "message": "操作参数包含无法保存的值"}
        if len(serialized.encode("utf-8")) > 8192:
            return {"success": False, "message": "操作参数过大，请控制在 8KB 以内"}

        with self._pending_lock:
            for action in self.pending_actions:
                if (
                    action.get("id") == action_id
                    and action.get("status") in {"pending", "failed"}
                ):
                    action["params"] = copy.deepcopy(params)
                    action["status"] = "pending"
                    action["updated_at"] = datetime.now().isoformat()
                    action.pop("last_error", None)
                    self._save_pending()
                    return {"success": True, "action": copy.deepcopy(action)}
        return {"success": False, "message": "操作不存在或当前不可修改"}

    def confirm_pending(self, action_id: str) -> Dict:
        with self._pending_lock:
            for action in self.pending_actions:
                if (
                    action.get("id") == action_id
                    and action.get("status") in {"pending", "failed"}
                ):
                    device_id = action.get("device_id", "")
                    if not device_id:
                        action["status"] = "failed"
                        action["last_error"] = "待确认操作缺少 device_id"
                        action["updated_at"] = datetime.now().isoformat()
                        self._save_pending()
                        return {
                            "success": False,
                            "message": action["last_error"],
                            "action_status": "failed",
                        }
                    action["status"] = "executing"
                    action["updated_at"] = datetime.now().isoformat()
                    action["attempt_count"] = int(action.get("attempt_count", 0)) + 1
                    self._save_pending()
                    # 在锁外执行，避免设备重试期间长期持锁。
                    break_action = copy.deepcopy(action)
                    break
            else:
                return {"success": False, "message": "操作不存在或已处理"}

        from devices.base import DeviceCommand

        cmd = DeviceCommand(
            command=break_action.get("command", "start"),
            params=break_action.get("params", {}),
        )
        try:
            result = self.execute_sync(
                break_action["device_id"], cmd,
                trigger="confirmed", decision="auto_execute",
                capability=break_action.get("capability"),
                policy_context=break_action.get("policy_context"),
            )
        except Exception as exc:
            logger.exception("待确认操作执行异常: %s", action_id)
            result = {"success": False, "message": str(exc)}

        succeeded = bool(result.get("success"))
        result_detail = result.get("result")
        failure_message = (
            result.get("message")
            or getattr(result_detail, "message", "")
            or "设备执行失败，可检查设备状态后重试"
        )
        with self._pending_lock:
            for action in self.pending_actions:
                if action.get("id") == action_id:
                    action["status"] = "executed" if succeeded else "failed"
                    action["updated_at"] = datetime.now().isoformat()
                    if succeeded:
                        action["completed_at"] = action["updated_at"]
                        action.pop("last_error", None)
                    else:
                        action["last_error"] = failure_message
                    self._save_pending()
                    break
        result["action_status"] = "executed" if succeeded else "failed"
        if not succeeded:
            result.setdefault("message", failure_message)
        return result

    def reject_pending(self, action_id: str) -> bool:
        with self._pending_lock:
            for action in self.pending_actions:
                if (
                    action.get("id") == action_id
                    and action.get("status") in {"pending", "failed"}
                ):
                    action["status"] = "rejected"
                    action["updated_at"] = datetime.now().isoformat()
                    self._save_pending()
                    return True
        return False

    # ── 日志 ──────────────────────────────────

    def record_decision(self, device_id: str, command, decision: str,
                        reason: str, trigger: str = "rule",
                        rule_id: Optional[str] = None,
                        add_pending: bool = False,
                        capability: Optional[str] = None,
                        policy_context: Optional[Dict] = None) -> Dict:
        """记录未执行的确认、拒绝或通知决策，并可加入待确认队列。"""
        from devices.base import DeviceResult

        blocked = DeviceResult(
            success=False,
            device_id=device_id,
            executed_command=getattr(command, "command", "unknown"),
            actual_params=getattr(command, "params", {}) or {},
            message=reason,
            error_code=(
                "POLICY_CONFIRM_REQUIRED"
                if decision == "need_confirm"
                else "POLICY_REJECTED"
            ),
        )
        pending_id = None
        if add_pending and decision == "need_confirm":
            pending_id = self.add_pending({
                "device_id": device_id,
                "command": getattr(command, "command", "start"),
                "params": getattr(command, "params", {}) or {},
                "capability": capability,
                "policy_context": dict(policy_context or {}),
                "reason": reason,
                "rule_id": rule_id,
            })
        self._write_log(
            device_id, command, blocked, trigger, rule_id, decision, 0
        )
        return {"success": False, "pending_id": pending_id, "result": blocked}

    def _resolve_policy_scope(self, device_id: str, capability: Optional[str],
                              context: Optional[Dict]) -> tuple[str, Dict]:
        """从设备配置解析能力、地块和预留分区范围。"""
        resolved_context = dict(context or {})
        resolved_capability = str(capability or "").lower()
        try:
            from core.device_registry_factory import load_custom_devices

            config = next(
                (item for item in load_custom_devices(self.username)
                 if item.get("device_id") == device_id),
                {},
            )
            if resolved_context.get("plot_id") in (None, ""):
                resolved_context["plot_id"] = config.get("plot_id")
            if resolved_context.get("zone_id") in (None, ""):
                resolved_context["zone_id"] = config.get("zone_id")
            if not resolved_capability:
                candidates = [
                    item for item in config.get("capabilities", [])
                    if item not in {"read_sensor", "capture"}
                ]
                if candidates:
                    resolved_capability = candidates[0]
        except Exception:
            pass
        return resolved_capability, resolved_context

    async def _resolve_registry_capability(self, device_id: str) -> str:
        """从驱动注册表声明读取能力，避免旧调用遗漏 capability。"""
        try:
            from core.device_safety_policy import ABSOLUTE_CEILINGS

            devices = await self.registry.discover_all()
            target = next(
                (item for item in devices if item.device_id == device_id),
                None,
            )
            if not target:
                return ""
            for capability in getattr(target, "capabilities", []):
                value = str(getattr(capability, "value", capability)).lower()
                if value in ABSOLUTE_CEILINGS:
                    return value
        except Exception as exc:
            logger.debug("从设备注册表解析能力失败: %s", exc)
        return ""

    @staticmethod
    def _infer_capability_from_device_id(device_id: str) -> str:
        """仅供旧设备兼容；新设备应显式声明 capability。"""
        lowered = str(device_id or "").lower()
        keyword_map = (
            (("fertigat", "fertil"), "fertigate"),
            (("ventilat", "fan"), "ventilate"),
            (("shade",), "shade"),
            (("light", "lamp"), "light"),
            (("heat",), "heat"),
            (("cool",), "cool"),
            (("irrigat", "water", "valve"), "irrigate"),
        )
        for keywords, capability in keyword_map:
            if any(keyword in lowered for keyword in keywords):
                return capability
        return ""

    def get_logs(self, limit: int = 50) -> List[Dict]:
        """从DB读取设备操作日志"""
        from core.database.repository.devices import DeviceLogRepository
        from core.database.repository.users import UserRepository
        user_repo = UserRepository()
        user = user_repo.get_by_username(self.username)
        if not user:
            user = user_repo.create(username=self.username, password_hash="")
        log_repo = DeviceLogRepository()
        db_logs = log_repo.get_recent(user.id, limit)
        return [{
            "timestamp": log.created_at.isoformat() if log.created_at else "",
            "device_id": log.device_id,
            "command": log.command,
            "params": json.loads(log.params) if log.params else {},
            "trigger": log.trigger,
            "rule_id": log.rule_id,
            "decision": log.decision,
            "success": bool(log.success),
            "attempts": log.attempts,
            "message": log.message,
            "error_code": log.error_code or "",
        } for log in db_logs]

    def _write_log(self, device_id, command, result, trigger, rule_id, decision, attempts):
        """写入操作日志到DB"""
        from core.database.repository.devices import DeviceLogRepository
        from core.database.repository.users import UserRepository
        user_repo = UserRepository()
        user = user_repo.get_by_username(self.username)
        if not user:
            user = user_repo.create(username=self.username, password_hash="")
        log_repo = DeviceLogRepository()
        db_rule_id = None
        if isinstance(rule_id, int) or (
            isinstance(rule_id, str) and rule_id.isdigit()
        ):
            db_rule_id = int(rule_id)
        log_repo.create(
            user_id=user.id,
            device_id=device_id,
            command=command.command,
            params=json.dumps(command.params, ensure_ascii=False) if command.params else "{}",
            trigger=trigger,
            rule_id=db_rule_id,
            decision=decision,
            status="success" if result.success else "failed",
            success=1 if result.success else 0,
            attempts=attempts,
            message=result.message or "",
            error_code=result.error_code or "",
        )

    def _make_log_entry(self, device_id, command, result, trigger, rule_id, decision, attempts) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "command": command.command,
            "params": command.params,
            "trigger": trigger,
            "rule_id": rule_id,
            "decision": decision,
            "success": result.success,
            "attempts": attempts,
            "message": result.message,
            "error_code": result.error_code or "",
        }

    def _log_path(self) -> str:
        return os.path.join(DEFAULT_DATA_DIR, self.username, "device_log.json")

    def _pending_path(self) -> str:
        return os.path.join(DEFAULT_DATA_DIR, self.username, "device_pending.json")

    def _load_pending(self):
        path = self._pending_path()
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    logger.warning("pending 文件格式异常(非列表)，保留备份并重置")
                    self._backup_corrupted(path)
                    self.pending_actions = []
                    return
                self.pending_actions = data
                recovered = False
                for action in self.pending_actions:
                    if action.get("status") == "executing":
                        action["status"] = "failed"
                        action["last_error"] = "上次执行被中断，请确认设备状态后重试"
                        action["updated_at"] = datetime.now().isoformat()
                        recovered = True
                if recovered:
                    self._save_pending()
            except Exception:
                logger.warning("pending 文件损坏，保留备份并重置")
                self._backup_corrupted(path)
                self.pending_actions = []

    def _save_pending(self):
        path = self._pending_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 原子写入
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.pending_actions, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.pending_actions, f, ensure_ascii=False, indent=2)

    def _backup_corrupted(self, path: str):
        """将损坏的文件备份为 .bak，防止数据完全丢失"""
        try:
            bak_path = path + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.exists(path):
                os.replace(path, bak_path)
                logger.info("已备份损坏文件: %s", bak_path)
        except Exception:
            logger.warning("备份损坏文件失败: %s", path)
