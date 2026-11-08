"""设备指令执行器 — 重试/超时/队列/审计日志"""

import asyncio
import copy
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core.storage_paths import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# 重试配置
MAX_RETRIES = 3
RETRY_DELAYS_SECONDS = [5, 15, 45]
MAX_LOG_ENTRIES = int(os.getenv("DEVICE_LOG_MAX_ENTRIES", "1000"))


class _PendingFileLock:
    """哥哥，用操作系统文件锁保护跨进程队列，不改写锁文件内容。"""

    def __init__(self, path: str):
        self.path = path
        self._handle = None

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


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
        with self._pending_transaction():
            pass

    @staticmethod
    def _validate_username(username: str) -> bool:
        """验证用户名不包含路径遍历字符"""
        if not username or not isinstance(username, str):
            return False
        forbidden = {"..", "/", "\\", "\0"}
        return not any(c in username for c in forbidden)

    # ── 指令执行 ──────────────────────────────

    async def execute(self, device_id: str, command, trigger: str = "manual",
                      rule_id: Optional[str] = None, decision: str = "auto_execute",
                      capability: Optional[str] = None, policy_context: Optional[Dict] = None,
                      skip_policy: bool = False) -> Dict:
        # 哥哥，设备及分区锁覆盖校验、下发和日志，其他进程也必须等待。
        keys = ["device:" + device_id]
        resolved, context = self._resolve_policy_scope(device_id, capability, policy_context)
        if resolved in {"heat", "cool"} and context.get("plot_id") not in (None, ""):
            keys.append("climate:" + str(context.get("plot_id")))
        locks = []
        try:
            for key in sorted(keys):
                lock = _PendingFileLock(self._pending_path() + ".resource." + hashlib.sha256(key.encode()).hexdigest() + ".lock")
                while not lock.acquire():
                    await asyncio.sleep(0.01)
                locks.append(lock)
            return await self._execute_locked(device_id, command, trigger, rule_id, decision,
                                              capability, policy_context, skip_policy)
        finally:
            for lock in reversed(locks):
                lock.release()

    async def reconcile_uncertain(self, device_id: str) -> Dict:
        """哥哥，只读取设备状态；证据必须关联原作业，核对不重发指令。"""
        key = hashlib.sha256(("device:" + device_id).encode()).hexdigest()
        lock = _PendingFileLock(self._pending_path() + ".resource." + key + ".lock")
        while not lock.acquire():
            await asyncio.sleep(0.01)
        try:
            marker = self._pending_path() + ".uncertain." + hashlib.sha256(device_id.encode()).hexdigest()
            try:
                with open(marker, encoding="utf-8") as handle:
                    record = json.load(handle)
                execution_id = record.get("execution_id")
                if record.get("device_id") != device_id or not isinstance(execution_id, str) or not execution_id:
                    raise ValueError("标记缺少执行编号")
            except FileNotFoundError:
                return {"success": False, "message": "没有可核对的未知执行记录"}
            except (ValueError, TypeError, OSError, json.JSONDecodeError):
                return {"success": False, "message": "未知执行记录无法读取或缺少执行编号"}
            try:
                state = await self.registry.read_state(device_id)
                observed = datetime.fromisoformat(str(state.get("_observed_at", "")).replace("Z", "+00:00"))
                now = datetime.now(timezone.utc) if observed.tzinfo else datetime.now()
                age = (now - observed).total_seconds()
                if (state.get("error") or state.get("_error") or state.get("_quality") != "good"
                    or not 0 <= age <= 300):
                    raise ValueError("状态不可用或已过期")
            except Exception:
                return {"success": False, "message": "设备状态不可用、时间不明或已过期"}
            if state.get("status") not in {"standby", "powered_off", "off", "idle"}:
                return {"success": False, "message": "设备未确认停止或运行状态不明"}
            history = state.get("run_history")
            entry = history.get(execution_id) if isinstance(history, dict) else None
            ended = isinstance(entry, dict) and entry.get("reason") in {"completed", "interrupted", "aborted", "stopped"}
            ack = state.get("_stop_ack")
            if not isinstance(ack, dict):
                ack = record.get("stop_ack")
            stopped = isinstance(ack, dict) and ack.get("acknowledged") is True and ack.get("execution_id") == execution_id
            if not ended and not stopped:
                return {"success": False, "message": "缺少与原执行编号匹配的结束记录或停止回执"}
            with self._pending_transaction():
                changed = False
                for action in self.pending_actions:
                    if (action.get("status") == "uncertain" and action.get("device_id") == device_id
                        and (action.get("execution_id") == execution_id
                             or (not action.get("execution_id") and action.get("id") == record.get("pending_action_id")))):
                        action["status"] = "reconciled"
                        action["updated_at"] = datetime.now().isoformat()
                        action["reconcile_reason"] = "设备结束记录" if ended else "设备停止回执"
                        changed = True
                if changed:
                    self._save_pending()
                os.remove(marker)
            return {"success": True, "status": "reconciled", "execution_id": execution_id,
                    "message": "已核对设备状态，原指令不会重新下发"}
        finally:
            lock.release()

    async def _execute_locked(self, device_id: str, command,
                      trigger: str = "manual",
                      rule_id: Optional[str] = None,
                      decision: str = "auto_execute",
                      capability: Optional[str] = None,
                      policy_context: Optional[Dict] = None,
                      skip_policy: bool = False) -> Dict:
        from devices.base import DeviceResult

        # 哥哥，设备不存在是明确执行失败，不是契约不足或可确认的策略拒绝。
        get_driver = getattr(self.registry, "get_driver", None)
        if callable(get_driver) and get_driver(device_id) is None:
            if callable(getattr(self.registry, "discover_all", None)):
                await self.registry.discover_all()
            if get_driver(device_id) is None:
                missing = DeviceResult(False, device_id, getattr(command, "command", "unknown"),
                                       message="设备不存在或未发现对应驱动", error_code="DEVICE_NOT_FOUND")
                self._write_log(device_id, command, missing, trigger, rule_id, decision, 0)
                return {"success": False, "result": missing, "attempts": 0,
                        "log_entry": self._make_log_entry(device_id, command, missing, trigger, rule_id, decision, 0)}

        # 哥哥，效果复查的停止只能针对刚刚核验的同一轮，锁内再次读取避免误停新作业。
        if trigger == "effect_check" and getattr(command, "command", "") in {"stop", "power_off", "shutdown"}:
            try:
                state = await self.registry.read_state(device_id)
                observed = datetime.fromisoformat(str(state.get("_observed_at", "")).replace("Z", "+00:00"))
                now = datetime.now(timezone.utc) if observed.tzinfo else datetime.now()
                age = (now - observed).total_seconds()
                run_id = getattr(command, "execution_id", None)
                valid = (bool(run_id) and not state.get("error") and not state.get("_error")
                         and state.get("_quality") == "good" and 0 <= age <= 300
                         and state.get("status") == "running"
                         and (state.get("run_id") or state.get("execution_id")) == run_id)
            except Exception:
                valid = False
            if not valid:
                blocked = DeviceResult(False, device_id, command.command, message="设备状态未知或运行编号已变化，未下发停止", error_code="RUN_MISMATCH")
                return {"success": False, "result": blocked, "attempts": 0, "decision": "rejected", "message": blocked.message}

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
            automatic = trigger in {"autonomous", "rule", "task_scheduler", "camera", "agent", "task"} or bool(resolved_context.get("automatic"))
            resolved_context["automatic"] = automatic
            policy_result = SafetyPolicyService(self.username).evaluate(
                device_id=device_id,
                capability=resolved_capability,
                params=getattr(command, "params", {}) or {},
                command=getattr(command, "command", "start"),
                context=resolved_context,
            )
            if policy_result.decision != RuleDecision.REJECTED and getattr(command, "command", "start") not in {"stop", "power_off", "shutdown"}:
                from core.device_rule_engine import RuleEngine
                engine = RuleEngine(self.username)
                for configured in engine.rules:
                    if not configured.get("enabled", True) or configured.get("action", {}).get("device_id") != device_id:
                        continue
                    mode = configured.get("execution_mode", "auto")
                    ok, reason = engine._check_constraints(configured.get("constraints", {}), policy_result.params, device_id, resolved_context)
                    if mode == "notify":
                        policy_result.decision, policy_result.reason = RuleDecision.REJECTED, "设备规则仅允许通知，不能执行"
                        break
                    if (not ok or mode == "confirm") and trigger != "confirmed":
                        policy_result.decision, policy_result.reason = RuleDecision.NEED_CONFIRM, reason or "设备规则要求确认"
            if policy_result.decision != RuleDecision.REJECTED and getattr(command, "command", "start") == "start":
                from core.device_registry_factory import load_custom_devices
                from core.device_execution_contract import unattended_contract_error
                configs = load_custom_devices(self.username)
                config = next ( (item for item in configs if item.get("device_id") == device_id), None )
                driver = self.registry.get_driver(device_id) if hasattr(self.registry, "get_driver") else None
                if config is None and getattr(driver, "driver_name", "") == "simulator":
                    config = {"driver": "simulator"}
                if automatic or config is not None:
                    error = unattended_contract_error(config or {}, "start", policy_result.params)
                    if error:
                        policy_result.decision = RuleDecision.REJECTED
                        policy_result.reason = "设备无人值守契约不足：" + error
                if policy_result.decision != RuleDecision.REJECTED and resolved_capability in {"heat", "cool"}:
                    if resolved_context.get("plot_id") in (None, ""):
                        policy_result.decision, policy_result.reason = RuleDecision.REJECTED, "缺少地块，无法确认冷热互斥"
                    else:
                        opposite = "cool" if resolved_capability == "heat" else "heat"
                        for peer in configs:
                            if (peer.get("device_id") == device_id or opposite not in peer.get("capabilities", [])
                                or str(peer.get("plot_id")) != str(resolved_context.get("plot_id"))
                                or (peer.get("zone_id") and resolved_context.get("zone_id")
                                    and str(peer.get("zone_id")) != str(resolved_context.get("zone_id")))):
                                continue
                            try:
                                state = await self.registry.read_state(peer["device_id"])
                            except Exception:
                                state = {}
                            try:
                                observed = datetime.fromisoformat(str(state.get("_observed_at", "")).replace("Z", "+00:00"))
                                now = datetime.now(timezone.utc) if observed.tzinfo else datetime.now()
                                age = (now - observed).total_seconds()
                                fresh = 0 <= age <= 300
                            except (AttributeError, TypeError, ValueError):
                                fresh = False
                            if (not isinstance(state, dict) or state.get("error") or state.get("_error")
                                or state.get("_quality") != "good" or not fresh
                                or state.get("status") not in {"standby", "powered_off", "off", "idle"}):
                                policy_result.decision, policy_result.reason = RuleDecision.REJECTED, "相反冷热设备运行中或状态未知"
                                break
            policy_decision = policy_result.decision
            if policy_decision == RuleDecision.NEED_CONFIRM and trigger == "confirmed":
                policy_decision = RuleDecision.AUTO_EXECUTE
            elif (
                policy_decision == RuleDecision.AUTO_EXECUTE
                and trigger != "confirmed"
                and getattr(command, "command", "start") not in {
                    "stop", "power_off", "shutdown"
                }
            ):
                # 低自主模式可以把原本安全的操作降级为确认；但高自主模式
                # 不能跳过用户在安全策略中明确要求的确认。
                from app.agent.config import get_autonomy_level

                autonomy = get_autonomy_level(self.username)
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

        # 哥哥，领取时落库的运行标识随下发传递，进程中断重试也不重启同一轮作业。
        task_id = (policy_context or {}).get("task_id")
        if task_id and getattr(command, "command", "") == "start":
            from core.database.repository.planting import PlantingTaskRepository
            from core.database.repository.users import UserRepository

            user = UserRepository().get_by_username(self.username)
            try:
                task = PlantingTaskRepository().find_one(id=int(task_id), user_id=user.id) if user else None
            except (ValueError, TypeError):
                task = None
            if task and task.execution_state:
                runtime = json.loads(task.execution_state)
                command.execution_id = runtime.get("run_id")

        driver = get_driver(device_id) if callable(get_driver) else None
        if (trigger != "effect_check" and getattr(command, "command", "") == "stop"
            and getattr(driver, "driver_name", "") == "simulator"):
            # 哥哥，普通停止的操作编号不是目标运行编号；锁内取当前轮，旧状态无编号也允许停机。
            command = copy.copy(command)
            try:
                current_state = await self.registry.read_state(device_id)
                command.execution_id = current_state.get("run_id") if isinstance(current_state, dict) else None
            except Exception:
                command.execution_id = None
        if not getattr(command, "execution_id", None):
            driver = get_driver(device_id) if callable(get_driver) else None
            if not (getattr(driver, "driver_name", "") == "simulator" and command.command in {"stop", "power_off", "shutdown"}):
                command.execution_id = uuid.uuid4().hex
        pending_id = (policy_context or {}).get("pending_action_id")
        # 哥哥，下发前留下持久标记；进程崩溃或回执不明时后续启动不能猜测空闲。
        marker = None
        if getattr(command, "command", "") == "start" and not skip_policy:
            marker = self._pending_path() + ".uncertain." + hashlib.sha256(device_id.encode()).hexdigest()
            if os.path.exists(marker):
                blocked = DeviceResult(False, device_id, command.command,
                                       message="上次启动结果未知，请检查设备后处理", error_code="EXECUTION_UNCERTAIN")
                return {"success": False, "result": blocked, "attempts": 0,
                        "decision": "rejected", "message": blocked.message}
            with open(marker, "x", encoding="utf-8") as handle:
                json.dump({"device_id": device_id, "execution_id": command.execution_id,
                           "pending_action_id": pending_id,
                           "created_at": datetime.now().isoformat()}, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
        if pending_id:
            with self._pending_transaction():
                for action in self.pending_actions:
                    if action.get("id") == pending_id and action.get("status") == "executing":
                        action["execution_id"] = command.execution_id
                        self._save_pending()
                        break
        last_result = None

        for attempt in range(MAX_RETRIES):
            try:
                last_result = await self.registry.execute(device_id, command)

                if last_result.success:
                    if (getattr(command, "command", "") in {"stop", "power_off", "shutdown"}
                        and getattr(last_result, "acknowledged", False)
                        and getattr(last_result, "execution_id", None) == command.execution_id):
                        stop_marker = self._pending_path() + ".uncertain." + hashlib.sha256(device_id.encode()).hexdigest()
                        try:
                            with open(stop_marker, encoding="utf-8") as handle:
                                original = json.load(handle)
                            if original.get("device_id") == device_id and original.get("execution_id") == command.execution_id:
                                original["stop_ack"] = {"acknowledged": True, "execution_id": command.execution_id}
                                temp = stop_marker + "." + uuid.uuid4().hex + ".tmp"
                                try:
                                    with open(temp, "w", encoding="utf-8") as handle:
                                        json.dump(original, handle, ensure_ascii=False)
                                        handle.flush()
                                        os.fsync(handle.fileno())
                                    os.replace(temp, stop_marker)
                                finally:
                                    if os.path.exists(temp):
                                        os.remove(temp)
                        except (OSError, ValueError, TypeError):
                            logger.warning("无法保存关联停止回执，仍需人工核对设备状态")
                    if marker and not pending_id and os.path.exists(marker):
                        os.remove(marker)
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

                if not getattr(last_result, "retry_safe", False):
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
                break

            if attempt < MAX_RETRIES - 1 and getattr(last_result, "retry_safe", False):
                delay = RETRY_DELAYS_SECONDS[min(attempt, len(RETRY_DELAYS_SECONDS) - 1)]
                logger.warning("设备 %s 执行失败（第%d次），%d秒后重试", device_id, attempt + 1, delay)
                await asyncio.sleep(delay)

        if marker and last_result and last_result.error_code == "DEVICE_NOT_FOUND" and os.path.exists(marker):
            os.remove(marker)

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
            self._write_log(device_id, command, last_result, trigger, rule_id, decision, attempt + 1)
        except Exception:
            logger.warning("最终失败日志写入失败")
        return {
            "success": False,
            "result": last_result,
            "attempts": attempt + 1,
            "log_entry": self._make_log_entry(device_id, command, last_result, trigger, rule_id, decision, attempt + 1),
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
        rule_id = str(action.get("rule_id") or "")
        task_id = str((action.get("policy_context") or {}).get("task_id") or "")
        with self._pending_transaction():
            # 哥哥，查重和追加共用跨进程锁，确认中的操作也只能保留一个编号。
            if rule_id or task_id:
                for pending in self.pending_actions:
                    if (
                        pending.get("status") in {"pending", "executing", "failed", "uncertain"}
                        and str(pending.get("rule_id") or "") == rule_id
                        and str((pending.get("policy_context") or {}).get("task_id") or "") == task_id
                        and pending.get("device_id") == action.get("device_id")
                        and pending.get("command", "start") == action.get("command", "start")
                        and (pending.get("params") or {}) == (action.get("params") or {})
                    ):
                        return pending["id"]
            action["id"] = f"pending_{uuid.uuid4().hex[:8]}"
            action["created_at"] = datetime.now().isoformat()
            action["status"] = "pending"
            self.pending_actions.append(action)
            self._save_pending()
        return action["id"]

    def list_pending(self) -> List[Dict]:
        """返回仍可由用户处理的操作副本，避免调用方意外修改内存状态。"""
        with self._pending_transaction():
            return copy.deepcopy([
                action for action in self.pending_actions
                if action.get("status") in {"pending", "failed", "uncertain"}
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

        with self._pending_transaction():
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
        with self._pending_transaction():
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
                    execution_lock = self._execution_lock(action_id)
                    if not execution_lock.acquire():
                        return {"success": False, "message": "操作不存在或已处理"}
                    action["status"] = "executing"
                    action["updated_at"] = datetime.now().isoformat()
                    action["attempt_count"] = int(action.get("attempt_count", 0)) + 1
                    try:
                        self._save_pending()
                    except BaseException:
                        execution_lock.release()
                        raise
                    # 在锁外执行，避免设备重试期间长期持锁。
                    break_action = copy.deepcopy(action)
                    break
            else:
                return {"success": False, "message": "操作不存在或已处理"}

        try:
            from devices.base import DeviceCommand

            cmd = DeviceCommand(
                command=break_action.get("command", "start"),
                params=break_action.get("params", {}),
            )
            try:
                result = self.execute_sync(
                    break_action["device_id"], cmd,
                    trigger="confirmed", decision="auto_execute",
                    rule_id=break_action.get("rule_id"),
                    capability=break_action.get("capability"),
                    policy_context={**(break_action.get("policy_context") or {}), "pending_action_id": action_id},
                )
            except Exception as exc:
                logger.exception("待确认操作执行异常: %s", action_id)
                result = {"success": False, "message": str(exc)}

            succeeded = bool(result.get("success"))
            result_detail = result.get("result")
            uncertain = not succeeded and (
                (result.get("attempts", 0) > 0 and getattr(result_detail, "error_code", "") != "DEVICE_NOT_FOUND"
                 and not getattr(result_detail, "retry_safe", False))
                or (result_detail is None and "attempts" not in result)
            )
            failure_message = (
                result.get("message")
                or getattr(result_detail, "message", "")
                or "设备执行失败，可检查设备状态后重试"
            )
            with self._pending_transaction():
                for action in self.pending_actions:
                    if action.get("id") == action_id:
                        action["status"] = "executed" if succeeded else ("uncertain" if uncertain else "failed")
                        action["updated_at"] = datetime.now().isoformat()
                        if succeeded:
                            action["completed_at"] = action["updated_at"]
                            action.pop("last_error", None)
                        else:
                            action["last_error"] = failure_message
                        self._save_pending()
                        if succeeded and cmd.command == "start":
                            marker = self._pending_path() + ".uncertain." + hashlib.sha256(break_action["device_id"].encode()).hexdigest()
                            try:
                                with open(marker, encoding="utf-8") as handle:
                                    original = json.load(handle)
                                if original.get("execution_id") == cmd.execution_id and original.get("pending_action_id") == action_id:
                                    os.remove(marker)
                            except (OSError, ValueError, TypeError):
                                logger.warning("无法清理已完成操作的标记，请核对设备状态")
                        break
            result["action_status"] = "executed" if succeeded else ("uncertain" if uncertain else "failed")
            if not succeeded:
                result.setdefault("message", failure_message)
            return result
        finally:
            execution_lock.release()

    def reject_pending(self, action_id: str) -> bool:
        with self._pending_transaction():
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

    @contextmanager
    def _pending_transaction(self):
        with self._pending_lock:
            queue_lock = _PendingFileLock(self._pending_path() + ".lock")
            deadline = time.monotonic() + 10
            while not queue_lock.acquire():
                if time.monotonic() >= deadline:
                    raise RuntimeError("待确认队列正忙，请稍后重试")
                time.sleep(0.01)
            try:
                # 哥哥，每次都在跨进程锁内重新读取，避免旧实例覆盖别人的操作。
                self._load_pending()
                yield
            finally:
                queue_lock.release()

    def _execution_lock(self, action_id: str):
        action_key = hashlib.sha256(str(action_id).encode("utf-8")).hexdigest()
        return _PendingFileLock(
            self._pending_path() + f".{action_key}.executing.lock"
        )

    def _load_pending(self):
        path = self._pending_path()
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list) or not all(
                    isinstance(action, dict) for action in data
                ):
                    logger.warning("pending 文件格式异常(非列表)，保留备份并重置")
                    self._backup_corrupted(path)
                    self.pending_actions = []
                    return
                self.pending_actions = data
                recovered = False
                for action in self.pending_actions:
                    if action.get("status") == "executing":
                        execution_lock = self._execution_lock(action.get("id", ""))
                        if execution_lock.acquire():
                            try:
                                # 哥哥，操作锁随进程退出释放，只有真正中断才能恢复。
                                action["status"] = "uncertain"
                                action["last_error"] = "上次执行被中断，结果未知，请检查设备状态"
                                action["updated_at"] = datetime.now().isoformat()
                                recovered = True
                            finally:
                                execution_lock.release()
                if recovered:
                    self._save_pending()
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("pending 文件损坏，保留备份并重置")
                self._backup_corrupted(path)
                self.pending_actions = []
        else:
            self.pending_actions = []

    def _save_pending(self):
        path = self._pending_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 原子写入
        tmp_path = path + f".{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.pending_actions, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _backup_corrupted(self, path: str):
        """将损坏的文件备份为 .bak，防止数据完全丢失"""
        try:
            bak_path = path + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.exists(path):
                os.replace(path, bak_path)
                logger.info("已备份损坏文件: %s", bak_path)
        except Exception:
            logger.warning("备份损坏文件失败: %s", path)
