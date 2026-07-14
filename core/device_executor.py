"""设备指令执行器 — 重试/超时/队列/审计日志"""

import asyncio
import json
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# DEFAULT_DATA_DIR: 空字符串当作未设置处理
_raw_dir = os.getenv("DATA_STORAGE_DIR", "data")
DEFAULT_DATA_DIR = _raw_dir if _raw_dir else "data"

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
                      decision: str = "auto_execute") -> Dict:
        from devices.base import DeviceResult

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
                     loop=None) -> Dict:
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
                self.execute(device_id, command, trigger, rule_id, decision))
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
        action["id"] = f"pending_{uuid.uuid4().hex[:8]}"
        action["created_at"] = datetime.now().isoformat()
        action["status"] = "pending"
        with self._pending_lock:
            self.pending_actions.append(action)
            self._save_pending()
        return action["id"]

    def list_pending(self) -> List[Dict]:
        with self._pending_lock:
            return [a for a in self.pending_actions if a.get("status") == "pending"]

    def confirm_pending(self, action_id: str) -> Dict:
        with self._pending_lock:
            for action in self.pending_actions:
                if action["id"] == action_id and action["status"] == "pending":
                    action["status"] = "confirmed"
                    self._save_pending()

                    from devices.base import DeviceCommand
                    device_id = action.get("device_id", "")
                    if not device_id:
                        return {"success": False, "message": "待确认操作缺少 device_id"}
                    cmd = DeviceCommand(
                        command=action.get("command", "start"),
                        params=action.get("params", {}),
                    )
                    # 在锁外执行，避免长时间持锁
                    break_action = action
                    break
            else:
                return {"success": False, "message": "操作不存在或已处理"}

        return self.execute_sync(break_action["device_id"], cmd, trigger="confirmed", decision="auto_execute")

    def reject_pending(self, action_id: str) -> bool:
        with self._pending_lock:
            for action in self.pending_actions:
                if action["id"] == action_id and action["status"] == "pending":
                    action["status"] = "rejected"
                    self._save_pending()
                    return True
        return False

    # ── 日志 ──────────────────────────────────

    def get_logs(self, limit: int = 50) -> List[Dict]:
        # 优先从数据库读取
        try:
            from core.database.repository.devices import DeviceLogRepository
            from core.database.repository.users import UserRepository
            user_repo = UserRepository()
            user = user_repo.get_by_username(self.username)
            if user:
                log_repo = DeviceLogRepository()
                db_logs = log_repo.get_recent(user.id, limit)
                if db_logs:
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
        except Exception:
            pass
        # JSON兜底
        path = self._log_path()
        if not os.path.exists(path):
            return []
        try:
            with self._log_lock:
                with open(path, encoding="utf-8") as f:
                    logs = json.load(f)
            if not isinstance(logs, list):
                return []
            return logs[-limit:]
        except Exception:
            return []

    def _write_log(self, device_id, command, result, trigger, rule_id, decision, attempts):
        entry = self._make_log_entry(device_id, command, result, trigger, rule_id, decision, attempts)
        path = self._log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with self._log_lock:
            logs = []
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            logs = data
                except Exception:
                    pass

            logs.append(entry)
            if len(logs) > MAX_LOG_ENTRIES:
                logs = logs[-MAX_LOG_ENTRIES:]

            # 原子写入：先写临时文件，再重命名
            tmp_path = path + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, path)  # 原子操作
            except Exception:
                # 原子替换失败，回退到直接写入
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2)

        # 同步写入数据库
        try:
            from core.database.repository.devices import DeviceLogRepository
            from core.database.repository.users import UserRepository
            user_repo = UserRepository()
            user = user_repo.get_by_username(self.username)
            if user:
                log_repo = DeviceLogRepository()
                log_repo.create(
                    user_id=user.id,
                    device_id=device_id,
                    command=command.command,
                    params=json.dumps(command.params, ensure_ascii=False) if command.params else "{}",
                    trigger=trigger,
                    rule_id=rule_id,
                    decision=decision,
                    status="success" if result.success else "failed",
                    success=1 if result.success else 0,
                    attempts=attempts,
                    message=result.message or "",
                    error_code=result.error_code or "",
                )
        except Exception as e:
            logger.debug("数据库写入操作日志失败: %s", e)

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
                    if isinstance(data, list):
                        self.pending_actions = data
                    else:
                        logger.warning("pending 文件格式异常(非列表)，保留备份并重置")
                        self._backup_corrupted(path)
                        self.pending_actions = []
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
