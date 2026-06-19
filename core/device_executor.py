"""设备指令执行器 — 重试/超时/队列/审计日志"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.getenv("DATA_STORAGE_DIR", "data")

# 重试配置
MAX_RETRIES = 3
RETRY_DELAYS_SECONDS = [5, 15, 45]


class DeviceExecutor:
    """设备指令执行器 — 重试/超时/队列/审计日志"""

    def __init__(self, registry, username: str = "default"):
        self.registry = registry
        self.username = username
        self.pending_actions: List[Dict] = []
        self._load_pending()

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
                    self._write_log(device_id, command, last_result,
                                    trigger, rule_id, decision, attempt + 1)
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

        self._write_log(device_id, command, last_result, trigger, rule_id, decision, MAX_RETRIES)
        return {
            "success": False,
            "result": last_result,
            "attempts": MAX_RETRIES,
            "log_entry": self._make_log_entry(device_id, command, last_result, trigger, rule_id, decision, MAX_RETRIES),
        }

    def execute_sync(self, device_id: str, command,
                     trigger: str = "manual",
                     rule_id: Optional[str] = None,
                     decision: str = "auto_execute") -> Dict:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.execute(device_id, command, trigger, rule_id, decision))

    # ── 待确认操作队列 ────────────────────────

    def add_pending(self, action: Dict) -> str:
        import uuid
        action["id"] = f"pending_{uuid.uuid4().hex[:8]}"
        action["created_at"] = datetime.now().isoformat()
        action["status"] = "pending"
        self.pending_actions.append(action)
        self._save_pending()
        return action["id"]

    def list_pending(self) -> List[Dict]:
        return [a for a in self.pending_actions if a.get("status") == "pending"]

    def confirm_pending(self, action_id: str) -> Dict:
        for action in self.pending_actions:
            if action["id"] == action_id and action["status"] == "pending":
                action["status"] = "confirmed"
                self._save_pending()

                from devices.base import DeviceCommand
                cmd = DeviceCommand(
                    command=action.get("command", "start"),
                    params=action.get("params", {}),
                )
                return self.execute_sync(action["device_id"], cmd, trigger="confirmed", decision="auto_execute")
        return {"success": False, "message": "操作不存在或已处理"}

    def reject_pending(self, action_id: str) -> bool:
        for action in self.pending_actions:
            if action["id"] == action_id and action["status"] == "pending":
                action["status"] = "rejected"
                self._save_pending()
                return True
        return False

    # ── 日志 ──────────────────────────────────

    def get_logs(self, limit: int = 50) -> List[Dict]:
        path = self._log_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, encoding="utf-8") as f:
                logs = json.load(f)
            return logs[-limit:]
        except Exception:
            return []

    def _write_log(self, device_id, command, result, trigger, rule_id, decision, attempts):
        entry = self._make_log_entry(device_id, command, result, trigger, rule_id, decision, attempts)
        path = self._log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        logs = []
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                pass

        logs.append(entry)
        if len(logs) > 1000:
            logs = logs[-1000:]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

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
                    self.pending_actions = json.load(f)
            except Exception:
                self.pending_actions = []

    def _save_pending(self):
        path = self._pending_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.pending_actions, f, ensure_ascii=False, indent=2)
