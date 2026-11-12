"""哥哥，对话卡片只保存所属引用，操作候选沿用原设备确认队列。"""

import copy
import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastapi import HTTPException

from core.device_action_schema import ACTION_SCHEMAS, normalize_action
from core.device_executor import DeviceExecutor
from core.device_registry_factory import RegistrySession, load_custom_devices, _validate_username
from core.storage_paths import DEFAULT_DATA_DIR
from devices.registry import DeviceDriverRegistry

logger = logging.getLogger(__name__)
MAX_CHAT_CARDS = 8


def _device_configs(username: str) -> dict[str, dict]:
    _validate_username(username)
    return {item["device_id"]: item for item in load_custom_devices(username)}


def _tracker(username: str):
    from core.planting_tracker import PlantingTracker

    _validate_username(username)
    return PlantingTracker(os.path.join(DEFAULT_DATA_DIR, username))


def _pending_actions(username: str) -> dict[str, dict]:
    executor = DeviceExecutor(DeviceDriverRegistry(), username=username)
    # 哥哥，历史卡也要能查询已执行和已拒绝操作，不能只看可处理队列。
    with executor._pending_transaction():
        return {action["id"]: copy.deepcopy(action) for action in executor.pending_actions if action.get("id")}


def sanitize_chat_cards(username: str, cards: Any) -> list[dict]:
    """哥哥，只保留当前账号的真实引用与白名单字段。"""
    if not isinstance(cards, list):
        return []
    configs = _device_configs(username)
    pending = None
    tracker = None
    result = []
    positions = {}
    for card in cards:
        if not isinstance(card, dict):
            continue
        if card.get("type") == "device":
            device_id = card.get("device_id")
            if not isinstance(device_id, str) or device_id not in configs:
                continue
            clean = {"type": "device", "device_id": device_id}
            action_id = card.get("pending_action_id")
            if action_id is not None:
                if not isinstance(action_id, str) or not action_id:
                    continue
                pending = _pending_actions(username) if pending is None else pending
                if pending.get(action_id, {}).get("device_id") != device_id:
                    continue
                clean["pending_action_id"] = action_id
            key = ("device", device_id)
        elif card.get("type") == "task":
            task_id = card.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                continue
            tracker = _tracker(username) if tracker is None else tracker
            if tracker.get_task_by_id(task_id) is None:
                continue
            clean = {"type": "task", "task_id": task_id}
            key = ("task", task_id)
        else:
            continue
        if key in positions:
            if clean.get("pending_action_id") and not result[positions[key]].get("pending_action_id"):
                result[positions[key]] = clean
            continue
        if len(result) < MAX_CHAT_CARDS:
            positions[key] = len(result)
            result.append(clean)
    return result


def build_chat_cards(username: str, state: Any) -> list[dict]:
    """哥哥，只从结构化执行与任务结果构造卡片，正文不能授予权限。"""
    def value(key, default=None):
        return state.get(key, default) if isinstance(state, dict) else getattr(state, key, default)

    cards = []
    pending = value("pending_action") or {}
    if isinstance(pending, dict) and pending.get("device_id") and pending.get("id"):
        cards.append({"type": "device", "device_id": pending["device_id"], "pending_action_id": pending["id"]})
    command = value("device_command") or {}
    if isinstance(command, dict) and command.get("device_id"):
        cards.append({"type": "device", "device_id": command["device_id"]})
    cards.extend({"type": "device", "device_id": device_id} for device_id in (value("chat_device_ids", []) or []))
    cards.extend({"type": "task", "task_id": str(task_id)} for task_id in (value("chat_task_ids", []) or []) if task_id)
    cards.extend({"type": "task", "task_id": str(reminder["task_id"])}
                 for reminder in (value("reminders", []) or [])
                 if isinstance(reminder, dict) and reminder.get("task_id"))
    return sanitize_chat_cards(username, cards)


def get_device_card(username: str, device_id: str, pending_action_id: str | None = None) -> dict:
    """哥哥，先核对设备归属，再读取本次真实状态与可选操作。"""
    config = _device_configs(username).get(device_id)
    if config is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    pending = None
    if pending_action_id is not None:
        pending = _pending_actions(username).get(pending_action_id)
        if pending is None or pending.get("device_id") != device_id:
            raise HTTPException(status_code=404, detail="关联操作不存在")
    from core.plot_manager import PlotManager

    plot_id = config.get("plot_id")
    plot_name = next((plot.get("name", "") for plot in PlotManager(username).list_plots()
                      if str(plot.get("plot_id")) == str(plot_id)), "") if plot_id is not None else ""
    state = {"online": False, "_fresh": False, "error": "设备离线，当前状态未知"}
    try:
        with RegistrySession(username) as (registry, loop):
            devices = loop.run_until_complete(registry.discover_all())
            device = next((item for item in devices if item.device_id == device_id), None)
            if device is not None:
                observed = loop.run_until_complete(registry.read_state(device_id))
                if isinstance(observed, dict):
                    state = copy.deepcopy(observed)
                    state["online"] = getattr(device.status, "value", device.status) not in {"offline", "error"}
    except Exception:
        logger.exception("读取对话设备卡失败: user=%s device=%s", username, device_id)
    return {
        "device_id": device_id, "name": config.get("name", device_id),
        "plot_id": plot_id, "plot_name": plot_name,
        "capabilities": config.get("capabilities", []), "state": state, "pending_action": pending,
    }


def get_task_card(username: str, task_id: str) -> dict:
    """哥哥，任务卡总是读取当前账号任务，并补充所属设备名称。"""
    tracker = _tracker(username)
    task = tracker.get_task_by_id(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    card = asdict(task)
    config = _device_configs(username).get(task.device_id, {})
    card["device_name"] = config.get("name", "")
    from core.database.repository.reminders import ReminderRepository

    # 哥哥，只读这位用户与这项任务关联的真实下次提醒，不从任务描述猜时间。
    reminders = ReminderRepository().find_by(user_id=tracker._uid, task_id=int(task.id), status="active")
    upcoming = min((reminder for reminder in reminders if reminder.next_trigger is not None),
                   key=lambda reminder: (reminder.next_trigger, reminder.id), default=None)
    card["reminder_time"] = upcoming.next_trigger.isoformat() if upcoming is not None else None
    card["reminder_frequency"] = upcoming.frequency if upcoming is not None else None
    return card


def prepare_device_action(username: str, device_id: str, command: str, params: dict, request_id: str) -> dict:
    """哥哥，只在同一个持久队列事务中准备候选，绝不发送设备命令。"""
    config = _device_configs(username).get(device_id)
    if config is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    try:
        serialized = json.dumps({"device_id": device_id, "command": command, "params": params},
                                sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    executor = DeviceExecutor(DeviceDriverRegistry(), username=username)
    with executor._pending_transaction():
        for action in executor.pending_actions:
            if action.get("chat_request_id") == request_id:
                if action.get("chat_request_fingerprint") != fingerprint:
                    raise HTTPException(status_code=409, detail="请求编号已用于其他操作，请刷新后重试")
                return {"success": True, "pending_action": copy.deepcopy(action)}
        capabilities = [cap for cap in config.get("capabilities", []) if cap in ACTION_SCHEMAS]
        capability = params.get("action") or params.get("capability")
        if capability is None and len(capabilities) == 1:
            capability = capabilities[0]
        if capability not in capabilities:
            raise HTTPException(status_code=422, detail="请指定设备支持的操作能力")
        try:
            if len(serialized.encode("utf-8")) > 8192:
                raise ValueError("操作参数过大，请控制在 8KB 以内")
            for key, value in params.items():
                if key in ACTION_SCHEMAS[capability]["parameters"] and isinstance(value, bool):
                    raise ValueError(f"参数 {key} 必须是数字")
            normalized = normalize_action(capability, command, params)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # 哥哥，查重和追加必须在同一把跨进程队列锁里，不能嵌套 add_pending。
        action = {
            "id": "pending_" + uuid.uuid4().hex[:8], "device_id": device_id,
            "command": command, "params": normalized, "capability": capability,
            "reason": "操作已准备，请核对设备与参数后确认。", "policy_context": {"device_id": device_id},
            "rule_id": None, "status": "pending", "created_at": datetime.now().isoformat(),
            "chat_request_id": request_id, "chat_request_fingerprint": fingerprint,
        }
        executor.pending_actions.append(action)
        executor._save_pending()
        return {"success": True, "pending_action": copy.deepcopy(action)}
