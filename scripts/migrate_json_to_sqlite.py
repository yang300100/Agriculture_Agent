#!/usr/bin/env python3
"""一次性 JSON → SQLite 数据迁移脚本。会清空DB数据后重新迁移。"""
import json
import os
import sys
from datetime import datetime, date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.database.engine import init_db, Session
from core.database.models import (
    User, FinanceRecord, DeviceConfig, DeviceRule,
    Field, ChatSession, ChatMessage, PlantingPlan, PlantingTask, Reminder,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _ensure_user(session, username: str):
    """确保用户存在，返回 user_id"""
    user = session.query(User).filter(User.username == username).first()
    if not user:
        user = User(username=username, password_hash="")
        session.add(user)
        session.flush()
    return user.id


def _parse_date(s: str):
    """将日期字符串转为 date 对象"""
    if not s:
        return date.today()
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return date.today()


def migrate_all(force: bool = False):
    print("开始迁移 JSON → SQLite...")
    init_db()
    session = Session()
    total = 0

    # ── 1. 用户 ──
    path = os.path.join(DATA_DIR, "users.json")
    if os.path.exists(path):
        existing = {u.username for u in session.query(User).all()}
        with open(path, encoding="utf-8") as f:
            users_data = json.load(f)
        count = 0
        for username, password in users_data.items():
            if username in existing:
                continue
            pwd = password if isinstance(password, str) else password.get("password", "")
            session.add(User(username=username, password_hash=pwd))
            count += 1
        session.flush()
        print(f"  users: {count} 条新增")

    # ── 2. 财务 ──
    if force:
        session.query(FinanceRecord).delete()
    if force or session.query(FinanceRecord).count() == 0:
        count = 0
        for d in os.listdir(DATA_DIR):
            user_dir = os.path.join(DATA_DIR, d)
            if not os.path.isdir(user_dir):
                continue
            for fname, rtype in [("finance_costs.json", "cost"), ("finance_income.json", "income")]:
                fpath = os.path.join(user_dir, fname)
                if not os.path.exists(fpath):
                    continue
                try:
                    with open(fpath, encoding="utf-8") as fh:
                        records = json.load(fh)
                except Exception:
                    continue
                if not records:
                    continue
                uid = _ensure_user(session, d)
                for r in records:
                    session.add(FinanceRecord(
                        user_id=uid,
                        date=_parse_date(r.get("date", "")),
                        crop=r.get("crop", ""),
                        plot=r.get("plot", ""),
                        record_type=rtype,
                        category=r.get("cost_type") or r.get("income_type", ""),
                        item_name=r.get("item_name", ""),
                        quantity=r.get("quantity"),
                        unit=r.get("unit", ""),
                        unit_price=r.get("unit_price"),
                        total_amount=r.get("total_amount", 0),
                        buyer=r.get("buyer", ""),
                        notes=r.get("notes", ""),
                    ))
                    count += 1
        session.flush()
        print(f"  finance_records: {count} 条")
    else:
        print("  finance_records: 已有数据，跳过")

    # ── 3. 设备配置 ──
    if force:
        session.query(DeviceConfig).delete()
    if force or session.query(DeviceConfig).count() == 0:
        count = 0
        for d in os.listdir(DATA_DIR):
            user_dir = os.path.join(DATA_DIR, d)
            if not os.path.isdir(user_dir):
                continue
            fpath = os.path.join(user_dir, "custom_devices.json")
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, encoding="utf-8") as fh:
                    devices = json.load(fh)
            except Exception:
                continue
            if not devices:
                continue
            uid = _ensure_user(session, d)
            for dev in devices:
                session.add(DeviceConfig(
                    user_id=uid,
                    device_id=dev.get("device_id", ""),
                    name=dev.get("name", ""),
                    driver=dev.get("driver", "simulator"),
                    capabilities=json.dumps(dev.get("capabilities", []), ensure_ascii=False),
                    sensors=json.dumps(dev.get("sensors", []), ensure_ascii=False),
                    connection=json.dumps(dev.get("connection", {}), ensure_ascii=False),
                    location=dev.get("location", ""),
                    plot_id=dev.get("plot_id"),
                ))
                count += 1
        session.flush()
        print(f"  device_configs: {count} 条")
    else:
        print("  device_configs: 已有数据，跳过")

    # ── 4. 设备规则 ──
    if force:
        session.query(DeviceRule).delete()
    if force or session.query(DeviceRule).count() == 0:
        count = 0
        for d in os.listdir(DATA_DIR):
            user_dir = os.path.join(DATA_DIR, d)
            if not os.path.isdir(user_dir):
                continue
            fpath = os.path.join(user_dir, "device_rules.json")
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            rules = data.get("rules", []) if isinstance(data, dict) else data
            if not rules:
                continue
            uid = _ensure_user(session, d)
            for rule in rules:
                session.add(DeviceRule(
                    user_id=uid,
                    name=rule.get("name", ""),
                    enabled=1 if rule.get("enabled", True) else 0,
                    conditions=json.dumps(rule.get("trigger", {}).get("conditions", []), ensure_ascii=False),
                    actions=json.dumps(rule.get("action", {}), ensure_ascii=False),
                    constraints=json.dumps(rule.get("constraints", {}), ensure_ascii=False),
                ))
                count += 1
        session.flush()
        print(f"  device_rules: {count} 条")
    else:
        print("  device_rules: 已有数据，跳过")

    # ── 5. 地块 (fields.json 全局) ──
    if force or session.query(Field).count() == 0:
        count = 0
        fpath = os.path.join(DATA_DIR, "fields.json")
        if os.path.exists(fpath):
            try:
                with open(fpath, encoding="utf-8") as fh:
                    fields_data = json.load(fh)
            except Exception:
                fields_data = []
            uid = _ensure_user(session, "default")
            for f in fields_data:
                coords = f.get("coordinates", [])
                history = f.get("history") or f.get("planting_history", [])
                session.add(Field(
                    user_id=uid,
                    name=f.get("name", "未命名地块"),
                    coordinates=json.dumps(coords, ensure_ascii=False),
                    center_lat=f.get("center_lat", 0),
                    center_lon=f.get("center_lon", 0),
                    area_mu=f.get("area_mu", 0),
                    area_m2=f.get("area_m2", 0),
                    soil_type=f.get("soil_type", ""),
                    current_crop=f.get("current_crop", ""),
                    planting_history=json.dumps(history, ensure_ascii=False),
                ))
                count += 1
        session.flush()
        print(f"  fields: {count} 条")
    else:
        print("  fields: 已有数据，跳过")

    # ── 6. 聊天历史 ──
    if force or session.query(ChatSession).count() == 0:
        count = 0
        fpath = os.path.join(DATA_DIR, "chat_history.json")
        if os.path.exists(fpath):
            try:
                with open(fpath, encoding="utf-8") as fh:
                    chat_data = json.load(fh)
            except Exception:
                chat_data = {"sessions": []}
            uid = _ensure_user(session, "default")
            for s in chat_data.get("sessions", []):
                sid = ChatSession(
                    user_id=uid,
                    title=s.get("title", "未命名"),
                )
                session.add(sid)
                session.flush()
                for m in s.get("messages", []):
                    session.add(ChatMessage(
                        session_id=sid.id,
                        role=m.get("role", "user"),
                        content=m.get("content", ""),
                    ))
                count += 1
        session.flush()
        print(f"  chat_sessions: {count} 条")
    else:
        print("  chat_sessions: 已有数据，跳过")

    # ── 7. 种植任务 + 进度 ──
    for d in os.listdir(DATA_DIR):
        user_dir = os.path.join(DATA_DIR, d)
        if not os.path.isdir(user_dir):
            continue
        uid = _ensure_user(session, d)

        # 任务
        tpath = os.path.join(user_dir, "planting_tasks.json")
        if os.path.exists(tpath) and (force or session.query(PlantingTask).filter(PlantingTask.user_id == uid).count() == 0):
            try:
                with open(tpath, encoding="utf-8") as fh:
                    tasks = json.load(fh)
            except Exception:
                tasks = []
            for t in tasks:
                session.add(PlantingTask(
                    user_id=uid,
                    crop=t.get("crop", ""),
                    task_type=t.get("task_type", ""),
                    title=t.get("title", ""),
                    description=t.get("description", ""),
                    status=t.get("status", "待办"),
                    priority=t.get("priority", "medium"),
                    start_date=_parse_date(t.get("start_date", "")),
                    end_date=_parse_date(t.get("end_date", "")) if t.get("end_date") else None,
                    device_id=t.get("device_id"),
                    device_command=t.get("device_command"),
                    device_params=json.dumps(t.get("device_params", {}), ensure_ascii=False) if t.get("device_params") else None,
                    notes=t.get("notes", ""),
                ))
            session.flush()
            print(f"  planting_tasks [{d}]: {len(tasks)} 条")

        # 进度
        ppath = os.path.join(user_dir, "planting_progress.json")
        if os.path.exists(ppath) and (force or session.query(PlantingPlan).filter(PlantingPlan.user_id == uid).count() == 0):
            try:
                with open(ppath, encoding="utf-8") as fh:
                    progresses = json.load(fh)
            except Exception:
                progresses = []
            for p in progresses:
                session.add(PlantingPlan(
                    user_id=uid,
                    crop=p.get("crop", ""),
                    stage=p.get("stage"),
                    stage_number=p.get("stage_number"),
                    total_stages=p.get("total_stages"),
                    start_date=_parse_date(p.get("start_date", "")),
                    expected_end_date=_parse_date(p.get("expected_end_date", "")) if p.get("expected_end_date") else None,
                    progress_percent=p.get("progress_percent", 0),
                    status=p.get("status", "active"),
                ))
            session.flush()
            print(f"  planting_progress [{d}]: {len(progresses)} 条")

    # ── 8. 提醒 ──
    rpath = os.path.join(DATA_DIR, "reminders.json")
    if os.path.exists(rpath) and (force or session.query(Reminder).count() == 0):
        try:
            with open(rpath, encoding="utf-8") as fh:
                reminders = json.load(fh)
        except Exception:
            reminders = []
        if reminders:
            uid = _ensure_user(session, "default")
            for r in reminders:
                session.add(Reminder(
                    user_id=uid,
                    crop=r.get("crop", ""),
                    reminder_type=r.get("reminder_type", ""),
                    task_description=r.get("task_description", ""),
                    frequency=r.get("frequency", "once"),
                    interval_days=r.get("interval_days"),
                    time_of_day=r.get("time_of_day"),
                    status=r.get("status", "active"),
                ))
            session.flush()
            print(f"  reminders: {len(reminders)} 条")

    session.commit()
    session.close()
    print("迁移完成!")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="强制重新迁移（清空已有数据）")
    args = p.parse_args()
    migrate_all(force=args.force)
