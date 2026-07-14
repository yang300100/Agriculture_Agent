#!/usr/bin/env python3
"""一次性 JSON → SQLite 数据迁移脚本。幂等：已存在数据则跳过。"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.database.engine import init_db, Session
from core.database.models import User, FinanceRecord, DeviceConfig, DeviceRule

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def migrate_users():
    session = Session()
    try:
        if session.query(User).count() > 0:
            print("  users: 已有数据，跳过")
            return 0
        path = os.path.join(DATA_DIR, "users.json")
        if not os.path.exists(path):
            print("  users.json: 文件不存在，跳过")
            return 0
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for username, info in data.items():
            user = User(username=username, password_hash=info.get("password", ""))
            session.add(user)
            count += 1
        session.commit()
        print(f"  users: {count} 条")
        return count
    finally:
        session.close()


def migrate_finance():
    session = Session()
    try:
        if session.query(FinanceRecord).count() > 0:
            print("  finance_records: 已有数据，跳过")
            return 0
        count = 0
        for username_dir in os.listdir(DATA_DIR):
            user_dir = os.path.join(DATA_DIR, username_dir)
            if not os.path.isdir(user_dir):
                continue
            user = session.query(User).filter(User.username == username_dir).first()
            uid = user.id if user else None
            for fname, rtype in [("finance_costs.json", "cost"), ("finance_income.json", "income")]:
                path = os.path.join(user_dir, fname)
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as f:
                    records = json.load(f)
                for r in records:
                    fr = FinanceRecord(
                        user_id=uid,
                        date=datetime.fromisoformat(r.get("date", "2000-01-01")).date() if r.get("date") else datetime.now().date(),
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
                    )
                    session.add(fr)
                    count += 1
        session.commit()
        print(f"  finance_records: {count} 条")
        return count
    finally:
        session.close()


def migrate_device_configs():
    session = Session()
    try:
        if session.query(DeviceConfig).count() > 0:
            print("  device_configs: 已有数据，跳过")
            return 0
        count = 0
        for username_dir in os.listdir(DATA_DIR):
            user_dir = os.path.join(DATA_DIR, username_dir)
            if not os.path.isdir(user_dir):
                continue
            user = session.query(User).filter(User.username == username_dir).first()
            uid = user.id if user else None
            path = os.path.join(user_dir, "custom_devices.json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                devices = json.load(f)
            for d in devices:
                dc = DeviceConfig(
                    user_id=uid,
                    device_id=d.get("device_id", ""),
                    name=d.get("name", ""),
                    driver=d.get("driver", "simulator"),
                    capabilities=json.dumps(d.get("capabilities", []), ensure_ascii=False),
                    sensors=json.dumps(d.get("sensors", []), ensure_ascii=False),
                    connection=json.dumps(d.get("connection", {}), ensure_ascii=False),
                    location=d.get("location", ""),
                    plot_id=d.get("plot_id"),
                    initial_state=json.dumps(d.get("initial_state", {}), ensure_ascii=False),
                )
                session.add(dc)
                count += 1
        session.commit()
        print(f"  device_configs: {count} 条")
        return count
    finally:
        session.close()


def migrate_device_rules():
    session = Session()
    try:
        if session.query(DeviceRule).count() > 0:
            print("  device_rules: 已有数据，跳过")
            return 0
        count = 0
        for username_dir in os.listdir(DATA_DIR):
            user_dir = os.path.join(DATA_DIR, username_dir)
            if not os.path.isdir(user_dir):
                continue
            user = session.query(User).filter(User.username == username_dir).first()
            uid = user.id if user else None
            path = os.path.join(user_dir, "device_rules.json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            rules = data.get("rules", []) if isinstance(data, dict) else data
            for rule in rules:
                dr = DeviceRule(
                    user_id=uid,
                    name=rule.get("name", ""),
                    enabled=1 if rule.get("enabled", True) else 0,
                    conditions=json.dumps(rule.get("trigger", {}).get("conditions", []), ensure_ascii=False),
                    actions=json.dumps(rule.get("action", {}), ensure_ascii=False),
                    constraints=json.dumps(rule.get("constraints", {}), ensure_ascii=False),
                )
                session.add(dr)
                count += 1
        session.commit()
        print(f"  device_rules: {count} 条")
        return count
    finally:
        session.close()


def migrate_all():
    print("开始迁移 JSON → SQLite...")
    init_db()
    total = 0
    total += migrate_users()
    total += migrate_finance()
    total += migrate_device_configs()
    total += migrate_device_rules()
    print(f"迁移完成！共 {total} 条记录。")


if __name__ == "__main__":
    migrate_all()
