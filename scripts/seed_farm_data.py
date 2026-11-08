#!/usr/bin/env python3
"""为已有账号补齐农场资料；只新增缺失记录，保留哥哥的现有数据。"""

import argparse
import json
import math
import sqlite3
import sys
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from sqlalchemy.orm import Session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 必须先读应用配置，避免数据库包初始化时选到错误的库。
from core.storage_paths import DEFAULT_DATA_DIR
from core.database.models import (
    DeviceConfig, DeviceRule, Field, FieldZone, FinanceRecord, PlantingPlan,
    PlantingTask, Reminder, User, UserProfile,
)


HK_TIMEZONE = timezone(timedelta(hours=8))
BUCKETS = ("fields", "zones", "plans", "tasks", "finance", "devices", "reminders", "rules", "profiles")
FIELD_SPECS = (
    ("东侧麦田", "小麦", "wheat", 114.5210, 38.0520, 5, 35, "分蘖期", 3, 220),
    ("北区番茄温室", "番茄", "tomato", 114.5230, 38.0540, 2, 75, "开花坐果期", 4, 150),
    ("油菜试验田", "油菜", "rapeseed", 114.5250, 38.0510, 6, 30, "苗期", 2, 210),
)


def _json(value):
    return json.dumps(value, ensure_ascii=False)


def _rectangle(lon, lat, area_mu):
    """按米制宽高生成矩形，让哥哥看到的坐标和亩数相符。"""
    width = 50.0
    half_lon = width / (111320 * math.cos(math.radians(lat))) / 2
    half_lat = (area_mu * 666.67 / width) / 111320 / 2
    return [[lon - half_lon, lat - half_lat], [lon + half_lon, lat - half_lat],
            [lon + half_lon, lat + half_lat], [lon - half_lon, lat + half_lat]]


def _zone_coordinates(coordinates):
    """新矩形按东西分区；旧边界不规则时只补分区名称。"""
    try:
        coords = json.loads(coordinates)
        xs = sorted({point[0] for point in coords})
        ys = sorted({point[1] for point in coords})
        if len(xs) == len(ys) == 2 and len({tuple(point) for point in coords}) == 4:
            middle = (xs[0] + xs[1]) / 2
            return [
                [[xs[0], ys[0]], [middle, ys[0]], [middle, ys[1]], [xs[0], ys[1]]],
                [[middle, ys[0]], [xs[1], ys[0]], [xs[1], ys[1]], [middle, ys[1]]],
            ]
    except (TypeError, ValueError, IndexError):
        pass
    return [[], []]


def _ensure(session, user_id, result, bucket, model, key, values, alternate=None):
    """先找自然键，同名记录也直接跳过，绝不替哥哥覆盖原值。"""
    query = session.query(model).filter_by(user_id=user_id)
    existing = query.filter_by(**key).first()
    if existing is None and alternate:
        existing = query.filter_by(**alternate).first()
    if existing is not None:
        result["existing_ids"][bucket].append(existing.id)
        return existing
    row = model(user_id=user_id, **key, **values)
    session.add(row)
    session.flush()
    result["added"][bucket] += 1
    result["created_ids"][bucket].append(row.id)
    return row


def seed_farm_data(session: Session, username: str, as_of: date | None = None) -> dict:
    """在调用方事务内补齐已有账号的农场资料，不提交或修改原记录。"""
    user = session.query(User).filter_by(username=username).first()
    if user is None:
        raise ValueError(f"账号不存在：{username}")
    as_of = as_of or datetime.now(HK_TIMEZONE).date()
    result = {
        "username": user.username, "as_of": as_of.isoformat(),
        "added": {key: 0 for key in BUCKETS},
        "created_ids": {key: [] for key in BUCKETS},
        "existing_ids": {key: [] for key in BUCKETS},
        "field_ids": {}, "device_ids": [], "skipped": [],
    }

    def ensure(bucket, model, key, values, alternate=None):
        return _ensure(session, user.id, result, bucket, model, key, values, alternate)

    fields, plans, zones = {}, {}, {}
    for name, crop, prefix, lon, lat, area, age, stage, stage_number, duration in FIELD_SPECS:
        field = ensure("fields", Field, {"name": name}, {
            "coordinates": _json(_rectangle(lon, lat, area)), "center_lon": lon, "center_lat": lat,
            "area_mu": area, "area_m2": area * 666.67, "soil_type": "壤土", "current_crop": crop,
            "planting_history": "[]",
        })
        fields[prefix] = field
        result["field_ids"][name] = field.id
        for index, coordinates in enumerate(_zone_coordinates(field.coordinates)):
            zone_id = ("west_bed", "east_bed")[index]
            zone = ensure("zones", FieldZone, {"field_id": field.id, "zone_id": zone_id}, {
                "name": ("西侧作业区", "东侧作业区")[index], "zone_type": "operation",
                "coordinates": _json(coordinates), "notes": "按垄分区开展田间管理。",
            }, alternate={"field_id": field.id, "name": ("西侧作业区", "东侧作业区")[index]})
            zones[prefix, index] = zone.zone_id
        start = as_of - timedelta(days=age)
        plans[prefix] = ensure("plans", PlantingPlan, {"crop": crop, "plot_id": field.id}, {
            "stage": stage, "stage_number": stage_number, "total_stages": 7,
            "start_date": start, "expected_end_date": start + timedelta(days=duration),
            "progress_percent": round(age / duration * 100, 1), "status": "进行中",
            "notes": f"{name}按墒情安排水肥，结合生长阶段开展巡查。",
        })

    task_specs = (
        ("wheat", "小麦播种记录", "播种", -35, -34, "已完成", 100),
        ("wheat", "麦田苗情巡查", "其他", -2, 1, "进行中", 45),
        ("wheat", "麦田越冬水准备", "浇水", 2, 5, "待办", 0),
        ("tomato", "番茄定植记录", "播种", -75, -73, "已完成", 100),
        ("tomato", "番茄整枝打杈", "修剪", -1, 2, "进行中", 60),
        ("tomato", "番茄灰霉病巡查", "病虫害防治", 0, 3, "待办", 0),
        ("rapeseed", "油菜播种记录", "播种", -30, -29, "已完成", 100),
        ("rapeseed", "油菜苗期补肥", "施肥", 1, 4, "待办", 0),
    )
    for prefix, title, kind, start_offset, end_offset, status, progress in task_specs:
        ensure("tasks", PlantingTask, {"title": title}, {
            "plan_id": plans[prefix].id, "crop": plans[prefix].crop, "task_type": kind,
            "description": f"在{fields[prefix].name}完成{title}，记录长势和作业情况。",
            "start_date": as_of + timedelta(days=start_offset),
            "end_date": as_of + timedelta(days=end_offset),
            "completed_date": as_of + timedelta(days=end_offset) if status == "已完成" else None,
            "status": status, "priority": "medium", "progress_percent": progress,
            "notes": "完成后登记实际作业情况。", "device_id": None,
            "device_command": None, "device_params": None,
        })

    # 本月及之前五个月都用月初日期，同月重跑的自然键保持不变。
    month_index = as_of.year * 12 + as_of.month - 1
    for offset in range(5, -1, -1):
        year, zero_month = divmod(month_index - offset, 12)
        record_date = date(year, zero_month + 1, 1)
        monthly = (
            ("wheat", "cost", "种子", "麦田生产资料", 20, "kg", 8.5, ""),
            ("tomato", "cost", "肥料", "番茄温室水溶肥", 15, "kg", 18, ""),
            ("rapeseed", "cost", "人工", "油菜田间管护", 3, "工日", 120, ""),
            ("tomato", "income", "销售", "番茄批次销售", 240 + record_date.month * 10,
             "kg", 3.6, "石家庄蔬菜收购站"),
        )
        for prefix, record_type, category, item, quantity, unit, price, buyer in monthly:
            ensure("finance", FinanceRecord, {
                "date": record_date, "record_type": record_type, "crop": plans[prefix].crop,
                "plot": fields[prefix].name, "item_name": item,
            }, {
                "category": category, "quantity": quantity, "unit": unit, "unit_price": price,
                "total_amount": round(quantity * price, 2), "buyer": buyer,
                "notes": f"{record_date:%Y年%m月}生产收支登记。",
            })

    device_specs = (
        ("wheat", 0, "wheat_soil_sensor", "麦田土壤传感器", "read_sensor",
         {"soil_moisture": 43.5, "temperature": 12.6, "ph": 7.2}),
        ("tomato", 0, "tomato_environment_sensor", "番茄温室环境传感器", "read_sensor",
         {"temperature": 24.2, "humidity": 68, "soil_moisture": 51, "co2_ppm": 620, "light_lux": 7600}),
        ("rapeseed", 0, "rapeseed_soil_sensor", "油菜田土壤传感器", "read_sensor",
         {"soil_moisture": 46, "temperature": 13.1, "ph": 7.1}),
        ("wheat", 1, "wheat_irrigation", "麦田灌溉阀", "irrigate", {"flow_rate": 0}),
        ("tomato", 1, "tomato_irrigation", "番茄滴灌泵", "irrigate", {"flow_rate": 0}),
        ("tomato", 0, "tomato_ventilation_fan", "番茄温室通风扇", "ventilate", {"rpm": 0}),
        ("tomato", 1, "tomato_grow_light", "番茄温室补光灯", "light", {"brightness_percent": 0}),
        ("tomato", 1, "tomato_heater", "番茄温室加热器", "heat", {"temperature": 24.2}),
    )
    available_devices = set()
    for prefix, zone_index, device_id, name, capability, readings in device_specs:
        device = ensure("devices", DeviceConfig, {"device_id": device_id}, {
            "name": name, "driver": "simulator", "capabilities": _json([capability]),
            "sensors": _json(list(readings)), "connection": "{}",
            "plot_id": fields[prefix].id, "zone_id": zones[prefix, zone_index],
            "location": f"{fields[prefix].name}·{('西侧作业区', '东侧作业区')[zone_index]}",
            "initial_state": _json({"power": True, "status": "standby", **readings}),
        }, alternate={"name": name})
        if device.device_id == device_id and device.driver == "simulator":
            available_devices.add(device_id)
            result["device_ids"].append(device_id)

    for prefix, kind, description in (
        ("wheat", "浇水", "检查麦田墒情并安排越冬水"),
        ("tomato", "病虫害防治", "巡查番茄叶片和温室湿度"),
        ("rapeseed", "施肥", "查看油菜苗情并核对补肥用量"),
    ):
        ensure("reminders", Reminder, {"task_description": description}, {
            "crop": plans[prefix].crop, "reminder_type": kind, "growth_stage": plans[prefix].stage,
            "frequency": "每周", "interval_days": 7, "time_of_day": "08:30",
            "advance_hours": 0, "channels": "[]", "status": "paused", "next_trigger": None,
        })

    for name, sensor, field, threshold, operator, target, capability, duration in (
        ("麦田墒情灌溉", "wheat_soil_sensor", "soil_moisture", 35, "<", "wheat_irrigation", "irrigate", 10),
        ("番茄温室高温通风", "tomato_environment_sensor", "temperature", 28, ">",
         "tomato_ventilation_fan", "ventilate", 15),
    ):
        if not {sensor, target}.issubset(available_devices):
            result["skipped"].append({"kind": "rule", "name": name, "reason": "关联设备标识已被其他设备占用"})
            continue
        ensure("rules", DeviceRule, {"name": name}, {
            "enabled": 0,
            "conditions": _json({"logic": "AND", "execution_mode": "confirm", "conditions": [
                {"type": "sensor", "device_id": sensor, "field": field, "op": operator, "value": threshold},
            ]}),
            "actions": _json({"device_id": target, "capability": capability,
                              "command": "start", "params": {"duration": duration}}),
            "constraints": _json({"max_duration_per_use": 20, "max_duration_per_day": 40,
                                  "min_interval_minutes": 120}),
        })

    total_area = sum(row.area_mu or 0 for row in session.query(Field).filter_by(user_id=user.id))
    ensure("profiles", UserProfile, {}, {
        "region": "河北省石家庄市", "soil_type": "壤土", "farm_size": total_area,
        "experience_level": "中级（3-5年）", "goals": _json(["高产", "节水", "经济效益"]),
    })
    return result


def backup_sqlite_database(database_path: Path, backup_dir: Path) -> Path:
    """使用 SQLite 在线备份，连同已提交的 WAL 记录一起保留。"""
    database_path = Path(database_path).resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"数据库文件不存在：{database_path}")
    backup_dir = Path(backup_dir).resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(HK_TIMEZONE).strftime("%Y%m%d_%H%M%S_%f")
    destination = backup_dir / f"{database_path.stem}_{timestamp}_{uuid4().hex[:8]}.sqlite3"
    with closing(sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(destination)) as target:
            source.backup(target)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="为已有账号补齐农场资料，执行前自动备份本地SQLite数据库。")
    parser.add_argument("--username", required=True, help="必须明确指定已有账号")
    parser.add_argument("--as-of", type=date.fromisoformat, help="基准日期YYYY-MM-DD，默认香港当天日期")
    args = parser.parse_args(argv)

    # 读取应用的同一份引擎配置；此处还不执行迁移。
    from core.database import engine as database

    url = database._engine.url
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:" or url.query:
        parser.error("仅支持当前配置指向的本地SQLite文件数据库")
    if url.database.startswith(("\\\\", "//")):
        parser.error("仅支持本地SQLite文件，不能使用网络共享路径")
    database_path = Path(url.database).resolve()
    data_dir = Path(DEFAULT_DATA_DIR)
    backup = backup_sqlite_database(database_path, data_dir / "backups" / "farm_seed")
    database.init_db()
    with database.Session.begin() as session:
        result = seed_farm_data(session, args.username, as_of=args.as_of)
    result["backup"] = str(backup)
    result["database"] = str(database_path)
    result["seeded_at"] = datetime.now(HK_TIMEZONE).isoformat()
    manifest_dir = data_dir / "seeded_farms"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{quote(args.username, safe='')}.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(_json(result), encoding="utf-8")
    temporary.replace(manifest)
    print(json.dumps({"username": args.username, "added": result["added"],
                      "backup": str(backup), "manifest": str(manifest)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
