#!/usr/bin/env python3
"""注册 7 个混合协议模拟设备到用户 123"""

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.database.engine import init_db, Session
from core.database.models import DeviceConfig, User

DEVICES = [
    # ── HTTP（port 5000）──
    {"device_id": "irrigation_pump_01", "name": "温室灌溉泵", "driver": "http",
     "capabilities": ["irrigate"], "sensors": ["flow_rate"],
     "connection": {"base_url": "http://127.0.0.1:5000"}, "location": "温室A区-灌溉区"},
    {"device_id": "grow_light_01", "name": "温室补光灯", "driver": "http",
     "capabilities": ["light"], "sensors": ["light_lux", "brightness_percent"],
     "connection": {"base_url": "http://127.0.0.1:5000"}, "location": "温室A区-种植区"},
    {"device_id": "fertilizer_pump_01", "name": "施肥一体机", "driver": "http",
     "capabilities": ["fertigate"], "sensors": ["flow_rate"],
     "connection": {"base_url": "http://127.0.0.1:5000"}, "location": "温室A区-灌溉区"},
    # ── MQTT ──
    {"device_id": "ventilation_fan_01", "name": "温室通风扇", "driver": "mqtt",
     "capabilities": ["ventilate"], "sensors": ["rpm"],
     "connection": {"host": "localhost", "port": 1883,
                    "control_topic": "devices/ventilation_fan_01/control",
                    "state_topic": "devices/ventilation_fan_01/state"},
     "location": "温室A区-通风区"},
    {"device_id": "heater_01", "name": "温室加热器", "driver": "mqtt",
     "capabilities": ["heat"], "sensors": ["temperature"],
     "connection": {"host": "localhost", "port": 1883,
                    "control_topic": "devices/heater_01/control",
                    "state_topic": "devices/heater_01/state"},
     "location": "温室A区-种植区"},
    # ── Modbus ──
    {"device_id": "env_sensor_01", "name": "环境温湿度传感器", "driver": "modbus",
     "capabilities": ["read_sensor"], "sensors": ["temperature", "humidity", "soil_moisture", "co2_ppm"],
     "connection": {"mode": "tcp", "host": "127.0.0.1", "port": "5020", "slave_id": 1},
     "location": "温室A区-中心"},
    # 摄像头图像走 HTTP；Modbus 寄存器无法传递 JPEG。
    {"device_id": "greenhouse_camera_01", "name": "温室监控摄像头", "driver": "http",
     "capabilities": ["capture"], "sensors": [],
     "connection": {"base_url": "http://127.0.0.1:5000"},
     "location": "温室A区-入口"},
]


def _is_legacy_local_camera(device):
    """只识别旧脚本写入的本机摄像头配置，保留其他接线。"""
    if device.device_id != "greenhouse_camera_01" or device.driver != "modbus":
        return False
    try:
        conn = json.loads(device.connection or "{}")
        return (
            conn.get("mode", "tcp") == "tcp"
            and conn.get("host") in ("127.0.0.1", "localhost", "::1")
            and int(conn.get("port", 0)) == 5020
            and int(conn.get("slave_id", 0)) == 2
        )
    except (TypeError, ValueError, AttributeError):
        return False


def register(username="123", *, repair_legacy_camera=False):
    init_db()
    session = Session()
    user = session.query(User).filter(User.username == username).first()
    if not user:
        print(f"用户 '{username}' 不存在")
        session.close()
        return

    added = repaired = skipped = 0
    for d in DEVICES:
        existing = session.query(DeviceConfig).filter(
            DeviceConfig.user_id == user.id,
            DeviceConfig.device_id == d["device_id"],
        ).first()
        if existing:
            if repair_legacy_camera and _is_legacy_local_camera(existing):
                existing.driver = d["driver"]
                existing.connection = json.dumps(d["connection"], ensure_ascii=False)
                repaired += 1
                print(f"  修复图像连接: {d['device_id']} [HTTP]")
                continue
            skipped += 1
            print(f"  跳过(已存在): {d['device_id']} ({d['name']})")
            continue
        session.add(DeviceConfig(
            user_id=user.id,
            device_id=d["device_id"], name=d["name"], driver=d["driver"],
            capabilities=json.dumps(d["capabilities"], ensure_ascii=False),
            sensors=json.dumps(d["sensors"], ensure_ascii=False),
            connection=json.dumps(d["connection"], ensure_ascii=False),
            location=d["location"],
        ))
        added += 1
        print(f"  注册: {d['device_id']} ({d['name']}) [{d['driver'].upper()}]")

    session.commit()
    session.close()
    print(f"\n完成! 用户 '{username}': 新增 {added}，修复 {repaired}，保留 {skipped}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--username", default="123", help="用户名")
    p.add_argument("--repair-legacy-camera", action="store_true",
                   help="将旧脚本的本机 Modbus 摄像头定向修复为 HTTP；请先备份数据库")
    args = p.parse_args()
    register(args.username, repair_legacy_camera=args.repair_legacy_camera)
