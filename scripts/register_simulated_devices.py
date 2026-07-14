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
     "connection": {"mode": "tcp", "host": "127.0.0.1", "port": 5020, "slave_id": 1},
     "location": "温室A区-中心"},
    {"device_id": "greenhouse_camera_01", "name": "温室监控摄像头", "driver": "modbus",
     "capabilities": ["capture"], "sensors": [],
     "connection": {"mode": "tcp", "host": "127.0.0.1", "port": 5020, "slave_id": 2},
     "location": "温室A区-入口"},
]


def register(username="123"):
    init_db()
    session = Session()
    user = session.query(User).filter(User.username == username).first()
    if not user:
        print(f"用户 '{username}' 不存在")
        session.close()
        return

    for d in DEVICES:
        existing = session.query(DeviceConfig).filter(
            DeviceConfig.user_id == user.id,
            DeviceConfig.device_id == d["device_id"],
        ).first()
        if existing:
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
        print(f"  注册: {d['device_id']} ({d['name']}) [{d['driver'].upper()}]")

    session.commit()
    session.close()
    print(f"\n完成! 共 {len(DEVICES)} 个设备已注册到用户 '{username}'")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--username", default="123", help="用户名")
    args = p.parse_args()
    register(args.username)
