"""
MQTT 协议模拟器服务器 — 多设备统一管理

模拟 7 种农业 IoT 设备，通过 MQTT 协议接收指令、上报状态。
Agent 通过 MQTTDriver 连接 MQTT Broker，与本模拟器通信。

要求: 需要 MQTT Broker（如 mosquitto）运行在 localhost:1883

依赖: pip install paho-mqtt
运行: python hardware_examples/mqtt_simulator_server.py [--broker localhost] [--port 1883]

Topic 格式（匹配 MQTTDriver 协议）:
  控制: devices/{device_id}/control  ← Agent 发送指令
  状态: devices/{device_id}/state    → 模拟器上报状态
"""

import json
import random
import sys
import threading
import time
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("[ERR] 请先安装 paho-mqtt: pip install paho-mqtt")
    sys.exit(1)

# ── 设备模板 ────────────────────────────
DEVICE_TEMPLATES = {
    "irrigation_pump_01": {
        "name": "温室灌溉泵-MQTT",
        "initial_state": {"power": False, "status": "powered_off", "flow_rate": 0.0},
    },
    "ventilation_fan_01": {
        "name": "温室通风扇-MQTT",
        "initial_state": {"power": False, "status": "powered_off", "rpm": 0},
    },
    "grow_light_01": {
        "name": "温室补光灯-MQTT",
        "initial_state": {"power": False, "status": "powered_off", "brightness_percent": 0},
    },
    "heater_01": {
        "name": "温室加热器-MQTT",
        "initial_state": {"power": False, "status": "powered_off", "target_temp": 22.0},
    },
    "env_sensor_01": {
        "name": "环境温湿度传感器-MQTT",
        "initial_state": {"power": True, "status": "standby",
                          "temperature": 22.0, "humidity": 65.0,
                          "soil_moisture": 45.0, "co2_ppm": 400.0},
    },
    "fertilizer_pump_01": {
        "name": "施肥一体机-MQTT",
        "initial_state": {"power": False, "status": "powered_off", "flow_rate": 0.0},
    },
    "greenhouse_camera_01": {
        "name": "温室监控摄像头-MQTT",
        "initial_state": {"power": False, "status": "powered_off"},
    },
}

_devices = {}
_lock = threading.Lock()


def _init_devices():
    for dev_id, template in DEVICE_TEMPLATES.items():
        _devices[dev_id] = {
            "name": template["name"],
            "state": dict(template["initial_state"]),
        }


def _sensor_drift(dev_id, state):
    """传感器数据漂移模拟"""
    if "temperature" in state:
        state["temperature"] = round(state["temperature"] + random.uniform(-0.3, 0.3), 1)
    if "humidity" in state:
        state["humidity"] = round(state["humidity"] + random.uniform(-1.5, 1.5), 1)
    if "soil_moisture" in state:
        irrigation_on = any(
            "irrigat" in did and _devices[did]["state"].get("power")
            and _devices[did]["state"].get("status") == "running"
            for did in _devices
        )
        if irrigation_on:
            state["soil_moisture"] = round(state["soil_moisture"] + random.uniform(0.5, 1.2), 1)
        else:
            state["soil_moisture"] = round(state["soil_moisture"] - random.uniform(0.05, 0.15), 1)
        state["soil_moisture"] = max(5.0, min(95.0, state["soil_moisture"]))
    if "co2_ppm" in state:
        state["co2_ppm"] = int(state["co2_ppm"] + random.uniform(-5, 5))


def _execute(dev_id, command, params):
    if dev_id not in _devices:
        return False, f"设备 '{dev_id}' 不存在"
    state = _devices[dev_id]["state"]
    name = _devices[dev_id]["name"]
    current = state.get("status", "powered_off")

    if command in ("power_on", "boot"):
        if current == "powered_off":
            state["power"] = True; state["status"] = "standby"
            return True, f"{name} 通电启动"
        return True, f"{name} 已在 {current} 状态"

    elif command in ("power_off", "shutdown"):
        state["power"] = False; state["status"] = "powered_off"
        return True, f"{name} 关机"

    elif command == "start":
        if current == "error": return False, f"{name} 故障状态"
        state["power"] = True; state["status"] = "running"
        if "duration" in params: state["last_duration"] = params["duration"]
        if "flow_rate" in params: state["flow_rate"] = params["flow_rate"]
        return True, f"{name} 开始工作"

    elif command == "stop":
        if current == "running":
            state["status"] = "standby"; state["flow_rate"] = 0; state["rpm"] = 0
        return True, f"{name} 已停止"

    elif command == "reset":
        state["power"] = False; state["status"] = "powered_off"
        return True, f"{name} 已复位"

    elif command == "capture":
        return True, f"{name} 拍照完成（模拟）"

    else:
        return False, f"不支持指令: {command}"


# ── MQTT 回调 ─────────────────────────────

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] 已连接 Broker")
        for dev_id in _devices:
            topic = f"devices/{dev_id}/control"
            client.subscribe(topic)
            print(f"  订阅: {topic}")
    else:
        print(f"[MQTT] 连接失败: rc={rc}")


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return
    dev_id = data.get("device_id", "")
    command = data.get("command", "")
    params = data.get("params", {})

    with _lock:
        success, message = _execute(dev_id, command, params)

    response = {"success": success, "message": message, "device_id": dev_id}
    # 发布响应到设备专属响应 topic
    resp_topic = f"devices/{dev_id}/response"
    client.publish(resp_topic, json.dumps(response, ensure_ascii=False))

    # 执行后立即上报最新状态
    if dev_id in _devices:
        state = dict(_devices[dev_id]["state"])
        _sensor_drift(dev_id, state)
        state["_read_at"] = datetime.now().isoformat()
        state_topic = f"devices/{dev_id}/state"
        client.publish(state_topic, json.dumps(state, ensure_ascii=False))


# ── 定时上报 ──────────────────────────────

def _publish_loop(client, interval=10):
    """每 N 秒上报所有设备的状态"""
    while True:
        time.sleep(interval)
        with _lock:
            for dev_id, dev in _devices.items():
                state = dict(dev["state"])
                _sensor_drift(dev_id, state)
                state["_read_at"] = datetime.now().isoformat()
                dev["state"].update(
                    {k: v for k, v in state.items() if k in dev["state"]}
                )
                topic = f"devices/{dev_id}/state"
                client.publish(topic, json.dumps(state, ensure_ascii=False))


# ── 启动 ──────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MQTT 农业设备模拟器")
    p.add_argument("--broker", default="localhost", help="MQTT Broker 地址")
    p.add_argument("--port", type=int, default=1883, help="MQTT Broker 端口")
    p.add_argument("--interval", type=int, default=10, help="状态上报间隔(秒)")
    args = p.parse_args()

    _init_devices()
    print(f"[MQTT模拟器] 已初始化 {len(_devices)} 个设备")
    for dev_id, dev in _devices.items():
        print(f"  {dev_id}: {dev['name']}")

    client = mqtt.Client(client_id="agriculture_simulator_mqtt")
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.broker, args.port, 60)
    except Exception as e:
        print(f"[ERR] 无法连接 MQTT Broker {args.broker}:{args.port}: {e}")
        print("请先启动 MQTT Broker，如: mosquitto -v")
        sys.exit(1)

    # 启动定时上报线程
    t = threading.Thread(target=_publish_loop, args=(client, args.interval), daemon=True)
    t.start()

    print(f"\n[MQTT模拟器] 运行中 (Broker: {args.broker}:{args.port}, 上报间隔: {args.interval}s)")
    print("按 Ctrl+C 停止")
    client.loop_forever()
