"""
HTTP 协议模拟器服务器 — 多设备统一管理

模拟 7 种农业 IoT 设备，通过 HTTP 协议接收指令、返回状态。
Agent 通过 HTTPDriver 连接本服务器，走真实 HTTP 协议。

设备:
  - irrigation_pump_01: 灌溉泵（IRRIGATE）
  - ventilation_fan_01: 通风扇（VENTILATE）
  - grow_light_01: 补光灯（LIGHT）
  - heater_01: 加热器（HEAT）
  - env_sensor_01: 温湿度传感器（READ_SENSOR）
  - fertilizer_pump_01: 施肥泵（FERTIGATE）
  - greenhouse_camera_01: 摄像头（CAPTURE）

端点（匹配 HTTPDriver 协议）:
  POST /command   Body: {"device_id":"...", "command":"...", "params":{...}}
  GET  /state?device_id=...

启动: python hardware_examples/http_simulator_server.py [--port 5000]
"""

import json
import math
import random
import sys
import threading
import time
from datetime import datetime

try:
    from flask import Flask, request, jsonify
except ImportError:
    print("[ERR] 请先安装 flask: pip install flask")
    sys.exit(1)

app = Flask(__name__)
_lock = threading.Lock()

# ── 传感器默认值 ────────────────────────────
SENSOR_DEFAULTS = {
    "temperature": 22.0,
    "humidity": 65.0,
    "soil_moisture": 45.0,
    "ph": 7.0,
    "light_lux": 5000.0,
    "co2_ppm": 400.0,
    "flow_rate": 0.0,
    "rpm": 0.0,
}

# ── 多设备状态存储 ──────────────────────────

DEVICE_TEMPLATES = {
    "irrigation_pump_01": {
        "name": "温室灌溉泵",
        "capabilities": ["irrigate"],
        "sensors": ["flow_rate", "total_water_liters"],
        "location": "温室A区-灌溉区",
    },
    "ventilation_fan_01": {
        "name": "温室通风扇",
        "capabilities": ["ventilate"],
        "sensors": ["rpm", "temperature"],
        "location": "温室A区-通风区",
    },
    "grow_light_01": {
        "name": "温室补光灯",
        "capabilities": ["light"],
        "sensors": ["light_lux", "brightness_percent"],
        "location": "温室A区-种植区",
    },
    "heater_01": {
        "name": "温室加热器",
        "capabilities": ["heat"],
        "sensors": ["temperature", "target_temp"],
        "location": "温室A区-种植区",
    },
    "env_sensor_01": {
        "name": "环境温湿度传感器",
        "capabilities": ["read_sensor"],
        "sensors": ["temperature", "humidity", "soil_moisture", "co2_ppm"],
        "location": "温室A区-中心",
    },
    "fertilizer_pump_01": {
        "name": "施肥一体机",
        "capabilities": ["fertigate"],
        "sensors": ["flow_rate", "last_amount_kg"],
        "location": "温室A区-灌溉区",
    },
    "greenhouse_camera_01": {
        "name": "温室监控摄像头",
        "capabilities": ["capture"],
        "sensors": [],
        "location": "温室A区-入口",
    },
}

_devices = {}

def _init_devices():
    """初始化所有设备状态"""
    for dev_id, template in DEVICE_TEMPLATES.items():
        state = {"power": False, "status": "powered_off"}
        for s in template["sensors"]:
            if s in SENSOR_DEFAULTS:
                state[s] = SENSOR_DEFAULTS[s]
        _devices[dev_id] = {
            "info": template,
            "state": state,
        }


def _sensor_drift(state):
    """模拟传感器数据漂移（每次读取时调用）"""
    if "temperature" in state:
        state["temperature"] = round(state["temperature"] + random.uniform(-0.3, 0.3), 1)
        state["temperature"] = max(-30.0, min(55.0, state["temperature"]))
    if "humidity" in state:
        state["humidity"] = round(state["humidity"] + random.uniform(-1.5, 1.5), 1)
        state["humidity"] = max(10.0, min(99.0, state["humidity"]))
    if "soil_moisture" in state:
        # 灌溉中 → 湿度上升，否则下降
        irrigation_on = False
        for did, dev in _devices.items():
            if "irrigat" in did and dev["state"].get("power") and dev["state"].get("status") == "running":
                irrigation_on = True
                break
        if irrigation_on:
            state["soil_moisture"] = round(state["soil_moisture"] + random.uniform(0.5, 1.2), 1)
        else:
            state["soil_moisture"] = round(state["soil_moisture"] - random.uniform(0.05, 0.15), 1)
        state["soil_moisture"] = max(5.0, min(95.0, state["soil_moisture"]))
    if "co2_ppm" in state:
        state["co2_ppm"] = round(state["co2_ppm"] + random.uniform(-5, 5), 0)
        state["co2_ppm"] = max(300, min(2000, state["co2_ppm"]))
    if "light_lux" in state:
        # 模拟日夜变化：基础值 ± 随机
        hour = datetime.now().hour
        base = 30000 if 6 <= hour <= 18 else 500
        state["light_lux"] = round(base + random.uniform(-2000, 2000), 0)
    return state


def _execute_command(dev_id, command, params):
    """执行指令，返回 (success, message)"""
    if dev_id not in _devices:
        return False, f"设备 '{dev_id}' 不存在"

    dev = _devices[dev_id]
    state = dev["state"]
    name = dev["info"]["name"]
    current = state.get("status", "powered_off")

    if command in ("power_on", "boot"):
        if current == "powered_off":
            state["power"] = True
            state["status"] = "standby"
            return True, f"{name} 通电启动，进入待机"
        elif current == "standby":
            return True, f"{name} 已在待机状态"
        elif current == "running":
            return True, f"{name} 正在工作中"
        elif current == "error":
            return False, f"{name} 故障状态，请先复位"

    elif command in ("power_off", "shutdown"):
        state["power"] = False
        state["status"] = "powered_off"
        return True, f"{name} 关机断电"

    elif command == "start":
        if current == "powered_off":
            state["power"] = True
            state["status"] = "running"
            msg = f"{name} 通电并启动"
        elif current == "standby":
            state["status"] = "running"
            msg = f"{name} 开始工作"
        elif current == "running":
            msg = f"{name} 工作中，参数已更新"
        elif current == "error":
            return False, f"{name} 故障状态，请先复位"
        else:
            msg = f"{name} 已启动"

        # 保存工作参数
        if "duration" in params:
            state["last_duration"] = params["duration"]
        if "flow_rate" in params:
            state["flow_rate"] = params["flow_rate"]
        if "target_temp" in params:
            state["target_temp"] = params["target_temp"]
        if "brightness_percent" in params:
            state["brightness_percent"] = params["brightness_percent"]
        if "amount_kg" in params:
            state["last_amount_kg"] = params["amount_kg"]
        return True, msg

    elif command == "stop":
        if current == "running":
            state["status"] = "standby"
            state["flow_rate"] = 0
            state["rpm"] = 0
            return True, f"{name} 已停止（保持通电待机）"
        return True, f"{name} 当前未工作"

    elif command == "reset":
        state["power"] = False
        state["status"] = "powered_off"
        return True, f"{name} 已复位"

    elif command == "capture":
        # 模拟拍照
        return True, f"{name} 拍照完成（模拟）"

    elif command == "set_param":
        for k, v in params.items():
            if k in state:
                state[k] = v
        return True, f"{name} 参数已更新"

    else:
        return False, f"不支持的指令: {command}"


# ── API 端点 ──────────────────────────────────

@app.route("/command", methods=["POST"])
def handle_command():
    """接收指令（匹配 HTTPDriver 协议）"""
    data = request.get_json(silent=True) or {}
    dev_id = data.get("device_id", "")
    command = data.get("command", "")
    params = data.get("params", {})

    with _lock:
        success, message = _execute_command(dev_id, command, params)

    return jsonify({
        "success": success,
        "message": message,
        "device_id": dev_id,
    })


@app.route("/state", methods=["GET"])
def handle_state():
    """返回设备状态（匹配 HTTPDriver 协议）"""
    dev_id = request.args.get("device_id", "")

    with _lock:
        if dev_id not in _devices:
            return jsonify({"error": f"设备 '{dev_id}' 不存在"})

        state = dict(_devices[dev_id]["state"])
        state = _sensor_drift(state)

        # 保存漂移后的传感器值
        for k, v in state.items():
            if k in _devices[dev_id]["state"]:
                _devices[dev_id]["state"][k] = v

    state["_simulator"] = True
    state["_read_at"] = datetime.now().isoformat()
    return jsonify(state)


@app.route("/devices", methods=["GET"])
def list_devices():
    """列出所有模拟设备"""
    result = {}
    for dev_id, dev in _devices.items():
        info = dev["info"]
        state = dev["state"]
        result[dev_id] = {
            "name": info["name"],
            "capabilities": info["capabilities"],
            "sensors": info["sensors"],
            "location": info["location"],
            "status": state.get("status"),
            "power": state.get("power"),
        }
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "devices": len(_devices)})


# ── 启动 ──────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="HTTP 农业设备模拟器")
    p.add_argument("--port", type=int, default=5000, help="监听端口（默认5000）")
    p.add_argument("--host", default="127.0.0.1", help="监听地址")
    args = p.parse_args()

    _init_devices()
    print(f"[模拟器] 已初始化 {len(_devices)} 个设备")
    for dev_id, dev in _devices.items():
        print(f"  {dev_id}: {dev['info']['name']} ({', '.join(dev['info']['capabilities'])})")

    print(f"\n[模拟器] HTTP 服务启动: http://{args.host}:{args.port}")
    print(f"  指令端点: POST http://{args.host}:{args.port}/command")
    print(f"  状态端点: GET  http://{args.host}:{args.port}/state?device_id=...")
    print(f"  设备列表: GET  http://{args.host}:{args.port}/devices")

    app.run(host=args.host, port=args.port, debug=False)
