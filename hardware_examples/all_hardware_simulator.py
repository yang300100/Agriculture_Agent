"""
一体化硬件模拟器

启动所有连接方式的模拟硬件，前端操控时实时打印反馈。
支持: Simulator / HTTP / MQTT / Modbus

运行: python hardware_examples/all_hardware_simulator.py

启动后:
  1. HTTP设备服务器 (端口5000) 自动启动
  2. 8个模拟设备全部在线
  3. 前端操控设备 → 此终端实时显示反馈
  4. 输入命令可手动控制设备 (输入 help 查看)


硬件设备生命周期状态机
========================

  powered_off(关机) ──[power_on/通电]──▶ standby(待机) ──[start/工作]──▶ running(工作中)
       ▲                                      ▲                            │
       │                                      │ [stop/停止工作]            │
       │                                      ◀────────────────────────────┘
       │                                      │
       └──────────────[power_off/关机]────────┘

  任意状态 ──[故障]──▶ error(故障)
  error    ──[reset]──▶ standby(待机)

状态说明:
  powered_off : 设备断电，完全关闭
  standby     : 设备已通电，等待工作指令
  running     : 设备正在执行工作任务
  error       : 设备发生故障，需复位

命令:
  power_on  / boot     : 通电启动  (powered_off → standby)
  power_off / shutdown : 关机断电  (standby → powered_off)
  start                : 开始工作  (standby → running)
  stop                 : 停止工作  (running → standby, 保持通电)
  reset                : 故障复位  (error → standby)
"""

import sys
import os
import json
import time
import asyncio
import base64
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════
# 颜色输出
# ═══════════════════════════════════════════════════

class C:
    """终端颜色"""
    R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"
    B = "\033[94m"; C = "\033[96m"; M = "\033[95m"
    W = "\033[0m"; BOLD = "\033[1m"

    @staticmethod
    def time(): return f"{C.C}{datetime.now().strftime('%H:%M:%S')}{C.W}"


def log_event(device_name, action, detail=""):
    """打印硬件事件"""
    detail_str = f" ({detail})" if detail else ""
    print(f"  [{C.time()}] {C.G}[HARDWARE]{C.W} {C.BOLD}{device_name}{C.W} -> {C.Y}{action}{C.W}{detail_str}")


def log_start(service_name, port=None):
    port_str = f" :{port}" if port else ""
    print(f"  [{C.time()}] {C.G}[START]{C.W}  {C.B}{service_name}{C.W}{port_str} {C.G}已启动模拟{C.W}")


def log_error(service_name, msg=""):
    print(f"  [{C.time()}] {C.R}[ERROR]{C.W} {service_name}: {msg}")


def log_cmd(cmd):
    print(f"  [{C.time()}] {C.M}[CMD]{C.W}   {cmd}")


# ═══════════════════════════════════════════════════
# 设备状态机工具函数
# ═══════════════════════════════════════════════════

# 有效状态及其中文名
POWER_STATE_LABELS = {
    "powered_off": "关机",
    "standby": "待机",
    "running": "工作中",
    "error": "故障",
}

# 状态转换规则: (当前状态, 命令) → (新状态, 是否合法)
# 格式: {当前status: {命令: (新status, 是否需要power=True)}}
STATE_TRANSITIONS = {
    "powered_off": {
        "power_on":  ("standby", True),
        "boot":      ("standby", True),
        "start":     ("running", True),   # 从关机直接start：先通电再工作
    },
    "standby": {
        "start":     ("running", True),
        "power_off": ("powered_off", False),
        "shutdown":  ("powered_off", False),
    },
    "running": {
        "stop":      ("standby", True),    # 停止工作但保持通电
        "power_off": ("powered_off", False),  # 工作中直接关机
        "shutdown":  ("powered_off", False),
    },
    "error": {
        "reset":     ("standby", True),
        "power_off": ("powered_off", False),
        "shutdown":  ("powered_off", False),
    },
}


def apply_state_transition(current_status: str, command: str) -> tuple:
    """
    执行状态转换，返回 (新status, 新power值, 是否合法, 错误消息)

    Args:
        current_status: 当前状态 (powered_off/standby/running/error)
        command: 要执行的命令

    Returns:
        (new_status, new_power, is_valid, message)
    """
    transitions = STATE_TRANSITIONS.get(current_status, {})
    if command in transitions:
        new_status, new_power = transitions[command]
        return new_status, new_power, True, ""

    # set_param 不改变状态
    if command == "set_param":
        return current_status, None, True, ""

    # 非法转换
    return current_status, None, False, \
        f"当前状态'{POWER_STATE_LABELS.get(current_status, current_status)}'不支持'{command}'操作"


def status_display(state: dict) -> str:
    """格式化状态显示"""
    status = state.get("status", "powered_off")
    power = state.get("power", False)
    label = POWER_STATE_LABELS.get(status, status)
    if status == "running":
        return f"{C.G}●{C.W} {label}"
    elif status == "standby":
        return f"{C.Y}○{C.W} {label}"
    elif status == "error":
        return f"{C.R}✕{C.W} {label}"
    else:  # powered_off
        return f"{C.R}○{C.W} {label}"


# ═══════════════════════════════════════════════════
# 多设备 HTTP 服务器 — 模拟多个真实硬件端点
# ═══════════════════════════════════════════════════

# 预定义的虚拟农业设备（模拟真实硬件）
FARM_DEVICE_TEMPLATES = {
    "irrigation_pump_01": {
        "name": "温室灌溉泵",
        "type": "irrigate",
        "sensors": ["flow_rate", "total_water_liters"],
        "initial": {"power": True, "status": "standby", "flow_rate": 0, "total_water_liters": 156.8},
    },
    "ventilation_fan_01": {
        "name": "温室通风扇",
        "type": "ventilate",
        "sensors": ["rpm"],
        "initial": {"power": True, "status": "standby", "rpm": 0},
    },
    "grow_light_01": {
        "name": "温室补光灯",
        "type": "light",
        "sensors": ["brightness_percent"],
        "initial": {"power": True, "status": "standby", "brightness_percent": 0},
    },
    "heater_01": {
        "name": "温室加热器",
        "type": "heat",
        "sensors": ["target_temp", "current_temp"],
        "initial": {"power": True, "status": "standby", "target_temp": 22, "current_temp": 18.5},
    },
    "env_sensor_01": {
        "name": "环境温湿度传感器",
        "type": "read_sensor",
        "sensors": ["temperature", "humidity", "soil_moisture", "ph", "light_lux"],
        "initial": {"power": True, "status": "standby",
                    "temperature": 24.5, "humidity": 62.0, "soil_moisture": 48.0, "ph": 6.8, "light_lux": 35000},
    },
    "fertilizer_pump_01": {
        "name": "施肥一体机",
        "type": "fertigate",
        "sensors": ["flow_rate", "total_fertilizer_kg"],
        "initial": {"power": True, "status": "standby", "flow_rate": 0, "total_fertilizer_kg": 23.5},
    },
    "greenhouse_camera_01": {
        "name": "温室监控摄像头",
        "type": "capture",
        "sensors": ["resolution", "last_capture_time"],
        "initial": {"power": True, "status": "standby", "resolution": "640x480", "last_capture_time": None},
    },
}


def start_http_server(port=5000):
    """启动多设备 HTTP 服务器（独立线程）— 模拟多个真实硬件端点。

    每个设备通过请求中的 device_id 字段区分，维护独立状态。
    前端/驱动通过 HTTPDriver 连接此服务器即可操控所有设备。
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        log_error("HTTP服务器", "请安装 flask: pip install flask")
        return None

    app = Flask(__name__)

    # 多设备状态存储: {device_id: state_dict}
    _devices: Dict[str, dict] = {}
    _lock = threading.Lock()

    # 初始化所有预定义设备
    for dev_id, template in FARM_DEVICE_TEMPLATES.items():
        _devices[dev_id] = dict(template["initial"])
        _devices[dev_id]["_name"] = template["name"]
        _devices[dev_id]["_type"] = template["type"]

    def _get_device(device_id: str) -> Optional[dict]:
        """获取设备状态，不存在返回 None"""
        return _devices.get(device_id)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "server": "farm-device-simulator",
            "device_count": len(_devices),
            "devices": list(_devices.keys()),
        })

    @app.route("/api/state", methods=["GET"])
    @app.route("/state", methods=["GET"])  # HTTPDriver 兼容路由
    def get_state():
        """查询设备状态 — 通过 query param 或 body 中的 device_id 区分设备"""
        device_id = request.args.get("device_id") or ""
        if not device_id:
            # 兼容：尝试从 body 获取（GET 通常无 body，但有些客户端会发）
            pass

        with _lock:
            if device_id and device_id in _devices:
                state = {k: v for k, v in _devices[device_id].items() if not k.startswith("_")}
                return jsonify(state)
            # 无 device_id 时返回所有设备概览
            summary = {}
            for did, dev in _devices.items():
                summary[did] = {
                    "name": dev.get("_name", did),
                    "type": dev.get("_type", "unknown"),
                    "power": dev.get("power", False),
                    "status": dev.get("status", "powered_off"),
                }
            return jsonify({"devices": summary, "total": len(_devices)})

    @app.route("/api/command", methods=["POST"])
    @app.route("/command", methods=["POST"])  # HTTPDriver 兼容路由
    def execute():
        """执行设备指令 — 通过 body 中的 device_id 区分目标设备"""
        data = request.get_json(silent=True) or {}
        device_id = data.get("device_id", "")
        command = data.get("command", "")
        params = data.get("params", {})

        if not device_id:
            return jsonify({"success": False, "message": "缺少 device_id"}), 400

        with _lock:
            dev = _get_device(device_id)
            if dev is None:
                return jsonify({"success": False, "message": f"设备 '{device_id}' 不存在"}), 404

            dev_name = dev.get("_name", device_id)
            current = dev.get("status", "powered_off")

            # ── 通电 / 关机 ──
            if command in ("power_on", "boot"):
                if current == "powered_off":
                    dev["power"] = True
                    dev["status"] = "standby"
                    log_event(dev_name, "通电启动", "进入待机")
                elif current == "standby":
                    log_event(dev_name, "通电启动", "已在待机")
                return jsonify({"success": True, "message": f"{dev_name} 已通电"})

            elif command in ("power_off", "shutdown"):
                if current in ("standby", "running"):
                    dev["power"] = False
                    dev["status"] = "powered_off"
                    log_event(dev_name, "关机断电")
                elif current == "powered_off":
                    log_event(dev_name, "关机断电", "已在关机状态")
                return jsonify({"success": True, "message": f"{dev_name} 已关机"})

            # ── 开始工作 ──
            elif command == "start":
                if current == "powered_off":
                    dev["power"] = True
                    dev["status"] = "running"
                    log_event(dev_name, "通电并启动", f"参数={params}")
                elif current == "standby":
                    dev["status"] = "running"
                    log_event(dev_name, "开始工作", f"参数={params}")
                elif current == "running":
                    log_event(dev_name, "开始工作", "已在工作中，更新参数")
                elif current == "error":
                    return jsonify({"success": False, "message": f"{dev_name} 故障中，请先复位"}), 400

                # 应用参数到设备状态
                for k, v in params.items():
                    if k in dev and not k.startswith("_"):
                        dev[k] = v

                # 模拟定时停止
                duration = params.get("duration", 0)
                if duration > 0:
                    captured_id = device_id
                    def auto_stop():
                        time.sleep(duration * 60)  # duration 单位是分钟
                        with _lock:
                            d = _devices.get(captured_id)
                            if d and d.get("status") == "running":
                                d["status"] = "standby"
                                log_event(d.get("_name", captured_id), "定时结束，回到待机", f"运行了{duration}分钟")
                    threading.Thread(target=auto_stop, daemon=True).start()

                return jsonify({"success": True, "message": f"{dev_name} 已开始工作"})

            # ── 停止工作 ──
            elif command == "stop":
                if current == "running":
                    dev["status"] = "standby"
                    # power 保持 True
                    log_event(dev_name, "停止工作", "回到待机（保持通电）")
                else:
                    log_event(dev_name, "停止工作", "当前未在工作")
                return jsonify({"success": True, "message": f"{dev_name} 已停止工作"})

            # ── 故障复位 ──
            elif command == "reset":
                if current == "error":
                    dev["power"] = True
                    dev["status"] = "standby"
                    log_event(dev_name, "故障复位", "恢复到待机")
                return jsonify({"success": True, "message": f"{dev_name} 已复位"})

            # ── 摄像头拍照 ──
            elif command == "capture":
                dev_type = dev.get("_type", "")
                if dev_type != "capture":
                    return jsonify({"success": False, "message": f"{dev_name} 不支持拍照功能"}), 400

                # 读取桌面病害图片模拟摄像头抓拍
                image_path = os.path.join(os.path.expanduser("~"), "Desktop", "病害1.jpg")
                if not os.path.exists(image_path):
                    log_event(dev_name, "拍照失败", f"图片不存在: {image_path}")
                    return jsonify({"success": False, "message": "模拟图片不存在，请将病害1.jpg放在桌面"}), 404

                try:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    dev["last_capture_time"] = datetime.now().isoformat()
                    log_event(dev_name, "拍照成功", f"图片大小: {len(image_bytes)} bytes")
                    return jsonify({
                        "success": True,
                        "message": f"抓拍成功 ({len(image_bytes)} bytes)",
                        "image_base64": image_b64,
                        "metadata": {
                            "width": 640, "height": 480,
                            "size_bytes": len(image_bytes),
                            "timestamp": datetime.now().isoformat(),
                            "source": image_path,
                        },
                    })
                except Exception as e:
                    log_event(dev_name, "拍照失败", str(e))
                    return jsonify({"success": False, "message": f"读取图片失败: {e}"}), 500

            elif command == "set_param":
                for k, v in params.items():
                    if k in dev and not k.startswith("_"):
                        dev[k] = v
                log_event(dev_name, "参数设置", str(params))
                return jsonify({"success": True, "message": "参数已更新"})

            return jsonify({"success": False, "message": f"不支持指令: {command}"}), 400

    def run():
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(1)

    # 验证启动
    try:
        import requests
        resp = requests.get(f"http://localhost:{port}/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            log_start(f"HTTP多设备服务器 ({data.get('device_count', 0)}台设备)", port)
            return app
    except Exception:
        pass

    log_start(f"HTTP多设备服务器", port)
    return app


# ═══════════════════════════════════════════════════
# MQTT 模拟硬件
# ═══════════════════════════════════════════════════

class MockMQTTNode:
    """模拟一个 MQTT 传感器/控制器节点（无需 broker）

    生命周期: powered_off → standby → running → standby → powered_off
    """

    def __init__(self, device_id, device_type="sensor"):
        self.device_id = device_id
        self.device_type = device_type
        # 初始状态：关机
        self.state = {
            "power": False,
            "status": "powered_off",
            "temperature": 24.0, "humidity": 60.0,
            "soil_moisture": 50.0, "ph": 6.8,
            "_read_at": ""
        }
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self, report_interval=5):
        """启动模拟节点（通电 + 待机）"""
        self._running = True
        with self._lock:
            self.state["power"] = True
            self.state["status"] = "standby"
        if self.device_type == "sensor":
            self._thread = threading.Thread(target=self._sensor_loop, args=(report_interval,), daemon=True)
        else:
            self._thread = threading.Thread(target=self._actuator_loop, args=(report_interval,), daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """停止模拟节点（关机）"""
        self._running = False
        with self._lock:
            self.state["power"] = False
            self.state["status"] = "powered_off"

    def _sensor_loop(self, interval):
        import random
        log_event(f"{self.device_id}(MQTT传感器)", "通电启动", f"每{interval}s上报数据")
        while self._running:
            with self._lock:
                self.state["temperature"] += random.uniform(-0.2, 0.2)
                self.state["temperature"] = round(max(-10, min(50, self.state["temperature"])), 1)
                self.state["humidity"] += random.uniform(-0.8, 0.8)
                self.state["humidity"] = round(max(0, min(100, self.state["humidity"])), 1)
                # 工作中土壤湿度上升，待机/关机则缓慢下降
                if self.state["status"] == "running":
                    self.state["soil_moisture"] += random.uniform(0.3, 1.0)
                else:
                    self.state["soil_moisture"] -= random.uniform(0.1, 0.2)
                self.state["soil_moisture"] = round(max(0, min(100, self.state["soil_moisture"])), 1)
                self.state["ph"] += random.uniform(-0.03, 0.03)
                self.state["ph"] = round(max(3.5, min(9.5, self.state["ph"])), 1)
                self.state["_read_at"] = datetime.now().isoformat()
            time.sleep(interval)

    def _actuator_loop(self, interval):
        import random
        log_event(f"{self.device_id}(MQTT控制器)", "待机", "等待指令")
        while self._running:
            time.sleep(interval)

    def execute(self, command, params=None):
        """执行指令，遵循状态机规则"""
        with self._lock:
            current = self.state["status"]

            # ── 通电 / 关机 ──
            if command in ("power_on", "boot"):
                if current == "powered_off":
                    self.state["power"] = True
                    self.state["status"] = "standby"
                    log_event(f"{self.device_id}(MQTT)", "通电启动", "进入待机")
                elif current == "standby":
                    log_event(f"{self.device_id}(MQTT)", "通电启动", "已在待机状态")
                return {"success": True, "message": f"{self.device_id} 已通电"}

            elif command in ("power_off", "shutdown"):
                if current in ("standby", "running"):
                    self.state["power"] = False
                    self.state["status"] = "powered_off"
                    log_event(f"{self.device_id}(MQTT)", "关机断电")
                elif current == "powered_off":
                    log_event(f"{self.device_id}(MQTT)", "关机断电", "已在关机状态")
                return {"success": True, "message": f"{self.device_id} 已关机"}

            # ── 开始工作 ──
            elif command == "start":
                if current == "powered_off":
                    self.state["power"] = True
                    self.state["status"] = "running"
                    log_event(f"{self.device_id}(MQTT)", "通电并启动", f"时长={(params or {}).get('duration', 0)}s")
                elif current == "standby":
                    self.state["status"] = "running"
                    duration = (params or {}).get("duration", 0)
                    log_event(f"{self.device_id}(MQTT)", "开始工作", f"时长={duration}s")
                elif current == "running":
                    log_event(f"{self.device_id}(MQTT)", "开始工作", "已在工作中")
                return {"success": True, "message": f"{self.device_id} 已开始工作"}

            # ── 停止工作 ──
            elif command == "stop":
                if current == "running":
                    self.state["status"] = "standby"
                    # power 保持 True，不断电！
                    log_event(f"{self.device_id}(MQTT)", "停止工作", "回到待机")
                elif current in ("standby", "powered_off"):
                    log_event(f"{self.device_id}(MQTT)", "停止工作", "当前未在工作")
                return {"success": True, "message": f"{self.device_id} 已停止工作"}

            # ── 故障复位 ──
            elif command == "reset":
                if current == "error":
                    self.state["power"] = True
                    self.state["status"] = "standby"
                    log_event(f"{self.device_id}(MQTT)", "故障复位", "恢复到待机")
                return {"success": True, "message": f"{self.device_id} 已复位"}

            return {"success": False, "message": f"不支持: {command}"}

    def read_state(self):
        with self._lock:
            return dict(self.state)


# ═══════════════════════════════════════════════════
# Simulator 驱动集成
# ═══════════════════════════════════════════════════

class PersistentSimulator:
    """持久化模拟器：保持设备状态，遵循完整生命周期状态机

    状态: powered_off(关机) → standby(待机) → running(工作中) → standby → powered_off
    """

    def __init__(self):
        self.devices: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def add_device(self, device_id, name, device_type, location="测试区"):
        """添加模拟设备，初始状态为关机(powered_off)"""
        initial_states = {
            "irrigate":   {"power": True, "status": "standby", "flow_rate": 0, "total_liters": 0},
            "ventilate":  {"power": True, "status": "standby", "rpm": 0},
            "light":      {"power": True, "status": "standby", "brightness": 0},
            "heat":       {"power": True, "status": "standby", "target_temp": 22},
            "cool":       {"power": True, "status": "standby", "target_temp": 24},
            "sensor":     {"power": True, "status": "standby",
                           "temperature": 24.5, "humidity": 62.0, "soil_moisture": 48.0, "ph": 6.8},
        }
        with self._lock:
            self.devices[device_id] = {
                "name": name,
                "type": device_type,
                "location": location,
                "state": dict(initial_states.get(device_type, {"power": True, "status": "standby"})),
            }
        log_start(f"{name}({device_type})", None)
        log_event(name, "初始状态: 关机", f"设备类型={device_type}")
        return self

    def execute(self, device_id, command, params=None):
        """执行设备指令，遵循状态机规则"""
        with self._lock:
            if device_id not in self.devices:
                return {"success": False, "message": "设备不存在"}

            dev = self.devices[device_id]
            params = params or {}
            current = dev["state"]["status"]
            name = dev["name"]

            # ── 通电启动 ──
            if command in ("power_on", "boot"):
                if current == "powered_off":
                    dev["state"]["power"] = True
                    dev["state"]["status"] = "standby"
                    log_event(name, "通电启动", "设备已进入待机状态")
                elif current == "standby":
                    log_event(name, "通电启动", "设备已在待机状态")
                elif current == "running":
                    log_event(name, "通电启动", "设备正在工作中，无需重复通电")
                elif current == "error":
                    log_event(name, "通电启动", "设备处于故障状态，请先复位(reset)")
                    return {"success": False, "message": "设备故障，请先执行 reset"}
                return {"success": True, "message": f"{name} 已通电"}

            # ── 关机断电 ──
            elif command in ("power_off", "shutdown"):
                if current in ("standby", "running"):
                    dev["state"]["power"] = False
                    dev["state"]["status"] = "powered_off"
                    log_event(name, "关机断电", f"关机前状态: {POWER_STATE_LABELS.get(current)}")
                elif current == "powered_off":
                    log_event(name, "关机断电", "设备已在关机状态")
                elif current == "error":
                    dev["state"]["power"] = False
                    dev["state"]["status"] = "powered_off"
                    log_event(name, "强制关机", "故障状态下断电")
                return {"success": True, "message": f"{name} 已关机"}

            # ── 开始工作 ──
            elif command == "start":
                if current == "powered_off":
                    # 从关机直接start：自动先通电再工作
                    dev["state"]["power"] = True
                    dev["state"]["status"] = "running"
                    detail = f"时长={params.get('duration', '?')}s" if "duration" in params else ""
                    log_event(name, "通电并启动", detail)
                elif current == "standby":
                    dev["state"]["status"] = "running"
                    if "duration" in params:
                        log_event(name, "开始工作", f"时长={params['duration']}s")
                    elif "flow_rate" in params:
                        log_event(name, "开始工作", f"流量={params['flow_rate']}")
                    elif "brightness" in params:
                        log_event(name, "开始工作", f"亮度={params['brightness']}")
                    elif "target_temp" in params:
                        log_event(name, "开始工作", f"目标温度={params['target_temp']}°C")
                    else:
                        log_event(name, "开始工作")
                elif current == "running":
                    log_event(name, "开始工作", "设备已在工作中，更新参数")
                    # 允许更新运行参数
                elif current == "error":
                    log_event(name, "开始工作", "设备故障，无法启动")
                    return {"success": False, "message": "设备故障，请先执行 reset"}
                return {"success": True, "message": f"{name} 已开始工作"}

            # ── 停止工作 ──
            elif command == "stop":
                if current == "running":
                    dev["state"]["status"] = "standby"
                    # 关键：power 保持 True，只停止工作，不断电！
                    log_event(name, "停止工作", "回到待机状态（保持通电）")
                elif current == "standby":
                    log_event(name, "停止工作", "设备当前未在工作（待机中）")
                elif current == "powered_off":
                    log_event(name, "停止工作", "设备处于关机状态")
                return {"success": True, "message": f"{name} 已停止工作"}

            # ── 故障复位 ──
            elif command == "reset":
                if current == "error":
                    dev["state"]["power"] = True
                    dev["state"]["status"] = "standby"
                    log_event(name, "故障复位", "已恢复到待机状态")
                else:
                    log_event(name, "复位", "设备未处于故障状态，无需复位")
                return {"success": True, "message": f"{name} 已复位"}

            # ── 参数设置 ──
            elif command == "set_param":
                for k, v in params.items():
                    if k in dev["state"]:
                        dev["state"][k] = v
                log_event(name, "参数设置", str(params))
                return {"success": True, "message": "参数已更新"}

            return {"success": False, "message": f"不支持: {command}"}

    def read_state(self, device_id):
        with self._lock:
            if device_id not in self.devices:
                return {}
            return dict(self.devices[device_id]["state"])

    def list_devices(self):
        with self._lock:
            return [
                {"id": did, "name": d["name"], "type": d["type"], "state": dict(d["state"])}
                for did, d in self.devices.items()
            ]


# ═══════════════════════════════════════════════════
# API 轮询监控
# ═══════════════════════════════════════════════════

class APIMonitor:
    """通过轮询后端API来监控前端设备操作"""

    def __init__(self, api_base="http://localhost:8000", username="123"):
        self.api_base = api_base
        self.username = username
        self._last_states: Dict[str, Dict] = {}
        self._running = False

    def _get_devices(self):
        import requests
        try:
            resp = requests.get(
                f"{self.api_base}/api/devices",
                params={"username": self.username},
                timeout=5
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return []

    def start(self, interval=2):
        self._running = True
        t = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        t.start()
        return t

    def stop(self):
        self._running = False

    def _monitor_loop(self, interval):
        time.sleep(3)  # 等后端就绪
        log_start("API设备监控", 8000)
        log_event("监控", "开始轮询", f"每{interval}s 检查设备状态变化")

        while self._running:
            try:
                devices = self._get_devices()
                for d in devices:
                    did = d.get("device_id", "")
                    state = d.get("state", {})
                    name = d.get("name", did)

                    if did in self._last_states:
                        prev = self._last_states[did]
                        # 检测状态变化
                        if prev.get("status") != state.get("status"):
                            old_label = POWER_STATE_LABELS.get(prev.get("status"), prev.get("status", "?"))
                            new_label = POWER_STATE_LABELS.get(state.get("status"), state.get("status", "?"))
                            log_event(name, f"状态变化: {old_label} -> {new_label}", "(前端操控)")
                        elif prev.get("power") != state.get("power"):
                            if state.get("power"):
                                log_event(name, "通电", "(前端操控)")
                            else:
                                log_event(name, "断电", "(前端操控)")

                    self._last_states[did] = dict(state)
            except Exception:
                pass
            time.sleep(interval)


# ═══════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════

def print_banner():
    print(f"""
{C.BOLD}{C.G}
  ╔══════════════════════════════════════════════════════╗
  ║     🌾 智能种植助手 — 硬件模拟器 v3.0               ║
  ║     Farm Hardware Simulator                          ║
  ╚══════════════════════════════════════════════════════╝{C.W}

  {C.Y}模拟温室设备 (HTTP 真实驱动):{C.W}
     灌溉泵      |  通风扇      |  补光灯
     加热器      | ️ 温湿度传感器 |  施肥一体机

""")


def build_command_handlers(sim, mqtt_nodes):
    """构建命令行处理函数"""

    def show_status():
        """显示所有设备状态"""
        print(f"\n  {C.BOLD}{'设备ID':38s} {'通电':6s} {'状态':10s} {'详细信息'}{C.W}")
        print(f"  {'-'*75}")
        for d in sim.list_devices():
            state = d["state"]
            power_icon = f"{C.G}是{C.W}" if state.get("power") else f"{C.R}否{C.W}"
            status_str = status_display(state)
            extras = []
            if "temperature" in state:
                extras.append(f"temp={state['temperature']}°C")
            if "humidity" in state:
                extras.append(f"hum={state['humidity']}%")
            if "soil_moisture" in state:
                extras.append(f"soil={state['soil_moisture']}%")
            if "flow_rate" in state:
                extras.append(f"flow={state['flow_rate']}")
            if "rpm" in state:
                extras.append(f"rpm={state['rpm']}")
            if "brightness" in state:
                extras.append(f"bri={state['brightness']}")
            print(f"  {d['id']:38s} {power_icon:8s} {status_str:14s} {', '.join(extras)}")

        # MQTT nodes
        for nid, node in mqtt_nodes.items():
            s = node.read_state()
            p_icon = f"{C.G}是{C.W}" if s.get("power") else f"{C.R}否{C.W}"
            st = status_display(s)
            print(f"  {f'{nid}(MQTT)':38s} {p_icon:8s} {st:14s} temp={s['temperature']}°C hum={s['humidity']}%")
        print()

    def do_command(device_id, command, params=None):
        # 先尝试持久化模拟器
        result = sim.execute(device_id, command, params)
        if result["success"]:
            return result

        # 再尝试 MQTT 节点
        if device_id in mqtt_nodes:
            return mqtt_nodes[device_id].execute(command, params)

        print(f"  {C.R}设备不存在: {device_id}{C.W}")
        return {"success": False, "message": "设备不存在"}

    return show_status, do_command


def main():
    print_banner()

    # ── 1. 多设备 HTTP 服务器（6台温室设备）──
    http_app = start_http_server(5000)

    # ── 2. API 监控（轮询后端检测前端操控）──
    monitor = APIMonitor()
    monitor.start(interval=2)

    # ── 3. 交互循环 ──
    print(f"\n  {C.BOLD}{'='*70}{C.W}")
    print(f"  {C.G}6台温室设备已就绪（初始状态: 关机），等待指令...{C.W}")
    print(f"  {C.Y}前端操控设备时，此终端实时显示硬件反馈{C.W}")
    print(f"  {C.BOLD}{'='*70}{C.W}\n")
    print(f"  {C.B}可用命令:{C.W}")
    print(f"    {C.C}list / l{C.W}         查看所有设备状态")
    print(f"    {C.C}boot <id>{C.W}         通电启动 (如: boot pump)")
    print(f"    {C.C}start <id>{C.W}        开始工作 (如: start pump)")
    print(f"    {C.C}stop <id>{C.W}         停止工作")
    print(f"    {C.C}shutdown <id>{C.W}    关机断电")
    print(f"    {C.C}help / h / quit / q{C.W}")
    print(f"\n  {C.B}设备快捷ID:{C.W}")
    print(f"    pump / fan / light / heat / sensor / fert")
    print()

    shortcuts = {
        "pump": "irrigation_pump_01", "fan": "ventilation_fan_01",
        "light": "grow_light_01", "heat": "heater_01",
        "sensor": "env_sensor_01", "fert": "fertilizer_pump_01",
    }

    def show_all_status():
        """通过 HTTP 请求查看服务器上所有设备状态"""
        try:
            import requests
            resp = requests.get("http://localhost:5000/health", timeout=3)
            health = resp.json()
            print(f"\n  {C.BOLD}HTTP服务器: {health.get('status')}, 设备总数: {health.get('device_count')}{C.W}")

            resp2 = requests.get("http://localhost:5000/api/state", timeout=3)
            summary = resp2.json().get("devices", {})
            print(f"  {C.BOLD}{'设备ID':30s} {'名称':16s} {'状态':10s}{C.W}")
            print(f"  {'-'*60}")
            for did, info in summary.items():
                st = info.get("status", "powered_off")
                label = POWER_STATE_LABELS.get(st, st)
                icon = "●" if st == "running" else "○" if st == "standby" else "○"
                print(f"  {did:30s} {info['name']:16s} {icon} {label}")
            print()
        except Exception as e:
            print(f"  {C.R}无法连接HTTP服务器: {e}{C.W}")

    def send_http_command(device_id, command, params=None):
        """发送HTTP指令到模拟硬件"""
        try:
            import requests
            payload = {"device_id": device_id, "command": command, "params": params or {}}
            resp = requests.post("http://localhost:5000/api/command", json=payload, timeout=5)
            result = resp.json()
            if result.get("success"):
                print(f"  {C.G}[OK]{C.W} {result.get('message', '')}")
            else:
                print(f"  {C.R}[FAIL]{C.W} {result.get('message', '')}")
        except Exception as e:
            print(f"  {C.R}指令发送失败: {e}{C.W}")

    try:
        while True:
            try:
                line = input(f"  {C.C}>>> {C.W}").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("quit", "q", "exit"):
                break
            elif cmd in ("help", "h"):
                print(f"\n  {C.B}快捷ID:{C.W}")
                for k, v in shortcuts.items():
                    print(f"    {k:8s} -> {v}")
                print()
            elif cmd in ("list", "l", "status", "s"):
                show_all_status()
            elif cmd in ("boot", "power_on") and len(parts) >= 2:
                target = shortcuts.get(parts[1], parts[1])
                send_http_command(target, "power_on")
            elif cmd == "start" and len(parts) >= 2:
                target = shortcuts.get(parts[1], parts[1])
                send_http_command(target, "start", {"duration": 10})
            elif cmd == "stop" and len(parts) >= 2:
                target = shortcuts.get(parts[1], parts[1])
                send_http_command(target, "stop")
            elif cmd in ("shutdown", "power_off") and len(parts) >= 2:
                target = shortcuts.get(parts[1], parts[1])
                send_http_command(target, "power_off")
            elif cmd == "set" and len(parts) >= 3:
                target = shortcuts.get(parts[1], parts[1])
                try:
                    k, v = parts[2].split("=")
                    send_http_command(target, "set_param", {k: float(v)})
                except ValueError:
                    print(f"  {C.R}格式: set <device> key=value{C.W}")
            else:
                print(f"  {C.R}未知命令: {cmd}{C.W}")

    except KeyboardInterrupt:
        pass

    # 清理
    print(f"\n  {C.Y}正在停止所有硬件模拟...{C.W}")
    monitor.stop()
    print(f"  {C.G}硬件模拟器已停止{C.W}\n")


if __name__ == "__main__":
    main()
