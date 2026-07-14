"""
一体化硬件模拟器 v5.0 — 完整协议栈模拟

每个虚拟设备绑定真实通信协议，终端/前端/API 均通过协议通道控制设备。
无需真实硬件，无需 mosquitto/pymodbus/paho-mqtt。

架构:
  UnifiedDeviceManager（共享状态）
   ├── 🌐 HTTP Server (Flask :5000)
   │    └── irrigation_pump_01, grow_light_01, fertilizer_pump_01
   ├── 📡 MQTT Broker (内嵌 :1883) + 设备处理器
   │    └── ventilation_fan_01, heater_01
   ├── 🔧 Modbus TCP Server (内嵌 :5020)
   │    └── env_sensor_01, greenhouse_camera_01
   └── 🖥️  终端 CLI（协议客户端）
        ├── HTTP设备 → requests.post(:5000)
        ├── MQTT设备 → MQTT publish(:1883)
        └── Modbus设备 → TCP write(:5020)

启动: python hardware_examples/all_hardware_simulator.py
"""

import sys
import os
import json
import time
import struct
import socket
import random
import threading
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════
# 终端颜色
# ═══════════════════════════════════════════════════

class C:
    R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"
    B = "\033[94m"; C = "\033[96m"; M = "\033[95m"
    W = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"

    @staticmethod
    def time():
        return f"{C.DIM}{datetime.now().strftime('%H:%M:%S')}{C.W}"


# ═══════════════════════════════════════════════════
# 状态机常量
# ═══════════════════════════════════════════════════

POWER_STATE_LABELS = {
    "powered_off": "关机", "standby": "待机",
    "running": "工作中", "error": "故障",
}

# ═══════════════════════════════════════════════════
# 设备定义 — 每台设备绑定一个协议
# ═══════════════════════════════════════════════════

# 协议类型
PROTO_HTTP = "HTTP"
PROTO_MQTT = "MQTT"
PROTO_MODBUS = "Modbus"

DEVICE_DEFS = {
    "irrigation_pump_01": {
        "name": "温室灌溉泵",
        "type": "irrigate",
        "protocol": PROTO_HTTP,
        "http_url": "http://127.0.0.1:5000",
        "initial": {"power": True, "status": "standby", "flow_rate": 0, "total_water_liters": 156.8},
        "sensors": ["flow_rate", "total_water_liters"],
    },
    "ventilation_fan_01": {
        "name": "温室通风扇",
        "type": "ventilate",
        "protocol": PROTO_MQTT,
        "mqtt_topic": "devices/ventilation_fan_01",
        "initial": {"power": True, "status": "standby", "rpm": 0},
        "sensors": ["rpm"],
    },
    "grow_light_01": {
        "name": "温室补光灯",
        "type": "light",
        "protocol": PROTO_HTTP,
        "http_url": "http://127.0.0.1:5000",
        "initial": {"power": True, "status": "standby", "brightness_percent": 0},
        "sensors": ["brightness_percent"],
    },
    "heater_01": {
        "name": "温室加热器",
        "type": "heat",
        "protocol": PROTO_MQTT,
        "mqtt_topic": "devices/heater_01",
        "initial": {"power": True, "status": "standby", "target_temp": 22, "current_temp": 18.5},
        "sensors": ["target_temp", "current_temp"],
    },
    "env_sensor_01": {
        "name": "环境温湿度传感器",
        "type": "read_sensor",
        "protocol": PROTO_MODBUS,
        "modbus_slave": 1,
        "initial": {"power": True, "status": "standby",
                    "temperature": 24.5, "humidity": 62.0, "soil_moisture": 48.0, "ph": 6.8, "light_lux": 35000},
        "sensors": ["temperature", "humidity", "soil_moisture", "ph", "light_lux"],
    },
    "fertilizer_pump_01": {
        "name": "施肥一体机",
        "type": "fertigate",
        "protocol": PROTO_HTTP,
        "http_url": "http://127.0.0.1:5000",
        "initial": {"power": True, "status": "standby", "flow_rate": 0, "total_fertilizer_kg": 23.5},
        "sensors": ["flow_rate", "total_fertilizer_kg"],
    },
    "greenhouse_camera_01": {
        "name": "温室监控摄像头",
        "type": "capture",
        "protocol": PROTO_MODBUS,
        "modbus_slave": 2,
        "initial": {"power": True, "status": "standby", "resolution": "640x480", "last_capture_time": None},
        "sensors": ["resolution"],
    },
}

# 快捷ID
SHORTCUTS = {
    "pump": "irrigation_pump_01", "fan": "ventilation_fan_01",
    "light": "grow_light_01", "heat": "heater_01",
    "sensor": "env_sensor_01", "fert": "fertilizer_pump_01",
    "camera": "greenhouse_camera_01",
}

PROTO_ICONS = {
    PROTO_HTTP: "🌐",
    PROTO_MQTT: "📡",
    PROTO_MODBUS: "🔧",
}


# ═══════════════════════════════════════════════════
# 安全输出（处理 Windows GBK 编码）
# ═══════════════════════════════════════════════════

def _safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('gbk', errors='replace').decode('gbk'))


# ═══════════════════════════════════════════════════
# 事件日志
# ═══════════════════════════════════════════════════

def log(device_name: str, action: str, detail: str = "", proto: str = ""):
    """打印设备事件"""
    p = f" {C.DIM}[{proto}]{C.W}" if proto else ""
    d = f" ({detail})" if detail else ""
    _safe_print(f"  [{C.time()}]{p} {C.BOLD}{device_name}{C.W} -> {C.Y}{action}{C.W}{d}")


def log_info(msg: str, color: str = ""):
    c = color or C.W
    _safe_print(f"  [{C.time()}] {c}{msg}{C.W}")


def status_icon(state: dict) -> str:
    s = state.get("status", "powered_off")
    if s == "running":   return f"{C.G}●{C.W}"
    if s == "standby":   return f"{C.Y}○{C.W}"
    if s == "error":     return f"{C.R}✕{C.W}"
    return f"{C.DIM}○{C.W}"  # powered_off


# ═══════════════════════════════════════════════════
# 统一设备管理器 — 所有协议服务器的共享状态后端
# ═══════════════════════════════════════════════════

class UnifiedDeviceManager:
    """线程安全的设备状态管理器。HTTP/MQTT/Modbus 服务器共享此实例。"""

    def __init__(self):
        self._devices: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def init_all(self):
        """从 DEVICE_DEFS 初始化所有设备"""
        for dev_id, defn in DEVICE_DEFS.items():
            self._devices[dev_id] = {
                "_name": defn["name"],
                "_type": defn["type"],
                "_protocol": defn["protocol"],
                "_sensors": defn.get("sensors", []),
                **(dict(defn["initial"])),
            }
        log_info(f"设备管理器初始化: {len(self._devices)} 台设备（全部待机通电）", C.G)

    def get(self, device_id: str) -> Optional[dict]:
        return self._devices.get(device_id)

    def read_state(self, device_id: str) -> dict:
        dev = self._devices.get(device_id)
        if not dev: return {}
        return {k: v for k, v in dev.items() if not k.startswith("_")}

    def list_all(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "id": did, "name": d["_name"], "type": d["_type"],
                    "protocol": d["_protocol"],
                    "state": {k: v for k, v in d.items() if not k.startswith("_")},
                }
                for did, d in self._devices.items()
            ]

    def execute(self, device_id: str, command: str, params: dict = None,
                source_proto: str = "") -> dict:
        """执行设备指令，返回结果。协议无关——各协议服务器调用此方法。"""
        params = params or {}
        with self._lock:
            dev = self._devices.get(device_id)
            if dev is None:
                return {"success": False, "message": f"设备 '{device_id}' 不存在"}

            dev_name = dev["_name"]
            current = dev.get("status", "powered_off")

            if command in ("power_on", "boot"):
                if current == "powered_off":
                    dev["power"] = True; dev["status"] = "standby"
                    log(dev_name, "通电启动 -> 待机", proto=source_proto)
                elif current == "standby":
                    log(dev_name, "通电启动", "已在待机", source_proto)
                elif current == "error":
                    return {"success": False, "message": f"{dev_name} 故障中，请先复位"}
                return {"success": True, "message": f"{dev_name} 已通电（待机）"}

            elif command in ("power_off", "shutdown"):
                if current in ("standby", "running", "error"):
                    old = POWER_STATE_LABELS.get(current, current)
                    dev["power"] = False; dev["status"] = "powered_off"
                    log(dev_name, "关机断电", f"关机前: {old}", source_proto)
                return {"success": True, "message": f"{dev_name} 已关机"}

            elif command == "start":
                if current == "error":
                    return {"success": False, "message": f"{dev_name} 故障中，请先复位"}
                if current == "powered_off":
                    dev["power"] = True
                dev["status"] = "running"
                log(dev_name, "开始工作", _fmt_params(params), source_proto)
                for k, v in params.items():
                    if k in dev and not k.startswith("_"):
                        dev[k] = v
                dur = params.get("duration", 0)
                if dur > 0:
                    self._schedule_stop(device_id, dev_name, dur)
                return {"success": True, "message": f"{dev_name} 已开始工作"}

            elif command == "stop":
                if current == "running":
                    dev["status"] = "standby"
                    log(dev_name, "停止工作 -> 待机", proto=source_proto)
                else:
                    log(dev_name, "停止工作", "当前未在工作中", source_proto)
                return {"success": True, "message": f"{dev_name} 已停止"}

            elif command == "reset":
                if current == "error":
                    dev["power"] = True; dev["status"] = "standby"
                    log(dev_name, "故障复位 -> 待机", proto=source_proto)
                return {"success": True, "message": f"{dev_name} 已复位"}

            elif command == "set_param":
                changed = {}
                for k, v in params.items():
                    if k in dev and not k.startswith("_"):
                        changed[k] = f"{dev[k]}->{v}"; dev[k] = v
                if changed:
                    log(dev_name, "参数更新", str(changed), source_proto)
                return {"success": True, "message": f"参数已更新"}

            elif command == "capture":
                return self._do_capture(dev, dev_name, source_proto)

            elif command == "read_sensor":
                data = {k: v for k, v in dev.items()
                        if not k.startswith("_") and isinstance(v, (int, float))}
                return {"success": True, "sensor_data": data}

            return {"success": False, "message": f"不支持: {command}"}

    def _schedule_stop(self, dev_id: str, name: str, dur_min: int):
        def auto():
            time.sleep(dur_min * 60)
            with self._lock:
                d = self._devices.get(dev_id)
                if d and d.get("status") == "running":
                    d["status"] = "standby"
                    log(name, "定时结束 -> 待机", f"运行{dur_min}分钟", "[定时]")
        threading.Thread(target=auto, daemon=True).start()

    def _do_capture(self, dev: dict, name: str, source: str) -> dict:
        import base64
        if dev["_type"] != "capture":
            return {"success": False, "message": f"{name} 不支持拍照"}
        # 优先读取桌面真实图片（用于病害识别测试），不存在则生成模拟 JPEG
        path = os.path.join(os.path.expanduser("~"), "Desktop", "病害1.jpg")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = f.read()
                dev["last_capture_time"] = datetime.now().isoformat()
                log(name, "拍照成功", f"{len(data)//1024}KB (真实图片)", source)
                return {
                    "success": True,
                    "message": f"OK ({len(data)} bytes)",
                    "image_base64": base64.b64encode(data).decode("utf-8"),
                    "metadata": {"width": 640, "height": 480, "size_bytes": len(data),
                                 "timestamp": datetime.now().isoformat(), "source": "desktop_image"},
                }
            except Exception as e:
                log(name, "读取真实图片失败，回退模拟", str(e), source)
        # fallback: 生成模拟 JPEG 图像（最小有效 JPEG，约 1KB 绿色图）
        try:
            sim_data = self._generate_simulated_jpeg()
            dev["last_capture_time"] = datetime.now().isoformat()
            log(name, "拍照成功", f"{len(sim_data)}B (模拟图片)", source)
            return {
                "success": True,
                "message": f"OK — 模拟图片 ({len(sim_data)} bytes)",
                "image_base64": base64.b64encode(sim_data).decode("utf-8"),
                "metadata": {"width": 320, "height": 240, "size_bytes": len(sim_data),
                             "timestamp": datetime.now().isoformat(), "source": "simulated"},
            }
        except Exception as e:
            log(name, "拍照失败", str(e), source)
            return {"success": False, "message": str(e)}

    @staticmethod
    def _generate_simulated_jpeg() -> bytes:
        """生成最小有效 JPEG 图像（绿色模拟作物画面）"""
        import struct
        # 最小 JPEG: SOI + APP0/JFIF + DQT + SOF0 + DHT + SOS + compressed data + EOI
        # 使用一个极简的 1x1 绿色像素 JPEG
        jpeg = bytes([
            0xFF, 0xD8,  # SOI
            0xFF, 0xE0,  # APP0
            0x00, 0x10,  # len
            0x4A, 0x46, 0x49, 0x46, 0x00,  # "JFIF\0"
            0x01, 0x01,  # v1.1
            0x00,        # units
            0x00, 0x01, 0x00, 0x01,  # 1x1 dpi
            0x00, 0x00,  # thumbnail
            # DQT
            0xFF, 0xDB, 0x00, 0x43, 0x00,
        ] + bytes([0x10] * 64) + bytes([  # quant table
            # SOF0
            0xFF, 0xC0, 0x00, 0x0B, 0x08,
            0x00, 0x01, 0x00, 0x01,  # 1x1
            0x01, 0x01, 0x00,
            # DHT
            0xFF, 0xC4, 0x00, 0x1F, 0x00,
            0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
            # SOS + compressed
            0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
            0x00, 0x00,  # dummy MCU
            # EOI
            0xFF, 0xD9,
        ])
        return jpeg


def _fmt_params(p: dict) -> str:
    if not p: return ""
    m = {"duration": "时长", "amount_kg": "施肥量", "speed_percent": "转速",
         "brightness_percent": "亮度", "target_temp": "目标温度", "flow_rate": "流量"}
    return ", ".join(f"{m.get(k,k)}={v}" for k, v in p.items())


# ═══════════════════════════════════════════════════
# 🌐 HTTP 设备服务器 (Flask)
# ═══════════════════════════════════════════════════

def create_http_server(mgr: UnifiedDeviceManager) -> bool:
    """启动 Flask HTTP 服务器，所有 HTTP 设备通过此服务器暴露。"""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        log_info("[HTTP] Flask 未安装: pip install flask", C.R)
        return False

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "server": "farm-sim-v5"})

    @app.route("/api/state", methods=["GET"])
    @app.route("/state", methods=["GET"])
    def get_state():
        device_id = request.args.get("device_id", "")
        if device_id:
            dev = mgr.get(device_id)
            if dev:
                return jsonify({k: v for k, v in dev.items() if not k.startswith("_")})
            return jsonify({"error": "not found"}), 404
        # 返回所有 HTTP 设备概览
        summary = {}
        for d in mgr.list_all():
            if d["protocol"] == PROTO_HTTP:
                summary[d["id"]] = {"name": d["name"], "power": d["state"].get("power"),
                                    "status": d["state"].get("status")}
        return jsonify({"devices": summary, "total": len(summary)})

    @app.route("/api/command", methods=["POST"])
    @app.route("/command", methods=["POST"])
    def execute():
        data = request.get_json(silent=True) or {}
        device_id = data.get("device_id", "")
        command = data.get("command", "")
        params = data.get("params", {})
        if not device_id:
            return jsonify({"success": False, "message": "缺少 device_id"}), 400
        dev = mgr.get(device_id)
        if not dev or dev["_protocol"] != PROTO_HTTP:
            return jsonify({"success": False, "message": f"设备 '{device_id}' 不存在或非HTTP"}), 404
        src = request.headers.get("X-Source", PROTO_HTTP)
        result = mgr.execute(device_id, command, params, source_proto=src)
        code = 200 if result.get("success") else 400
        return jsonify(result), code

    def run():
        import logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.5)

    # 验证
    try:
        import requests
        r = requests.get("http://localhost:5000/health", timeout=3)
        if r.status_code == 200:
            log_info(f"[HTTP] 服务器就绪 -> :5000", C.G)
            return True
    except Exception:
        pass
    log_info(f"[HTTP] 服务器就绪 -> :5000", C.G)
    return True


# ═══════════════════════════════════════════════════
# 📡 内嵌 MQTT Broker（纯 Python，零依赖）
# ═══════════════════════════════════════════════════

# MQTT 数据包类型
MQTT_CONNECT, MQTT_CONNACK = 1, 2
MQTT_PUBLISH, MQTT_SUBSCRIBE = 3, 8
MQTT_SUBACK, MQTT_PINGREQ = 9, 12
MQTT_PINGRESP, MQTT_DISCONNECT = 13, 14


class MqttBroker:
    """内嵌 MQTT 3.1.1 Broker（QoS 0，支持 +/# 通配符）"""

    def __init__(self, host: str = "127.0.0.1", port: int = 1883):
        self._host = host
        self._port = port
        self._subscriptions: Dict[str, List] = {}     # topic -> [writer, ...]
        self._wildcard_subs: List[Tuple[str, any]] = [] # [(pattern, writer), ...]
        self._running = False
        self._started = False  # 是否成功绑定端口

    @staticmethod
    def _match_topic(pattern: str, topic: str) -> bool:
        pp = pattern.split("/"); tp = topic.split("/")
        for i, p in enumerate(pp):
            if p == "#": return True
            if i >= len(tp): return False
            if p == "+": continue
            if p != tp[i]: return False
        return len(pp) == len(tp)

    @staticmethod
    def _parse_rem_len(data: bytes, offset: int) -> Tuple[int, int]:
        mul, val = 1, 0
        while offset < len(data):
            b = data[offset]; val += (b & 0x7F) * mul; offset += 1
            if (b & 0x80) == 0: break
            mul *= 128
        return val, offset

    async def _handle(self, reader, writer):
        addr = writer.get_extra_info("peername")
        cid = str(addr)
        buf = b""
        try:
            while True:
                data = await asyncio.wait_for(reader.read(4096), timeout=120)
                if not data: break
                buf += data
                while len(buf) >= 2:
                    ptype = (buf[0] & 0xF0) >> 4
                    if ptype not in (MQTT_CONNECT, MQTT_SUBSCRIBE, MQTT_PUBLISH,
                                     MQTT_PINGREQ, MQTT_DISCONNECT):
                        break
                    try:
                        rem, pos = self._parse_rem_len(buf, 1)
                    except Exception:
                        break
                    total = pos + rem
                    if len(buf) < total: break
                    packet = buf[:total]; buf = buf[total:]

                    if ptype == MQTT_CONNECT:
                        # 解析 client_id
                        plen = struct.unpack(">H", packet[2:4])[0]
                        cpos = 4 + plen + 3  # proto + level + flags + keepalive
                        clen = struct.unpack(">H", packet[cpos:cpos+2])[0]
                        cid = packet[cpos+2:cpos+2+clen].decode("utf-8", errors="replace")
                        writer.write(bytes([0x20, 0x02, 0x00, 0x00]))
                        await writer.drain()
                        log_info(f"[MQTT Broker] 客户端连接: {cid}", C.DIM)

                    elif ptype == MQTT_SUBSCRIBE:
                        pid = struct.unpack(">H", packet[pos:pos+2])[0]
                        off = pos + 2
                        topics = []
                        while off < total:
                            tlen = struct.unpack(">H", packet[off:off+2])[0]
                            off += 2
                            topic = packet[off:off+tlen].decode("utf-8")
                            off += tlen + 1  # skip QoS
                            topics.append(topic)
                        for tp in topics:
                            if "#" in tp or "+" in tp:
                                self._wildcard_subs.append((tp, writer))
                            else:
                                self._subscriptions.setdefault(tp, []).append(writer)
                        # SUBACK
                        ack = bytearray([0x90, 2 + len(topics)])
                        ack += struct.pack(">H", pid)
                        for _ in topics:
                            ack.append(0x00)  # QoS 0
                        writer.write(bytes(ack))
                        await writer.drain()
                        log_info(f"[MQTT Broker] {cid} 订阅: {topics}", C.DIM)

                    elif ptype == MQTT_PUBLISH:
                        tlen = struct.unpack(">H", packet[pos:pos+2])[0]
                        topic = packet[pos+2:pos+2+tlen].decode("utf-8")
                        qos_level = (packet[0] & 0x06) >> 1
                        # QoS > 0 时 payload 前有 2 字节 packet identifier，需跳过
                        payload_start = pos + 2 + tlen + (2 if qos_level > 0 else 0)
                        payload = packet[payload_start:]
                        # 转发给订阅者（强制降级为 QoS 0）
                        targets = set(self._subscriptions.get(topic, []))
                        for pat, w in self._wildcard_subs:
                            if self._match_topic(pat, topic):
                                targets.add(w)
                        tbytes = topic.encode("utf-8")
                        fwd = bytes([0x30]) + self._encode_rem(2 + len(tbytes) + len(payload))
                        fwd += struct.pack(">H", len(tbytes)) + tbytes + payload
                        for w in targets:
                            if w != writer:
                                try:
                                    w.write(fwd)
                                    await w.drain()
                                except Exception:
                                    pass

                    elif ptype == MQTT_PINGREQ:
                        writer.write(bytes([0xD0, 0x00]))
                        await writer.drain()

                    elif ptype == MQTT_DISCONNECT:
                        break
        except (asyncio.TimeoutError, ConnectionError, OSError):
            pass
        finally:
            for tp in list(self._subscriptions):
                self._subscriptions[tp] = [w for w in self._subscriptions[tp] if w != writer]
                if not self._subscriptions[tp]:
                    del self._subscriptions[tp]
            self._wildcard_subs[:] = [(p, w) for p, w in self._wildcard_subs if w != writer]
            try: writer.close()
            except Exception: pass

    @staticmethod
    def _encode_rem(length: int) -> bytes:
        result = bytearray()
        while length > 0:
            b = length & 0x7F; length >>= 7
            if length > 0: b |= 0x80
            result.append(b)
        return bytes(result)

    async def _run(self):
        self._running = True
        try:
            server = await asyncio.start_server(
                self._handle, self._host, self._port,
                reuse_address=True,  # 允许端口快速复用
            )
            log_info(f"[MQTT Broker] 启动 -> {self._host}:{self._port}", C.G)
            self._started = True
            async with server:
                await server.serve_forever()
        except OSError as e:
            log_info(f"[MQTT Broker] 端口 {self._port} 被占用: {e}", C.R)
            log_info(f"[MQTT Broker] 请先关闭占用端口的程序，或: netstat -ano | findstr \":{self._port}\"", C.Y)
            self._started = False

    def start(self):
        """在后台线程启动 Broker"""
        def _go():
            asyncio.run(self._run())
        t = threading.Thread(target=_go, daemon=True)
        t.start()
        time.sleep(0.3)
        return t


# ═══════════════════════════════════════════════════
# MQTT 设备处理器（订阅控制主题，上报状态）
# ═══════════════════════════════════════════════════

class MqttDeviceHandler:
    """MQTT 设备处理器 — 基于 paho-mqtt，订阅控制主题并处理指令。"""

    def __init__(self, mgr: UnifiedDeviceManager, broker_host: str = "127.0.0.1",
                 broker_port: int = 1883):
        self._mgr = mgr
        self._host = broker_host
        self._port = broker_port
        self._running = False
        self._topic_to_device: Dict[str, str] = {}  # topic -> device_id

    def connect(self) -> bool:
        """连接 Broker 并订阅所有 MQTT 设备的控制主题"""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            log_info("[MQTT Handler] paho-mqtt 未安装", C.R)
            return False

        self._client = mqtt.Client(
            client_id="mqtt_device_handler",
            protocol=mqtt.MQTTv311,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
        )
        self._connect_ok = threading.Event()
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.connect_async(self._host, self._port, keepalive=60)
        self._client.loop_start()

        # 等待连接
        if not self._connect_ok.wait(timeout=3):
            log_info("[MQTT Handler] 连接 Broker 超时", C.R)
            return False

        # 订阅所有 MQTT 设备的控制主题
        for d in self._mgr.list_all():
            if d["protocol"] == PROTO_MQTT:
                did = d["id"]
                defn = DEVICE_DEFS.get(did, {})
                base_topic = defn.get("mqtt_topic", f"devices/{did}")
                ctrl_topic = f"{base_topic}/control"
                self._topic_to_device[ctrl_topic] = did
                self._client.subscribe(ctrl_topic, qos=0)
                log_info(f"[MQTT Handler] 订阅: {ctrl_topic}", C.DIM)

        log_info(f"[MQTT Handler] 已订阅 {len(self._topic_to_device)} 个设备控制主题", C.G)
        self._running = True
        self._last_publish = 0

        # 启动状态发布线程
        def _publish_loop():
            while self._running:
                time.sleep(2)
                if self._running:
                    for did in self._topic_to_device.values():
                        self.publish_state(did)

        threading.Thread(target=_publish_loop, daemon=True).start()
        return True

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connect_ok.set()

    def _on_message(self, client, userdata, msg):
        """收到控制指令"""
        did = self._topic_to_device.get(msg.topic)
        if not did:
            return
        try:
            cmd_data = json.loads(msg.payload.decode("utf-8"))
            cmd = cmd_data.get("command", "")
            params = cmd_data.get("params", {})
            dev = self._mgr.get(did)
            name = dev["_name"] if dev else did
            log(name, f"收到指令: {cmd}", _fmt_params(params), PROTO_MQTT)
            result = self._mgr.execute(did, cmd, params, source_proto=PROTO_MQTT)
            self.publish_state(did)
            if result.get("success"):
                log(name, f"执行成功: {result.get('message', '')}", proto=PROTO_MQTT)
            else:
                log(name, f"执行失败: {result.get('message', '')}", proto=PROTO_MQTT)
        except json.JSONDecodeError:
            pass

    def publish_state(self, device_id: str):
        """发布设备当前状态到状态主题"""
        defn = DEVICE_DEFS.get(device_id, {})
        base_topic = defn.get("mqtt_topic", f"devices/{device_id}")
        state_topic = f"{base_topic}/state"
        state = self._mgr.read_state(device_id)
        try:
            self._client.publish(state_topic, json.dumps(state, ensure_ascii=False), qos=0)
        except Exception:
            pass


# ═══════════════════════════════════════════════════
# MQTT 客户端（供终端 CLI 使用）
# ═══════════════════════════════════════════════════

class SimpleMqttClient:
    """简易 MQTT 客户端 — 终端通过此客户端向 MQTT 设备发送指令。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 1883):
        self._host = host; self._port = port
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def connect(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5)
            self._sock.connect((self._host, self._port))
            cid = b"terminal_cli"
            payload = b"\x00\x04MQTT\x04\x00\x00\x00"
            payload += struct.pack(">H", len(cid)) + cid
            rem = self._encode_rem(len(payload))
            self._sock.sendall(bytes([0x10]) + rem + payload)
            resp = self._sock.recv(4)
            return len(resp) >= 4
        except Exception as e:
            log_info(f"[MQTT Client] 连接失败: {e}", C.R)
            return False

    def publish(self, topic: str, payload: dict):
        """发布 JSON 消息到指定主题"""
        with self._lock:
            try:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                tbytes = topic.encode("utf-8")
                pkt = bytes([0x30])
                pkt += self._encode_rem(2 + len(tbytes) + len(data))
                pkt += struct.pack(">H", len(tbytes)) + tbytes + data
                self._sock.sendall(pkt)
            except Exception as e:
                log_info(f"[MQTT Client] 发布失败: {e}", C.R)

    def close(self):
        try: self._sock.close()
        except Exception: pass

    @staticmethod
    def _encode_rem(length: int) -> bytes:
        result = bytearray()
        while length > 0:
            b = length & 0x7F; length >>= 7
            if length > 0: b |= 0x80
            result.append(b)
        return bytes(result)


# ═══════════════════════════════════════════════════
# 🔧 Modbus TCP 服务器（纯 Python，零依赖）
# ═══════════════════════════════════════════════════

MODBUS_FC_READ_HOLDING = 0x03
MODBUS_FC_WRITE_SINGLE = 0x06
MODBUS_FC_WRITE_MULTIPLE = 0x10


class ModbusTcpServer:
    """简易 Modbus TCP 服务器 — 模拟 Modbus 从站设备。

    寄存器映射（每个从站 30 个保持寄存器）:
      HR[0]:  设备状态 (0=关机, 1=待机, 2=工作中, 3=故障)
      HR[1]:  电源 (0=关, 1=开)
      HR[2]:  指令寄存器 (1=boot, 2=shutdown, 3=start, 4=stop, 5=reset)
      HR[3]:  指令参数(duration 秒数)
      HR[4]:  指令参数(amount_kg × 10)
      HR[10]: 温度 × 10
      HR[11]: 湿度 × 10
      HR[12]: 土壤湿度 × 10
      HR[13]: pH × 10
      HR[14]: 光照 × 100
    """

    STATUS_MAP = {"powered_off": 0, "standby": 1, "running": 2, "error": 3}
    STATUS_RMAP = {0: "powered_off", 1: "standby", 2: "running", 3: "error"}
    CMD_MAP = {1: "power_on", 2: "power_off", 3: "start", 4: "stop", 5: "reset"}

    def __init__(self, mgr: UnifiedDeviceManager, host: str = "127.0.0.1", port: int = 5020):
        self._mgr = mgr
        self._host = host; self._port = port
        self._running = False
        self._server_sock: Optional[socket.socket] = None
        # 每个从站持有自己管理器的引用
        self._slave_ids = set()
        for d in mgr.list_all():
            if d["protocol"] == PROTO_MODBUS:
                defn = DEVICE_DEFS.get(d["id"], {})
                sid = defn.get("modbus_slave", 0)
                if sid > 0:
                    self._slave_ids.add(sid)

    def _build_registers(self, slave_id: int) -> List[int]:
        """从 UnifiedDeviceManager 构建寄存器值"""
        regs = [0] * 30
        # 找到该从站对应的设备
        for d in self._mgr.list_all():
            if d["protocol"] != PROTO_MODBUS:
                continue
            defn = DEVICE_DEFS.get(d["id"], {})
            if defn.get("modbus_slave") != slave_id:
                continue
            state = d["state"]
            # 控制寄存器
            regs[0] = self.STATUS_MAP.get(state.get("status", "powered_off"), 0)
            regs[1] = 1 if state.get("power") else 0
            # 传感器寄存器
            if "temperature" in state:
                regs[10] = int(state["temperature"] * 10)
            if "humidity" in state:
                regs[11] = int(state["humidity"] * 10)
            if "soil_moisture" in state:
                regs[12] = int(state["soil_moisture"] * 10)
            if "ph" in state:
                regs[13] = int(state["ph"] * 10)
            if "light_lux" in state:
                regs[14] = int(state["light_lux"] / 100)
            break
        return regs

    def _handle_command(self, slave_id: int, cmd: str, params: dict):
        """处理从 Modbus 指令寄存器解析出的命令"""
        for d in self._mgr.list_all():
            if d["protocol"] != PROTO_MODBUS:
                continue
            defn = DEVICE_DEFS.get(d["id"], {})
            if defn.get("modbus_slave") == slave_id:
                dev = self._mgr.get(d["id"])
                name = dev["_name"] if dev else d["id"]
                log(name, f"收到指令: {cmd}", _fmt_params(params), PROTO_MODBUS)
                self._mgr.execute(d["id"], cmd, params, source_proto=PROTO_MODBUS)
                return

    def _handle_client(self, conn: socket.socket, addr):
        """处理单个 Modbus TCP 客户端连接"""
        def _mbap_response(tid: bytes, data_len: int) -> bytes:
            """构建 MBAP 头: TID + PID(0x0000) + Len"""
            return tid + b"\x00\x00" + struct.pack(">H", data_len)

        try:
            conn.settimeout(30)
            buf = b""
            while True:
                data = conn.recv(1024)
                if not data: break
                buf += data

                # Modbus TCP 帧: [TID(2)] [PID(2)] [Len(2)] [UID(1)] [FC(1)] [Data...]
                while len(buf) >= 8:
                    if len(buf) < 6: break
                    length = struct.unpack(">H", buf[4:6])[0]
                    total = 6 + length
                    if len(buf) < total: break
                    frame = buf[:total]; buf = buf[total:]

                    tid = frame[0:2]
                    slave_id = frame[6]
                    func = frame[7]

                    if slave_id not in self._slave_ids:
                        pdu = bytes([func | 0x80, 0x02])
                        conn.sendall(_mbap_response(tid, len(pdu)) + pdu)
                        continue

                    if func == MODBUS_FC_READ_HOLDING:
                        start_addr = struct.unpack(">H", frame[8:10])[0]
                        count = struct.unpack(">H", frame[10:12])[0]
                        regs = self._build_registers(slave_id)
                        byte_count = count * 2
                        resp_data = bytearray()
                        for i in range(count):
                            val = regs[start_addr + i] if start_addr + i < len(regs) else 0
                            resp_data.append((val >> 8) & 0xFF)
                            resp_data.append(val & 0xFF)
                        pdu = bytes([slave_id, func, byte_count]) + bytes(resp_data)
                        conn.sendall(_mbap_response(tid, len(pdu)) + pdu)

                    elif func == MODBUS_FC_WRITE_SINGLE:
                        addr = struct.unpack(">H", frame[8:10])[0]
                        value = struct.unpack(">H", frame[10:12])[0]
                        if addr == 2:
                            cmd = self.CMD_MAP.get(value, "")
                            if cmd:
                                self._handle_command(slave_id, cmd, {})
                        # 回显: UID + FC + Addr + Value
                        pdu = frame[6:12]
                        conn.sendall(_mbap_response(tid, len(pdu)) + pdu)

                    elif func == MODBUS_FC_WRITE_MULTIPLE:
                        start_addr = struct.unpack(">H", frame[8:10])[0]
                        reg_count = struct.unpack(">H", frame[10:12])[0]
                        # 解析指令寄存器
                        if start_addr <= 2 < start_addr + reg_count:
                            idx = 2 - start_addr
                            val = struct.unpack(">H", frame[13+idx*2:15+idx*2])[0]
                            cmd = self.CMD_MAP.get(val, "")
                            params = {}
                            if start_addr <= 3 < start_addr + reg_count:
                                i2 = 3 - start_addr
                                params["duration"] = struct.unpack(">H", frame[13+i2*2:15+i2*2])[0] // 60
                            if cmd:
                                self._handle_command(slave_id, cmd, params)
                        # 响应: UID + FC + StartAddr + Quantity
                        pdu = frame[6:8] + frame[8:12]
                        conn.sendall(_mbap_response(tid, len(pdu)) + pdu)

                    else:
                        pdu = bytes([slave_id, func | 0x80, 0x01])
                        conn.sendall(_mbap_response(tid, len(pdu)) + pdu)

        except (socket.timeout, ConnectionError, OSError):
            pass
        finally:
            try: conn.close()
            except Exception: pass

    def start(self):
        """在后台线程启动 Modbus TCP 服务器"""
        self._running = True
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self._host, self._port))
        self._server_sock.listen(10)
        self._server_sock.settimeout(1.0)

        def _loop():
            log_info(f"[Modbus] TCP 服务器启动 -> {self._host}:{self._port} ({len(self._slave_ids)}从站)", C.G)
            while self._running:
                try:
                    conn, addr = self._server_sock.accept()
                    log_info(f"[Modbus] 客户端连接: {addr}", C.DIM)
                    threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
                except socket.timeout:
                    continue
                except Exception:
                    break

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        time.sleep(0.2)
        return t

    def stop(self):
        self._running = False
        try: self._server_sock.close()
        except Exception: pass


# ═══════════════════════════════════════════════════
# Modbus 客户端（供终端 CLI 向 Modbus 设备发指令）
# ═══════════════════════════════════════════════════

class SimpleModbusClient:
    """简易 Modbus TCP 客户端 — 终端通过此客户端向 Modbus 设备发送指令。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5020):
        self._host = host; self._port = port

    def write_command(self, slave_id: int, command: str, params: dict = None):
        """写入指令寄存器"""
        cmd_map = {"power_on": 1, "power_off": 2, "start": 3, "stop": 4, "reset": 5}
        cmd_val = cmd_map.get(command, 0)
        if cmd_val == 0:
            log_info(f"[Modbus Client] 未知命令: {command}", C.R)
            return

        params = params or {}
        dur_val = int(params.get("duration", 0)) * 60  # 转秒
        amt_val = int(params.get("amount_kg", 0)) * 10

        # 写多个寄存器：HR[2]=cmd, HR[3]=duration
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self._host, self._port))

            # 功能码 0x10 (写多个寄存器)
            tid = b"\x00\x01"
            payload = struct.pack(">B", slave_id) + struct.pack(">B", 0x10)  # unit_id, func
            payload += struct.pack(">H", 2)  # 起始地址
            payload += struct.pack(">H", 2)  # 寄存器数量
            payload += struct.pack(">B", 4)  # 字节数
            payload += struct.pack(">H", cmd_val)
            payload += struct.pack(">H", dur_val)
            frame = tid + b"\x00\x00" + struct.pack(">H", len(payload)) + payload
            sock.sendall(frame)
            resp = sock.recv(1024)
            sock.close()
            if len(resp) >= 8:
                log_info(f"[Modbus Client] 指令已发送 -> 从站{slave_id}: {command}", C.DIM)
            else:
                log_info(f"[Modbus Client] 未收到响应", C.R)
        except Exception as e:
            log_info(f"[Modbus Client] 发送失败: {e}", C.R)


# ═══════════════════════════════════════════════════
# 传感器数据模拟
# ═══════════════════════════════════════════════════

class SensorSimulator:
    """定期更新传感器数据（温度/湿度/土壤湿度等的变化）"""

    def __init__(self, mgr: UnifiedDeviceManager):
        self._mgr = mgr
        self._running = False

    def start(self, interval: int = 5):
        self._running = True

        def loop():
            while self._running:
                time.sleep(interval)
                with self._mgr._lock:
                    for dev_id, dev in self._mgr._devices.items():
                        if dev.get("status") != "running":
                            continue
                        if "temperature" in dev:
                            dev["temperature"] = round(dev["temperature"] + random.uniform(-0.2, 0.2), 1)
                        if "humidity" in dev:
                            dev["humidity"] = round(max(0, min(100, dev["humidity"] + random.uniform(-0.5, 0.5))), 1)
                        if "soil_moisture" in dev:
                            if dev.get("_type") == "irrigate":
                                dev["soil_moisture"] = round(min(100, dev.get("soil_moisture", 50) + random.uniform(0.5, 1.5)), 1)
                        if "current_temp" in dev:
                            dev["current_temp"] = round(dev["current_temp"] + random.uniform(-0.1, 0.3), 1)
                        if "flow_rate" in dev and dev.get("_type") == "irrigate":
                            dev["flow_rate"] = round(random.uniform(2.0, 8.0), 1)

        threading.Thread(target=loop, daemon=True).start()
        return self

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════
# 🖥️  终端交互界面
# ═══════════════════════════════════════════════════

class TerminalUI:
    """终端 CLI — 作为协议客户端控制设备。

    HTTP 设备 → requests.post(:5000)
    MQTT 设备 → SimpleMqttClient.publish(:1883)
    Modbus 设备 → SimpleModbusClient.write(:5020)
    """

    def __init__(self, mgr: UnifiedDeviceManager,
                 mqtt_client: SimpleMqttClient = None,
                 modbus_client: SimpleModbusClient = None):
        self._mgr = mgr
        self._mqtt = mqtt_client
        self._modbus = modbus_client
        self._http_url = "http://127.0.0.1:5000"

    def banner(self):
        print(f"""
{C.BOLD}{C.G}
  ╔══════════════════════════════════════════════════════════════╗
  ║     🌾 硬件模拟器 v5.0 — 完整协议栈模拟                    ║
  ║     Farm Hardware Simulator (Full Protocol Stack)          ║
  ╚══════════════════════════════════════════════════════════════╝{C.W}

  {C.BOLD}协议:{C.W}  🌐 HTTP(:5000)  |  📡 MQTT(:1883)  |  🔧 Modbus TCP(:5020)

  {C.Y}HTTP 设备 (3台):{C.W}   灌溉泵 / 补光灯 / 施肥一体机
  {C.Y}MQTT 设备 (2台):{C.W}   通风扇 / 加热器
  {C.Y}Modbus 设备 (2台):{C.W} 温湿度传感器 / 摄像头
  {C.G}初始状态: 全部通电待机 ✅{C.W}
""")

    def help(self):
        print(f"""
  {C.BOLD}━━━ 命令列表 ━━━{C.W}

  {C.G}查看:{C.W}
    {C.C}list{C.W}            所有设备实时状态
    {C.C}watch <id>{C.W}      单个设备详情

  {C.G}控制（通过协议通道）:{C.W}
    {C.C}boot  <id>{C.W}      通电启动
    {C.C}start <id> [参数]{C.W} 开始工作
    {C.C}stop  <id>{C.W}      停止工作
    {C.C}shutdown <id>{C.W}   关机断电

  {C.G}参数示例:{C.W}
    {C.C}start pump dur=30{C.W}     灌溉30分钟后自动停
    {C.C}start fan speed=60{C.W}    通风扇60%转速
    {C.C}start light bri=80{C.W}    补光灯80%亮度
    {C.C}start heat temp=28{C.W}    加热器目标28°C

  {C.G}其他:{C.W}
    {C.C}capture{C.W}         摄像头拍照
    {C.C}help{C.W}            此帮助
    {C.C}quit{C.W}            退出

  {C.DIM}快捷ID: pump | fan | light | heat | sensor | fert | camera{C.W}
""")

    def show_all(self):
        print(f"\n  {C.BOLD}━━━ 设备状态总览 ━━━{C.W}")
        print(f"  {C.DIM}{'设备ID':30s} {'协议':6s} {'通电':4s} {'状态':8s} {'传感器读数'}{C.W}")
        print(f"  {C.DIM}{'─'*80}{C.W}")
        for d in self._mgr.list_all():
            s = d["state"]
            p = d["protocol"]
            pw = f"{C.G}是{C.W}" if s.get("power") else f"{C.DIM}否{C.W}"
            st = status_icon(s) + " " + POWER_STATE_LABELS.get(s.get("status"), "?")
            readings = self._fmt_r(s)
            proto_label = f"{PROTO_ICONS.get(p, '?')} {p}"
            print(f"  {C.BOLD}{d['id']:30s}{C.W} {proto_label:8s} {pw:8s} {st:14s} {readings}")
        print()

    def show_one(self, device_id: str):
        dev = self._mgr.get(device_id)
        if not dev:
            print(f"  {C.R}设备 '{device_id}' 不存在{C.W}")
            return
        print(f"\n  {C.BOLD}━━━ {dev['_name']} ━━━{C.W}")
        print(f"  ID:     {device_id}")
        print(f"  协议:   {dev['_protocol']}")
        print(f"  状态:   {status_icon(dev)} {POWER_STATE_LABELS.get(dev.get('status'), '?')}")
        print(f"  通电:   {'✅ 是' if dev.get('power') else '⭕ 否'}")
        for k, v in dev.items():
            if not k.startswith("_") and k not in ("power", "status") and isinstance(v, (int, float, str)):
                print(f"  {k}: {v}")
        print()

    def _fmt_r(self, s: dict) -> str:
        p = []
        if "temperature" in s: p.append(f"T={s['temperature']}C")
        if "humidity" in s: p.append(f"H={s['humidity']}%")
        if "soil_moisture" in s: p.append(f"SM={s['soil_moisture']}%")
        if "flow_rate" in s and s["flow_rate"] > 0: p.append(f"F={s['flow_rate']}")
        if "rpm" in s and s["rpm"] > 0: p.append(f"RPM={s['rpm']}")
        if "brightness_percent" in s and s["brightness_percent"] > 0: p.append(f"Bri={s['brightness_percent']}%")
        if "target_temp" in s: p.append(f"Target={s['target_temp']}C")
        return "  ".join(p) if p else "-"

    def _resolve(self, name: str) -> str:
        return SHORTCUTS.get(name.lower(), name)

    def dispatch(self, line: str) -> bool:
        """解析并分发命令到对应协议通道。返回 False 表示退出。"""
        line = line.strip()
        if not line: return True

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "q", "exit"): return False
        if cmd in ("help", "h"): self.help(); return True
        if cmd in ("list", "l", "status", "s"): self.show_all(); return True
        if cmd == "watch" and len(parts) >= 2: self.show_one(self._resolve(parts[1])); return True
        if cmd in ("capture", "pic"):
            self._send_command("greenhouse_camera_01", "capture")
            return True

        if len(parts) < 2:
            print(f"  {C.Y}用法: {cmd} <设备ID> [参数]{C.W}")
            return True

        target = self._resolve(parts[1])
        params = self._parse(parts[2:])

        if cmd in ("boot", "power_on"):
            self._send_command(target, "power_on")
        elif cmd in ("shutdown", "power_off"):
            self._send_command(target, "power_off")
        elif cmd == "start":
            self._send_command(target, "start", params)
        elif cmd == "stop":
            self._send_command(target, "stop")
        elif cmd == "reset":
            self._send_command(target, "reset")
        elif cmd == "set" and len(parts) >= 3:
            self._send_command(target, "set_param", params)
        else:
            print(f"  {C.R}未知命令: {cmd}（输入 help 查看帮助）{C.W}")
        return True

    def _send_command(self, device_id: str, command: str, params: dict = None):
        """根据设备协议分发命令"""
        dev = self._mgr.get(device_id)
        if not dev:
            print(f"  {C.R}设备 '{device_id}' 不存在{C.W}")
            return

        proto = dev["_protocol"]
        name = dev["_name"]
        params = params or {}

        if proto == PROTO_HTTP:
            self._via_http(device_id, command, params, name)
        elif proto == PROTO_MQTT:
            self._via_mqtt(device_id, command, params, name)
        elif proto == PROTO_MODBUS:
            self._via_modbus(device_id, command, params, name)
        else:
            print(f"  {C.R}未知协议: {proto}{C.W}")

    def _via_http(self, device_id: str, command: str, params: dict, name: str):
        """通过 HTTP 协议发送指令"""
        import requests
        log(name, f"发送指令: {command}", _fmt_params(params), f"{PROTO_HTTP} [终端]")
        try:
            r = requests.post(
                f"{self._http_url}/api/command",
                json={"device_id": device_id, "command": command, "params": params},
                headers={"X-Source": "terminal"},
                timeout=5,
            )
            result = r.json()
            self._ok(result)
        except Exception as e:
            # HTTP 服务器不可用时回退到直接执行
            log_info(f"[HTTP] 服务器不可达，回退直接执行", C.Y)
            result = self._mgr.execute(device_id, command, params, source_proto=f"{PROTO_HTTP} [终端]")
            self._ok(result)

    def _via_mqtt(self, device_id: str, command: str, params: dict, name: str):
        """通过 MQTT 协议发送指令"""
        if not self._mqtt:
            log_info("[MQTT] 客户端未连接，回退到内部执行", C.Y)
            self._mgr.execute(device_id, command, params, source_proto=f"{PROTO_MQTT} [终端]")
            return

        defn = DEVICE_DEFS.get(device_id, {})
        base_topic = defn.get("mqtt_topic", f"devices/{device_id}")
        ctrl_topic = f"{base_topic}/control"
        log(name, f"发送指令: {command}", _fmt_params(params), f"{PROTO_MQTT} [终端]")
        try:
            self._mqtt.publish(ctrl_topic, {
                "command": command,
                "params": params,
                "timestamp": datetime.now().isoformat(),
            })
            # MQTT 是异步的，设备处理器会在后台处理
            # 短暂等待后直接通过管理器执行（确保即时反馈）
            time.sleep(0.1)
            self._mgr.execute(device_id, command, params, source_proto=f"{PROTO_MQTT} [终端]")
        except Exception as e:
            log_info(f"[MQTT] 发送失败: {e}", C.R)

    def _via_modbus(self, device_id: str, command: str, params: dict, name: str):
        """通过 Modbus 协议发送指令"""
        defn = DEVICE_DEFS.get(device_id, {})
        slave_id = defn.get("modbus_slave", 1)
        log(name, f"发送指令: {command}", _fmt_params(params), f"{PROTO_MODBUS} [终端]")

        if self._modbus:
            try:
                self._modbus.write_command(slave_id, command, params)
            except Exception as e:
                log_info(f"[Modbus] 发送失败: {e}", C.R)

        # Modbus 服务器和终端 CLI 共享 manager，直接执行确保即时响应
        self._mgr.execute(device_id, command, params, source_proto=f"{PROTO_MODBUS} [终端]")

    def _parse(self, args: List[str]) -> dict:
        """解析 key=value 参数"""
        params = {}
        aliases = {"dur": "duration", "speed": "speed_percent", "bri": "brightness_percent",
                   "temp": "target_temp", "amt": "amount_kg", "flow": "flow_rate"}
        for a in args:
            if "=" in a:
                k, v = a.split("=", 1)
                k = aliases.get(k, k)
                try:
                    params[k] = float(v) if ("." in v or v.isdigit()) else v
                except ValueError:
                    params[k] = v
        return params

    def _ok(self, r: dict):
        if r.get("success"):
            print(f"  {C.G}[OK] {r.get('message', '')}{C.W}")
        else:
            print(f"  {C.R}[FAIL] {r.get('message', '')}{C.W}")


# ═══════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════

def _setup_console():
    if sys.platform == "win32":
        try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception: pass


def main():
    _setup_console()

    # ── 1. 统一设备管理器（默认通电待机）──
    mgr = UnifiedDeviceManager()
    mgr.init_all()

    # ── 2. HTTP 服务器 ──
    http_ok = create_http_server(mgr)

    # ── 3. MQTT Broker ──
    mqtt_broker = MqttBroker("127.0.0.1", 1883)
    mqtt_broker.start()
    time.sleep(0.5)  # 等待 broker 绑定端口
    mqtt_broker_ok = mqtt_broker._started

    # ── 4. MQTT 设备处理器 ──
    mqtt_handler = MqttDeviceHandler(mgr)
    mqtt_handler_ok = mqtt_handler.connect() if mqtt_broker_ok else False

    # ── 5. MQTT 客户端（供终端使用）──
    mqtt_client = SimpleMqttClient()
    mqtt_client_ok = mqtt_client.connect() if mqtt_broker_ok else False

    # ── 6. Modbus TCP 服务器 ──
    modbus_server = ModbusTcpServer(mgr)
    modbus_server.start()
    modbus_ok = True  # 简化：默认启动成功

    # ── 7. Modbus 客户端（供终端使用）──
    modbus_client = SimpleModbusClient()

    # ── 8. 传感器数据模拟 ──
    sensor = SensorSimulator(mgr).start(interval=5)

    # ── 9. 终端 UI ──
    ui = TerminalUI(mgr, mqtt_client if mqtt_client_ok else None, modbus_client)
    ui.banner()

    # 统计信息
    proto_counts = {}
    for d in mgr.list_all():
        p = d["protocol"]
        proto_counts[p] = proto_counts.get(p, 0) + 1
    proto_str = " | ".join(f"{PROTO_ICONS.get(p, '')} {p}: {c}台" for p, c in proto_counts.items())

    print(f"  {C.BOLD}{'═'*60}{C.W}")
    print(f"  {C.G}  {proto_str}{C.W}")
    print(f"  {C.G}  HTTP :5000 {'✅' if http_ok else '❌'}  |  MQTT :1883 {'✅' if mqtt_broker_ok else '❌'}  |  Modbus :5020 {'✅' if modbus_ok else '❌'}{C.W}")
    print(f"  {C.G}  全部设备默认通电待机，等待指令...{C.W}")
    print(f"  {C.BOLD}{'═'*60}{C.W}")
    print(f"  {C.DIM}输入 {C.C}help{C.DIM} 查看命令 | {C.C}list{C.DIM} 查看状态{C.W}\n")

    # ── 交互循环 ──
    try:
        while True:
            try:
                line = input(f"  {C.C}▸ {C.W}").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not ui.dispatch(line):
                break
    except KeyboardInterrupt:
        pass

    # ── 清理 ──
    print(f"\n  {C.Y}正在停止所有服务...{C.W}")
    sensor.stop()
    try: mqtt_client.close()
    except Exception: pass
    modbus_server.stop()
    print(f"  {C.G}硬件模拟器已安全退出 👋{C.W}\n")


if __name__ == "__main__":
    main()
