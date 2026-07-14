"""
HTTP 智能设备服务端

启动一个简单的 HTTP 服务，模拟智能插座/控制器。
提供 /api/state（查询状态）和 /api/command（执行指令）两个端点。

设备生命周期状态机:
  关机(powered_off) ──[power_on]──▶ 待机(standby) ──[start]──▶ 工作中(running)
       ▲                                  ▲                       │
       │                                  │ [stop]                │
       │                                  ◀───────────────────────┘
       │
       └────────────[power_off]───────────┘

依赖: pip install flask
运行: python hardware_examples/http_device_server.py
"""
import json
import time
import sys
import threading

try:
    from flask import Flask, request, jsonify
except ImportError:
    print("[ERR] 请先安装 flask: pip install flask")
    exit(1)

app = Flask(__name__)

# ── 设备状态（初始: 关机）──────────────────────────
device_state = {
    "power": False,
    "status": "powered_off",   # powered_off | standby | running | error
    "runtime_seconds": 0,
    "temperature": 25.0,
    "humidity": 60.0,
}

_lock = threading.Lock()

# ── API 端点 ──────────────────────────────────────


@app.route("/api/state", methods=["GET"])
def get_state():
    """查询设备当前状态"""
    with _lock:
        return jsonify(dict(device_state))


@app.route("/api/command", methods=["POST"])
def execute():
    """接收并执行指令，遵循状态机规则"""
    data = request.get_json(silent=True) or {}
    command = data.get("command", "")
    params = data.get("params", {})

    with _lock:
        current = device_state["status"]

        # ── 通电启动 ──
        if command in ("power_on", "boot"):
            if current == "powered_off":
                device_state["power"] = True
                device_state["status"] = "standby"
                print("[BOOT] 设备通电启动，进入待机状态")
            elif current == "standby":
                print("[BOOT] 设备已在待机状态")
            elif current == "running":
                print("[BOOT] 设备正在工作中")
            return jsonify({"success": True, "message": "设备已通电"})

        # ── 关机断电 ──
        elif command in ("power_off", "shutdown"):
            if current in ("standby", "running"):
                device_state["power"] = False
                device_state["status"] = "powered_off"
                print("[SHUTDOWN] 设备关机断电")
            elif current == "powered_off":
                print("[SHUTDOWN] 设备已在关机状态")
            elif current == "error":
                device_state["power"] = False
                device_state["status"] = "powered_off"
                print("[SHUTDOWN] 故障状态下强制关机")
            return jsonify({"success": True, "message": "设备已关机"})

        # ── 开始工作 ──
        elif command == "start":
            if current == "powered_off":
                device_state["power"] = True
                device_state["status"] = "running"
                print(f"[START] 通电并启动 (时长={params.get('duration', 0)}s)")
            elif current == "standby":
                device_state["status"] = "running"
                duration = params.get("duration", 0)
                print(f"[START] 设备开始工作 (时长={duration}s)")
            elif current == "running":
                print("[START] 设备已在工作中")
                # 更新参数
                if params.get("duration"):
                    print(f"[START] 更新工作时长={params['duration']}s")
            elif current == "error":
                return jsonify({"success": False, "message": "设备故障，请先复位(reset)"}), 400

            # 模拟定时停止（回到待机，不断电）
            duration = params.get("duration", 0)
            if duration > 0:
                def auto_stop():
                    time.sleep(duration)
                    with _lock:
                        if device_state["status"] == "running":
                            device_state["status"] = "standby"
                            # power 保持 True
                            print(f"[AUTO] 定时结束，回到待机 (运行了 {duration}s)")
                threading.Thread(target=auto_stop, daemon=True).start()

            return jsonify({"success": True, "message": "设备已开始工作"})

        # ── 停止工作 ──
        elif command == "stop":
            if current == "running":
                device_state["status"] = "standby"
                # 关键：power 保持 True，不断电！
                print("[STOP] 设备停止工作，回到待机状态（保持通电）")
            elif current == "standby":
                print("[STOP] 设备当前未在工作（待机中）")
            elif current == "powered_off":
                print("[STOP] 设备处于关机状态")
            return jsonify({"success": True, "message": "设备已停止工作"})

        # ── 故障复位 ──
        elif command == "reset":
            if current == "error":
                device_state["power"] = True
                device_state["status"] = "standby"
                print("[RESET] 故障复位，恢复到待机状态")
            else:
                print("[RESET] 设备未处于故障状态")
            return jsonify({"success": True, "message": "设备已复位"})

        # ── 参数设置 ──
        elif command == "set_param":
            for k, v in params.items():
                if k in device_state and k not in ("power", "status"):
                    device_state[k] = v
            print(f"[CFG] 参数已设置: {params}")
            return jsonify({"success": True, "message": f"参数已更新: {params}"})

        else:
            return jsonify({
                "success": False,
                "message": f"不支持的指令: {command}"
            }), 400


@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({"status": "ok", "device": "http_device_01"})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"\n{'='*50}")
    print(f"HTTP 智能设备服务端")
    print(f"   地址: http://localhost:{port}")
    print(f"   状态: GET  /api/state")
    print(f"   控制: POST /api/command")
    print(f"   健康: GET  /health")
    print(f"{'='*50}")
    print(f"   生命周期: boot → start → stop → shutdown")
    print(f"   初始状态: 关机(powered_off)")
    print(f"{'='*50}\n")
    print(f"[...] 等待指令... 按 Ctrl+C 停止\n")

    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
