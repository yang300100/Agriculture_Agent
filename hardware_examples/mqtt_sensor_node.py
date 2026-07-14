"""
MQTT 传感器模拟节点

模拟一个土壤传感器，定期向 MQTT broker 上报温湿度、土壤湿度数据。
同时订阅控制指令 topic，模拟接收灌溉命令。

设备生命周期:
  关机(powered_off) ──[power_on]──▶ 待机(standby) ──[start]──▶ 工作中(running)
                                        ▲                       │
                                        │ [stop]                │
                                        ◀───────────────────────┘

依赖: pip install paho-mqtt
运行: python hardware_examples/mqtt_sensor_node.py
"""

import json
import random
import time
import threading

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("[ERR] 请先安装 paho-mqtt: pip install paho-mqtt")
    exit(1)

# ── 配置 ──────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883
DEVICE_ID = "mqtt_soil_sensor_01"
STATE_TOPIC = f"devices/{DEVICE_ID}/state"
CONTROL_TOPIC = f"devices/{DEVICE_ID}/control"

# 设备当前状态（初始：关机）
device_state = {
    "temperature": 25.0,
    "humidity": 65.0,
    "soil_moisture": 45.0,
    "ph": 6.8,
    "power": False,
    "status": "powered_off",   # powered_off | standby | running | error
    "_driver": "mqtt",
    "_read_at": "",
}

_state_lock = threading.Lock()


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[OK] 已连接到 MQTT Broker: {BROKER_HOST}:{BROKER_PORT}")
        client.subscribe(CONTROL_TOPIC)
        print(f"[SUB] 已订阅控制主题: {CONTROL_TOPIC}")
        # 连接后自动通电进入待机
        with _state_lock:
            device_state["power"] = True
            device_state["status"] = "standby"
        print(f"[BOOT] 设备通电启动，进入待机状态")
    else:
        print(f"[ERR] 连接失败, rc={rc}")


def on_message(client, userdata, msg):
    """接收控制指令，遵循状态机规则"""
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        command = payload.get("command", "")
        params = payload.get("params", {})

        with _state_lock:
            current = device_state["status"]

            # ── 通电启动 ──
            if command in ("power_on", "boot"):
                if current == "powered_off":
                    device_state["power"] = True
                    device_state["status"] = "standby"
                    print(f"[BOOT] 设备通电启动，进入待机")

            # ── 关机断电 ──
            elif command in ("power_off", "shutdown"):
                if current in ("standby", "running"):
                    device_state["power"] = False
                    device_state["status"] = "powered_off"
                    print(f"[SHUTDOWN] 设备关机断电")
                elif current == "powered_off":
                    print(f"[SHUTDOWN] 设备已在关机状态")

            # ── 开始工作 ──
            elif command == "start":
                if current == "powered_off":
                    device_state["power"] = True
                    device_state["status"] = "running"
                    print(f"[START] 通电并启动 | 参数: {params}")
                elif current == "standby":
                    device_state["status"] = "running"
                    print(f"[START] 开始工作 | 参数: {params}")
                elif current == "running":
                    print(f"[START] 已在工作中 | 参数: {params}")

                # 模拟运行指定时长后自动停止（回到待机，不断电）
                duration = params.get("duration", 30)
                if duration > 0:
                    def auto_stop():
                        time.sleep(duration)
                        with _state_lock:
                            if device_state["status"] == "running":
                                device_state["status"] = "standby"
                                # power 保持 True
                                print(f"[AUTO] 定时结束，回到待机 (运行了 {duration}s)")
                    threading.Thread(target=auto_stop, daemon=True).start()

            # ── 停止工作 ──
            elif command == "stop":
                if current == "running":
                    device_state["status"] = "standby"
                    # power 保持 True，不断电！
                    print(f"[STOP] 停止工作，回到待机状态（保持通电）")
                elif current in ("standby", "powered_off"):
                    print(f"[STOP] 当前未在工作")

            # ── 参数设置 ──
            elif command == "set_param":
                for k, v in params.items():
                    if k in device_state and k not in ("power", "status"):
                        device_state[k] = v
                print(f"[CFG] 参数设置: {params}")

            else:
                print(f"[WARN] 未知指令: {command}")

    except json.JSONDecodeError:
        print(f"[WARN] 收到非 JSON 消息: {msg.payload.decode('utf-8', errors='replace')[:80]}")


def simulate_sensor_drift():
    """模拟传感器数据自然波动"""
    with _state_lock:
        device_state["temperature"] += random.uniform(-0.3, 0.3)
        device_state["temperature"] = round(max(-10, min(50, device_state["temperature"])), 1)

        device_state["humidity"] += random.uniform(-1.0, 1.0)
        device_state["humidity"] = round(max(0, min(100, device_state["humidity"])), 1)

        # 土壤湿度：工作中（灌溉）时上升，待机/关机时缓慢下降
        if device_state["status"] == "running":
            device_state["soil_moisture"] += random.uniform(0.5, 1.5)
        else:
            device_state["soil_moisture"] -= random.uniform(0.1, 0.3)
        device_state["soil_moisture"] = round(max(0, min(100, device_state["soil_moisture"])), 1)

        device_state["ph"] += random.uniform(-0.05, 0.05)
        device_state["ph"] = round(max(3.5, min(9.5, device_state["ph"])), 1)

        device_state["_read_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")


def main():
    print(f"\n{'='*50}")
    print(f"MQTT 土壤传感器模拟节点")
    print(f"   设备 ID: {DEVICE_ID}")
    print(f"   状态主题: {STATE_TOPIC}")
    print(f"   控制主题: {CONTROL_TOPIC}")
    print(f"   生命周期: boot -> start -> stop -> shutdown")
    print(f"   初始状态: 关机(powered_off)")
    print(f"{'='*50}\n")

    client = mqtt.Client(client_id=DEVICE_ID)
    client.on_connect = on_connect
    client.on_message = on_message

    # 设置遗嘱消息（断开时通知关机）
    client.will_set(STATE_TOPIC, json.dumps({
        "power": False, "status": "powered_off", "_driver": "mqtt"
    }), qos=1, retain=True)

    try:
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    except ConnectionRefusedError:
        print(f"[ERR] 无法连接到 MQTT Broker {BROKER_HOST}:{BROKER_PORT}")
        print("   请先启动 Mosquitto: mosquitto -v")
        print("   或安装: sudo apt install mosquitto (Linux)")
        exit(1)

    client.loop_start()

    # 上报初始状态
    client.publish(STATE_TOPIC, json.dumps(device_state, ensure_ascii=False), qos=1)

    print("\n[SENSOR] 开始上报传感器数据 (每5秒)...")
    print("   按 Ctrl+C 停止\n")

    try:
        while True:
            simulate_sensor_drift()
            payload = json.dumps(device_state, ensure_ascii=False)
            client.publish(STATE_TOPIC, payload, qos=1)

            status_labels = {"powered_off": "关机", "standby": "待机", "running": "工作中", "error": "故障"}
            label = status_labels.get(device_state["status"], device_state["status"])
            print(f"[PUB] 上报: temp={device_state['temperature']}°C "
                  f"hum={device_state['humidity']}% "
                  f"soil={device_state['soil_moisture']}% "
                  f"状态={label}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n[STOP] 停止上报")
        with _state_lock:
            device_state["power"] = False
            device_state["status"] = "powered_off"
        client.publish(STATE_TOPIC, json.dumps(device_state, ensure_ascii=False), qos=1)
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
