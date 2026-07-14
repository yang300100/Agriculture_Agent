"""
MQTT 继电器控制器

模拟一个可通过 MQTT 控制的继电器/水泵设备。
订阅控制 topic，收到指令后执行并反馈状态。

设备生命周期:
  关机(powered_off) ──[power_on]──▶ 待机(standby) ──[start]──▶ 工作中(running)
       ▲                                ▲                       │
       │                                │ [stop]                │
       │                                ◀───────────────────────┘
       │
       └──────────[power_off]───────────┘

依赖: pip install paho-mqtt
运行: python hardware_examples/mqtt_relay_controller.py
"""

import json
import time

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("[ERR] 请先安装 paho-mqtt: pip install paho-mqtt")
    exit(1)

# ── 配置 ──────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883
DEVICE_ID = "mqtt_relay_01"
STATE_TOPIC = f"devices/{DEVICE_ID}/state"
CONTROL_TOPIC = f"devices/{DEVICE_ID}/control"

# 设备状态（初始：关机）
device = {
    "state": {
        "power": False,
        "status": "powered_off",   # powered_off | standby | running | error
        "runtime_seconds": 0,
        "_driver": "mqtt",
        "_read_at": "",
    }
}


def publish_state(client):
    device["state"]["_read_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    client.publish(STATE_TOPIC, json.dumps(device["state"], ensure_ascii=False), qos=1)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[OK] 继电器控制器已连接 MQTT Broker")
        client.subscribe(CONTROL_TOPIC)
        # 连接后自动通电进入待机
        device["state"]["power"] = True
        device["state"]["status"] = "standby"
        print(f"[BOOT] 设备通电启动，进入待机状态")
        print(f"[SUB] 等待控制指令: {CONTROL_TOPIC}")
    else:
        print(f"[ERR] 连接失败, rc={rc}")


def on_message(client, userdata, msg):
    """接收控制指令，遵循状态机规则"""
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        command = payload.get("command", "")
        params = payload.get("params", {})
        current = device["state"]["status"]

        # ── 通电启动 ──
        if command in ("power_on", "boot"):
            if current == "powered_off":
                device["state"]["power"] = True
                device["state"]["status"] = "standby"
                print(f"[BOOT] 设备通电启动，进入待机")
            elif current == "standby":
                print(f"[BOOT] 设备已在待机状态")

        # ── 关机断电 ──
        elif command in ("power_off", "shutdown"):
            if current in ("standby", "running"):
                device["state"]["power"] = False
                device["state"]["status"] = "powered_off"
                print(f"[SHUTDOWN] 设备关机断电")
            elif current == "powered_off":
                print(f"[SHUTDOWN] 设备已在关机状态")

        # ── 开始工作（继电器闭合）──
        elif command == "start":
            if current == "powered_off":
                device["state"]["power"] = True
                device["state"]["status"] = "running"
                print(f"[START] 通电并闭合继电器 (运行 {params.get('duration', 0)}s)")
            elif current == "standby":
                device["state"]["status"] = "running"
                duration = params.get("duration", 0)
                print(f"[START] 继电器闭合 (运行 {duration}s)")
            elif current == "running":
                print(f"[START] 继电器已闭合，更新参数: {params}")

        # ── 停止工作（继电器断开）──
        elif command == "stop":
            if current == "running":
                device["state"]["status"] = "standby"
                # power 保持 True，不断电！
                print(f"[STOP] 继电器断开，回到待机（保持通电）")
            elif current in ("standby", "powered_off"):
                print(f"[STOP] 继电器未闭合")

        # ── 参数设置 ──
        elif command == "set_param":
            for k, v in params.items():
                if k in device["state"] and k not in ("power", "status"):
                    device["state"][k] = v
            print(f"[CFG] 参数设置: {params}")

        else:
            print(f"[WARN] 未知指令: {command}")
            return

        publish_state(client)

    except json.JSONDecodeError:
        print(f"[WARN] 非 JSON 消息: {msg.payload.decode('utf-8', errors='replace')[:80]}")


def main():
    print(f"\n{'='*50}")
    print(f"MQTT 继电器控制器")
    print(f"   设备 ID: {DEVICE_ID}")
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
        print(f"[ERR] 无法连接 MQTT Broker {BROKER_HOST}:{BROKER_PORT}")
        print("   请先启动 Mosquitto: mosquitto -v")
        exit(1)

    publish_state(client)
    client.loop_start()

    print("\n[READY] 等待控制指令... 按 Ctrl+C 停止\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] 断开连接")
        device["state"]["power"] = False
        device["state"]["status"] = "powered_off"
        publish_state(client)
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
