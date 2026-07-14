"""
统一硬件模拟器启动器 — 同时启动 HTTP / MQTT / Modbus 三协议模拟器

协议分配:
  HTTP   (port 5000): irrigation_pump_01  灌溉泵      — 简单开关量
          (port 5001): grow_light_01       补光灯      — 调光控制
          (port 5002): fertilizer_pump_01  施肥泵      — 流量控制
  MQTT   (需要 Broker): ventilation_fan_01 通风扇      — 变频调速
                      : heater_01          加热器      — 温控PID
  Modbus (port 5020): env_sensor_01        温湿度传感器  — 工业传感器
                      slave#2: greenhouse_camera_01  监控摄像头

HTTP 设备各自独立端口，真实模拟分布式部署场景。

启动: python hardware_examples/start_all_simulators.py [--no-mqtt] [--no-modbus]
"""

import subprocess
import sys
import os
import time
import signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 设备配置 ─────────────────────────────

HTTP_DEVICES = [
    {"port": 5000, "device_id": "irrigation_pump_01", "name": "灌溉泵"},
    {"port": 5001, "device_id": "grow_light_01", "name": "补光灯"},
    {"port": 5002, "device_id": "fertilizer_pump_01", "name": "施肥泵"},
]

MQTT_DEVICES = [
    {"device_id": "ventilation_fan_01", "name": "通风扇"},
    {"device_id": "heater_01", "name": "加热器"},
]

MODBUS_DEVICES = [
    {"slave": 1, "device_id": "env_sensor_01", "name": "温湿度传感器"},
    {"slave": 2, "device_id": "greenhouse_camera_01", "name": "监控摄像头"},
]

procs = []


def start_http(port, device_id):
    """启动单设备 HTTP 模拟器"""
    script = os.path.join(SCRIPT_DIR, "http_simulator_server.py")
    # 用环境变量限制只启动指定设备
    env = os.environ.copy()
    env["SIM_DEVICE_FILTER"] = device_id
    proc = subprocess.Popen(
        [sys.executable, script, "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env,
    )
    procs.append(proc)
    return proc


def start_mqtt():
    """启动 MQTT 模拟器（多设备）"""
    script = os.path.join(SCRIPT_DIR, "mqtt_simulator_server.py")
    proc = subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(proc)
    return proc


def start_modbus():
    """启动 Modbus 模拟器"""
    script = os.path.join(SCRIPT_DIR, "modbus_simulator_server.py")
    proc = subprocess.Popen(
        [sys.executable, script, "--port", "5020"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(proc)
    return proc


def cleanup():
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass


def main():
    import argparse
    p = argparse.ArgumentParser(description="统一硬件模拟器启动器")
    p.add_argument("--no-http", action="store_true")
    p.add_argument("--no-mqtt", action="store_true")
    p.add_argument("--no-modbus", action="store_true")
    args = p.parse_args()

    signal.signal(signal.SIGINT, lambda *_: cleanup())
    signal.signal(signal.SIGTERM, lambda *_: cleanup())

    print("=" * 55)
    print("  农业 IoT 硬件模拟器 — 三协议统一启动")
    print("=" * 55)

    # ── HTTP ──
    if not args.no_http:
        print("\n[HTTP] 启动模拟设备...")
        for dev in HTTP_DEVICES:
            proc = start_http(dev["port"], dev["device_id"])
            print(f"  {dev['device_id']} ({dev['name']}): http://127.0.0.1:{dev['port']}")
            time.sleep(0.5)
    else:
        print("\n[HTTP] 已跳过")

    # ── MQTT ──
    if not args.no_mqtt:
        print("\n[MQTT] 启动模拟设备...")
        try:
            proc = start_mqtt()
            time.sleep(1)
            for dev in MQTT_DEVICES:
                print(f"  {dev['device_id']} ({dev['name']}): mqtt://localhost:1883")
        except Exception as e:
            print(f"  MQTT 启动失败: {e}")
    else:
        print("\n[MQTT] 已跳过")

    # ── Modbus ──
    if not args.no_modbus:
        print("\n[Modbus] 启动模拟设备...")
        try:
            proc = start_modbus()
            time.sleep(1)
            for dev in MODBUS_DEVICES:
                print(f"  {dev['device_id']} ({dev['name']}): modbus://127.0.0.1:5020 (slave#{dev['slave']})")
        except Exception as e:
            print(f"  Modbus 启动失败: {e}")
    else:
        print("\n[Modbus] 已跳过")

    print("\n" + "=" * 55)
    print("  全部就绪！按 Ctrl+C 停止所有模拟器")
    print("=" * 55)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
        print("\n所有模拟器已停止。")


if __name__ == "__main__":
    main()
