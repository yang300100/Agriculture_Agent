"""
统一硬件模拟器启动器

同时启动:
  1. MQTT Broker（内嵌，无需 mosquitto）
  2. HTTP 模拟器（port 5000）
  3. MQTT 模拟器
  4. Modbus TCP 模拟器（port 5020）

启动: python hardware_examples/start_all_simulators.py [--no-http] [--no-mqtt] [--no-modbus]
"""

import subprocess
import sys
import os
import time
import signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
procs = []


def start_process(name, args):
    proc = subprocess.Popen(
        [sys.executable] + args,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(proc)
    print(f"  [{name}] PID={proc.pid}")
    return proc


def cleanup(*_):
    print("\n正在停止所有模拟器...")
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    print("已停止")


def main():
    import argparse
    p = argparse.ArgumentParser(description="统一硬件模拟器启动器")
    p.add_argument("--no-http", action="store_true")
    p.add_argument("--no-mqtt", action="store_true")
    p.add_argument("--no-modbus", action="store_true")
    args = p.parse_args()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("=" * 50)
    print("  农业 IoT 硬件模拟器 — 全协议启动")
    print("=" * 50)

    if not args.no_mqtt:
        print("\n[MQTT Broker]")
        start_process("MQTT Broker", [os.path.join(SCRIPT_DIR, "mqtt_broker.py")])
        time.sleep(0.5)
        print("\n[MQTT 模拟器]")
        start_process("MQTT 模拟器", [os.path.join(SCRIPT_DIR, "mqtt_simulator_server.py")])
    else:
        print("\n[MQTT] 已跳过")

    if not args.no_http:
        print("\n[HTTP 模拟器]")
        start_process("HTTP 模拟器", [os.path.join(SCRIPT_DIR, "http_simulator_server.py")])
    else:
        print("\n[HTTP] 已跳过")

    if not args.no_modbus:
        print("\n[Modbus 模拟器]")
        start_process("Modbus 模拟器", [os.path.join(SCRIPT_DIR, "modbus_simulator_server.py")])
    else:
        print("\n[Modbus] 已跳过")

    print("\n" + "=" * 50)
    print("  全部就绪！按 Ctrl+C 停止")
    print("  浏览器控制面板: http://127.0.0.1:5000/")
    print("=" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
