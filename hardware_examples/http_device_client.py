"""
HTTP 设备客户端

调用 HTTP 设备服务端的 API，演示完整的"查询→控制→验证"流程。
可配合 http_device_server.py 一起运行测试。

依赖: pip install requests
运行: python hardware_examples/http_device_client.py
"""

import json
import time
import sys

try:
    import requests
except ImportError:
    print("❌ 请先安装 requests: pip install requests")
    exit(1)

# ── 配置 ──────────────────────────────────────────
DEVICE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"


def get_state():
    """查询设备状态"""
    resp = requests.get(f"{DEVICE_URL}/api/state", timeout=5)
    resp.raise_for_status()
    return resp.json()


def send_command(command, params=None):
    """向设备发送指令"""
    data = {"command": command, "params": params or {}}
    resp = requests.post(f"{DEVICE_URL}/api/command", json=data, timeout=5)
    return resp.json()


def health_check():
    """健康检查"""
    try:
        resp = requests.get(f"{DEVICE_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def main():
    print(f"\n{'='*50}")
    print(f"🌐 HTTP 设备客户端测试")
    print(f"   目标设备: {DEVICE_URL}")
    print(f"{'='*50}\n")

    # 1. 健康检查
    if not health_check():
        print(f"❌ 设备不可达: {DEVICE_URL}")
        print(f"   请先启动 HTTP 设备服务端:")
        print(f"   python hardware_examples/http_device_server.py")
        exit(1)
    print("✅ 设备在线")

    # 2. 查询初始状态
    state = get_state()
    status_labels = {"powered_off": "关机", "standby": "待机", "running": "工作中", "error": "故障"}
    label = status_labels.get(state['status'], state['status'])
    print(f"\n[SENSOR] 初始状态: 通电={'是' if state['power'] else '否'}, 状态={label}")

    # 3. 通电启动
    print("\n[CMD] 通电启动 (power_on)...")
    result = send_command("power_on")
    print(f"   响应: {result}")
    state = get_state()
    label = status_labels.get(state['status'], state['status'])
    print(f"[SENSOR] 通电后: 通电={'是' if state['power'] else '否'}, 状态={label}")

    # 4. 启动设备
    print("\n[CMD] 开始工作 (start, duration=10s)...")
    result = send_command("start", {"duration": 10})
    print(f"   响应: {result}")

    time.sleep(1)
    state = get_state()
    label = status_labels.get(state['status'], state['status'])
    print(f"[SENSOR] 启动后: 通电={'是' if state['power'] else '否'}, 状态={label}")

    # 5. 等待自动定时停止
    print("\n[WAIT] 等待设备工作中...")
    for i in range(12):
        time.sleep(1)
        state = get_state()
        is_running = state["status"] == "running"
        icon = "[RUN]" if is_running else "[STBY]"
        label = status_labels.get(state['status'], state['status'])
        print(f"   {icon} [{i+1}s] 通电={'是' if state['power'] else '否'} 状态={label}")
        if state["status"] != "running":
            break

    # 6. 手动停止（如果还在运行）
    if state["status"] == "running":
        print("\n[CMD] 停止工作 (stop)...")
        send_command("stop")
        time.sleep(0.5)
        state = get_state()
        label = status_labels.get(state['status'], state['status'])
        print(f"[SENSOR] 停止后: 通电={'是' if state['power'] else '否'}, 状态={label} (保持通电!)")

    # 7. 关机断电
    print("\n[CMD] 关机断电 (power_off)...")
    send_command("power_off")
    time.sleep(0.5)
    state = get_state()
    label = status_labels.get(state['status'], state['status'])
    print(f"[SENSOR] 关机后: 通电={'是' if state['power'] else '否'}, 状态={label}")

    # 8. 设置参数
    print("\n[CMD] 测试参数设置...")
    result = send_command("set_param", {"temperature": 28.5, "humidity": 55.0})
    print(f"   响应: {result}")
    state = get_state()
    print(f"[SENSOR] 更新后温度: {state['temperature']}°C, 湿度: {state['humidity']}%")

    print(f"\n{'='*50}")
    print("[OK] HTTP 设备客户端测试完成！")
    print("   完整生命周期: power_on -> start -> (auto-stop) -> power_off")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
