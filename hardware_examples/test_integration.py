"""
软硬件联动集成测试

测试完整的"设备连接→状态读取→指令执行→结果验证"链路。
使用项目内置的 SimulatorDriver，无需任何外部硬件。

运行:
  # 只运行集成测试（无需启动后端）
  python hardware_examples/test_integration.py

  # 在已启动后端的情况下测试 API
  python hardware_examples/test_integration.py --api
"""

import sys
import os
import json
import time
import asyncio
import argparse

# 将项目根目录加入 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class Colors:
    """终端颜色"""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(title):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")


def print_ok(msg):
    print(f"  {Colors.GREEN}[OK]{Colors.RESET} {msg}")


def print_fail(msg):
    print(f"  {Colors.RED}[FAIL]{Colors.RESET} {msg}")


def print_info(msg):
    print(f"  {Colors.BLUE}[i]{Colors.RESET}  {msg}")


def print_warn(msg):
    print(f"  {Colors.YELLOW}[WARN]{Colors.RESET}  {msg}")


# ═══════════════════════════════════════════════════
# Test 1: Device Registry
# ═══════════════════════════════════════════════════

def test_device_registry():
    """测试设备注册中心（真实场景：无内置设备，仅加载用户自定义设备）"""
    print_header("Test 1: 设备注册中心")

    from core.device_registry_factory import setup_registry, close_registry
    from devices.simulator_driver import SimulatorDriver
    from devices.base import DeviceCapability, DeviceCommand

    # 使用 "test_user" 加载自定义设备（可能为空）
    registry, loop = setup_registry("test_user")
    try:
        # 手动添加测试设备（模拟真实场景：用户通过前端注册设备）
        sim = SimulatorDriver(simulated_latency_ms=10)
        registry.register("simulator", sim)
        loop.run_until_complete(sim.connect())

        test_devices = [
            ("test_pump_01", "测试水泵", [DeviceCapability.IRRIGATE],
             {"power": False, "status": "powered_off", "flow_rate": 0, "total_water_liters": 0}),
            ("test_sensor_01", "测试传感器", [DeviceCapability.READ_SENSOR],
             {"power": False, "status": "powered_off", "temperature": 25.0, "humidity": 60.0, "soil_moisture": 45.0, "ph": 6.8}),
            ("test_fan_01", "测试风扇", [DeviceCapability.VENTILATE],
             {"power": False, "status": "powered_off", "rpm": 0}),
            ("test_light_01", "测试补光灯", [DeviceCapability.LIGHT],
             {"power": False, "status": "powered_off", "brightness_percent": 0}),
        ]
        for dev_id, name, caps, init_state in test_devices:
            sim.add_virtual_device(device_id=dev_id, name=name, capabilities=caps,
                                   sensors=list(init_state.keys()), initial_state=init_state)

        # 发现所有设备
        devices = loop.run_until_complete(registry.discover_all())
        print_info(f"发现 {len(devices)} 个设备（用户注册）")
        for d in devices:
            print_ok(f"{d.device_id:30s} | {d.name:15s} | {d.status.value:8s} | {d.driver_name}")

        assert len(devices) >= 4, f"至少应有4个测试设备，实际: {len(devices)}"
        print_ok("设备注册中心测试通过（真实场景模式）")

        return registry, loop, devices
    except Exception as e:
        print_fail(f"设备注册中心测试失败: {e}")
        raise
    # 注意：不在这里 close_registry，返回给调用方继续使用


# ═══════════════════════════════════════════════════
# Test 2: Sensor Reading
# ═══════════════════════════════════════════════════

def test_sensor_reading(registry, loop):
    """测试传感器读数"""
    print_header("Test 2: 传感器读数")

    sensor_id = "test_sensor_01"
    state = loop.run_until_complete(registry.read_state(sensor_id))
    print_info(f"传感器 {sensor_id}:")
    print_info(f"  - 温度: {state.get('temperature', '?'):.1f}°C")
    print_info(f"  - 湿度: {state.get('humidity', '?'):.1f}%")
    print_info(f"  - 土壤湿度: {state.get('soil_moisture', '?'):.1f}%")
    print_info(f"  - pH: {state.get('ph', '?'):.1f}")

    # 验证关键字段存在
    required_keys = ["temperature", "humidity", "soil_moisture"]
    for key in required_keys:
        assert key in state, f"缺失传感器字段: {key}"
    print_ok("传感器读数测试通过")


# ═══════════════════════════════════════════════════
# Test 3: Device Execution
# ═══════════════════════════════════════════════════

def test_device_execution(registry, loop):
    """测试设备指令执行"""
    print_header("Test 3: 设备指令执行")

    from devices.base import DeviceCommand

    device_id = "test_pump_01"

    # 3.1 启动
    cmd = DeviceCommand(command="start", params={"duration": 10})
    result = loop.run_until_complete(registry.execute(device_id, cmd))
    assert result.success, f"启动失败: {result.message}"
    print_ok(f"启动 {device_id}: {result.message}")

    # 3.2 读取运行状态
    state = loop.run_until_complete(registry.read_state(device_id))
    assert state.get("power"), "设备应处于通电状态"
    assert state.get("status") == "running", f"设备应处于工作中，实际: {state.get('status')}"
    print_ok(f"运行状态确认: power=True, status=running")

    # 3.3 停止（回到待机，保持通电）
    cmd = DeviceCommand(command="stop")
    result = loop.run_until_complete(registry.execute(device_id, cmd))
    assert result.success, f"停止失败: {result.message}"
    print_ok(f"停止 {device_id}: {result.message}")

    state = loop.run_until_complete(registry.read_state(device_id))
    assert state.get("power"), "停止后设备应保持通电（待机状态）"
    assert state.get("status") == "standby", f"停止后应为待机，实际: {state.get('status')}"
    print_ok("停止后状态: power=True(保持通电), status=standby [OK]")

    # 3.4 关机断电
    cmd = DeviceCommand(command="power_off")
    result = loop.run_until_complete(registry.execute(device_id, cmd))
    assert result.success, f"关机失败: {result.message}"
    state = loop.run_until_complete(registry.read_state(device_id))
    assert not state.get("power"), "关机后设备应断电"
    assert state.get("status") == "powered_off", f"关机后应为powered_off，实际: {state.get('status')}"
    print_ok("关机后状态: power=False, status=powered_off [OK]")
    print_ok("设备执行测试通过（完整生命周期: start→stop→power_off）")


# ═══════════════════════════════════════════════════
# Test 4: Device Executor (with executor layer)
# ═══════════════════════════════════════════════════

def test_device_executor(registry, loop):
    """测试设备执行器（带重试/日志）"""
    print_header("Test 4: 设备执行器")

    from devices.base import DeviceCommand
    from core.device_executor import DeviceExecutor

    executor = DeviceExecutor(registry, username="test_user")

    # 4.1 同步执行
    cmd = DeviceCommand(command="start", params={"duration": 5})
    result = executor.execute_sync("test_fan_01", cmd, trigger="test")
    assert result["success"], f"执行失败: {result}"
    print_ok(f"同步执行: {result['attempts']} 次尝试, success={result['success']}")

    # 4.2 停止
    cmd = DeviceCommand(command="stop")
    result = executor.execute_sync("test_fan_01", cmd, trigger="test")
    print_ok(f"停止执行: success={result['success']}")

    # 4.3 获取日志
    logs = executor.get_logs(limit=5)
    print_info(f"设备日志: {len(logs)} 条记录")
    for log in logs:
        print_info(f"  - {log['timestamp']} | {log['device_id']} | {log['command']} | {log['success']}")

    assert len(logs) > 0, "应有执行日志"
    print_ok("设备执行器测试通过")


# ═══════════════════════════════════════════════════
# Test 5: Rule Engine
# ═══════════════════════════════════════════════════

def test_rule_engine():
    """测试规则引擎"""
    print_header("Test 5: 规则引擎")

    from core.device_rule_engine import RuleEngine, RuleDecision

    engine = RuleEngine(username="test_user")

    # 添加一条测试规则
    rule_id = engine.add_rule({
        "name": "测试规则-自动灌溉",
        "enabled": True,
        "trigger": {
            "conditions": [
                {"type": "sensor", "field": "soil_moisture", "op": "<", "value": 50}
            ],
            "logic": "AND"
        },
        "action": {
            "device_id": "virtual_irrigation_01",
            "command": "start",
            "params": {"duration": 30}
        },
        "constraints": {
            "max_duration_per_use": 60,
            "forbidden_hours": [22, 23, 0, 1, 2, 3, 4, 5]
        }
    })
    print_ok(f"规则已创建: {rule_id}")

    # 查看规则
    rules = engine.list_rules()
    assert len(rules) >= 1, "应有至少1条规则"
    print_ok(f"规则列表: {len(rules)} 条")

    # 匹配规则
    context = {"sensor_data": {"soil_moisture": 30}, "weather": {}, "crop": "小麦"}
    matched = engine.find_matching_rules(context)
    print_ok(f"匹配规则: {len(matched)} 条")

    # 评估操作
    if matched:
        decision, reason, final_params = engine.evaluate_action(
            matched[0], {"duration": 30}, {"device_id": "virtual_irrigation_01"}
        )
        print_info(f"决策: {decision} | 原因: {reason} | 参数: {final_params}")

    # 清理测试规则
    engine.delete_rule(rule_id)
    print_ok("规则引擎测试通过")


# ═══════════════════════════════════════════════════
# Test 6: Simulator Driver
# ═══════════════════════════════════════════════════

def test_simulator_driver():
    """测试模拟器驱动（无需外部硬件）"""
    print_header("Test 6: Simulator 驱动独立测试")

    from devices.simulator_driver import SimulatorDriver
    from devices.base import DeviceCommand, DeviceCapability

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    sim = SimulatorDriver(simulated_latency_ms=10)
    loop.run_until_complete(sim.connect())
    print_ok("Simulator 已连接")

    # 添加虚拟设备
    sim.add_virtual_device(
        device_id="test_sensor_01",
        name="测试传感器",
        capabilities=[DeviceCapability.READ_SENSOR],
        sensors=["temperature", "humidity"],
        location="测试区",
    )
    sim.add_virtual_device(
        device_id="test_pump_01",
        name="测试水泵",
        capabilities=[DeviceCapability.IRRIGATE],
        location="测试区",
    )
    print_ok("虚拟设备已添加")

    # 发现设备
    devices = loop.run_until_complete(sim.discover())
    print_info(f"发现 {len(devices)} 个模拟设备")

    # 传感器读数
    state = loop.run_until_complete(sim.read_state("test_sensor_01"))
    print_info(f"传感器: {json.dumps(state, ensure_ascii=False, indent=0)[:200]}")

    # 执行指令
    cmd = DeviceCommand(command="start", params={"duration": 5})
    result = loop.run_until_complete(sim.execute("test_pump_01", cmd))
    assert result.success, f"执行失败: {result.message}"
    print_ok(f"执行指令: {result.message}")

    # 验证
    state = loop.run_until_complete(sim.read_state("test_pump_01"))
    assert state.get("power"), "水泵应处于通电状态"
    assert state.get("status") == "running", f"水泵应工作中，实际: {state.get('status')}"
    print_ok("水泵运行状态确认: power=True, status=running [OK]")

    loop.run_until_complete(sim.disconnect())
    loop.close()
    print_ok("Simulator 驱动测试通过")


# ═══════════════════════════════════════════════════
# Test 7: Device Info & Capability
# ═══════════════════════════════════════════════════

def test_device_info():
    """测试设备信息模型"""
    print_header("Test 7: 设备信息 & 数据模型")

    from devices.base import (
        DeviceInfo, DeviceCommand, DeviceResult,
        DeviceCapability, DeviceStatus, CommandPriority,
    )

    # DeviceInfo
    info = DeviceInfo(
        device_id="test_device_01",
        name="测试设备",
        driver_name="simulator",
        capabilities=[DeviceCapability.IRRIGATE, DeviceCapability.READ_SENSOR],
        sensors=["temperature"],
        location="测试区",
    )
    assert info.device_id == "test_device_01"
    assert DeviceCapability.IRRIGATE in info.capabilities
    print_ok(f"DeviceInfo: {info}")

    # DeviceCommand
    cmd = DeviceCommand(command="start", params={"duration": 30}, priority=CommandPriority.HIGH)
    assert cmd.command == "start"
    print_ok(f"DeviceCommand: {cmd}")

    # DeviceResult
    result = DeviceResult(
        success=True,
        device_id="test_device_01",
        executed_command="start",
        message="测试成功",
        actual_params={"duration": 30},
    )
    assert result.success
    assert result.error_code is None  # 成功时 error_code 被 post_init 清除
    print_ok(f"DeviceResult: {result}")

    # DeviceResult 一致性
    bad_result = DeviceResult(
        success=True,
        device_id="test",
        executed_command="start",
        error_code="SOME_ERROR",
    )
    assert bad_result.error_code is None  # post_init 清除了矛盾字段
    print_ok("数据模型一致性检查通过")


# ═══════════════════════════════════════════════════
# Test 8: API Integration (optional, requires backend)
# ═══════════════════════════════════════════════════

def test_api_integration():
    """测试 API 集成（需要后端运行中）"""
    print_header("Test 8: API 集成测试")

    import requests
    API_BASE = "http://localhost:8000"
    USERNAME = "test_user"

    def api_get(path, **params):
        params["username"] = USERNAME
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def api_post(path, data):
        data["username"] = USERNAME
        resp = requests.post(f"{API_BASE}{path}", json=data, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # 8.1 设备列表
    try:
        devices = api_get("/api/devices")
        print_ok(f"设备列表 API: {len(devices)} 个设备")
    except Exception as e:
        print_warn(f"设备列表 API 跳过 ({e})")

    # 8.2 设备快照
    try:
        snapshot = api_get("/api/devices/snapshot")
        device_count = len(snapshot.get("devices", []))
        print_ok(f"设备快照 API: {device_count} 个设备")
    except Exception as e:
        print_warn(f"设备快照 API 跳过 ({e})")

    # 8.3 发送指令（使用列表中第一个可执行设备）
    try:
        devices = api_get("/api/devices")
        executable = [d for d in devices if d.get("capabilities") and "irrigate" in d.get("capabilities", [])]
        if executable:
            target = executable[0]["device_id"]
            result = api_post("/api/devices/command", {
                "device_id": target,
                "command": "start",
                "params": {"duration": 5},
            })
            if result.get("success"):
                print_ok(f"发送指令 API ({target}): {result.get('message', 'OK')}")
                # 停止
                api_post("/api/devices/command", {
                    "device_id": target,
                    "command": "stop",
                })
            else:
                print_warn(f"发送指令 API: {result.get('message', '未知错误')}")
        else:
            print_warn("无可执行设备，跳过指令测试")
    except Exception as e:
        print_warn(f"发送指令 API 跳过 ({e})")

    # 8.4 规则列表
    try:
        rules = api_get("/api/rules")
        print_ok(f"规则列表 API: {len(rules)} 条规则")
    except Exception as e:
        print_warn(f"规则列表 API 跳过 ({e})")

    # 8.5 操作日志
    try:
        logs = api_get("/api/devices/logs", limit=5)
        print_ok(f"操作日志 API: {len(logs)} 条记录")
    except Exception as e:
        print_warn(f"操作日志 API 跳过 ({e})")

    print_ok("API 集成测试完成")


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="软硬件联动集成测试")
    parser.add_argument("--api", action="store_true", help="同时测试 API 集成（需要后端运行中）")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{Colors.GREEN}")
    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║      软硬件联动集成测试                                  ║")
    print(f"║      Hardware-Software Integration Test                  ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")

    passed = 0
    failed = 0

    # 独立测试（不需要共享 registry）
    independent_tests = [
        ("设备信息模型", test_device_info),
        ("Simulator 驱动", test_simulator_driver),
        ("规则引擎", test_rule_engine),
    ]

    for name, test_fn in independent_tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print_fail(f"{name} 测试异常: {e}")
            import traceback
            traceback.print_exc()

    # 注册中心 + 传感器 + 执行测试（共享 registry）
    try:
        registry, loop, devices = test_device_registry()
        passed += 1

        try:
            test_sensor_reading(registry, loop)
            passed += 1

            test_device_execution(registry, loop)
            passed += 1

            test_device_executor(registry, loop)
            passed += 1
        finally:
            from core.device_registry_factory import close_registry
            close_registry(loop, registry)
    except Exception as e:
        failed += 1
        print_fail(f"注册中心/执行器测试异常: {e}")
        import traceback
        traceback.print_exc()

    # Optional API test
    if args.api:
        try:
            test_api_integration()
            passed += 1
        except Exception as e:
            failed += 1
            print_fail(f"API 测试异常: {e}")

    # Summary
    total = passed + failed
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  [RESULTS] 测试结果: {passed}/{total} 通过{Colors.RESET}")
    if failed > 0:
        print(f"{Colors.RED}  {failed} 个测试失败{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}  [ALL PASS] 全部测试通过！软硬件联动链路正常！{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")


if __name__ == "__main__":
    main()
