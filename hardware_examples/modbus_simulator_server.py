"""
Modbus TCP 协议模拟器服务器 — 多从站设备统一管理

模拟 7 种农业 IoT 设备作为 Modbus TCP 从站。
Agent 通过 ModbusDriver 连接本服务器，走真实 Modbus TCP 协议。

依赖: pip install pymodbus
运行: python hardware_examples/modbus_simulator_server.py [--port 5020]

寄存器映射（每个从站）:
  地址 0:  设备状态 (0=powered_off, 1=standby, 2=running, 3=error)
  地址 1:  电源 (0=off, 1=on)
  地址 10:  温度 (x10, 如 220 = 22.0°C)
  地址 11:  湿度 (x10, 如 650 = 65.0%)
  地址 12:  土壤湿度 (x10)
  地址 13:  CO2 (ppm)
  地址 20:  指令寄存器 (写1=power_on, 2=power_off, 3=start, 4=stop, 5=reset)
  地址 21:  指令参数 duration 秒数
"""

import random
import struct
import sys
import threading
import time
from datetime import datetime

try:
    from pymodbus.server import StartTcpServer
    from pymodbus.device import ModbusDeviceIdentification
    from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
    from pymodbus.version import version
except ImportError:
    print("[ERR] 请先安装 pymodbus: pip install pymodbus")
    sys.exit(1)

# ── 从站设备 ─────────────────────────────
SLAVE_CONFIG = {
    1: {"name": "温室灌溉泵-Modbus", "sensors": ["flow_rate"]},
    2: {"name": "温室通风扇-Modbus", "sensors": ["rpm"]},
    3: {"name": "温室补光灯-Modbus", "sensors": ["light_lux"]},
    4: {"name": "温室加热器-Modbus", "sensors": ["temperature"]},
    5: {"name": "环境温湿度传感器-Modbus", "sensors": ["temperature", "humidity", "soil_moisture", "co2_ppm"]},
    6: {"name": "施肥一体机-Modbus", "sensors": ["flow_rate"]},
    7: {"name": "温室监控摄像头-Modbus", "sensors": []},
}

# 每个从站的内部状态
_slave_states = {}
_lock = threading.Lock()

# 寄存器布局: 0-9 控制, 10-19 传感器, 20-29 指令
REG_COUNT = 30


def _init_slave(slave_id):
    registers = [0] * REG_COUNT
    registers[0] = 0   # powered_off
    registers[1] = 0   # power off
    registers[10] = 220  # 温度 22.0°C
    registers[11] = 650  # 湿度 65.0%
    registers[12] = 450  # 土壤湿度 45.0%
    registers[13] = 400  # CO2 400ppm
    return registers


for sid in SLAVE_CONFIG:
    _slave_states[sid] = _init_slave(sid)


def _sensor_drift(sid, registers):
    """传感器漂移模拟"""
    registers[10] = max(50, min(550, registers[10] + random.randint(-3, 3)))  # 温度
    registers[11] = max(100, min(990, registers[11] + random.randint(-15, 15)))  # 湿度
    irrigation_on = any(
        _slave_states[s][1] == 1 and _slave_states[s][0] == 2
        for s in SLAVE_CONFIG if "irrigat" in SLAVE_CONFIG[s].get("name", "").lower()
    )
    if irrigation_on:
        registers[12] = min(950, registers[12] + random.randint(5, 12))
    else:
        registers[12] = max(50, registers[12] - random.randint(1, 3))
    registers[13] = max(300, min(2000, registers[13] + random.randint(-5, 5)))


def _process_command(sid, registers):
    """检查指令寄存器，执行状态机"""
    cmd = registers[20]
    if cmd == 0:
        return  # 无指令

    duration = registers[21]
    name = SLAVE_CONFIG[sid]["name"]
    current = registers[0]

    if cmd == 1:  # power_on
        if current == 0:
            registers[1] = 1; registers[0] = 1  # standby
    elif cmd == 2:  # power_off
        registers[1] = 0; registers[0] = 0
    elif cmd == 3:  # start
        registers[1] = 1; registers[0] = 2  # running
    elif cmd == 4:  # stop
        if current == 2:
            registers[0] = 1  # back to standby
    elif cmd == 5:  # reset
        registers[1] = 0; registers[0] = 0

    registers[20] = 0  # 清除指令
    registers[21] = 0


# ── 自定义 DataBlock（每次读写时触发传感器漂移和指令处理）──

class SimulatorDataBlock(ModbusSequentialDataBlock):
    """自定义数据块 — 读写时自动更新传感器数据"""

    def __init__(self, slave_id, address, values):
        super().__init__(address, values)
        self._sid = slave_id

    def getValues(self, address, count=1):
        with _lock:
            _sensor_drift(self._sid, _slave_states[self._sid])
        return super().getValues(address, count)

    def setValues(self, address, values):
        result = super().setValues(address, values)
        with _lock:
            _process_command(self._sid, _slave_states[self._sid])
        return result


# ── 构建 Modbus Server ────────────────────

def _build_context():
    slaves = {}
    for sid, config in SLAVE_CONFIG.items():
        block = SimulatorDataBlock(sid, 0, _slave_states[sid])
        store = ModbusSlaveContext(hr=block, ir=block)  # holding + input registers
        slaves[sid] = store
    return ModbusServerContext(slaves=slaves, single=False)


# ── 漂移更新线程 ──────────────────────────

def _drift_loop(interval=5):
    while True:
        time.sleep(interval)
        with _lock:
            for sid in SLAVE_CONFIG:
                _sensor_drift(sid, _slave_states[sid])


# ── 启动 ──────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Modbus TCP 农业设备模拟器")
    p.add_argument("--port", type=int, default=5020, help="监听端口（默认5020，避免与系统502冲突）")
    p.add_argument("--host", default="127.0.0.1", help="监听地址")
    args = p.parse_args()

    print(f"[Modbus模拟器] 已初始化 {len(SLAVE_CONFIG)} 个从站设备")
    for sid, config in SLAVE_CONFIG.items():
        print(f"  从站#{sid}: {config['name']} (寄存器地址 0-{REG_COUNT-1})")

    context = _build_context()

    # 启动漂移线程
    t = threading.Thread(target=_drift_loop, args=(5,), daemon=True)
    t.start()

    print(f"\n[Modbus模拟器] Modbus TCP: {args.host}:{args.port}")
    print("[Modbus模拟器] 下方可直接输入命令操控设备（输入 help 查看帮助）")
    print("  指令映射: on=写寄存器20=1, off=2, start=3, stop=4, reset=5, error=写状态=3")

    # Modbus server 在后台线程
    def run_modbus():
        StartTcpServer(context=context, identity=ModbusDeviceIdentification(),
                       address=(args.host, args.port))
    threading.Thread(target=run_modbus, daemon=True).start()
    import time as _time; _time.sleep(1)

    HELP = "命令: on/off/start/stop/reset/error <从站号> | set <从站号> <寄存器地址> <值> | state <从站号> | list | quit"
    print(HELP)
    while True:
        try:
            line = input("modbus> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line: continue
        parts = line.split()
        cmd = parts[0].lower()
        if cmd == "quit": break
        elif cmd == "help": print(HELP)
        elif cmd == "list":
            with _lock:
                for sid, cfg in SLAVE_CONFIG.items():
                    regs = _slave_states[sid]
                    status = ["powered_off","standby","running","error"][min(regs[0], 3)]
                    print(f"  从站#{sid}: {cfg['name']:15s} [{status:12s}] power={regs[1]}")
        elif cmd == "state" and len(parts) >= 2:
            sid = int(parts[1])
            with _lock:
                if sid in _slave_states:
                    regs = _slave_states[sid]
                    print(f"  status={regs[0]} power={regs[1]} temp={regs[10]/10:.1f}°C humidity={regs[11]/10:.1f}% soil={regs[12]/10:.1f}% co2={regs[13]}ppm")
        elif cmd in ("on","off","start","stop","reset","error") and len(parts) >= 2:
            cmd_map = {"on":1,"off":2,"start":3,"stop":4,"reset":5,"error":-1}
            val = cmd_map[cmd]
            sid = int(parts[1])
            with _lock:
                if val == -1:
                    _slave_states[sid][0] = 3  # error
                else:
                    _slave_states[sid][20] = val
                    _process_command(sid, _slave_states[sid])
            status = ["powered_off","standby","running","error"][min(_slave_states[sid][0], 3)]
            print(f"  从站#{sid}: {SLAVE_CONFIG[sid]['name']} → {status}")
        elif cmd == "set" and len(parts) >= 4:
            sid, addr = int(parts[1]), int(parts[2])
            val = float(parts[3])
            with _lock:
                _slave_states[sid][addr] = int(val * 10) if addr >= 10 else int(val)
            print(f"  从站#{sid} 寄存器{addr} = {val}")
