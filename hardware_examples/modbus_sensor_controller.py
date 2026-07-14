"""
Modbus RTU 传感器+控制器模拟

模拟一个通过 Modbus RTU 协议通信的传感器+阀门控制器。
使用 Holding Registers 存储传感器数据和设备状态。

设备生命周期:
  关机(powered_off) ──[power_on]──▶ 待机(standby) ──[start]──▶ 工作中(running)
       ▲                                ▲                       │
       │                                │ [stop]                │
       │                                ◀───────────────────────┘

寄存器映射:
  HR[0] = 电源状态 (0=断电, 1=通电)
  HR[1] = 运行状态 (0=关机powered_off, 1=待机standby, 2=工作中running, 3=故障error)
  HR[2] = 设定值 (如温度设定点 x10)
  HR[3] = 当前值 (如当前温度 x10)

依赖: pip install pymodbus (可选, 无硬件时可运行内置模拟)
运行:
  # 纯模拟模式 (无需硬件)
  python hardware_examples/modbus_sensor_controller.py

  # 连接真实 Modbus 设备
  python hardware_examples/modbus_sensor_controller.py --port COM3 --slave 1
"""

import time
import random
import argparse
import threading

# ── 状态常量 ──────────────────────────────────────
STATUS_POWERED_OFF = 0   # 关机
STATUS_STANDBY = 1       # 待机
STATUS_RUNNING = 2       # 工作中
STATUS_ERROR = 3         # 故障

STATUS_LABELS = {
    0: "关机(powered_off)",
    1: "待机(standby)",
    2: "工作中(running)",
    3: "故障(error)",
}

# ── 模拟设备状态 ──────────────────────────────────

class ModbusSimulator:
    """纯软件模拟的 Modbus 设备（无需真实硬件）

    完整生命周期状态机:
      powered_off(0) ──[power_on]──▶ standby(1) ──[start]──▶ running(2)
          ▲                               ▲                    │
          │                               │ [stop]             │
          │                               ◀────────────────────┘
          │
          └──────────[power_off]──────────┘
    """

    def __init__(self, slave_id=1):
        self.slave_id = slave_id
        # HR[0]=电源(0/1), HR[1]=状态(0-3), HR[2]=设定值, HR[3]=当前值
        # 初始状态: 关机 (power=0, status=0)
        self.registers = [0, STATUS_POWERED_OFF, 250, 225]
        self._running = False
        self._lock = threading.Lock()

    def write_register(self, address, value):
        """写入单个寄存器，自动处理状态转换"""
        with self._lock:
            if not (0 <= address < len(self.registers)):
                return False

            if address == 0:
                # HR[0]: 电源控制
                # 1 = 通电(power_on), 0 = 断电(power_off)
                current_status = self.registers[1]
                if value == 1:
                    if current_status == STATUS_POWERED_OFF:
                        self.registers[1] = STATUS_STANDBY  # 通电→待机
                    # 如果已在待机或工作中，不改变状态
                elif value == 0:
                    if current_status in (STATUS_STANDBY, STATUS_RUNNING):
                        self.registers[1] = STATUS_POWERED_OFF  # 断电→关机
                    elif current_status == STATUS_ERROR:
                        self.registers[1] = STATUS_POWERED_OFF  # 故障下强制关机

            elif address == 1:
                # HR[1]: 直接状态控制
                # 0=关机, 1=待机, 2=工作中, 3=故障
                if 0 <= value <= 3:
                    self.registers[1] = value
                    # 同步 HR[0] 电源状态
                    if value in (STATUS_STANDBY, STATUS_RUNNING):
                        self.registers[0] = 1  # 通电
                    else:
                        self.registers[0] = 0  # 断电

            self.registers[address] = value
            return True

    def read_registers(self, start, count):
        with self._lock:
            end = min(start + count, len(self.registers))
            return self.registers[start:end]

    def execute_command(self, command):
        """执行高级指令，遵循状态机规则"""
        with self._lock:
            current = self.registers[1]

            if command in ("power_on", "boot"):
                if current == STATUS_POWERED_OFF:
                    self.registers[0] = 1
                    self.registers[1] = STATUS_STANDBY
                    return True, "通电启动，进入待机"
                return True, "已在待机或工作中"

            elif command in ("power_off", "shutdown"):
                if current in (STATUS_STANDBY, STATUS_RUNNING):
                    self.registers[0] = 0
                    self.registers[1] = STATUS_POWERED_OFF
                    return True, "关机断电"
                elif current == STATUS_ERROR:
                    self.registers[0] = 0
                    self.registers[1] = STATUS_POWERED_OFF
                    return True, "故障状态强制关机"
                return True, "已在关机状态"

            elif command == "start":
                if current == STATUS_POWERED_OFF:
                    self.registers[0] = 1
                    self.registers[1] = STATUS_RUNNING
                    return True, "通电并开始工作"
                elif current == STATUS_STANDBY:
                    self.registers[1] = STATUS_RUNNING
                    return True, "开始工作"
                elif current == STATUS_RUNNING:
                    return True, "已在工作中"
                elif current == STATUS_ERROR:
                    return False, "设备故障，请先复位(reset)"

            elif command == "stop":
                if current == STATUS_RUNNING:
                    self.registers[1] = STATUS_STANDBY
                    # power 保持 1，不断电
                    return True, "停止工作，回到待机（保持通电）"
                return True, "当前未在工作"

            elif command == "reset":
                if current == STATUS_ERROR:
                    self.registers[0] = 1
                    self.registers[1] = STATUS_STANDBY
                    return True, "故障复位，恢复到待机"
                return True, "设备未处于故障状态"

            return False, f"未知指令: {command}"

    def simulate(self, interval=2.0):
        """模拟传感器数据变化"""
        while self._running:
            with self._lock:
                # 温度波动 (±0.5°C, 以 x10 存储)
                self.registers[3] += random.randint(-5, 5)
                self.registers[3] = max(0, min(500, self.registers[3]))

                # 设定值不变
            time.sleep(interval)

    def start_simulation(self):
        self._running = True
        t = threading.Thread(target=self.simulate, daemon=True)
        t.start()
        return t

    def stop_simulation(self):
        self._running = False


def test_with_pymodbus(port, slave_id):
    """使用 pymodbus 库测试真实 Modbus 设备"""
    try:
        from pymodbus.client import ModbusSerialClient
    except ImportError:
        print("[ERR] pymodbus 未安装: pip install pymodbus")
        return False

    client = ModbusSerialClient(
        port=port,
        baudrate=9600,
        timeout=1,
    )

    if not client.connect():
        print(f"[ERR] 无法连接到 {port}")
        return False

    print(f"[OK] 已连接到 {port}, 从站 ID={slave_id}")

    try:
        # 读取寄存器
        result = client.read_holding_registers(0, 4, slave=slave_id)
        if result.isError():
            print(f"[ERR] 读取失败: {result}")
            return False

        regs = result.registers
        status_label = STATUS_LABELS.get(regs[1], f"unknown({regs[1]})")
        print(f"\n[SENSOR] 寄存器状态:")
        print(f"  HR[0] 电源: {'通电' if regs[0] else '断电'}")
        print(f"  HR[1] 状态: {regs[1]} ({status_label})")
        print(f"  HR[2] 设定值: {regs[2]/10:.1f}")
        print(f"  HR[3] 当前值: {regs[3]/10:.1f}")

        # 通电启动
        print(f"\n[CMD] 通电启动 (HR[0]=1)...")
        result = client.write_register(0, 1, slave=slave_id)
        if not result.isError():
            print(f"  [OK] 通电成功")
        else:
            print(f"  [ERR] 通电失败")

        time.sleep(0.5)

        # 开始工作
        print(f"[CMD] 开始工作 (HR[1]=2)...")
        result = client.write_register(1, 2, slave=slave_id)
        if not result.isError():
            print(f"  [OK] 开始工作")
        else:
            print(f"  [ERR] 启动失败")

        time.sleep(1)

        # 验证
        result = client.read_holding_registers(0, 4, slave=slave_id)
        if not result.isError():
            s = STATUS_LABELS.get(result.registers[1], "?")
            print(f"  [VERIFY] HR[0]={result.registers[0]}({'通电' if result.registers[0] else '断电'}), HR[1]={result.registers[1]}({s})")

        # 停止工作（回到待机）
        print(f"[CMD] 停止工作 (HR[1]=1)...")
        client.write_register(1, 1, slave=slave_id)
        time.sleep(0.5)
        result = client.read_holding_registers(0, 1, slave=slave_id)
        if not result.isError():
            print(f"  [VERIFY] HR[0]={result.registers[0]}(通电保持), HR[1]={result.registers[1]}({STATUS_LABELS.get(result.registers[1], '?')})")

        # 关机断电
        print(f"[CMD] 关机断电 (HR[0]=0)...")
        client.write_register(0, 0, slave=slave_id)
        time.sleep(0.5)
        result = client.read_holding_registers(0, 2, slave=slave_id)
        if not result.isError():
            s = STATUS_LABELS.get(result.registers[1], "?")
            print(f"  [VERIFY] HR[0]={result.registers[0]}(断电), HR[1]={result.registers[1]}({s})")

        return True

    finally:
        client.close()


def test_simulated():
    """纯模拟测试（无需硬件）"""
    print("\n[TEST] 使用纯软件模拟测试 Modbus 协议")

    sim = ModbusSimulator(slave_id=1)
    sim.start_simulation()

    try:
        # 1. 读取初始状态
        print("\n--- 步骤1: 读取初始状态(关机) ---")
        regs = sim.read_registers(0, 4)
        print(f"  HR[0]={regs[0]}, HR[1]={regs[1]}({STATUS_LABELS.get(regs[1])}), "
              f"HR[2]={regs[2]/10:.1f}, HR[3]={regs[3]/10:.1f}")

        # 2. 通电启动
        print("\n--- 步骤2: 通电启动(power_on) ---")
        ok, msg = sim.execute_command("power_on")
        regs = sim.read_registers(0, 4)
        print(f"  {msg}: HR[0]={regs[0]}, HR[1]={regs[1]}({STATUS_LABELS.get(regs[1])})")

        # 3. 开始工作
        print("\n--- 步骤3: 开始工作(start) ---")
        ok, msg = sim.execute_command("start")
        regs = sim.read_registers(0, 4)
        print(f"  {msg}: HR[0]={regs[0]}, HR[1]={regs[1]}({STATUS_LABELS.get(regs[1])})")

        # 4. 传感器数据模拟
        print("\n--- 步骤4: 传感器数据波动(5次) ---")
        for i in range(5):
            time.sleep(1)
            regs = sim.read_registers(0, 4)
            print(f"  [{i+1}] HR[3] 当前值: {regs[3]/10:.1f}")

        # 5. 停止工作（回到待机，保持通电）
        print("\n--- 步骤5: 停止工作(stop) → 待机 ---")
        ok, msg = sim.execute_command("stop")
        regs = sim.read_registers(0, 4)
        print(f"  {msg}: HR[0]={regs[0]}(通电保持!), HR[1]={regs[1]}({STATUS_LABELS.get(regs[1])})")

        # 6. 关机断电
        print("\n--- 步骤6: 关机断电(power_off) ---")
        ok, msg = sim.execute_command("power_off")
        regs = sim.read_registers(0, 4)
        print(f"  {msg}: HR[0]={regs[0]}, HR[1]={regs[1]}({STATUS_LABELS.get(regs[1])})")

        print("\n[OK] 模拟 Modbus 状态机测试通过!")
        print("  验证: powered_off → standby → running → standby → powered_off")

    finally:
        sim.stop_simulation()


def main():
    parser = argparse.ArgumentParser(description="Modbus 传感器+控制器测试")
    parser.add_argument("--port", default=None, help="串口名称 (如 COM3 或 /dev/ttyUSB0)")
    parser.add_argument("--slave", type=int, default=1, help="从站 ID (默认1)")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"Modbus 传感器+控制器测试")
    print(f"   生命周期: power_on -> start -> stop -> power_off")
    print(f"   初始状态: 关机(powered_off)")
    print(f"{'='*50}")

    if args.port:
        # 真实硬件模式
        ok = test_with_pymodbus(args.port, args.slave)
        if ok:
            print(f"\n[OK] Modbus 硬件测试通过!")
        else:
            print(f"\n[ERR] Modbus 硬件测试失败")
    else:
        # 模拟模式
        print("\n[INFO] 未指定 --port，使用纯软件模拟...")
        print("   指定 --port COM3 可测试真实 Modbus 设备")
        test_simulated()


if __name__ == "__main__":
    main()
