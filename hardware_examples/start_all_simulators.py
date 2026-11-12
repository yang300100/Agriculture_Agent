"""启动共享状态的硬件模拟器，保留旧入口与协议开关。

运行：python hardware_examples/start_all_simulators.py [--headless]
可选：--no-http、--no-mqtt、--no-modbus。
"""

if __package__:
    from .all_hardware_simulator import main
else:
    from all_hardware_simulator import main


if __name__ == "__main__":
    raise SystemExit(main())
