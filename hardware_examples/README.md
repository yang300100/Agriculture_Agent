# 硬件联动示例代码

本文件夹包含各硬件连接方式的**可独立运行**示例代码，可直接运行测试软硬件联动。

## 目录

| 文件 | 说明 | 硬件依赖 |
|------|------|----------|
| `mqtt_sensor_node.py` | MQTT 传感器模拟节点 | `paho-mqtt` |
| `mqtt_relay_controller.py` | MQTT 继电器控制器 | `paho-mqtt` |
| `modbus_sensor_controller.py` | Modbus RTU 传感器+阀门 | `pymodbus` (*可选*) |
| `http_device_server.py` | HTTP 智能设备服务端 | `flask` |
| `http_device_client.py` | HTTP 设备客户端调用 | `requests` |
| `camera_capture_test.py` | 摄像头拍照测试 | `opencv-python` |
| `custom_devices_template.json` | 设备配置JSON模板 | 无 |
| `test_integration.py` | 软硬件联动集成测试 | 见文件注释 |

## 快速开始

```bash
# 1. 安装依赖
pip install paho-mqtt requests flask opencv-python numpy

# 2. 启动后端
python app/start.py

# 3. 运行集成测试（使用内置虚拟设备）
python hardware_examples/test_integration.py

# 4. 可选：启动 MQTT broker + 传感器节点
# 终端1: 启动 Mosquitto MQTT broker
mosquitto -v
# 终端2: 启动模拟传感器
python hardware_examples/mqtt_sensor_node.py
```

## 文件树

```
hardware_examples/
├── README.md
├── mqtt_sensor_node.py           # MQTT 传感器模拟
├── mqtt_relay_controller.py      # MQTT 继电器控制
├── modbus_sensor_controller.py   # Modbus RTU 传感器+阀门
├── http_device_server.py         # HTTP 设备服务端
├── http_device_client.py         # HTTP 设备客户端
├── camera_capture_test.py        # 摄像头拍照测试
├── custom_devices_template.json  # 设备配置模板
└── test_integration.py           # 集成测试
```
