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
| `all_hardware_simulator.py` | HTTP/MQTT/Modbus 共享状态服务、终端与网页面板 | `flask`、`paho-mqtt`、`Pillow` |
| `start_all_simulators.py` | 统一模拟器的兼容启动入口 | 同上 |

## 快速开始

```bash
# 1. 安装依赖
pip install paho-mqtt requests flask Pillow opencv-python numpy

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

### 统一硬件模拟器

```powershell
# 默认端口：HTTP 5000、MQTT 1883、Modbus TCP 5020
python hardware_examples/start_all_simulators.py

# 无交互终端时持续运行；两个启动脚本使用同一实现，不要同时启动
python hardware_examples/start_all_simulators.py --headless

# 可选择禁用协议，例如只启用 HTTP 面板与 HTTP 设备
python hardware_examples/start_all_simulators.py --no-mqtt --no-modbus

# Windows 端口被 Hyper-V/WSL 保留时可安全覆盖
$env:HARDWARE_HTTP_PORT = "15000"
$env:HARDWARE_MQTT_PORT = "18883"
$env:HARDWARE_MODBUS_PORT = "15020"
python hardware_examples/all_hardware_simulator.py
```

注册到主系统的设备连接参数必须使用同一组覆盖端口。

默认网页控制面板为 <http://127.0.0.1:5000/>，健康检查为 `/health`。
启动失败会显示具体错误并返回非零退出码；终端输入结束后服务继续运行，按 Ctrl+C 退出。
旧版独立 HTTP/MQTT/Modbus 脚本保留为示例，启动器不再调用它们。

设备分工为 HTTP（灌溉泵、补光灯、施肥泵、摄像头）、MQTT（通风扇、加热器）、Modbus（环境传感器，从站 1）。
模拟器摄像头通过 HTTP `capture` 返回 JPEG，设备中心每 5 秒更新一张快照。默认图源由程序绘制；可显式设置
`HARDWARE_CAMERA_IMAGE` 为本地图片路径，图源不可读时会返回错误。

接入本机摄像头时，在设备中心将对应设备的驱动设为 `camera`，能力选择 `capture`，连接参数填写：

```json
{"camera_type": "usb", "source": "0"}
```

`source` 为本机摄像头索引。设备中心会自动连接连续实时预览（约 10 FPS）；同一后端进程内，
预览、抓拍和状态检测共享一个摄像头采集句柄。离开页面会关闭视频连接，最后一个消费者离开后释放摄像头。
预览与快照刷新不保存照片；手动拍照和巡检抓拍仍按原逻辑保存。视频中断会保留最后画面并标记过期，可点击重新连接。
视频接口为 `GET /api/devices/{device_id}/stream`，沿用账号和 Bearer 认证，返回带长度信息的 multipart JPEG。
本机摄像头由农业助手后端直接读取；硬件模拟器网页中的虚拟摄像头保持独立。

新账号可运行 `python scripts/register_simulated_devices.py --username 123` 注册设备。
若账号已有旧版摄像头配置，先备份数据库，再运行：

```powershell
python scripts/register_simulated_devices.py --username 123 --repair-legacy-camera
```

修复只调整旧脚本在本机 5020 端口、从站 2 的摄像头连接，保留名称、地块和其他设备配置。

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
├── all_hardware_simulator.py     # 统一三协议终端模拟器
└── test_integration.py           # 集成测试
```
