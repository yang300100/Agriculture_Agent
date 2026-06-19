"""设备注册中心工厂 — 为所有模块(API/Agent/Scheduler)提供统一的设备驱动初始化"""

import asyncio
import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

# 项目根目录（本文件位于 <project_root>/core/device_registry_factory.py）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据存储目录：优先使用环境变量 DATA_STORAGE_DIR，否则使用项目根下的 data/
_raw_data_dir = os.getenv("DATA_STORAGE_DIR")
if _raw_data_dir:
    DEFAULT_DATA_DIR = _raw_data_dir if os.path.isabs(_raw_data_dir) else os.path.join(_PROJECT_ROOT, _raw_data_dir)
else:
    DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# 内置虚拟设备 ID 集合 — 防止自定义设备与内置设备 ID 冲突
BUILTIN_DEVICE_IDS = {
    "virtual_irrigation_01", "virtual_soil_sensor_01",
    "virtual_ventilation_01", "virtual_light_01",
    "virtual_fertigator_01", "virtual_heater_01",
}


def load_custom_devices(username: str) -> list:
    """加载用户自定义设备配置"""
    path = os.path.join(DEFAULT_DATA_DIR, username, "custom_devices.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_custom_devices(username: str, devices: list) -> None:
    """保存用户自定义设备配置"""
    path = os.path.join(DEFAULT_DATA_DIR, username, "custom_devices.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(devices, f, ensure_ascii=False, indent=2)


def setup_registry(username: str = "default"):
    """初始化设备注册中心，加载所有驱动和设备。

    Returns:
        (DeviceDriverRegistry, asyncio.AbstractEventLoop)
    """
    from devices.registry import DeviceDriverRegistry
    from devices.simulator_driver import SimulatorDriver
    from devices.base import DeviceCapability

    registry = DeviceDriverRegistry()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # ── 内置虚拟设备 ──
    sim = SimulatorDriver(simulated_latency_ms=50)
    registry.register("simulator", sim)
    loop.run_until_complete(sim.connect())

    # ── 自定义设备按驱动分组 ──
    custom_devices = load_custom_devices(username)
    mqtt_configs, http_configs, modbus_configs, camera_configs = [], [], [], []

    for cd in custom_devices:
        driver_type = cd.get("driver", "simulator")
        caps = [DeviceCapability(c) for c in cd.get("capabilities", ["irrigate"])]

        if driver_type == "simulator":
            sim.add_virtual_device(
                device_id=cd["device_id"], name=cd["name"],
                capabilities=caps, sensors=cd.get("sensors", []),
                location=cd.get("location", ""),
                initial_state=cd.get("initial_state", {"power": False, "status": "idle"}),
            )
        elif driver_type == "mqtt":
            mqtt_configs.append(cd)
        elif driver_type == "http":
            http_configs.append(cd)
        elif driver_type == "modbus":
            modbus_configs.append(cd)
        elif driver_type == "camera":
            camera_configs.append(cd)

    # ── MQTT 驱动 ──
    if mqtt_configs:
        try:
            from devices.mqtt_driver import MQTTDriver
            first = mqtt_configs[0]
            conn = first.get("connection", {})
            mqtt_drv = MQTTDriver(
                broker_host=conn.get("host", "localhost"),
                broker_port=conn.get("port", 1883),
            )
            for cd in mqtt_configs:
                conn = cd.get("connection", {})
                mqtt_drv.register_device(
                    device_id=cd["device_id"], name=cd["name"],
                    capabilities=[DeviceCapability(c) for c in cd.get("capabilities", ["irrigate"])],
                    sensors=cd.get("sensors", []), location=cd.get("location", ""),
                    control_topic=conn.get("control_topic", f"devices/{cd['device_id']}/control"),
                    state_topic=conn.get("state_topic"),
                )
            registry.register("mqtt", mqtt_drv)
            loop.run_until_complete(mqtt_drv.connect())
        except Exception as e:
            logger.warning("MQTT 驱动注册失败，设备将以错误状态显示: %s", e)
            # 降级：将设备注册为 simulator 虚拟设备，标记错误原因
            for cd in mqtt_configs:
                caps = [DeviceCapability(c) for c in cd.get("capabilities", ["irrigate"])]
                err_msg = "paho-mqtt 未安装" if "No module named" in str(e) else str(e)[:120]
                conn = cd.get("connection", {})
                sim.add_virtual_device(
                    device_id=cd["device_id"], name=cd["name"],
                    capabilities=caps, sensors=cd.get("sensors", []),
                    location=cd.get("location", ""),
                    initial_state={
                        "power": False, "status": "error",
                        "error_reason": f"MQTT 驱动不可用: {err_msg}",
                        "original_driver": "mqtt",
                        "mqtt_host": conn.get("host", "?"),
                        "mqtt_port": conn.get("port", "?"),
                    },
                )

    # ── HTTP 驱动 ──
    if http_configs:
        try:
            from devices.http_driver import HTTPDriver
            http_drv = HTTPDriver()
            for cd in http_configs:
                conn = cd.get("connection", {})
                http_drv.register_device(
                    device_id=cd["device_id"], name=cd["name"],
                    capabilities=[DeviceCapability(c) for c in cd.get("capabilities", ["irrigate"])],
                    sensors=cd.get("sensors", []), location=cd.get("location", ""),
                    base_url=conn.get("base_url", ""), api_key=conn.get("api_key"),
                )
            registry.register("http", http_drv)
            loop.run_until_complete(http_drv.connect())
        except Exception as e:
            logger.warning("HTTP 驱动注册失败，设备将以错误状态显示: %s", e)
            for cd in http_configs:
                caps = [DeviceCapability(c) for c in cd.get("capabilities", ["irrigate"])]
                err_msg = "requests 库异常" if "No module named" in str(e) else str(e)[:120]
                conn = cd.get("connection", {})
                sim.add_virtual_device(
                    device_id=cd["device_id"], name=cd["name"],
                    capabilities=caps, sensors=cd.get("sensors", []),
                    location=cd.get("location", ""),
                    initial_state={
                        "power": False, "status": "error",
                        "error_reason": f"HTTP 驱动不可用: {err_msg}",
                        "original_driver": "http",
                        "http_url": conn.get("base_url", "?"),
                    },
                )

    # ── Modbus 驱动 ──
    if modbus_configs:
        try:
            from devices.modbus_driver import ModbusDriver
            # 按 port 分组，同端口设备共享一个驱动实例
            port_groups: Dict[str, List[dict]] = {}
            for cd in modbus_configs:
                conn = cd.get("connection", {})
                port = conn.get("port", "/dev/ttyUSB0")
                if port not in port_groups:
                    port_groups[port] = []
                port_groups[port].append(cd)

            for port, devices_list in port_groups.items():
                first_dev = devices_list[0]
                conn = first_dev.get("connection", {})
                modbus_drv = ModbusDriver(
                    mode=conn.get("mode", "rtu"),
                    port=port,
                )
                for cd in devices_list:
                    conn = cd.get("connection", {})
                    modbus_drv.register_device(
                        device_id=cd["device_id"], name=cd["name"],
                        capabilities=[DeviceCapability(c) for c in cd.get("capabilities", ["irrigate"])],
                        sensors=cd.get("sensors", []), location=cd.get("location", ""),
                        slave_id=conn.get("slave_id", 1),
                    )
                registry.register(f"modbus_{port}", modbus_drv)
                loop.run_until_complete(modbus_drv.connect())
        except Exception as e:
            logger.warning("Modbus 驱动注册失败，设备将以错误状态显示: %s", e)
            for cd in modbus_configs:
                caps = [DeviceCapability(c) for c in cd.get("capabilities", ["irrigate"])]
                err_msg = "pymodbus 未安装" if "No module named" in str(e) else str(e)[:120]
                conn = cd.get("connection", {})
                sim.add_virtual_device(
                    device_id=cd["device_id"], name=cd["name"],
                    capabilities=caps, sensors=cd.get("sensors", []),
                    location=cd.get("location", ""),
                    initial_state={
                        "power": False, "status": "error",
                        "error_reason": f"Modbus 驱动不可用: {err_msg}",
                        "original_driver": "modbus",
                        "modbus_mode": conn.get("mode", "?"),
                        "modbus_port": conn.get("port", "?"),
                        "slave_id": conn.get("slave_id", "?"),
                    },
                )

    # ── 摄像头驱动 ──
    if camera_configs:
        try:
            from devices.camera_driver import CameraDriver
            camera_drv = CameraDriver()
            for cd in camera_configs:
                conn = cd.get("connection", {})
                caps = [DeviceCapability(c) for c in cd.get("capabilities", ["capture"])]
                camera_drv.register_device(
                    device_id=cd["device_id"], name=cd["name"],
                    capabilities=caps, sensors=cd.get("sensors", []),
                    location=cd.get("location", ""),
                    camera_type=conn.get("camera_type", "usb"),
                    source=conn.get("source", "0"),
                    username=conn.get("username", ""),
                    password=conn.get("password", ""),
                )
            registry.register("camera", camera_drv)
            loop.run_until_complete(camera_drv.connect())
        except Exception as e:
            logger.warning("摄像头驱动注册失败，设备将以错误状态显示: %s", e)
            for cd in camera_configs:
                caps = [DeviceCapability(c) for c in cd.get("capabilities", ["capture"])]
                err_msg = "opencv-python 未安装" if "No module named" in str(e) else str(e)[:120]
                conn = cd.get("connection", {})
                sim.add_virtual_device(
                    device_id=cd["device_id"], name=cd["name"],
                    capabilities=caps, sensors=cd.get("sensors", []),
                    location=cd.get("location", ""),
                    initial_state={
                        "power": False, "status": "error",
                        "error_reason": f"摄像头驱动不可用: {err_msg}",
                        "original_driver": "camera",
                        "camera_type": conn.get("camera_type", "?"),
                        "source": conn.get("source", "?"),
                    },
                )

    return registry, loop


def close_registry(loop):
    """关闭与 registry 关联的事件循环，防止泄漏"""
    try:
        if loop and not loop.is_closed():
            loop.close()
    except Exception:
        pass
    finally:
        # 恢复默认事件循环策略，避免后续代码拿到已关闭的 loop
        try:
            asyncio.set_event_loop(None)
        except Exception:
            pass
