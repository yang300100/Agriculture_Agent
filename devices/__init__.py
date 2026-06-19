"""设备驱动模块 — 统一设备控制接口"""

from .base import (
    BaseDeviceDriver, DeviceCapability, DeviceStatus,
    DeviceInfo, DeviceCommand, DeviceResult, CommandPriority,
)
from .registry import DeviceDriverRegistry
from .simulator_driver import SimulatorDriver

# 可选依赖：MQTT / HTTP / Modbus 驱动
try:
    from .mqtt_driver import MQTTDriver
    _has_mqtt = True
except ImportError:
    MQTTDriver = None
    _has_mqtt = False

try:
    from .http_driver import HTTPDriver
    _has_http = True
except ImportError:
    HTTPDriver = None
    _has_http = False

try:
    from .modbus_driver import ModbusDriver
    _has_modbus = True
except ImportError:
    ModbusDriver = None
    _has_modbus = False

try:
    from .camera_driver import CameraDriver
    _has_camera = True
except ImportError:
    CameraDriver = None
    _has_camera = False

__all__ = [
    "BaseDeviceDriver",
    "DeviceCapability",
    "DeviceStatus",
    "DeviceInfo",
    "DeviceCommand",
    "DeviceResult",
    "CommandPriority",
    "DeviceDriverRegistry",
    "SimulatorDriver",
    "MQTTDriver",
    "HTTPDriver",
    "ModbusDriver",
    "CameraDriver",
]
