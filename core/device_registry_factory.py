"""设备注册中心工厂 — 为所有模块(API/Agent/Scheduler)提供统一的设备驱动初始化

设备注册原则（真实场景模式）:
  - 不再自动注入内置虚拟设备
  - 每种驱动类型独立初始化，失败时直接跳过（不降级为模拟器）
  - SimulatorDriver 仅当用户显式注册 driver="simulator" 设备时才创建
  - 驱动不可用时（如 paho-mqtt 未安装），对应设备不会出现在列表中
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 项目根目录（本文件位于 <project_root>/core/device_registry_factory.py）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据存储目录：优先使用环境变量 DATA_STORAGE_DIR，否则使用项目根下的 data/
_raw_data_dir = os.getenv("DATA_STORAGE_DIR")
if _raw_data_dir and _raw_data_dir.strip():
    DEFAULT_DATA_DIR = _raw_data_dir if os.path.isabs(_raw_data_dir) else os.path.join(_PROJECT_ROOT, _raw_data_dir)
else:
    DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# 内置虚拟设备 ID 集合 — 保留用于 ID 冲突检测，但不再自动创建
BUILTIN_DEVICE_IDS = {
    "virtual_irrigation_01", "virtual_soil_sensor_01",
    "virtual_ventilation_01", "virtual_light_01",
    "virtual_fertigator_01", "virtual_heater_01",
}

# username 合法字符正则 — 防止路径穿越
_USERNAME_RE = re.compile(r'^[a-zA-Z0-9_\-.@]+$')


def _validate_username(username: str) -> None:
    """验证 username 参数，防止路径穿越攻击"""
    if not username or not isinstance(username, str):
        raise ValueError(f"无效的用户名: {username!r}")
    if ".." in username or "/" in username or "\\" in username:
        raise ValueError(f"用户名包含危险字符: {username!r}")
    if not _USERNAME_RE.match(username):
        raise ValueError(f"用户名包含非法字符: {username!r}")


def _backup_corrupted(path: str):
    """将损坏文件备份，防止数据完全丢失"""
    from datetime import datetime
    try:
        corrupted = path + ".corrupted." + datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.path.exists(path):
            os.rename(path, corrupted)
            logger.info("已保留损坏文件: %s", corrupted)
    except Exception as e:
        logger.warning("备份损坏文件失败: %s", e)


def load_custom_devices(username: str) -> list:
    """加载用户自定义设备配置"""
    _validate_username(username)
    path = os.path.join(DEFAULT_DATA_DIR, username, "custom_devices.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error("custom_devices.json 格式错误(非列表)，保留备份并返回空列表")
            _backup_corrupted(path)
            return []
        return data
    except json.JSONDecodeError:
        logger.error("custom_devices.json JSON 解析失败，尝试从备份恢复")
        bak_path = path + ".bak"
        if os.path.exists(bak_path):
            try:
                with open(bak_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    logger.warning("已从备份恢复 custom_devices.json")
                    save_custom_devices(username, data)
                    return data
            except Exception:
                pass
        _backup_corrupted(path)
        return []
    except Exception as e:
        logger.error("custom_devices.json 加载异常: %s", e)
        _backup_corrupted(path)
        return []


def save_custom_devices(username: str, devices: list) -> None:
    """保存用户自定义设备配置（原子写入）"""
    _validate_username(username)
    if not isinstance(devices, list):
        raise TypeError(f"devices 必须是列表类型，收到: {type(devices)}")
    path = os.path.join(DEFAULT_DATA_DIR, username, "custom_devices.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 原子写入：先写临时文件，再重命名
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(devices, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        # 保留一份备份
        try:
            import shutil
            shutil.copy2(path, path + ".bak")
        except Exception:
            pass
    except Exception as e:
        logger.error("自定义设备保存失败: %s", e)
        raise


def _safe_parse_capabilities(cap_strs: List[str]) -> list:
    """安全解析设备能力，跳过无效值并记录警告"""
    from devices.base import DeviceCapability
    caps = []
    for c in cap_strs:
        try:
            caps.append(DeviceCapability(c))
        except ValueError:
            logger.warning("忽略无效的设备能力: %r", c)
    if not caps:
        logger.warning("设备能力解析结果为空，将不赋予任何能力")
    return caps


def setup_registry(username: str = "default", loop=None):
    """初始化设备注册中心，按驱动类型加载设备。

    真实场景模式:
      - 不再自动注入内置虚拟设备
      - 每种驱动独立初始化，失败时直接跳过（不降级为模拟器）
      - 仅当用户显式注册 driver="simulator" 设备时才创建 SimulatorDriver

    Args:
        username: 用户名
        loop: 可选，外部传入的 event loop。不传则创建新 loop。

    Returns:
        (DeviceDriverRegistry, asyncio.AbstractEventLoop)
    """
    _validate_username(username)

    from devices.registry import DeviceDriverRegistry

    registry = DeviceDriverRegistry()
    created_loop = False
    if loop is None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        created_loop = True

    # ── 加载用户自定义设备，按驱动类型分组 ──
    custom_devices = load_custom_devices(username)
    sim_configs, mqtt_configs, http_configs, modbus_configs, camera_configs = [], [], [], [], []

    for cd in custom_devices:
        driver_type = cd.get("driver", "mqtt")  # 默认 MQTT（最常见的真实 IoT 协议）
        if driver_type == "simulator":
            sim_configs.append(cd)
        elif driver_type == "mqtt":
            mqtt_configs.append(cd)
        elif driver_type == "http":
            http_configs.append(cd)
        elif driver_type == "modbus":
            modbus_configs.append(cd)
        elif driver_type == "camera":
            camera_configs.append(cd)

    # ── Simulator 驱动（仅当用户显式注册了 simulator 设备时创建）──
    if sim_configs:
        from devices.simulator_driver import SimulatorDriver
        sim = SimulatorDriver(simulated_latency_ms=50)
        registry.register("simulator", sim)
        loop.run_until_complete(sim.connect())

        for cd in sim_configs:
            caps = _safe_parse_capabilities(cd.get("capabilities", ["irrigate"]))
            sim.add_virtual_device(
                device_id=cd["device_id"], name=cd["name"],
                capabilities=caps, sensors=cd.get("sensors", []),
                location=cd.get("location", ""),
                initial_state=cd.get("initial_state", {"power": False, "status": "powered_off"}),
            )
        logger.info("SimulatorDriver: 已加载 %d 个用户模拟设备", len(sim_configs))

    # ── MQTT 驱动 ──
    if mqtt_configs:
        try:
            from devices.mqtt_driver import MQTTDriver
        except ImportError:
            logger.warning(
                "MQTT 驱动不可用（paho-mqtt 未安装），%d 个 MQTT 设备将不会加载。"
                "安装: pip install paho-mqtt", len(mqtt_configs)
            )
            mqtt_configs = []  # 清空，不降级为模拟器

    if mqtt_configs:
        try:
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
                    capabilities=_safe_parse_capabilities(cd.get("capabilities", ["irrigate"])),
                    sensors=cd.get("sensors", []), location=cd.get("location", ""),
                    control_topic=conn.get("control_topic", f"devices/{cd['device_id']}/control"),
                    state_topic=conn.get("state_topic"),
                )
            registry.register("mqtt", mqtt_drv)
            connected = loop.run_until_complete(mqtt_drv.connect())
            if not connected:
                logger.info("MQTT 驱动已注册但 Broker 不可达，设备将显示为离线")
            else:
                logger.info("MQTT 驱动已连接: %d 个设备", len(mqtt_configs))
        except Exception as e:
            logger.warning("MQTT 驱动初始化失败: %s，%d 个设备将不可用", e, len(mqtt_configs))

    # ── HTTP 驱动 ──
    if http_configs:
        try:
            from devices.http_driver import HTTPDriver
        except ImportError:
            logger.warning(
                "HTTP 驱动不可用（requests 库异常），%d 个 HTTP 设备将不会加载。",
                len(http_configs)
            )
            http_configs = []

    if http_configs:
        try:
            http_drv = HTTPDriver()
            for cd in http_configs:
                conn = cd.get("connection", {})
                http_drv.register_device(
                    device_id=cd["device_id"], name=cd["name"],
                    capabilities=_safe_parse_capabilities(cd.get("capabilities", ["irrigate"])),
                    sensors=cd.get("sensors", []), location=cd.get("location", ""),
                    base_url=conn.get("base_url", ""), api_key=conn.get("api_key"),
                )
            registry.register("http", http_drv)
            connected = loop.run_until_complete(http_drv.connect())
            if not connected:
                logger.info("HTTP 驱动已注册但设备不可达，设备将显示为离线")
            else:
                logger.info("HTTP 驱动已连接: %d 个设备", len(http_configs))
        except Exception as e:
            logger.warning("HTTP 驱动初始化失败: %s，%d 个设备将不可用", e, len(http_configs))

    # ── Modbus 驱动 ──
    if modbus_configs:
        try:
            from devices.modbus_driver import ModbusDriver
        except ImportError:
            logger.warning(
                "Modbus 驱动不可用（pymodbus 未安装），%d 个 Modbus 设备将不会加载。"
                "安装: pip install pymodbus", len(modbus_configs)
            )
            modbus_configs = []

    if modbus_configs:
        try:
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
                        capabilities=_safe_parse_capabilities(cd.get("capabilities", ["irrigate"])),
                        sensors=cd.get("sensors", []), location=cd.get("location", ""),
                        slave_id=conn.get("slave_id", 1),
                    )
                registry.register(f"modbus_{port}", modbus_drv)
                connected = loop.run_until_complete(modbus_drv.connect())
                if not connected:
                    logger.info("Modbus 驱动已注册但设备不可达 (%s)，设备将显示为离线", port)
                else:
                    logger.info("Modbus 驱动已连接: %d 个设备 @ %s", len(devices_list), port)
        except Exception as e:
            logger.warning("Modbus 驱动初始化失败: %s，%d 个设备将不可用", e, len(modbus_configs))

    # ── 摄像头驱动 ──
    if camera_configs:
        try:
            from devices.camera_driver import CameraDriver
        except ImportError:
            logger.warning(
                "摄像头驱动不可用（opencv-python 未安装），%d 个摄像头设备将不会加载。"
                "安装: pip install opencv-python", len(camera_configs)
            )
            camera_configs = []

    if camera_configs:
        try:
            camera_drv = CameraDriver()
            for cd in camera_configs:
                conn = cd.get("connection", {})
                caps = _safe_parse_capabilities(cd.get("capabilities", ["capture"]))
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
            connected = loop.run_until_complete(camera_drv.connect())
            if not connected:
                logger.info("摄像头驱动已注册但设备不可达，设备将显示为离线")
            else:
                logger.info("摄像头驱动已连接: %d 个设备", len(camera_configs))
        except Exception as e:
            logger.warning("摄像头驱动初始化失败: %s，%d 个设备将不可用", e, len(camera_configs))

    return registry, loop


def close_registry(loop, registry=None):
    """关闭事件循环并清理驱动连接，防止资源泄漏。

    Args:
        loop: 要关闭的 event loop
        registry: 可选，DeviceDriverRegistry 实例。传入时会先断开所有驱动连接。
    """
    # 先断开所有驱动连接（MQTT/Modbus/串口等资源）
    if registry is not None:
        try:
            loop.run_until_complete(registry.disconnect_all())
        except Exception:
            logger.warning("驱动断开失败，继续关闭事件循环")

    # 关闭事件循环（不调用 set_event_loop(None)，避免污染线程）
    try:
        if loop and not loop.is_closed():
            # 取消所有pending任务
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()
    except Exception:
        pass


# ═══════════════════════════════════════════════════
# Registry 缓存 — 避免每次设备操作都重建驱动连接
# ═══════════════════════════════════════════════════
#
# 注意：只缓存 registry 对象（含已连接的驱动），不缓存 event loop。
# asyncio event loop 不是线程安全的，FastAPI 不同请求可能在不同线程处理，
# 跨线程共享 loop 会导致连接断裂。
# 每次 API 请求创建新的轻量 loop，复用已连接的 registry 驱动。

import time as _time
import threading as _threading

# 缓存: {username: (registry, timestamp)}
_registry_cache: Dict[str, tuple] = {}
_cache_lock = _threading.Lock()
_CACHE_TTL_SECONDS = 120  # 缓存2分钟


def get_cached_registry(username: str = "default"):
    """获取缓存的设备注册中心（不含 event loop）。

    驱动连接（HTTP/MQTT/Modbus）在缓存有效期内复用，
    每次调用需自行创建 event loop 来驱动异步操作。

    Returns:
        DeviceDriverRegistry — 已连接驱动的注册中心实例
    """
    now = _time.time()
    with _cache_lock:
        cached = _registry_cache.get(username)
        if cached:
            registry, timestamp = cached
            if (now - timestamp) < _CACHE_TTL_SECONDS:
                return registry
            logger.debug("Registry 缓存过期 (%.0fs)，重建", now - timestamp)
            _registry_cache.pop(username, None)

    # 重建（setup_registry 内部创建 loop 用于初始化连接）
    registry, init_loop = setup_registry(username)
    # 初始化完成后关闭临时 loop，驱动连接保留在 registry 中
    close_registry(init_loop, None)
    with _cache_lock:
        _registry_cache[username] = (registry, now)
    return registry


class RegistrySession:
    """Registry 会话上下文管理器 — 自动管理 event loop 生命周期。

    用法:
        with RegistrySession("123") as (registry, loop):
            loop.run_until_complete(registry.discover_all())
            ...
        # 退出时自动关闭 loop
    """

    def __init__(self, username: str = "default"):
        self.username = username
        self.registry = None
        self.loop = None

    def __enter__(self):
        self.registry = get_cached_registry(self.username)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        return self.registry, self.loop

    def __exit__(self, *args):
        if self.loop and not self.loop.is_closed():
            self.loop.close()
        try:
            asyncio.set_event_loop(None)
        except Exception:
            pass


def invalidate_registry_cache(username: str = None):
    """使指定用户（或所有用户）的 registry 缓存失效。"""
    with _cache_lock:
        if username:
            popped = _registry_cache.pop(username, None)
            if popped:
                registry, _ = popped
                try:
                    # 断开所有驱动连接
                    import asyncio as _asyncio
                    tmp_loop = _asyncio.new_event_loop()
                    _asyncio.set_event_loop(tmp_loop)
                    try:
                        tmp_loop.run_until_complete(registry.disconnect_all())
                    except Exception:
                        pass
                    tmp_loop.close()
                except Exception:
                    pass
        else:
            import asyncio as _asyncio
            for u, (registry, _) in list(_registry_cache.items()):
                try:
                    tmp_loop = _asyncio.new_event_loop()
                    tmp_loop.run_until_complete(registry.disconnect_all())
                    tmp_loop.close()
                except Exception:
                    pass
            _registry_cache.clear()
