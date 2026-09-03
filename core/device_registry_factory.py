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
from typing import Dict, List

from core.storage_paths import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# 内置虚拟设备 ID 集合 — 保留用于 ID 冲突检测，但不再自动创建
BUILTIN_DEVICE_IDS = {
    "virtual_irrigation_01", "virtual_soil_sensor_01",
    "virtual_ventilation_01", "virtual_light_01",
    "virtual_fertigator_01", "virtual_heater_01",
}

SUPPORTED_DEVICE_DRIVERS = {
    "simulator", "mqtt", "http", "modbus", "coap", "opcua", "camera",
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
    """加载用户自定义设备配置（纯DB）"""
    _validate_username(username)
    from core.database.repository.devices import DeviceConfigRepository
    from core.database.repository.users import UserRepository
    user_repo = UserRepository()
    user = user_repo.get_by_username(username)
    if not user:
        return []
    repo = DeviceConfigRepository()
    configs = repo.find_by(user_id=user.id)
    result = []
    for c in configs:
        result.append({
            "device_id": c.device_id,
            "name": c.name,
            "driver": c.driver,
            "capabilities": json.loads(c.capabilities) if c.capabilities else [],
            "sensors": json.loads(c.sensors) if c.sensors else [],
            "connection": json.loads(c.connection) if c.connection else {},
            "location": c.location or "",
            "plot_id": c.plot_id,
            "zone_id": c.zone_id or "",
            "initial_state": json.loads(c.initial_state) if c.initial_state else {},
        })
    return result

def save_custom_devices(username: str, devices: list) -> None:
    """保存用户自定义设备配置（纯DB）"""
    _validate_username(username)
    if not isinstance(devices, list):
        raise TypeError(f"devices 必须是列表类型，收到: {type(devices)}")
    from core.database.repository.devices import DeviceConfigRepository
    from core.database.repository.users import UserRepository
    user_repo = UserRepository()
    user = user_repo.get_by_username(username)
    if not user:
        user = user_repo.create(username=username, password_hash="")
    repo = DeviceConfigRepository()
    items = [{
        "device_id": d.get("device_id", ""),
        "name": d.get("name", ""),
        "driver": d.get("driver", "simulator"),
        "capabilities": json.dumps(d.get("capabilities", []), ensure_ascii=False),
        "sensors": json.dumps(d.get("sensors", []), ensure_ascii=False),
        "connection": json.dumps(d.get("connection", {}), ensure_ascii=False),
        "location": d.get("location", ""),
        "plot_id": d.get("plot_id"),
        "zone_id": d.get("zone_id") or None,
        "initial_state": json.dumps(d.get("initial_state", {}), ensure_ascii=False),
    } for d in devices]
    repo.replace_all_for_user(user.id, items)


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
    """初始化设备注册中心，按驱动类型加载设备。"""
    _validate_username(username)
    from devices.registry import DeviceDriverRegistry

    registry = DeviceDriverRegistry()
    if loop is None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    custom_devices = load_custom_devices(username)
    sim_configs, mqtt_configs, http_configs, modbus_configs = [], [], [], []
    coap_configs, opcua_configs, camera_configs = [], [], []

    for cd in custom_devices:
        driver_type = str(cd.get("driver", "mqtt")).lower()
        if driver_type == "simulator":
            sim_configs.append(cd)
        elif driver_type == "mqtt":
            mqtt_configs.append(cd)
        elif driver_type == "http":
            http_configs.append(cd)
        elif driver_type == "modbus":
            modbus_configs.append(cd)
        elif driver_type == "coap":
            coap_configs.append(cd)
        elif driver_type == "opcua":
            opcua_configs.append(cd)
        elif driver_type == "camera":
            camera_configs.append(cd)
        else:
            logger.warning(
                "设备 %s 使用未知驱动 %r，已跳过",
                cd.get("device_id", "<unknown>"),
                driver_type,
            )

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

    if mqtt_configs:
        try:
            from devices.mqtt_driver import MQTTDriver
        except ImportError:
            logger.warning("MQTT 驱动不可用（paho-mqtt 未安装），%d 个 MQTT 设备将不会加载", len(mqtt_configs))
            mqtt_configs = []
    if mqtt_configs:
        try:
            broker_groups: Dict[tuple, List[dict]] = {}
            for cd in mqtt_configs:
                conn = cd.get("connection", {})
                key = (
                    str(conn.get("host", "localhost")),
                    int(conn.get("port", 1883)),
                    conn.get("username"), conn.get("password"),
                    bool(conn.get("use_tls", False)), conn.get("ca_cert"),
                    conn.get("client_cert"), conn.get("client_key"),
                    bool(conn.get("tls_insecure", False)), conn.get("client_id"),
                )
                broker_groups.setdefault(key, []).append(cd)
            for index, (key, devices_list) in enumerate(broker_groups.items(), start=1):
                (
                    host, port, username, password, use_tls, ca_cert,
                    client_cert, client_key, tls_insecure, client_id,
                ) = key
                mqtt_drv = MQTTDriver(
                    broker_host=host, broker_port=port,
                    username=username, password=password, client_id=client_id,
                    use_tls=use_tls, ca_cert=ca_cert,
                    client_cert=client_cert, client_key=client_key,
                    tls_insecure=tls_insecure,
                )
                for cd in devices_list:
                    conn = cd.get("connection", {})
                    mqtt_drv.register_device(
                        device_id=cd["device_id"], name=cd["name"],
                        capabilities=_safe_parse_capabilities(cd.get("capabilities", ["irrigate"])),
                        sensors=cd.get("sensors", []), location=cd.get("location", ""),
                        control_topic=conn.get("control_topic", f"devices/{cd['device_id']}/control"),
                        state_topic=conn.get("state_topic"),
                        qos=conn.get("qos", 0),
                    )
                registry.register(f"mqtt_{index}", mqtt_drv)
                connected = loop.run_until_complete(mqtt_drv.connect())
                if not connected:
                    logger.info("MQTT 驱动已注册但 Broker %s:%s 不可达", host, port)
                else:
                    logger.info("MQTT 驱动已连接: %d 个设备 @ %s:%s", len(devices_list), host, port)
        except Exception as e:
            logger.warning("MQTT 驱动初始化失败: %s，%d 个设备将不可用", e, len(mqtt_configs))

    if http_configs:
        try:
            from devices.http_driver import HTTPDriver
        except ImportError:
            logger.warning("HTTP 驱动不可用，%d 个 HTTP 设备将不会加载", len(http_configs))
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

    if modbus_configs:
        try:
            from devices.modbus_driver import ModbusDriver
        except ImportError:
            logger.warning("Modbus 驱动不可用（pymodbus 未安装），%d 个 Modbus 设备将不会加载", len(modbus_configs))
            modbus_configs = []
    if modbus_configs:
        try:
            port_groups: Dict[tuple, List[dict]] = {}
            for cd in modbus_configs:
                conn = cd.get("connection", {})
                mode = str(conn.get("mode", "rtu")).lower()
                # 兼容 int/str 类型的 port，TCP 模式组合为 "host:port"
                if mode == "tcp":
                    raw_port = conn.get("port", 502)
                    configured_host = conn.get("host")
                    if configured_host:
                        host = str(configured_host)
                        # IPv6 地址需要用方括号包裹: [::1]:502
                        if ":" in host and not host.startswith("["):
                            port = f"[{host}]:{raw_port}"
                        else:
                            port = f"{host}:{raw_port}"
                    elif isinstance(raw_port, str) and ":" in raw_port:
                        # 前端兼容格式：port 字段直接保存 "host:port"。
                        port = raw_port
                    else:
                        port = f"127.0.0.1:{raw_port}"
                else:
                    port = str(conn.get("port", "/dev/ttyUSB0"))
                group_key = (
                    mode,
                    port,
                    int(conn.get("baudrate", 9600)),
                    float(conn.get("timeout", 2.0)),
                )
                port_groups.setdefault(group_key, []).append(cd)
            for index, (group_key, devices_list) in enumerate(port_groups.items(), start=1):
                mode, port, baudrate, timeout = group_key
                modbus_drv = ModbusDriver(
                    mode=mode, port=port, baudrate=baudrate, timeout=timeout
                )
                for cd in devices_list:
                    conn = cd.get("connection", {})
                    modbus_drv.register_device(
                        device_id=cd["device_id"], name=cd["name"],
                        capabilities=_safe_parse_capabilities(cd.get("capabilities", ["irrigate"])),
                        sensors=cd.get("sensors", []), location=cd.get("location", ""),
                        slave_id=conn.get("slave_id", 1),
                    )
                registry.register(f"modbus_{index}", modbus_drv)
                connected = loop.run_until_complete(modbus_drv.connect())
                if not connected:
                    logger.info("Modbus 驱动已注册但设备不可达 (%s)，设备将显示为离线", port)
                else:
                    logger.info("Modbus 驱动已连接: %d 个设备 @ %s", len(devices_list), port)
        except Exception as e:
            logger.warning("Modbus 驱动初始化失败: %s，%d 个设备将不可用", e, len(modbus_configs))

    if coap_configs:
        try:
            from devices.coap_driver import CoAPDriver
            coap_drv = CoAPDriver()
            for cd in coap_configs:
                conn = cd.get("connection", {})
                coap_drv.register_device(
                    device_id=cd["device_id"], name=cd["name"],
                    capabilities=_safe_parse_capabilities(cd.get("capabilities", ["read_sensor"])),
                    sensors=cd.get("sensors", []), location=cd.get("location", ""),
                    base_uri=conn.get("base_uri", ""),
                    command_path=conn.get("command_path", "/command"),
                    state_path=conn.get("state_path", "/state"),
                    auth_token=conn.get("auth_token"),
                )
            registry.register("coap", coap_drv)
            connected = loop.run_until_complete(coap_drv.connect())
            logger.info("CoAP 驱动已加载: %d 个设备，在线=%s", len(coap_configs), connected)
        except Exception as e:
            logger.warning("CoAP 驱动初始化失败: %s，%d 个设备将不可用", e, len(coap_configs))

    if opcua_configs:
        try:
            from devices.opcua_driver import OPCUADriver
            opcua_drv = OPCUADriver()
            for cd in opcua_configs:
                conn = cd.get("connection", {})
                opcua_drv.register_device(
                    device_id=cd["device_id"], name=cd["name"],
                    capabilities=_safe_parse_capabilities(cd.get("capabilities", ["read_sensor"])),
                    sensors=cd.get("sensors", []), location=cd.get("location", ""),
                    endpoint=conn.get("endpoint", ""),
                    command_nodes=conn.get("command_nodes", {}),
                    state_nodes=conn.get("state_nodes", {}),
                    username=conn.get("username"), password=conn.get("password"),
                    security_string=conn.get("security_string"),
                )
            registry.register("opcua", opcua_drv)
            connected = loop.run_until_complete(opcua_drv.connect())
            logger.info("OPC UA 驱动已加载: %d 个设备，在线=%s", len(opcua_configs), connected)
        except Exception as e:
            logger.warning("OPC UA 驱动初始化失败: %s，%d 个设备将不可用", e, len(opcua_configs))

    if camera_configs:
        try:
            from devices.camera_driver import CameraDriver
        except ImportError:
            logger.warning("摄像头驱动不可用（opencv-python 未安装），%d 个摄像头设备将不会加载", len(camera_configs))
            camera_configs = []
    if camera_configs:
        try:
            camera_drv = CameraDriver(username=username)
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
    """关闭事件循环并清理驱动连接，防止资源泄漏。"""
    if registry is not None:
        try:
            loop.run_until_complete(registry.disconnect_all())
        except Exception:
            logger.warning("驱动断开失败，继续关闭事件循环")
    try:
        if loop and not loop.is_closed():
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


# Registry 缓存
import time as _time
import threading as _threading

_registry_cache: Dict[str, tuple] = {}
_cache_lock = _threading.Lock()
_CACHE_TTL_SECONDS = 120


def _disconnect_cached_registry(registry) -> None:
    """同步释放缓存驱动，避免 MQTT 后台线程和串口句柄泄漏。"""
    tmp_loop = asyncio.new_event_loop()
    try:
        tmp_loop.run_until_complete(registry.disconnect_all())
    except Exception:
        logger.warning("缓存 Registry 断开失败", exc_info=True)
    finally:
        tmp_loop.close()

def get_cached_registry(username: str = "default"):
    """获取缓存的设备注册中心（不含 event loop）。

    驱动连接（HTTP/MQTT/Modbus）在缓存有效期内复用，
    每次调用需自行创建 event loop 来驱动异步操作。

    Returns:
        DeviceDriverRegistry — 已连接驱动的注册中心实例
    """
    now = _time.time()
    expired_registry = None
    with _cache_lock:
        cached = _registry_cache.get(username)
        if cached:
            registry, timestamp = cached
            if (now - timestamp) < _CACHE_TTL_SECONDS:
                return registry
            logger.debug("Registry 缓存过期 (%.0fs)，重建", now - timestamp)
            _registry_cache.pop(username, None)
            expired_registry = registry

    if expired_registry is not None:
        _disconnect_cached_registry(expired_registry)

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
                    _disconnect_cached_registry(registry)
                except Exception:
                    pass
        else:
            for u, (registry, _) in list(_registry_cache.items()):
                try:
                    _disconnect_cached_registry(registry)
                except Exception:
                    logger.warning("缓存清理时断开驱动失败 [%s]", u, exc_info=True)
            _registry_cache.clear()
