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
        "initial_state": json.dumps(d.get("initial_state", {}), ensure_ascii=False),
    } for d in devices]
    repo.replace_all_for_user(user.id, items)

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
                        logger.warning("缓存清理时断开驱动失败", exc_info=True)
                    finally:
                        try:
                            tmp_loop.close()
                        except Exception:
                            pass
                        _asyncio.set_event_loop(None)
                except Exception:
                    pass
        else:
            import asyncio as _asyncio
            for u, (registry, _) in list(_registry_cache.items()):
                tmp_loop = None
                try:
                    tmp_loop = _asyncio.new_event_loop()
                    tmp_loop.run_until_complete(registry.disconnect_all())
                except Exception:
                    logger.warning("缓存清理时断开驱动失败 [%s]", u, exc_info=True)
                finally:
                    if tmp_loop is not None:
                        try:
                            tmp_loop.close()
                        except Exception:
                            pass
            _registry_cache.clear()
