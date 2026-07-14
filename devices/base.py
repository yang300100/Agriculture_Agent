"""设备抽象基类：所有设备驱动必须实现此接口"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


class DeviceCapability(Enum):
    """设备能力枚举 — 描述设备能做什么"""
    IRRIGATE    = "irrigate"      # 灌溉（阀门/水泵）
    FERTIGATE   = "fertigate"     # 施肥
    VENTILATE   = "ventilate"     # 通风（风机/天窗）
    HEAT        = "heat"          # 加热
    COOL        = "cool"          # 降温（湿帘）
    SHADE       = "shade"         # 遮阳
    LIGHT       = "light"         # 补光
    READ_SENSOR = "read_sensor"   # 传感器读数
    CAPTURE     = "capture"       # 摄像头拍摄


class DeviceStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    BUSY = "busy"


@dataclass
class DeviceInfo:
    """设备元信息"""
    device_id: str
    name: str
    driver_name: str              # 驱动标识，如 "simulator" | "mqtt" | "modbus" | "tuya"
    capabilities: List[DeviceCapability]
    sensors: List[str] = field(default_factory=list)
    status: DeviceStatus = DeviceStatus.ONLINE
    location: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """校验必填字段"""
        if not self.device_id or not isinstance(self.device_id, str) or not self.device_id.strip():
            raise ValueError(f"device_id 不能为空: {self.device_id!r}")
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"设备名称不能为空: {self.name!r}")
        if not self.driver_name or self.driver_name == "base":
            # driver_name 不应使用默认的 "base"，提醒开发者覆写
            import logging
            logging.getLogger(__name__).warning(
                "设备 %s 的 driver_name 仍为默认值 'base'，请确认驱动已正确覆写", self.device_id)


class CommandPriority(Enum):
    """指令优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EMERGENCY = "emergency"


@dataclass
class DeviceCommand:
    """下发给设备的指令"""
    command: str                  # "start" | "stop" | "set_param"
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 30000
    priority: CommandPriority = CommandPriority.NORMAL


@dataclass
class DeviceResult:
    """设备执行结果"""
    success: bool
    device_id: str
    executed_command: str
    actual_params: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error_code: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """保证 success/error_code 一致性"""
        if self.success and self.error_code:
            import logging
            logging.getLogger(__name__).warning(
                "DeviceResult 不一致: success=True 但 error_code=%r，已清除 error_code",
                self.error_code)
            self.error_code = None


class BaseDeviceDriver(ABC):
    """设备驱动抽象基类 — 所有协议驱动必须继承此类"""

    driver_name: str = "base"

    @abstractmethod
    async def connect(self) -> bool:
        """建立连接，返回 True 表示成功"""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        ...

    def disconnect_sync(self) -> None:
        """同步断开连接（用于 register/unregister 等同步方法中清理资源）。
        默认实现用 asyncio.run() 桥接异步 disconnect；
        如果已有运行中的事件循环则只打 warning。
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            logger = __import__("logging").getLogger(__name__)
            logger.warning(
                "disconnect_sync: 检测到运行中的事件循环，"
                "无法通过 asyncio.run 清理驱动 %s，请手动调用 disconnect()",
                self.driver_name,
            )
            return

        try:
            asyncio.run(self.disconnect())
        except Exception as e:
            logger = __import__("logging").getLogger(__name__)
            logger.warning("disconnect_sync 断开驱动 %s 失败: %s", self.driver_name, e)

    @abstractmethod
    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        """向指定设备发送指令并返回结果"""
        ...

    @abstractmethod
    async def read_state(self, device_id: str) -> Dict[str, Any]:
        """读取设备当前状态/传感器数据"""
        ...

    @abstractmethod
    async def discover(self) -> List[DeviceInfo]:
        """发现该驱动管理的所有设备"""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """检查驱动/连接健康状态"""
        ...
