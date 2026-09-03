"""MQTT 设备驱动 — 通过 MQTT 协议与真实 IoT 设备通信

依赖: pip install paho-mqtt

使用方式:
    driver = MQTTDriver(broker_host="192.168.1.100", broker_port=1883)
    driver.register_device("pump_01", "水泵#1", [...], control_topic="greenhouse/pump/control")
    await driver.connect()
    result = await driver.execute("pump_01", DeviceCommand("start", {"duration": 30}))
"""

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base import (
    BaseDeviceDriver, DeviceCapability, DeviceStatus,
    DeviceInfo, DeviceCommand, DeviceResult,
)

logger = logging.getLogger(__name__)

# paho-mqtt 是可选依赖
try:
    import paho.mqtt.client as mqtt
    HAS_PAHO = True
except ImportError:
    HAS_PAHO = False
    logger.warning("paho-mqtt 未安装，MQTT 驱动不可用。安装: pip install paho-mqtt")


class MQTTDriver(BaseDeviceDriver):
    """MQTT 设备驱动 — 通过 MQTT Broker 控制设备

    连接方式:
    - 设备通过 MQTT Broker 收发消息
    - 控制指令发布到设备的 control_topic (JSON payload)
    - 状态通过订阅 state_topic 获取

    设备端要求:
    - 订阅 control_topic 接收指令
    - 指令格式: {"command": "start", "params": {...}, "timestamp": "..."}
    - 状态发布到 state_topic (可选)
    """

    driver_name = "mqtt"

    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883,
                 username: str = None, password: str = None,
                 client_id: str = None, use_tls: bool = False,
                 ca_cert: str = None, client_cert: str = None,
                 client_key: str = None, tls_insecure: bool = False):
        if not HAS_PAHO:
            raise ImportError("paho-mqtt 未安装。请运行: pip install paho-mqtt")

        self._broker_host = broker_host
        self._broker_port = int(broker_port)
        self._username = username
        self._password = password
        self._client_id = client_id or f"agri_agent_{uuid.uuid4().hex[:10]}"
        self._use_tls = use_tls
        self._ca_cert = ca_cert
        self._client_cert = client_cert
        self._client_key = client_key
        self._tls_insecure = tls_insecure

        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._connect_event: Optional[threading.Event] = None
        self._connect_rc: Optional[int] = None
        self._devices: Dict[str, Dict] = {}  # device_id → {info, state}
        self._state_cache: Dict[str, Dict] = {}  # device_id → latest state from state_topic
        self._topic_to_device: Dict[str, str] = {}  # topic → device_id 反向映射，O(1) 查找
        self._event_loop = None
        self._state_lock = threading.Lock()  # 保护 _state_cache 和 _devices state 的并发访问

    # ── 设备注册 ──────────────────────────────

    def register_device(self, device_id: str, name: str,
                        capabilities: List[DeviceCapability],
                        sensors: List[str] = None,
                        location: str = "",
                        control_topic: str = None,
                        state_topic: str = None,
                        qos: int = 0) -> None:
        """注册一个 MQTT 设备

        Args:
            device_id: 设备唯一标识
            name: 设备名称
            capabilities: 设备能力列表
            sensors: 传感器字段列表
            location: 物理位置
            control_topic: MQTT 控制主题（发布指令到此主题）
            state_topic: MQTT 状态主题（订阅此主题获取状态，可选）
        """
        # 校验 device_id 不能包含 MQTT 通配符
        if "+" in device_id or "#" in device_id:
            logger.warning("MQTT 设备 ID '%s' 包含通配符 '+' 或 '#'，已拒绝注册", device_id)
            raise ValueError(f"设备 ID 不能包含 MQTT 通配符 '+' 或 '#': {device_id}")

        state_topic = state_topic or f"devices/{device_id}/state"
        control_topic = control_topic or f"devices/{device_id}/control"
        qos = int(qos)
        if qos not in (0, 1, 2):
            raise ValueError(f"MQTT QoS 必须为 0、1 或 2，收到: {qos}")
        existing_device = self._topic_to_device.get(state_topic)
        if existing_device and existing_device != device_id:
            raise ValueError(
                f"状态主题 '{state_topic}' 已被设备 '{existing_device}' 使用"
            )

        self._devices[device_id] = {
            "info": {
                "device_id": device_id,
                "name": name,
                "capabilities": capabilities,
                "sensors": sensors or [],
                "location": location,
                "control_topic": control_topic,
                "state_topic": state_topic,
                "qos": qos,
            },
            "state": {"power": False, "status": "powered_off"},
        }
        self._state_cache[device_id] = {"power": False, "status": "powered_off"}
        # 建立 topic → device_id 反向映射，用于 O(1) 消息分发
        self._topic_to_device[state_topic] = device_id
        logger.info("MQTT 设备已注册: %s → %s", device_id, control_topic)

        # 如果已经连接，自动订阅新设备的状态主题
        if self._connected and self._client is not None:
            try:
                self._client.subscribe(state_topic, qos=qos)
                logger.info("MQTT 自动订阅新设备状态: %s → %s", device_id, state_topic)
            except Exception as e:
                logger.warning("MQTT 自动订阅失败 %s: %s", device_id, e)

    # ── 生命周期 ──────────────────────────────

    async def connect(self) -> bool:
        """连接 MQTT Broker"""
        if self._connected:
            return True

        try:
            client_kwargs = {
                "client_id": self._client_id,
                "protocol": mqtt.MQTTv311,
            }
            # 同时兼容 paho-mqtt 1.x 与 2.x。
            if hasattr(mqtt, "CallbackAPIVersion"):
                client_kwargs["callback_api_version"] = mqtt.CallbackAPIVersion.VERSION1
            self._client = mqtt.Client(**client_kwargs)
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
            self._client.on_disconnect = self._on_disconnect

            if self._username:
                self._client.username_pw_set(self._username, self._password)

            if self._use_tls:
                self._client.tls_set(
                    ca_certs=self._ca_cert,
                    certfile=self._client_cert,
                    keyfile=self._client_key,
                )
                self._client.tls_insecure_set(self._tls_insecure)

            # 使用 threading.Event（非 asyncio.Event）避免 Windows 跨线程唤醒问题
            self._connect_event = threading.Event()
            self._connect_rc = None
            self._client.connect_async(self._broker_host, self._broker_port, keepalive=60)
            self._client.loop_start()

            # 等待连接确认：用 to_thread 避免阻塞事件循环
            connected = await asyncio.to_thread(
                self._connect_event.wait, 3.0  # 3 秒超时
            )
            if not connected or not self._connected:
                if connected:
                    logger.warning(
                        "MQTT Broker 拒绝连接 (%s:%d, rc=%s)",
                        self._broker_host,
                        self._broker_port,
                        self._connect_rc,
                    )
                else:
                    logger.debug("MQTT 连接超时 (%s:%d)", self._broker_host, self._broker_port)
                self._connected = False
                self._client.loop_stop()
                self._client.disconnect()
                self._client = None
                return False

            logger.info("MQTTDriver: 已连接到 %s:%d (%d 设备)",
                       self._broker_host, self._broker_port, len(self._devices))
            return True

        except Exception as e:
            logger.error("MQTT 连接失败: %s", e)
            self._connected = False
            # 确保 loop_stop() 总是被调用，避免后台线程泄露
            if self._client is not None:
                try:
                    self._client.loop_stop()
                except Exception:
                    pass
                try:
                    self._client.disconnect()
                except Exception:
                    pass
                self._client = None
            return False

    async def disconnect(self) -> None:
        """断开 MQTT 连接"""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
        self._connected = False
        logger.info("MQTTDriver: 已断开")

    async def health_check(self) -> bool:
        return self._connected and self._client is not None

    # ── 设备发现 ──────────────────────────────

    async def discover(self) -> List[DeviceInfo]:
        """返回所有注册的设备"""
        result = []
        for dev_id, dev in self._devices.items():
            info = dev["info"]
            state = dev["state"]
            status = DeviceStatus.ONLINE if self._connected else DeviceStatus.OFFLINE
            if state.get("status") == "error":
                status = DeviceStatus.ERROR

            result.append(DeviceInfo(
                device_id=info["device_id"],
                name=info["name"],
                driver_name=self.driver_name,
                capabilities=info["capabilities"],
                sensors=info["sensors"],
                status=status,
                location=info.get("location", ""),
                metadata={
                    "mqtt_broker": f"{self._broker_host}:{self._broker_port}",
                    "control_topic": info.get("control_topic"),
                    "state_topic": info.get("state_topic"),
                },
            ))
        return result

    # ── 指令执行 ──────────────────────────────

    async def execute(self, device_id: str, command: DeviceCommand) -> DeviceResult:
        """通过 MQTT 发布控制指令"""
        if device_id not in self._devices:
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message=f"设备 '{device_id}' 未注册",
                error_code="DEVICE_NOT_FOUND",
            )

        if not self._connected:
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message="MQTT Broker 未连接",
                error_code="NOT_CONNECTED",
            )

        try:
            dev = self._devices[device_id]
            topic = dev["info"]["control_topic"]

            # 构建 MQTT 消息
            payload = {
                "command": command.command,
                "params": command.params,
                "timestamp": datetime.now().isoformat(),
                "device_id": device_id,
            }

            # 默认 QoS 0 兼容内嵌 Broker，真实 Broker 可按设备配置 1/2。
            qos = dev["info"].get("qos", 0)
            msg_info = self._client.publish(
                topic, json.dumps(payload, ensure_ascii=False), qos=qos
            )
            success_rc = getattr(mqtt, "MQTT_ERR_SUCCESS", 0)
            if getattr(msg_info, "rc", success_rc) != success_rc:
                raise RuntimeError(f"MQTT publish 返回错误码: {msg_info.rc}")
            try:
                await asyncio.to_thread(msg_info.wait_for_publish, timeout=3)
            except (asyncio.TimeoutError, RuntimeError, ValueError):
                if qos > 0:
                    raise

            # 乐观状态更新：最佳努力，仅反映"已下发指令"，实际状态由 state_topic 异步回传确认
            # 注意：如果设备离线，此乐观更新可能会与实际状态不一致，state_topic 消息会覆盖它
            with self._state_lock:
                current = dev["state"].get("status", "powered_off")
                if command.command in ("power_on", "boot"):
                    if current == "powered_off":
                        dev["state"]["power"] = True
                        dev["state"]["status"] = "standby"
                elif command.command in ("power_off", "shutdown"):
                    if current in ("standby", "running"):
                        dev["state"]["power"] = False
                        dev["state"]["status"] = "powered_off"
                    elif current == "error":
                        dev["state"]["power"] = False
                        dev["state"]["status"] = "powered_off"
                elif command.command == "start":
                    if current == "powered_off":
                        dev["state"]["power"] = True
                        dev["state"]["status"] = "running"
                    elif current == "standby":
                        dev["state"]["status"] = "running"
                elif command.command == "stop":
                    if current == "running":
                        dev["state"]["status"] = "standby"
                        # power 保持 True，不断电！
                elif command.command == "reset":
                    if current == "error":
                        dev["state"]["power"] = True
                        dev["state"]["status"] = "standby"

            return DeviceResult(
                success=True,
                device_id=device_id,
                executed_command=command.command,
                actual_params=command.params,
                message=f"[MQTT] 指令已发布到 {topic}: {command.command}",
                raw_response={"topic": topic, "qos": qos},
            )

        except Exception as e:
            logger.error("MQTT 发布失败: %s → %s", device_id, e)
            dev["state"]["status"] = "error"
            return DeviceResult(
                success=False, device_id=device_id,
                executed_command=command.command,
                message=f"MQTT 发布失败: {e}",
                error_code="MQTT_ERROR",
            )

    # ── 状态读取 ──────────────────────────────

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        """读取设备状态（缓存的最新状态 + 本地状态）"""
        if device_id not in self._devices:
            return {"error": f"设备 '{device_id}' 不存在"}

        cached = dict(self._state_cache.get(device_id, {}))
        local = dict(self._devices[device_id]["state"])
        # 合并：MQTT 上报的状态优先
        merged = {**local, **cached}
        merged["_read_at"] = datetime.now().isoformat()
        merged["_driver"] = "mqtt"
        return merged

    # ── MQTT 回调 ──────────────────────────────

    def _on_connect(self, client, userdata, flags, rc):
        self._connect_rc = int(rc)
        if rc == 0:
            self._connected = True
            logger.info("MQTT 连接成功: %s:%d", self._broker_host, self._broker_port)
            # clean_session=True 时重连会丢失订阅，所以每次连接成功都重订阅。
            for dev_id, dev in self._devices.items():
                state_topic = dev["info"].get("state_topic")
                if state_topic:
                    qos = dev["info"].get("qos", 0)
                    client.subscribe(state_topic, qos=qos)
                    logger.info("MQTT 订阅状态: %s → %s", dev_id, state_topic)
        else:
            logger.warning("MQTT 连接失败, rc=%d", rc)
            self._connected = False
        # 成功和失败都唤醒 connect()，由 _connected 决定最终结果。
        if self._connect_event:
            self._connect_event.set()

    def _on_message(self, client, userdata, msg):
        """接收设备上报的状态消息"""
        try:
            payload_str = msg.payload.decode("utf-8")
            payload = json.loads(payload_str)
            topic = msg.topic
            if not isinstance(payload, dict):
                logger.warning("MQTT 状态消息必须是 JSON 对象: %s", topic)
                return

            # O(1) 反向映射：直接用 topic 找到 device_id
            dev_id = self._topic_to_device.get(topic)
            if dev_id:
                with self._state_lock:
                    self._state_cache[dev_id] = payload
                logger.debug("MQTT 状态更新: %s ← %s", dev_id, topic)
            else:
                logger.debug("MQTT 收到未注册 topic: %s", topic)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.debug("MQTT 消息非 JSON 或编码错误: %s", msg.topic)
        except Exception as e:
            logger.warning("MQTT 消息处理异常: %s", e)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning("MQTT 异常断开, rc=%d", rc)
