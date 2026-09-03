"""真实协议驱动的回归测试——覆盖模拟器测试触达不到的协议边界。"""

import asyncio
import io
import json
from types import SimpleNamespace

from devices.base import DeviceCapability, DeviceCommand


def test_hardware_simulator_embedded_jpeg_is_valid():
    from PIL import Image
    from hardware_examples.all_hardware_simulator import UnifiedDeviceManager

    data = UnifiedDeviceManager._generate_simulated_jpeg()
    assert data.startswith(b"\xff\xd8")
    assert data.endswith(b"\xff\xd9")
    image = Image.open(io.BytesIO(data))
    image.verify()


def test_registry_enforces_command_timeout():
    from devices.registry import DeviceDriverRegistry
    from devices.simulator_driver import SimulatorDriver

    async def run():
        registry = DeviceDriverRegistry()
        driver = SimulatorDriver(simulated_latency_ms=0)
        driver.add_virtual_device("slow_01", "慢设备", [DeviceCapability.IRRIGATE])
        await driver.connect()
        registry.register("sim", driver)
        await registry.discover_all()

        async def slow_execute(device_id, command):
            await asyncio.sleep(0.05)

        driver.execute = slow_execute
        result = await registry.execute(
            "slow_01", DeviceCommand("start", timeout_ms=5)
        )
        assert not result.success
        assert result.error_code == "TIMEOUT"

    asyncio.run(run())


def test_mqtt_rejected_connection_is_not_reported_online(monkeypatch):
    import devices.mqtt_driver as module

    class FakeClient:
        def __init__(self, **kwargs):
            self.on_connect = None
            self.on_message = None
            self.on_disconnect = None

        def connect_async(self, host, port, keepalive):
            self.on_connect(self, None, {}, 5)

        def loop_start(self):
            pass

        def loop_stop(self):
            pass

        def disconnect(self):
            pass

    monkeypatch.setattr(module.mqtt, "Client", FakeClient)
    driver = module.MQTTDriver()
    assert asyncio.run(driver.connect()) is False
    assert asyncio.run(driver.health_check()) is False


def test_mqtt_ignores_non_object_state_and_reports_real_qos():
    import devices.mqtt_driver as module

    class PublishInfo:
        rc = 0

        def wait_for_publish(self, timeout=None):
            return True

    class FakeClient:
        def __init__(self):
            self.published = None

        def publish(self, topic, payload, qos):
            self.published = (topic, payload, qos)
            return PublishInfo()

    driver = module.MQTTDriver()
    driver.register_device(
        "mqtt_01", "MQTT泵", [DeviceCapability.IRRIGATE], qos=1
    )
    driver._client = FakeClient()
    driver._connected = True

    message = SimpleNamespace(
        topic="devices/mqtt_01/state",
        payload=json.dumps(["不是对象"]).encode("utf-8"),
    )
    driver._on_message(None, None, message)
    assert driver._state_cache["mqtt_01"]["status"] == "powered_off"

    result = asyncio.run(driver.execute("mqtt_01", DeviceCommand("start")))
    assert result.success
    assert result.raw_response["qos"] == 1
    assert driver._client.published[2] == 1


def test_modbus_uses_new_device_id_keyword_and_reads_state():
    from devices.modbus_driver import ModbusDriver

    class Response:
        def __init__(self, registers=None):
            self.registers = registers or []

        def isError(self):
            return False

    class FakeClient:
        def __init__(self):
            self.write_device_id = None
            self.read_device_id = None

        def write_registers(self, address, values, *, device_id=1):
            self.write_device_id = device_id
            return Response()

        def read_holding_registers(self, address, *, count=1, device_id=1):
            self.read_device_id = device_id
            registers = [2, 1] + [0] * 8 + [235, 680, 450, 65, 120]
            return Response(registers)

    driver = ModbusDriver(mode="tcp", port="127.0.0.1:502")
    driver.register_device("plc_01", "PLC", [DeviceCapability.READ_SENSOR], slave_id="7")
    driver._client = FakeClient()
    driver._connected = True

    result = asyncio.run(
        driver.execute("plc_01", DeviceCommand("start", {"duration": 1}))
    )
    state = asyncio.run(driver.read_state("plc_01"))
    assert result.success
    assert driver._client.write_device_id == 7
    assert driver._client.read_device_id == 7
    assert state["temperature"] == 23.5
    assert state["status"] == "running"


def test_http_rejection_does_not_optimistically_change_state(monkeypatch):
    from devices.http_driver import HTTPDriver

    class Response:
        status_code = 200
        text = '{"success": false}'

        @staticmethod
        def json():
            return {"success": False, "message": "设备互锁拒绝"}

    driver = HTTPDriver()
    driver.register_device(
        "http_01", "HTTP泵", [DeviceCapability.IRRIGATE],
        base_url="http://127.0.0.1:5000",
    )

    async def fake_request(*args, **kwargs):
        return Response()

    monkeypatch.setattr(driver, "_async_request", fake_request)
    result = asyncio.run(driver.execute("http_01", DeviceCommand("start")))
    assert not result.success
    assert result.error_code == "DEVICE_REJECTED"
    assert driver._devices["http_01"]["state"]["status"] == "powered_off"


def test_http_validates_state_shape_and_bearer_header(monkeypatch):
    from devices.http_driver import HTTPDriver

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return ["错误格式"]

    driver = HTTPDriver()
    driver.register_device(
        "http_01", "HTTP传感器", [DeviceCapability.READ_SENSOR],
        base_url="http://127.0.0.1:5000", api_key="Bearer abc",
    )

    async def fake_request(*args, **kwargs):
        return Response()

    monkeypatch.setattr(driver, "_async_request", fake_request)
    state = asyncio.run(driver.read_state("http_01"))
    assert "_error" in state
    headers = driver._make_headers(driver._devices["http_01"]["info"])
    assert headers["Authorization"] == "Bearer abc"


def test_coap_driver_reads_and_executes_with_mock_transport(monkeypatch):
    import devices.coap_driver as module

    monkeypatch.setattr(module, "HAS_AIOCOAP", True)
    driver = module.CoAPDriver()
    driver.register_device(
        "coap_01", "CoAP传感器", [DeviceCapability.READ_SENSOR],
        base_uri="coap://127.0.0.1:5683",
    )

    class Code:
        @staticmethod
        def is_successful():
            return True

        def __str__(self):
            return "2.05 Content"

    responses = [
        SimpleNamespace(code=Code(), payload=b'{"temperature":24.5}'),
        SimpleNamespace(code=Code(), payload=b'{"success":true,"message":"ok"}'),
    ]

    async def fake_request(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(driver, "_request", fake_request)
    state = asyncio.run(driver.read_state("coap_01"))
    result = asyncio.run(driver.execute("coap_01", DeviceCommand("start")))
    assert state["temperature"] == 24.5
    assert result.success


def test_opcua_driver_uses_whitelisted_nodes(monkeypatch):
    import devices.opcua_driver as module

    class FakeNode:
        def __init__(self, node_id):
            self.node_id = node_id
            self.written = None

        async def write_value(self, value):
            self.written = value

        async def read_value(self):
            return 26.0

    class FakeClient:
        instances = []

        def __init__(self, url, timeout):
            self.url = url
            self.nodes = {}
            FakeClient.instances.append(self)

        def set_user(self, username):
            self.username = username

        def set_password(self, password):
            self.password = password

        async def connect(self):
            pass

        async def disconnect(self):
            pass

        def get_node(self, node_id):
            return self.nodes.setdefault(node_id, FakeNode(node_id))

    monkeypatch.setattr(module, "HAS_ASYNCUA", True)
    monkeypatch.setattr(module, "Client", FakeClient)
    driver = module.OPCUADriver()
    driver.register_device(
        "opc_01", "OPC泵", [DeviceCapability.IRRIGATE],
        endpoint="opc.tcp://127.0.0.1:4840",
        command_nodes={"start": {"node_id": "ns=2;s=Pump.Start", "value": True}},
        state_nodes={"temperature": "ns=2;s=Temperature"},
    )

    result = asyncio.run(driver.execute("opc_01", DeviceCommand("start")))
    state = asyncio.run(driver.read_state("opc_01"))
    rejected = asyncio.run(driver.execute("opc_01", DeviceCommand("reset")))
    assert result.success
    assert state["temperature"] == 26.0
    assert rejected.error_code == "UNSUPPORTED_COMMAND"


def test_factory_preserves_combined_modbus_tcp_endpoint(monkeypatch):
    import core.device_registry_factory as factory
    import devices.modbus_driver as modbus_module

    class FakeModbusDriver:
        driver_name = "modbus"
        instances = []

        def __init__(self, mode, port, baudrate, timeout):
            self.mode = mode
            self.port = port
            self.devices = []
            FakeModbusDriver.instances.append(self)

        def register_device(self, **kwargs):
            self.devices.append(kwargs)

        async def connect(self):
            return True

        async def disconnect(self):
            pass

    configs = [{
        "device_id": "modbus_01",
        "name": "Modbus设备",
        "driver": "modbus",
        "capabilities": ["read_sensor"],
        "connection": {"mode": "tcp", "port": "192.168.1.200:502", "slave_id": 1},
    }]
    monkeypatch.setattr(factory, "load_custom_devices", lambda username: configs)
    monkeypatch.setattr(modbus_module, "ModbusDriver", FakeModbusDriver)
    loop = asyncio.new_event_loop()
    registry, returned_loop = factory.setup_registry("test", loop=loop)
    try:
        assert returned_loop is loop
        assert FakeModbusDriver.instances[0].port == "192.168.1.200:502"
        assert registry.driver_names == ["modbus_1"]
    finally:
        loop.run_until_complete(registry.disconnect_all())
        loop.close()


def test_factory_creates_one_mqtt_driver_per_broker(monkeypatch):
    import core.device_registry_factory as factory
    import devices.mqtt_driver as mqtt_module

    class FakeMQTTDriver:
        driver_name = "mqtt"
        instances = []

        def __init__(self, **kwargs):
            self.options = kwargs
            self.devices = []
            FakeMQTTDriver.instances.append(self)

        def register_device(self, **kwargs):
            self.devices.append(kwargs)

        async def connect(self):
            return True

        async def disconnect(self):
            pass

    configs = [
        {
            "device_id": "mqtt_a", "name": "A", "driver": "mqtt",
            "capabilities": ["irrigate"],
            "connection": {"host": "broker-a", "port": 1883},
        },
        {
            "device_id": "mqtt_b", "name": "B", "driver": "mqtt",
            "capabilities": ["irrigate"],
            "connection": {"host": "broker-b", "port": 8883, "use_tls": True},
        },
    ]
    monkeypatch.setattr(factory, "load_custom_devices", lambda username: configs)
    monkeypatch.setattr(mqtt_module, "MQTTDriver", FakeMQTTDriver)
    loop = asyncio.new_event_loop()
    registry, _ = factory.setup_registry("test", loop=loop)
    try:
        assert len(FakeMQTTDriver.instances) == 2
        assert registry.driver_names == ["mqtt_1", "mqtt_2"]
        assert FakeMQTTDriver.instances[1].options["use_tls"] is True
    finally:
        loop.run_until_complete(registry.disconnect_all())
        loop.close()
