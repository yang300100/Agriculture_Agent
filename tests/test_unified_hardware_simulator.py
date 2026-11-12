"""哥哥，统一模拟器只用本机隔离端口和虚拟图片验证。"""

import base64
import io
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from PIL import Image

from hardware_examples import all_hardware_simulator as simulator


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.delenv("HARDWARE_CAMERA_IMAGE", raising=False)
    def forbid_private_path(*args):
        raise AssertionError("不允许读取隐式的用户桌面路径")
    monkeypatch.setattr(simulator.os.path, "expanduser", forbid_private_path)
    mgr = simulator.UnifiedDeviceManager()
    mgr.init_all()
    return mgr


def _app(manager):
    factory = getattr(simulator, "create_http_app", None)
    assert callable(factory), "需要可测试的 HTTP 应用工厂"
    return factory(manager)


def test_camera_capture_decodes_at_reported_size_without_private_files(manager):
    result = manager.execute("greenhouse_camera_01", "capture")
    assert result["success"], result
    data = base64.b64decode(result["image_base64"], validate=True)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        assert image.size == (640, 480)
        assert image.size == (result["metadata"]["width"], result["metadata"]["height"])
        assert image.getextrema()[0][0] != image.getextrema()[0][1]
    assert result["metadata"]["source"] == "simulated"


def test_explicit_camera_image_uses_actual_dimensions(manager, monkeypatch, tmp_path):
    image_path = tmp_path / "configured.png"
    Image.new("RGB", (432, 280), "#287f50").save(image_path)
    monkeypatch.setenv("HARDWARE_CAMERA_IMAGE", str(image_path))

    result = manager.execute("greenhouse_camera_01", "capture")

    assert result["success"], result
    with Image.open(io.BytesIO(base64.b64decode(result["image_base64"]))) as image:
        image.load()
        assert image.size == (432, 280)
    assert (result["metadata"]["width"], result["metadata"]["height"]) == (432, 280)
    assert result["metadata"]["source"] == "configured_image"


@pytest.mark.parametrize("invalid", ["missing.png", "invalid.png"])
def test_explicit_camera_image_errors_are_not_hidden_by_fallback(manager, monkeypatch, tmp_path, invalid):
    image_path = tmp_path / invalid
    if invalid == "invalid.png":
        image_path.write_bytes(b"not an image")
    monkeypatch.setenv("HARDWARE_CAMERA_IMAGE", str(image_path))

    result = manager.execute("greenhouse_camera_01", "capture")

    assert result["success"] is False
    assert "HARDWARE_CAMERA_IMAGE" in result["message"]
    assert "image_base64" not in result
    assert manager.read_state("greenhouse_camera_01")["last_capture_time"] is None


def test_http_homepage_lists_seven_devices_and_captures_camera(manager):
    client = _app(manager).test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "硬件模拟器" in response.get_data(as_text=True)
    panel = client.get("/api/panel/state").get_json()
    assert len(panel["devices"]) == 7
    assert {d["id"] for d in panel["devices"]} == {
        "irrigation_pump_01", "ventilation_fan_01", "grow_light_01", "heater_01",
        "env_sensor_01", "fertilizer_pump_01", "greenhouse_camera_01",
    }
    captured = client.post("/command", json={"device_id": "greenhouse_camera_01", "command": "capture"})
    assert captured.status_code == 200
    with Image.open(io.BytesIO(base64.b64decode(captured.get_json()["image_base64"]))) as image:
        image.load()
        assert image.size == (640, 480)


def test_panel_controls_shared_manager_while_http_command_retains_filter(manager):
    client = _app(manager).test_client()
    command = {"device_id": "ventilation_fan_01", "command": "start"}
    assert client.post("/command", json=command).status_code == 404
    assert client.post("/api/panel/command", json=command).status_code == 200
    assert manager.read_state("ventilation_fan_01")["status"] == "running"
    state = client.get("/api/panel/state").get_json()
    assert next(d for d in state["devices"] if d["id"] == "ventilation_fan_01")["state"]["status"] == "running"


def test_disabled_protocol_is_visible_and_cannot_execute_panel_command(manager):
    manager.protocol_status = {"HTTP": True, "MQTT": False, "Modbus": True}
    client = _app(manager).test_client()
    state = client.get("/api/panel/state").get_json()
    assert state["protocols"]["MQTT"] is False
    response = client.post("/api/panel/command", json={"device_id": "ventilation_fan_01", "command": "start"})
    assert response.status_code == 503
    assert manager.read_state("ventilation_fan_01")["status"] == "standby"


def test_http_state_rejects_disabled_http_and_non_http_device(manager):
    client = _app(manager).test_client()
    assert client.get("/state?device_id=greenhouse_camera_01").status_code == 200
    assert client.get("/state?device_id=env_sensor_01").status_code == 404
    manager.protocol_status["HTTP"] = False
    assert client.get("/state?device_id=greenhouse_camera_01").status_code == 503
    assert client.get("/state").status_code == 503


def test_device_names_are_not_interpolated_into_executable_homepage(manager):
    manager.get("greenhouse_camera_01")["_name"] = '<img src=x onerror="alert(1)">'
    client = _app(manager).test_client()
    html = client.get("/").get_data(as_text=True)
    assert '<img src=x onerror="alert(1)">' not in html
    device = next(d for d in client.get("/api/panel/state").get_json()["devices"] if d["id"] == "greenhouse_camera_01")
    assert device["name"] == '<img src=x onerror="alert(1)">'


@pytest.mark.parametrize("payload", [[], ["bad"], {"device_id": []}, {"device_id": "grow_light_01", "params": []}])
def test_invalid_command_payload_returns_clear_error(manager, payload):
    client = _app(manager).test_client()
    response = client.post("/command", json=payload)
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def _start_process(extra_args=(), *, stdin=subprocess.DEVNULL):
    env = os.environ.copy()
    env.pop("HARDWARE_CAMERA_IMAGE", None)
    for key in ("HARDWARE_HTTP_PORT", "HARDWARE_MQTT_PORT", "HARDWARE_MODBUS_PORT"):
        with socket.socket() as temporary:
            temporary.bind(("127.0.0.1", 0))
            env[key] = str(temporary.getsockname()[1])
    script = Path(simulator.__file__).resolve()
    process = subprocess.Popen(
        [sys.executable, "-u", str(script), *extra_args], stdin=stdin,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
    )
    return process, env


def _wait_ready(process, port):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"模拟器提前退出: {process.communicate()[0].decode('utf-8', errors='replace')}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=.3) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, ConnectionError):
            time.sleep(.05)
    pytest.fail("模拟器 HTTP 没有及时就绪")


@pytest.mark.parametrize("arguments", [(), ("--headless", "--no-mqtt", "--no-modbus")])
def test_stdin_eof_keeps_servers_alive(arguments):
    process, env = _start_process(arguments)
    try:
        _wait_ready(process, int(env["HARDWARE_HTTP_PORT"]))
        time.sleep(.7)
        assert process.poll() is None, "stdin EOF 不应关闭模拟器服务"
        with urlopen(f"http://127.0.0.1:{env['HARDWARE_HTTP_PORT']}/api/panel/state", timeout=1) as response:
            assert response.status == 200
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=8)


@pytest.mark.parametrize("protocol,port_env,args", [
    ("HTTP", "HARDWARE_HTTP_PORT", ("--no-mqtt", "--no-modbus")),
    ("MQTT", "HARDWARE_MQTT_PORT", ("--no-http", "--no-modbus")),
    ("Modbus", "HARDWARE_MODBUS_PORT", ("--no-http", "--no-mqtt")),
])
def test_binding_failure_returns_nonzero(protocol, port_env, args):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        env = os.environ.copy()
        for key in ("HARDWARE_HTTP_PORT", "HARDWARE_MQTT_PORT", "HARDWARE_MODBUS_PORT"):
            with socket.socket() as temporary:
                temporary.bind(("127.0.0.1", 0))
                env[key] = str(temporary.getsockname()[1])
        env[port_env] = str(listener.getsockname()[1])
        result = subprocess.run(
            [sys.executable, "-u", str(Path(simulator.__file__).resolve()), "--headless", *args],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, timeout=10,
        )
    assert result.returncode != 0, result.stdout.decode("utf-8", errors="replace")
    assert protocol in result.stdout.decode("utf-8", errors="replace")


def test_sigterm_returns_cleanly_and_releases_owned_servers():
    env = os.environ.copy()
    ports = []
    for key in ("HARDWARE_HTTP_PORT", "HARDWARE_MQTT_PORT", "HARDWARE_MODBUS_PORT"):
        with socket.socket() as temporary:
            temporary.bind(("127.0.0.1", 0))
            port = temporary.getsockname()[1]
            env[key] = str(port)
            ports.append(port)
    script = """
import signal, socket, threading
from hardware_examples.all_hardware_simulator import main
timer = threading.Timer(2, lambda: signal.raise_signal(signal.SIGTERM))
timer.start()
code = main(['--headless'])
timer.join()
assert code == 0, code
for port in PORTS:
    with socket.socket() as client:
        client.settimeout(.3)
        assert client.connect_ex(('127.0.0.1', port)) != 0, port
print('CLEAN_STOP')
""".replace("PORTS", repr(ports))
    result = subprocess.run(
        [sys.executable, "-u", "-c", script], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=15,
        cwd=Path(simulator.__file__).resolve().parents[1],
    )
    output = result.stdout.decode("utf-8", errors="replace")
    assert result.returncode == 0, output
    assert "CLEAN_STOP" in output


@pytest.mark.parametrize("protocol", ["MQTT", "Modbus"])
def test_server_stop_closes_existing_protocol_connections(manager, protocol):
    with socket.socket() as temporary:
        temporary.bind(("127.0.0.1", 0))
        port = temporary.getsockname()[1]
    if protocol == "MQTT":
        service = simulator.MqttBroker(port=port)
        service.start()
        mqtt_client = simulator.SimpleMqttClient(port=port)
        assert mqtt_client.connect()
        client = mqtt_client._sock
    else:
        service = simulator.ModbusTcpServer(manager, port=port)
        service.start()
        client = socket.create_connection(("127.0.0.1", port), timeout=1)
        client.sendall(bytes.fromhex("000100000006010300000001"))
        assert client.recv(1024)
    try:
        service.stop()
        client.settimeout(.5)
        assert client.recv(1024) == b"", "服务停止后，已有客户端也应收到断开连接"
    finally:
        client.close()
        service.stop()
