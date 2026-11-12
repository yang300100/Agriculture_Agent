"""MQTT 认证用户名不能覆盖摄像头所属账号。"""

def test_camera_factory_preserves_user_with_mqtt_credentials(monkeypatch):
    import core.device_registry_factory as factory
    import devices.mqtt_driver as mqtt_module
    import devices.camera_driver as camera_module

    owners = []

    class Driver:
        driver_name = "mqtt"
        def __init__(self, **kwargs):
            pass
        def register_device(self, **kwargs):
            pass
        async def connect(self):
            return True
        async def disconnect(self):
            pass

    class Camera(Driver):
        driver_name = "camera"
        def __init__(self, username):
            owners.append(username)

    monkeypatch.setattr(mqtt_module, "MQTTDriver", Driver)
    monkeypatch.setattr(camera_module, "CameraDriver", Camera)
    monkeypatch.setattr(factory, "load_custom_devices", lambda username: [
        {"device_id": "fan", "driver": "mqtt", "name": "风扇", "connection": {
            "host": "127.0.0.1", "username": "broker_service_account", "password": "test-only",
        }},
        {"device_id": "camera", "driver": "camera", "name": "摄像头", "connection": {
            "camera_type": "usb", "source": "0",
        }},
    ])
    registry, loop = factory.setup_registry("farm_owner")
    factory.close_registry(loop, registry)
    assert owners == ["farm_owner"]
