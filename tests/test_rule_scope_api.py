"""哥哥，规则测试接口也必须按目标地块取真实读数。"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api_routes import register_routes
from core.device_registry_factory import save_custom_devices


def test_规则测试接口不混用其他地块的湿度():
    username = "demo_rule_api_scope"
    save_custom_devices(username, [
        {"device_id": "dry_sensor", "name": "干燥地块传感器", "driver": "simulator",
         "capabilities": ["read_sensor"], "plot_id": 101, "zone_id": "west",
         "initial_state": {"soil_moisture": 10}},
        {"device_id": "wet_sensor", "name": "湿润地块传感器", "driver": "simulator",
         "capabilities": ["read_sensor"], "plot_id": 202, "zone_id": "west",
         "initial_state": {"soil_moisture": 70}},
        {"device_id": "wet_pump", "name": "湿润地块水泵", "driver": "simulator",
         "capabilities": ["irrigate"], "plot_id": 202, "zone_id": "west",
         "initial_state": {"power": True, "status": "standby"}},
    ])
    app = FastAPI()
    register_routes(app)
    with TestClient(app) as client:
        response = client.post(f"/api/rules?username={username}", json={
            "name": "湿润地块低湿灌溉", "enabled": True,
            "trigger": {"logic": "AND", "conditions": [{
                "type": "sensor", "field": "soil_moisture", "op": "<", "value": 30,
            }]},
            "action": {"device_id": "wet_pump", "capability": "irrigate", "command": "start", "params": {"duration": 1}},
        })
        assert response.json()["success"]
        rule_id = response.json()["rule_id"]
        result = client.post(f"/api/rules/{rule_id}/test?username={username}").json()
        assert result["success"]
        assert result["rule_matched"] is False
        assert result["sensor_snapshot"]["soil_moisture"] > 60
        assert "dry_sensor.soil_moisture" not in result["sensor_snapshot"]
