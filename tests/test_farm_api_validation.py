"""哥哥，用真实接口回归农场数据校验和完整列表。"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api_routes import register_routes


@pytest.fixture
def farm_client():
    app = FastAPI()
    register_routes(app)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.mark.parametrize("kind", ["costs", "income"])
def test_重复收支返回明确客户端错误而非服务器异常(farm_client, kind):
    path = f"/api/finance/{kind}?username=finance_duplicate_{kind}"
    payload = {"crop": "小麦", "quantity": 8, "unit_price": 12}
    assert farm_client.post(path, json=payload).status_code == 200
    response = farm_client.post(path, json=payload)
    assert response.status_code == 409
    assert "重复" in response.json()["detail"]
    assert len(farm_client.get(path).json()) == 1


@pytest.mark.parametrize("kind", ["costs", "income"])
def test_空作物收支返回可理解的校验错误(farm_client, kind):
    response = farm_client.post(
        f"/api/finance/{kind}?username=finance_invalid_{kind}",
        json={"crop": "  ", "quantity": 1, "unit_price": 2},
    )
    assert response.status_code == 400
    assert "作物" in response.json()["detail"]


@pytest.mark.parametrize("coordinates", [
    [[116.3], [116.31, 39.9], [116.3, 39.91]],
    [[181, 39.9], [116.31, 39.9], [116.3, 39.91]],
    [[116.3, 91], [116.31, 39.9], [116.3, 39.91]],
])
def test_非法地理坐标不写入数据库(farm_client, coordinates):
    path = "/api/fields?username=invalid_coordinates"
    before = farm_client.get(path).json()
    response = farm_client.post(path, json={"name": "无效边界", "coordinates": coordinates})
    assert response.status_code == 422
    assert farm_client.get(path).json() == before


@pytest.mark.parametrize("number", ["1e999", "-1e999", "NaN"])
def test_非有限坐标也能返回可序列化的校验错误(farm_client, number):
    path = "/api/fields?username=nonfinite_coordinates"
    response = farm_client.post(
        path,
        content='{"name":"无效边界","coordinates":[[' + number +
        ',38],[114.5,38],[114.5,38.1]]}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "coordinates"]
    assert farm_client.get(path).json() == []


def test_种植列表显示全部记录而非仅前五条(farm_client):
    path = "/api/progress?username=progress_full_list"
    for index in range(6):
        assert farm_client.post(path, json={
            "crop": f"测试作物{index}", "start_date": "2099-01-01",
        }).status_code == 200
    assert {row["crop"] for row in farm_client.get(path).json()} == {
        f"测试作物{index}" for index in range(6)
    }


def test_仪表盘天气使用当前农场档案地区(farm_client, monkeypatch):
    from core import weather_alerts

    regions = []
    monkeypatch.setattr(
        weather_alerts, "check_weather_alert_for_region", regions.append,
    )
    assert farm_client.post("/api/profile?username=hebei_farm", json={
        "user_region": "河北石家庄", "user_farm_size": 13,
    }).status_code == 200
    assert farm_client.get("/api/dashboard?username=hebei_farm").status_code == 200
    assert regions == ["河北石家庄"]
