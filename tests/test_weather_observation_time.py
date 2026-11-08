"""哥哥，天气观测时间来自服务商，不连接外网。"""
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
import pytest

from core.weather_service import WeatherService


def response_with(value):
    data = {'main': {'temp': 21, 'humidity': 40}, 'weather': [{}]}
    if value is not ...:
        data['dt'] = value
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)


def test_openweather_time_uses_provider_epoch(monkeypatch):
    from core import weather_service
    stamp = datetime.now(timezone.utc) - timedelta(hours=2)
    monkeypatch.setattr(weather_service.requests, 'get', lambda *a, **kw: response_with(stamp.timestamp()))
    data = WeatherService()._fetch_openweather_by_coords(120, 30)
    assert datetime.fromisoformat(data['obsTime']) == stamp
    assert data['source'] == 'openweathermap'


@pytest.mark.parametrize('value', [..., None, 'bad', float('nan'), float('inf'), True, -1, (datetime.now(timezone.utc) + timedelta(days=1)).timestamp()])
def test_openweather_invalid_observation_time_is_not_fabricated(monkeypatch, value):
    from core import weather_service
    monkeypatch.setattr(weather_service.requests, 'get', lambda *a, **kw: response_with(value))
    data = WeatherService()._fetch_openweather_by_coords(120, 30)
    assert data['obsTime'] is None


def test_grid_cache_keeps_time_source_and_separates_coordinates(monkeypatch):
    service = WeatherService(); service.provider = 'openweathermap'
    calls = []
    def fetch(lon, lat):
        calls.append((lon, lat))
        return {'temp': len(calls), 'obsTime': '2020-01-01T00:00:00+00:00', 'source':'openweathermap'}
    monkeypatch.setattr(service, '_fetch_openweather_by_coords', fetch)
    first = service.get_grid_weather(120.01, 30.01)
    again = service.get_grid_weather(120.01, 30.01)
    second = service.get_grid_weather(120.02, 30.02)
    assert first == again and first['obsTime'] == '2020-01-01T00:00:00+00:00'
    assert second['temp'] == 2 and len(calls) == 2
