"""测试 weather_service 模块"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.weather_service import WeatherService, WeatherAlertLevel


class TestWeatherService:
    def setup_method(self):
        self.service = WeatherService()

    def test_init_default(self):
        assert self.service is not None
        assert hasattr(self.service, 'provider')

    def test_cache_key_generation(self):
        key = self.service._get_cache_key("北京", "current")
        assert "北京" in key
        assert "current" in key

    def test_cache_set_get(self):
        test_data = {"test": True}
        self.service._set_cache("test_key", test_data)
        cached = self.service._get_cached("test_key")
        assert cached == test_data

    def test_wind_direction(self):
        assert self.service._get_wind_direction(0) == "北"
        assert self.service._get_wind_direction(90) == "东"
        assert self.service._get_wind_direction(180) == "南"
        assert self.service._get_wind_direction(270) == "西"

    def test_classify_weather_rain(self):
        from core.weather_service import WeatherInfo
        info = WeatherInfo(
            location="test", date="2026-01-01",
            temperature=20, temperature_high=25, temperature_low=15,
            humidity=80, weather_desc="小雨", wind_speed=5, wind_direction="北",
            precipitation=5.0, uv_index=3, visibility=10.0, pressure=1013,
            sunrise="06:00", sunset="18:00"
        )
        weather_type = self.service._classify_weather(info)
        assert weather_type == "rain"

    def test_classify_weather_sunny(self):
        from core.weather_service import WeatherInfo
        info = WeatherInfo(
            location="test", date="2026-01-01",
            temperature=25, temperature_high=30, temperature_low=20,
            humidity=50, weather_desc="晴朗", wind_speed=5, wind_direction="南",
            precipitation=0, uv_index=5, visibility=15.0, pressure=1015,
            sunrise="06:00", sunset="18:00"
        )
        weather_type = self.service._classify_weather(info)
        assert weather_type == "sunny"

    def test_format_weather_report(self):
        from core.weather_service import WeatherInfo
        info = WeatherInfo(
            location="北京", date="2026-05-07",
            temperature=22, temperature_high=28, temperature_low=16,
            humidity=55, weather_desc="晴", wind_speed=12, wind_direction="南",
            precipitation=0, uv_index=6, visibility=12.0, pressure=1013,
            sunrise="05:30", sunset="19:00"
        )
        report = self.service.format_weather_report(info)
        assert "北京" in report
        assert "22" in report

    def test_alert_level_enum(self):
        assert WeatherAlertLevel.LOW.value == "低"
        assert WeatherAlertLevel.CRITICAL.value == "紧急"

    def test_geocode_known_city(self):
        coords = self.service._geocode("北京")
        assert len(coords) == 2
        assert 116 < coords[0] < 117
        assert 39 < coords[1] < 40

    def test_geocode_unknown_fallback(self):
        coords = self.service._geocode("完全不存在的城市名")
        assert len(coords) == 2
        # 应回退到默认坐标
