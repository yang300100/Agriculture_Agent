"""
天气服务模块
功能：
- 获取实时天气和未来天气预报
- 农业气象灾害预警
- 基于天气的农事建议
- 对接第三方天气API（和风天气、OpenWeatherMap等）
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import dotenv

dotenv.load_dotenv()

# 天气API配置
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
WEATHER_API_PROVIDER = os.getenv("WEATHER_API_PROVIDER", "openweathermap")  # 默认使用OpenWeatherMap


class WeatherAlertLevel(Enum):
    """天气预警等级"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "紧急"


@dataclass
class WeatherInfo:
    """天气信息"""
    location: str
    date: str
    temperature: float  # 摄氏度
    temperature_high: float
    temperature_low: float
    humidity: int  # 湿度百分比
    weather_desc: str  # 天气描述
    wind_speed: float  # 风速 km/h
    wind_direction: str  # 风向
    precipitation: float  # 降水量 mm
    uv_index: int  # 紫外线指数
    visibility: float  # 能见度 km
    pressure: int  # 气压 hPa
    sunrise: str
    sunset: str


@dataclass
class WeatherAlert:
    """天气预警"""
    alert_type: str  # 霜冻、暴雨、干旱、高温等
    level: str  # 预警等级
    description: str
    start_time: str
    end_time: str
    suggestions: List[str]
    affected_crops: List[str]


@dataclass
class FarmingWeatherAdvice:
    """农事天气建议"""
    date: str
    suitable_activities: List[str]  # 适宜活动
    unsuitable_activities: List[str]  # 不适宜活动
    warnings: List[str]  # 注意事项
    irrigation_advice: str  # 灌溉建议
    spraying_advice: str  # 喷药建议


class WeatherService:
    """天气服务主类"""

    # 天气现象与农事建议映射
    WEATHER_ADVICE_MAP = {
        "rain": {
            "suitable": ["施肥", "播种（雨前）", "收获（雨前抢收）"],
            "unsuitable": ["喷洒农药", "田间作业", "晾晒"],
            "warnings": ["注意排水防涝", "雨后及时中耕松土"]
        },
        "sunny": {
            "suitable": ["喷洒农药", "收获", "晾晒", "播种", "整地"],
            "unsuitable": ["高温时段浇水"],
            "warnings": ["注意防晒", "及时灌溉", "监测病虫害"]
        },
        "cloudy": {
            "suitable": ["播种", "移栽", "施肥", "除草"],
            "unsuitable": ["喷洒农药（效果差）"],
            "warnings": ["注意通风透光"]
        },
        "windy": {
            "suitable": ["授粉作物传粉"],
            "unsuitable": ["喷洒农药", "架设支架", "高空作业"],
            "warnings": ["加固设施", "防风倒伏"]
        },
        "frost": {
            "suitable": [],
            "unsuitable": ["播种", "移栽", "灌溉"],
            "warnings": ["覆盖保温", "熏烟防冻", "延迟出苗"]
        }
    }

    def __init__(self, api_key: str = None, provider: str = None):
        self.api_key = api_key or WEATHER_API_KEY
        self.provider = provider or WEATHER_API_PROVIDER
        self.cache = {}  # 简单缓存
        self.cache_time = 1800  # 缓存30分钟

    def _get_cache_key(self, location: str, date: str) -> str:
        """生成缓存键"""
        return f"{location}_{date}"

    def _get_cached(self, key: str) -> Optional[Dict]:
        """获取缓存数据"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now().timestamp() - timestamp < self.cache_time:
                return data
        return None

    def _set_cache(self, key: str, data: Dict):
        """设置缓存"""
        self.cache[key] = (data, datetime.now().timestamp())

    def get_current_weather(self, location: str) -> Optional[WeatherInfo]:
        """
        获取当前天气

        Args:
            location: 地区名称（城市名）

        Returns:
            WeatherInfo对象
        """
        cache_key = self._get_cache_key(location, "current")
        cached = self._get_cached(cache_key)
        if cached:
            return WeatherInfo(**cached)

        try:
            if self.provider == "openweathermap":
                data = self._fetch_openweather_current(location)
            else:
                data = self._fetch_mock_weather(location)  # 模拟数据

            weather_info = self._parse_weather_data(data)
            self._set_cache(cache_key, asdict(weather_info))
            return weather_info

        except Exception as e:
            print(f"获取天气失败: {e}")
            return None

    def get_forecast(self, location: str, days: int = 7) -> List[WeatherInfo]:
        """
        获取未来天气预报

        Args:
            location: 地区名称
            days: 预报天数

        Returns:
            WeatherInfo列表
        """
        cache_key = self._get_cache_key(location, f"forecast_{days}")
        cached = self._get_cached(cache_key)
        if cached:
            return [WeatherInfo(**item) for item in cached]

        try:
            if self.provider == "openweathermap":
                data_list = self._fetch_openweather_forecast(location, days)
            else:
                data_list = self._generate_mock_forecast(location, days)

            weather_list = [self._parse_weather_data(data) for data in data_list]
            self._set_cache(cache_key, [asdict(w) for w in weather_list])
            return weather_list

        except Exception as e:
            print(f"获取天气预报失败: {e}")
            return []

    def check_weather_alerts(self, location: str, crop: str = None) -> List[WeatherAlert]:
        """
        检查天气预警

        Args:
            location: 地区名称
            crop: 作物名称（用于判断影响）

        Returns:
            预警列表
        """
        forecast = self.get_forecast(location, 7)
        alerts = []

        for weather in forecast:
            # 霜冻预警
            if weather.temperature_low < 2:
                alerts.append(WeatherAlert(
                    alert_type="霜冻预警",
                    level=WeatherAlertLevel.HIGH.value if weather.temperature_low < -2 else WeatherAlertLevel.MEDIUM.value,
                    description=f"预计{weather.date}最低气温降至{weather.temperature_low}℃，可能出现霜冻",
                    start_time=weather.date + " 02:00",
                    end_time=weather.date + " 08:00",
                    suggestions=["覆盖保温", "熏烟防冻", "延迟播种/移栽", "喷施防冻液"],
                    affected_crops=["蔬菜", "果树", "瓜类"] if not crop else [crop]
                ))

            # 暴雨预警
            if weather.precipitation > 50:
                alerts.append(WeatherAlert(
                    alert_type="暴雨预警",
                    level=WeatherAlertLevel.HIGH.value if weather.precipitation > 100 else WeatherAlertLevel.MEDIUM.value,
                    description=f"预计{weather.date}降水量达{weather.precipitation}mm",
                    start_time=weather.date + " 00:00",
                    end_time=weather.date + " 23:59",
                    suggestions=["疏通沟渠", "抢收成熟作物", "加固设施", "停止田间作业"],
                    affected_crops=["粮食作物", "经济作物"] if not crop else [crop]
                ))

            # 高温预警
            if weather.temperature_high > 35:
                alerts.append(WeatherAlert(
                    alert_type="高温预警",
                    level=WeatherAlertLevel.MEDIUM.value,
                    description=f"预计{weather.date}最高气温达{weather.temperature_high}℃",
                    start_time=weather.date + " 10:00",
                    end_time=weather.date + " 16:00",
                    suggestions=["增加灌溉", "遮阴降温", "避免中午作业", "防暑降温"],
                    affected_crops=["蔬菜", "果树"] if not crop else [crop]
                ))

            # 大风预警
            if weather.wind_speed > 20:  # 8级风
                alerts.append(WeatherAlert(
                    alert_type="大风预警",
                    level=WeatherAlertLevel.HIGH.value if weather.wind_speed > 28 else WeatherAlertLevel.MEDIUM.value,
                    description=f"预计{weather.date}风速达{weather.wind_speed}km/h",
                    start_time=weather.date + " 00:00",
                    end_time=weather.date + " 23:59",
                    suggestions=["加固设施", "停止高空作业", "防风倒伏", "保护幼苗"],
                    affected_crops=["高秆作物", "设施农业"] if not crop else [crop]
                ))

        return alerts

    def get_farming_advice(self, location: str, crop: str = None,
                          growth_stage: str = None) -> List[FarmingWeatherAdvice]:
        """
        获取农事天气建议

        Args:
            location: 地区名称
            crop: 作物名称
            growth_stage: 生长阶段

        Returns:
            农事建议列表
        """
        forecast = self.get_forecast(location, 5)
        advice_list = []

        for weather in forecast:
            # 判断天气类型
            weather_type = self._classify_weather(weather)
            advice_map = self.WEATHER_ADVICE_MAP.get(weather_type, self.WEATHER_ADVICE_MAP["sunny"])

            # 灌溉建议
            irrigation = self._generate_irrigation_advice(weather, crop, growth_stage)
            # 喷药建议
            spraying = self._generate_spraying_advice(weather)

            advice = FarmingWeatherAdvice(
                date=weather.date,
                suitable_activities=advice_map["suitable"],
                unsuitable_activities=advice_map["unsuitable"],
                warnings=advice_map["warnings"],
                irrigation_advice=irrigation,
                spraying_advice=spraying
            )
            advice_list.append(advice)

        return advice_list

    def _classify_weather(self, weather: WeatherInfo) -> str:
        """分类天气类型"""
        desc = weather.weather_desc.lower()
        if "rain" in desc or "雨" in desc:
            return "rain"
        elif "cloud" in desc or "云" in desc:
            return "cloudy"
        elif "wind" in desc or "风" in desc:
            return "windy"
        elif weather.temperature_low < 0:
            return "frost"
        else:
            return "sunny"

    def _generate_irrigation_advice(self, weather: WeatherInfo, crop: str = None,
                                   growth_stage: str = None) -> str:
        """生成灌溉建议"""
        if "rain" in weather.weather_desc.lower() or weather.precipitation > 10:
            return "今天有雨，无需灌溉"
        elif weather.temperature_high > 32:
            return f"高温干旱，建议清晨或傍晚灌溉，避开中午高温时段"
        elif crop and growth_stage:
            return f"{crop}{growth_stage}，建议适量灌溉保持土壤湿润"
        else:
            return "建议根据土壤墒情适时灌溉"

    def _generate_spraying_advice(self, weather: WeatherInfo) -> str:
        """生成喷药建议"""
        if "rain" in weather.weather_desc.lower():
            return "雨天不宜喷药，药效会降低"
        elif weather.wind_speed > 15:
            return f"风力较大({weather.wind_speed}km/h)，喷药易飘移，建议改日"
        elif weather.temperature_high > 30:
            return "高温时段不宜喷药，建议早晚进行"
        elif weather.temperature_low < 10:
            return "温度较低，药效可能受影响"
        else:
            return "天气适宜，可进行喷药作业"

    def _fetch_openweather_current(self, location: str) -> Dict:
        """从OpenWeatherMap获取当前天气"""
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": self.api_key,
            "units": "metric",
            "lang": "zh_cn"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def _fetch_openweather_forecast(self, location: str, days: int) -> List[Dict]:
        """从OpenWeatherMap获取天气预报"""
        url = f"http://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": location,
            "appid": self.api_key,
            "units": "metric",
            "lang": "zh_cn"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 聚合每日数据
        daily_data = {}
        for item in data.get("list", []):
            date = item["dt_txt"][:10]
            if date not in daily_data:
                daily_data[date] = []
            daily_data[date].append(item)

        result = []
        for date in sorted(daily_data.keys())[:days]:
            items = daily_data[date]
            temps = [i["main"]["temp"] for i in items]
            result.append({
                "dt_txt": date + " 12:00:00",
                "main": {
                    "temp": sum(temps) / len(temps),
                    "temp_max": max(temps),
                    "temp_min": min(temps),
                    "humidity": items[0]["main"]["humidity"],
                    "pressure": items[0]["main"]["pressure"]
                },
                "weather": items[0]["weather"],
                "wind": items[0]["wind"],
                "pop": sum(i.get("pop", 0) for i in items) / len(items)
            })

        return result

    def _fetch_mock_weather(self, location: str) -> Dict:
        """获取模拟天气数据（用于测试）"""
        import random
        base_temp = {"华北": 18, "华东": 22, "华南": 28, "东北": 15, "西北": 16, "西南": 20}.get(location[:2], 20)

        return {
            "name": location,
            "dt": int(datetime.now().timestamp()),
            "main": {
                "temp": base_temp + random.randint(-3, 3),
                "temp_max": base_temp + 5,
                "temp_min": base_temp - 5,
                "humidity": 50 + random.randint(-10, 20),
                "pressure": 1013
            },
            "weather": [{"description": "晴朗", "main": "Clear"}],
            "wind": {"speed": random.randint(2, 8), "deg": random.randint(0, 360)},
            "sys": {"sunrise": "06:00", "sunset": "18:30"}
        }

    def _generate_mock_forecast(self, location: str, days: int) -> List[Dict]:
        """生成模拟天气预报"""
        import random
        result = []
        base_temp = {"华北": 18, "华东": 22, "华南": 28, "东北": 15, "西北": 16, "西南": 20}.get(location[:2], 20)

        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            weather_types = ["Clear", "Clouds", "Rain"]
            weights = [0.5, 0.3, 0.2]
            weather_main = random.choices(weather_types, weights)[0]

            result.append({
                "dt_txt": date + " 12:00:00",
                "main": {
                    "temp": base_temp + random.randint(-5, 5),
                    "temp_max": base_temp + 8,
                    "temp_min": base_temp - 6,
                    "humidity": 50 + random.randint(-15, 25),
                    "pressure": 1013 + random.randint(-10, 10)
                },
                "weather": [{"description": "晴朗" if weather_main == "Clear" else "多云" if weather_main == "Clouds" else "小雨",
                           "main": weather_main}],
                "wind": {"speed": random.randint(2, 12), "deg": random.randint(0, 360)},
                "pop": 0.1 if weather_main != "Rain" else 0.6
            })

        return result

    def _parse_weather_data(self, data: Dict) -> WeatherInfo:
        """解析天气数据"""
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})
        sys = data.get("sys", {})

        return WeatherInfo(
            location=data.get("name", "未知"),
            date=data.get("dt_txt", datetime.now().strftime("%Y-%m-%d"))[:10],
            temperature=main.get("temp", 0),
            temperature_high=main.get("temp_max", main.get("temp", 0)),
            temperature_low=main.get("temp_min", main.get("temp", 0)),
            humidity=main.get("humidity", 0),
            weather_desc=weather.get("description", "未知"),
            wind_speed=wind.get("speed", 0) * 3.6,  # m/s to km/h
            wind_direction=self._get_wind_direction(wind.get("deg", 0)),
            precipitation=data.get("pop", 0) * 50,  # 概率转降水量估算
            uv_index=data.get("uvi", 0),
            visibility=data.get("visibility", 10000) / 1000,
            pressure=main.get("pressure", 1013),
            sunrise=sys.get("sunrise", "06:00"),
            sunset=sys.get("sunset", "18:30")
        )

    def _get_wind_direction(self, degree: int) -> str:
        """根据角度获取风向"""
        directions = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
        index = round(degree / 45) % 8
        return directions[index]

    def format_weather_report(self, weather: WeatherInfo) -> str:
        """格式化天气报告"""
        return f"""📍 {weather.location} {weather.date} 天气
🌡️ 温度: {weather.temperature}℃ ({weather.temperature_low}℃ ~ {weather.temperature_high}℃)
☁️ 天气: {weather.weather_desc}
💧 湿度: {weather.humidity}%
💨 风力: {weather.wind_direction}风 {weather.wind_speed:.1f}km/h
🌧️ 降水: {weather.precipitation:.1f}mm
🌅 日出/日落: {weather.sunrise} / {weather.sunset}
"""

    def format_alert_report(self, alerts: List[WeatherAlert]) -> str:
        """格式化预警报告"""
        if not alerts:
            return "✅ 未来7天暂无气象灾害预警"

        text = "⚠️ **气象预警提醒**\n\n"
        for alert in alerts[:5]:  # 最多显示5条
            level_emoji = {"低": "⚪", "中": "🟡", "高": "🔴", "紧急": "🚨"}
            emoji = level_emoji.get(alert.level, "⚪")
            text += f"{emoji} **{alert.alert_type}** ({alert.level}等级)\n"
            text += f"   时间: {alert.start_time} ~ {alert.end_time}\n"
            text += f"   {alert.description}\n"
            text += f"   建议: {', '.join(alert.suggestions[:3])}\n\n"

        return text

    def format_farming_advice(self, advice_list: List[FarmingWeatherAdvice]) -> str:
        """格式化农事建议"""
        text = "🌾 **未来5天农事建议**\n\n"

        for advice in advice_list:
            text += f"📅 **{advice.date}**\n"
            if advice.suitable_activities:
                text += f"   ✅ 适宜: {', '.join(advice.suitable_activities[:3])}\n"
            if advice.unsuitable_activities:
                text += f"   ❌ 不宜: {', '.join(advice.unsuitable_activities)}\n"
            text += f"   💧 灌溉: {advice.irrigation_advice}\n"
            text += f"   🧪 喷药: {advice.spraying_advice}\n\n"

        return text


# 便捷函数
def get_weather_advice_for_crop(location: str, crop: str, growth_stage: str = None) -> str:
    """获取针对特定作物的天气建议"""
    service = WeatherService()

    # 获取当前天气
    current = service.get_current_weather(location)

    # 获取预警
    alerts = service.check_weather_alerts(location, crop)

    # 获取农事建议
    advice = service.get_farming_advice(location, crop, growth_stage)

    result = ""
    if current:
        result += service.format_weather_report(current) + "\n"
    if alerts:
        result += service.format_alert_report(alerts) + "\n"
    if advice:
        result += service.format_farming_advice(advice)

    return result


if __name__ == "__main__":
    # 测试
    service = WeatherService()

    # 测试获取天气
    weather = service.get_current_weather("北京")
    if weather:
        print(service.format_weather_report(weather))

    # 测试预警
    alerts = service.check_weather_alerts("北京", "小麦")
    print(service.format_alert_report(alerts))

    # 测试农事建议
    advice = service.get_farming_advice("北京", "小麦", "拔节期")
    print(service.format_farming_advice(advice))
