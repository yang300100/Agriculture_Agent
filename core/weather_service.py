"""
天气服务模块
功能：
- 获取实时天气和未来天气预报
- 农业气象灾害预警
- 基于天气的农事建议
- 对接第三方天气API（和风天气、OpenWeatherMap等）
"""

import logging
import os
import json
import requests

logger = logging.getLogger(__name__)
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import dotenv

dotenv.load_dotenv(override=True)

# 天气API配置
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
WEATHER_API_PROVIDER = os.getenv("WEATHER_API_PROVIDER", "openweathermap")
QWEATHER_API_HOST = os.getenv("QWEATHER_API_HOST", "https://devapi.qweather.com")


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

    def __init__(self, api_key: str = None, provider: str = None,
                 max_consecutive_failures: int = 5):
        self.api_key = api_key or WEATHER_API_KEY
        self.provider = provider or WEATHER_API_PROVIDER
        self.cache = {}  # 简单缓存
        self.cache_time = 1800  # 缓存30分钟
        self._consecutive_failures = 0  # 连续失败计数
        self._max_consecutive_failures = max_consecutive_failures  # 连续失败上限
        self._aborted = False  # 是否已触发熔断

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
            elif self.provider == "qweather":
                data = self._fetch_qweather_current(location)
            else:
                data = self._fetch_mock_weather(location)

            weather_info = self._parse_weather_data(data)
            self._set_cache(cache_key, asdict(weather_info))
            return weather_info

        except Exception as e:
            print(f"获取天气失败: {e}")
            return None

    def get_grid_weather(self, lon: float, lat: float) -> Optional[Dict]:
        """
        获取指定坐标的格点实时天气（和风天气）

        Args:
            lon: 经度 (longitude)
            lat: 纬度 (latitude)

        Returns:
            dict with keys: temp, text, humidity, precip, windDir,
            windSpeed, windScale, cloud, pressure, obsTime
            Returns None on failure.
        """
        # 已触发熔断，不再发起新请求
        if self._aborted:
            return None

        cache_key = f"grid_{lon:.2f}_{lat:.2f}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            if self.provider == "qweather":
                try:
                    data = self._fetch_qweather_grid_now(lon, lat)
                except Exception as qw_error:
                    # QWeather 格点天气失败时，回退到 OpenWeatherMap
                    logger.info("和风天气格点接口不可用(%s, %s)，回退 OpenWeatherMap", lon, lat)
                    data = self._fetch_openweather_by_coords(lon, lat)

                self._consecutive_failures = 0  # 成功，重置计数
            else:
                data = self._fetch_openweather_by_coords(lon, lat)
                self._consecutive_failures = 0

            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            self._consecutive_failures += 1
            print(f"获取格点天气失败 ({lon}, {lat}) [连续失败 {self._consecutive_failures}/{self._max_consecutive_failures}]: {e}")
            if self._consecutive_failures >= self._max_consecutive_failures:
                self._aborted = True
                print(f"已连续失败 {self._max_consecutive_failures} 次，停止后续所有天气请求。")
            return None

    def _fetch_qweather_grid_now(self, lon: float, lat: float) -> Dict:
        """调用和风天气格点实时天气 API（使用 key 参数认证）"""
        url = f"{QWEATHER_API_HOST}/v7/grid-weather/now"
        params = {
            "location": f"{lon:.2f},{lat:.2f}",
            "key": self.api_key,
        }

        response = requests.get(url, params=params, timeout=10)

        # 403 通常是免费订阅不包含格点天气接口
        if response.status_code == 403:
            raise Exception(
                f"和风天气API 403 权限不足: 当前密钥可能为免费订阅，"
                f"不包含格点天气(grid-weather)接口。"
                f"请升级为付费订阅，或将 WEATHER_API_HOST 改为 https://api.qweather.com"
            )

        response.raise_for_status()
        result = response.json()

        if result.get("code") != "200":
            raise Exception(f"和风天气API错误: code={result.get('code')}")

        now = result.get("now", {})
        return {
            "temp": float(now.get("temp", 0)),
            "text": now.get("text", "未知"),
            "icon": now.get("icon", ""),
            "humidity": int(now.get("humidity", 0)),
            "precip": float(now.get("precip", 0)),
            "windDir": now.get("windDir", ""),
            "windSpeed": float(now.get("windSpeed", 0)),
            "windScale": now.get("windScale", ""),
            "wind360": now.get("wind360", ""),
            "cloud": now.get("cloud", ""),
            "pressure": float(now.get("pressure", 0)),
            "dew": now.get("dew", ""),
            "obsTime": now.get("obsTime", ""),
        }

    def _geocode(self, location: str) -> tuple:
        """将地名转换为坐标，国内城市/省份/区域有内置映射，失败回退北京"""
        city_coords = {
            # 城市
            "北京": (116.40, 39.90), "上海": (121.47, 31.23), "广州": (113.26, 23.13),
            "深圳": (114.06, 22.54), "成都": (104.07, 30.67), "武汉": (114.30, 30.60),
            "杭州": (120.15, 30.28), "南京": (118.78, 32.06), "西安": (108.94, 34.26),
            "重庆": (106.55, 29.57), "天津": (117.19, 39.12), "沈阳": (123.43, 41.80),
            "哈尔滨": (126.53, 45.80), "郑州": (113.66, 34.76), "济南": (116.98, 36.67),
            "长沙": (112.97, 28.23), "昆明": (102.83, 24.88), "福州": (119.30, 26.07),
            "南宁": (108.37, 22.82), "海口": (110.32, 20.03), "石家庄": (114.50, 38.04),
            "太原": (112.55, 37.87), "呼和浩特": (111.75, 40.84), "长春": (125.32, 43.90),
            "合肥": (117.28, 31.86), "南昌": (115.86, 28.68), "贵阳": (106.71, 26.57),
            "兰州": (103.83, 36.06), "西宁": (101.78, 36.62), "银川": (106.27, 38.47),
            "乌鲁木齐": (87.62, 43.82), "拉萨": (91.13, 29.65),
            # 省份
            "河北": (114.50, 38.04), "山西": (112.55, 37.87), "内蒙古": (111.75, 40.84),
            "辽宁": (123.43, 41.80), "吉林": (125.32, 43.90), "黑龙江": (126.53, 45.80),
            "江苏": (118.78, 32.06), "浙江": (120.15, 30.28), "安徽": (117.28, 31.86),
            "福建": (119.30, 26.07), "江西": (115.86, 28.68), "山东": (116.98, 36.67),
            "河南": (113.66, 34.76), "湖北": (114.30, 30.60), "湖南": (112.97, 28.23),
            "广东": (113.26, 23.13), "广西": (108.37, 22.82), "海南": (110.32, 20.03),
            "四川": (104.07, 30.67), "贵州": (106.71, 26.57), "云南": (102.83, 24.88),
            "西藏": (91.13, 29.65), "陕西": (108.94, 34.26), "甘肃": (103.83, 36.06),
            "青海": (101.78, 36.62), "宁夏": (106.27, 38.47), "新疆": (87.62, 43.82),
            "台湾": (121.50, 25.05), "香港": (114.17, 22.30), "澳门": (113.55, 22.19),
            # 农业区域
            "华北": (116.40, 39.90), "东北": (125.32, 43.90), "华东": (118.78, 32.06),
            "华中": (114.30, 30.60), "华南": (113.26, 23.13), "西南": (104.07, 30.67),
            "西北": (103.83, 36.06), "黄淮海": (116.98, 36.67), "长江流域": (114.30, 30.60),
        }
        for name, coords in city_coords.items():
            if name in location:
                return coords
        # Nominatim 仅作兜底，不阻塞
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="agriculture_agent")
            loc = geolocator.geocode(location, timeout=3)
            if loc:
                return (loc.longitude, loc.latitude)
        except Exception:
            pass
        return (116.40, 39.90)

    def _fetch_qweather_current(self, location: str) -> Dict:
        """通过和风天气获取当前天气（先地理编码转坐标，再调格点API）"""
        lon, lat = self._geocode(location)
        grid_data = self.get_grid_weather(lon, lat)
        if grid_data:
            return {
                "name": location,
                "dt": int(datetime.now().timestamp()),
                "main": {
                    "temp": grid_data.get("temp", 0),
                    "temp_max": grid_data.get("temp", 0),
                    "temp_min": grid_data.get("temp", 0),
                    "humidity": grid_data.get("humidity", 0),
                    "pressure": grid_data.get("pressure", 1013),
                },
                "weather": [{"description": grid_data.get("text", ""), "main": grid_data.get("text", "")}],
                "wind": {"speed": grid_data.get("windSpeed", 0) / 3.6, "deg": 0},
                "sys": {"sunrise": "", "sunset": ""},
            }
        raise Exception("和风天气格点数据获取失败")

    def _fetch_qweather_forecast(self, location: str, days: int) -> List[Dict]:
        """通过和风天气获取天气预报"""
        lon, lat = self._geocode(location)
        url = f"{QWEATHER_API_HOST}/v7/grid-weather/7d"
        params = {"location": f"{lon:.2f},{lat:.2f}", "key": self.api_key}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result.get("code") != "200":
            raise Exception(f"和风天气预报API错误: code={result.get('code')}")
        daily_list = result.get("daily", [])[:days]
        data_list = []
        for day in daily_list:
            data_list.append({
                "dt_txt": day.get("fxDate", ""),
                "main": {
                    "temp": (float(day.get("tempMax", 0)) + float(day.get("tempMin", 0))) / 2,
                    "temp_max": float(day.get("tempMax", 0)),
                    "temp_min": float(day.get("tempMin", 0)),
                    "humidity": int(day.get("humidity", 60)),
                    "pressure": int(day.get("pressure", 1013)),
                },
                "weather": [{"description": day.get("textDay", ""), "main": day.get("textDay", "")}],
                "wind": {"speed": 0, "deg": 0},
                "pop": float(day.get("precip", 0)) / 100 if day.get("precip") else 0,
            })
        if not data_list:
            return self._generate_mock_forecast(location, days)
        return data_list

    def _fetch_openweather_by_coords(self, lon: float, lat: float) -> Dict:
        """通过坐标从 OpenWeatherMap 获取天气（备用）"""
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
            "lang": "zh_cn",
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})
        return {
            "temp": main.get("temp", 0),
            "text": weather.get("description", ""),
            "humidity": main.get("humidity", 0),
            "precip": data.get("rain", {}).get("1h", 0),
            "windDir": "",
            "windSpeed": wind.get("speed", 0),
            "windScale": "",
            "cloud": str(data.get("clouds", {}).get("all", "")),
            "pressure": main.get("pressure", 0),
            "obsTime": datetime.now().isoformat(),
        }

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
            elif self.provider == "qweather":
                data_list = self._fetch_qweather_forecast(location, days)
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
