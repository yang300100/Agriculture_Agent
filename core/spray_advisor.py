"""施药气象建议 — 结合天气判断喷药/施肥适宜性"""

from datetime import datetime, timedelta
from typing import Dict, Any


def assess_spray_conditions(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    评估当前是否适合喷药/施肥

    Args:
        weather_data: {
            "temperature": float (℃),
            "humidity": float (%),
            "wind_speed": float (m/s),
            "weather_desc": str,
            "forecast": [{"date": str, "weather_desc": str, "temp_high": float, "temp_low": float, "wind_speed": float}]
        }

    Returns:
        {"suitable": bool, "score": 0-100, "risks": [...], "advice": str, "best_window": str}
    """
    risks = []
    score = 100

    temp = weather_data.get("temperature", 20)
    humidity = weather_data.get("humidity", 60)
    wind = weather_data.get("wind_speed", 0)
    desc = weather_data.get("weather_desc", "")

    # 1. 风力检查
    if wind > 8:
        risks.append(f"💨 风力过大 ({wind:.0f}m/s)，严禁喷药，漂移风险极高")
        score -= 40
    elif wind > 5:
        risks.append(f"💨 风速偏高 ({wind:.0f}m/s)，建议暂缓喷药")
        score -= 25
    elif wind < 1:
        risks.append("🌫 基本无风，注意药剂沉降不均匀")
        score -= 5

    # 2. 降雨检查
    rain_keywords = ["雨", "暴雨", "阵雨", "雷阵雨", "小雨", "中雨", "大雨", "毛毛雨",
                     "shower", "rain", "drizzle", "thunderstorm"]
    if any(w in desc for w in rain_keywords):
        risks.append(f"🌧 当前有降雨 ({desc})，药剂会被冲刷")
        score -= 25

    # 3. 温度检查 — 先检查更严重的条件
    if temp > 35:
        risks.append(f"🌡 高温 ({temp:.0f}℃)，药剂蒸发快且易产生药害")
        score -= 30
    elif temp > 30:
        risks.append(f"🌡 温度偏高 ({temp:.0f}℃)，建议清晨或傍晚施药")
        score -= 15
    elif temp < 5:
        risks.append(f"❄ 低温 ({temp:.0f}℃)，严禁施药")
        score -= 40
    elif temp < 10:
        risks.append(f"❄ 温度过低 ({temp:.0f}℃)，药效不佳且作物代谢慢")
        score -= 20

    # 4. 湿度检查
    if humidity > 90:
        risks.append(f"💧 湿度过高 ({humidity:.0f}%)，药液不易干燥")
        score -= 10
    elif humidity < 30:
        risks.append(f"🏜 空气干燥 ({humidity:.0f}%)，雾滴蒸发快，建议加大用水量")
        score -= 5

    # 5. 降雨预报检查（未来6h）
    forecast = weather_data.get("forecast", [])
    if forecast:
        near_term = forecast[:2]  # 最近两个预报时段
        for f in near_term:
            f_desc = f.get("weather_desc", "")
            if any(w in f_desc for w in rain_keywords):
                risks.append(f"⚠ 预报 {f.get('date','')} 有{f_desc}，喷药后可能被冲刷")
                score -= 20
                break

    # 最佳窗口推荐（确保降雨已检查）
    has_rain = any(w in desc for w in rain_keywords)
    if 15 <= temp <= 28 and 1 <= wind <= 3 and humidity < 80 and not has_rain:
        best_window = "✅ 当前时段非常适合喷药作业"
    elif has_rain:
        best_window = "🌧 当前有雨，不建议喷药，请等待雨停"
    elif temp < 15 and wind <= 3:
        best_window = "⏰ 建议等到气温回升至15℃以上再进行喷药"
    elif temp > 30:
        best_window = "🌅 建议在清晨 (5:00-8:00) 或傍晚 (17:00-19:00) 喷药"
    elif wind > 3:
        best_window = "🕐 建议等待风力减弱（<3m/s）后再喷药"
    else:
        best_window = "⏳ 请关注天气变化，择机施药"

    if score >= 80:
        suitable = True
    elif score >= 60:
        suitable = True  # 可进行但需注意
    else:
        suitable = False

    return {
        "suitable": suitable,
        "score": max(0, score),
        "risks": risks,
        "advice": _get_spray_advice(suitable, score, risks),
        "best_window": best_window,
    }


def _get_spray_advice(suitable: bool, score: int, risks: list) -> str:
    if score >= 80:
        return "当前气象条件适宜喷药/施肥作业，按正常操作规程进行即可。"
    elif score >= 60:
        return "气象条件基本适宜，但存在一定风险，建议关注天气变化并采取防护措施。"
    elif score >= 40:
        return "气象条件不太理想，建议推迟施药计划，等待更好的天气窗口。"
    else:
        return "当前气象条件不适宜喷药/施肥，请务必推迟作业，否则药效大打折扣且可能造成药害。"


def format_spray_report(assessment: Dict[str, Any]) -> str:
    """格式化施药气象报告"""
    lines = ["## 🌤 施药气象分析\n"]
    score = assessment["score"]
    emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
    lines.append(f"**适宜度评分：{emoji} {score}/100**")
    lines.append(f"> {assessment['advice']}")
    lines.append(f"\n**最佳窗口：**{assessment['best_window']}")

    if assessment["risks"]:
        lines.append("\n**风险提示：**")
        for r in assessment["risks"]:
            lines.append(f"- {r}")

    lines.append("\n---")
    lines.append("*建议：喷药前务必查看最新天气预报，避开降雨和大风天气。*")
    return "\n".join(lines)
