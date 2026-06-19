"""天气查询节点 - 获取天气预报和农事建议"""

import logging

from langchain_core.messages import AIMessage

from ..state import AgentState

from core.weather_service import WeatherService

logger = logging.getLogger(__name__)


def weather_query_node(state: AgentState) -> AgentState:
    """
    天气查询节点 - 获取天气预报和农事建议
    """
    if state.intent_type == "weather_query":
        user_question = state.user_question or ""

        # 获取用户地区（优先使用用户档案中的地区）
        location = (state.short_term_facts.get("region") or
                   state.user_profile.get("region", "北京"))

        # 获取当前作物
        crop = state.short_term_facts.get("crop") or state.user_profile.get("crop", "")

        # 获取生长阶段
        growth_stage = state.short_term_facts.get("growth_stage", "")

        try:
            # 初始化天气服务
            weather_service = WeatherService()

            # 获取当前天气
            current = weather_service.get_current_weather(location)

            # 获取未来5天预报
            forecast = weather_service.get_forecast(location, 5)

            # 获取农事建议
            farming_advice = weather_service.get_farming_advice(location, crop, growth_stage)

            # 获取预警信息
            alerts = weather_service.check_weather_alerts(location, crop)

            # 构建回答
            answer_parts = []

            # 1. 当前天气
            if current:
                answer_parts.append(weather_service.format_weather_report(current))

            # 2. 天气预警
            if alerts:
                answer_parts.append(weather_service.format_alert_report(alerts))

            # 3. 农事建议
            if farming_advice:
                answer_parts.append(weather_service.format_farming_advice(farming_advice))

            # 4. 未来3天简要预报
            if forecast:
                answer_parts.append("\n📅 **未来3天预报**：")
                for w in forecast[:3]:
                    answer_parts.append(f"   {w.date}: {w.weather_desc} {w.temperature_low}℃~{w.temperature_high}℃")

            # 5. 施药气象分析（用户问喷药/打药/施肥时自动附加）
            spray_keywords = ["喷药", "打药", "施药", "喷洒", "喷雾", "施肥", "撒肥", "追肥",
                              "农药", "除草剂", "杀虫剂", "杀菌剂", "叶面肥"]
            if any(w in user_question for w in spray_keywords) and current and forecast:
                try:
                    from core.spray_advisor import assess_spray_conditions, format_spray_report
                    wdata = {
                        "temperature": current.temperature,
                        "humidity": getattr(current, "humidity", 60),
                        "wind_speed": getattr(current, "wind_speed", 0),
                        "weather_desc": current.weather_desc,
                        "forecast": [
                            {"date": str(w.date), "weather_desc": w.weather_desc,
                             "temp_high": w.temperature_high, "temp_low": w.temperature_low}
                            for w in forecast[:3]
                        ],
                    }
                    spray = assess_spray_conditions(wdata)
                    answer_parts.append("\n---")
                    answer_parts.append(format_spray_report(spray))
                except Exception as e:
                    logger.warning("施药分析失败: %s", e)

            state.final_answer = "\n".join(answer_parts)
            state.messages.append(AIMessage(content=state.final_answer))

        except Exception as e:
            state.final_answer = f"获取天气信息时出现错误：{str(e)}。请检查天气服务配置。"
            state.messages.append(AIMessage(content=state.final_answer))

    return state
