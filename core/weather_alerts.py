"""天气主动提醒 + 收获倒计时"""

import os
import json
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional

import dotenv

dotenv.load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


def check_weather_alert_for_region(region: str, crop: str = "") -> Optional[Dict]:
    """主动检查天气预警（轻量版，不调完整天气服务）"""
    from core.weather_service import WeatherService
    try:
        service = WeatherService()
        alerts = service.check_weather_alerts(region, crop if crop else None)
        if alerts:
            return {
                "has_alert": True,
                "count": len(alerts),
                "alerts": [
                    {
                        "type": a.alert_type,
                        "level": a.level,
                        "desc": a.description,
                        "suggestions": a.suggestions[:3],
                    }
                    for a in alerts[:3]
                ],
                "region": region,
            }
    except Exception as e:
        logger.warning("天气预警检查失败: %s", e)
    return None


def calculate_harvest_countdown(progress_records: List[Dict]) -> List[Dict]:
    """计算每个种植进度的收获倒计时"""
    results = []
    today = date.today()

    for p in progress_records:
        start_str = p.get("start_date", "")
        crop = p.get("crop", "")
        if not start_str or not crop:
            continue

        try:
            start_date = datetime.strptime(start_str[:10], "%Y-%m-%d").date()
        except Exception:
            continue

        # 从知识库获取生长周期
        total_days = _get_crop_total_days(crop)
        harvest_date = start_date + timedelta(days=total_days)
        days_left = (harvest_date - today).days

        status = "growing"
        if days_left < 0:
            status = "harvested"
        elif days_left <= 7:
            status = "soon"
        elif days_left <= 30:
            status = "approaching"

        results.append({
            "crop": crop,
            "stage": p.get("stage", ""),
            "start_date": start_str,
            "harvest_date": harvest_date.strftime("%Y-%m-%d"),
            "days_left": max(0, days_left),
            "total_days": total_days,
            "status": status,
            "progress_percent": p.get("progress_percent", 0),
        })

    results.sort(key=lambda x: x["days_left"])
    return results


def _get_crop_total_days(crop: str) -> int:
    """从作物知识库获取总生长天数"""
    crops_dir = os.path.join(PROJECT_ROOT, "agriculture_knowledge", "crops")
    if not os.path.exists(crops_dir):
        return 120  # 默认

    for fname in os.listdir(crops_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(crops_dir, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get("crop_name") == crop:
                stages = data.get("growth_stages", [])
                return sum(s.get("duration_days", 30) for s in stages)
        except Exception:
            pass
    return 120


def format_harvest_countdown(countdowns: List[Dict]) -> str:
    """格式化收获倒计时显示"""
    if not countdowns:
        return ""

    lines = []
    for c in countdowns[:3]:
        status_emoji = {
            "soon": "🔴", "approaching": "🟡", "growing": "🟢", "harvested": "✅"
        }
        emoji = status_emoji.get(c["status"], "⚪")
        if c["status"] == "harvested":
            lines.append(f"{emoji} **{c['crop']}**: 已到收获期")
        elif c["days_left"] == 0:
            lines.append(f"{emoji} **{c['crop']}**: 今天收获！")
        elif c["days_left"] <= 7:
            lines.append(f"{emoji} **{c['crop']}**: 还有 **{c['days_left']}天** 收获")
        else:
            lines.append(f"{emoji} **{c['crop']}**: 还有 {c['days_left']}天 ({c['harvest_date']})")
    return "\n".join(lines)
