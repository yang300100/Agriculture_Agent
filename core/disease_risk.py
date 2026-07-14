"""病虫害风险计算引擎 — 融合气象数据与病害阈值"""

import os, json, logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 内置病害气象阈值（℃、%、mm）
BUILTIN_THRESHOLDS = {
    "小麦": {
        "赤霉病": {"temp_min": 15, "temp_max": 28, "humidity_min": 85, "rain_24h": 5, "stage": "抽穗扬花期"},
        "锈病": {"temp_min": 15, "temp_max": 25, "humidity_min": 80, "stage": "拔节期-灌浆期"},
        "白粉病": {"temp_min": 15, "temp_max": 25, "humidity_min": 70, "stage": "拔节期-灌浆期"},
    },
    "水稻": {
        "稻瘟病": {"temp_min": 20, "temp_max": 28, "humidity_min": 90, "stage": "分蘖期-抽穗期"},
        "纹枯病": {"temp_min": 22, "temp_max": 32, "humidity_min": 85, "stage": "分蘖期-孕穗期"},
        "稻飞虱": {"temp_min": 22, "temp_max": 30, "humidity_min": 80, "stage": "拔节期-灌浆期"},
    },
    "玉米": {
        "大斑病": {"temp_min": 20, "temp_max": 28, "humidity_min": 85, "stage": "拔节期-灌浆期"},
        "锈病": {"temp_min": 20, "temp_max": 30, "humidity_min": 80, "stage": "抽穗期-灌浆期"},
    },
    "番茄": {
        "晚疫病": {"temp_min": 10, "temp_max": 25, "humidity_min": 85, "rain_24h": 5, "stage": "全生育期"},
        "早疫病": {"temp_min": 20, "temp_max": 30, "humidity_min": 80, "stage": "全生育期"},
    },
    "棉花": {
        "枯萎病": {"temp_min": 20, "temp_max": 30, "humidity_min": 70, "stage": "苗期-蕾期"},
        "棉铃虫": {"temp_min": 22, "temp_max": 30, "humidity_min": 65, "stage": "蕾期-铃期"},
    },
    "花生": {
        "叶斑病": {"temp_min": 20, "temp_max": 30, "humidity_min": 80, "stage": "结荚期-饱果期"},
    },
    "大豆": {
        "锈病": {"temp_min": 18, "temp_max": 28, "humidity_min": 80, "stage": "开花期-结荚期"},
    },
    "油菜": {
        "菌核病": {"temp_min": 15, "temp_max": 25, "humidity_min": 85, "rain_24h": 5, "stage": "开花结荚期"},
    },
}


def _load_crop_thresholds() -> Dict:
    """合并内置阈值与 JSON 中的 risk_conditions"""
    merged = dict(BUILTIN_THRESHOLDS)
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    crops_dir = os.path.join(_project_root, "agriculture_knowledge", "crops")
    if not os.path.exists(crops_dir):
        return merged
    for fname in os.listdir(crops_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(crops_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            crop = data["crop_name"]
            if crop not in merged:
                merged[crop] = {}
            for d in data.get("common_diseases", []):
                rc = d.get("risk_conditions")
                if rc:
                    merged[crop][d["name"]] = rc
        except Exception:
            pass
    return merged


def assess_disease_risk(crop: str, current_stage: str = "",
                        temperature: float = 20, humidity: float = 60,
                        rain_24h: float = 0, forecast_rain: bool = False) -> List[Dict]:
    """
    评估指定作物在当前气象条件下的病害风险

    Returns:
        [{disease: str, risk: "高"/"中"/"低", score: 0-100, advice: str, matched: [str]}]
    """
    thresholds = _load_crop_thresholds()
    crop_thresholds = thresholds.get(crop, {})
    if not crop_thresholds:
        return []

    results = []
    for disease, cond in crop_thresholds.items():
        score = 0
        matched = []

        # 温度检查
        t_min = cond.get("temp_min")
        t_max = cond.get("temp_max")
        if t_min is not None and t_max is not None:
            if t_min <= temperature <= t_max:
                score += 30
                matched.append(f"温度{temperature:.0f}℃在适宜范围{t_min}-{t_max}℃")
            elif abs(temperature - t_min) <= 3 or abs(temperature - t_max) <= 3:
                score += 15
                matched.append(f"温度{temperature:.0f}℃接近适宜范围{t_min}-{t_max}℃")

        # 湿度检查
        h_min = cond.get("humidity_min")
        if h_min is not None and humidity >= h_min:
            score += 30
            matched.append(f"湿度{humidity:.0f}%≥阈值{h_min}%")
        elif h_min is not None and humidity >= h_min * 0.8:
            score += 15

        # 降雨检查
        r24 = cond.get("rain_24h", 0)
        if rain_24h >= r24 > 0:
            score += 20
            matched.append(f"24h降雨{rain_24h:.0f}mm≥阈值{r24}mm")
        if forecast_rain:
            score += 15
            matched.append("预报有降雨")

        # 阶段匹配（宽松匹配）
        stage_cond = cond.get("stage", "")
        if stage_cond and current_stage:
            stage_keywords = stage_cond.replace("期", "").split("-")
            if any(kw in current_stage for kw in stage_keywords):
                score += 20
                matched.append(f"当前处于易感阶段({current_stage})")
            elif "全生育期" in stage_cond:
                score += 20
        else:
            score += 10  # 无阶段信息，给基础分

        # 评分分级
        if score >= 70:
            risk = "高"
        elif score >= 45:
            risk = "中"
        else:
            risk = "低"

        if risk != "低":
            results.append({
                "disease": disease, "crop": crop, "risk": risk, "score": score,
                "matched": matched,
                "advice": _get_prevention_advice(crop, disease),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _get_prevention_advice(crop: str, disease: str) -> str:
    """从作物 JSON 获取防治建议"""
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    crops_dir = os.path.join(_project_root, "agriculture_knowledge", "crops")
    if not os.path.exists(crops_dir):
        return "请咨询当地农技部门"
    for fname in os.listdir(crops_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(crops_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            if data.get("crop_name") != crop:
                continue
            for d in data.get("common_diseases", []):
                if d["name"] == disease:
                    return d.get("prevention", "请咨询当地农技部门")
        except Exception:
            pass
    return "请咨询当地农技部门"


def assess_all_active_crops(weather_data: Dict) -> List[Dict]:
    """评估所有活跃作物的病害风险"""
    from core.planting_tracker import PlantingTracker
    tracker = PlantingTracker()
    progresses = tracker.get_progress()
    active = [p for p in progresses if p.status == "进行中"]

    all_risks = []
    for p in active:
        risks = assess_disease_risk(
            crop=p.crop, current_stage=p.stage,
            temperature=weather_data.get("temperature", 20),
            humidity=weather_data.get("humidity", 60),
            rain_24h=weather_data.get("rain_24h", 0),
            forecast_rain=weather_data.get("forecast_rain", False),
        )
        all_risks.extend(risks)
    return all_risks
