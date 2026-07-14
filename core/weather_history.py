"""历史天气记录 + 持续异常检测 + 农事建议"""

import os, json, logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_FILE = os.path.join(_PROJECT_ROOT, "data", "weather_history.json")

# 持续异常阈值
PERSISTENCE_RULES = {
    "持续降雨": {"condition": lambda w: w.get("rain", False), "days": 3,
                "advice": "连续降雨易导致根系缺氧、病害滋生。建议：1.及时清沟排水 2.雨后喷施杀菌剂预防病害 3.天晴后追施叶面肥恢复长势"},
    "持续高温": {"condition": lambda w: w.get("temp_high", 0) > 35, "days": 3,
                "advice": "持续高温易导致作物萎蔫、授粉不良。建议：1.早晚灌溉降温 2.覆盖遮阳网 3.喷施抗旱剂或磷酸二氢钾"},
    "持续低温": {"condition": lambda w: w.get("temp_low", 0) < 2, "days": 2,
                "advice": "低温冻害风险。建议：1.覆盖地膜或秸秆保温 2.熏烟防霜 3.提前灌水增加地温 4.喷施防冻液"},
    "持续干旱": {"condition": lambda w: w.get("rain", True) is False and w.get("humidity", 100) < 40, "days": 7,
                "advice": "持续干旱影响作物正常生长。建议：1.及时灌溉保墒 2.中耕松土减少蒸发 3.覆盖保水 4.优先保障需水关键期作物"},
}

# 不同作物的针对性建议
CROP_SPECIFIC_ADVICE = {
    "小麦": {"持续降雨": "小麦拔节后遇连续降雨易发赤霉病、锈病。抢晴天喷施戊唑醇+咪鲜胺防赤霉病。雨后及时清沟排渍。",
             "持续高温": "小麦灌浆期遇高温易逼熟减产。早晚喷水降温，喷施磷酸二氢钾增强抗逆性。",
             "持续低温": "小麦拔节后遇低温（倒春寒）易冻伤幼穗。提前灌水增温，喷施芸苔素内酯缓解冻害。"},
    "水稻": {"持续降雨": "水稻分蘖期遇连续降雨需控制无效分蘖。排水晒田，雨后追施钾肥壮秆。",
             "持续高温": "水稻抽穗扬花期遇高温易导致结实率下降。日灌夜排降温，喷施叶面肥。"},
    "玉米": {"持续降雨": "玉米大喇叭口期遇连续降雨易发大斑病。雨后及时喷施苯醚甲环唑。注意排涝防倒伏。"},
    "番茄": {"持续降雨": "番茄遇连续降雨易发晚疫病。雨后立即喷施霜脲·锰锌或嘧菌酯。及时摘除病叶病果。"},
    "花生": {"持续降雨": "花生结荚期遇连续降雨易烂果。及时清沟排水，雨后喷施钙肥防烂果。"},
    "棉花": {"持续高温": "棉花花铃期遇高温易脱落。早晚喷水降温，喷施硼肥保花保铃。"},
    "大豆": {"持续降雨": "大豆开花结荚期遇连续降雨易发锈病。雨后喷施三唑酮防锈病，注意排涝。"},
    "油菜": {"持续降雨": "油菜开花期遇连续降雨易发菌核病。雨后喷施多菌灵或嘧霉胺。清沟排渍。"},
}


def _load_history() -> List[Dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_history(records: List[Dict]):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    # 只保留最近 60 天
    cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    records = [r for r in records if r.get("date", "") >= cutoff]
    # 先写临时文件，再原子替换，避免写入失败时丢失全部数据
    tmp_file = HISTORY_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, HISTORY_FILE)  # 原子替换（Windows 也支持）


def record_today(weather: Dict):
    """记录今天天气"""
    today = datetime.now().strftime("%Y-%m-%d")
    records = _load_history()
    # 今天已记录则更新
    for r in records:
        if r.get("date") == today:
            r.update(weather)
            _save_history(records)
            return
    weather["date"] = today
    records.append(weather)
    _save_history(records)


def check_persistence(active_crops: List[str] = None) -> List[Dict]:
    """检测持续异常天气，返回需触发的提醒列表"""
    records = _load_history()
    if len(records) < 2:
        return []

    records.sort(key=lambda r: r["date"])
    alerts = []

    for rule_name, rule in PERSISTENCE_RULES.items():
        check_fn = rule["condition"]
        threshold = rule["days"]
        # 检查最近 N 个日历天是否持续满足条件（而非最近 N 条记录）
        cutoff_date = (datetime.now() - timedelta(days=threshold)).strftime("%Y-%m-%d")
        recent = [r for r in records if r.get("date", "0000-00-00") >= cutoff_date]
        if len(recent) < threshold:
            continue
        if all(check_fn(r) for r in recent):
            advice = rule["advice"]
            # 追加作物特定建议
            if active_crops:
                crop_tips = []
                for crop in active_crops:
                    if crop in CROP_SPECIFIC_ADVICE and rule_name in CROP_SPECIFIC_ADVICE[crop]:
                        crop_tips.append(f"🌾 {crop}：{CROP_SPECIFIC_ADVICE[crop][rule_name]}")
                if crop_tips:
                    advice += "\n\n" + "\n".join(crop_tips[:3])

            alerts.append({
                "type": rule_name,
                "days": threshold,
                "period": f"{recent[0]['date']} 至 {recent[-1]['date']}",
                "advice": advice,
            })

    return alerts
