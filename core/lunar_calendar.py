"""农历/二十四节气模块 — 提供农历日期转换和节气农事指导"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from zhdate import ZhDate
    _ZH_VIA_SDK = False
except ImportError:
    ZhDate = None
    _ZH_VIA_SDK = False

SOLAR_TERMS = [
    "立春", "雨水", "惊蛰", "春分", "清明", "谷雨",
    "立夏", "小满", "芒种", "夏至", "小暑", "大暑",
    "立秋", "处暑", "白露", "秋分", "寒露", "霜降",
    "立冬", "小雪", "大雪", "冬至", "小寒", "大寒",
]

# 21世纪节气计算常数: (月份, C值)
# 公式: day = floor(Y * 0.2422 + C) - floor(Y / 4), Y = 年份后两位
_TERM_CONSTANTS = [
    (2, 4.63),   # 0  立春
    (2, 19.12),  # 1  雨水
    (3, 6.11),   # 2  惊蛰
    (3, 20.84),  # 3  春分
    (4, 5.17),   # 4  清明
    (4, 20.22),  # 5  谷雨
    (5, 5.94),   # 6  立夏
    (5, 21.31),  # 7  小满
    (6, 6.22),   # 8  芒种
    (6, 21.94),  # 9  夏至
    (7, 7.44),   # 10 小暑
    (7, 23.09),  # 11 大暑
    (8, 7.92),   # 12 立秋
    (8, 23.47),  # 13 处暑
    (9, 8.31),   # 14 白露
    (9, 23.73),  # 15 秋分
    (10, 9.04),  # 16 寒露
    (10, 23.88), # 17 霜降
    (11, 7.82),  # 18 立冬
    (11, 22.71), # 19 小雪
    (12, 7.49),  # 20 大雪
    (12, 22.28), # 21 冬至
    (1, 6.11),   # 22 小寒 (次年1月)
    (1, 20.84),  # 23 大寒 (次年1月)
]


def _calc_solar_term_dates(year: int) -> dict:
    """用天文公式计算指定年份的24节气日期（21世纪适用）"""
    y = year % 100
    result = {}
    for i, (name, (month, C)) in enumerate(zip(SOLAR_TERMS, _TERM_CONSTANTS)):
        day = int(y * 0.2422 + C) - int(y / 4)
        # 前两个节气(小寒/大寒)属于上一年，月份是1月
        actual_year = year + 1 if name in ("小寒", "大寒") and month == 1 else year
        # 如果前一年的大寒/小寒需要用前一年计算
        calc_year = year - 1 if i <= 1 else year
        if i <= 1:
            calc_y = (year - 1) % 100
            day = int(calc_y * 0.2422 + C) - int(calc_y / 4)
        try:
            result[name] = f"{month:02d}-{day:02d}"
        except (ValueError, OverflowError):
            result[name] = "01-01"
    return result


# 缓存已计算的年份
_TERM_CACHE: dict = {}


def _get_term_dates(year: int) -> dict:
    """获取某年节气日期（带缓存）"""
    if year not in _TERM_CACHE:
        _TERM_CACHE[year] = _calc_solar_term_dates(year)
    return _TERM_CACHE[year]

SOLAR_TERM_FARMING_MAP: Dict[str, Dict[str, Any]] = {
    "立春": {
        "season": "春", "activities": ["备耕整地", "检修农具", "育苗准备"],
        "advice": "立春一年端，种地早盘算。开始备耕整地，做好春播准备。",
        "crops": ["冬小麦返青管理", "早春蔬菜育苗"],
    },
    "雨水": {
        "season": "春", "activities": ["春灌", "施肥", "果树修剪"],
        "advice": "春雨贵如油。做好春灌保墒，小麦追施返青肥。",
        "crops": ["冬小麦追肥", "油菜清沟排水"],
    },
    "惊蛰": {
        "season": "春", "activities": ["春耕整地", "播种准备", "病虫害防治"],
        "advice": "惊蛰一犁土，春分地气通。开始春耕整地，关注越冬害虫防治。",
        "crops": ["春玉米备播", "马铃薯播种", "蔬菜育苗"],
    },
    "春分": {
        "season": "春", "activities": ["播种", "施肥", "灌溉"],
        "advice": "春分麦起身，一刻值千金。小麦拔节期管理，春播作物开始播种。",
        "crops": ["春小麦播种", "春玉米播种", "棉花备播"],
    },
    "清明": {
        "season": "春", "activities": ["大规模播种", "果树授粉", "茶园管理"],
        "advice": "清明前后，种瓜点豆。春播进入关键期，注意倒春寒防范。",
        "crops": ["大豆播种", "花生播种", "棉花播种", "春茶采摘"],
    },
    "谷雨": {
        "season": "春", "activities": ["播种移苗", "埯瓜点豆", "水稻育秧"],
        "advice": "谷雨前后种瓜点豆。抓住最后春播时机，准备夏收工作。",
        "crops": ["水稻育秧", "甘薯扦插", "蔬菜定植"],
    },
    "立夏": {
        "season": "夏", "activities": ["中耕除草", "灌溉防旱", "病虫害监测"],
        "advice": "立夏三天遍地锄。加强中耕除草，注意蚜虫等害虫防治。",
        "crops": ["冬小麦灌浆管理", "早稻田间管理", "棉花苗期管理"],
    },
    "小满": {
        "season": "夏", "activities": ["夏收准备", "灌溉", "病虫害防治"],
        "advice": "小满麦渐黄，夏收准备忙。检修收割机具，防干热风。",
        "crops": ["冬小麦最后管理", "夏玉米备播", "水稻插秧"],
    },
    "芒种": {
        "season": "夏", "activities": ["夏收", "夏种", "夏管"],
        "advice": "芒种忙收忙种。抢收冬小麦，抢种夏玉米，确保颗粒归仓。",
        "crops": ["冬小麦收割", "夏玉米播种", "夏大豆播种", "水稻插秧"],
    },
    "夏至": {
        "season": "夏", "activities": ["灌溉降温", "追肥", "病虫害防治"],
        "advice": "夏至不锄根边草，如同养下毒蛇咬。加强除草和病虫害防治。",
        "crops": ["夏玉米追肥", "水稻田间管理", "棉花整枝"],
    },
    "小暑": {
        "season": "夏", "activities": ["抗旱灌溉", "追肥", "病虫害防治"],
        "advice": "小暑天气热，棉花整枝不停歇。注意抗旱保墒和棉铃虫防治。",
        "crops": ["水稻晒田", "棉花管理", "蔬菜遮阳"],
    },
    "大暑": {
        "season": "夏", "activities": ["抗旱防涝", "追肥", "病虫害防治"],
        "advice": "大暑前后，衣裳溻透。注意防暑抗旱，同时防范暴雨洪涝。",
        "crops": ["晚稻插秧", "夏玉米大喇叭口期管理"],
    },
    "立秋": {
        "season": "秋", "activities": ["秋收准备", "秋菜播种", "积肥"],
        "advice": "立秋十八天，寸草皆结籽。开始秋收准备，播种秋菜。",
        "crops": ["秋白菜播种", "萝卜播种", "棉花采摘"],
    },
    "处暑": {
        "season": "秋", "activities": ["秋收", "秋耕", "蓄水保墒"],
        "advice": "处暑收黍，白露收谷。开始秋收，做好秋耕蓄水。",
        "crops": ["高粱收割", "谷子收割", "甘薯管理"],
    },
    "白露": {
        "season": "秋", "activities": ["秋收", "秋种准备", "茶园管理"],
        "advice": "白露早寒露迟，秋分种麦正当时。开始筹备秋播。",
        "crops": ["玉米收割", "水稻收割", "大豆收割", "秋茶采摘"],
    },
    "秋分": {
        "season": "秋", "activities": ["大规模秋收", "秋播", "秸秆还田"],
        "advice": "秋分种麦正当时。冬小麦最佳播种期，收获晚秋作物。",
        "crops": ["冬小麦播种", "油菜播种", "甘薯收获"],
    },
    "寒露": {
        "season": "秋", "activities": ["秋收扫尾", "秋播收尾", "防霜冻"],
        "advice": "寒露收山楂，霜降刨地瓜。完成秋收扫尾，预防早霜。",
        "crops": ["晚稻收割", "甘薯收获", "棉花收尾"],
    },
    "霜降": {
        "season": "秋", "activities": ["秋耕整地", "防冻准备", "设施农业管理"],
        "advice": "霜降拔葱，不拔就空。完成秋耕，做好大棚等设施防寒。",
        "crops": ["大葱收获", "白菜收获", "大棚蔬菜管理"],
    },
    "立冬": {
        "season": "冬", "activities": ["冬灌", "防寒防冻", "农田水利"],
        "advice": "立冬之日起大雾，冬水田里点萝卜。做好冬灌和水利维修。",
        "crops": ["冬小麦越冬管理", "油菜防寒"],
    },
    "小雪": {
        "season": "冬", "activities": ["农田水利建设", "果树冬剪", "积肥"],
        "advice": "小雪雪满天，来年必丰年。利用农闲修理水利设施和农具。",
        "crops": ["冬小麦镇压保墒", "果树冬季修剪"],
    },
    "大雪": {
        "season": "冬", "activities": ["农田基本建设", "设施农业管理", "技术培训"],
        "advice": "大雪不冻倒春寒。加强大棚等设施保温，参加农技培训。",
        "crops": ["设施蔬菜管理", "茶园冬季管理"],
    },
    "冬至": {
        "season": "冬", "activities": ["深翻改土", "积肥造肥", "冬修水利"],
        "advice": "冬至数九，冷在三九。开始数九寒天，做好防寒防冻。",
        "crops": ["冬小麦越冬", "油菜越冬", "来年种植规划"],
    },
    "小寒": {
        "season": "冬", "activities": ["防寒防冻", "检修温室", "种子准备"],
        "advice": "小寒大寒，冷成冰团。注意极端低温防范，准备春播种子。",
        "crops": ["设施农业保温", "果树防冻"],
    },
    "大寒": {
        "season": "冬", "activities": ["备耕整地", "种子选购", "农技学习"],
        "advice": "大寒过了就是年，备耕工作走在前。选购良种，规划新年种植。",
        "crops": ["制定种植计划", "预订农资"],
    },
}


def _get_term_for_date(dt: date) -> Optional[Dict[str, Any]]:
    """根据日期查找所属节气（公式计算，不限年份）"""
    # 当年和上年的节气表
    this_terms = _get_term_dates(dt.year)
    prev_terms = _get_term_dates(dt.year - 1)
    next_terms = _get_term_dates(dt.year + 1)

    # 合并：上一年小寒大寒 + 今年全部 + 下一年小寒大寒
    all_terms = {}
    for name in ("小寒", "大寒"):
        if name in prev_terms:
            all_terms[(name, dt.year - 1)] = prev_terms[name]
    for name, md in sorted(this_terms.items(), key=lambda x: x[1]):
        if name in ("小寒", "大寒"):
            all_terms[(name, dt.year + 1)] = md  # 小寒大寒实际在次年1月
        else:
            all_terms[(name, dt.year)] = md

    # 按日期排序
    current_md = dt.strftime("%m-%d")
    sorted_all = sorted(all_terms.items(), key=lambda x: x[1])

    current_term = None
    next_term = None
    current_date_str = ""
    next_date_str = ""

    for i, ((name, term_year), md) in enumerate(sorted_all):
        if md <= current_md:
            current_term = name
            current_date_str = f"{term_year}-{md}"
        if md >= current_md and next_term is None:
            next_term = name
            next_date_str = f"{term_year}-{md}"

    if current_term is None:
        current_term = "大寒"
    if next_term is None:
        next_term = "立春"

    return {
        "current": current_term,
        "next": next_term,
        "current_date": current_date_str or dt.strftime("%Y-") + "??-??",
        "next_date": next_date_str or dt.strftime("%Y-") + "??-??",
    }


def get_lunar_today() -> Dict[str, Any]:
    """获取今日农历和节气信息"""
    today = date.today()
    result = {
        "date": today.strftime("%Y-%m-%d"),
        "lunar_month": "",
        "lunar_day": "",
        "lunar_year": "",
        "zodiac": "",
        "solar_term_current": "",
        "solar_term_next": "",
        "solar_term_advice": "",
    }

    # 农历转换
    if ZhDate is not None:
        try:
            lunar = ZhDate.from_datetime(datetime.now())
            gan = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
            zhi = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
            zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
            year_ganzhi = gan[lunar.lunar_year % 10] + zhi[lunar.lunar_year % 12]
            result["lunar_year"] = f"{year_ganzhi}年"
            result["lunar_month"] = f"{lunar.lunar_month}月"
            result["lunar_day"] = f"{lunar.lunar_day}日"
            result["zodiac"] = zodiacs[(lunar.lunar_year - 1900) % 12]
        except Exception as e:
            logger.warning("农历转换失败: %s", e)

    # 节气
    try:
        term_info = _get_term_for_date(today)
        if term_info:
            result["solar_term_current"] = term_info["current"]
            result["solar_term_next"] = term_info["next"]
            advice = SOLAR_TERM_FARMING_MAP.get(
                term_info["current"],
                SOLAR_TERM_FARMING_MAP.get("立春", {}),
            )
            result["solar_term_advice"] = advice.get("advice", "")
    except Exception as e:
        logger.warning("节气计算失败: %s", e)

    return result


def get_solar_terms_in_range(start_date: date, end_date: date) -> List[Dict[str, Any]]:
    """获取指定日期范围内的所有节气（公式计算，不限年份）"""
    terms = []
    seen = set()
    for y in range(start_date.year - 1, end_date.year + 2):
        term_dates = _get_term_dates(y)
        for name, md in term_dates.items():
            # 小寒/大寒实际在次年1月
            actual_year = y + 1 if name in ("小寒", "大寒") else y
            try:
                term_date = datetime.strptime(f"{actual_year}-{md}", "%Y-%m-%d").date()
            except ValueError:
                continue
            key = f"{name}-{term_date}"
            if key in seen:
                continue
            seen.add(key)
            if start_date <= term_date <= end_date:
                farming = SOLAR_TERM_FARMING_MAP.get(name, {})
                terms.append({
                    "name": name,
                    "date": term_date.strftime("%Y-%m-%d"),
                    "activities": farming.get("activities", []),
                    "advice": farming.get("advice", ""),
                    "crops": farming.get("crops", []),
                })
    terms.sort(key=lambda x: x["date"])
    return terms


def get_farming_context_for_query(query: str) -> str:
    """根据当前节气注入农事上下文，用于增强 RAG 检索"""
    try:
        info = get_lunar_today()
        term = info.get("solar_term_current", "")
        advice = info.get("solar_term_advice", "")
        if term:
            return f"当前节气：{term}。{advice}"
    except Exception:
        pass
    return ""


def get_today_solar_term_display() -> str:
    """获取今日节气展示文本（侧边栏用）"""
    info = get_lunar_today()
    parts = []
    if info["lunar_month"] and info["lunar_day"]:
        parts.append(f"农历 {info['lunar_month']}{info['lunar_day']}")
    if info["solar_term_current"]:
        parts.append(f"节气：{info['solar_term_current']}")
    if info["solar_term_next"] and info["solar_term_next"] != info["solar_term_current"]:
        parts.append(f"下个节气：{info['solar_term_next']}")
    if info["solar_term_advice"]:
        parts.append(info["solar_term_advice"])
    return "\n".join(parts) if parts else "农历信息不可用"
