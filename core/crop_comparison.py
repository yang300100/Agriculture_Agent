"""种植方案多作物对比 — 生成多套方案并排比较"""

import logging
from typing import Dict, List, Any
from core.planting_planner import PlantingPlanner
from core.market_service import MarketService

logger = logging.getLogger(__name__)


# 各区域的推荐作物列表
REGION_CROPS = {
    "华北": ["小麦", "玉米", "大豆", "棉花", "花生"],
    "东北": ["玉米", "大豆", "水稻", "小麦", "土豆"],
    "黄淮海": ["小麦", "玉米", "大豆", "棉花", "花生"],
    "西北": ["小麦", "玉米", "棉花", "土豆", "油菜"],
    "华东": ["水稻", "小麦", "油菜", "大豆", "番茄"],
    "华南": ["水稻", "甘蔗", "花生", "番茄", "土豆"],
    "西南": ["水稻", "玉米", "土豆", "油菜", "大豆"],
}


def generate_multi_crop_plan(user_info: Dict[str, Any], num_options: int = 3) -> List[Dict]:
    """为同一区域生成多套作物方案并对比"""
    region = user_info.get("region", "")
    soil = user_info.get("soil_type", "")
    area = user_info.get("farm_size", 1.0)
    goals = user_info.get("goals", [])
    preferred_crop = user_info.get("crop", "")

    # 获取区域推荐作物
    candidates = REGION_CROPS.get(region[:2], ["小麦", "玉米", "大豆"])
    if preferred_crop and preferred_crop in candidates:
        # 将用户倾向的作物放在首位
        candidates.remove(preferred_crop)
        candidates.insert(0, preferred_crop)

    planner = PlantingPlanner()
    market = MarketService()

    options = []
    for crop in candidates[:num_options]:
        info = dict(user_info)
        info["crop"] = crop
        try:
            plan = planner.generate_plan(info)
        except Exception:
            plan = None

        price = market.get_price(crop)
        revenue = market.estimate_revenue(crop, area)

        # 计算适宜度评分
        score = _calculate_suitability(crop, region, soil, goals)

        options.append({
            "crop": crop,
            "suitability_score": score,
            "score_breakdown": {
                "climate": score * 0.8,  # 简化
                "soil": score * 0.7,
                "market": min(90, score + 5),
            },
            "price_range": f"{price['price_low']} - {price['price_high']} {price['unit']}",
            "avg_price": (price["price_low"] + price["price_high"]) / 2,
            "est_revenue": revenue.get("avg_revenue", 0),
            "est_yield": revenue.get("yield_medium", "参考知识库"),
            "growing_days": _get_crop_days(crop),
            "risk_level": _assess_risk(crop, region),
            "advantages": _get_advantages(crop),
            "disadvantages": _get_disadvantages(crop),
        })

    # 按适宜度评分降序
    options.sort(key=lambda x: x["suitability_score"], reverse=True)
    return options


def format_comparison_table(options: List[Dict]) -> str:
    """格式化多方案对比表"""
    if not options:
        return "暂无可用方案"

    lines = ["## 种植方案多作物对比\n"]
    lines.append("| 作物 | 适宜度 | 价格参考 | 预估收益 | 生长周期 | 风险 |")
    lines.append("|------|--------|----------|----------|----------|------|")

    for opt in options:
        risk_emoji = {"低": "🟢", "中": "🟡", "高": "🔴"}.get(opt["risk_level"], "⚪")
        lines.append(
            f"| **{opt['crop']}** | {opt['suitability_score']}分 | "
            f"{opt['price_range']} | ¥{opt['est_revenue']:,} | "
            f"约{opt['growing_days']}天 | {risk_emoji}{opt['risk_level']} |"
        )

    lines.append("")
    lines.append(f"**推荐**: {options[0]['crop']}（综合评分最高）")

    # 详述
    for i, opt in enumerate(options):
        lines.append(f"\n### {'🥇' if i==0 else '🥈' if i==1 else '🥉'} {opt['crop']}")
        lines.append(f"- 适宜度评分: {opt['suitability_score']}分")
        lines.append(f"- 优势: {', '.join(opt['advantages'][:3])}")
        lines.append(f"- 注意: {', '.join(opt['disadvantages'][:2])}")

    return "\n".join(lines)


def _calculate_suitability(crop: str, region: str, soil: str, goals: List[str]) -> int:
    """计算作物适宜度评分 (0-100)"""
    score = 70  # 基础分

    # 区域匹配加分
    for r, crops in REGION_CROPS.items():
        if region[:2] in r or r in region[:2]:
            if crop in crops:
                score += 15
            break

    # 目标匹配
    cash_crops = ["棉花", "大豆", "花生"]
    food_crops = ["小麦", "水稻", "玉米"]
    if "经济效益" in goals and crop in cash_crops:
        score += 10
    elif "高产" in goals and crop in food_crops:
        score += 10

    return min(100, score)


def _assess_risk(crop: str, region: str) -> str:
    """评估种植风险"""
    high_risk_crops = {"棉花": ["虫害风险高"], "番茄": ["病害风险高"]}
    if crop in high_risk_crops:
        return "中"
    return "低"


def _get_advantages(crop: str) -> List[str]:
    adv = {
        "小麦": ["管理成熟", "机械化程度高", "政策有补贴"],
        "玉米": ["适应性强", "产量高", "用途广泛"],
        "水稻": ["产量稳定", "需求大", "政策保护"],
        "大豆": ["固氮养地", "政策扶持", "进口替代需求"],
        "棉花": ["经济价值高", "耐旱", "适应性强"],
        "土豆": ["产量高", "适应性强", "营养丰富"],
        "花生": ["经济价值高", "耐旱耐瘠", "可榨油"],
        "油菜": ["冬季利用", "油料作物", "蜜源植物"],
        "高粱": ["极耐旱", "耐盐碱", "酿酒原料需求大"],
        "谷子": ["极耐旱", "营养价值高", "耐贮藏"],
        "甘薯": ["产量极高", "适应性强", "用途广泛"],
        "甘蔗": ["经济价值高", "糖料战略作物", "可宿根多年"],
        "烟草": ["经济效益高", "有专卖政策保障"],
        "茶叶": ["经济价值高", "多年生", "品牌溢价大"],
        "番茄": ["经济价值高", "生长期短", "市场需求大"],
    }
    return adv.get(crop, ["适应性好", "有一定市场"])


def _get_disadvantages(crop: str) -> List[str]:
    disadv = {
        "小麦": ["病虫害较多", "价格波动"],
        "玉米": ["需肥量大", "价格受国际影响"],
        "水稻": ["需水量大", "劳动强度高"],
        "大豆": ["产量偏低", "除草要求高"],
        "棉花": ["虫害严重", "采收成本高"],
        "土豆": ["储存要求高", "价格波动大"],
        "花生": ["连作障碍", "收获费工"],
        "油菜": ["产量偏低", "机械化程度低"],
        "高粱": ["口味限制需求", "价格波动大"],
        "谷子": ["产量较低", "除草费工"],
        "甘薯": ["不耐贮运", "价格低"],
        "甘蔗": ["需水肥量大", "地域受限"],
        "烟草": ["技术要求高", "种植需许可"],
        "茶叶": ["投资回收期长", "采摘人工高"],
        "番茄": ["病害多", "保鲜期短"],
    }
    return disadv.get(crop, ["需注意管理"])


def _get_crop_days(crop: str) -> int:
    days = {"小麦": 240, "玉米": 120, "水稻": 150, "大豆": 110, "棉花": 160,
            "土豆": 90, "花生": 130, "油菜": 220, "番茄": 100,
            "高粱": 130, "谷子": 110, "甘薯": 150, "甘蔗": 360,
            "烟草": 120, "茶叶": 365}
    return days.get(crop, 120)
