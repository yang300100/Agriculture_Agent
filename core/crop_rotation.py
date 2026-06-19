"""轮作建议 — 检测连作障碍风险，推荐科学的轮作方案"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger(__name__)


# 轮作兼容性矩阵：{作物: {前茬作物: 兼容性说明}}
ROTATION_RULES = {
    "小麦": {
        "大豆": "✅ 最佳前茬 — 大豆固氮，小麦增产明显",
        "玉米": "✅ 良好前茬 — 玉米残茬利于小麦播种",
        "棉花": "⚠️ 可接受 — 需注意残膜清理",
        "土豆": "✅ 良好前茬 — 马铃薯收获后土壤疏松",
        "花生": "✅ 良好前茬",
        "水稻": "⚠️ 水旱轮作需整地",
        "小麦": "❌ 不宜连作 — 病虫害加重，建议间隔1-2年",
    },
    "玉米": {
        "大豆": "✅ 最佳前茬 — 大豆固氮养地",
        "小麦": "✅ 良好前茬 — 麦茬玉米是传统模式",
        "土豆": "✅ 良好前茬",
        "玉米": "❌ 不宜连作 — 土壤养分失衡，病虫害积累",
        "棉花": "⚠️ 需增施有机肥",
    },
    "水稻": {
        "小麦": "✅ 水旱轮作 — 改善土壤结构",
        "油菜": "✅ 良好前茬",
        "大豆": "✅ 良好前茬",
        "水稻": "⚠️ 可连作2-3年 — 但需注意纹枯病积累",
    },
    "大豆": {
        "玉米": "✅ 最佳前茬",
        "小麦": "✅ 良好前茬",
        "土豆": "✅ 良好前茬",
        "大豆": "❌ 不宜连作 — 根腐病、孢囊线虫严重，必须间隔2年以上",
        "棉花": "⚠️ 可接受",
    },
    "棉花": {
        "小麦": "✅ 良好前茬",
        "玉米": "✅ 良好前茬",
        "大豆": "✅ 良好前茬",
        "棉花": "❌ 严禁连作 — 枯黄萎病严重，需间隔3-5年",
        "土豆": "⚠️ 可接受",
    },
    "土豆": {
        "玉米": "✅ 最佳前茬",
        "小麦": "✅ 良好前茬",
        "大豆": "✅ 良好前茬",
        "土豆": "❌ 不宜连作 — 晚疫病、环腐病加重，需间隔2-3年",
    },
    "番茄": {
        "水稻": "✅ 水旱轮作 — 有效减少土传病害",
        "小麦": "✅ 良好前茬",
        "大豆": "✅ 最佳前茬",
        "番茄": "❌ 不宜连作 — 青枯病、根结线虫严重，需间隔3年以上",
    },
    "花生": {
        "玉米": "✅ 最佳前茬",
        "小麦": "✅ 良好前茬",
        "土豆": "✅ 良好前茬",
        "花生": "❌ 不宜连作 — 根腐病加重，需间隔2年以上",
    },
}


class CropRotationAdvisor:
    """轮作顾问"""

    def __init__(self):
        self.rules = ROTATION_RULES

    def check_continuous_cropping_risk(self, crop: str, years: int = 1) -> Dict:
        """检查连作风险"""
        if crop not in self.rules:
            return {"risk": "unknown", "message": f"暂无{crop}的轮作数据"}

        own_rule = self.rules[crop].get(crop, "")
        if "严禁" in own_rule:
            return {"risk": "high", "message": own_rule, "min_interval": 3}
        elif "不宜" in own_rule:
            return {"risk": "medium", "message": own_rule, "min_interval": 2}
        elif "可连作" in own_rule:
            return {"risk": "low", "message": own_rule, "min_interval": 0}
        return {"risk": "low", "message": "可正常种植"}

    def recommend_rotation(self, current_crop: str, goals: List[str] = None) -> List[Dict]:
        """推荐下一季轮作作物"""
        if current_crop not in self.rules:
            return []

        compat = self.rules[current_crop]
        recommendations = []
        for next_crop, advice in compat.items():
            if next_crop == current_crop:
                continue
            if "❌" not in advice:
                score = 100
                if "最佳" in advice:
                    score = 95
                elif "良好" in advice:
                    score = 80
                elif "可接受" in advice:
                    score = 60

                # 根据目标调整打分
                if goals:
                    cash_crops = ["棉花", "花生"]
                    food_crops = ["小麦", "水稻", "玉米"]
                    if "经济效益" in goals and next_crop in cash_crops:
                        score += 5
                    if "高产" in goals and next_crop in food_crops:
                        score += 5

                recommendations.append({
                    "crop": next_crop,
                    "score": min(100, score),
                    "reason": advice,
                })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:5]

    def suggest_rotation_plan(self, crops: List[str], years: int = 3) -> List[Dict]:
        """为一系列作物生成轮作计划"""
        plan = []
        for year in range(years):
            year_plan = {"year": year + 1, "seasons": []}
            for i, crop in enumerate(crops):
                prev = crops[i - 1] if i > 0 else None
                risk = "normal"
                if prev and prev == crop:
                    risk_result = self.check_continuous_cropping_risk(crop)
                    risk = risk_result["risk"]
                year_plan["seasons"].append({
                    "crop": crop,
                    "previous": prev,
                    "continuous_risk": risk,
                })
            plan.append(year_plan)
        return plan

    def format_rotation_report(self, current_crop: str, field_name: str = "") -> str:
        """格式化轮作建议报告"""
        risk = self.check_continuous_cropping_risk(current_crop)
        recommendations = self.recommend_rotation(current_crop)

        lines = [f"## 轮作建议{f' — {field_name}' if field_name else ''}\n"]

        # 连作风险
        risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢", "unknown": "⚪"}
        emoji = risk_emoji.get(risk["risk"], "⚪")
        lines.append(f"**当前作物**: {current_crop}")
        lines.append(f"**连作风险**: {emoji} {risk['message']}\n")

        # 推荐轮作
        if recommendations:
            lines.append("**推荐下一季种植**:\n")
            for i, rec in enumerate(recommendations):
                lines.append(f"{i+1}. **{rec['crop']}** — {rec['reason']}")

        lines.append(f"\n> 科学轮作可减少病虫害、改善土壤结构、提高产量。")
        return "\n".join(lines)
