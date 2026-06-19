"""农产品市场价格查询服务"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import dotenv
import requests

dotenv.load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

# 静态价格参考数据（元/kg）
PRICE_REFERENCE = {
    "小麦": {"range": (2.2, 3.0), "unit": "元/kg", "season_high": "1-3月", "season_low": "6-8月"},
    "玉米": {"range": (2.0, 2.8), "unit": "元/kg", "season_high": "4-6月", "season_low": "9-11月"},
    "水稻": {"range": (2.6, 4.0), "unit": "元/kg", "season_high": "1-3月", "season_low": "9-11月"},
    "大豆": {"range": (4.0, 6.5), "unit": "元/kg", "season_high": "3-5月", "season_low": "10-12月"},
    "棉花": {"range": (12.0, 18.0), "unit": "元/kg", "season_high": "3-5月", "season_low": "9-11月"},
    "土豆": {"range": (1.5, 3.5), "unit": "元/kg", "season_high": "1-3月", "season_low": "6-8月"},
    "番茄": {"range": (2.0, 6.0), "unit": "元/kg", "season_high": "12-2月", "season_low": "6-8月"},
    "花生": {"range": (6.0, 10.0), "unit": "元/kg", "season_high": "1-3月", "season_low": "9-11月"},
    "油菜": {"range": (4.0, 6.5), "unit": "元/kg", "season_high": "3-5月", "season_low": "6-8月"},
}


class MarketService:
    """农产品市场价格服务"""

    def __init__(self):
        self.price_data = PRICE_REFERENCE
        self._load_crop_prices()

    def _load_crop_prices(self):
        """从作物知识库加载市场信息补充价格数据"""
        crops_dir = os.path.join(PROJECT_ROOT, "agriculture_knowledge", "crops")
        if not os.path.exists(crops_dir):
            return
        for fname in os.listdir(crops_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(crops_dir, fname), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                crop_name = data.get("crop_name", "")
                market = data.get("market_info", {})
                if crop_name and crop_name not in self.price_data and market:
                    self.price_data[crop_name] = {
                        "range": (0, 0),
                        "unit": "元/kg",
                        "season_high": market.get("peak_season", ""),
                        "season_low": "",
                        "factors": market.get("price_factors", ""),
                        "storage": market.get("storage_tips", ""),
                    }
            except Exception:
                pass

    def get_price(self, crop: str) -> Dict[str, Any]:
        """获取指定作物的市场价格参考"""
        for name, info in self.price_data.items():
            if crop in name or name in crop:
                return {
                    "crop": name,
                    "price_low": info["range"][0],
                    "price_high": info["range"][1],
                    "unit": info["unit"],
                    "season_high": info.get("season_high", ""),
                    "season_low": info.get("season_low", ""),
                    "factors": info.get("factors", ""),
                    "storage": info.get("storage", ""),
                    "updated_at": datetime.now().strftime("%Y-%m-%d"),
                    "source": "参考数据",
                }
        return {
            "crop": crop,
            "price_low": 0,
            "price_high": 0,
            "unit": "元/kg",
            "updated_at": datetime.now().strftime("%Y-%m-%d"),
            "source": "暂无数据",
        }

    def get_all_prices(self) -> List[Dict]:
        """获取所有作物的价格参考"""
        results = []
        for name, info in self.price_data.items():
            results.append({
                "crop": name,
                "price_low": info["range"][0],
                "price_high": info["range"][1],
                "unit": info["unit"],
            })
        return results

    def estimate_revenue(self, crop: str, area_mu: float) -> Dict[str, Any]:
        """估算种植收益"""
        price_info = self.get_price(crop)
        avg_price = (price_info["price_low"] + price_info["price_high"]) / 2
        if avg_price == 0:
            return {"error": f"暂无{crop}的价格数据"}

        # 从知识库获取产量参考
        yield_range = (300, 500)  # 默认 kg/亩
        crops_dir = os.path.join(PROJECT_ROOT, "agriculture_knowledge", "crops")
        for fname in os.listdir(crops_dir) if os.path.exists(crops_dir) else []:
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(crops_dir, fname), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get("crop_name") == crop:
                    yi = data.get("yield_info", {})
                    med = yi.get("medium_yield", "")
                    import re
                    nums = re.findall(r'(\d+)', med)
                    if len(nums) >= 2:
                        yield_range = (int(nums[0]), int(nums[1]))
                    break
            except Exception:
                pass

        low_yield, high_yield = yield_range
        low_revenue = low_yield * price_info["price_low"] * area_mu
        high_revenue = high_yield * price_info["price_high"] * area_mu

        return {
            "crop": crop,
            "area_mu": area_mu,
            "avg_price": round(avg_price, 2),
            "price_unit": price_info["unit"],
            "yield_low": f"{low_yield} kg/亩",
            "yield_high": f"{high_yield} kg/亩",
            "revenue_low": round(low_revenue, 0),
            "revenue_high": round(high_revenue, 0),
            "avg_revenue": round((low_revenue + high_revenue) / 2, 0),
            "source": price_info["source"],
        }

    def format_market_report(self, crop: str = None) -> str:
        """格式化市场价格报告"""
        if crop:
            info = self.get_price(crop)
            if info["price_low"] == 0:
                return f"📊 **{crop}** 暂无市场价格参考数据。"
            lines = [
                f"📊 **{info['crop']} 市场价格参考**",
                f"价格区间: **{info['price_low']} - {info['price_high']} {info['unit']}**",
            ]
            if info.get("season_high"):
                lines.append(f"价格旺季: {info['season_high']}")
            if info.get("season_low"):
                lines.append(f"价格淡季: {info['season_low']}")
            if info.get("factors"):
                lines.append(f"影响因素: {info['factors']}")
            lines.append(f"更新时间: {info['updated_at']} ({info['source']})")
            return "\n".join(lines)

        lines = ["📊 **农产品市场价格参考**\n"]
        for name, info in sorted(self.price_data.items()):
            lines.append(
                f"- **{name}**: {info['range'][0]} - {info['range'][1]} {info['unit']}"
            )
        lines.append(f"\n*数据仅供参考，实际价格以当地市场为准*")
        return "\n".join(lines)


# 便捷函数
def get_market_service() -> MarketService:
    return MarketService()
