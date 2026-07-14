"""地块管理器 — 管理种植地块的坐标、作物、面积，为设备提供精确的天气定位。

每个地块有独立的经纬度坐标，设备通过 plot_id 绑定到地块。
巡检时按地块获取天气，而非通过设备 location 字符串 geocode。
"""

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# 内置中国城市坐标（用于设备 location 字面量匹配 → 自动创建地块）
CITY_COORDS = {
    "北京": (39.90, 116.40), "上海": (31.23, 121.47), "广州": (23.13, 113.26),
    "深圳": (22.54, 114.06), "成都": (30.67, 104.07), "武汉": (30.60, 114.30),
    "杭州": (30.28, 120.15), "南京": (32.06, 118.78), "西安": (34.26, 108.94),
    "重庆": (29.57, 106.55), "天津": (39.12, 117.19), "沈阳": (41.80, 123.43),
    "哈尔滨": (45.80, 126.53), "郑州": (34.76, 113.66), "济南": (36.67, 116.98),
    "长沙": (28.23, 112.97), "昆明": (24.88, 102.83), "福州": (26.07, 119.30),
    "南宁": (22.82, 108.37), "海口": (20.03, 110.32), "石家庄": (38.04, 114.47),
    "太原": (37.87, 112.55), "呼和浩特": (40.84, 111.75), "长春": (43.90, 125.32),
    "合肥": (31.86, 117.28), "南昌": (28.68, 115.86), "贵阳": (26.57, 106.71),
    "兰州": (36.06, 103.83), "西宁": (36.62, 101.78), "银川": (38.47, 106.27),
    "乌鲁木齐": (43.82, 87.62), "拉萨": (29.65, 91.13),
    "河北": (38.04, 114.47), "华北": (39.90, 116.40), "东北": (43.90, 125.32),
    "华东": (32.06, 118.78), "华中": (30.60, 114.30), "华南": (23.13, 113.26),
    "西南": (30.67, 104.07), "西北": (36.06, 103.83),
}


class PlotManager:
    """地块管理器 — 每个用户独立的地块配置"""

    def __init__(self, username: str = "default"):
        self.username = username
        self._plot_path = os.path.join(DEFAULT_DATA_DIR, username, "plots.json")
        os.makedirs(os.path.dirname(self._plot_path), exist_ok=True)

    # ── CRUD ──────────────────────────────────────

    def list_plots(self) -> List[Dict]:
        """列出所有地块"""
        if not os.path.exists(self._plot_path):
            return []
        try:
            with open(self._plot_path, "r", encoding="utf-8") as f:
                plots = json.load(f)
            return plots if isinstance(plots, list) else []
        except Exception:
            return []

    def get_plot(self, plot_id: str) -> Optional[Dict]:
        """获取单个地块"""
        for p in self.list_plots():
            if p.get("plot_id") == plot_id:
                return p
        return None

    def save_plots(self, plots: List[Dict]):
        """保存地块列表（原子写入 + DB同步）"""
        tmp = self._plot_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(plots, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._plot_path)
        # DB同步（地块坐标信息同步到fields表的plot相关字段）
        try:
            from core.database.repository.users import UserRepository
            user_repo = UserRepository()
            user = user_repo.get_by_username(self.username)
            if user:
                from core.database.repository.fields import FieldRepository
                repo = FieldRepository()
                existing = repo.find_by(user_id=user.id)
                existing_names = {f.name for f in existing}
                for p in plots:
                    if p.get("name") not in existing_names:
                        repo.create(
                            user_id=user.id,
                            name=p.get("name", p.get("plot_id", "")),
                            coordinates=json.dumps([[p.get("lon", 0), p.get("lat", 0)]], ensure_ascii=False),
                            center_lat=p.get("lat"),
                            center_lon=p.get("lon"),
                            area_mu=p.get("area_mu", 0),
                            current_crop=p.get("crop", ""),
                        )
        except Exception as e:
            logger.debug("数据库同步地块失败: %s", e)

    def add_plot(self, plot_id: str, name: str, lat: float, lon: float,
                 crop: str = "", area_mu: float = 0.0) -> Dict:
        """添加地块"""
        plots = self.list_plots()
        for p in plots:
            if p["plot_id"] == plot_id:
                raise ValueError(f"地块ID '{plot_id}' 已存在")
        plot = {
            "plot_id": plot_id,
            "name": name,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "crop": crop,
            "area_mu": area_mu,
        }
        plots.append(plot)
        self.save_plots(plots)
        logger.info("地块已添加: %s (%s) → (%.4f, %.4f)", plot_id, name, lat, lon)
        return plot

    def update_plot(self, plot_id: str, **kwargs) -> Optional[Dict]:
        """更新地块"""
        plots = self.list_plots()
        for p in plots:
            if p["plot_id"] == plot_id:
                for k in ("name", "lat", "lon", "crop", "area_mu"):
                    if k in kwargs and kwargs[k] is not None:
                        p[k] = kwargs[k]
                self.save_plots(plots)
                return p
        return None

    def delete_plot(self, plot_id: str) -> bool:
        """删除地块"""
        plots = self.list_plots()
        new_plots = [p for p in plots if p["plot_id"] != plot_id]
        if len(new_plots) < len(plots):
            self.save_plots(new_plots)
            return True
        return False

    # ── 坐标解析 ──────────────────────────────────

    def get_coords(self, plot_id: str = None) -> Optional[tuple]:
        """获取地块坐标 → (lat, lon)，无匹配时返回 None"""
        if plot_id:
            plot = self.get_plot(plot_id)
            if plot:
                return (plot["lat"], plot["lon"])
        return None

    def resolve_coords(self, plot_id: str = None, location: str = "") -> tuple:
        """智能解析坐标。

        优先级: plot_id 匹配 > location 内置城市匹配 > 本机 IP 定位
        """
        # 1. 地块ID精确匹配
        if plot_id:
            coords = self.get_coords(plot_id)
            if coords:
                return coords

        # 2. location 内置城市匹配
        if location:
            for name, coords in CITY_COORDS.items():
                if name in location:
                    return coords

        # 3. 本机 IP 定位
        from core.weather_service import get_local_coords
        return get_local_coords()

    # ── 设备→地块关联 ─────────────────────────────

    def get_devices_for_plot(self, plot_id: str,
                             all_devices: List) -> List:
        """获取属于某地块的所有设备"""
        return [d for d in all_devices
                if getattr(d, 'metadata', {}).get('plot_id', '') == plot_id
                or getattr(d, 'location', '') == plot_id]

    def auto_create_from_devices(self, devices_config: List[Dict]) -> int:
        """从设备配置自动创建地块（设备 location 作为地块名）。

        只为尚未存在的地块创建，已有地块不覆盖。
        """
        existing_ids = {p["plot_id"] for p in self.list_plots()}
        created = 0

        for dev in devices_config:
            location = dev.get("location", "").strip()
            plot_id = dev.get("plot_id", "").strip() or location

            if not plot_id or plot_id in existing_ids:
                continue

            # 尝试从 location 匹配坐标
            coords = None
            for name, c in CITY_COORDS.items():
                if name in location:
                    coords = c
                    break
            if not coords:
                from core.weather_service import get_local_coords
                coords = get_local_coords()

            self.add_plot(
                plot_id=plot_id,
                name=location or plot_id,
                lat=coords[0], lon=coords[1],
                crop="",
                area_mu=0.0,
            )
            existing_ids.add(plot_id)
            created += 1

        return created
