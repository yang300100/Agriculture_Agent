"""
地图管理模块 - 支持地块多边形绘制、定位与面积计算
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from shapely.geometry import Polygon
import math


class FieldBoundary(BaseModel):
    """地块边界数据模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "未命名地块"  # 地块名称
    center_lat: float = 0.0  # 中心点纬度
    center_lon: float = 0.0  # 中心点经度
    coordinates: List[List[float]] = Field(default_factory=list)  # 多边形坐标 [[lon, lat], ...]
    area_mu: float = 0.0  # 面积（亩）
    area_m2: float = 0.0  # 面积（平方米）
    soil_type: str = ""  # 土壤类型
    current_crop: str = ""  # 当前种植作物
    created_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    updated_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class MapManager:
    """地图管理器 - 处理地块的增删改查和面积计算"""

    DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fields.json")

    def __init__(self):
        self.fields: List[FieldBoundary] = []
        self._ensure_data_dir()
        self._load_data()

    def _ensure_data_dir(self):
        """确保数据目录存在"""
        data_dir = os.path.dirname(self.DATA_FILE)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def _load_data(self):
        """从JSON文件加载地块数据"""
        if os.path.exists(self.DATA_FILE):
            try:
                with open(self.DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.fields = [FieldBoundary(**item) for item in data]
            except Exception as e:
                print(f"加载地块数据失败: {e}")
                self.fields = []

    def _save_data(self):
        """保存地块数据到JSON文件"""
        try:
            with open(self.DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump([field.model_dump() for field in self.fields], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存地块数据失败: {e}")

    @staticmethod
    def calculate_area(coordinates: List[List[float]]) -> Tuple[float, float]:
        """
        计算多边形面积

        Args:
            coordinates: 坐标列表 [[lon, lat], [lon, lat], ...] (GeoJSON格式)

        Returns:
            (area_m2, area_mu): 平方米和亩
        """
        if len(coordinates) < 3:
            return 0.0, 0.0

        # 转换为Shapely Polygon (lon, lat) -> (x, y)
        # 注意：Shapely使用平面坐标系，需要将球面坐标转换为近似平面坐标
        polygon = Polygon(coordinates)

        # 计算中心点用于更准确的面积计算
        center_lon = sum(coord[0] for coord in coordinates) / len(coordinates)
        center_lat = sum(coord[1] for coord in coordinates) / len(coordinates)

        # 使用Haversine公式近似计算面积
        # 在中心纬度处，1度经度对应的米数
        meters_per_deg_lon = 111320 * math.cos(math.radians(center_lat))
        meters_per_deg_lat = 111320

        # 将经纬度坐标转换为近似米坐标
        points_m = []
        for lon, lat in coordinates:
            x = (lon - center_lon) * meters_per_deg_lon
            y = (lat - center_lat) * meters_per_deg_lat
            points_m.append((x, y))

        # 使用Shoelace公式计算面积
        area_m2 = 0.0
        n = len(points_m)
        for i in range(n):
            j = (i + 1) % n
            area_m2 += points_m[i][0] * points_m[j][1]
            area_m2 -= points_m[j][0] * points_m[i][1]
        area_m2 = abs(area_m2) / 2.0

        # 转换为亩 (1亩 = 666.67平方米)
        area_mu = area_m2 / 666.67

        return area_m2, area_mu

    @staticmethod
    def calculate_center(coordinates: List[List[float]]) -> Tuple[float, float]:
        """
        计算多边形中心点

        Returns:
            (lat, lon): 中心点纬度和经度
        """
        if not coordinates:
            return 0.0, 0.0

        # 使用多边形质心或简单平均值
        if len(coordinates) >= 3:
            try:
                polygon = Polygon(coordinates)
                centroid = polygon.centroid
                return centroid.y, centroid.x  # lat, lon
            except:
                pass

        # 降级到平均值
        avg_lon = sum(coord[0] for coord in coordinates) / len(coordinates)
        avg_lat = sum(coord[1] for coord in coordinates) / len(coordinates)
        return avg_lat, avg_lon

    def create_field(self, name: str, coordinates: List[List[float]],
                     soil_type: str = "", current_crop: str = "") -> FieldBoundary:
        """
        创建新地块

        Args:
            name: 地块名称
            coordinates: 多边形坐标 [[lon, lat], ...]
            soil_type: 土壤类型
            current_crop: 当前作物

        Returns:
            FieldBoundary: 创建的地块对象
        """
        # 计算面积和中心点
        area_m2, area_mu = self.calculate_area(coordinates)
        center_lat, center_lon = self.calculate_center(coordinates)

        field = FieldBoundary(
            name=name,
            coordinates=coordinates,
            center_lat=center_lat,
            center_lon=center_lon,
            area_m2=area_m2,
            area_mu=area_mu,
            soil_type=soil_type,
            current_crop=current_crop
        )

        self.fields.append(field)
        self._save_data()
        return field

    def update_field(self, field_id: str, **kwargs) -> Optional[FieldBoundary]:
        """
        更新地块信息

        Args:
            field_id: 地块ID
            **kwargs: 要更新的字段

        Returns:
            FieldBoundary: 更新后的地块，如果不存在返回None
        """
        for field in self.fields:
            if field.id == field_id:
                # 如果更新了坐标，重新计算面积和中心点
                if 'coordinates' in kwargs:
                    coords = kwargs['coordinates']
                    kwargs['area_m2'], kwargs['area_mu'] = self.calculate_area(coords)
                    kwargs['center_lat'], kwargs['center_lon'] = self.calculate_center(coords)

                # 更新时间戳
                kwargs['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 更新字段
                for key, value in kwargs.items():
                    if hasattr(field, key):
                        setattr(field, key, value)

                self._save_data()
                return field
        return None

    def delete_field(self, field_id: str) -> bool:
        """
        删除地块

        Args:
            field_id: 地块ID

        Returns:
            bool: 是否删除成功
        """
        for i, field in enumerate(self.fields):
            if field.id == field_id:
                del self.fields[i]
                self._save_data()
                return True
        return False

    def get_field(self, field_id: str) -> Optional[FieldBoundary]:
        """获取单个地块"""
        for field in self.fields:
            if field.id == field_id:
                return field
        return None

    def get_all_fields(self) -> List[FieldBoundary]:
        """获取所有地块"""
        return self.fields

    def get_total_area(self) -> float:
        """获取所有地块总面积（亩）"""
        return sum(field.area_mu for field in self.fields)

    def get_field_count(self) -> int:
        """获取地块数量"""
        return len(self.fields)

    def format_field_info(self, field: FieldBoundary) -> str:
        """格式化地块信息为文本"""
        lines = [
            f"🌾 **{field.name}**",
            f"   📍 中心位置: {field.center_lat:.4f}°N, {field.center_lon:.4f}°E",
            f"   📐 面积: {field.area_mu:.2f}亩 ({field.area_m2:.0f}平方米)",
        ]
        if field.soil_type:
            lines.append(f"   🌍 土壤类型: {field.soil_type}")
        if field.current_crop:
            lines.append(f"   🌱 当前作物: {field.current_crop}")
        return "\n".join(lines)

    def get_fields_by_crop(self, crop: str) -> List[FieldBoundary]:
        """按作物筛选地块"""
        return [f for f in self.fields if crop in f.current_crop]

    def get_fields_summary(self) -> Dict[str, Any]:
        """获取地块汇总信息"""
        if not self.fields:
            return {
                "count": 0,
                "total_area_mu": 0.0,
                "fields": []
            }

        return {
            "count": len(self.fields),
            "total_area_mu": self.get_total_area(),
            "fields": [
                {
                    "id": f.id,
                    "name": f.name,
                    "area_mu": f.area_mu,
                    "crop": f.current_crop
                }
                for f in self.fields
            ]
        }


# Folium地图组件相关函数

def create_folium_map(center_lat: float = 39.9, center_lon: float = 116.4,
                      zoom: int = 12, drawn_shapes: List[Dict] = None) -> "folium.Map":
    """
    创建Folium地图，集成绘制工具

    Args:
        center_lat: 中心纬度
        center_lon: 中心经度
        zoom: 缩放级别
        drawn_shapes: 已绘制的形状数据

    Returns:
        folium.Map: 地图对象
    """
    import folium
    from folium.plugins import Draw, LocateControl

    # 创建地图
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap"
    )

    # 添加绘制工具
    draw = Draw(
        draw_options={
            "polyline": False,
            "polygon": True,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        },
        edit_options={
            "edit": True,
            "remove": True,
        }
    )
    draw.add_to(m)

    # 添加定位控件 - 使用更明显的配置
    locate = LocateControl(
        position="topright",
        strings={
            "title": "定位到我的位置 📍",
            "popup": "您的当前位置"
        },
        locate_options={
            "enableHighAccuracy": True,
            "timeout": 10000,
            "maximumAge": 0
        },
        flyTo=True,  # 定位后自动飞移到位置
        returnToPrevBounds=True
    )
    locate.add_to(m)

    # 添加图层选择
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="卫星影像",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="标准地图",
        overlay=False,
        control=True
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # 添加已有地块显示
    if drawn_shapes:
        for shape in drawn_shapes:
            coords = shape.get("coordinates", [])
            if coords:
                # 转换坐标格式 [lon, lat] -> [lat, lon] for folium
                folium_coords = [[c[1], c[0]] for c in coords]
                folium.Polygon(
                    locations=folium_coords,
                    popup=shape.get("name", "地块"),
                    color="blue",
                    fill=True,
                    fill_color="blue",
                    fill_opacity=0.3
                ).add_to(m)

    return m


def extract_polygon_from_map_data(map_data: Dict) -> Optional[List[List[float]]]:
    """
    从streamlit-folium返回的数据中提取多边形坐标

    Args:
        map_data: st_folium返回的数据

    Returns:
        List[List[float]]: 多边形坐标 [[lon, lat], ...]，如果没有则返回None
    """
    if not map_data:
        return None

    # 检查最后绘制的形状
    last_drawing = map_data.get("last_active_drawing")
    if last_drawing:
        geometry = last_drawing.get("geometry", {})
        geom_type = geometry.get("type", "")

        if geom_type == "Polygon":
            # GeoJSON格式: coordinates[0] 是外环 [[lon, lat], ...]
            coords = geometry.get("coordinates", [[]])[0]
            if len(coords) >= 3:
                return coords

    # 检查所有绘制的形状
    all_drawings = map_data.get("all_drawings", [])
    if all_drawings:
        last = all_drawings[-1]
        geometry = last.get("geometry", {})
        if geometry.get("type") == "Polygon":
            coords = geometry.get("coordinates", [[]])[0]
            if len(coords) >= 3:
                return coords

    return None


def get_location_from_address(address: str) -> Optional[Tuple[float, float]]:
    """
    将地址转换为坐标（需要geopy）

    Args:
        address: 地址字符串

    Returns:
        (lat, lon): 纬度和经度
    """
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="agri_policy_qa_agent")
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"地址解析失败: {e}")
    return None


def format_coordinates_for_display(coordinates: List[List[float]]) -> str:
    """格式化坐标用于显示"""
    if not coordinates:
        return "未绘制"

    # 显示边界框信息
    lons = [c[0] for c in coordinates]
    lats = [c[1] for c in coordinates]

    return f"顶点数: {len(coordinates)}, 经度: {min(lons):.4f}~{max(lons):.4f}, 纬度: {min(lats):.4f}~{max(lats):.4f}"
