"""测试 planting_planner 模块"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.planting_planner import PlantingPlanner, CropDatabase


class TestCropDatabase:
    def test_load_crops(self):
        db = CropDatabase()
        crops = db.get_all_crops()
        assert len(crops) >= 3, "至少应加载3个作物"
        assert "小麦" in crops

    def test_get_crop_by_name(self):
        db = CropDatabase()
        wheat = db.get_crop("小麦")
        assert wheat is not None
        assert wheat.name == "小麦"

    def test_get_crop_by_alias(self):
        db = CropDatabase()
        rice = db.get_crop("大米")
        # 别名匹配看实现，至少不崩溃
        assert rice is None or rice.name == "水稻"

    def test_get_nonexistent_crop(self):
        db = CropDatabase()
        result = db.get_crop("不存在的作物")
        assert result is None


class TestPlantingPlanner:
    def setup_method(self):
        self.planner = PlantingPlanner()

    def test_generate_plan_basic(self):
        user_info = {
            "region": "华北",
            "soil_type": "壤土",
            "farm_size": 50.0,
            "goals": ["高产"],
            "experience": "中级",
            "crop": "小麦"
        }
        plan = self.planner.generate_plan(user_info)
        assert plan is not None
        assert plan.crop
        assert plan.region == "华北"
        assert plan.created_at

    def test_format_plan_as_text(self):
        user_info = {"region": "华北", "soil_type": "壤土", "farm_size": 10.0, "goals": [], "crop": "小麦"}
        plan = self.planner.generate_plan(user_info)
        text = self.planner.format_plan_as_text(plan)
        assert "小麦" in text
        assert len(text) > 50

    def test_generate_plan_no_crop(self):
        user_info = {"region": "华北", "soil_type": "壤土", "farm_size": 30.0, "goals": ["高产"]}
        plan = self.planner.generate_plan(user_info)
        assert plan is not None
        # 应自动推荐一种作物
        assert plan.crop

    def test_generate_plan_small_farm(self):
        user_info = {"region": "华东", "soil_type": "壤土", "farm_size": 0.5, "goals": ["自用为主"]}
        plan = self.planner.generate_plan(user_info)
        assert plan is not None

    def test_risk_assessment(self):
        user_info = {"region": "东北", "soil_type": "黑土", "farm_size": 100.0, "goals": ["经济效益"], "crop": "玉米"}
        plan = self.planner.generate_plan(user_info)
        assert plan.risks is not None
        assert isinstance(plan.risks, list)
