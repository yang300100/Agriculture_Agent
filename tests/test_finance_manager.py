"""测试 finance_manager 模块"""

import sys
import os
import tempfile
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.finance_manager import FinanceManager


class TestFinanceManager:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.manager = FinanceManager(storage_dir=self.tmp_dir)

    def test_add_cost(self):
        record = self.manager.add_cost({
            "crop": "小麦",
            "cost_type": "种子",
            "item_name": "优质种子",
            "quantity": 20,
            "unit": "斤",
            "unit_price": 5.0
        })
        assert record.crop == "小麦"
        assert record.total_amount == 100.0

    def test_add_cost_empty_crop_rejected(self):
        with pytest.raises(ValueError, match="作物名称"):
            self.manager.add_cost({
                "crop": "",
                "cost_type": "种子",
                "item_name": "",
                "quantity": 1,
                "unit": "项",
                "unit_price": 10.0
            })

    def test_add_cost_duplicate_rejected(self):
        self.manager.add_cost({
            "crop": "玉米", "cost_type": "肥料",
            "item_name": "复合肥", "quantity": 10, "unit": "kg", "unit_price": 3.0
        })
        with pytest.raises(ValueError, match="重复"):
            self.manager.add_cost({
                "crop": "玉米", "cost_type": "肥料",
                "item_name": "复合肥", "quantity": 10, "unit": "kg", "unit_price": 3.0
            })

    def test_add_income(self):
        record = self.manager.add_income({
            "crop": "小麦",
            "quantity": 2000,
            "unit_price": 2.5,
            "buyer": "收购站"
        })
        assert record.crop == "小麦"
        assert record.total_amount == 5000.0

    def test_add_income_empty_crop_rejected(self):
        with pytest.raises(ValueError, match="作物名称"):
            self.manager.add_income({
                "crop": "",
                "quantity": 100,
                "unit_price": 2.0
            })

    def test_get_crop_summary(self):
        self.manager.add_cost({
            "crop": "大豆", "cost_type": "种子",
            "item_name": "豆种", "quantity": 5, "unit": "kg", "unit_price": 10.0
        })
        self.manager.add_income({
            "crop": "大豆", "quantity": 300, "unit_price": 6.0
        })
        summary = self.manager.get_crop_financial_summary("大豆")
        assert summary is not None
        assert summary.total_cost == 50.0
        assert summary.total_income == 1800.0
        assert summary.net_profit == 1750.0

    def test_get_annual_report_empty(self):
        report = self.manager.get_annual_report()
        assert "crop_reports" in report

    def test_csv_export_import(self):
        self.manager.add_cost({
            "crop": "棉花", "cost_type": "农药",
            "item_name": "杀虫剂", "quantity": 2, "unit": "瓶", "unit_price": 50.0
        })
        export_path = os.path.join(self.tmp_dir, "export.csv")
        success = self.manager.export_to_csv(export_path)
        assert success
        assert os.path.exists(export_path)

    def test_validate_crop_name(self):
        assert self.manager._validate_crop_name("小麦") is True
        assert self.manager._validate_crop_name("") is False
        assert self.manager._validate_crop_name("   ") is False
