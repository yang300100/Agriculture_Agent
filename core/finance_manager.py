"""
财务管理模块
功能：
- 种植成本记录与核算
- 收入记录与收益计算
- 财务报表生成
- 支持从CSV/Excel文件导入历史数据
- 多作物、多地块成本对比
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import dotenv

dotenv.load_dotenv()
DEFAULT_STORAGE_DIR = os.getenv("DATA_STORAGE_DIR", "data")


class CostType(Enum):
    """成本类型"""
    SEED = "种子"
    FERTILIZER = "肥料"
    PESTICIDE = "农药"
    MACHINERY = "农机"
    LABOR = "人工"
    IRRIGATION = "水电"
    LAND_RENT = "地租"
    OTHER = "其他"


class IncomeType(Enum):
    """收入类型"""
    SALES = "销售"
    SUBSIDY = "补贴"
    OTHER = "其他"


@dataclass
class CostRecord:
    """成本记录"""
    id: str
    date: str
    crop: str
    plot: str  # 地块名称
    cost_type: str
    item_name: str
    quantity: float
    unit: str
    unit_price: float
    total_amount: float
    notes: str
    created_at: str


@dataclass
class IncomeRecord:
    """收入记录"""
    id: str
    date: str
    crop: str
    plot: str
    income_type: str
    quantity: float  # 产量(kg)
    unit_price: float  # 单价(元/kg)
    total_amount: float
    buyer: str  # 收购方
    notes: str
    created_at: str


@dataclass
class CropFinancialSummary:
    """作物财务汇总"""
    crop: str
    plot: str
    year: str
    total_cost: float
    total_income: float
    net_profit: float
    profit_per_mu: float
    yield_per_mu: float
    cost_breakdown: Dict[str, float]  # 各项成本占比


class FinanceStorage:
    """财务数据存储"""

    def __init__(self, storage_dir: str = None):
        self.storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        self.costs_file = os.path.join(self.storage_dir, "finance_costs.json")
        self.income_file = os.path.join(self.storage_dir, "finance_income.json")
        self._ensure_storage()

    def _ensure_storage(self):
        """确保存储目录和文件存在"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        for file_path in [self.costs_file, self.income_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([], f)

    def load_costs(self) -> List[Dict]:
        """加载所有成本记录"""
        try:
            with open(self.costs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载成本记录失败: {e}")
            return []

    def save_costs(self, costs: List[Dict]):
        """保存成本记录"""
        with open(self.costs_file, 'w', encoding='utf-8') as f:
            json.dump(costs, f, ensure_ascii=False, indent=2)

    def load_income(self) -> List[Dict]:
        """加载所有收入记录"""
        try:
            with open(self.income_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载收入记录失败: {e}")
            return []

    def save_income(self, income: List[Dict]):
        """保存收入记录"""
        with open(self.income_file, 'w', encoding='utf-8') as f:
            json.dump(income, f, ensure_ascii=False, indent=2)


class FinanceManager:
    """财务管理主类"""

    def __init__(self, storage_dir: str = None):
        self.storage = FinanceStorage(storage_dir)

    def add_cost(self, cost_data: Dict[str, Any]) -> CostRecord:
        """
        添加成本记录

        Args:
            cost_data: {
                "date": "2024-03-15",
                "crop": "小麦",
                "plot": "地块A",
                "cost_type": "种子",
                "item_name": "优质小麦种子",
                "quantity": 20,
                "unit": "斤",
                "unit_price": 5.5,
                "notes": ""
            }

        Returns:
            CostRecord对象
        """
        total = cost_data.get("quantity", 0) * cost_data.get("unit_price", 0)

        record = CostRecord(
            id=self._generate_id(),
            date=cost_data.get("date", datetime.now().strftime("%Y-%m-%d")),
            crop=cost_data.get("crop", ""),
            plot=cost_data.get("plot", "默认地块"),
            cost_type=cost_data.get("cost_type", "其他"),
            item_name=cost_data.get("item_name", ""),
            quantity=cost_data.get("quantity", 0),
            unit=cost_data.get("unit", ""),
            unit_price=cost_data.get("unit_price", 0),
            total_amount=total,
            notes=cost_data.get("notes", ""),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        costs = self.storage.load_costs()
        costs.append(asdict(record))
        self.storage.save_costs(costs)

        return record

    def add_income(self, income_data: Dict[str, Any]) -> IncomeRecord:
        """
        添加收入记录

        Args:
            income_data: {
                "date": "2024-06-20",
                "crop": "小麦",
                "plot": "地块A",
                "income_type": "销售",
                "quantity": 2000,
                "unit_price": 2.8,
                "buyer": "粮食收购站",
                "notes": ""
            }
        """
        total = income_data.get("quantity", 0) * income_data.get("unit_price", 0)

        record = IncomeRecord(
            id=self._generate_id(),
            date=income_data.get("date", datetime.now().strftime("%Y-%m-%d")),
            crop=income_data.get("crop", ""),
            plot=income_data.get("plot", "默认地块"),
            income_type=income_data.get("income_type", "销售"),
            quantity=income_data.get("quantity", 0),
            unit_price=income_data.get("unit_price", 0),
            total_amount=total,
            buyer=income_data.get("buyer", ""),
            notes=income_data.get("notes", ""),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        income = self.storage.load_income()
        income.append(asdict(record))
        self.storage.save_income(income)

        return record

    def get_crop_financial_summary(self, crop: str, plot: str = None,
                                   year: str = None) -> Optional[CropFinancialSummary]:
        """
        获取作物财务汇总

        Args:
            crop: 作物名称
            plot: 地块名称（可选）
            year: 年份（可选，默认当前年）

        Returns:
            CropFinancialSummary对象
        """
        year = year or datetime.now().strftime("%Y")

        # 获取成本记录
        costs = self.storage.load_costs()
        crop_costs = [c for c in costs
                     if c.get("crop") == crop
                     and c.get("date", "").startswith(year)]
        if plot:
            crop_costs = [c for c in crop_costs if c.get("plot") == plot]

        # 获取收入记录
        income = self.storage.load_income()
        crop_income = [i for i in income
                      if i.get("crop") == crop
                      and i.get("date", "").startswith(year)]
        if plot:
            crop_income = [i for i in crop_income if i.get("plot") == plot]

        if not crop_costs and not crop_income:
            return None

        # 计算总成本
        total_cost = sum(c.get("total_amount", 0) for c in crop_costs)

        # 成本明细
        cost_breakdown = {}
        for c in crop_costs:
            cost_type = c.get("cost_type", "其他")
            cost_breakdown[cost_type] = cost_breakdown.get(cost_type, 0) + c.get("total_amount", 0)

        # 计算总收入
        total_income = sum(i.get("total_amount", 0) for i in crop_income)

        # 计算产量
        total_yield = sum(i.get("quantity", 0) for i in crop_income)

        # 获取种植面积（从成本记录中估算）
        plot_size = self._estimate_plot_size(crop_costs)

        return CropFinancialSummary(
            crop=crop,
            plot=plot or "全部地块",
            year=year,
            total_cost=total_cost,
            total_income=total_income,
            net_profit=total_income - total_cost,
            profit_per_mu=(total_income - total_cost) / plot_size if plot_size > 0 else 0,
            yield_per_mu=total_yield / plot_size if plot_size > 0 else 0,
            cost_breakdown=cost_breakdown
        )

    def get_annual_report(self, year: str = None) -> Dict[str, Any]:
        """
        生成年度财务报告

        Args:
            year: 年份（默认当前年）

        Returns:
            年度报告数据
        """
        year = year or datetime.now().strftime("%Y")

        costs = self.storage.load_costs()
        income = self.storage.load_income()

        year_costs = [c for c in costs if c.get("date", "").startswith(year)]
        year_income = [i for i in income if i.get("date", "").startswith(year)]

        # 按作物统计
        crops = set(c.get("crop") for c in year_costs + year_income)
        crop_reports = []

        for crop in crops:
            summary = self.get_crop_financial_summary(crop, year=year)
            if summary:
                crop_reports.append(asdict(summary))

        # 总体统计
        total_cost = sum(c.get("total_amount", 0) for c in year_costs)
        total_income = sum(i.get("total_amount", 0) for i in year_income)

        # 月度统计
        monthly_data = {}
        for month in range(1, 13):
            month_str = f"{year}-{month:02d}"
            month_costs = sum(c.get("total_amount", 0) for c in year_costs if c.get("date", "").startswith(month_str))
            month_income = sum(i.get("total_amount", 0) for i in year_income if i.get("date", "").startswith(month_str))
            monthly_data[month_str] = {
                "cost": month_costs,
                "income": month_income,
                "profit": month_income - month_costs
            }

        return {
            "year": year,
            "total_cost": total_cost,
            "total_income": total_income,
            "net_profit": total_income - total_cost,
            "crop_reports": crop_reports,
            "monthly_data": monthly_data,
            "cost_count": len(year_costs),
            "income_count": len(year_income)
        }

    def import_from_csv(self, file_path: str, data_type: str = "cost") -> Dict[str, Any]:
        """
        从CSV文件导入数据

        成本CSV格式：
        date,crop,plot,cost_type,item_name,quantity,unit,unit_price,notes
        2024-03-15,小麦,地块A,种子,优质种子,20,斤,5.5,备注

        收入CSV格式：
        date,crop,plot,income_type,quantity,unit_price,buyer,notes
        2024-06-20,小麦,地块A,销售,2000,2.8,收购站,备注

        Args:
            file_path: CSV文件路径
            data_type: "cost" 或 "income"

        Returns:
            导入结果统计
        """
        imported = 0
        failed = 0
        errors = []

        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if data_type == "cost":
                            self.add_cost({
                                "date": row.get("date", ""),
                                "crop": row.get("crop", ""),
                                "plot": row.get("plot", "默认地块"),
                                "cost_type": row.get("cost_type", ""),
                                "item_name": row.get("item_name", ""),
                                "quantity": float(row.get("quantity", 0)),
                                "unit": row.get("unit", ""),
                                "unit_price": float(row.get("unit_price", 0)),
                                "notes": row.get("notes", "")
                            })
                        else:
                            self.add_income({
                                "date": row.get("date", ""),
                                "crop": row.get("crop", ""),
                                "plot": row.get("plot", "默认地块"),
                                "income_type": row.get("income_type", "销售"),
                                "quantity": float(row.get("quantity", 0)),
                                "unit_price": float(row.get("unit_price", 0)),
                                "buyer": row.get("buyer", ""),
                                "notes": row.get("notes", "")
                            })
                        imported += 1
                    except Exception as e:
                        failed += 1
                        errors.append(f"行 {imported + failed}: {str(e)}")

            return {
                "success": True,
                "imported": imported,
                "failed": failed,
                "errors": errors[:10]  # 最多返回10条错误
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def export_to_csv(self, file_path: str, data_type: str = "cost",
                     year: str = None, crop: str = None) -> bool:
        """
        导出数据到CSV文件

        Args:
            file_path: 导出文件路径
            data_type: "cost" 或 "income"
            year: 筛选年份
            crop: 筛选作物

        Returns:
            是否成功
        """
        try:
            if data_type == "cost":
                data = self.storage.load_costs()
                fieldnames = ["date", "crop", "plot", "cost_type", "item_name",
                            "quantity", "unit", "unit_price", "total_amount", "notes"]
            else:
                data = self.storage.load_income()
                fieldnames = ["date", "crop", "plot", "income_type", "quantity",
                            "unit_price", "total_amount", "buyer", "notes"]

            # 筛选
            if year:
                data = [d for d in data if d.get("date", "").startswith(year)]
            if crop:
                data = [d for d in data if d.get("crop") == crop]

            with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

            return True

        except Exception as e:
            print(f"导出失败: {e}")
            return False

    def get_cost_records(self, crop: str = None, plot: str = None,
                        start_date: str = None, end_date: str = None) -> List[CostRecord]:
        """获取成本记录列表"""
        costs = self.storage.load_costs()

        if crop:
            costs = [c for c in costs if c.get("crop") == crop]
        if plot:
            costs = [c for c in costs if c.get("plot") == plot]
        if start_date:
            costs = [c for c in costs if c.get("date", "") >= start_date]
        if end_date:
            costs = [c for c in costs if c.get("date", "") <= end_date]

        return [CostRecord(**c) for c in costs]

    def get_income_records(self, crop: str = None, plot: str = None,
                          start_date: str = None, end_date: str = None) -> List[IncomeRecord]:
        """获取收入记录列表"""
        income = self.storage.load_income()

        if crop:
            income = [i for i in income if i.get("crop") == crop]
        if plot:
            income = [i for i in income if i.get("plot") == plot]
        if start_date:
            income = [i for i in income if i.get("date", "") >= start_date]
        if end_date:
            income = [i for i in income if i.get("date", "") <= end_date]

        return [IncomeRecord(**i) for i in income]

    def delete_cost(self, record_id: str) -> bool:
        """删除成本记录"""
        costs = self.storage.load_costs()
        original_count = len(costs)
        costs = [c for c in costs if c.get("id") != record_id]
        if len(costs) < original_count:
            self.storage.save_costs(costs)
            return True
        return False

    def delete_income(self, record_id: str) -> bool:
        """删除收入记录"""
        income = self.storage.load_income()
        original_count = len(income)
        income = [i for i in income if i.get("id") != record_id]
        if len(income) < original_count:
            self.storage.save_income(income)
            return True
        return False

    def _generate_id(self) -> str:
        """生成唯一ID"""
        return f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"

    def _estimate_plot_size(self, costs: List[Dict]) -> float:
        """从成本记录估算种植面积"""
        # 简单估算：种子成本通常占总成本的3-5%，根据种子用量反推面积
        seed_costs = [c for c in costs if c.get("cost_type") == "种子"]
        if seed_costs:
            # 假设小麦种子15-20斤/亩，玉米3-5斤/亩
            total_seed = sum(c.get("quantity", 0) for c in seed_costs)
            crop = costs[0].get("crop", "") if costs else ""
            if "玉米" in crop:
                return total_seed / 4  # 估算
            else:
                return total_seed / 17  # 默认按小麦估算
        return 1.0  # 默认1亩

    def format_summary_report(self, summary: CropFinancialSummary) -> str:
        """格式化汇总报告"""
        text = f"""📊 **{summary.crop} ({summary.plot}) - {summary.year}年度财务汇总**

💰 **收益情况**
   总收入: ¥{summary.total_income:.2f}
   总成本: ¥{summary.total_cost:.2f}
   净利润: ¥{summary.net_profit:.2f} ({"盈利" if summary.net_profit >= 0 else "亏损"})

📈 **亩均效益**
   亩均收益: ¥{summary.profit_per_mu:.2f}/亩
   亩产量: {summary.yield_per_mu:.2f}kg/亩

💸 **成本构成**"""

        for cost_type, amount in summary.cost_breakdown.items():
            percentage = (amount / summary.total_cost * 100) if summary.total_cost > 0 else 0
            text += f"\n   {cost_type}: ¥{amount:.2f} ({percentage:.1f}%)"

        return text

    def format_annual_report(self, report: Dict[str, Any]) -> str:
        """格式化年度报告"""
        text = f"""📊 **{report['year']}年度种植财务报告**

💰 **总体情况**
   总收入: ¥{report['total_income']:.2f}
   总成本: ¥{report['total_cost']:.2f}
   净利润: ¥{report['net_profit']:.2f}

🌾 **各作物收益情况**
"""
        for crop_report in report['crop_reports']:
            emoji = "✅" if crop_report['net_profit'] >= 0 else "❌"
            text += f"   {emoji} {crop_report['crop']}: ¥{crop_report['net_profit']:.2f} "
            text += f"(亩均¥{crop_report['profit_per_mu']:.2f})\n"

        # 找出收益最高和最低的月份
        monthly = report.get('monthly_data', {})
        if monthly:
            profits = [(m, d['profit']) for m, d in monthly.items()]
            if profits:
                best_month = max(profits, key=lambda x: x[1])
                worst_month = min(profits, key=lambda x: x[1])
                text += f"\n📅 **收支月度分布**\n"
                text += f"   最佳月份: {best_month[0]} (¥{best_month[1]:.2f})\n"
                text += f"   最差月份: {worst_month[0]} (¥{worst_month[1]:.2f})\n"

        return text


# 便捷函数
def quick_add_cost(crop: str, cost_type: str, amount: float, plot: str = "默认地块") -> str:
    """快速添加成本"""
    manager = FinanceManager()
    record = manager.add_cost({
        "crop": crop,
        "plot": plot,
        "cost_type": cost_type,
        "item_name": f"{cost_type}支出",
        "quantity": 1,
        "unit": "项",
        "unit_price": amount,
        "notes": ""
    })
    return f"已记录{crop}的{cost_type}成本 ¥{amount}"


def quick_add_income(crop: str, quantity: float, unit_price: float, plot: str = "默认地块") -> str:
    """快速添加收入"""
    manager = FinanceManager()
    record = manager.add_income({
        "crop": crop,
        "plot": plot,
        "income_type": "销售",
        "quantity": quantity,
        "unit_price": unit_price,
        "buyer": "",
        "notes": ""
    })
    return f"已记录{crop}销售收入 ¥{record.total_amount:.2f}"


def get_crop_profit(crop: str, year: str = None) -> str:
    """获取作物盈亏情况"""
    manager = FinanceManager()
    summary = manager.get_crop_financial_summary(crop, year=year)
    if summary:
        return manager.format_summary_report(summary)
    return f"未找到{crop}的财务记录"


if __name__ == "__main__":
    # 测试
    manager = FinanceManager()

    # 添加成本记录
    manager.add_cost({
        "date": "2024-03-15",
        "crop": "小麦",
        "plot": "地块A",
        "cost_type": "种子",
        "item_name": "优质小麦种子",
        "quantity": 100,
        "unit": "斤",
        "unit_price": 4.5,
        "notes": "春播用种"
    })

    manager.add_cost({
        "date": "2024-03-20",
        "crop": "小麦",
        "plot": "地块A",
        "cost_type": "肥料",
        "item_name": "复合肥",
        "quantity": 200,
        "unit": "kg",
        "unit_price": 3.2,
        "notes": "基肥"
    })

    # 添加收入记录
    manager.add_income({
        "date": "2024-06-25",
        "crop": "小麦",
        "plot": "地块A",
        "income_type": "销售",
        "quantity": 2500,
        "unit_price": 2.6,
        "buyer": "粮食收购站",
        "notes": "一等麦"
    })

    # 生成报告
    summary = manager.get_crop_financial_summary("小麦", "地块A", "2024")
    if summary:
        print(manager.format_summary_report(summary))

    # 生成年度报告
    report = manager.get_annual_report("2024")
    print(manager.format_annual_report(report))
