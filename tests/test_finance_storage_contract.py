"""帮哥哥检查数据库记录能按成本、收入契约读回来。"""

from uuid import uuid4

from core.finance_manager import FinanceManager


def test_reloaded_cost_records_match_cost_contract(tmp_path):
    storage_dir = str(tmp_path / f"finance_cost_contract_{uuid4().hex}")
    FinanceManager(storage_dir).add_cost({
        "date": "2026-03-15", "crop": "小麦", "plot": "北地",
        "cost_type": "种子", "item_name": "优质种子", "quantity": 20,
        "unit": "斤", "unit_price": 5, "notes": "春播",
    })

    records = FinanceManager(storage_dir).get_cost_records(
        crop="小麦", plot="北地", start_date="2026-03-01", end_date="2026-03-31",
    )

    assert len(records) == 1
    record = records[0]
    assert (record.cost_type, record.item_name, record.unit) == ("种子", "优质种子", "斤")
    assert (record.quantity, record.unit_price, record.total_amount) == (20, 5, 100)
    assert (record.date, record.notes) == ("2026-03-15", "春播")


def test_reloaded_income_records_match_income_contract(tmp_path):
    storage_dir = str(tmp_path / f"finance_income_contract_{uuid4().hex}")
    FinanceManager(storage_dir).add_income({
        "date": "2026-06-20", "crop": "小麦", "plot": "北地",
        "income_type": "销售", "quantity": 2000, "unit_price": 2.8,
        "buyer": "粮食收购站", "notes": "夏收",
    })

    records = FinanceManager(storage_dir).get_income_records(
        crop="小麦", plot="北地", start_date="2026-06-01", end_date="2026-06-30",
    )

    assert len(records) == 1
    record = records[0]
    assert (record.income_type, record.buyer) == ("销售", "粮食收购站")
    assert (record.quantity, record.unit_price, record.total_amount) == (2000, 2.8, 5600)
    assert (record.date, record.notes) == ("2026-06-20", "夏收")
