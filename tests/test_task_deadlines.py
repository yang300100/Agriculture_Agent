"""哥哥，按真实日期验证逾期任务及剩余天数，不依赖后台调度器。"""

from datetime import datetime, timedelta

from core.database.repository.planting import PlantingTaskRepository
from core.planting_tracker import PlantingTracker


def test_未完成任务过期后列表立即显示逾期且不篡改数据库状态(tmp_path):
    tracker = PlantingTracker(str(tmp_path / "task_deadline_read"))
    yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
    task = tracker.create_task({"crop": "小麦", "title": "麦田巡查", "end_date": yesterday})
    card = tracker.get_task_cards()[0]
    assert card["status"] == "已逾期"
    assert card["status_color"] == "red"
    assert card["days_left"] == 0
    assert tracker.get_tasks()[0].status == "已逾期"
    assert PlantingTaskRepository().get_by_id(int(task.id)).status == "待办"


def test_按日历日期计算截止天数且完成任务不会变回逾期(tmp_path):
    tracker = PlantingTracker(str(tmp_path / "task_deadline_days"))
    today = datetime.now().date()
    for offset, title in [(0, "今天截止"), (1, "明天截止"), (-1, "已经完成")]:
        task = tracker.create_task({
            "crop": "小麦", "title": title, "end_date": (today + timedelta(days=offset)).isoformat(),
        })
        if offset < 0:
            tracker.update_task_status(task.id, "已完成")
    cards = {card["title"]: card for card in tracker.get_task_cards()}
    assert cards["今天截止"]["status"] == "待办"
    assert cards["今天截止"]["days_left"] == 0
    assert cards["明天截止"]["days_left"] == 1
    assert cards["已经完成"]["status"] == "已完成"
    assert cards["已经完成"]["progress"] == 100
