"""测试种植阶段推进的持久化语义。"""

from core.planting_tracker import PlantingTracker


def test_手动完成阶段后保留记录id并更新阶段(tmp_path):
    tracker = PlantingTracker(str(tmp_path / "progress_user"))
    created = tracker.create_progress({
        "crop": "测试作物",
        "stage": "准备期",
        "stage_number": 1,
        "total_stages": 3,
        "progress_percent": 20,
        "status": "进行中",
    })
    persisted = tracker.get_progress()[0]

    result = tracker.advance_to_next_stage(persisted.id)
    refreshed = tracker.get_progress()[0]

    assert result["success"] is True
    assert refreshed.id == persisted.id
    assert refreshed.stage_number == 2
    assert refreshed.stage == "播种期"
    assert refreshed.progress_percent == 0


def test_最后阶段完成后记录标记为已完成(tmp_path):
    tracker = PlantingTracker(str(tmp_path / "completed_progress_user"))
    tracker.create_progress({
        "crop": "测试作物",
        "stage": "收获期",
        "stage_number": 1,
        "total_stages": 1,
        "status": "进行中",
    })
    persisted = tracker.get_progress()[0]

    result = tracker.advance_to_next_stage(persisted.id)
    refreshed = tracker.get_progress()[0]

    assert result["is_completed"] is True
    assert refreshed.id == persisted.id
    assert refreshed.status == "已完成"
    assert refreshed.progress_percent == 100
    assert refreshed.actual_end_date
