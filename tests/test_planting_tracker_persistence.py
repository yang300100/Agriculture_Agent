"""验证哥哥的农事记录在真实数据库中稳定保存。"""

from datetime import datetime, timedelta

import pytest

from core.database.repository.planting import PlantingTaskRepository
from core.planting_tracker import PlantingTracker


def test_created_task_returns_the_persisted_id(tmp_path):
    tracker = PlantingTracker(str(tmp_path / "task_returned_id"))

    task = tracker.create_task({"crop": "小麦", "title": "施肥"})

    persisted = tracker.get_task_by_id(task.id)
    assert persisted is not None
    assert persisted.title == "施肥"


def test_task_mutations_preserve_other_ids_and_creation_dates(tmp_path):
    tracker = PlantingTracker(str(tmp_path / "task_stable_a"))
    other = PlantingTracker(str(tmp_path / "task_stable_b"))
    tracker.create_task({"crop": "小麦", "title": "施肥"})
    first = tracker.get_tasks()[0]
    other.create_task({"crop": "玉米", "title": "除草"})
    tracker.create_task({"crop": "小麦", "title": "浇水"})

    saved = {task.title: task for task in tracker.get_tasks()}
    assert saved["施肥"].id == first.id
    assert saved["施肥"].created_at == first.created_at
    watering = saved["浇水"]
    assert tracker.update_task_status(first.id, "已完成", 100) is True
    assert tracker.get_task_by_id(watering.id).title == "浇水"
    assert tracker.delete_task(first.id) is True
    assert tracker.get_task_by_id(watering.id).title == "浇水"
    assert tracker.get_task_by_id(watering.id).created_at == watering.created_at


def test_task_device_parameters_survive_database_roundtrip(tmp_path):
    tracker = PlantingTracker(str(tmp_path / "task_device_params"))
    tracker.create_task({
        "crop": "小麦", "title": "模拟灌溉", "device_id": "mock-pump",
        "device_command": "irrigate", "device_params": {"duration": 60, "zone": "北区"},
    })

    task = tracker.get_tasks()[0]
    assert task.device_params == {"duration": 60, "zone": "北区"}
    tracker.update_task_status(task.id, "进行中", 40)
    assert tracker.get_task_by_id(task.id).device_params == {"duration": 60, "zone": "北区"}


@pytest.mark.parametrize("stored,expected", [
    ('{"duration": 60}', {"duration": 60}),
    ("invalid-json", None),
    ('[1, 2]', None),
])
def test_task_legacy_device_parameters_are_decoded_safely(tmp_path, stored, expected):
    tracker = PlantingTracker(str(tmp_path / ("legacy_params_" + str(len(stored)))))
    row = PlantingTaskRepository().create(
        user_id=tracker._uid, crop="小麦", title="旧灌溉任务", device_params=stored,
    )

    assert tracker.get_task_by_id(str(row.id)).device_params == expected


def test_task_progress_and_completion_are_persisted(tmp_path):
    tracker = PlantingTracker(str(tmp_path / "task_progress"))
    tracker.create_task({"crop": "小麦", "title": "收获", "progress_percent": 25})
    task = tracker.get_tasks()[0]
    assert task.progress_percent == 25

    tracker.update_task_status(task.id, "进行中", 40)
    assert tracker.get_task_by_id(task.id).progress_percent == 40
    tracker.update_task_status(task.id, "已完成")
    assert tracker.get_task_cards()[0]["progress"] == 100
    assert tracker.get_task_by_id(task.id).completed_date
    assert PlantingTaskRepository().get_by_id(int(task.id)).progress_percent == 100


def test_progress_id_and_notes_survive_database_roundtrip(tmp_path):
    tracker = PlantingTracker(str(tmp_path / "progress_returned_id"))

    progress = tracker.create_progress({
        "crop": "小麦", "stage": "准备期", "start_date": "2099-01-01",
        "notes": "北坡地块，采用滴灌",
    })

    saved = tracker.get_progress()[0]
    assert saved.id == progress.id
    assert saved.notes == "北坡地块，采用滴灌"
    tracker.update_progress(saved.id, {"notes": "完成土壤检测"})
    assert tracker.get_progress()[0].notes == "完成土壤检测"


def test_progress_mutations_preserve_other_record_ids(tmp_path):
    tracker = PlantingTracker(str(tmp_path / "progress_stable_a"))
    other = PlantingTracker(str(tmp_path / "progress_stable_b"))
    tracker.create_progress({"crop": "小麦", "start_date": "2099-01-01"})
    first = tracker.get_progress()[0]
    other.create_progress({"crop": "玉米"})
    tracker.create_progress({"crop": "水稻", "start_date": "2099-01-01"})

    saved = {progress.crop: progress for progress in tracker.get_progress()}
    assert saved["小麦"].id == first.id
    assert saved["小麦"].created_at == first.created_at
    rice = saved["水稻"]
    tracker.update_progress(first.id, {"progress_percent": 40})
    assert tracker._load_progress_by_id(rice.id).crop == "水稻"
    assert tracker.delete_progress(first.id) is True
    assert tracker._load_progress_by_id(rice.id).crop == "水稻"
    assert tracker._load_progress_by_id(rice.id).created_at == rice.created_at


def test_foreign_record_ids_cannot_modify_or_delete_records(tmp_path):
    owner = PlantingTracker(str(tmp_path / "tracker_owner"))
    visitor = PlantingTracker(str(tmp_path / "tracker_visitor"))
    owner.create_task({"crop": "小麦", "title": "施肥"})
    owner.create_progress({"crop": "小麦", "stage": "准备期", "total_stages": 3})
    task = owner.get_tasks()[0]
    progress = owner.get_progress()[0]

    assert visitor.update_task_status(task.id, "已完成", 100) is False
    assert visitor.delete_task(task.id) is False
    visitor.update_progress(progress.id, {"crop": "水稻"})
    assert visitor.delete_progress(progress.id) is False
    assert owner.get_task_by_id(task.id).status == "待办"
    assert owner._load_progress_by_id(progress.id).crop == "小麦"


def test_auto_calculation_keeps_progress_id_and_notes(tmp_path):
    tracker = PlantingTracker(str(tmp_path / "progress_auto_id"))
    other = PlantingTracker(str(tmp_path / "progress_auto_other"))
    tracker.create_progress({
        "crop": "测试作物", "stage": "准备期", "stage_number": 1,
        "start_date": "2020-01-01", "notes": "留存旧记录",
    })
    saved = tracker.get_progress()[0]
    other.create_progress({"crop": "玉米"})

    result = tracker.auto_calculate_progress(saved.id)

    assert result["success"] is True
    refreshed = tracker._load_progress_by_id(saved.id)
    assert refreshed is not None
    assert refreshed.status == "已完成"
    assert refreshed.notes == "留存旧记录"


def test_recent_manual_stage_change_is_not_immediately_overwritten(tmp_path):
    tracker = PlantingTracker(str(tmp_path / "progress_recent_manual"))
    progress = tracker.create_progress({
        "crop": "秋季轮作试验", "stage": "准备期", "stage_number": 1,
        "total_stages": 8,
        "start_date": (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d"),
    })

    result = tracker.advance_to_next_stage(progress.id)
    assert result["stage_number"] == 4

    card = tracker.get_progress_cards()[0]
    assert card["stage_number"] == 4
    assert card["progress"] == 0
