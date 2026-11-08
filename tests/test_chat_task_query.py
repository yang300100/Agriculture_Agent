"""哥哥，按日期查询任务时，只展示相应日期的真实记录。"""

from datetime import datetime, timedelta

from app.agent.nodes.progress import progress_tracking_node
from app.agent.state import AgentState
from core.planting_tracker import PlantingTracker
from core.storage_paths import DEFAULT_DATA_DIR
from pathlib import Path


def test_查询明天任务不会混入其他日期():
    username = "chat_tomorrow_filter"
    tracker = PlantingTracker(str(Path(DEFAULT_DATA_DIR) / username))
    tomorrow = (datetime.now() + timedelta(days=1)).date().isoformat()
    tracker.create_task({"crop": "小麦", "title": "明天巡田", "end_date": tomorrow})
    tracker.create_task({"crop": "小麦", "title": "其他日期任务", "end_date": "2099-01-01"})
    result = progress_tracking_node(AgentState(username=username, user_question="查看明天的农事任务"))
    assert "明天巡田" in result.final_answer
    assert "其他日期任务" not in result.final_answer
    assert tomorrow in result.final_answer
