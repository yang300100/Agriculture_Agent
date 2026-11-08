"""哥哥，任务领取到结果落库期间持有跨进程锁，退出时由系统释放。"""

import inspect
import os
from functools import wraps


def guard_device_task(handler):
    signature = inspect.signature(handler)

    @wraps(handler)
    def guarded(*args, **kwargs):
        from core.planting_tracker import PlantingTracker
        from core.storage_paths import DEFAULT_DATA_DIR

        values = signature.bind(*args, **kwargs)
        values.apply_defaults()
        username = values.arguments.get("username", "default")
        if not str(values.arguments["tid"]).isdigit():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="任务不存在")
        tracker = PlantingTracker(os.path.join(DEFAULT_DATA_DIR, username))
        lock = tracker.device_task_execution_lock(str(values.arguments["tid"]))
        if not lock.acquire():
            if handler.__name__ == "execute_task":
                return {"success": False, "decision": "executing", "error": "任务正在处理，请稍后刷新状态"}
            from fastapi import HTTPException
            raise HTTPException(status_code=409, detail="任务正在处理，请稍后刷新状态")
        try:
            return handler(*args, **kwargs)
        finally:
            lock.release()

    return guarded
