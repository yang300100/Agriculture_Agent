"""设备操作日志与待确认队列 API。"""

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.device_registry_factory import close_registry, setup_registry


logger = logging.getLogger(__name__)


class PendingActionUpdate(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)


def register_device_action_routes(app: FastAPI) -> None:
    """注册设备操作队列接口，集中维护错误语义和资源释放。"""

    @app.get("/api/actions/log")
    def get_action_log(limit: int = 50, username: str = "default"):
        try:
            from core.device_executor import DeviceExecutor
            from devices.registry import DeviceDriverRegistry
            from devices.simulator_driver import SimulatorDriver

            registry = DeviceDriverRegistry()
            registry.register("simulator", SimulatorDriver())
            return DeviceExecutor(registry, username=username).get_logs(limit=limit)
        except Exception as exc:
            logger.exception("读取设备操作日志失败: user=%s", username)
            raise HTTPException(
                status_code=500, detail="设备操作日志暂时无法读取"
            ) from exc

    @app.get("/api/actions/pending")
    def get_pending_actions(username: str = "default"):
        try:
            from core.device_executor import DeviceExecutor
            from devices.registry import DeviceDriverRegistry
            from devices.simulator_driver import SimulatorDriver

            registry = DeviceDriverRegistry()
            registry.register("simulator", SimulatorDriver())
            return DeviceExecutor(registry, username=username).list_pending()
        except Exception as exc:
            logger.exception("读取待确认操作失败: user=%s", username)
            raise HTTPException(
                status_code=500, detail="待确认操作暂时无法读取"
            ) from exc

    @app.put("/api/actions/{action_id}")
    def update_pending_action(
        action_id: str,
        data: PendingActionUpdate,
        username: str = "default",
    ):
        try:
            from core.device_executor import DeviceExecutor
            from devices.registry import DeviceDriverRegistry

            executor = DeviceExecutor(DeviceDriverRegistry(), username=username)
            result = executor.update_pending(action_id, data.params)
            if not result.get("success"):
                raise HTTPException(
                    status_code=409,
                    detail=result.get("message", "待确认操作无法修改"),
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("修改待确认操作失败: %s", action_id)
            raise HTTPException(
                status_code=500, detail="待确认操作修改失败"
            ) from exc

    @app.post("/api/actions/{action_id}/confirm")
    def confirm_action(action_id: str, username: str = "default"):
        try:
            from core.device_executor import DeviceExecutor

            registry, loop = setup_registry(username)
            try:
                loop.run_until_complete(registry.discover_all())
                result = DeviceExecutor(
                    registry, username=username
                ).confirm_pending(action_id)
                if (
                    not result.get("success")
                    and result.get("message") == "操作不存在或已处理"
                ):
                    raise HTTPException(status_code=409, detail=result["message"])
                return result
            finally:
                close_registry(loop, registry)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("确认待确认操作失败: %s", action_id)
            raise HTTPException(
                status_code=500, detail="待确认操作执行失败"
            ) from exc

    @app.post("/api/actions/{action_id}/reject")
    def reject_action(action_id: str, username: str = "default"):
        try:
            from core.device_executor import DeviceExecutor

            registry, loop = setup_registry(username)
            try:
                ok = DeviceExecutor(
                    registry, username=username
                ).reject_pending(action_id)
                if not ok:
                    raise HTTPException(status_code=409, detail="操作不存在或已处理")
                return {"success": True}
            finally:
                close_registry(loop, registry)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("拒绝待确认操作失败: %s", action_id)
            raise HTTPException(
                status_code=500, detail="待确认操作拒绝失败"
            ) from exc

