"""哥哥，对话业务卡片接口不提供设备直接执行入口。"""

from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.chat_cards import get_device_card, get_task_card, prepare_device_action


class PrepareChatDeviceAction(BaseModel):
    command: Literal["start", "stop"]
    params: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(min_length=1, max_length=200, pattern=r"^\S+$")


def register_chat_card_routes(app: FastAPI) -> None:
    """哥哥，设备与任务读取都由服务按当前请求账号验证归属。"""

    @app.get("/api/chat/cards/device/{device_id}")
    def read_device_card(device_id: str, username: str = "default", pending_action_id: str | None = None):
        return get_device_card(username, device_id, pending_action_id)

    @app.get("/api/chat/cards/task/{task_id}")
    def read_task_card(task_id: str, username: str = "default"):
        return get_task_card(username, task_id)

    @app.post("/api/chat/cards/device/{device_id}/prepare")
    def prepare_card_action(device_id: str, data: PrepareChatDeviceAction, username: str = "default"):
        return prepare_device_action(username, device_id, data.command, data.params, data.request_id)
