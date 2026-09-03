"""测试模型推理执行器的同步事件循环桥接。"""

import asyncio

from core.model_executor import ModelExecutor
from models.base import ModelInput, ModelOutput


class _FakeRegistry:
    """返回固定成功结果的最小模拟注册中心。"""

    async def infer(self, model_id, model_input):
        await asyncio.sleep(0)
        return ModelOutput(
            success=True,
            model_id=model_id,
            predictions=[],
            inference_time_ms=1,
        )


def _model_input():
    return ModelInput(image_bytes=b"test")


def test_infer_sync_recovers_from_closed_current_loop():
    """线程残留已关闭事件循环时，图片推理仍应创建新循环完成。"""
    closed_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(closed_loop)
    closed_loop.close()

    try:
        result = ModelExecutor(_FakeRegistry()).infer_sync("image_model", _model_input())
    finally:
        asyncio.set_event_loop(None)

    assert result.success is True
    assert result.model_id == "image_model"


def test_infer_sync_inside_running_loop_uses_worker_thread():
    """调用线程已有运行中事件循环时，应在线程内完成同步推理。"""
    async def _run():
        return ModelExecutor(_FakeRegistry()).infer_sync("image_model", _model_input())

    result = asyncio.run(_run())

    assert result.success is True
    assert result.model_id == "image_model"
