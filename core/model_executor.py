"""模型推理执行器 — 重试 + 超时 + 同步桥接"""
import time
import logging
import asyncio
from typing import Optional

from models.base import ModelInput, ModelOutput
from models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelExecutor:
    """模型推理执行器，在 Registry 之上叠加重试/超时"""

    def __init__(self, registry: ModelRegistry, max_retries: int = 2, timeout_ms: int = 30000):
        self.registry = registry
        self.max_retries = max_retries
        self.timeout_ms = timeout_ms

    async def infer(self, model_id: str, model_input: ModelInput) -> ModelOutput:
        last_result = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self.registry.infer(model_id, model_input),
                    timeout=self.timeout_ms / 1000,
                )
                if result.success:
                    return result
                last_result = result
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "模型推理失败(attempt %d/%d): %s, %d秒后重试",
                        attempt + 1, self.max_retries + 1, result.error_code, wait,
                    )
                    await asyncio.sleep(wait)
            except asyncio.TimeoutError:
                logger.error("模型推理超时(attempt %d/%d): %s", attempt + 1, self.max_retries + 1, model_id)
                last_result = ModelOutput.error(model_id, "TIMEOUT")
        return last_result or ModelOutput.error(model_id, "MAX_RETRIES_EXCEEDED")

    def infer_sync(self, model_id: str, model_input: ModelInput) -> ModelOutput:
        """同步推理桥接"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.infer(model_id, model_input))
            finally:
                loop.close()
        return loop.run_until_complete(self.infer(model_id, model_input))
