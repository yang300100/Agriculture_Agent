"""模型注册中心 - 管理多个推理后端和模型"""
import asyncio
import logging
from typing import Dict, List, Optional

from models.base import BaseModelBackend, ModelInfo, ModelInput, ModelOutput, ModelCapability

logger = logging.getLogger(__name__)


class ModelRegistry:
    """模型注册中心 — 与 DeviceDriverRegistry 对等设计"""

    def __init__(self):
        self._backends: Dict[str, BaseModelBackend] = {}       # backend_name → 后端实例
        self._model_map: Dict[str, str] = {}                    # model_id → backend_name
        self._model_info: Dict[str, ModelInfo] = {}             # model_id → ModelInfo

    def register(self, name: str, backend: BaseModelBackend):
        """注册推理后端"""
        if name in self._backends:
            old = self._backends[name]
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                pass
        self._backends[name] = backend

    async def discover_all(self) -> int:
        """发现所有后端注册的模型，原子更新映射表"""
        new_map = {}
        new_info = {}
        for name, backend in self._backends.items():
            try:
                models = await asyncio.wait_for(backend.discover_models(), timeout=30)
                for model in models:
                    new_map[model.model_id] = name
                    new_info[model.model_id] = model
            except asyncio.TimeoutError:
                logger.warning("后端 %s 模型发现超时", name)
            except Exception as e:
                logger.error("后端 %s 模型发现失败: %s", name, e)
        self._model_map = new_map
        self._model_info = new_info
        return len(self._model_map)

    async def infer(self, model_id: str, model_input: ModelInput) -> ModelOutput:
        """路由推理请求到对应后端"""
        backend_name = self._model_map.get(model_id)
        if backend_name is None:
            return ModelOutput.error(model_id, "MODEL_NOT_FOUND")
        backend = self._backends.get(backend_name)
        if backend is None:
            return ModelOutput.error(model_id, "BACKEND_NOT_FOUND")
        return await backend.infer(model_id, model_input)

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        return self._model_info.get(model_id)

    def list_models(self) -> List[ModelInfo]:
        return list(self._model_info.values())

    def get_models_by_capability(self, cap: ModelCapability) -> List[ModelInfo]:
        return [m for m in self._model_info.values() if m.capability == cap]

    def unregister(self, name: str):
        """移除后端及其所有模型"""
        if name in self._backends:
            self._backends.pop(name)
            self._model_map = {k: v for k, v in self._model_map.items() if v != name}
            self._model_info = {}

    @property
    def backend_names(self) -> List[str]:
        return list(self._backends.keys())

    @property
    def model_count(self) -> int:
        return len(self._model_map)
