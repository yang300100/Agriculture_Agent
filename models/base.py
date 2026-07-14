"""DL模型接口的抽象基类与数据结构"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Tuple


class ModelCapability(Enum):
    DISEASE_CLASSIFY = "disease_classify"
    CROP_IDENTIFY = "crop_identify"
    PEST_DETECT = "pest_detect"
    SEVERITY_ASSESS = "severity_assess"


@dataclass
class ModelInfo:
    model_id: str
    model_name: str
    backend_name: str                          # "onnx" | "torch"
    capability: ModelCapability
    model_path: str                            # 权重文件路径
    input_shape: Tuple[int, int, int] = (3, 224, 224)
    classes: List[str] = field(default_factory=list)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInput:
    image_bytes: bytes
    top_k: int = 3
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    class_name: str
    confidence: float
    index: int


@dataclass
class ModelOutput:
    success: bool
    model_id: str
    predictions: List[Prediction]
    inference_time_ms: float
    error_code: str = ""
    raw_output: Any = None

    @classmethod
    def error(cls, model_id: str, error_code: str, message: str = "") -> "ModelOutput":
        return cls(success=False, model_id=model_id, predictions=[], inference_time_ms=0, error_code=error_code)


class BaseModelBackend(ABC):
    backend_name: str = "base"

    @abstractmethod
    async def load_model(self, model_info: ModelInfo) -> bool:
        ...

    @abstractmethod
    async def unload_model(self, model_id: str) -> None:
        ...

    @abstractmethod
    async def infer(self, model_id: str, model_input: ModelInput) -> ModelOutput:
        ...

    @abstractmethod
    async def discover_models(self) -> List[ModelInfo]:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...
