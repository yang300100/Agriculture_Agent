"""深度学习模型推理接口

支持 ONNX Runtime 和 PyTorch 两种推理后端。
通过 ModelRegistry 注册中心统一管理模型的发现、加载和推理。
"""
from models.base import BaseModelBackend, ModelInfo, ModelInput, ModelOutput, Prediction, ModelCapability
from models.registry import ModelRegistry

_ONNX_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import onnxruntime  # noqa: F401
    _ONNX_AVAILABLE = True
except ImportError:
    pass

try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def is_onnx_available() -> bool:
    return _ONNX_AVAILABLE


def is_torch_available() -> bool:
    return _TORCH_AVAILABLE
