"""模型注册中心工厂 — 初始化 + 自动发现"""
import os
import logging
from typing import Optional

from models.registry import ModelRegistry
from models.base import ModelInfo
from models.onnx_backend import ONNXBackend, ONNX_AVAILABLE
from models.torch_backend import TorchBackend, TORCH_AVAILABLE
from models.presets import PRESETS

logger = logging.getLogger(__name__)

_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """获取全局模型注册中心（单例）"""
    global _model_registry
    if _model_registry is None:
        _model_registry = setup_model_registry()
    return _model_registry


def setup_model_registry() -> ModelRegistry:
    """初始化模型注册中心"""
    registry = ModelRegistry()
    backend_type = os.getenv("DL_BACKEND", "onnx")
    device = os.getenv("DL_DEVICE", "cpu")
    models_dir = os.getenv("DL_MODELS_DIR", "models/weights")

    # 注册后端
    if backend_type == "onnx" and ONNX_AVAILABLE:
        registry.register("onnx", ONNXBackend(device=device))
        logger.info("ONNX后端已注册，设备: %s", device)
    elif backend_type == "onnx" and not ONNX_AVAILABLE:
        logger.warning("ONNX后端配置但onnxruntime未安装，跳过")

    if backend_type == "torch" and TORCH_AVAILABLE:
        registry.register("torch", TorchBackend(device=device))
        logger.info("Torch后端已注册，设备: %s", device)
    elif backend_type == "torch" and not TORCH_AVAILABLE:
        logger.warning("Torch后端配置但PyTorch未安装，跳过")

    # 扫描预设对应的权重文件并加载
    for preset_id, preset in PRESETS.items():
        backend = preset.get("preferred_backend", "onnx")
        ext = ".onnx" if backend == "onnx" else ".pt"
        weight_path = os.path.join(models_dir, f"{preset_id}{ext}")
        if os.path.exists(weight_path):
            info = ModelInfo(
                model_id=preset_id,
                model_name=preset["model_name"],
                backend_name=backend,
                capability=preset["capability"],
                model_path=os.path.abspath(weight_path),
                input_shape=preset.get("input_shape", (3, 224, 224)),
                classes=preset.get("classes", []),
                preprocessing=preset.get("preprocessing", {}),
            )
            # 同步加载模型
            backend_instance = registry._backends.get(backend)
            if backend_instance:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import threading
                        def _load():
                            new_loop = asyncio.new_event_loop()
                            new_loop.run_until_complete(backend_instance.load_model(info))
                            new_loop.close()
                        threading.Thread(target=_load, daemon=True).start()
                    else:
                        loop.run_until_complete(backend_instance.load_model(info))
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(backend_instance.load_model(info))
                    loop.close()
                registry._model_map[info.model_id] = backend
                registry._model_info[info.model_id] = info

    return registry
