"""模型注册中心工厂 — 初始化 + 自动发现"""
import os
import logging
import threading
from typing import Optional

from models.registry import ModelRegistry
from models.base import ModelInfo
from models.onnx_backend import ONNXBackend, ONNX_AVAILABLE
from models.torch_backend import TorchBackend, TORCH_AVAILABLE
from models.presets import PRESETS

logger = logging.getLogger(__name__)

_model_registry: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_model_registry() -> ModelRegistry:
    """获取全局模型注册中心（单例，线程安全）"""
    global _model_registry
    if _model_registry is None:
        with _registry_lock:
            # 双重检查：锁内再次检查防止竞态
            if _model_registry is None:
                _model_registry = setup_model_registry()
    return _model_registry


def setup_model_registry() -> ModelRegistry:
    """初始化模型注册中心"""
    registry = ModelRegistry()
    from app.agent.config import DL_BACKEND, DL_DEVICE, DL_MODELS_DIR
    backend_type = DL_BACKEND
    device = DL_DEVICE
    models_dir = DL_MODELS_DIR

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
            # 同步加载模型（确保注册表映射与模型加载同步完成）
            backend_instance = registry._backends.get(backend)
            if backend_instance:
                import asyncio
                loaded = [False]  # 用列表包装以便在线程中修改
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # 在独立线程中加载模型，加载完成后更新映射
                        def _load():
                            new_loop = asyncio.new_event_loop()
                            try:
                                ok = new_loop.run_until_complete(backend_instance.load_model(info))
                                if ok:
                                    registry._model_map[info.model_id] = backend
                                    registry._model_info[info.model_id] = info
                                    loaded[0] = True
                            finally:
                                new_loop.close()
                        t = threading.Thread(target=_load, daemon=True)
                        t.start()
                        t.join(timeout=10)  # 等待最多10秒加载完成
                        if not loaded[0]:
                            logger.warning("模型 %s 加载超时或失败，注册表未更新", info.model_id)
                    else:
                        ok = loop.run_until_complete(backend_instance.load_model(info))
                        if ok:
                            registry._model_map[info.model_id] = backend
                            registry._model_info[info.model_id] = info
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    try:
                        ok = loop.run_until_complete(backend_instance.load_model(info))
                        if ok:
                            registry._model_map[info.model_id] = backend
                            registry._model_info[info.model_id] = info
                    finally:
                        loop.close()

    return registry
