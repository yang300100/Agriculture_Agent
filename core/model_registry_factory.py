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


def _ensure_backend_registered(registry, backend_name: str, device: str):
    """按需注册后端，允许模型权重格式覆盖配置中的首选后端"""
    if backend_name in registry._backends:
        return registry._backends[backend_name]

    if backend_name == "onnx" and ONNX_AVAILABLE:
        backend = ONNXBackend(device=device)
    elif backend_name == "torch" and TORCH_AVAILABLE:
        backend = TorchBackend(device=device)
    else:
        logger.warning("模型需要 %s 后端，但对应依赖未安装", backend_name)
        return None

    registry.register(backend_name, backend)
    logger.info("%s 后端已注册，设备: %s", backend_name, device)
    return backend


def _load_model_sync(registry, backend_instance, info, backend_name):
    """同步加载模型并更新注册表（处理异步循环冲突）"""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 当前线程没有运行中的事件循环，可以直接执行协程
        ok = asyncio.run(backend_instance.load_model(info))
    else:
        # 当前线程已有事件循环，在独立线程中执行，避免嵌套运行事件循环
        result = {"ok": False, "error": None}

        def _load():
            try:
                result["ok"] = asyncio.run(backend_instance.load_model(info))
            except Exception as exc:
                result["error"] = exc

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()
        thread.join(timeout=30)
        if thread.is_alive():
            logger.warning("模型 %s 加载超时，注册表未更新", info.model_id)
            return
        if result["error"] is not None:
            logger.error("模型 %s 加载失败: %s", info.model_id, result["error"])
            return
        ok = result["ok"]

    if ok:
        registry._model_map[info.model_id] = backend_name
        registry._model_info[info.model_id] = info
    else:
        logger.warning("模型 %s 加载失败，注册表未更新", info.model_id)


def get_model_registry() -> ModelRegistry:
    """获取全局模型注册中心（单例，线程安全）"""
    global _model_registry
    if _model_registry is None:
        with _registry_lock:
            # 双重检查：锁内再次检查防止竞态
            if _model_registry is None:
                _model_registry = setup_model_registry()
    return _model_registry


def reload_model_registry() -> ModelRegistry:
    """重新扫描权重并替换全局注册中心，供运行中新增模型后恢复。"""
    global _model_registry
    with _registry_lock:
        _model_registry = setup_model_registry()
        return _model_registry


def resolve_inference_model(requested_model_id: str = ""):
    """返回可用注册中心和模型 ID；配置模型缺失时自动重扫并安全回退。"""
    registry = get_model_registry()
    if requested_model_id and registry.get_model_info(requested_model_id):
        return registry, requested_model_id

    registry = reload_model_registry()
    if requested_model_id and registry.get_model_info(requested_model_id):
        return registry, requested_model_id

    models = registry.list_models()
    if models:
        fallback_id = models[0].model_id
        if requested_model_id:
            logger.warning("配置模型 %s 未加载，回退到 %s", requested_model_id, fallback_id)
        return registry, fallback_id

    configured = requested_model_id or "未指定"
    raise RuntimeError(
        f"本地图片模型未加载（配置: {configured}）。请检查 DL_MODELS_DIR、"
        "DL_DEFAULT_MODEL 和推理依赖，随后重启后端。"
    )


def setup_model_registry() -> ModelRegistry:
    """初始化模型注册中心"""
    registry = ModelRegistry()
    from app.agent.config import DL_BACKEND, DL_DEVICE, DL_MODELS_DIR
    backend_type = DL_BACKEND
    device = DL_DEVICE
    models_dir = DL_MODELS_DIR

    # 先注册配置指定的后端，发现其他格式权重时再按需注册对应后端
    _ensure_backend_registered(registry, backend_type, device)

    # 扫描预设对应的权重文件并加载
    for preset_id, preset in PRESETS.items():
        backend_name = preset.get("preferred_backend", "onnx")
        architecture = preset.get("model_architecture", "")

        # ── 确定权重文件扩展名和搜索路径 ──
        if architecture == "hmpd_net":
            # HMPD-Net 当前使用包含类别体系与多任务配置的 PyTorch 检查点。
            candidates = [f"{preset_id}.pth", f"{preset_id}.pt"]
        elif architecture == "convnext_v2":
            # ConvNeXt V2 支持 PyTorch state_dict 和 ONNX 两种格式。
            if backend_name == "onnx":
                candidates = [
                    f"{preset_id}.onnx",
                    f"{preset_id}.pth",
                    f"{preset_id}.pt",
                ]
            else:
                candidates = [f"{preset_id}.pth", f"{preset_id}.pt", f"{preset_id}.onnx"]
        else:
            # 旧版预设: .onnx (onnx后端) 或 .pt (torch后端)
            ext = ".onnx" if backend_name == "onnx" else ".pt"
            candidates = [f"{preset_id}{ext}"]

        # 搜索第一个存在的权重文件
        weight_path = None
        actual_backend = backend_name
        for candidate in candidates:
            candidate_path = os.path.join(models_dir, candidate)
            if os.path.exists(candidate_path):
                weight_path = candidate_path
                # 如果加载的是 .onnx 但原定 torch，自动切换后端
                if candidate.endswith(".onnx") and backend_name == "torch":
                    actual_backend = "onnx"
                # PyTorch 权重必须使用 torch 后端
                if candidate.endswith((".pth", ".pt")):
                    actual_backend = "torch"
                break

        # ConvNeXt V2: 额外检查 model_all/weights/ 下的默认输出名
        if weight_path is None and architecture == "convnext_v2":
            alt_dir = os.path.join(os.path.dirname(__file__), "..", "model_all", "weights")
            alt_dir = os.path.abspath(alt_dir)
            alt_candidates = ["pest_disease_model.onnx", "pest_disease_model.pth"]
            for alt_name in alt_candidates:
                alt_path = os.path.join(alt_dir, alt_name)
                if os.path.exists(alt_path):
                    weight_path = alt_path
                    actual_backend = "onnx" if alt_name.endswith(".onnx") else "torch"
                    logger.info("在 model_all/weights/ 发现权重: %s", alt_path)
                    break

        if weight_path is None:
            continue

        info = ModelInfo(
            model_id=preset_id,
            model_name=preset["model_name"],
            backend_name=actual_backend,
            capability=preset["capability"],
            model_path=os.path.abspath(weight_path),
            input_shape=preset.get("input_shape", (3, 224, 224)),
            classes=preset.get("classes", []),
            preprocessing=preset.get("preprocessing", {}),
            metadata={
                "model_architecture": architecture,
                "num_classes": preset.get("num_classes", 0),
                "dropout_rate": preset.get("dropout_rate", 0.3),
            },
        )

        # 同步加载模型（确保注册表映射与模型加载同步完成）
        backend_instance = _ensure_backend_registered(registry, actual_backend, device)
        if backend_instance:
            _load_model_sync(registry, backend_instance, info, actual_backend)

    return registry
