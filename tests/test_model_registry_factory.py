"""测试模型注册中心对不同权重格式的发现与后端选择。"""
import asyncio
import sys
import types

from models.base import ModelCapability, ModelInfo
from models.registry import ModelRegistry
import core.model_registry_factory as factory


class _FakeBackend:
    """不读取真实权重的模拟后端。"""

    def __init__(self, device="cpu"):
        self.device = device
        self.loaded = []

    async def load_model(self, model_info):
        self.loaded.append(model_info)
        return True


def _setup_with_fake_weight(
    monkeypatch,
    tmp_path,
    extension,
    model_id="plant_village_38",
    architecture="convnext_v2",
):
    """使用空权重文件验证发现链路，不执行真实模型推理。"""
    (tmp_path / f"{model_id}{extension}").touch()

    fake_config = types.SimpleNamespace(
        DL_BACKEND="onnx",
        DL_DEVICE="cpu",
        DL_MODELS_DIR=str(tmp_path),
    )
    monkeypatch.setitem(sys.modules, "app.agent.config", fake_config)
    monkeypatch.setattr(factory, "ONNX_AVAILABLE", True)
    monkeypatch.setattr(factory, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(factory, "ONNXBackend", _FakeBackend)
    monkeypatch.setattr(factory, "TorchBackend", _FakeBackend)
    monkeypatch.setattr(factory, "PRESETS", {
        model_id: {
            "model_name": "测试模型",
            "capability": ModelCapability.DISEASE_CLASSIFY,
            "preferred_backend": "onnx",
            "model_architecture": architecture,
            "classes": ["健康", "病害"],
        }
    })

    return factory.setup_model_registry()


def test_pth_weight_registers_torch_backend(monkeypatch, tmp_path):
    """即使首选 ONNX，发现 .pth 后也应按需注册 Torch 并加载。"""
    registry = _setup_with_fake_weight(monkeypatch, tmp_path, ".pth")

    info = registry.get_model_info("plant_village_38")
    assert info is not None
    assert info.backend_name == "torch"
    assert "torch" in registry.backend_names


def test_pt_weight_registers_torch_backend(monkeypatch, tmp_path):
    """兼容使用 .pt 后缀保存的 state_dict 权重。"""
    registry = _setup_with_fake_weight(monkeypatch, tmp_path, ".pt")

    info = registry.get_model_info("plant_village_38")
    assert info is not None
    assert info.backend_name == "torch"


def test_onnx_weight_keeps_onnx_backend(monkeypatch, tmp_path):
    """发现 ONNX 权重时仍保持原有 ONNX 路径。"""
    registry = _setup_with_fake_weight(monkeypatch, tmp_path, ".onnx")

    info = registry.get_model_info("plant_village_38")
    assert info is not None
    assert info.backend_name == "onnx"


def test_hmpd_checkpoint_registers_torch_backend(monkeypatch, tmp_path):
    """HMPD-Net 的 .pth 检查点应被发现并交给 Torch 后端。"""
    registry = _setup_with_fake_weight(
        monkeypatch,
        tmp_path,
        ".pth",
        model_id="hmpd_net",
        architecture="hmpd_net",
    )

    info = registry.get_model_info("hmpd_net")
    assert info is not None
    assert info.backend_name == "torch"
    assert info.metadata["model_architecture"] == "hmpd_net"


def test_load_model_inside_running_event_loop():
    """已有事件循环时，应在线程中加载且不触发嵌套循环异常。"""
    registry = ModelRegistry()
    backend = _FakeBackend()
    info = ModelInfo(
        model_id="loop_test",
        model_name="事件循环测试模型",
        backend_name="onnx",
        capability=ModelCapability.DISEASE_CLASSIFY,
        model_path="fake.onnx",
    )

    async def _run():
        factory._load_model_sync(registry, backend, info, "onnx")

    asyncio.run(_run())
    assert registry.get_model_info("loop_test") is info


def test_resolve_inference_model_reloads_and_falls_back(monkeypatch):
    """配置模型缺失时应重扫，并回退到实际已经加载的图片模型。"""
    stale = ModelRegistry()
    refreshed = ModelRegistry()
    backend = _FakeBackend()
    info = ModelInfo(
        model_id="available_model",
        model_name="可用模型",
        backend_name="torch",
        capability=ModelCapability.DISEASE_CLASSIFY,
        model_path="available.pth",
    )
    refreshed.register("torch", backend)
    refreshed._model_map[info.model_id] = "torch"
    refreshed._model_info[info.model_id] = info
    monkeypatch.setattr(factory, "_model_registry", stale)
    monkeypatch.setattr(factory, "reload_model_registry", lambda: refreshed)

    registry, model_id = factory.resolve_inference_model("missing_model")

    assert registry is refreshed
    assert model_id == "available_model"
