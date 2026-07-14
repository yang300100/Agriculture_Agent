"""PyTorch 推理后端"""
import logging
import time
from io import BytesIO
from typing import List

from PIL import Image

from models.base import BaseModelBackend, ModelInfo, ModelInput, ModelOutput, Prediction

logger = logging.getLogger(__name__)

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TorchBackend(BaseModelBackend):
    backend_name = "torch"

    def __init__(self, device: str = "cpu"):
        self._models = {}  # model_id → (model_instance, ModelInfo)
        if TORCH_AVAILABLE:
            self._device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        else:
            self._device = None

    async def load_model(self, model_info: ModelInfo) -> bool:
        if not TORCH_AVAILABLE:
            logger.error("PyTorch 未安装")
            return False
        try:
            model = torch.load(model_info.model_path, map_location=self._device, weights_only=False)
            model.eval()
            self._models[model_info.model_id] = (model, model_info)
            logger.info("PyTorch模型已加载: %s", model_info.model_id)
            return True
        except Exception as e:
            logger.error("加载PyTorch模型失败 %s: %s", model_info.model_id, e)
            return False

    async def unload_model(self, model_id: str) -> None:
        self._models.pop(model_id, None)

    async def infer(self, model_id: str, model_input: ModelInput) -> ModelOutput:
        entry = self._models.get(model_id)
        if entry is None:
            return ModelOutput.error(model_id, "MODEL_NOT_LOADED")
        model, model_info = entry

        try:
            image = Image.open(BytesIO(model_input.image_bytes)).convert("RGB")
            preprocess = model_info.preprocessing
            resize = preprocess.get("resize", model_info.input_shape[1:])
            if isinstance(resize, (list, tuple)):
                resize = (resize[1], resize[0])

            mean = preprocess.get("mean", [0.485, 0.456, 0.406])
            std = preprocess.get("std", [0.229, 0.224, 0.225])

            transform = T.Compose([
                T.Resize(resize),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
            tensor = transform(image).unsqueeze(0).to(self._device)

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(tensor)
            elapsed_ms = (time.perf_counter() - start) * 1000

            probs = torch.softmax(outputs[0], dim=0)
            top_k = min(model_input.top_k, len(model_info.classes))
            top_probs, top_indices = torch.topk(probs, top_k)

            predictions = [
                Prediction(
                    class_name=model_info.classes[idx] if idx < len(model_info.classes) else f"class_{idx}",
                    confidence=float(conf),
                    index=int(idx),
                )
                for conf, idx in zip(top_probs, top_indices)
            ]

            return ModelOutput(
                success=True, model_id=model_id,
                predictions=predictions, inference_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error("PyTorch推理失败 %s: %s", model_id, e)
            return ModelOutput.error(model_id, "INFERENCE_ERROR", str(e))

    async def discover_models(self) -> List[ModelInfo]:
        return [info for _, info in self._models.values()]

    async def health_check(self) -> bool:
        return TORCH_AVAILABLE and len(self._models) > 0
