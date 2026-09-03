"""ONNX Runtime 推理后端"""
import logging
import time
from io import BytesIO
from typing import List

import numpy as np
from PIL import Image

from models.base import BaseModelBackend, ModelInfo, ModelInput, ModelOutput, Prediction

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except (ImportError, OSError):
    ONNX_AVAILABLE = False


class ONNXBackend(BaseModelBackend):
    backend_name = "onnx"

    def __init__(self, device: str = "cpu"):
        self._sessions = {}     # model_id → InferenceSession
        self._models = {}       # model_id → ModelInfo
        self._device = device
        if device == "cuda":
            self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self._providers = ["CPUExecutionProvider"]

    async def load_model(self, model_info: ModelInfo) -> bool:
        if not ONNX_AVAILABLE:
            logger.error("onnxruntime 未安装")
            return False
        try:
            session = ort.InferenceSession(model_info.model_path, providers=self._providers)
            self._sessions[model_info.model_id] = session
            self._models[model_info.model_id] = model_info
            logger.info("ONNX模型已加载: %s", model_info.model_id)
            return True
        except Exception as e:
            logger.error("加载ONNX模型失败 %s: %s", model_info.model_id, e)
            return False

    async def unload_model(self, model_id: str) -> None:
        self._sessions.pop(model_id, None)
        self._models.pop(model_id, None)

    async def infer(self, model_id: str, model_input: ModelInput) -> ModelOutput:
        model_info = self._models.get(model_id)
        session = self._sessions.get(model_id)
        if model_info is None or session is None:
            return ModelOutput.error(model_id, "MODEL_NOT_LOADED")

        try:
            image = Image.open(BytesIO(model_input.image_bytes)).convert("RGB")
            preprocess = model_info.preprocessing
            resize = preprocess.get("resize", model_info.input_shape[1:])
            if isinstance(resize, (list, tuple)):
                image = image.resize((resize[1], resize[0]))

            img_array = np.array(image).astype(np.float32) / 255.0
            mean = np.array(preprocess.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
            std = np.array(preprocess.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
            img_array = (img_array - mean) / std
            img_array = img_array.transpose(2, 0, 1)
            img_array = np.expand_dims(img_array, axis=0)

            input_name = session.get_inputs()[0].name
            start = time.perf_counter()
            outputs = session.run(None, {input_name: img_array})
            elapsed_ms = (time.perf_counter() - start) * 1000

            logits = outputs[0][0]
            # softmax 归一化，确保 confidence 在 0-1 范围内（对齐 TorchBackend 行为）
            exp_logits = np.exp(logits - np.max(logits))  # 减去 max 防数值溢出
            probs = exp_logits / np.sum(exp_logits)
            top_k = min(model_input.top_k, len(model_info.classes))
            top_indices = np.argsort(probs)[::-1][:top_k]

            predictions = [
                Prediction(
                    class_name=model_info.classes[idx] if idx < len(model_info.classes) else f"class_{idx}",
                    confidence=float(probs[idx]),
                    index=int(idx),
                )
                for idx in top_indices
            ]

            return ModelOutput(
                success=True, model_id=model_id,
                predictions=predictions, inference_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error("ONNX推理失败 %s: %s", model_id, e)
            return ModelOutput.error(model_id, "INFERENCE_ERROR", str(e))

    async def discover_models(self) -> List[ModelInfo]:
        return list(self._models.values())

    async def health_check(self) -> bool:
        return ONNX_AVAILABLE and len(self._sessions) > 0
