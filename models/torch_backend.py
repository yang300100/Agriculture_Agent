"""PyTorch 推理后端

支持两种模型加载模式:
1. 通用模式 (默认): torch.load() 加载完整序列化模型 (pickle)
2. 安全模式 (state_dict): 加载 state_dict 格式权重，需指定 model_architecture
   当前支持的 model_architecture:
   - "convnext_v2": 加载 model_all.model.architecture.PestDiseaseClassifier
   - "hmpd_net": 加载 HMPD-Net 多任务检查点
"""
import logging
import os
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
except (ImportError, OSError):
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
        """加载模型，支持通用 pickle 和安全 state_dict 两种格式"""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch 未安装")
            return False

        architecture = model_info.metadata.get("model_architecture", "")
        model_path = model_info.model_path

        try:
            if architecture == "convnext_v2":
                # ── 安全 state_dict 加载: ConvNeXt V2-Base 病虫害分类模型 ──
                model = self._load_convnext_v2(model_path, model_info)
            elif architecture == "hmpd_net":
                # ── HMPD-Net 多任务检查点加载 ──
                model = self._load_hmpd_net(model_path, model_info)
            else:
                # ── 通用 pickle 加载（兼容旧模型）──
                if not os.path.exists(model_path):
                    # 尝试 .pth → .pt 后缀 fallback
                    alt_path = model_path.rsplit(".", 1)[0] + ".pt"
                    if os.path.exists(alt_path):
                        model_path = alt_path
                model = torch.load(model_path, map_location=self._device, weights_only=False)

            model.eval()
            model.to(self._device)
            self._models[model_info.model_id] = (model, model_info)
            logger.info("PyTorch模型已加载: %s (架构: %s, 设备: %s)",
                       model_info.model_id, architecture or "pickle", self._device)
            return True
        except Exception as e:
            logger.error("加载PyTorch模型失败 %s: %s", model_info.model_id, e)
            return False

    def _load_convnext_v2(self, model_path: str, model_info: ModelInfo):
        """使用 model_all 的安全 state_dict 加载方法加载 ConvNeXt V2 模型"""
        # 确保 model_all 在 Python path 中
        model_all_dir = os.path.join(os.path.dirname(__file__), "..", "model_all")
        if model_all_dir not in __import__("sys").path:
            __import__("sys").path.insert(0, os.path.abspath(model_all_dir))

        from model.architecture import load_model_state
        from model.config import NUM_CLASSES, DROPOUT_RATE

        num_classes = model_info.metadata.get("num_classes", NUM_CLASSES)
        dropout_rate = model_info.metadata.get("dropout_rate", DROPOUT_RATE)

        model = load_model_state(
            model_path,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            map_location=self._device,
        )
        return model

    def _load_hmpd_net(self, model_path: str, model_info: ModelInfo):
        """加载 HMPD-Net，并用检查点内的类别体系补全注册信息。"""
        model_all_dir = os.path.join(os.path.dirname(__file__), "..", "model_all")
        if model_all_dir not in __import__("sys").path:
            __import__("sys").path.insert(0, os.path.abspath(model_all_dir))

        from model.architecture import load_hmpd_checkpoint

        model, taxonomy, checkpoint_metadata = load_hmpd_checkpoint(
            model_path,
            map_location=self._device,
        )
        image_size = int(
            checkpoint_metadata["model_config"].get("image_size", 256)
        )
        model_info.classes = [
            str(class_name).replace("__", "-")
            for class_name in taxonomy["joint_classes"]
        ]
        model_info.input_shape = (3, image_size, image_size)
        model_info.preprocessing = {
            **model_info.preprocessing,
            "resize": [image_size, image_size],
        }
        model_info.metadata.update({
            "taxonomy": taxonomy,
            "checkpoint_epoch": checkpoint_metadata.get("epoch"),
            "checkpoint_metrics": checkpoint_metadata.get("metrics", {}),
        })
        return model

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

            architecture = model_info.metadata.get("model_architecture", "")
            raw_output = None
            if architecture == "hmpd_net":
                logits = outputs["final_logits"][0]
                taxonomy = model_info.metadata.get("taxonomy", {})

                def _task_result(output_key, taxonomy_key):
                    task_probs = torch.softmax(outputs[output_key][0], dim=0)
                    index = int(task_probs.argmax().item())
                    labels = taxonomy.get(taxonomy_key, [])
                    label = labels[index] if index < len(labels) else f"class_{index}"
                    return {
                        "class_name": label,
                        "confidence": float(task_probs[index].item()),
                        "index": index,
                    }

                raw_output = {
                    "crop": _task_result("crop_logits", "crops"),
                    "disease": _task_result("disease_logits", "diseases"),
                    "severity": _task_result("severity_logits", "severities"),
                    "scale_weights": outputs["scale_weights"][0].detach().cpu().tolist(),
                    "checkpoint_epoch": model_info.metadata.get("checkpoint_epoch"),
                }
            else:
                logits = outputs[0]

            probs = torch.softmax(logits, dim=0)
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
                raw_output=raw_output,
            )
        except Exception as e:
            logger.error("PyTorch推理失败 %s: %s", model_id, e)
            return ModelOutput.error(model_id, "INFERENCE_ERROR", str(e))

    async def discover_models(self) -> List[ModelInfo]:
        return [info for _, info in self._models.values()]

    async def health_check(self) -> bool:
        return TORCH_AVAILABLE and len(self._models) > 0
