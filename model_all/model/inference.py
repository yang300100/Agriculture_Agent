"""
病虫害识别模型 —— 推理接口（Agent 就绪）
提供简洁的预测 API，返回 JSON 兼容的字典结构，可直接接入 Agent 框架

支持两种推理后端:
  - PyTorch: PestDiseaseDetector（默认，训练后直接用）
  - ONNX:   PestDiseaseDetectorONNX（需先 export_to_onnx 导出）
"""
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

from .config import CLASS_NAMES, IMAGE_SIZE, MEAN, STD
from .architecture import load_model_state, load_hmpd_checkpoint, export_to_onnx


def _build_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def _softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class PestDiseaseDetector:
    """
    病虫害检测器 —— PyTorch 推理后端

    使用方式:
        detector = PestDiseaseDetector("weights/model.pth")
        result = detector.predict("leaf.jpg")
        # {"class": "番茄早疫病", "confidence": 0.95, "top_5": [...]}
    """

    def __init__(self, model_path, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = load_model_state(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.transform = _build_transform()

    def predict(self, image_path):
        image = self._load_image(image_path)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(input_tensor)[0].cpu().numpy()

        top5_idx = np.argsort(probs)[::-1][:5]

        return {
            "class": CLASS_NAMES[top5_idx[0]],
            "class_index": int(top5_idx[0]),
            "confidence": round(float(probs[top5_idx[0]]), 4),
            "top_5": [
                {
                    "class": CLASS_NAMES[i],
                    "class_index": int(i),
                    "confidence": round(float(probs[i]), 4),
                }
                for i in top5_idx
            ],
        }

    def predict_batch(self, image_paths):
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

    def predict_from_array(self, image_array):
        image = Image.fromarray(image_array.astype("uint8"), "RGB")
        return self.predict(image)

    def _load_image(self, image):
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise TypeError(f"不支持的类型: {type(image)}，请传入路径或 PIL Image")


class PestDiseaseDetectorONNX:
    """
    病虫害检测器 —— ONNX 推理后端（比 PyTorch 快 2-3 倍）

    使用前需先导出 ONNX 模型:
        from model.architecture import export_to_onnx, load_model_state
        model = load_model_state("weights/model.pth")
        export_to_onnx(model, "weights/model.onnx")

    使用方式:
        detector = PestDiseaseDetectorONNX("weights/model.onnx")
        result = detector.predict("leaf.jpg")
    """

    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.transform = _build_transform()

    def predict(self, image_path):
        image = self._load_image(image_path)
        input_tensor = self.transform(image).unsqueeze(0)
        ort_inputs = {"input": input_tensor.numpy()}
        logits = self.session.run(None, ort_inputs)[0][0]
        probs = _softmax(logits)

        top5_idx = np.argsort(probs)[::-1][:5]

        return {
            "class": CLASS_NAMES[top5_idx[0]],
            "class_index": int(top5_idx[0]),
            "confidence": round(float(probs[top5_idx[0]]), 4),
            "top_5": [
                {
                    "class": CLASS_NAMES[i],
                    "class_index": int(i),
                    "confidence": round(float(probs[i]), 4),
                }
                for i in top5_idx
            ],
        }

    def predict_batch(self, image_paths):
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

    def _load_image(self, image):
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise TypeError(f"不支持的类型: {type(image)}，请传入路径或 PIL Image")


class HMPDDetector:
    """优化后HMPD-Net的多任务推理接口。"""

    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model, self.taxonomy, self.metadata = load_hmpd_checkpoint(
            checkpoint_path, map_location=self.device
        )
        self.model.to(self.device)
        image_size = self.metadata["model_config"].get("image_size", IMAGE_SIZE)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def predict(self, image):
        image = self._load_image(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)

        joint_probs = torch.softmax(outputs["final_logits"][0], dim=0)
        crop_probs = torch.softmax(outputs["crop_logits"][0], dim=0)
        disease_probs = torch.softmax(outputs["disease_logits"][0], dim=0)
        severity_probs = torch.softmax(outputs["severity_logits"][0], dim=0)
        top_probs, top_indices = torch.topk(joint_probs, min(5, joint_probs.numel()))
        best_joint = int(top_indices[0])
        attention = outputs["attention_maps"].mean(dim=1)[0].cpu().numpy()

        return {
            "class": self.taxonomy["joint_classes"][best_joint],
            "confidence": round(float(top_probs[0]), 4),
            "crop": self.taxonomy["crops"][int(crop_probs.argmax())],
            "disease": self.taxonomy["diseases"][int(disease_probs.argmax())],
            "severity": self.taxonomy["severities"][int(severity_probs.argmax())],
            "top_5": [
                {
                    "class": self.taxonomy["joint_classes"][int(index)],
                    "confidence": round(float(probability), 4),
                }
                for probability, index in zip(top_probs, top_indices)
            ],
            "scale_weights": [
                round(float(value), 4)
                for value in outputs["scale_weights"][0].cpu()
            ],
            # 注意力图用于解释模型关注区域，不等同于病斑分割掩码。
            "attention_map": attention,
        }

    @staticmethod
    def _load_image(image):
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"不支持的类型: {type(image)}")


# ==================== Agent 工具函数 ====================

_DETECTOR = None


def get_detector(model_path="weights/pest_disease_model.pth"):
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = PestDiseaseDetector(model_path)
    return _DETECTOR


def agent_tool_predict(image_path: str) -> dict:
    detector = get_detector()
    return detector.predict(image_path)


def agent_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "identify_pest_disease",
            "description": "识别农作物叶片图像中的病虫害类型，支持苹果、番茄、玉米、马铃薯等作物共38类病虫害",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "农作物叶片图像的本地文件路径",
                    }
                },
                "required": ["image_path"],
            },
        },
    }
