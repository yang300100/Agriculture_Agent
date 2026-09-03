"""
病虫害识别 —— 单函数调用
传入 numpy 数组图像，返回类别名称字符串
"""
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model.architecture import load_model_state
from model.config import CLASS_NAMES, CROP_DISEASE_MAP, IMAGE_SIZE, MEAN, STD

_MODEL = None
_TRANSFORM = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        import os
        model_path = os.path.join(os.path.dirname(__file__), "weights", "pest_disease_model.pth")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = load_model_state(model_path, map_location=device)
        _MODEL.to(device)
        _MODEL.eval()
    return _MODEL


def _get_transform():
    global _TRANSFORM
    if _TRANSFORM is None:
        _TRANSFORM = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    return _TRANSFORM


def predict(image_array):
    """
    病虫害识别

    Args:
        image_array: numpy 数组，形状 (H, W, 3)，RGB 格式，dtype=uint8

    Returns:
        tuple[str, str]: (作物名称, 病害名称)
            作物名称: 如 "番茄"、"玉米"、"马铃薯"
            病害名称: 如 "早疫病"、"锈病"、"健康"
    """
    image = Image.fromarray(image_array.astype("uint8"), "RGB")
    tensor = _get_transform()(image).unsqueeze(0)

    model = _get_model()
    device = next(model.parameters()).device

    with torch.no_grad():
        probs = model.predict_proba(tensor.to(device))[0].cpu().numpy()

    best_idx = int(np.argmax(probs))
    crop, disease = CROP_DISEASE_MAP[best_idx]
    return crop, disease
