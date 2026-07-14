"""内置预训练模型配置预设

用户将对应的 .onnx / .pt 权重文件放入 models/weights/ 即可使用。
"""
from models.base import ModelCapability

PRESETS = {
    "plant_village_wheat": {
        "model_name": "PlantVillage 小麦病害分类",
        "capability": ModelCapability.DISEASE_CLASSIFY,
        "classes": ["健康", "条锈病", "叶锈病", "秆锈病", "白粉病", "赤霉病"],
        "input_shape": (3, 224, 224),
        "preprocessing": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize": [224, 224],
        },
        "preferred_backend": "onnx",
    },
    "plant_village_tomato": {
        "model_name": "PlantVillage 番茄病害分类",
        "capability": ModelCapability.DISEASE_CLASSIFY,
        "classes": [
            "健康", "早疫病", "晚疫病", "叶霉病", "斑枯病",
            "细菌性斑点病", "黄化曲叶病", "花叶病毒病",
        ],
        "input_shape": (3, 224, 224),
        "preprocessing": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize": [224, 224],
        },
        "preferred_backend": "onnx",
    },
    "plant_village_rice": {
        "model_name": "PlantVillage 水稻病害分类",
        "capability": ModelCapability.DISEASE_CLASSIFY,
        "classes": ["健康", "稻瘟病", "纹枯病", "白叶枯病", "胡麻斑病", "恶苗病"],
        "input_shape": (3, 224, 224),
        "preprocessing": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize": [224, 224],
        },
        "preferred_backend": "onnx",
    },
}
