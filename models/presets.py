"""内置预训练模型配置预设

用户将对应的 .onnx / .pth 权重文件放入 models/weights/ 即可使用。

plant_village_38 为基于 ConvNeXt V2-Base 的全品类病虫害分类模型（主力预设）：
- 骨干: ConvNeXt V2-Base (timm, FCMAE 自监督预训练), ~88M 参数
- 分类头: 自定义 MLP (LayerNorm→Dropout→Linear→GELU→Dropout→Linear→GELU→Linear), ~1M 参数
- 输入: (3, 256, 256) RGB
- 输出: 38 类 (14 种作物, PlantVillage 标准类别)
"""
from models.base import ModelCapability

# PlantVillage 标准 38 类别标签
# ⚠️ 顺序必须与 model_all/model/config.py 中的 CLASS_NAMES / CROP_DISEASE_MAP 索引严格一致
# 模型输出的第 i 个 logit 对应此列表的第 i 个标签
PLANT_VILLAGE_38_CLASSES = [
    # 0-3: 苹果
    "苹果-黑星病", "苹果-黑腐病", "苹果-锈病", "苹果-健康",
    # 4: 蓝莓
    "蓝莓-健康",
    # 5-6: 樱桃
    "樱桃-白粉病", "樱桃-健康",
    # 7-10: 玉米
    "玉米-灰斑病", "玉米-锈病", "玉米-大斑病", "玉米-健康",
    # 11-14: 葡萄
    "葡萄-黑腐病", "葡萄-黑麻疹", "葡萄-叶枯病", "葡萄-健康",
    # 15: 柑橘
    "柑橘-黄龙病",
    # 16-17: 桃
    "桃-细菌性斑点病", "桃-健康",
    # 18-19: 辣椒
    "辣椒-细菌性斑点病", "辣椒-健康",
    # 20-22: 马铃薯
    "马铃薯-早疫病", "马铃薯-晚疫病", "马铃薯-健康",
    # 23: 覆盆子
    "覆盆子-健康",
    # 24: 大豆
    "大豆-健康",
    # 25: 南瓜
    "南瓜-白粉病",
    # 26-27: 草莓
    "草莓-叶枯病", "草莓-健康",
    # 28-37: 番茄
    "番茄-细菌性斑点病", "番茄-早疫病", "番茄-晚疫病",
    "番茄-叶霉病", "番茄-斑枯病",
    "番茄-红蜘蛛危害", "番茄-靶斑病",
    "番茄-黄化曲叶病毒病", "番茄-花叶病毒病", "番茄-健康",
]

PRESETS = {
    # ── 默认模型：HMPD-Net 47 类联合分类，类别表从检查点动态读取 ──
    "hmpd_net": {
        "model_name": "HMPD-Net 多任务病虫害识别",
        "capability": ModelCapability.DISEASE_CLASSIFY,
        "classes": [],
        "input_shape": (3, 256, 256),
        "preprocessing": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize": [256, 256],
        },
        "preferred_backend": "torch",
        "model_architecture": "hmpd_net",
    },
    # ── 主力预设：ConvNeXt V2-Base 全品类 38 类病虫害分类 ──
    "plant_village_38": {
        "model_name": "PlantVillage 38类病虫害分类 (ConvNeXt V2-Base)",
        "capability": ModelCapability.DISEASE_CLASSIFY,
        "classes": PLANT_VILLAGE_38_CLASSES,
        "input_shape": (3, 256, 256),
        "preprocessing": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize": [256, 256],
        },
        "preferred_backend": "onnx",          # 首选 ONNX Runtime（快速推理）
        "model_architecture": "convnext_v2",   # 标识模型架构类型
        "num_classes": 38,
        "dropout_rate": 0.3,
    },
    # ── 以下为旧版单作物预设（兼容已有权重文件）──
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
