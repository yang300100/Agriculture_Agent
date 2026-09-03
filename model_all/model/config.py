"""
病虫害识别模型 —— 配置文件
包含类别定义、超参数、预处理参数
"""

# ==================== 病虫害类别定义 ====================
# PlantVillage 标准 38 类别（中英对照）
CLASS_NAMES = [
    # 苹果病害 (4)
    "苹果黑星病 (Apple Scab)",
    "苹果黑腐病 (Apple Black Rot)",
    "苹果锈病 (Apple Cedar Rust)",
    "苹果健康 (Apple Healthy)",
    # 蓝莓 (1)
    "蓝莓健康 (Blueberry Healthy)",
    # 樱桃 (2)
    "樱桃白粉病 (Cherry Powdery Mildew)",
    "樱桃健康 (Cherry Healthy)",
    # 玉米 (4)
    "玉米灰斑病 (Corn Cercospora Leaf Spot)",
    "玉米锈病 (Corn Common Rust)",
    "玉米大斑病 (Corn Northern Leaf Blight)",
    "玉米健康 (Corn Healthy)",
    # 葡萄 (4)
    "葡萄黑腐病 (Grape Black Rot)",
    "葡萄黑麻疹 (Grape Esca)",
    "葡萄叶枯病 (Grape Leaf Blight)",
    "葡萄健康 (Grape Healthy)",
    # 柑橘 (1)
    "柑橘黄龙病 (Orange Huanglongbing)",
    # 桃子 (2)
    "桃细菌性斑点病 (Peach Bacterial Spot)",
    "桃健康 (Peach Healthy)",
    # 辣椒 (2)
    "辣椒细菌性斑点病 (Pepper Bacterial Spot)",
    "辣椒健康 (Pepper Healthy)",
    # 马铃薯 (3)
    "马铃薯早疫病 (Potato Early Blight)",
    "马铃薯晚疫病 (Potato Late Blight)",
    "马铃薯健康 (Potato Healthy)",
    # 覆盆子 (1)
    "覆盆子健康 (Raspberry Healthy)",
    # 大豆 (1)
    "大豆健康 (Soybean Healthy)",
    # 南瓜 (1)
    "南瓜白粉病 (Squash Powdery Mildew)",
    # 草莓 (2)
    "草莓叶枯病 (Strawberry Leaf Scorch)",
    "草莓健康 (Strawberry Healthy)",
    # 番茄 (10)
    "番茄细菌性斑点病 (Tomato Bacterial Spot)",
    "番茄早疫病 (Tomato Early Blight)",
    "番茄晚疫病 (Tomato Late Blight)",
    "番茄叶霉病 (Tomato Leaf Mold)",
    "番茄斑枯病 (Tomato Septoria Leaf Spot)",
    "番茄红蜘蛛危害 (Tomato Spider Mite)",
    "番茄靶斑病 (Tomato Target Spot)",
    "番茄黄化曲叶病毒病 (Tomato Yellow Leaf Curl Virus)",
    "番茄花叶病毒病 (Tomato Mosaic Virus)",
    "番茄健康 (Tomato Healthy)",
]

NUM_CLASSES = len(CLASS_NAMES)  # 38

# 每类对应的 (作物名称, 病害名称)，与 CLASS_NAMES 索引一一对应
CROP_DISEASE_MAP = [
    ("苹果", "黑星病"),
    ("苹果", "黑腐病"),
    ("苹果", "锈病"),
    ("苹果", "健康"),
    ("蓝莓", "健康"),
    ("樱桃", "白粉病"),
    ("樱桃", "健康"),
    ("玉米", "灰斑病"),
    ("玉米", "锈病"),
    ("玉米", "大斑病"),
    ("玉米", "健康"),
    ("葡萄", "黑腐病"),
    ("葡萄", "黑麻疹"),
    ("葡萄", "叶枯病"),
    ("葡萄", "健康"),
    ("柑橘", "黄龙病"),
    ("桃", "细菌性斑点病"),
    ("桃", "健康"),
    ("辣椒", "细菌性斑点病"),
    ("辣椒", "健康"),
    ("马铃薯", "早疫病"),
    ("马铃薯", "晚疫病"),
    ("马铃薯", "健康"),
    ("覆盆子", "健康"),
    ("大豆", "健康"),
    ("南瓜", "白粉病"),
    ("草莓", "叶枯病"),
    ("草莓", "健康"),
    ("番茄", "细菌性斑点病"),
    ("番茄", "早疫病"),
    ("番茄", "晚疫病"),
    ("番茄", "叶霉病"),
    ("番茄", "斑枯病"),
    ("番茄", "红蜘蛛危害"),
    ("番茄", "靶斑病"),
    ("番茄", "黄化曲叶病毒病"),
    ("番茄", "花叶病毒病"),
    ("番茄", "健康"),
]

# ==================== 模型超参数 ====================
BACKBONE = "convnextv2_base"  # 骨干网络: ConvNeXt V2-Base (CVPR 2023)
IMAGE_SIZE = 256              # 输入图像尺寸（ConvNeXtV2 支持任意分辨率）
DROPOUT_RATE = 0.3            # Dropout 比例（防止过拟合）
FREEZE_BACKBONE = True        # 是否冻结骨干网络（微调时建议先冻结后解冻）

# ==================== 图像预处理参数 ====================
MEAN = [0.485, 0.456, 0.406]  # ImageNet 均值
STD = [0.229, 0.224, 0.225]   # ImageNet 标准差

# ==================== 训练超参数（供 train.py 使用）====================
BATCH_SIZE = 32
EPOCHS = 30                   # 阶段一 epoch 数
PHASE2_EPOCHS = 20            # 阶段二 epoch 数（全模型微调，建议 ≥20）
LEARNING_RATE = 1e-4          # 分类头学习率
BACKBONE_LR_RATIO = 0.1       # 阶段二骨干学习率比例（lr * ratio）
WEIGHT_DECAY = 5e-2           # AdamW 权重衰减（增大到 5e-2，当前推荐范围）
WARMUP_EPOCHS = 3             # 学习率预热 epoch 数
GRAD_CLIP = 1.0               # 梯度裁剪阈值
LABEL_SMOOTHING = 0.1         # 标签平滑（0.1 对 38 类效果最佳）
MIXUP_ALPHA = 0.2             # MixUp 混合强度
CUTMIX_ALPHA = 1.0            # CutMix 混合强度
SEED = 42                     # 随机种子（确保可复现）
NUM_WORKERS = 4
TRAIN_VAL_SPLIT = 0.8
