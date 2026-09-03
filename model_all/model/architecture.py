"""
病虫害识别模型 —— 网络结构定义
骨干网络：ConvNeXt V2-Base（timm 预训练，FCMAE 自监督）
分类头：自定义全连接层 + Dropout

ConvNeXt V2 (CVPR 2023) 特点：
- FCMAE 自监督预训练，特征泛化性强于纯监督预训练
- 全卷积架构，推理速度约为同级别 Transformer 的 2 倍
- 训练更稳定，数据量小时过拟合风险更低
"""
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _configure_hf_endpoint():
    """配置 HuggingFace Hub 端点，优先使用国内镜像以避免下载失败。

    当检测到未设置 HF_ENDPOINT 且不在离线模式时，自动切换到 hf-mirror.com。
    用户可通过 HF_ENDPOINT 环境变量自行指定其他镜像。
    """
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        return  # 离线模式，不修改

    if os.environ.get("HF_ENDPOINT"):
        return  # 用户已自行配置，尊重用户选择

    # 国内用户自动使用 hf-mirror.com 镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _safe_create_backbone(backbone_name, pretrained, **kwargs):
    """安全创建 timm 骨干网络，下载失败时自动降级为随机初始化。

    当 pretrained=True 但网络不可用时：
    1. 打印警告信息
    2. 自动回退到 pretrained=False（随机初始化）
    3. 后续可通过微调弥补精度
    """
    if not pretrained:
        return timm.create_model(backbone_name, pretrained=False, **kwargs)

    try:
        return timm.create_model(backbone_name, pretrained=True, **kwargs)
    except (RuntimeError, OSError, IOError) as exc:
        warnings.warn(
            f"预训练权重下载失败: {exc}\\n"
            f"将使用随机初始化的 {backbone_name} 继续训练。\\n"
            f"提示：设置环境变量 HF_ENDPOINT=https://hf-mirror.com 可使用国内镜像。\\n"
            f"或手动下载权重到 HuggingFace 缓存目录后重试。"
        )
        return timm.create_model(backbone_name, pretrained=False, **kwargs)


# 模块加载时自动配置镜像
_configure_hf_endpoint()


class PestDiseaseClassifier(nn.Module):
    """
    病虫害图像分类模型
    基于 ConvNeXt V2-Base 迁移学习，适用于 38 类农作物病虫害识别

    ConvNeXt V2-B 参数量: ~89M（骨干 ~88M + 分类头 ~1M）

    输入: (B, 3, H, W) RGB 图像张量
    输出: (B, num_classes) 类别 logits
    """

    def __init__(self, num_classes=38, dropout_rate=0.3, freeze_backbone=True):
        super().__init__()

        self.backbone = _safe_create_backbone(
            "convnextv2_base",
            pretrained=True,
            num_classes=0,
        )
        self.feature_dim = self.backbone.num_features  # 1024

        if freeze_backbone:
            self.freeze_backbone()

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

        self.num_classes = num_classes
        self.image_size = 256

    def forward(self, x):
        features = self.backbone(x)        # (B, 1024)
        logits = self.classifier(features)  # (B, num_classes)
        return logits

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


class LesionGuidedMultiScaleFusion(nn.Module):
    """病斑引导的多尺度动态融合模块。

    各阶段特征先投影到相同通道数，再由空间注意力增强疑似病斑区域，
    同时根据当前图像动态计算每个尺度的融合权重。
    """

    def __init__(self, input_channels, fusion_channels=256):
        super().__init__()
        self.num_scales = len(input_channels)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, fusion_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fusion_channels),
                nn.GELU(),
            )
            for channels in input_channels
        ])
        merged_channels = fusion_channels * self.num_scales
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(merged_channels, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.GELU(),
            nn.Conv2d(fusion_channels, self.num_scales, kernel_size=1),
            nn.Sigmoid(),
        )
        hidden = max(fusion_channels // 2, 32)
        self.scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(merged_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_scales),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.GELU(),
        )

    def forward(self, features):
        target_size = features[0].shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, features):
            value = projection(feature)
            if value.shape[-2:] != target_size:
                value = F.interpolate(
                    value, size=target_size, mode="bilinear", align_corners=False
                )
            projected.append(value)

        merged = torch.cat(projected, dim=1)
        attention_maps = self.spatial_attention(merged)
        scale_weights = torch.softmax(self.scale_gate(merged), dim=1)

        fused = torch.zeros_like(projected[0])
        for index, feature in enumerate(projected):
            # 残差式注意力避免训练初期错误抹除有效信息。
            enhanced = feature * (1.0 + attention_maps[:, index:index + 1])
            weight = scale_weights[:, index].view(-1, 1, 1, 1)
            fused = fused + weight * enhanced
        return self.refine(fused), attention_maps, scale_weights


class HMPDNet(nn.Module):
    """病斑引导的层次化多任务病害识别网络。

    输出四个任务：
    - 作物分类；
    - 病害属性分类；
    - 作物-病害联合分类；
    - 严重度分类（仅对具备严重度标注的样本计算损失）。
    """

    def __init__(
        self,
        num_crops,
        num_diseases,
        num_joint_classes,
        joint_to_crop,
        joint_to_disease,
        num_severities=2,
        backbone_name="convnextv2_base",
        pretrained=True,
        fusion_channels=256,
        dropout_rate=0.3,
        consistency_strength=0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = _safe_create_backbone(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        feature_channels = self.backbone.feature_info.channels()
        self.fusion = LesionGuidedMultiScaleFusion(feature_channels, fusion_channels)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(fusion_channels),
            nn.Dropout(dropout_rate),
        )
        self.crop_head = nn.Linear(fusion_channels, num_crops)
        self.disease_head = nn.Linear(fusion_channels, num_diseases)
        self.joint_head = nn.Linear(fusion_channels, num_joint_classes)
        self.severity_head = nn.Linear(fusion_channels, num_severities)
        self.consistency_strength = float(consistency_strength)

        self.register_buffer(
            "joint_to_crop", torch.as_tensor(joint_to_crop, dtype=torch.long)
        )
        self.register_buffer(
            "joint_to_disease", torch.as_tensor(joint_to_disease, dtype=torch.long)
        )
        if len(joint_to_crop) != num_joint_classes or len(joint_to_disease) != num_joint_classes:
            raise ValueError("联合类别映射长度必须等于 num_joint_classes")

    def forward(self, x):
        multi_scale_features = self.backbone(x)
        fused_map, attention_maps, scale_weights = self.fusion(multi_scale_features)
        features = self.pool(fused_map)

        crop_logits = self.crop_head(features)
        disease_logits = self.disease_head(features)
        joint_logits = self.joint_head(features)
        severity_logits = self.severity_head(features)

        crop_log_probs = F.log_softmax(crop_logits, dim=1)
        disease_log_probs = F.log_softmax(disease_logits, dim=1)
        hierarchical_logits = (
            crop_log_probs.index_select(1, self.joint_to_crop)
            + disease_log_probs.index_select(1, self.joint_to_disease)
        )
        final_logits = joint_logits + self.consistency_strength * hierarchical_logits

        return {
            "crop_logits": crop_logits,
            "disease_logits": disease_logits,
            "joint_logits": joint_logits,
            "severity_logits": severity_logits,
            "hierarchical_logits": hierarchical_logits,
            "final_logits": final_logits,
            "attention_maps": attention_maps,
            "scale_weights": scale_weights,
        }

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def unfreeze_last_stages(self, count=2):
        """仅解冻骨干最后若干阶段，适合第二阶段微调。"""
        self.freeze_backbone()
        stages = getattr(self.backbone, "stages", None)
        if stages is None:
            self.unfreeze_backbone()
            return
        for stage in list(stages)[-count:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True


# ── 安全序列化（state_dict，不使用 pickle 整模型）──

def save_model_state(model, path):
    """保存模型权重为 state_dict（安全、标准格式）"""
    torch.save(model.state_dict(), path)


def load_model_state(path, num_classes=38, dropout_rate=0.3, map_location=None):
    """从 state_dict 安全加载模型"""
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PestDiseaseClassifier(num_classes=num_classes, dropout_rate=dropout_rate)
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_hmpd_checkpoint(path, map_location=None):
    """安全加载HMPD-Net检查点，并返回模型、类别体系和训练元数据。"""
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
        mmap=True,
    )
    if checkpoint.get("architecture") != "HMPDNet":
        raise ValueError("检查点不是HMPDNet格式")
    taxonomy = checkpoint["taxonomy"]
    config = checkpoint["model_config"]
    model = HMPDNet(
        num_crops=len(taxonomy["crops"]),
        num_diseases=len(taxonomy["diseases"]),
        num_joint_classes=len(taxonomy["joint_classes"]),
        joint_to_crop=taxonomy["joint_to_crop"],
        joint_to_disease=taxonomy["joint_to_disease"],
        num_severities=len(taxonomy["severities"]),
        backbone_name=config["backbone_name"],
        pretrained=False,
        fusion_channels=config["fusion_channels"],
        dropout_rate=config["dropout_rate"],
        consistency_strength=config["consistency_strength"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, taxonomy, {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
        "model_config": config,
    }


# ── ONNX 导出 ──

def export_to_onnx(model, onnx_path, image_size=256):
    """
    导出模型为 ONNX 格式，供 onnxruntime 推理

    使用方式:
        export_to_onnx(model, "model.onnx")
        # 然后用 PestDiseaseDetectorONNX("model.onnx") 进行推理
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )
    return onnx_path
