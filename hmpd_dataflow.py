# -*- coding: utf-8 -*-
"""生成 HMPD-Net 完整数据流向图。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

W, H = 100, 152
fig, ax = plt.subplots(figsize=(16.5, 24))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(cy, w, h, text, fc="#eef4fb", ec="#3366aa", fs=9.5, tc="#1a1a1a"):
    ax.add_patch(FancyBboxPatch(
        (50 - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.35,rounding_size=1.1",
        linewidth=1.3, edgecolor=ec, facecolor=fc))
    ax.text(50, cy, text, ha="center", va="center",
            fontsize=fs, linespacing=1.45, color=tc)


def label(cy, text, color="#333333", fs=12):
    ax.text(50, cy, text, ha="center", va="center",
            fontsize=fs, color=color, fontweight="bold")


def arrow(y_from, y_to, x=50, style="-|>", lw=1.3, color="#3366aa"):
    ax.add_patch(FancyArrowPatch(
        (x, y_from), (x, y_to), arrowstyle=style,
        mutation_scale=13, linewidth=lw, color=color))


def subbox(cx, cy, w, h, text, fc="#ffffff", ec="#88aacc", fs=8.5):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.25,rounding_size=0.8",
        linewidth=1.0, edgecolor=ec, facecolor=fc))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, linespacing=1.2)


# ───────────── Section A 数据准备 ─────────────
label(148, "A. 数据准备（离线，一次完成）", "#7a1f1f")
lbls = [("data1\n(作物__病害)", 24), ("data2\n(小麦病害)", 50), ("data3\n(作物/病害/严重度)", 76)]
for t, x in lbls:
    ax.add_patch(FancyBboxPatch((x - 11, 142 - 3.5), 22, 7,
                                boxstyle="round,pad=0.25,rounding_size=0.8",
                                linewidth=1.1, edgecolor="#8899bb", facecolor="#f2f5fa"))
    ax.text(x, 142, t, ha="center", va="center", fontsize=9)
for _, x in lbls:
    arrow(138.5, 134.6, x=x, lw=1.0)

clean = box(131, 82, 8, "统一清洗与对齐\n格式校验 · SHA-256 全库去重 · 同图异标冲突检查")
arrow(127, 125.2)

box(121, 86, 8.5, "统一清单 + 类别体系\nmanifest.csv · taxonomy.json\n(crops/diseases/joint/severity · joint_to_crop/disease)", fc="#eaf3e6", ec="#44884a")
arrow(116.7, 115.2)

box(111, 86, 8, "MultiTaskDiseaseDataset\n每样本 → image · crop · disease · joint · severity", fc="#eaf3e6", ec="#44884a")

# ───────────── Section B 模型前向 ─────────────
label(101, "B. 模型前向（训练与推理共用）", "#123c6b")
arrow(107, 99.2)

box(96, 60, 7, "输入图像      (B, 3, 256×256)")
arrow(92.5, 90.2)

box(86, 86, 9, "骨干  ConvNeXt V2-Base（timm · FCMAE 自监督预训练）\n→ 输出 4 阶段多尺度特征  multi_scale_features")
arrow(81.5, 76.8)

# 融合模块
ax.add_patch(FancyBboxPatch((50 - 47, 68 - 9.5), 94, 19,
                            boxstyle="round,pad=0.35,rounding_size=1.1",
                            linewidth=1.5, edgecolor="#b06a1f", facecolor="#fdf3e6"))
ax.text(50, 74.6, "病斑引导的多尺度动态融合   LesionGuidedMultiScaleFusion",
        ha="center", va="center", fontsize=9.5, color="#7a4a10", fontweight="bold")
for tx, ty in [
    ("[逐尺度]  1×1 Conv → BN → GELU  投影到 fusion_channels", 70.6),
    ("[空间注意力]  3×3 Conv → Sigmoid  →  attention_maps（高亮疑似病斑区）", 67.2),
    ("[尺度门控]  AdaptiveAvgPool → MLP → Softmax  →  scale_weights（动态选倍率）", 63.8),
    ("[残差增强]  feature×(1+attention)  加权求和  →  fused_map", 60.4),
]:
    ax.text(50, ty, tx, ha="center", va="center", fontsize=8.6)
arrow(58.5, 58.2)

box(54, 82, 8, "共享编码池  pool\nAdaptiveAvgPool2d → Flatten → LayerNorm → Dropout\n→  features (B, 256)")
arrow(50, 45.9)  # arrow 到 heads 容器顶（下方容器顶约 45.5）

# 四个任务头（容器 + 4 内框）
# 容器: top ~45.5, bottom ~38.5 → cy 42, h 7.5 ; 加一点高度容纳标题
ax.add_patch(FancyBboxPatch((50 - 47, 42 - 6.5), 94, 13,
                            boxstyle="round,pad=0.35,rounding_size=1.1",
                            linewidth=1.5, edgecolor="#123c6b", facecolor="#eef2fb"))
ax.text(50, 46.8, "四个任务头（共享 features）", ha="center", va="center",
        fontsize=9.5, color="#123c6b", fontweight="bold")
subx = [22, 41, 60, 79]
subtxt = [
    "crop_head\n→ 作物",
    "disease_head\n→ 病害",
    "joint_head\n→ 作物×病害\n★主任务",
    "severity_head\n→ 严重度\n(掩码辅助)",
]
for x, t in zip(subx, subtxt):
    subbox(x, 40, 17, 9, t)
arrow(35.5, 33.7)  # heads 容器底 → 一致性

box(29, 88, 9, "层次化一致性融合\nhierarchical_logits = log P(crop)[joint→crop] ⊕ log P(disease)[joint→disease]\nfinal_logits = joint_logits + λ·hierarchical_logits     (λ = 0.3)")
arrow(24.5, 18.7)

box(19, 90, 8, "输出\nfinal_logits（主任务）· crop/disease/severity 概率\n· attention_maps · scale_weights（可解释）", fc="#f1e8fb", ec="#6a4aa8")

# ───────────── Section C 训练目标 ─────────────
label(10, "C. 训练目标 HMPDLoss（反向传播）", "#7a1f1f")
box(3.5, 92, 8.5, "总损失 = joint★ ＋ 0.3·crop ＋ 0.3·disease ＋ 0.2·severity(掩码≥0) ＋ 0.1·KL一致性\n（KL：joint 边缘化 → crop/disease，与各头分布对齐）", fc="#fbeaea", ec="#a83a3a")

fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.02)
out = r"C:\Users\User\Desktop\Agriculture_Agent\hmpd_dataflow.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)
