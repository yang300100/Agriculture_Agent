"""
病虫害识别模型 —— 训练脚本（改进版）
改进点:
  1. Label Smoothing —— 防止 38 类过拟合，提升泛化
  2. MixUp + CutMix —— 混合增强，打破背景依赖
  3. AdamW —— 解耦权重衰减，收敛更好
  4. Warmup + Cosine —— 前期稳定，后期精细收敛
  5. Gradient Clipping —— 防止全模型微调时梯度爆炸
  6. 阶段间重载最佳模型 —— 微调起点最优
  7. TrivialAugment —— 自动搜索最优增强策略
"""
import sys
import os
import random
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from timm.data import Mixup

from model.architecture import PestDiseaseClassifier, save_model_state
from model.config import (
    NUM_CLASSES, IMAGE_SIZE, MEAN, STD,
    DROPOUT_RATE, BATCH_SIZE, EPOCHS, PHASE2_EPOCHS,
    LEARNING_RATE, BACKBONE_LR_RATIO, WEIGHT_DECAY,
    WARMUP_EPOCHS, GRAD_CLIP, LABEL_SMOOTHING,
    MIXUP_ALPHA, CUTMIX_ALPHA, SEED,
    NUM_WORKERS, TRAIN_VAL_SPLIT, CLASS_NAMES,
)

# ==================== 配置 ====================
DATA_DIR = "data"
OUTPUT_DIR = os.path.join("weights", "pest_disease_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==================== 数据增强 ====================
# 训练增强：TrivialAugmentWide 自动搜索最优增强组合，比手写更强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load_data(data_dir, batch_size=BATCH_SIZE):
    """加载数据，训练/验证集各自独立 transform，避免互相覆盖"""
    full_ds = datasets.ImageFolder(data_dir, transform=None)
    classes = full_ds.classes
    n = len(full_ds)

    rng = np.random.RandomState(SEED)
    indices = np.arange(n)
    rng.shuffle(indices)
    train_size = int(TRAIN_VAL_SPLIT * n)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    # 两个独立 ImageFolder，各自带正确的 transform
    train_base = datasets.ImageFolder(data_dir, transform=train_transform)
    val_base = datasets.ImageFolder(data_dir, transform=val_transform)

    train_ds = torch.utils.data.Subset(train_base, train_idx.tolist())
    val_ds = torch.utils.data.Subset(val_base, val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, classes


def create_optimizer(model, phase="head"):
    """分阶段创建优化器"""
    if phase == "head":
        return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    else:
        return torch.optim.AdamW([
            {"params": model.backbone.parameters(),
             "lr": LEARNING_RATE * BACKBONE_LR_RATIO},
            {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
        ], weight_decay=WEIGHT_DECAY)


def create_scheduler(optimizer, total_epochs):
    """Warmup + Cosine 学习率调度"""
    if WARMUP_EPOCHS >= total_epochs:
        return CosineAnnealingLR(optimizer, T_max=total_epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - WARMUP_EPOCHS)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])


def train_epoch(model, loader, criterion, optimizer, device, mixup_fn=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        # MixUp / CutMix
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # 精度只在非 mixup 时计算（mixup 后标签是软的，硬准确率无意义）
        if mixup_fn is None:
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        total += labels.size(0)

    acc = 100.0 * correct / total if mixup_fn is None else None
    return running_loss / total, acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def run_phase(model, train_loader, val_loader, epochs, phase_name,
              optimizer, scheduler, mixup_fn=None):
    """通用训练阶段"""
    best_acc = 0.0
    best_path = OUTPUT_DIR
    history = []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, mixup_fn)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        if isinstance(scheduler, SequentialLR):
            scheduler.step()
        else:
            scheduler.step()

        # 打印信息
        if train_acc is not None:
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        else:
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_model_state(model, best_path)
            print(f"  -> 最佳模型已保存 (Acc: {best_acc:.2f}%)")

        history.append((train_loss, val_loss, val_acc))

    return best_acc


def main():
    print(f"设备: {DEVICE}")
    print(f"类别数: {NUM_CLASSES}")
    print(f"图像尺寸: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"训练增强: TrivialAugmentWide + MixUp/CutMix")
    print(f"Label Smoothing: {LABEL_SMOOTHING}")
    print()

    # 1. 加载数据
    if not os.path.exists(DATA_DIR):
        print(f"\n错误: 数据目录 '{DATA_DIR}' 不存在！")
        print("请将数据集按以下结构放置：")
        print("  data/")
        for name in CLASS_NAMES[:5]:
            print(f"    {name}/")
        print("    ...")
        return

    print("加载数据...")
    train_loader, val_loader, classes = load_data(DATA_DIR)
    print(f"  训练集: {len(train_loader.dataset)} 张")
    print(f"  验证集: {len(val_loader.dataset)} 张")

    # 2. MixUp + CutMix（仅在阶段一使用，阶段二不需要）
    mixup_fn = Mixup(
        mixup_alpha=MIXUP_ALPHA,
        cutmix_alpha=CUTMIX_ALPHA,
        num_classes=NUM_CLASSES,
        label_smoothing=LABEL_SMOOTHING,
    )

    # 3. Label Smoothing 损失函数
    global criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # 4. 构建模型
    print("\n构建模型...")
    model = PestDiseaseClassifier(
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        freeze_backbone=True,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量:   {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # ── 阶段一：冻结骨干，训练分类头 ──
    print("\n" + "=" * 50)
    print("阶段一：训练分类头（MixUp/CutMix + 骨干冻结）")
    print("=" * 50)

    optimizer = create_optimizer(model, phase="head")
    scheduler = create_scheduler(optimizer, EPOCHS)
    best1 = run_phase(model, train_loader, val_loader, EPOCHS, "阶段一",
                      optimizer, scheduler, mixup_fn=mixup_fn)
    print(f"\n阶段一完成，最佳验证准确率: {best1:.2f}%")

    # 重载阶段一最佳模型，作为阶段二的起点
    from model.architecture import load_model_state
    model = load_model_state(OUTPUT_DIR, num_classes=NUM_CLASSES,
                              dropout_rate=DROPOUT_RATE)
    model.unfreeze_backbone()
    model = model.to(DEVICE)

    # ── 阶段二：全模型微调 ──
    print("\n" + "=" * 50)
    print(f"阶段二：全模型微调（骨干学习率 ×{BACKBONE_LR_RATIO}，无 MixUp）")
    print("=" * 50)

    optimizer = create_optimizer(model, phase="finetune")
    scheduler = create_scheduler(optimizer, PHASE2_EPOCHS)
    best2 = run_phase(model, train_loader, val_loader, PHASE2_EPOCHS, "阶段二",
                      optimizer, scheduler, mixup_fn=None)

    print(f"\n{'=' * 50}")
    print(f"训练完成！")
    print(f"  阶段一最佳: {best1:.2f}%")
    print(f"  阶段二最佳: {best2:.2f}%")
    print(f"  模型保存在: {OUTPUT_DIR}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
