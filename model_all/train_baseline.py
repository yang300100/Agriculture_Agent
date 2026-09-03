"""
病虫害识别 Baseline 训练脚本 —— 仅训练 ConvNeXt V2-Base 基线模型
═══════════════════════════════════════════════════════════════
用途：训练未经修改的 PestDiseaseClassifier，用于对比后续模型改进效果。
支持断点续训——训练中断后可以从最新 checkpoint 无缝恢复。

数据集：与正式训练（train_hmpd.py）共用 train_data/prepared/ 目录下的
       manifest.csv + taxonomy.json，保证 train/val/test 划分一致。

训练策略（两阶段）：
  阶段一（冻结骨干）：MixUp + CutMix 增强，仅训练分类头
  阶段二（全模型微调）：解冻骨干，无 MixUp，全参数精细调优

断点续训机制：
  - 每个 epoch 结束后保存完整训练状态到 checkpoint
  - 包含：模型权重、优化器状态、调度器状态、随机种子、训练历史
  - 使用 --resume 参数自动查找最新 checkpoint 并恢复

使用示例：
  # 从头开始训练
  python model_all/train_baseline.py

  # 从断点恢复训练
  python model_all/train_baseline.py --resume

  # 指定实验目录
  python model_all/train_baseline.py --experiment-dir experiments/baseline_test

  # 自定义超参数
  python model_all/train_baseline.py --epochs 40 --phase2-epochs 25 --batch-size 16 --lr 5e-5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from timm.data import Mixup

# ── 确保能找到 model 包 ──
sys.path.insert(0, os.path.dirname(__file__))

from model.architecture import PestDiseaseClassifier, save_model_state
from model.dataset import MultiTaskDiseaseDataset
from model.config import (
    NUM_CLASSES, IMAGE_SIZE, MEAN, STD,
    DROPOUT_RATE, LEARNING_RATE, BACKBONE_LR_RATIO, WEIGHT_DECAY,
    WARMUP_EPOCHS, GRAD_CLIP, LABEL_SMOOTHING,
    MIXUP_ALPHA, CUTMIX_ALPHA, SEED,
    NUM_WORKERS, CLASS_NAMES,
)

# ══════════════════════════════════════════════════════════════
# 默认配置（可通过命令行覆盖）
# ══════════════════════════════════════════════════════════════
DEFAULT_PREPARED_DIR = "../train_data/prepared"  # 与正式训练共用
DEFAULT_EPOCHS = 30          # 阶段一轮数
DEFAULT_PHASE2_EPOCHS = 20   # 阶段二轮数
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4


def seed_everything(seed: int) -> None:
    """固定所有随机种子，确保训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict:
    """捕获当前所有随机数生成器的状态，用于 checkpoint 保存。"""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict) -> None:
    """从 checkpoint 恢复随机数生成器状态。"""
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ══════════════════════════════════════════════════════════════
# 数据增强与加载（与正式训练共用 MultiTaskDiseaseDataset）
# ══════════════════════════════════════════════════════════════

def build_transforms(image_size: int):
    """构建训练/验证数据增强。"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    return train_transform, val_transform


def load_data(prepared_dir: str, batch_size: int, num_workers: int):
    """使用 MultiTaskDiseaseDataset 加载数据，与正式训练完全一致。

    Baseline 仅使用 joint（联合类别）标签做单任务分类，
    与 HMPD-Net 使用同一份 manifest.csv 和 taxonomy.json，
    保证 train/val/test 划分完全相同。

    Returns:
        train_loader, val_loader, test_loader, taxonomy
    """
    prepared_path = Path(prepared_dir).resolve()
    manifest_path = prepared_path / "manifest.csv"
    taxonomy_path = prepared_path / "taxonomy.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到数据清单: {manifest_path}")
    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"找不到类别体系: {taxonomy_path}")

    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))

    train_transform, val_transform = build_transforms(IMAGE_SIZE)

    train_dataset = MultiTaskDiseaseDataset(
        manifest_path, "train", train_transform,
    )
    val_dataset = MultiTaskDiseaseDataset(
        manifest_path, "val", val_transform,
    )
    test_dataset = MultiTaskDiseaseDataset(
        manifest_path, "test", val_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, taxonomy


# ══════════════════════════════════════════════════════════════
# 优化器与调度器
# ══════════════════════════════════════════════════════════════

def create_optimizer(model: nn.Module, phase: str, lr: float, backbone_lr_ratio: float,
                     weight_decay: float) -> torch.optim.Optimizer:
    """分阶段创建 AdamW 优化器。

    Args:
        phase: "head" 仅训练分类头；"finetune" 全模型微调
    """
    if phase == "head":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
    else:
        return torch.optim.AdamW([
            {"params": model.backbone.parameters(),
             "lr": lr * backbone_lr_ratio},
            {"params": model.classifier.parameters(),
             "lr": lr},
        ], weight_decay=weight_decay)


def create_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int,
                     warmup_epochs: int) -> torch.optim.lr_scheduler.LRScheduler:
    """Warmup + CosineAnnealing 学习率调度策略。"""
    if warmup_epochs >= total_epochs:
        return CosineAnnealingLR(optimizer, T_max=total_epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


# ══════════════════════════════════════════════════════════════
# 训练 / 验证循环
# ══════════════════════════════════════════════════════════════

def _extract_batch(batch: dict, device: torch.device) -> tuple:
    """从 MultiTaskDiseaseDataset 的 dict batch 中提取 (images, joint_labels)。"""
    images = batch["image"].to(device)
    labels = batch["joint"].to(device)
    return images, labels


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device,
                mixup_fn=None, grad_clip: float = 1.0,
                scaler=None, grad_accum_steps: int = 1) -> tuple:
    """执行单个训练 epoch（支持 AMP 混合精度 + 梯度累积）。

    Args:
        scaler: torch.amp.GradScaler，启用 AMP 混合精度训练（显存减半）
        grad_accum_steps: 梯度累积步数，小 batch 时可保持等效大 batch 效果

    Returns:
        (avg_loss, accuracy_or_None) — mixup 模式下 accuracy 返回 None
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="训练中", leave=False)):
        images, labels = _extract_batch(batch, device)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # AMP 前向传播
        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 梯度累积时按累积步数缩放 loss
            loss = loss / grad_accum_steps

        # AMP 反向传播
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 每 grad_accum_steps 步或最后一步时更新参数
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        # 累计 loss（用于日志，还原为原始尺度）
        running_loss += loss.item() * grad_accum_steps * images.size(0)

        if mixup_fn is None:
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        total += labels.size(0)

    acc = 100.0 * correct / total if mixup_fn is None else None
    return running_loss / total, acc


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple:
    """执行验证。

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="验证中", leave=False):
        images, labels = _extract_batch(batch, device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate_test(model: nn.Module, loader: DataLoader, device: torch.device,
                  class_names: list) -> dict:
    """在独立测试集上评估模型，返回详细指标。

    Returns:
        dict with keys: accuracy, per_class_acc, confusion_matrix, predictions, targets
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc="测试集评估", leave=False):
        images, labels = _extract_batch(batch, device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.append(predicted.cpu())
        all_targets.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    num_classes = len(class_names)

    # 混淆矩阵
    confusion = torch.bincount(
        all_targets * num_classes + all_preds,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    # 各类别准确率
    per_class_acc = confusion.diag().float() / confusion.sum(1).clamp_min(1)
    overall_acc = float(confusion.diag().sum() / confusion.sum().clamp_min(1))

    return {
        "accuracy": overall_acc,
        "per_class_accuracy": {
            class_names[i]: round(float(per_class_acc[i]), 4)
            for i in range(num_classes) if confusion.sum(1)[i] > 0
        },
        "confusion_matrix": confusion,
        "predictions": all_preds,
        "targets": all_targets,
    }


# ══════════════════════════════════════════════════════════════
# Checkpoint 保存与恢复
# ══════════════════════════════════════════════════════════════

def save_checkpoint(checkpoint_path: Path, **payload) -> None:
    """保存完整训练状态到 checkpoint 文件。

    保存内容：
        - model_state: 模型权重
        - optimizer_state: 优化器状态
        - scheduler_state: 学习率调度器状态（如有）
        - phase: 当前训练阶段 ("head" / "finetune")
        - epoch: 当前阶段内的 epoch 编号（从 0 开始）
        - best_acc: 当前最佳验证准确率
        - history: 完整训练历史记录
        - rng_state: 随机数生成器状态
        - config: 训练超参数快照
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    print(f"  📦 Checkpoint 已保存 → {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    """从 checkpoint 文件恢复训练状态。"""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint 文件: {checkpoint_path}")

    print(f"  📂 加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"     └ 阶段={checkpoint.get('phase')}, "
          f"阶段内epoch={checkpoint.get('epoch', 0) + 1}, "
          f"最佳acc={checkpoint.get('best_acc', 0):.2f}%")
    return checkpoint


def find_latest_checkpoint(experiment_dir: Path) -> Path | None:
    """在实验目录下查找最新的 checkpoint。

    优先级：
        1. latest_checkpoint.pth
        2. checkpoint_epoch_*.pth 中修改时间最新的
    """
    latest = experiment_dir / "checkpoints" / "latest_checkpoint.pth"
    if latest.is_file():
        return latest

    checkpoints_dir = experiment_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None

    checkpoint_files = sorted(
        checkpoints_dir.glob("checkpoint_epoch_*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    return checkpoint_files[-1] if checkpoint_files else None


# ══════════════════════════════════════════════════════════════
# 训练主逻辑
# ══════════════════════════════════════════════════════════════

def run_training_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: str,
    total_epochs: int,
    lr: float,
    backbone_lr_ratio: float,
    weight_decay: float,
    warmup_epochs: int,
    grad_clip: float,
    mixup_fn,
    experiment_dir: Path,
    best_acc: float = 0.0,
    history: list | None = None,
    start_epoch: int = 0,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    scaler_state: dict | None = None,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
) -> tuple:
    """运行一个训练阶段（阶段一或阶段二）。

    Args:
        start_epoch: 从该 epoch 开始（用于断点续训），0 表示从头开始
        best_acc: 当前最佳准确率（用于断点续训时保持历史最佳）
        optimizer_state: 断点续训时，从 checkpoint 恢复的优化器状态
        scheduler_state: 断点续训时，从 checkpoint 恢复的调度器状态
        scaler_state: 断点续训时，从 checkpoint 恢复的 AMP scaler 状态
        grad_accum_steps: 梯度累积步数（>1 时用小 batch 模拟大 batch 效果）
        use_amp: 是否启用 AMP 混合精度（默认 True，CUDA 时生效）

    Returns:
        (final_best_acc, history) — 阶段结束后的最佳准确率和更新后的历史
    """
    if history is None:
        history = []

    # 创建 optimizer 和 scheduler
    optimizer = create_optimizer(model, phase, lr, backbone_lr_ratio, weight_decay)
    scheduler = create_scheduler(optimizer, total_epochs, warmup_epochs)

    # 创建 AMP scaler（CUDA 时可用，可手动关闭）
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    # ── 断点续训：恢复 optimizer/scheduler/scaler 状态 ──
    if start_epoch > 0 and optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print(f"  ↻ 已恢复优化器状态")
        except Exception as e:
            print(f"  ⚠ 优化器状态恢复失败（将从头累积动量）: {e}")

    if start_epoch > 0 and scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
            print(f"  ↻ 已恢复学习率调度器状态")
        except Exception as e:
            print(f"  ⚠ 调度器状态恢复失败（将从当前 epoch 重新开始）: {e}")

    if start_epoch > 0 and scaler_state is not None and scaler is not None:
        try:
            scaler.load_state_dict(scaler_state)
            print(f"  ↻ 已恢复 AMP scaler 状态")
        except Exception as e:
            print(f"  ⚠ AMP scaler 状态恢复失败: {e}")

    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    phase_label = "阶段一（冻结骨干 + MixUp）" if phase == "head" else "阶段二（全模型微调）"
    amp_info = "AMP(fp16)" if use_amp else "fp32"
    accum_info = f" + 梯度累积×{grad_accum_steps}" if grad_accum_steps > 1 else ""
    print(f"\n{'=' * 55}")
    print(f"  {phase_label}")
    print(f"  精度: {amp_info} | 等效 Batch: {train_loader.batch_size * grad_accum_steps}{accum_info}")
    print(f"  共 {total_epochs} epochs，从第 {start_epoch + 1} 轮开始")
    print(f"{'=' * 55}")

    for epoch in range(start_epoch, total_epochs):
        epoch_num = epoch + 1  # 显示用（从1开始）

        # ── 训练 & 验证 ──
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_fn=mixup_fn, grad_clip=grad_clip,
            scaler=scaler, grad_accum_steps=grad_accum_steps,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # ── 更新学习率 ──
        scheduler.step()
        current_lr = scheduler.get_last_lr()
        lr_str = ", ".join(f"{lr:.2e}" for lr in current_lr)

        # ── 打印进度 ──
        if train_acc is not None:
            print(f"  Epoch {epoch_num:3d}/{total_epochs} | "
                  f"LR: {lr_str} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%",
                  end="")
        else:
            print(f"  Epoch {epoch_num:3d}/{total_epochs} | "
                  f"LR: {lr_str} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%",
                  end="")

        # ── 保存最佳模型 ──
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_path = experiment_dir / "best_model.pth"
            save_model_state(model, str(best_path))
            print(f"  ⭐ 最佳！(Acc: {best_acc:.2f}%)", end="")

        print()  # 换行

        # ── 记录历史 ──
        history.append({
            "phase": phase,
            "epoch": epoch_num,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4) if train_acc is not None else None,
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 4),
            "learning_rate": lr_str,
            "best_acc": round(best_acc, 4),
            "is_best": is_best,
        })

        # ── 保存 checkpoint（每个 epoch 一次，支持断点续训）──
        checkpoint_payload = {
            "format_version": 2,
            "architecture": "PestDiseaseClassifier",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "phase": phase,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "best_acc": best_acc,
            "history": history,
            "rng_state": get_rng_state(),
            "grad_accum_steps": grad_accum_steps,
            "config": {
                "num_classes": model.num_classes,
                "dropout_rate": DROPOUT_RATE,
                "image_size": IMAGE_SIZE,
                "learning_rate": lr,
                "backbone_lr_ratio": backbone_lr_ratio,
                "weight_decay": weight_decay,
                "warmup_epochs": warmup_epochs,
                "grad_clip": grad_clip,
                "label_smoothing": LABEL_SMOOTHING,
                "batch_size": train_loader.batch_size,
                "grad_accum_steps": grad_accum_steps,
            },
        }

        # 保存带编号的 checkpoint
        epoch_ckpt_path = checkpoints_dir / f"checkpoint_epoch_{epoch_num:03d}.pth"
        save_checkpoint(epoch_ckpt_path, **checkpoint_payload)

        # 更新 latest 指针
        latest_path = checkpoints_dir / "latest_checkpoint.pth"
        save_checkpoint(latest_path, **checkpoint_payload)

        # ── 写入 CSV 历史 ──
        _write_history_csv(experiment_dir / "history.csv", history)

    return best_acc, history


def _write_history_csv(csv_path: Path, history: list) -> None:
    """将训练历史写入 CSV 文件。"""
    if not history:
        return
    fieldnames = list(history[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _save_confusion_matrix_csv(output_path: Path, matrix, class_names: list) -> None:
    """保存混淆矩阵为 CSV 文件。"""
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["真实\\预测", *class_names])
        for name, row in zip(class_names, matrix.tolist()):
            writer.writerow([name, *row])


def _plot_history(experiment_dir: Path, history: list) -> None:
    """绘制训练曲线图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib 未安装，跳过绘图")
        return

    if not history:
        return

    # 分离阶段
    phase1 = [h for h in history if h["phase"] == "head"]
    phase2 = [h for h in history if h["phase"] == "finetune"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Loss 曲线 ──
    ax = axes[0]
    if phase1:
        ax.plot([h["epoch"] for h in phase1],
                [h["train_loss"] for h in phase1], "b-", alpha=0.6, label="Train Loss (P1)")
        ax.plot([h["epoch"] for h in phase1],
                [h["val_loss"] for h in phase1], "b--", alpha=0.8, label="Val Loss (P1)")
    if phase2:
        offset = len(phase1)
        ax.plot([h["epoch"] + offset for h in phase2],
                [h["train_loss"] for h in phase2], "r-", alpha=0.6, label="Train Loss (P2)")
        ax.plot([h["epoch"] + offset for h in phase2],
                [h["val_loss"] for h in phase2], "r--", alpha=0.8, label="Val Loss (P2)")
    ax.set_title("Loss 曲线")
    ax.set_xlabel("Epoch（全局）")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ── Accuracy 曲线 ──
    ax = axes[1]
    if phase1:
        p1_acc = [h["val_acc"] for h in phase1]
        ax.plot([h["epoch"] for h in phase1], p1_acc, "b-", alpha=0.8, label="Val Acc (P1)")
    if phase2:
        offset = len(phase1)
        p2_acc = [h["val_acc"] for h in phase2]
        ax.plot([h["epoch"] + offset for h in phase2], p2_acc, "r-", alpha=0.8, label="Val Acc (P2)")
    ax.set_title("验证准确率")
    ax.set_xlabel("Epoch（全局）")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(experiment_dir / "training_curves.png", dpi=180)
    plt.close(fig)
    print(f"  📈 训练曲线已保存 → {experiment_dir / 'training_curves.png'}")


# ══════════════════════════════════════════════════════════════
# 命令行参数
# ══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="训练 Baseline ConvNeXt V2-Base 病虫害分类模型（支持断点续训）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_baseline.py                           # 从头训练（默认参数）
  python train_baseline.py --resume                  # 从最新 checkpoint 恢复
  python train_baseline.py --resume --experiment-dir experiments/baseline_20260801_120000
  python train_baseline.py --epochs 40 --phase2-epochs 30 --batch-size 16
        """,
    )

    # ── 路径参数 ──
    parser.add_argument("--prepared-dir", type=str, default=DEFAULT_PREPARED_DIR,
                        help="prepared 数据集目录（默认: ../train_data/prepared）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="模型和日志输出目录（默认: experiments/baseline_<时间戳>）")
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="指定已有实验目录（用于 --resume 或覆盖输出）")

    # ── 训练控制 ──
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"阶段一（冻结骨干）epoch 数（默认: {DEFAULT_EPOCHS}）")
    parser.add_argument("--phase2-epochs", type=int, default=DEFAULT_PHASE2_EPOCHS,
                        help=f"阶段二（全模型微调）epoch 数（默认: {DEFAULT_PHASE2_EPOCHS}）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"阶段一批次大小（默认: {DEFAULT_BATCH_SIZE}）")
    parser.add_argument("--phase2-batch-size", type=int, default=None,
                        help="阶段二批次大小（默认: batch_size // 4，全模型训练显存占用大，建议减小）")
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                        help="梯度累积步数（默认: 阶段一=1，阶段二自动计算使等效 batch 等于 batch_size）")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help=f"数据加载线程数（默认: {NUM_WORKERS}）")

    # ── 优化器参数 ──
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"分类头学习率（默认: {DEFAULT_LR}）")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help=f"AdamW 权重衰减（默认: {WEIGHT_DECAY}）")
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS,
                        help=f"学习率预热轮数（默认: {WARMUP_EPOCHS}）")
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP,
                        help=f"梯度裁剪阈值（默认: {GRAD_CLIP}）")

    # ── 数据增强 ──
    parser.add_argument("--no-mixup-phase1", action="store_true",
                        help="阶段一禁用 MixUp/CutMix 增强")
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING,
                        help=f"标签平滑系数（默认: {LABEL_SMOOTHING}）")

    # ── 显存优化 ──
    parser.add_argument("--no-amp", action="store_true",
                        help="禁用 AMP 混合精度（默认 CUDA 时自动启用 fp16，显存减半）")

    # ── 断点续训 ──
    parser.add_argument("--resume", action="store_true",
                        help="从最新 checkpoint 恢复训练")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="指定具体的 checkpoint 文件路径恢复训练")
    parser.add_argument("--resume", action="store_true",
                        help="从最新 checkpoint 恢复训练")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="指定具体的 checkpoint 文件路径恢复训练")

    # ── 其他 ──
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"随机种子（默认: {SEED}）")
    parser.add_argument("--no-phase2", action="store_true",
                        help="仅执行阶段一，跳过阶段二全模型微调")

    return parser


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def main():
    args = build_parser().parse_args()

    # ── 设备检测 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  设备: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── 确定输出目录 ──
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir).resolve()
    elif args.output_dir:
        experiment_dir = Path(args.output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = (Path(__file__).resolve().parents[1]
                          / "experiments" / f"baseline_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # ── 固定随机种子 ──
    seed_everything(args.seed)

    # ── 解析 prepared 目录 ──
    prepared_dir = args.prepared_dir
    if not os.path.isabs(prepared_dir):
        prepared_dir = Path(__file__).resolve().parent / prepared_dir
    prepared_dir = Path(prepared_dir).resolve()

    # ── 加载数据（与正式训练共用 MultiTaskDiseaseDataset）──
    print(f"\n📂 数据目录: {prepared_dir}")
    print("加载数据...")
    try:
        train_loader, val_loader, test_loader, taxonomy = load_data(
            str(prepared_dir), args.batch_size, args.workers,
        )
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("请先运行数据准备脚本：")
        print("  python run_hmpd_experiment.py    # 自动整理数据")
        print("或：")
        print("  python model_all/tools/prepare_dataset.py --output-dir train_data/prepared")
        sys.exit(1)

    joint_classes = taxonomy["joint_classes"]
    num_classes = len(joint_classes)
    print(f"  训练集: {len(train_loader.dataset)} 张")
    print(f"  验证集: {len(val_loader.dataset)} 张")
    print(f"  测试集: {len(test_loader.dataset)} 张")
    print(f"  类别数: {num_classes} (joint 联合类别)")

    # ── 损失函数 ──
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── 断点续训：尝试加载 checkpoint ──
    resume_checkpoint = None
    if args.checkpoint:
        resume_checkpoint = Path(args.checkpoint).resolve()
        if not resume_checkpoint.is_file():
            print(f"\n❌ 指定的 checkpoint 不存在: {resume_checkpoint}")
            sys.exit(1)
    elif args.resume:
        resume_checkpoint = find_latest_checkpoint(experiment_dir)
        if resume_checkpoint is None:
            print(f"\n⚠ 未找到可恢复的 checkpoint ({experiment_dir / 'checkpoints'})")
            print(f"  将从头开始训练。")
        else:
            print(f"\n🔍 找到 checkpoint: {resume_checkpoint}")

    start_phase = "head"
    start_epoch = 0           # 阶段内 epoch（从0开始）
    best_acc = 0.0
    history = []
    optimizer_state = None    # 用于恢复 optimizer
    scheduler_state = None    # 用于恢复 scheduler
    scaler_state = None       # 用于恢复 AMP scaler
    model = None

    if resume_checkpoint is not None and resume_checkpoint.is_file():
        print(f"\n🔄 断点续训模式")
        ckpt = load_checkpoint(resume_checkpoint, device)

        # 恢复随机状态
        if ckpt.get("rng_state"):
            set_rng_state(ckpt["rng_state"])

        # 重建模型并加载权重
        saved_config = ckpt.get("config", {})
        model_num_classes = saved_config.get("num_classes", num_classes)
        model = PestDiseaseClassifier(
            num_classes=model_num_classes,
            dropout_rate=saved_config.get("dropout_rate", DROPOUT_RATE),
            freeze_backbone=(ckpt.get("phase") == "head"),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.train()  # 切回训练模式

        # 恢复训练状态
        saved_phase = ckpt.get("phase", "head")
        saved_epoch = ckpt.get("epoch", -1)    # 阶段内 epoch（从0开始）
        best_acc = ckpt.get("best_acc", 0.0)
        history = ckpt.get("history", [])

        print(f"  恢复状态: 阶段={saved_phase}, "
              f"已完成epoch={saved_epoch + 1}, 最佳acc={best_acc:.2f}%")

        # 判断从哪个阶段继续
        total_phase1 = args.epochs
        total_phase2 = args.phase2_epochs

        if saved_phase == "head":
            if saved_epoch + 1 >= total_phase1:
                print(f"  阶段一已完成，切换到阶段二")
                model.unfreeze_backbone()
                model = model.to(device)
                start_phase = "finetune"
                start_epoch = 0
            else:
                start_phase = "head"
                start_epoch = saved_epoch + 1
                optimizer_state = ckpt.get("optimizer_state")
                scheduler_state = ckpt.get("scheduler_state")
                scaler_state = ckpt.get("scaler_state")
        elif saved_phase == "finetune":
            if saved_epoch + 1 >= total_phase2:
                print(f"  ⚠ 阶段二也已完成！如需重新训练请删除 checkpoint。")
            model.unfreeze_backbone()
            model = model.to(device)
            start_phase = "finetune"
            start_epoch = saved_epoch + 1
            optimizer_state = ckpt.get("optimizer_state")
            scheduler_state = ckpt.get("scheduler_state")
            scaler_state = ckpt.get("scaler_state")

        if start_phase == "head":
            model.freeze_backbone()
    else:
        print(f"\n🆕 从头开始训练")

    # ── 构建模型（如果还没创建）──
    if model is None:
        model = PestDiseaseClassifier(
            num_classes=num_classes,
            dropout_rate=DROPOUT_RATE,
            freeze_backbone=True,
        ).to(device)

    # ── 模型信息 ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔧 模型: PestDiseaseClassifier (ConvNeXt V2-Base)")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练: {trainable_params:,}")
    print(f"   类别数: {num_classes}")
    print(f"   图像尺寸: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"   输出目录: {experiment_dir}")

    # ── 阶段二显存优化：计算小 batch 和梯度累积 ──
    use_amp = device.type == "cuda" and not args.no_amp
    phase2_batch_size = args.phase2_batch_size
    if phase2_batch_size is None:
        # 默认阶段二 batch = 阶段一的 1/4（全模型训练显存暴增）
        phase2_batch_size = max(2, args.batch_size // 4)

    phase2_grad_accum = args.grad_accum_steps
    if phase2_grad_accum is None:
        # 自动计算梯度累积使等效 batch 与阶段一相同
        phase2_grad_accum = max(1, args.batch_size // phase2_batch_size)

    phase1_grad_accum = args.grad_accum_steps if args.grad_accum_steps else 1

    if phase2_batch_size < args.batch_size:
        print(f"   阶段二 Batch: {phase2_batch_size} (×{phase2_grad_accum}梯度累积 "
              f"→ 等效 {phase2_batch_size * phase2_grad_accum})")
    print(f"   AMP 混合精度: {'✅ fp16' if use_amp else '❌ fp32'}")

    # ── 保存运行配置 ──
    run_config = {
        "architecture": "PestDiseaseClassifier",
        "backbone": "convnextv2_base",
        "num_classes": num_classes,
        "image_size": IMAGE_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "batch_size": args.batch_size,
        "phase2_batch_size": phase2_batch_size,
        "phase2_grad_accum": phase2_grad_accum,
        "use_amp": use_amp,
        "phase1_epochs": args.epochs,
        "phase2_epochs": args.phase2_epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip": args.grad_clip,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": MIXUP_ALPHA,
        "cutmix_alpha": CUTMIX_ALPHA,
        "seed": args.seed,
        "prepared_dir": str(prepared_dir),
        "resumed_from": str(resume_checkpoint) if resume_checkpoint else None,
    }
    (experiment_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # ══════════════════════════════════════════════════════════
    # 阶段一：冻结骨干，训练分类头（MixUp/CutMix）
    # ══════════════════════════════════════════════════════════
    if start_phase == "head":
        if start_epoch >= args.epochs:
            print(f"\n✅ 阶段一已完成 (epoch {start_epoch}/{args.epochs})，跳过")
            best_path = experiment_dir / "best_model.pth"
            if best_path.is_file():
                from model.architecture import load_model_state
                model = load_model_state(str(best_path), num_classes=num_classes,
                                         dropout_rate=DROPOUT_RATE)
                model.unfreeze_backbone()
                model = model.to(device)
            start_phase = "finetune"
            start_epoch = 0
            optimizer_state = None
            scheduler_state = None
            scaler_state = None
        else:
            mixup_fn = None if args.no_mixup_phase1 else Mixup(
                mixup_alpha=MIXUP_ALPHA,
                cutmix_alpha=CUTMIX_ALPHA,
                num_classes=num_classes,
                label_smoothing=args.label_smoothing,
            )

            best_acc, history = run_training_phase(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                phase="head",
                total_epochs=args.epochs,
                lr=args.lr,
                backbone_lr_ratio=BACKBONE_LR_RATIO,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                grad_clip=args.grad_clip,
                mixup_fn=mixup_fn,
                experiment_dir=experiment_dir,
                best_acc=best_acc,
                history=history,
                start_epoch=start_epoch,
                optimizer_state=optimizer_state,
                scheduler_state=scheduler_state,
                scaler_state=scaler_state,
                grad_accum_steps=phase1_grad_accum,
                use_amp=use_amp,
            )

            # 阶段一完成，加载最佳模型进入阶段二
            best_path = experiment_dir / "best_model.pth"
            if best_path.is_file():
                from model.architecture import load_model_state
                model = load_model_state(str(best_path), num_classes=num_classes,
                                         dropout_rate=DROPOUT_RATE)
                model.unfreeze_backbone()
                model = model.to(device)
            start_phase = "finetune"
            start_epoch = 0
            optimizer_state = None
            scheduler_state = None
            scaler_state = None

    # ══════════════════════════════════════════════════════════
    # 阶段二：全模型微调（无 MixUp，AMP + 小 batch + 梯度累积）
    # ══════════════════════════════════════════════════════════
    if start_phase == "finetune" and not args.no_phase2:
        if start_epoch >= args.phase2_epochs:
            print(f"\n✅ 阶段二已完成 (epoch {start_epoch}/{args.phase2_epochs})，跳过")
        else:
            model.unfreeze_backbone()

            # ── 阶段二重建 train_loader（更小的 batch 防止 OOM）──
            if phase2_batch_size != args.batch_size:
                from model.dataset import MultiTaskDiseaseDataset as Mtd
                from torchvision import transforms as T
                train_transform_p2 = T.Compose([
                    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.TrivialAugmentWide(),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ])
                prepared_path = Path(prepared_dir).resolve()
                manifest_path = prepared_path / "manifest.csv"
                train_dataset_p2 = Mtd(manifest_path, "train", train_transform_p2)
                train_loader = DataLoader(
                    train_dataset_p2, batch_size=phase2_batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True,
                )
                print(f"  🔄 阶段二重建 DataLoader: batch={phase2_batch_size}, "
                      f"梯度累积×{phase2_grad_accum}")

            best_acc, history = run_training_phase(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                phase="finetune",
                total_epochs=args.phase2_epochs,
                lr=args.lr,
                backbone_lr_ratio=BACKBONE_LR_RATIO,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                grad_clip=args.grad_clip,
                mixup_fn=None,
                experiment_dir=experiment_dir,
                best_acc=best_acc,
                history=history,
                start_epoch=start_epoch,
                optimizer_state=optimizer_state,
                scheduler_state=scheduler_state,
                scaler_state=scaler_state,
                grad_accum_steps=phase2_grad_accum,
                use_amp=use_amp,
            )

    # ══════════════════════════════════════════════════════════
    # 独立测试集评估
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 55}")
    print(f"  独立测试集评估")
    print(f"{'=' * 55}")

    # 加载最佳模型进行评估
    best_path = experiment_dir / "best_model.pth"
    if best_path.is_file():
        from model.architecture import load_model_state
        model = load_model_state(str(best_path), num_classes=num_classes,
                                 dropout_rate=DROPOUT_RATE)
        model = model.to(device)
        model.eval()

        test_results = evaluate_test(model, test_loader, device, joint_classes)
        confusion = test_results.pop("confusion_matrix")

        # 保存混淆矩阵
        _save_confusion_matrix_csv(
            experiment_dir / "test_confusion_matrix.csv", confusion, joint_classes,
        )

        # 保存测试指标
        test_payload = {
            "accuracy": test_results["accuracy"],
            "per_class_accuracy": test_results["per_class_accuracy"],
        }
        (experiment_dir / "test_metrics.json").write_text(
            json.dumps(test_payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        print(f"  测试集准确率: {test_results['accuracy']:.4f} "
              f"({test_results['accuracy'] * 100:.2f}%)")
        print(f"  混淆矩阵: {experiment_dir / 'test_confusion_matrix.csv'}")
        print(f"  测试指标: {experiment_dir / 'test_metrics.json'}")
    else:
        print(f"  ⚠ 未找到最佳模型，跳过测试集评估")

    # ══════════════════════════════════════════════════════════
    # 训练完成
    # ══════════════════════════════════════════════════════════
    _plot_history(experiment_dir, history)

    (experiment_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"\n{'=' * 55}")
    print(f"  🎉 训练完成！")
    print(f"  最佳验证准确率: {best_acc:.2f}%")
    print(f"  最佳模型: {experiment_dir / 'best_model.pth'}")
    print(f"  训练历史: {experiment_dir / 'history.csv'}")
    print(f"  训练曲线: {experiment_dir / 'training_curves.png'}")
    print(f"  Checkpoints: {experiment_dir / 'checkpoints' / ''}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
