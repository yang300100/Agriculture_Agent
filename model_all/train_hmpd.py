"""训练病斑引导的层次化多任务病害识别网络。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

# ── 优先配置 HuggingFace 镜像，避免国内下载失败 ──
if not os.environ.get("HF_ENDPOINT") and os.environ.get("HF_HUB_OFFLINE", "0") != "1":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model.architecture import HMPDNet
from model.dataset import MultiTaskDiseaseDataset
from model.losses import HMPDLoss


def seed_everything(seed):
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


def build_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def classification_metrics(predictions, targets, num_classes):
    predictions = torch.cat(predictions).cpu()
    targets = torch.cat(targets).cpu()
    confusion = torch.bincount(
        targets * num_classes + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(0).clamp_min(1)
    recall = true_positive / confusion.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    active = confusion.sum(1) > 0
    return {
        "accuracy": float(true_positive.sum() / confusion.sum().clamp_min(1)),
        "macro_precision": float(precision[active].mean()) if active.any() else 0.0,
        "macro_recall": float(recall[active].mean()) if active.any() else 0.0,
        "macro_f1": float(f1[active].mean()) if active.any() else 0.0,
        "confusion_matrix": confusion,
    }


def run_epoch(
    model, loader, criterion, device, optimizer=None, scaler=None, max_batches=None
):
    training = optimizer is not None
    model.train(training)
    loss_names = ["total", "joint", "crop", "disease", "severity", "consistency"]
    totals = {name: 0.0 for name in loss_names}
    totals["count"] = 0
    totals["joint_correct"] = 0
    predictions = {"joint": [], "crop": [], "disease": [], "severity": []}
    targets_all = {"joint": [], "crop": [], "disease": [], "severity": []}

    context = torch.enable_grad if training else torch.no_grad
    with context():
        progress = tqdm(loader, leave=False, desc="训练" if training else "评估")
        for batch_index, batch in enumerate(progress):
            if max_batches is not None and batch_index >= max_batches:
                break
            targets = move_batch(batch, device)
            images = targets.pop("image")
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs = model(images)
                losses = criterion(outputs, targets)

            if training:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            batch_size = images.shape[0]
            for name in loss_names:
                totals[name] += float(losses[name].detach()) * batch_size
            totals["count"] += batch_size
            predictions["joint"].append(outputs["final_logits"].argmax(1).detach())
            predictions["crop"].append(outputs["crop_logits"].argmax(1).detach())
            predictions["disease"].append(outputs["disease_logits"].argmax(1).detach())
            targets_all["joint"].append(targets["joint"].detach())
            targets_all["crop"].append(targets["crop"].detach())
            targets_all["disease"].append(targets["disease"].detach())
            totals["joint_correct"] += int(
                (predictions["joint"][-1] == targets["joint"]).sum()
            )
            severity_mask = targets["severity"] >= 0
            if severity_mask.any():
                predictions["severity"].append(
                    outputs["severity_logits"][severity_mask].argmax(1).detach()
                )
                targets_all["severity"].append(targets["severity"][severity_mask].detach())
            progress.set_postfix(
                loss=f"{totals['total'] / totals['count']:.4f}",
                joint_acc=f"{totals['joint_correct'] / totals['count']:.3f}",
            )

    count = max(totals["count"], 1)
    result = {f"{name}_loss": totals[name] / count for name in loss_names}
    task_sizes = {
        "joint": model.joint_head.out_features,
        "crop": model.crop_head.out_features,
        "disease": model.disease_head.out_features,
        "severity": model.severity_head.out_features,
    }
    for task, size in task_sizes.items():
        if predictions[task]:
            metrics = classification_metrics(predictions[task], targets_all[task], size)
            result.update({
                f"{task}_accuracy": metrics["accuracy"],
                f"{task}_macro_precision": metrics["macro_precision"],
                f"{task}_macro_recall": metrics["macro_recall"],
                f"{task}_macro_f1": metrics["macro_f1"],
            })
            if task == "joint":
                result["confusion_matrix"] = metrics["confusion_matrix"]
    return result


def serializable_metrics(metrics):
    return {
        key: value.tolist() if isinstance(value, torch.Tensor) else value
        for key, value in metrics.items()
    }


def write_history(path, history):
    fieldnames = sorted({key for row in history for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_history(output_dir, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, [row["train_total_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_total_loss"] for row in history], label="Validation")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs, [row["train_joint_macro_f1"] for row in history], label="Train Macro-F1"
    )
    axes[1].plot(
        epochs, [row["val_joint_macro_f1"] for row in history], label="Validation Macro-F1"
    )
    axes[1].plot(
        epochs, [row["val_joint_accuracy"] for row in history], label="Validation Accuracy"
    )
    axes[1].set_title("Joint Classification Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def save_confusion_matrix(output_dir, matrix, class_names):
    with (output_dir / "test_confusion_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["真实\\预测", *class_names])
        for name, row in zip(class_names, matrix.tolist()):
            writer.writerow([name, *row])


def save_checkpoint(path, model, taxonomy, args, epoch, metrics):
    payload = {
        "format_version": 1,
        "architecture": "HMPDNet",
        "model_state": model.state_dict(),
        "taxonomy": taxonomy,
        "model_config": {
            "backbone_name": args.backbone,
            "fusion_channels": args.fusion_channels,
            "dropout_rate": args.dropout,
            "consistency_strength": args.consistency_strength,
            "image_size": args.image_size,
        },
        "epoch": epoch,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


# ══════════════════════════════════════════════════════════════
# 训练断点续训：保存与恢复
# ══════════════════════════════════════════════════════════════

def save_training_checkpoint(checkpoint_path, model, taxonomy, args, epoch, metrics,
                             optimizer, scaler, best_score, history):
    """保存完整训练状态，支持随时恢复训练。"""
    payload = {
        "format_version": 2,  # v2 包含 optimizer/scaler 状态
        "architecture": "HMPDNet",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "taxonomy": taxonomy,
        "model_config": {
            "backbone_name": args.backbone,
            "fusion_channels": args.fusion_channels,
            "dropout_rate": args.dropout,
            "consistency_strength": args.consistency_strength,
            "image_size": args.image_size,
        },
        "training_config": {
            "epochs": args.epochs,
            "freeze_epochs": args.freeze_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "epoch": epoch,
        "best_score": best_score,
        "metrics": metrics,
        "history": history,
        "rng_state": get_rng_state(),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_training_checkpoint(checkpoint_path, device="cpu"):
    """加载完整训练 checkpoint，返回所有恢复所需的状态。"""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到训练 checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def find_latest_checkpoint(experiment_dir):
    """在实验目录下查找最新的训练 checkpoint。

    优先级：
        1. latest_checkpoint.pth
        2. checkpoint_epoch_*.pth 中修改时间最新的
    """
    experiment_dir = Path(experiment_dir)
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


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared_dir = args.prepared_dir.resolve()

    # ── 确定实验目录 ──
    experiment_dir = args.experiment_dir.resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    args.output = args.output.resolve()

    # ── 加载类别体系（断点续训时从 checkpoint 获取）──
    taxonomy = None
    start_epoch = 0
    best_score = -1.0
    history = []
    checkpoint = None

    if args.resume or args.checkpoint:
        # ── 断点续训模式 ──
        if args.checkpoint:
            resume_path = args.checkpoint.resolve()
        else:
            resume_path = find_latest_checkpoint(experiment_dir)

        if resume_path is None or not resume_path.is_file():
            if args.resume:
                print(f"\n⚠ 未找到可恢复的 checkpoint ({experiment_dir / 'checkpoints'})")
                print(f"  将从头开始训练。")
            elif args.checkpoint:
                raise FileNotFoundError(f"指定的 checkpoint 不存在: {args.checkpoint}")
        else:
            print(f"\n🔄 断点续训模式")
            checkpoint = load_training_checkpoint(resume_path, device)
            taxonomy = checkpoint["taxonomy"]
            start_epoch = checkpoint["epoch"]  # 已完成 epoch 数（从1开始）
            best_score = checkpoint.get("best_score", -1.0)
            history = checkpoint.get("history", [])

            print(f"  📂 加载 checkpoint: {resume_path}")
            print(f"     已完成 epoch: {start_epoch}/{args.epochs}")
            print(f"     历史最佳 macro_f1: {best_score:.4f}")

            # 恢复随机状态
            if checkpoint.get("rng_state"):
                set_rng_state(checkpoint["rng_state"])

    # ── 加载类别体系 ──
    if taxonomy is None:
        taxonomy = json.loads((prepared_dir / "taxonomy.json").read_text(encoding="utf-8"))

    # ── 构建数据集（每次都需要，不能 pickle）──
    train_transform, val_transform = build_transforms(args.image_size)
    train_dataset = MultiTaskDiseaseDataset(
        prepared_dir / "manifest.csv", "train", train_transform
    )
    val_dataset = MultiTaskDiseaseDataset(
        prepared_dir / "manifest.csv", "val", val_transform
    )
    test_dataset = MultiTaskDiseaseDataset(
        prepared_dir / "manifest.csv", "test", val_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    # ── 构建/恢复模型 ──
    if checkpoint is not None:
        # 从 checkpoint 恢复模型
        saved_config = checkpoint["model_config"]
        model = HMPDNet(
            num_crops=len(taxonomy["crops"]),
            num_diseases=len(taxonomy["diseases"]),
            num_joint_classes=len(taxonomy["joint_classes"]),
            joint_to_crop=taxonomy["joint_to_crop"],
            joint_to_disease=taxonomy["joint_to_disease"],
            num_severities=len(taxonomy["severities"]),
            backbone_name=saved_config["backbone_name"],
            pretrained=False,  # 权重从 checkpoint 加载
            fusion_channels=saved_config["fusion_channels"],
            dropout_rate=saved_config["dropout_rate"],
            consistency_strength=saved_config["consistency_strength"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        # 根据当前 epoch 决定骨干是否冻结
        if start_epoch >= args.freeze_epochs:
            model.unfreeze_backbone()
        else:
            model.freeze_backbone()
        print(f"  ✓ 模型权重已从 checkpoint 恢复")
    else:
        model = HMPDNet(
            num_crops=len(taxonomy["crops"]),
            num_diseases=len(taxonomy["diseases"]),
            num_joint_classes=len(taxonomy["joint_classes"]),
            joint_to_crop=taxonomy["joint_to_crop"],
            joint_to_disease=taxonomy["joint_to_disease"],
            num_severities=len(taxonomy["severities"]),
            backbone_name=args.backbone,
            pretrained=not args.no_pretrained,
            fusion_channels=args.fusion_channels,
            dropout_rate=args.dropout,
            consistency_strength=args.consistency_strength,
        ).to(device)
        model.freeze_backbone()

    criterion = HMPDLoss(
        taxonomy["joint_to_crop"],
        taxonomy["joint_to_disease"],
        crop_weight=args.crop_loss_weight,
        disease_weight=args.disease_loss_weight,
        severity_weight=args.severity_loss_weight,
        consistency_weight=args.consistency_loss_weight,
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ── 构建/恢复优化器 ──
    optimizer = torch.optim.AdamW([
        {
            "params": list(model.backbone.parameters()),
            "lr": 0.0 if start_epoch < args.freeze_epochs else args.learning_rate * 0.1,
            "name": "backbone",
        },
        {
            "params": [
                p for name, p in model.named_parameters()
                if not name.startswith("backbone.")
            ],
            "lr": args.learning_rate,
            "name": "heads",
        },
    ], weight_decay=args.weight_decay)

    if checkpoint is not None:
        # 恢复优化器和 AMP scaler 状态
        if checkpoint.get("optimizer_state"):
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                print(f"  ✓ 优化器状态已恢复")
            except Exception as e:
                print(f"  ⚠ 优化器状态恢复失败（将从头累积动量）: {e}")
        if checkpoint.get("scaler_state"):
            try:
                scaler.load_state_dict(checkpoint["scaler_state"])
                print(f"  ✓ AMP scaler 状态已恢复")
            except Exception as e:
                print(f"  ⚠ AMP scaler 状态恢复失败: {e}")

    # ── 模型信息 ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔧 模型: HMPDNet ({args.backbone})")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练: {trainable_params:,}")
    print(f"   输出目录: {experiment_dir}")
    print(f"   从 epoch {start_epoch + 1} 开始训练（共 {args.epochs} 轮）")

    # ── 训练循环 ──
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        # 冻结期结束，解冻骨干
        if epoch == args.freeze_epochs:
            model.unfreeze_backbone()
            optimizer.param_groups[0]["lr"] = args.learning_rate * 0.1
            print(f"\n  🔓 解冻骨干，骨干学习率设为 {args.learning_rate * 0.1:.2e}")

        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, max_batches=args.max_eval_batches
        )
        train_public = {k: v for k, v in train_metrics.items() if k != "confusion_matrix"}
        val_public = {k: v for k, v in val_metrics.items() if k != "confusion_matrix"}
        row = {"epoch": epoch + 1}
        row.update({f"train_{key}": value for key, value in train_public.items()})
        row.update({f"val_{key}": value for key, value in val_public.items()})
        history.append(row)
        write_history(experiment_dir / "history.csv", history)
        (experiment_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        plot_history(experiment_dir, history)
        print(
            f"\nEpoch {epoch + 1:03d}/{args.epochs}\n"
            f"  Train: loss={train_metrics['total_loss']:.4f} "
            f"acc={train_metrics['joint_accuracy']:.4f} "
            f"macro_f1={train_metrics['joint_macro_f1']:.4f}\n"
            f"  Val:   loss={val_metrics['total_loss']:.4f} "
            f"acc={val_metrics['joint_accuracy']:.4f} "
            f"precision={val_metrics['joint_macro_precision']:.4f} "
            f"recall={val_metrics['joint_macro_recall']:.4f} "
            f"macro_f1={val_metrics['joint_macro_f1']:.4f}"
        )
        if val_metrics["joint_macro_f1"] > best_score:
            best_score = val_metrics["joint_macro_f1"]
            save_checkpoint(
                args.output, model, taxonomy, args, epoch + 1,
                {"train": train_public, "val": val_public},
            )
            print(f"  ⭐ 最佳模型已保存 (macro_f1={best_score:.4f})")

        # ── 每个 epoch 保存训练断点（支持随时恢复）──
        epoch_ckpt_path = checkpoints_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth"
        save_training_checkpoint(
            epoch_ckpt_path, model, taxonomy, args, epoch + 1,
            {"train": train_public, "val": val_public},
            optimizer, scaler, best_score, history,
        )
        # 更新 latest 指针
        latest_path = checkpoints_dir / "latest_checkpoint.pth"
        save_training_checkpoint(
            latest_path, model, taxonomy, args, epoch + 1,
            {"train": train_public, "val": val_public},
            optimizer, scaler, best_score, history,
        )

    # ── 测试集评估（使用最佳模型）──
    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(
        model, test_loader, criterion, device, max_batches=args.max_eval_batches
    )
    confusion = test_metrics.pop("confusion_matrix")
    save_confusion_matrix(experiment_dir, confusion, taxonomy["joint_classes"])
    test_payload = serializable_metrics(test_metrics)
    test_payload["best_epoch"] = checkpoint.get("epoch")
    test_payload["best_validation_metrics"] = checkpoint.get("metrics", {}).get("val", {})
    (experiment_dir / "test_metrics.json").write_text(
        json.dumps(test_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "\n独立测试集结果："
        f"accuracy={test_metrics['joint_accuracy']:.4f} "
        f"precision={test_metrics['joint_macro_precision']:.4f} "
        f"recall={test_metrics['joint_macro_recall']:.4f} "
        f"macro_f1={test_metrics['joint_macro_f1']:.4f}"
    )
    print(f"最佳权重：{args.output}")
    print(f"实验记录：{experiment_dir}")
    return {"history": history, "test": test_payload}


def build_parser():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="训练 HMPD-Net")
    parser.add_argument(
        "--prepared-dir", type=Path, default=project_root / "train_data" / "prepared"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "weights" / "hmpd_best.pth"
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=project_root / "experiments" / "hmpd_latest",
    )
    parser.add_argument("--backbone", default="convnextv2_base")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--fusion-channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--consistency-strength", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--crop-loss-weight", type=float, default=0.3)
    parser.add_argument("--disease-loss-weight", type=float, default=0.3)
    parser.add_argument("--severity-loss-weight", type=float, default=0.2)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)

    # ── 断点续训 ──
    parser.add_argument("--resume", action="store_true",
                        help="从最新 checkpoint 恢复训练")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="指定具体的训练 checkpoint 文件路径恢复训练")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
