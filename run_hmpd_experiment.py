"""HMPD-Net 一键数据整理、训练、验证与独立测试入口。

正式训练：
    python run_hmpd_experiment.py

已有清洁清单时跳过数据整理：
    python run_hmpd_experiment.py --skip-prepare

快速验证完整链路：
    python run_hmpd_experiment.py --skip-prepare --smoke-test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(command: list[str], title: str) -> None:
    """继承当前终端执行子进程，确保训练进度条与指标实时显示。"""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        },
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"{title}失败，退出码：{completed.returncode}")


def check_environment() -> None:
    """在启动长任务前检查训练核心依赖。"""
    missing = []
    for module_name in ("torch", "torchvision", "timm", "PIL", "numpy", "tqdm", "matplotlib"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        joined = "、".join(missing)
        raise SystemExit(
            f"当前Python环境缺少依赖：{joined}\n"
            "请切换到项目训练环境，或执行：\n"
            "python -m pip install -r model_all/requirements.txt matplotlib"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一键运行HMPD-Net训练与独立测试")
    parser.add_argument("--skip-prepare", action="store_true", help="跳过数据清洗和清单重建")
    parser.add_argument("--smoke-test", action="store_true", help="仅运行少量批次验证完整链路")
    parser.add_argument("--no-pretrained", action="store_true", help="不加载通用预训练骨干")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone", default="convnextv2_base")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--prepared-dir", type=Path, default=Path("train_data/prepared"))
    parser.add_argument("--experiment-dir", type=Path)

    # ── 断点续训 ──
    parser.add_argument("--resume", action="store_true",
                        help="从最新 checkpoint 恢复训练")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="指定具体的训练 checkpoint 文件路径恢复训练")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    check_environment()

    # ── 断点续训：自动跳过数据整理 ──
    is_resuming = args.resume or args.checkpoint is not None

    if is_resuming:
        # 断点续训时强制跳过数据准备（数据已经准备好了）
        args.skip_prepare = True

    if not args.skip_prepare:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "model_all" / "tools" / "prepare_dataset.py"),
                "--output-dir",
                str(args.prepared_dir),
            ],
            "[1/3] 整理、校验并去重数据集",
        )
    else:
        manifest = PROJECT_ROOT / args.prepared_dir / "manifest.csv"
        taxonomy = PROJECT_ROOT / args.prepared_dir / "taxonomy.json"
        if not manifest.is_file() or not taxonomy.is_file():
            raise SystemExit(
                "--skip-prepare要求已有manifest.csv和taxonomy.json，"
                "请先不带该参数运行一次。"
            )
        if is_resuming:
            print("\n[1/3] 断点续训模式，使用现有数据清单。", flush=True)
        else:
            print("\n[1/3] 已跳过数据整理，使用现有清洁清单。", flush=True)

    # ── 确定实验目录 ──
    experiment_dir = args.experiment_dir
    if experiment_dir is None:
        if is_resuming:
            # 自动查找最新的实验目录
            experiments_root = PROJECT_ROOT / "experiments"
            if experiments_root.is_dir():
                hmpd_dirs = sorted(
                    experiments_root.glob("hmpd_*"),
                    key=lambda p: p.stat().st_mtime,
                )
                if hmpd_dirs:
                    experiment_dir = hmpd_dirs[-1].relative_to(PROJECT_ROOT)
                    print(f"  自动定位到最新实验目录: {experiment_dir}", flush=True)
            if experiment_dir is None:
                raise SystemExit(
                    "未找到可恢复的实验目录，请用 --experiment-dir 指定，"
                    "或不带 --resume 从头训练。"
                )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = Path("experiments") / f"hmpd_{timestamp}"
    weight_path = experiment_dir / "hmpd_best.pth"

    epochs = args.epochs
    batch_size = args.batch_size
    workers = args.workers
    backbone = args.backbone
    train_args = [
        sys.executable,
        str(PROJECT_ROOT / "model_all" / "train_hmpd.py"),
        "--prepared-dir",
        str(args.prepared_dir),
        "--experiment-dir",
        str(experiment_dir),
        "--output",
        str(weight_path),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--workers",
        str(workers),
        "--backbone",
        backbone,
        "--image-size",
        str(args.image_size),
        "--learning-rate",
        str(args.learning_rate),
    ]

    if args.no_pretrained:
        train_args.append("--no-pretrained")

    # ── 断点续训参数传递 ──
    if args.resume:
        train_args.append("--resume")

    if args.checkpoint is not None:
        train_args.extend(["--checkpoint", str(args.checkpoint)])

    if args.smoke_test:
        # 冒烟测试不下载预训练权重，仅验证数据、网络、指标和文件输出。
        train_args = [
            sys.executable,
            str(PROJECT_ROOT / "model_all" / "train_hmpd.py"),
            "--prepared-dir",
            str(args.prepared_dir),
            "--experiment-dir",
            str(experiment_dir),
            "--output",
            str(weight_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--backbone",
            "convnextv2_tiny",
            "--image-size",
            str(args.image_size),
            "--no-pretrained",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "2",
        ]
        print("已启用冒烟测试：1轮、Tiny骨干、每阶段最多2个批次。", flush=True)

    step_label = "[2/3] 续训" if is_resuming else "[2/3] 训练、验证并在独立测试集评估"
    run_command(train_args, step_label)

    absolute_experiment = (PROJECT_ROOT / experiment_dir).resolve()
    print(
        f"\n{'=' * 72}\n"
        "[3/3] 全部完成\n"
        f"最佳权重：{absolute_experiment / 'hmpd_best.pth'}\n"
        f"训练历史：{absolute_experiment / 'history.csv'}\n"
        f"训练曲线：{absolute_experiment / 'training_curves.png'}\n"
        f"测试指标：{absolute_experiment / 'test_metrics.json'}\n"
        f"混淆矩阵：{absolute_experiment / 'test_confusion_matrix.csv'}\n"
        f"{'=' * 72}",
        flush=True,
    )


if __name__ == "__main__":
    main()
