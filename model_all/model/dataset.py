"""基于统一清单的多任务病害数据集。"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class MultiTaskDiseaseDataset(Dataset):
    """同时返回作物、病害、联合类别和可选严重度标签。"""

    def __init__(self, manifest_path, split, transform=None):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            self.rows = [
                row for row in csv.DictReader(stream)
                if row["split"] == split
            ]
        if not self.rows:
            raise ValueError(f"清单中没有 split={split!r} 的样本")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["path"])
        if not image_path.is_absolute():
            image_path = (self.manifest_path.parent / image_path).resolve()
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return {
            "image": image,
            "crop": torch.tensor(int(row["crop_index"]), dtype=torch.long),
            "disease": torch.tensor(int(row["disease_index"]), dtype=torch.long),
            "joint": torch.tensor(int(row["joint_index"]), dtype=torch.long),
            # -1 表示该数据源没有严重度标注，损失函数会自动忽略。
            "severity": torch.tensor(int(row["severity_index"]), dtype=torch.long),
            "path": str(image_path),
            "source": row["source"],
        }
