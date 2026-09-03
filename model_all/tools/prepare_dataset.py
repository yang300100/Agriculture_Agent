"""统一整理并清洗三个病害识别数据集。

脚本不会修改或删除原始数据，而是生成：
1. manifest.csv：训练直接使用的干净样本清单；
2. taxonomy.json：稳定的作物、病害、联合类别与严重度索引；
3. rejected.csv：损坏、缺标注、重复或格式异常样本；
4. summary.json：各数据源与划分的统计摘要。

用法：
    python model_all/tools/prepare_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"
}
HEALTHY_WORDS = {"healthy", "health", "健康", "heathy"}
# “未知”使用索引-1并由损失掩码忽略，不作为可学习类别。
SEVERITY_NAMES = ["一般", "严重"]


@dataclass(frozen=True)
class Sample:
    path: Path
    source: str
    split: str
    crop: str
    disease: str
    severity: str = "未知"
    source_label: str = ""

    @property
    def joint(self) -> str:
        return f"{self.crop}__{self.disease}"


def _clean_token(value: str) -> str:
    value = value.strip().replace("\ufeff", "")
    value = re.sub(r"\s+", " ", value)
    return value


def _normal_key(value: str) -> str:
    value = _clean_token(value).lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value)


CROP_ALIASES = {
    "apple": "苹果",
    "blueberry": "蓝莓",
    "cherry": "樱桃",
    "corn": "玉米",
    "maize": "玉米",
    "grape": "葡萄",
    "orange": "柑橘",
    "citrus": "柑橘",
    "peach": "桃",
    "pepperbell": "辣椒",
    "pepper": "辣椒",
    "potato": "马铃薯",
    "raspberry": "覆盆子",
    "soybean": "大豆",
    "squash": "南瓜",
    "strawberry": "草莓",
    "tomato": "番茄",
    "wheat": "小麦",
}


DISEASE_ALIASES = {
    "healthy": "健康",
    "health": "健康",
    "heathy": "健康",
    "健康": "健康",
    "疮痂病": "细菌性斑点病",
    "细菌性斑点病": "细菌性斑点病",
    "轮斑病": "黑麻疹",
    "黑麻疹": "黑麻疹",
    "褐斑病": "叶枯病",
    "斑点病": "靶斑病",
    "叶斑病": "弯孢叶斑病",
    "红蜘蛛损伤": "红蜘蛛危害",
    "晚疫病菌": "晚疫病",
    "applescab": "黑星病",
    "blackrot": "黑腐病",
    "cedarapplerust": "雪松锈病",
    "frogeyespot": "灰斑病",
    "powderymildew": "白粉病",
    "cercosporaleafspotgrayleafspot": "灰斑病",
    "cercosporazeaemaydistehonanddaniels": "灰斑病",
    "commonrust": "锈病",
    "pucciniapolysora": "锈病",
    "northernleafblight": "大斑病",
    "corncurvularialeafspotfungus": "弯孢叶斑病",
    "maizedwarfmosaicvirus": "花叶病毒病",
    "esca": "黑麻疹",
    "esblackmeasles": "黑麻疹",
    "escablackmeasles": "黑麻疹",
    "grapeblackmeaslesfungus": "黑麻疹",
    "leafblightisariopsisleafspot": "叶枯病",
    "leafblight": "叶枯病",
    "grapeleafblightfungus": "叶枯病",
    "haunglongbingcitrusgreening": "黄龙病",
    "haunglongbing": "黄龙病",
    "citrusgreeningjune": "黄龙病",
    "citrushealthy": "健康",
    "bacterialspot": "细菌性斑点病",
    "peachbacterialspot": "细菌性斑点病",
    "pepperscab": "细菌性斑点病",
    "earlyblight": "早疫病",
    "potatoearlyblightfungus": "早疫病",
    "tomatoearlyblightfungus": "早疫病",
    "lateblight": "晚疫病",
    "potatolateblightfungus": "晚疫病",
    "tomatolateblightwatermold": "晚疫病",
    "leafscorch": "叶枯病",
    "strawberryscorch": "叶枯病",
    "leafmold": "叶霉病",
    "tomatoleafmoldfungus": "叶霉病",
    "septorialeafspot": "斑枯病",
    "tomatoseptorialeafspotfungus": "斑枯病",
    "spidermitestwospottedspidermite": "红蜘蛛危害",
    "tomatospidermitedamage": "红蜘蛛危害",
    "targetspot": "靶斑病",
    "tomatotargetspotbacteria": "靶斑病",
    "tomatoyellowleafcurlvirus": "黄化曲叶病毒病",
    "tomatoylcvvirus": "黄化曲叶病毒病",
    "tomatomosaicvirus": "花叶病毒病",
    "tomatotomv": "花叶病毒病",
    "tomatopowderymildew": "白粉病",
    "leafrust": "叶锈病",
    "stemrust": "茎锈病",
}


def normalize_crop(raw: str) -> str:
    key = _normal_key(raw)
    if key in CROP_ALIASES:
        return CROP_ALIASES[key]
    for alias, name in CROP_ALIASES.items():
        if key.startswith(alias):
            return name
    return _clean_token(raw)


def normalize_disease(raw: str) -> str:
    cleaned = re.sub(r"\([^)]*\)", "", raw)
    cleaned = re.sub(r"\b(general|serious)\b", "", cleaned, flags=re.IGNORECASE)
    key = _normal_key(cleaned)
    if key in DISEASE_ALIASES:
        return DISEASE_ALIASES[key]
    return _clean_token(cleaned).replace("_", " ")


def parse_data1(root: Path) -> Iterable[Sample]:
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        raw = class_dir.name
        if _normal_key(raw) == "backgroundwithoutleaves":
            crop, disease = "背景", "无叶片"
        else:
            parts = re.split(r"_{2,}", raw, maxsplit=1)
            if len(parts) != 2:
                continue
            crop, disease = normalize_crop(parts[0]), normalize_disease(parts[1])
        for path in sorted(class_dir.iterdir()):
            if path.is_file():
                yield Sample(path, "data1", "unassigned", crop, disease, "未知", raw)


def parse_data2(root: Path) -> Iterable[Sample]:
    train_root = root / "train"
    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir():
            continue
        raw = class_dir.name
        disease = normalize_disease(raw.replace("healthy_wheat", "healthy"))
        for path in sorted(class_dir.iterdir()):
            if path.is_file():
                yield Sample(path, "data2", "unassigned", "小麦", disease, "未知", raw)


def parse_label_file(path: Path) -> dict[int, tuple[str, str, str, str]]:
    labels: dict[int, tuple[str, str, str, str]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s*(\d+)\s+(.+?)\s*$", line)
        if not match:
            continue
        class_id = int(match.group(1))
        raw = match.group(2)
        chinese = re.search(r"\(([^()]*)\)\s*$", raw)
        readable = chinese.group(1) if chinese else raw
        severity = "严重" if readable.endswith("严重") else "一般" if readable.endswith("一般") else "未知"
        if severity != "未知":
            readable = readable[: -len(severity)]

        crop = ""
        for alias, canonical in sorted(CROP_ALIASES.items(), key=lambda item: -len(item[0])):
            if _normal_key(raw).startswith(alias) or readable.startswith(canonical):
                crop = canonical
                break
        if not crop:
            raise ValueError(f"无法从 data3 标签解析作物: {class_id} {raw}")

        disease_text = readable[len(crop):].strip() if readable.startswith(crop) else raw
        disease = normalize_disease(disease_text)
        labels[class_id] = (crop, disease, severity, raw)
    if not labels:
        raise ValueError(f"没有从 {path} 解析到任何标签")
    return labels


def parse_data3(root: Path) -> tuple[list[Sample], list[dict[str, str]]]:
    labels = parse_label_file(root / "label.txt")
    samples: list[Sample] = []
    rejected: list[dict[str, str]] = []
    split_dirs = {
        "train": root / "AgriculturalDisease_trainingset",
        # 官方验证集不参与调参，保留为最终独立测试集。
        "test": root / "AgriculturalDisease_validationset",
    }
    for split, split_dir in split_dirs.items():
        annotation_files = sorted(split_dir.glob("*annotations.json"))
        if len(annotation_files) != 1:
            raise ValueError(f"{split_dir} 应恰好包含一个 annotations.json")
        records = json.loads(annotation_files[0].read_text(encoding="utf-8-sig"))
        seen_names: set[str] = set()
        for record in records:
            image_id = str(record.get("image_id", "")).strip()
            class_id = record.get("disease_class")
            if not image_id or class_id not in labels:
                rejected.append(_reject(split_dir, "data3", "invalid_annotation", image_id))
                continue
            if image_id in seen_names:
                rejected.append(_reject(split_dir / "images" / image_id, "data3", "duplicate_annotation", image_id))
                continue
            seen_names.add(image_id)
            crop, disease, severity, raw = labels[int(class_id)]
            samples.append(
                Sample(split_dir / "images" / image_id, "data3", split, crop, disease, severity, raw)
            )

        for image in (split_dir / "images").iterdir():
            if image.is_file() and image.name not in seen_names:
                rejected.append(_reject(image, "data3", "unannotated_file", ""))
    return samples, rejected


def _reject(path: Path, source: str, reason: str, detail: str = "") -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "source": source,
        "reason": reason,
        "detail": detail,
    }


def validate_image(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing_file"
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False, "unsupported_extension"
    if path.stat().st_size == 0:
        return False, "empty_file"
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
            if image.width < 16 or image.height < 16:
                return False, "image_too_small"
    except (UnidentifiedImageError, OSError, ValueError):
        return False, "corrupt_image"
    return True, ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_splits(samples: list[Sample], val_ratio: float, seed: int) -> list[Sample]:
    """为没有官方划分的数据按联合类别分层划分。"""
    rng = random.Random(seed)
    grouped: dict[str, list[Sample]] = defaultdict(list)
    fixed: list[Sample] = []
    for sample in samples:
        if sample.split in {"train", "val", "test"}:
            fixed.append(sample)
        else:
            grouped[sample.joint].append(sample)

    assigned = list(fixed)
    for items in grouped.values():
        rng.shuffle(items)
        val_count = max(1, round(len(items) * val_ratio)) if len(items) > 1 else 0
        for index, sample in enumerate(items):
            split = "val" if index < val_count else "train"
            assigned.append(
                Sample(
                    sample.path, sample.source, split, sample.crop, sample.disease,
                    sample.severity, sample.source_label,
                )
            )
    return assigned


def build_taxonomy(samples: list[Sample]) -> dict:
    crops = sorted({sample.crop for sample in samples})
    diseases = sorted({sample.disease for sample in samples})
    joints = sorted({sample.joint for sample in samples})
    crop_to_index = {name: index for index, name in enumerate(crops)}
    disease_to_index = {name: index for index, name in enumerate(diseases)}
    return {
        "version": 1,
        "crops": crops,
        "diseases": diseases,
        "joint_classes": joints,
        "severities": SEVERITY_NAMES,
        "joint_to_crop": [crop_to_index[name.split("__", 1)[0]] for name in joints],
        "joint_to_disease": [disease_to_index[name.split("__", 1)[1]] for name in joints],
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> dict:
    root = args.input_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    data3_samples, rejected = parse_data3(root / "data_3")
    # 优先保留标注更丰富且具有官方划分的data3，再处理其他数据源中的重复图。
    samples = list(data3_samples)
    samples.extend(parse_data2(root / "data_2"))
    samples.extend(parse_data1(root / "data_1"))

    clean: list[Sample] = []
    hashes: dict[str, Sample] = {}
    for index, sample in enumerate(samples, 1):
        ok, reason = validate_image(sample.path)
        if not ok:
            rejected.append(_reject(sample.path, sample.source, reason, sample.source_label))
            continue
        digest = file_sha256(sample.path)
        previous = hashes.get(digest)
        if previous is not None:
            same_label = previous.joint == sample.joint and previous.severity == sample.severity
            reason = "exact_duplicate" if same_label else "duplicate_label_conflict"
            detail = f"保留文件: {previous.path.resolve()}；保留标签: {previous.joint}/{previous.severity}"
            rejected.append(_reject(sample.path, sample.source, reason, detail))
            continue
        hashes[digest] = sample
        clean.append(sample)
        if index % 5000 == 0:
            print(f"已校验 {index}/{len(samples)} 个候选样本")

    clean = assign_splits(clean, args.val_ratio, args.seed)
    taxonomy = build_taxonomy(clean)
    crop_index = {name: i for i, name in enumerate(taxonomy["crops"])}
    disease_index = {name: i for i, name in enumerate(taxonomy["diseases"])}
    joint_index = {name: i for i, name in enumerate(taxonomy["joint_classes"])}
    severity_index = {name: i for i, name in enumerate(taxonomy["severities"])}

    manifest_rows = []
    for sample in sorted(clean, key=lambda item: (item.split, item.source, item.joint, str(item.path))):
        manifest_rows.append({
            "path": str(sample.path.resolve()),
            "split": sample.split,
            "source": sample.source,
            "crop": sample.crop,
            "disease": sample.disease,
            "joint_class": sample.joint,
            "severity": sample.severity,
            "crop_index": crop_index[sample.crop],
            "disease_index": disease_index[sample.disease],
            "joint_index": joint_index[sample.joint],
            "severity_index": -1 if sample.severity == "未知" else severity_index[sample.severity],
            "source_label": sample.source_label,
        })

    write_csv(
        output / "manifest.csv",
        manifest_rows,
        [
            "path", "split", "source", "crop", "disease", "joint_class", "severity",
            "crop_index", "disease_index", "joint_index", "severity_index", "source_label",
        ],
    )
    write_csv(output / "rejected.csv", rejected, ["path", "source", "reason", "detail"])
    (output / "taxonomy.json").write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "candidate_samples": len(samples),
        "accepted_samples": len(clean),
        "rejected_samples": len(rejected),
        "num_crops": len(taxonomy["crops"]),
        "num_diseases": len(taxonomy["diseases"]),
        "num_joint_classes": len(taxonomy["joint_classes"]),
        "by_source": dict(Counter(sample.source for sample in clean)),
        "by_split": dict(Counter(sample.split for sample in clean)),
        "rejected_by_reason": dict(Counter(row["reason"] for row in rejected)),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="整理并清洗多源病虫害识别数据")
    parser.add_argument("--input-root", type=Path, default=project_root / "train_data")
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "train_data" / "prepared"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = prepare(arguments)
    print(json.dumps(result, ensure_ascii=False, indent=2))
