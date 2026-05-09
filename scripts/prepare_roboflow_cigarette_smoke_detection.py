from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = Path("datasets/raw/roboflow_cigarette_smoke_detection_v4_20260504/Cigarette Smoke Detection.v4-version-4.yolov8.zip")
DEFAULT_OUTPUT = Path("datasets/interim/roboflow_cigarette_smoke_detection_v4_yolo3cls")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Roboflow Cigarette Smoke Detection v4 for the 3-class project dataset.")
    parser.add_argument("--zip", default=str(DEFAULT_ZIP), help="Roboflow YOLOv8 ZIP path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT), help="Normalized output dataset root.")
    parser.add_argument("--work-dir", default="D:/rf_csd_v4_prepare", help="Short temporary extraction path to avoid Windows path limits.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def convert_label(src_label: Path) -> tuple[list[str], Counter[str]]:
    converted: list[str] = []
    stats: Counter[str] = Counter()
    text = src_label.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        stats["empty_source_label"] += 1
        return converted, stats
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            # This Roboflow export contains polygon labels for cold breath/sunlight.
            # They are not compatible with our box-detection task, so keep the image
            # as a negative sample and drop the polygon line.
            stats["skipped_non_bbox_line"] += 1
            continue
        try:
            cls_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            stats["invalid_bbox_line"] += 1
            continue
        if cls_id in (0, 1):
            converted.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            stats["mapped_to_cigarette"] += 1
        elif cls_id in (2, 3):
            stats[f"skipped_class_{cls_id}"] += 1
        else:
            stats[f"unknown_class_{cls_id}"] += 1
    if not converted:
        stats["empty_output_label"] += 1
    return converted, stats


def prepare_split(source_root: Path, output_root: Path, source_split: str, target_split: str) -> dict[str, Any]:
    image_dir = source_root / source_split / "images"
    label_dir = source_root / source_split / "labels"
    out_image_dir = output_root / "images" / target_split
    out_label_dir = output_root / "labels" / target_split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    split_stats: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    for idx, src_image in enumerate(images, start=1):
        sample_id = f"rfcsd4_{target_split}_{idx:05d}"
        dst_image = out_image_dir / f"{sample_id}.jpg"
        dst_label = out_label_dir / f"{sample_id}.txt"
        shutil.copy2(src_image, dst_image)

        src_label = label_dir / f"{src_image.stem}.txt"
        if not src_label.exists():
            converted: list[str] = []
            line_stats: Counter[str] = Counter({"missing_source_label": 1, "empty_output_label": 1})
        else:
            converted, line_stats = convert_label(src_label)
        dst_label.write_text("\n".join(converted), encoding="utf-8")
        split_stats.update(line_stats)
        split_stats["images"] += 1
        split_stats["labels"] += 1
        split_stats["boxes"] += len(converted)
        if converted:
            class_counts["0"] += len(converted)
        else:
            split_stats["negative_images"] += 1

    return {"source_split": source_split, "target_split": target_split, "stats": dict(split_stats), "class_counts": dict(class_counts)}


def main() -> None:
    args = parse_args()
    zip_path = repo_path(args.zip)
    output_root = repo_path(args.output_root)
    work_dir = Path(args.work_dir)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {output_root}")

    reset_dir(work_dir)
    reset_dir(output_root)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work_dir)

    source_config = read_yaml(work_dir / "data.yaml")
    split_reports = []
    total_stats: Counter[str] = Counter()
    total_class_counts: Counter[str] = Counter()
    for source_split, target_split in SPLIT_MAP.items():
        split_report = prepare_split(work_dir, output_root, source_split, target_split)
        split_reports.append(split_report)
        total_stats.update(split_report["stats"])
        total_class_counts.update(split_report["class_counts"])

    data_yaml = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "cigarette", 1: "smoking_person", 2: "smoke"},
    }
    (output_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = {
        "source_zip": str(zip_path),
        "source_names": source_config.get("names"),
        "output_root": str(output_root),
        "mapping": {
            "source 0 Cigar in hand": "target 0 cigarette",
            "source 1 Cigar near mouth": "target 0 cigarette",
            "source 2 cold breath vapor": "dropped as polygon/negative image",
            "source 3 sunlight": "dropped as polygon/negative image",
        },
        "total_stats": dict(total_stats),
        "total_class_counts": dict(total_class_counts),
        "splits": split_reports,
        "warning": "This dataset contributes cigarette boxes and negative images only; it does not add smoking_person or smoke boxes.",
    }
    report_path = output_root / "prepare_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Prepared dataset saved to: {output_root}")


if __name__ == "__main__":
    main()
