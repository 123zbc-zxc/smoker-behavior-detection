from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz"}
LABEL_EXTS = {".txt"}
STAGES = ("raw", "interim", "final")
SPLITS = ("train", "val", "valid", "test")
DEFAULT_PROJECT_NAMES = {"0": "cigarette", "1": "smoking_person", "2": "smoke"}


@dataclass
class YoloStats:
    label_files: int = 0
    empty_label_files: int = 0
    valid_box_lines: int = 0
    invalid_lines: int = 0
    unknown_classes: int = 0
    out_of_range_boxes: int = 0
    class_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.class_counts is None:
            self.class_counts = Counter()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_files": self.label_files,
            "empty_label_files": self.empty_label_files,
            "valid_box_lines": self.valid_box_lines,
            "invalid_lines": self.invalid_lines,
            "unknown_classes": self.unknown_classes,
            "out_of_range_boxes": self.out_of_range_boxes,
            "class_counts": dict(self.class_counts or {}),
        }


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Inventory all dataset assets under datasets/raw, datasets/interim, and datasets/final.")
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--output-dir", default=f"datasets/reports/dataset_asset_inventory_{stamp}")
    parser.add_argument("--hash-images", action="store_true", help="Hash image bytes to estimate duplicates. Slower but more accurate.")
    parser.add_argument("--max-hash-mb", type=float, default=20.0, help="Skip hashing files larger than this size.")
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in VIDEO_EXTS:
        return "video"
    if suffix in ARCHIVE_EXTS:
        return "archive"
    if suffix in LABEL_EXTS:
        return "label_or_text"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix in {".json", ".csv"}:
        return "metadata"
    return "other"


def hash_file(path: Path, max_bytes: int) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def load_names(dataset_root: Path) -> dict[str, str]:
    candidates = [dataset_root / "data.yaml", dataset_root / "dataset.yaml"]
    candidates.extend(dataset_root.glob("*.yaml"))
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
        except Exception:
            continue
        names = data.get("names")
        if isinstance(names, dict):
            return {str(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {str(i): str(v) for i, v in enumerate(names)}
    classes = dataset_root / "classes.txt"
    if classes.exists():
        return {str(i): line.strip() for i, line in enumerate(classes.read_text(encoding="utf-8", errors="ignore").splitlines()) if line.strip()}
    return dict(DEFAULT_PROJECT_NAMES)


def parse_yolo_labels(label_dir: Path, known_names: dict[str, str]) -> YoloStats:
    stats = YoloStats()
    if not label_dir.exists():
        return stats
    known_classes = set(known_names) if known_names else {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    for label_path in sorted(label_dir.glob("*.txt")):
        stats.label_files += 1
        text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            stats.empty_label_files += 1
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                stats.invalid_lines += 1
                continue
            cls = parts[0]
            try:
                cls_value = int(float(cls))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                stats.invalid_lines += 1
                continue
            cls_key = str(cls_value)
            if cls_key not in known_classes:
                stats.unknown_classes += 1
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                stats.out_of_range_boxes += 1
                continue
            stats.valid_box_lines += 1
            label = known_names.get(cls_key, cls_key)
            stats.class_counts[f"{cls_key}:{label}"] += 1
    return stats


def find_dataset_roots(stage_root: Path) -> list[Path]:
    roots: set[Path] = set()
    if not stage_root.exists():
        return []
    for label_dir in stage_root.rglob("labels"):
        if label_dir.is_dir():
            parent = label_dir.parent
            if (parent / "images").exists() or any((parent / split / "images").exists() for split in SPLITS):
                roots.add(parent)
    for yaml_path in stage_root.rglob("data.yaml"):
        roots.add(yaml_path.parent)
    sorted_roots = sorted(roots, key=lambda p: len(p.parts))
    pruned: list[Path] = []
    for root in sorted_roots:
        if any(root != existing and root.is_relative_to(existing) for existing in pruned):
            continue
        pruned.append(root)
    return pruned


def split_image_count(dataset_root: Path, split: str) -> int:
    candidates = [dataset_root / "images" / split, dataset_root / split / "images"]
    total = 0
    for path in candidates:
        if path.exists():
            total += sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return total


def split_label_count(dataset_root: Path, split: str) -> int:
    candidates = [dataset_root / "labels" / split, dataset_root / split / "labels"]
    total = 0
    for path in candidates:
        if path.exists():
            total += sum(1 for p in path.rglob("*.txt") if p.is_file())
    return total


def merge_yolo_stats(target: YoloStats, source: YoloStats) -> None:
    target.label_files += source.label_files
    target.empty_label_files += source.empty_label_files
    target.valid_box_lines += source.valid_box_lines
    target.invalid_lines += source.invalid_lines
    target.unknown_classes += source.unknown_classes
    target.out_of_range_boxes += source.out_of_range_boxes
    target.class_counts.update(source.class_counts or {})


def collect_yolo_stats(dataset_root: Path, names: dict[str, str]) -> YoloStats:
    aggregate = YoloStats()
    candidate_dirs: list[Path] = []
    for split in SPLITS:
        candidate_dirs.append(dataset_root / "labels" / split)
        candidate_dirs.append(dataset_root / split / "labels")
    candidate_dirs.append(dataset_root / "labels")

    seen: set[Path] = set()
    for label_dir in candidate_dirs:
        if not label_dir.exists() or not label_dir.is_dir():
            continue
        resolved = label_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        merge_yolo_stats(aggregate, parse_yolo_labels(label_dir, names))
    return aggregate


def image_count_under(path: Path) -> int:
    total = 0
    for candidate in [path / "images", *(path / split / "images" for split in SPLITS)]:
        if candidate.exists():
            total += sum(1 for p in candidate.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if total:
        return total
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_count_under(path: Path) -> int:
    total = 0
    for candidate in [path / "labels", *(path / split / "labels" for split in SPLITS)]:
        if candidate.exists():
            total += sum(1 for p in candidate.rglob("*.txt") if p.is_file())
    if total:
        return total
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def inventory_stage(stage_root: Path, stage_name: str, *, hash_images: bool, max_hash_bytes: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[str]]]:
    file_counts: Counter[str] = Counter()
    bytes_by_type: Counter[str] = Counter()
    image_hashes: dict[str, list[str]] = defaultdict(list)
    total_files = 0
    total_bytes = 0

    for path in stage_root.rglob("*") if stage_root.exists() else []:
        if not path.is_file():
            continue
        total_files += 1
        kind = classify_file(path)
        file_counts[kind] += 1
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        total_bytes += size
        bytes_by_type[kind] += size
        if hash_images and kind == "image":
            digest = hash_file(path, max_hash_bytes)
            if digest:
                image_hashes[digest].append(str(path))

    dataset_rows: list[dict[str, Any]] = []
    for dataset_root in find_dataset_roots(stage_root):
        names = load_names(dataset_root)
        yolo_stats = collect_yolo_stats(dataset_root, names)
        row = {
            "stage": stage_name,
            "dataset_root": str(dataset_root),
            "relative_root": str(dataset_root.relative_to(ROOT)) if dataset_root.is_relative_to(ROOT) else str(dataset_root),
            "image_files_total": image_count_under(dataset_root),
            "label_txt_total": label_count_under(dataset_root),
            "names": names,
            "yolo": yolo_stats.to_dict(),
            "split_images": {split: split_image_count(dataset_root, split) for split in ("train", "val", "valid", "test")},
            "split_labels": {split: split_label_count(dataset_root, split) for split in ("train", "val", "valid", "test")},
        }
        dataset_rows.append(row)

    stage_summary = {
        "stage": stage_name,
        "root": str(stage_root),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "file_counts": dict(file_counts),
        "bytes_by_type": dict(bytes_by_type),
        "detected_dataset_roots": len(dataset_rows),
        "image_files_total": file_counts.get("image", 0),
        "video_files_total": file_counts.get("video", 0),
        "archive_files_total": file_counts.get("archive", 0),
        "label_or_text_files_total": file_counts.get("label_or_text", 0),
    }
    return stage_summary, dataset_rows, image_hashes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "stage",
        "relative_root",
        "image_files_total",
        "label_txt_total",
        "yolo_label_files",
        "yolo_empty_label_files",
        "yolo_valid_box_lines",
        "yolo_invalid_lines",
        "yolo_unknown_classes",
        "yolo_out_of_range_boxes",
        "class_counts",
        "split_images",
        "split_labels",
        "names",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            yolo = row.get("yolo", {})
            writer.writerow(
                {
                    "stage": row.get("stage"),
                    "relative_root": row.get("relative_root"),
                    "image_files_total": row.get("image_files_total"),
                    "label_txt_total": row.get("label_txt_total"),
                    "yolo_label_files": yolo.get("label_files"),
                    "yolo_empty_label_files": yolo.get("empty_label_files"),
                    "yolo_valid_box_lines": yolo.get("valid_box_lines"),
                    "yolo_invalid_lines": yolo.get("invalid_lines"),
                    "yolo_unknown_classes": yolo.get("unknown_classes"),
                    "yolo_out_of_range_boxes": yolo.get("out_of_range_boxes"),
                    "class_counts": json.dumps(yolo.get("class_counts", {}), ensure_ascii=False),
                    "split_images": json.dumps(row.get("split_images", {}), ensure_ascii=False),
                    "split_labels": json.dumps(row.get("split_labels", {}), ensure_ascii=False),
                    "names": json.dumps(row.get("names", {}), ensure_ascii=False),
                }
            )


def write_markdown(path: Path, summary: dict[str, Any], dataset_rows: list[dict[str, Any]], duplicate_summary: dict[str, Any]) -> None:
    lines = [
        "# 数据资产盘点报告",
        "",
        f"- 扫描根目录: `{summary['datasets_root']}`",
        f"- 图片文件总数: `{summary['totals']['image_files']}`",
        f"- 视频文件总数: `{summary['totals']['video_files']}`",
        f"- 压缩包总数: `{summary['totals']['archive_files']}`",
        f"- 检测到的数据集根目录: `{len(dataset_rows)}`",
        f"- YOLO 有效框总数: `{summary['totals']['valid_yolo_boxes']}`",
        "",
        "## 分阶段资产",
        "",
    ]
    for stage in summary["stages"]:
        lines.append(
            f"- `{stage['stage']}`: 图片 `{stage['image_files_total']}`, 视频 `{stage['video_files_total']}`, "
            f"压缩包 `{stage['archive_files_total']}`, 数据集根 `{stage['detected_dataset_roots']}`"
        )
    lines.extend(["", "## 主要 YOLO 数据集", ""])
    for row in sorted(dataset_rows, key=lambda r: int(r["yolo"]["valid_box_lines"]), reverse=True)[:20]:
        yolo = row["yolo"]
        lines.append(
            f"- `{row['relative_root']}`: 图片 `{row['image_files_total']}`, 标签 `{yolo['label_files']}`, "
            f"有效框 `{yolo['valid_box_lines']}`, 空标签 `{yolo['empty_label_files']}`, 无效行 `{yolo['invalid_lines']}`, "
            f"类别 `{yolo['class_counts']}`"
        )
    lines.extend(["", "## 重复图片估计", ""])
    if duplicate_summary.get("enabled"):
        lines.append(f"- 已 hash 图片: `{duplicate_summary['hashed_images']}`")
        lines.append(f"- 重复 hash 组: `{duplicate_summary['duplicate_groups']}`")
        lines.append(f"- 重复图片数: `{duplicate_summary['duplicate_images']}`")
    else:
        lines.append("- 未启用图片 hash。需要去重时运行 `--hash-images`。")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 当前项目不是没有数据，而是存在多阶段重复资产、原始归档、训练子集和候选伪标签混在一起的问题。",
            "- 下一轮 Google 训练应优先使用 `datasets/final/smoke_bal` 加确认过的 `datasets/interim/*_yolo3cls`，不要把 raw 目录直接混入训练。",
            "- `hmdb51_smoke` 和自定义视频主要用于视频时序验证/抽帧候选，不是天然 YOLO 检测训练集。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    datasets_root = repo_path(args.datasets_root)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_hash_bytes = int(args.max_hash_mb * 1024 * 1024)

    stages: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    all_hashes: dict[str, list[str]] = defaultdict(list)
    for stage_name in STAGES:
        stage_summary, rows, hashes = inventory_stage(
            datasets_root / stage_name,
            stage_name,
            hash_images=args.hash_images,
            max_hash_bytes=max_hash_bytes,
        )
        stages.append(stage_summary)
        dataset_rows.extend(rows)
        for digest, paths in hashes.items():
            all_hashes[digest].extend(paths)

    duplicate_groups = {digest: paths for digest, paths in all_hashes.items() if len(paths) > 1}
    duplicate_summary = {
        "enabled": bool(args.hash_images),
        "hashed_images": sum(len(paths) for paths in all_hashes.values()),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_images": sum(len(paths) for paths in duplicate_groups.values()),
    }
    totals = {
        "image_files": sum(stage["image_files_total"] for stage in stages),
        "video_files": sum(stage["video_files_total"] for stage in stages),
        "archive_files": sum(stage["archive_files_total"] for stage in stages),
        "label_or_text_files": sum(stage["label_or_text_files_total"] for stage in stages),
        "valid_yolo_boxes": sum(int(row["yolo"]["valid_box_lines"]) for row in dataset_rows),
        "empty_yolo_label_files": sum(int(row["yolo"]["empty_label_files"]) for row in dataset_rows),
        "invalid_yolo_lines": sum(int(row["yolo"]["invalid_lines"]) for row in dataset_rows),
    }
    summary = {
        "datasets_root": str(datasets_root),
        "output_dir": str(output_dir),
        "hash_images": bool(args.hash_images),
        "totals": totals,
        "stages": stages,
        "duplicate_summary": duplicate_summary,
    }

    (output_dir / "dataset_asset_inventory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dataset_asset_inventory_datasets.json").write_text(json.dumps(dataset_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "dataset_asset_inventory_datasets.csv", dataset_rows)
    if args.hash_images:
        (output_dir / "duplicate_image_hashes.json").write_text(json.dumps(duplicate_groups, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_dir / "dataset_asset_inventory_report.md", summary, dataset_rows, duplicate_summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Dataset inventory saved to: {output_dir}")


if __name__ == "__main__":
    main()
