from __future__ import annotations

import json
import random
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


RANDOM_SEED = 42
DEFAULT_SPLIT = (0.7, 0.15, 0.15)
REPORT_PATH = Path("datasets/reports/added_datasets_prepare_report.json")


@dataclass(frozen=True)
class ZipDatasetSpec:
    name: str
    zip_path: Path
    output_root: Path
    class_map: dict[str, int]
    ignored_classes: set[str]
    split_ratio: tuple[float, float, float] = DEFAULT_SPLIT
    enabled: bool = True
    skip_reason: str = ""


DATASETS = (
    ZipDatasetSpec(
        name="cigarette_new",
        zip_path=Path("Cigarette.yolov8.zip"),
        output_root=Path("datasets/interim/cigarette_yolo"),
        class_map={"0": 0},
        ignored_classes={"1"},
    ),
    ZipDatasetSpec(
        name="smoke_new",
        zip_path=Path("smoke.yolov8 (1).zip"),
        output_root=Path("datasets/interim/smoke_yolo"),
        class_map={"0": 2},
        ignored_classes=set(),
    ),
    ZipDatasetSpec(
        name="smoke_legacy",
        zip_path=Path("smoke.yolov8.zip"),
        output_root=Path("datasets/interim/smoke_legacy_yolo"),
        class_map={"0": 2},
        ignored_classes={"1"},
        enabled=True,
        skip_reason="",
    ),
)


def reset_output(root: Path) -> None:
    for split in ("train", "val", "test"):
        for folder in ("images", "labels"):
            target = root / folder / split
            target.mkdir(parents=True, exist_ok=True)
            for item in target.iterdir():
                if item.is_file():
                    item.unlink()


def build_split_map(stems: list[str], seed: int, split_ratio: tuple[float, float, float]) -> dict[str, set[str]]:
    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])
    return {
        "train": set(shuffled[:train_end]),
        "val": set(shuffled[train_end:val_end]),
        "test": set(shuffled[val_end:]),
    }


def find_split_for_stem(stem: str, split_map: dict[str, set[str]]) -> str:
    for split, stems in split_map.items():
        if stem in stems:
            return split
    raise KeyError(f"Stem not found in split map: {stem}")


def remap_label_text(text: str, spec: ZipDatasetSpec, stats: dict) -> str:
    output_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            stats["malformed_lines"] += 1
            continue

        cls = parts[0]
        stats["original_class_counts"][cls] += 1
        if cls in spec.ignored_classes:
            stats["ignored_class_counts"][cls] += 1
            continue
        if cls not in spec.class_map:
            stats["unknown_class_counts"][cls] += 1
            continue

        try:
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            stats["malformed_lines"] += 1
            continue

        x, y, w, h = coords
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            stats["out_of_range_boxes"] += 1
            continue

        mapped_cls = spec.class_map[cls]
        output_lines.append(
            f"{mapped_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )
        stats["mapped_class_counts"][str(mapped_cls)] += 1
    return "\n".join(output_lines)


def prepare_dataset(spec: ZipDatasetSpec, seed_offset: int) -> dict:
    if not spec.enabled:
        return {
            "enabled": False,
            "skip_reason": spec.skip_reason,
            "zip_path": str(spec.zip_path),
        }

    reset_output(spec.output_root)
    stats = {
        "enabled": True,
        "zip_path": str(spec.zip_path),
        "output_root": str(spec.output_root),
        "split_ratio": list(spec.split_ratio),
        "images_written": Counter(),
        "labels_written": Counter(),
        "empty_labels_after_remap": 0,
        "malformed_lines": 0,
        "out_of_range_boxes": 0,
        "original_class_counts": Counter(),
        "mapped_class_counts": Counter(),
        "ignored_class_counts": Counter(),
        "unknown_class_counts": Counter(),
        "missing_label_files": [],
    }

    with zipfile.ZipFile(spec.zip_path) as zf:
        image_members = {
            Path(member).stem: member
            for member in zf.namelist()
            if member.startswith("train/images/") and not member.endswith("/")
        }
        label_members = {
            Path(member).stem: member
            for member in zf.namelist()
            if member.startswith("train/labels/") and member.endswith(".txt")
        }

        split_map = build_split_map(sorted(image_members), RANDOM_SEED + seed_offset, spec.split_ratio)

        for stem, image_member in sorted(image_members.items()):
            label_member = label_members.get(stem)
            if label_member is None:
                stats["missing_label_files"].append(stem)
                continue

            split = find_split_for_stem(stem, split_map)
            image_name = Path(image_member).name
            label_name = f"{stem}.txt"

            label_text = zf.read(label_member).decode("utf-8", errors="ignore").strip()
            remapped = remap_label_text(label_text, spec, stats)
            if not remapped:
                stats["empty_labels_after_remap"] += 1
                continue

            image_target = spec.output_root / "images" / split / image_name
            label_target = spec.output_root / "labels" / split / label_name
            image_target.write_bytes(zf.read(image_member))
            label_target.write_text(remapped, encoding="utf-8")

            stats["images_written"][split] += 1
            stats["labels_written"][split] += 1

    return {
        **stats,
        "images_written": dict(stats["images_written"]),
        "labels_written": dict(stats["labels_written"]),
        "original_class_counts": dict(stats["original_class_counts"]),
        "mapped_class_counts": dict(stats["mapped_class_counts"]),
        "ignored_class_counts": dict(stats["ignored_class_counts"]),
        "unknown_class_counts": dict(stats["unknown_class_counts"]),
    }


def main() -> None:
    report = {
        "random_seed": RANDOM_SEED,
        "datasets": {},
    }

    for index, spec in enumerate(DATASETS):
        report["datasets"][spec.name] = prepare_dataset(spec, index)

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
