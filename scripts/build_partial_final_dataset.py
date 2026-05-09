from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


RANDOM_SEED = 42
AISTUDIO_SPLIT = (0.7, 0.15, 0.15)

AISTUDIO_ROOT = Path("datasets/interim/aistudio_yolo")
ROBOFLOW_ROOT = Path("datasets/interim/roboflow_remap")
CIGARETTE_ROOT = Path("datasets/interim/cigarette_yolo")
SMOKE_ROOT = Path("datasets/interim/smoke_yolo")
SMOKE_LEGACY_ROOT = Path("datasets/interim/smoke_legacy_yolo")
FINAL_ROOT = Path("datasets/final/smoking_yolo_3cls_full")
REPORT_PATH = Path("datasets/reports/partial_final_dataset_report.json")
INCLUDE_SMOKE_LEGACY = os.environ.get("INCLUDE_SMOKE_LEGACY", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
SOURCE_PREFIX = {
    "aistudio": "ai",
    "roboflow": "rf",
    "cigarette": "cg",
    "smoke": "sm",
    "smoke_legacy": "sl",
}


def reset_split_dirs() -> None:
    for split in ("train", "val", "test"):
        for folder in ("images", "labels"):
            target = FINAL_ROOT / folder / split
            target.mkdir(parents=True, exist_ok=True)
            for item in target.iterdir():
                if item.is_file():
                    item.unlink()


def copy_pair(src_img: Path, src_label: Path, dst_img: Path, dst_label: Path) -> None:
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_label, dst_label)


def split_aistudio_files() -> dict[str, list[str]]:
    image_dir = AISTUDIO_ROOT / "images"
    stems = sorted(p.stem for p in image_dir.glob("*") if p.is_file())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(stems)

    total = len(stems)
    train_end = int(total * AISTUDIO_SPLIT[0])
    val_end = train_end + int(total * AISTUDIO_SPLIT[1])
    return {
        "train": stems[:train_end],
        "val": stems[train_end:val_end],
        "test": stems[val_end:],
    }


def merge_source(source_name: str, source_root: Path, split_map: dict[str, list[str]] | None, stats: dict) -> None:
    if split_map is None:
        split_map = {}
        for split in ("train", "val", "test"):
            stems = [p.stem for p in (source_root / "images" / split).glob("*") if p.is_file()]
            split_map[split] = sorted(stems)

    for split, stems in split_map.items():
        if source_name == "aistudio":
            src_image_dir = source_root / "images"
            src_label_dir = source_root / "labels"
        else:
            src_image_dir = source_root / "images" / split
            src_label_dir = source_root / "labels" / split

        for stem in stems:
            src_img = next(iter(list(src_image_dir.glob(f"{stem}.*"))), None)
            src_label = src_label_dir / f"{stem}.txt"
            if src_img is None or not src_label.exists():
                stats["missing_pairs"].append({"source": source_name, "split": split, "stem": stem})
                continue

            prefix = SOURCE_PREFIX.get(source_name, source_name)
            prefixed_stem = f"{prefix}_{stem}"
            dst_img = FINAL_ROOT / "images" / split / f"{prefixed_stem}{src_img.suffix.lower()}"
            dst_label = FINAL_ROOT / "labels" / split / f"{prefixed_stem}.txt"
            copy_pair(src_img, src_label, dst_img, dst_label)

            stats["images_per_split"][split] += 1
            stats["labels_per_split"][split] += 1
            stats["source_counts"][source_name] += 1

            text = src_label.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                stats["empty_labels"] += 1
                continue
            for line in text.splitlines():
                cls = line.split()[0]
                stats["class_counts"][cls] += 1


def main() -> None:
    reset_split_dirs()

    stats = {
        "random_seed": RANDOM_SEED,
        "aistudio_split_ratio": AISTUDIO_SPLIT,
        "include_smoke_legacy": INCLUDE_SMOKE_LEGACY,
        "images_per_split": Counter(),
        "labels_per_split": Counter(),
        "source_counts": Counter(),
        "class_counts": Counter(),
        "empty_labels": 0,
        "missing_pairs": [],
    }

    aistudio_splits = split_aistudio_files()
    merge_source("aistudio", AISTUDIO_ROOT, aistudio_splits, stats)
    merge_source("roboflow", ROBOFLOW_ROOT, None, stats)
    merge_source("cigarette", CIGARETTE_ROOT, None, stats)
    merge_source("smoke", SMOKE_ROOT, None, stats)
    if INCLUDE_SMOKE_LEGACY:
        merge_source("smoke_legacy", SMOKE_LEGACY_ROOT, None, stats)

    serializable = {
        **stats,
        "images_per_split": dict(stats["images_per_split"]),
        "labels_per_split": dict(stats["labels_per_split"]),
        "source_counts": dict(stats["source_counts"]),
        "class_counts": dict(stats["class_counts"]),
    }
    REPORT_PATH.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(serializable, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
