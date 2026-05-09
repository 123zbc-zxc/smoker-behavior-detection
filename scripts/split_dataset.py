from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_utils import (
    DatasetPair,
    collect_yolo_pairs,
    ensure_clean_yolo_output,
    transfer_file,
)
from scripts.yolo_utils import dump_json


DEFAULT_REPORT = "datasets/reports/split_dataset_report.json"
OUTPUT_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a YOLO dataset into train/val/test directories.")
    parser.add_argument("--source-root", required=True, help="Source YOLO dataset root containing images/ and labels/.")
    parser.add_argument("--output-root", required=True, help="Output YOLO dataset root.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to materialize files into the output dataset.",
    )
    parser.add_argument("--report", default=DEFAULT_REPORT, help="JSON report output path.")
    return parser.parse_args()


def validate_ratios(args: argparse.Namespace) -> tuple[float, float, float]:
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(value < 0 for value in ratios):
        raise ValueError("Split ratios must be non-negative.")
    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum:.6f}."
        )
    return ratios


def parse_label_counts(label_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5:
            counts[parts[0]] += 1
    return counts


def assign_output_stems(pairs: list[DatasetPair]) -> dict[DatasetPair, str]:
    usage: Counter[str] = Counter()
    assigned: dict[DatasetPair, str] = {}
    for pair in pairs:
        base_stem = pair.stem
        if usage[base_stem] == 0:
            output_stem = base_stem
        else:
            group_prefix = pair.group or "flat"
            output_stem = f"{group_prefix}_{base_stem}"
            while usage[output_stem] > 0:
                output_stem = f"{output_stem}_{usage[output_stem]}"
        usage[output_stem] += 1
        assigned[pair] = output_stem
    return assigned


def split_pairs(
    pairs: list[DatasetPair],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[DatasetPair]]:
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def build_report(
    args: argparse.Namespace,
    split_records: dict[str, list[DatasetPair]],
    output_stems: dict[DatasetPair, str],
    source_group_counts: Counter[str],
) -> dict[str, Any]:
    per_split_images = {split: len(records) for split, records in split_records.items()}
    class_counts = {
        split: dict(sum((parse_label_counts(pair.label_path) for pair in records), Counter()))
        for split, records in split_records.items()
    }
    renamed_pairs = [
        {
            "source_group": pair.group or "flat",
            "source_stem": pair.stem,
            "output_stem": output_stems[pair],
        }
        for split in OUTPUT_SPLITS
        for pair in split_records[split]
        if output_stems[pair] != pair.stem
    ]
    return {
        "source_root": str(Path(args.source_root)),
        "output_root": str(Path(args.output_root)),
        "report_path": str(Path(args.report)),
        "copy_mode": args.copy_mode,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "source_group_counts": dict(source_group_counts),
        "images_per_split": per_split_images,
        "class_counts_per_split": class_counts,
        "renamed_pair_count": len(renamed_pairs),
        "renamed_pairs": renamed_pairs[:100],
    }


def main() -> None:
    args = parse_args()
    train_ratio, val_ratio, _ = validate_ratios(args)
    pairs, issues = collect_yolo_pairs(args.source_root)
    if issues["missing_images"] or issues["missing_labels"]:
        raise ValueError(
            "Source dataset has unmatched image/label files. "
            "Resolve them before splitting."
        )
    if not pairs:
        raise ValueError(f"No image-label pairs found in {args.source_root}.")

    split_records = split_pairs(
        pairs=pairs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=args.seed,
    )
    output_stems = assign_output_stems(pairs)
    output_root = ensure_clean_yolo_output(args.output_root, OUTPUT_SPLITS)

    source_group_counts: Counter[str] = Counter((pair.group or "flat") for pair in pairs)
    for split, records in split_records.items():
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        for pair in records:
            output_stem = output_stems[pair]
            output_image = image_dir / f"{output_stem}{pair.image_path.suffix.lower()}"
            output_label = label_dir / f"{output_stem}.txt"
            transfer_file(pair.image_path, output_image, mode=args.copy_mode)
            transfer_file(pair.label_path, output_label, mode=args.copy_mode)

    report = build_report(args, split_records, output_stems, source_group_counts)
    report["source_issues"] = issues
    dump_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSplit dataset saved to: {Path(args.output_root)}")
    print(f"Report saved to: {Path(args.report)}")


if __name__ == "__main__":
    main()
