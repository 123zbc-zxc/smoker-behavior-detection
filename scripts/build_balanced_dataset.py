from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = {
    "0": "cigarette",
    "1": "smoking_person",
    "2": "smoke",
}
DEFAULT_SOURCE_ROOT = Path("datasets/final/smoking_yolo_3cls_full")
DEFAULT_OUTPUT_ROOT = Path("datasets/final/smoke_bal")
DEFAULT_REPORT_PATH = Path("datasets/reports/balanced_dataset_report.json")
DEFAULT_SEED = 42
DEFAULT_MULTIPLIER = 1.5


@dataclass(frozen=True)
class SampleRecord:
    split: str
    stem: str
    image_path: Path
    label_path: Path
    class_counts: Counter[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mildly balanced smoker dataset for CPU training.")
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Source YOLO dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Balanced dataset output root.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--smoking-person-target",
        type=int,
        help="Explicit target count for class 1 instances after downsampling.",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=DEFAULT_MULTIPLIER,
        help="If no explicit target is set, cap class 1 at multiplier * max(class0, class2).",
    )
    return parser.parse_args()


def reset_output_dirs(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        for folder in ("images", "labels"):
            target = output_root / folder / split
            target.mkdir(parents=True, exist_ok=True)
            for item in target.iterdir():
                if item.is_file():
                    item.unlink()


def build_image_index(image_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in image_dir.iterdir() if path.is_file()}


def parse_label_counts(label_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] in CLASS_NAMES:
            counts[parts[0]] += 1
    return counts


def collect_records(source_root: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for split in ("train", "val", "test"):
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
        image_index = build_image_index(image_dir)
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = image_index.get(label_path.stem)
            if image_path is None:
                raise FileNotFoundError(f"Missing image for {label_path.stem} in {image_dir}")
            records.append(
                SampleRecord(
                    split=split,
                    stem=label_path.stem,
                    image_path=image_path,
                    label_path=label_path,
                    class_counts=parse_label_counts(label_path),
                )
            )
    return records


def total_class_counts(records: list[SampleRecord]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for record in records:
        totals.update(record.class_counts)
    return totals


def choose_target_class1(records: list[SampleRecord], explicit_target: int | None, multiplier: float) -> int:
    totals = total_class_counts(records)
    minority_peak = max(totals.get("0", 0), totals.get("2", 0))
    computed = int(round(minority_peak * multiplier))
    if explicit_target is not None:
        computed = explicit_target
    return min(totals.get("1", 0), max(computed, 0))


def split_priority_and_optional(records: list[SampleRecord]) -> tuple[list[SampleRecord], list[SampleRecord]]:
    priority: list[SampleRecord] = []
    optional: list[SampleRecord] = []
    for record in records:
        has_minority = record.class_counts.get("0", 0) > 0 or record.class_counts.get("2", 0) > 0
        if has_minority:
            priority.append(record)
        elif record.class_counts.get("1", 0) > 0:
            optional.append(record)
    return priority, optional


def sample_records(
    records: list[SampleRecord],
    target_class1: int,
    seed: int,
) -> tuple[list[SampleRecord], Counter[str], int]:
    priority, optional = split_priority_and_optional(records)
    chosen = list(priority)
    chosen_counts = total_class_counts(chosen)

    if chosen_counts.get("1", 0) >= target_class1:
        return chosen, chosen_counts, 0

    rng = random.Random(seed)
    rng.shuffle(optional)
    added = 0
    for record in optional:
        next_total = chosen_counts.get("1", 0) + record.class_counts.get("1", 0)
        if next_total > target_class1:
            continue
        chosen.append(record)
        chosen_counts.update(record.class_counts)
        added += 1
        if chosen_counts.get("1", 0) >= target_class1:
            break

    return chosen, chosen_counts, added


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_records(records: list[SampleRecord], output_root: Path) -> Counter[str]:
    per_split_images: Counter[str] = Counter()
    for record in records:
        dst_image = output_root / "images" / record.split / f"{record.stem}{record.image_path.suffix.lower()}"
        dst_label = output_root / "labels" / record.split / f"{record.stem}.txt"
        link_or_copy(record.image_path, dst_image)
        link_or_copy(record.label_path, dst_label)
        per_split_images[record.split] += 1
    return per_split_images


def build_report(
    source_root: Path,
    output_root: Path,
    target_class1: int,
    all_records: list[SampleRecord],
    chosen_records: list[SampleRecord],
    chosen_counts: Counter[str],
    per_split_images: Counter[str],
    optional_added: int,
    seed: int,
    multiplier: float,
) -> dict:
    original_counts = total_class_counts(all_records)
    priority_records, optional_records = split_priority_and_optional(all_records)
    chosen_split_counts: Counter[str] = Counter(record.split for record in chosen_records)
    return {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": seed,
        "multiplier": multiplier,
        "smoking_person_target": target_class1,
        "original_class_counts": dict(original_counts),
        "balanced_class_counts": dict(chosen_counts),
        "original_image_count": len(all_records),
        "balanced_image_count": len(chosen_records),
        "priority_image_count": len(priority_records),
        "optional_class1_only_count": len(optional_records),
        "optional_class1_only_added": optional_added,
        "images_per_split": dict(per_split_images),
        "selected_images_per_split": dict(chosen_split_counts),
        "class_names": CLASS_NAMES,
    }


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    report_path = Path(args.report)

    all_records = collect_records(source_root)
    target_class1 = choose_target_class1(all_records, args.smoking_person_target, args.multiplier)
    chosen_records, chosen_counts, optional_added = sample_records(all_records, target_class1, args.seed)

    reset_output_dirs(output_root)
    per_split_images = copy_records(chosen_records, output_root)

    report = build_report(
        source_root=source_root,
        output_root=output_root,
        target_class1=target_class1,
        all_records=all_records,
        chosen_records=chosen_records,
        chosen_counts=chosen_counts,
        per_split_images=per_split_images,
        optional_added=optional_added,
        seed=args.seed,
        multiplier=args.multiplier,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nBalanced dataset saved to: {output_root}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
