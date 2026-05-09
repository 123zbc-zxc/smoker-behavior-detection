from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image


DATASET_ROOT = Path("datasets/final/smoking_yolo_3cls_full")
REPORT_PATH = Path("datasets/reports/final_dataset_check_report.json")
KNOWN_CLASSES = {"0", "1", "2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO smoker dataset.")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT), help="Dataset root containing images/ and labels/.")
    parser.add_argument("--report", default=str(REPORT_PATH), help="Validation report output path.")
    return parser.parse_args()


def iter_pairs(dataset_root: Path, split: str) -> Iterable[tuple[Path, Path]]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    for image_path in sorted(image_dir.iterdir()):
        if image_path.is_file():
            yield image_path, label_dir / f"{image_path.stem}.txt"


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    report_path = Path(args.report)

    report: dict = {
        "dataset_root": str(dataset_root),
        "splits": {},
        "global_class_counts": Counter(),
        "global_issues": {
            "missing_label_files": [],
            "missing_image_files": [],
            "empty_label_files": [],
            "invalid_label_lines": [],
            "unknown_classes": [],
            "out_of_range_boxes": [],
        },
    }

    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        split_info = {
            "images": 0,
            "labels": 0,
            "matched_pairs": 0,
            "class_counts": Counter(),
            "image_sizes": Counter(),
        }

        image_stems = {p.stem for p in image_dir.iterdir() if p.is_file()}
        label_stems = {p.stem for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}
        split_info["images"] = len(image_stems)
        split_info["labels"] = len(label_stems)

        for missing in sorted(image_stems - label_stems):
            report["global_issues"]["missing_label_files"].append({"split": split, "stem": missing})
        for missing in sorted(label_stems - image_stems):
            report["global_issues"]["missing_image_files"].append({"split": split, "stem": missing})

        for image_path, label_path in iter_pairs(dataset_root, split):
            if not label_path.exists():
                continue
            split_info["matched_pairs"] += 1

            with Image.open(image_path) as img:
                split_info["image_sizes"][f"{img.width}x{img.height}"] += 1

            text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                report["global_issues"]["empty_label_files"].append({"split": split, "file": label_path.name})
                continue

            for line_no, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    report["global_issues"]["invalid_label_lines"].append(
                        {"split": split, "file": label_path.name, "line_no": line_no, "line": line}
                    )
                    continue

                cls, x, y, w, h = parts
                if cls not in KNOWN_CLASSES:
                    report["global_issues"]["unknown_classes"].append(
                        {"split": split, "file": label_path.name, "line_no": line_no, "class": cls}
                    )
                    continue

                try:
                    x, y, w, h = map(float, (x, y, w, h))
                except ValueError:
                    report["global_issues"]["invalid_label_lines"].append(
                        {"split": split, "file": label_path.name, "line_no": line_no, "line": line}
                    )
                    continue

                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    report["global_issues"]["out_of_range_boxes"].append(
                        {"split": split, "file": label_path.name, "line_no": line_no, "line": line}
                    )
                    continue

                split_info["class_counts"][cls] += 1
                report["global_class_counts"][cls] += 1

        split_info["class_counts"] = dict(split_info["class_counts"])
        split_info["image_sizes"] = dict(split_info["image_sizes"].most_common(20))
        report["splits"][split] = split_info

    report["global_class_counts"] = dict(report["global_class_counts"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
