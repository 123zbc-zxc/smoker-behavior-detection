from __future__ import annotations

import json
from pathlib import Path


DATASET_ROOT = Path("datasets/final/smoking_yolo_3cls_full")
REPORT_PATH = Path("datasets/reports/final_dataset_clean_report.json")


def is_standard_yolo_label(text: str) -> bool:
    if not text:
        return False
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            return False
        try:
            cls = parts[0]
            values = list(map(float, parts[1:]))
        except ValueError:
            return False
        if cls not in {"0", "1", "2"}:
            return False
        x, y, w, h = values
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            return False
    return True


def main() -> None:
    report = {
        "deleted_empty_label_pairs": [],
        "deleted_invalid_label_pairs": [],
        "scanned_files": 0,
    }

    for split in ("train", "val", "test"):
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split
        image_index: dict[str, list[Path]] = {}

        for image_path in image_dir.iterdir():
            if image_path.is_file():
                image_index.setdefault(image_path.stem, []).append(image_path)

        for label_path in sorted(label_dir.glob("*.txt")):
            report["scanned_files"] += 1
            text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
            image_matches = image_index.get(label_path.stem, [])

            if not text:
                label_path.unlink(missing_ok=True)
                for image_path in image_matches:
                    image_path.unlink(missing_ok=True)
                report["deleted_empty_label_pairs"].append(
                    {
                        "split": split,
                        "label": label_path.name,
                        "images": [p.name for p in image_matches],
                    }
                )
                continue

            if not is_standard_yolo_label(text):
                label_path.unlink(missing_ok=True)
                for image_path in image_matches:
                    image_path.unlink(missing_ok=True)
                report["deleted_invalid_label_pairs"].append(
                    {
                        "split": split,
                        "label": label_path.name,
                        "images": [p.name for p in image_matches],
                    }
                )

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
