from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path


SOURCE_ZIP = Path("datasets/raw/roboflow_smoking_drinking/SmokingAndDrinking.v1i.yolov8.zip")
OUTPUT_ROOT = Path("datasets/interim/roboflow_remap")
REPORT_PATH = Path("datasets/reports/roboflow_prepare_report.json")

# Roboflow exports two near-duplicate behavior classes, both mapped into the
# project's unified "smoking_person" class id.
CLASS_REMAP = {"0": 1, "1": 1}
SPLIT_REMAP = {"train": "train", "valid": "val", "test": "test"}


def ensure_dirs() -> None:
    for split in SPLIT_REMAP.values():
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def remap_label_text(text: str, stats: dict) -> str:
    lines_out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        cls = parts[0]
        stats["original_class_counts"][cls] += 1
        if cls not in CLASS_REMAP:
            stats["unknown_class_counts"][cls] += 1
            continue
        mapped_cls = CLASS_REMAP[cls]
        parts[0] = str(mapped_cls)
        stats["remapped_class_counts"][str(mapped_cls)] += 1
        lines_out.append(" ".join(parts))
    return "\n".join(lines_out)


def main() -> None:
    ensure_dirs()

    stats = {
        "source_zip": str(SOURCE_ZIP),
        "output_root": str(OUTPUT_ROOT),
        "files_in_zip": 0,
        "image_files_extracted": Counter(),
        "label_files_written": Counter(),
        "empty_label_files": 0,
        "original_class_counts": Counter(),
        "remapped_class_counts": Counter(),
        "unknown_class_counts": Counter(),
    }

    with zipfile.ZipFile(SOURCE_ZIP) as zf:
        stats["files_in_zip"] = len(zf.namelist())
        for member in zf.infolist():
            parts = member.filename.split("/")
            if len(parts) < 3:
                continue

            split_raw, folder, filename = parts[0], parts[1], parts[-1]
            if split_raw not in SPLIT_REMAP:
                continue
            split = SPLIT_REMAP[split_raw]

            if folder == "images" and not member.is_dir():
                target = OUTPUT_ROOT / "images" / split / filename
                target.write_bytes(zf.read(member))
                stats["image_files_extracted"][split] += 1
            elif folder == "labels" and filename.endswith(".txt"):
                raw_text = zf.read(member).decode("utf-8", errors="ignore").strip()
                remapped = remap_label_text(raw_text, stats)
                if not remapped:
                    stats["empty_label_files"] += 1
                target = OUTPUT_ROOT / "labels" / split / filename
                target.write_text(remapped, encoding="utf-8")
                stats["label_files_written"][split] += 1

    serializable = {
        **stats,
        "image_files_extracted": dict(stats["image_files_extracted"]),
        "label_files_written": dict(stats["label_files_written"]),
        "original_class_counts": dict(stats["original_class_counts"]),
        "remapped_class_counts": dict(stats["remapped_class_counts"]),
        "unknown_class_counts": dict(stats["unknown_class_counts"]),
    }
    REPORT_PATH.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(serializable, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
