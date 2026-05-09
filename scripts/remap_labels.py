from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_utils import (
    FLAT_GROUP,
    collect_yolo_pairs,
    ensure_clean_yolo_output,
    group_output_dir,
    transfer_file,
)
from scripts.yolo_utils import dump_json


DEFAULT_REPORT = "datasets/reports/remap_labels_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap YOLO label classes into a new dataset root.")
    parser.add_argument("--source-root", required=True, help="Source YOLO dataset root containing images/ and labels/.")
    parser.add_argument("--output-root", required=True, help="Output YOLO dataset root.")
    parser.add_argument(
        "--mapping-json",
        help='Inline JSON object, e.g. "{\\"0\\": 1, \\"1\\": 1, \\"2\\": 2}".',
    )
    parser.add_argument("--mapping-file", help="Path to a JSON file containing class remapping.")
    parser.add_argument(
        "--unknown-policy",
        choices=("drop", "keep", "error"),
        default="drop",
        help="How to handle classes that are missing from the mapping.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to materialize images into the output dataset.",
    )
    parser.add_argument("--report", default=DEFAULT_REPORT, help="JSON report output path.")
    return parser.parse_args()


def load_mapping(args: argparse.Namespace) -> dict[str, str]:
    provided = [bool(args.mapping_json), bool(args.mapping_file)]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of `--mapping-json` or `--mapping-file`.")

    if args.mapping_json:
        raw_mapping: Any = json.loads(args.mapping_json)
    else:
        raw_mapping = json.loads(Path(args.mapping_file).read_text(encoding="utf-8-sig"))

    if not isinstance(raw_mapping, dict):
        raise ValueError("Class mapping must be a JSON object.")

    mapping: dict[str, str] = {}
    for source_cls, target_cls in raw_mapping.items():
        mapping[str(source_cls)] = str(target_cls)
    return mapping


def remap_label_text(
    text: str,
    mapping: dict[str, str],
    unknown_policy: str,
    stats: dict[str, Any],
    pair_name: str,
) -> str:
    output_lines: list[str] = []
    stripped = text.strip()
    if not stripped:
        stats["empty_label_files"] += 1
        return ""

    for line_no, raw_line in enumerate(stripped.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            stats["invalid_label_lines"].append(
                {"file": pair_name, "line_no": line_no, "line": raw_line}
            )
            continue

        original_cls = parts[0]
        stats["original_class_counts"][original_cls] += 1
        mapped_cls = mapping.get(original_cls)
        if mapped_cls is None:
            stats["unknown_class_counts"][original_cls] += 1
            if unknown_policy == "error":
                raise ValueError(
                    f"Unknown class `{original_cls}` in `{pair_name}` line {line_no}."
                )
            if unknown_policy == "drop":
                continue
            mapped_cls = original_cls

        parts[0] = mapped_cls
        stats["remapped_class_counts"][mapped_cls] += 1
        output_lines.append(" ".join(parts))

    if not output_lines:
        stats["empty_label_files"] += 1
    return "\n".join(output_lines)


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args)
    pairs, issues = collect_yolo_pairs(args.source_root)
    if issues["missing_images"] or issues["missing_labels"]:
        raise ValueError(
            "Source dataset has unmatched image/label files. "
            "Resolve them before remapping labels."
        )

    groups = sorted({pair.group for pair in pairs}) or [FLAT_GROUP]
    output_root = ensure_clean_yolo_output(args.output_root, groups)

    stats: dict[str, Any] = {
        "source_root": str(Path(args.source_root)),
        "output_root": str(output_root),
        "report_path": str(Path(args.report)),
        "copy_mode": args.copy_mode,
        "unknown_policy": args.unknown_policy,
        "mapping": mapping,
        "groups": [group or "flat" for group in groups],
        "total_pairs": len(pairs),
        "images_written": Counter(),
        "labels_written": Counter(),
        "empty_label_files": 0,
        "original_class_counts": Counter(),
        "remapped_class_counts": Counter(),
        "unknown_class_counts": Counter(),
        "invalid_label_lines": [],
        "source_issues": issues,
    }

    for pair in pairs:
        output_group = pair.group
        image_dir = group_output_dir(output_root / "images", output_group)
        label_dir = group_output_dir(output_root / "labels", output_group)
        output_image = image_dir / f"{pair.stem}{pair.image_path.suffix.lower()}"
        output_label = label_dir / f"{pair.stem}.txt"

        transfer_file(pair.image_path, output_image, mode=args.copy_mode)
        remapped_text = remap_label_text(
            pair.label_path.read_text(encoding="utf-8", errors="ignore"),
            mapping=mapping,
            unknown_policy=args.unknown_policy,
            stats=stats,
            pair_name=str(pair.label_path),
        )
        output_label.write_text(remapped_text, encoding="utf-8")
        stats["images_written"][output_group or "flat"] += 1
        stats["labels_written"][output_group or "flat"] += 1

    serializable = {
        **stats,
        "images_written": dict(stats["images_written"]),
        "labels_written": dict(stats["labels_written"]),
        "original_class_counts": dict(stats["original_class_counts"]),
        "remapped_class_counts": dict(stats["remapped_class_counts"]),
        "unknown_class_counts": dict(stats["unknown_class_counts"]),
    }
    dump_json(args.report, serializable)
    print(json.dumps(serializable, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {Path(args.report)}")


if __name__ == "__main__":
    main()
