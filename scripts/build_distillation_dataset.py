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

from scripts.dataset_utils import ensure_clean_yolo_output, transfer_file
from scripts.yolo_utils import dump_json, ensure_exists, load_yaml, resolve_repo_path, validate_data_config


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a teacher-guided distillation dataset by merging ground-truth labels with teacher targets."
    )
    parser.add_argument("--data", default="configs/data_smoking_balanced.yaml", help="Source dataset YAML path.")
    parser.add_argument("--teacher-targets", required=True, help="Teacher targets JSON exported by export_teacher_targets.py.")
    parser.add_argument(
        "--output-root",
        default="datasets/final/smoke_bal_distill",
        help="Output YOLO dataset root receiving merged labels.",
    )
    parser.add_argument(
        "--distill-splits",
        nargs="+",
        default=["train"],
        choices=SPLITS,
        help="Dataset splits that should receive teacher pseudo labels.",
    )
    parser.add_argument(
        "--target-classes",
        default="0",
        help="Comma-separated class ids to accept from the teacher. Default keeps only cigarette class 0.",
    )
    parser.add_argument(
        "--pseudo-label-conf",
        type=float,
        default=0.40,
        help="Minimum teacher confidence required before a pseudo label can be merged.",
    )
    parser.add_argument(
        "--duplicate-iou",
        type=float,
        default=0.60,
        help="Skip teacher boxes whose IoU with an existing label of the same class exceeds this value.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0,
        help="Optional minimum normalized box area ratio required for pseudo labels.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to materialize images into the output dataset.",
    )
    parser.add_argument(
        "--output-yaml",
        help="Optional output dataset YAML path. Defaults to <output-root>/data_distill.yaml.",
    )
    parser.add_argument(
        "--report",
        default="datasets/reports/distillation_dataset_report.json",
        help="JSON report output path.",
    )
    return parser.parse_args()


def parse_target_classes(raw_value: str) -> set[int]:
    classes: set[int] = set()
    for item in raw_value.split(","):
        value = item.strip()
        if not value:
            continue
        classes.add(int(value))
    if not classes:
        raise ValueError("At least one target class id is required.")
    return classes


def resolve_dataset_layout(data_path: Path) -> tuple[Path, dict[str, Path], dict[str, Path], dict[str, Any]]:
    config = load_yaml(data_path)
    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    image_dirs = {split: (dataset_root / config[split]).resolve() for split in SPLITS}
    label_dirs = {split: (dataset_root / "labels" / split).resolve() for split in SPLITS}
    return dataset_root, image_dirs, label_dirs, config


def parse_yolo_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        x, y, w, h = (float(value) for value in parts[1:])
    except ValueError:
        return None
    return class_id, x, y, w, h


def yolo_to_xyxy_abs(x: float, y: float, w: float, h: float, width: int, height: int) -> list[float]:
    cx = x * width
    cy = y * height
    bw = w * width
    bh = h * height
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def xyxy_abs_to_yolo(xyxy: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    if len(xyxy) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = xyxy
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    box_w = max(x2 - x1, 0.0)
    box_h = max(y2 - y1, 0.0)
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    cx = x1 + box_w / 2
    cy = y1 + box_h / 2
    return cx / width, cy / height, box_w / width, box_h / height


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def load_teacher_targets(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["image_name"]: item for item in payload.get("items", [])}


def build_output_yaml(
    *,
    output_root: Path,
    names: dict[str, Any] | list[str],
    output_yaml_path: Path,
) -> None:
    try:
        relative_root = output_root.relative_to(ROOT)
        path_value = relative_root.as_posix()
    except ValueError:
        path_value = str(output_root)
    yaml_text = "\n".join(
        [
            f"path: {path_value}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            *(
                [f"  {key}: {value}" for key, value in names.items()]
                if isinstance(names, dict)
                else [f"  {idx}: {value}" for idx, value in enumerate(names)]
            ),
            "",
        ]
    )
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    output_yaml_path.write_text(yaml_text, encoding="utf-8")


def build_distillation_dataset(
    *,
    data_path: str | Path,
    teacher_targets_path: str | Path,
    output_root: str | Path,
    distill_splits: list[str],
    target_classes: set[int],
    pseudo_label_conf: float,
    duplicate_iou: float,
    min_area_ratio: float,
    copy_mode: str,
    output_yaml_path: str | Path | None,
    report_path: str | Path,
) -> dict[str, Any]:
    data_path = ensure_exists(data_path, "Dataset config")
    teacher_targets_path = ensure_exists(teacher_targets_path, "Teacher targets")
    validate_data_config(data_path)
    output_root = resolve_repo_path(output_root)
    report_path = resolve_repo_path(report_path)
    generated_yaml_path = resolve_repo_path(output_yaml_path or (output_root / "data_distill.yaml"))

    _, image_dirs, label_dirs, config = resolve_dataset_layout(data_path)
    teacher_targets = load_teacher_targets(teacher_targets_path)
    ensure_clean_yolo_output(output_root, SPLITS)

    report: dict[str, Any] = {
        "source_data": str(data_path),
        "teacher_targets": str(teacher_targets_path),
        "output_root": str(output_root),
        "output_yaml": str(generated_yaml_path),
        "distill_splits": distill_splits,
        "target_classes": sorted(target_classes),
        "pseudo_label_conf": pseudo_label_conf,
        "duplicate_iou": duplicate_iou,
        "min_area_ratio": min_area_ratio,
        "copy_mode": copy_mode,
        "images_per_split": {},
        "ground_truth_boxes_per_split": {},
        "teacher_candidates_per_split": {},
        "teacher_appended_per_split": {},
        "images_with_pseudo_labels_per_split": {},
        "skipped_missing_teacher_targets": [],
        "sample_augmented_images": [],
    }

    for split in SPLITS:
        image_dir = image_dirs[split]
        label_dir = label_dirs[split]
        output_image_dir = output_root / "images" / split
        output_label_dir = output_root / "labels" / split
        image_paths = [
            path
            for path in sorted(image_dir.iterdir())
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        gt_boxes = 0
        teacher_candidates = 0
        teacher_appended = 0
        images_with_pseudo = 0

        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            output_image_path = output_image_dir / image_path.name
            output_label_path = output_label_dir / f"{image_path.stem}.txt"
            transfer_file(image_path, output_image_path, mode=copy_mode)

            gt_lines_raw = label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if label_path.exists() else []
            gt_lines = [line.strip() for line in gt_lines_raw if line.strip()]
            gt_boxes += len(gt_lines)
            merged_lines = list(gt_lines)
            appended_for_image = 0

            if split in distill_splits:
                teacher_item = teacher_targets.get(image_path.name)
                if teacher_item is None:
                    report["skipped_missing_teacher_targets"].append({"split": split, "image_name": image_path.name})
                else:
                    width = int(teacher_item.get("width") or 0)
                    height = int(teacher_item.get("height") or 0)
                    gt_boxes_xyxy: list[tuple[int, list[float]]] = []
                    for line in gt_lines:
                        parsed = parse_yolo_label_line(line)
                        if parsed is None:
                            continue
                        class_id, x, y, w, h = parsed
                        gt_boxes_xyxy.append((class_id, yolo_to_xyxy_abs(x, y, w, h, width, height)))

                    pseudo_boxes_xyxy: list[tuple[int, list[float]]] = []
                    for detection in teacher_item.get("detections", []):
                        class_id = int(detection.get("class_id", -1))
                        confidence = float(detection.get("confidence", 0.0))
                        xyxy = detection.get("xyxy", [])
                        if class_id not in target_classes or confidence < pseudo_label_conf or len(xyxy) != 4:
                            continue
                        normalized = xyxy_abs_to_yolo([float(value) for value in xyxy], width, height)
                        if normalized is None:
                            continue
                        x, y, w, h = normalized
                        if (w * h) < min_area_ratio:
                            continue
                        teacher_candidates += 1
                        teacher_box_abs = yolo_to_xyxy_abs(x, y, w, h, width, height)
                        duplicate = any(
                            gt_class == class_id and iou_xyxy(box_abs, teacher_box_abs) >= duplicate_iou
                            for gt_class, box_abs in [*gt_boxes_xyxy, *pseudo_boxes_xyxy]
                        )
                        if duplicate:
                            continue
                        pseudo_boxes_xyxy.append((class_id, teacher_box_abs))
                        merged_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                        teacher_appended += 1
                        appended_for_image += 1

            if appended_for_image:
                images_with_pseudo += 1
                if len(report["sample_augmented_images"]) < 50:
                    report["sample_augmented_images"].append(
                        {"split": split, "image_name": image_path.name, "pseudo_boxes_added": appended_for_image}
                    )

            output_label_path.write_text("\n".join(merged_lines), encoding="utf-8")

        report["images_per_split"][split] = len(image_paths)
        report["ground_truth_boxes_per_split"][split] = gt_boxes
        report["teacher_candidates_per_split"][split] = teacher_candidates
        report["teacher_appended_per_split"][split] = teacher_appended
        report["images_with_pseudo_labels_per_split"][split] = images_with_pseudo

    build_output_yaml(output_root=output_root, names=config["names"], output_yaml_path=generated_yaml_path)
    dump_json(report_path, report)
    return {
        "report_path": str(report_path),
        "output_root": str(output_root),
        "output_yaml": str(generated_yaml_path),
        "report": report,
    }


def main() -> None:
    args = parse_args()
    result = build_distillation_dataset(
        data_path=args.data,
        teacher_targets_path=args.teacher_targets,
        output_root=args.output_root,
        distill_splits=list(args.distill_splits),
        target_classes=parse_target_classes(args.target_classes),
        pseudo_label_conf=args.pseudo_label_conf,
        duplicate_iou=args.duplicate_iou,
        min_area_ratio=args.min_area_ratio,
        copy_mode=args.copy_mode,
        output_yaml_path=args.output_yaml,
        report_path=args.report,
    )
    print(f"Distillation dataset saved to: {result['output_root']}")
    print(f"Generated dataset YAML: {result['output_yaml']}")
    print(f"Report saved to: {result['report_path']}")


if __name__ == "__main__":
    main()
