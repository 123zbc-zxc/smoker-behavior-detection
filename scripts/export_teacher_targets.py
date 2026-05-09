from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_utils import build_model, dump_json, ensure_exists, load_yaml, validate_data_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export teacher detections as reusable targets for student distillation or pseudo-label review."
    )
    parser.add_argument("--weights", required=True, help="Teacher model weights (.pt).")
    parser.add_argument("--data", default="configs/data_smoking_balanced.yaml", help="Dataset YAML path.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to export.")
    parser.add_argument("--imgsz", type=int, default=512, help="Prediction image size for teacher export.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for teacher predictions.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for teacher NMS.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max image count. 0 means all images.")
    parser.add_argument(
        "--output",
        default="runs/reports/teacher_targets.json",
        help="JSON output path containing per-image teacher detections.",
    )
    parser.add_argument(
        "--output-label-dir",
        help="Optional folder that receives YOLO-format pseudo labels for high-confidence teacher detections.",
    )
    parser.add_argument(
        "--pseudo-label-conf",
        type=float,
        default=0.40,
        help="Minimum teacher confidence required when writing pseudo labels.",
    )
    return parser.parse_args()


def resolve_split_dir(data_path: Path, split: str) -> Path:
    config = load_yaml(data_path)
    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    image_dir = dataset_root / config[split]
    if not image_dir.exists():
        raise FileNotFoundError(f"Dataset split `{split}` not found: {image_dir}")
    return image_dir


def list_images(image_dir: Path, limit: int) -> list[Path]:
    image_paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    if limit > 0:
        return image_paths[:limit]
    return image_paths


def to_yolo_line(class_id: int, xyxy: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = xyxy
    box_width = max(x2 - x1, 0.0)
    box_height = max(y2 - y1, 0.0)
    cx = x1 + box_width / 2
    cy = y1 + box_height / 2
    return (
        f"{class_id} "
        f"{cx / width:.6f} "
        f"{cy / height:.6f} "
        f"{box_width / width:.6f} "
        f"{box_height / height:.6f}"
    )


def main() -> None:
    args = parse_args()
    data_path = ensure_exists(args.data, "Dataset config")
    validate_data_config(data_path)
    teacher_weights = ensure_exists(args.weights, "Teacher weights")
    image_dir = resolve_split_dir(data_path, args.split)
    image_paths = list_images(image_dir, args.limit)
    model = build_model(str(teacher_weights))

    label_dir = Path(args.output_label_dir) if args.output_label_dir else None
    if label_dir is not None:
        label_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    total_boxes = 0
    pseudo_label_count = 0

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device="cpu",
            verbose=False,
        )
        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
        shape = getattr(result, "orig_shape", None) or (0, 0)
        height, width = int(shape[0]), int(shape[1])

        entries: list[dict[str, Any]] = []
        pseudo_lines: list[str] = []
        if boxes is not None and boxes.cls is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.tolist()
            for idx, (class_id, score) in enumerate(zip(boxes.cls.tolist(), boxes.conf.tolist())):
                box_xyxy = [float(value) for value in xyxy[idx]] if idx < len(xyxy) else []
                entries.append(
                    {
                        "class_id": int(class_id),
                        "class_name": str(names.get(int(class_id), class_id)),
                        "confidence": float(score),
                        "xyxy": box_xyxy,
                    }
                )
                if (
                    label_dir is not None
                    and width > 0
                    and height > 0
                    and float(score) >= args.pseudo_label_conf
                    and len(box_xyxy) == 4
                ):
                    pseudo_lines.append(to_yolo_line(int(class_id), box_xyxy, width, height))

        if label_dir is not None:
            (label_dir / f"{image_path.stem}.txt").write_text("\n".join(pseudo_lines), encoding="utf-8")
            pseudo_label_count += len(pseudo_lines)

        total_boxes += len(entries)
        items.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "split": args.split,
                "width": width,
                "height": height,
                "num_detections": len(entries),
                "detections": entries,
            }
        )

    dump_json(
        args.output,
        {
            "weights": str(teacher_weights),
            "data": str(data_path),
            "split": args.split,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "image_count": len(items),
            "total_detections": total_boxes,
            "output_label_dir": str(label_dir) if label_dir is not None else "",
            "pseudo_label_conf": args.pseudo_label_conf,
            "pseudo_label_count": pseudo_label_count,
            "items": items,
        },
    )
    print(f"Teacher targets saved to: {args.output}")


if __name__ == "__main__":
    main()
