from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_utils import build_model, dump_json, ensure_exists, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cigarette small-object difficulty from the YOLO dataset and optional predictions."
    )
    parser.add_argument("--data", default="configs/data_smoking_balanced.yaml", help="Dataset YAML path.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split to inspect.")
    parser.add_argument("--weights", help="Optional .pt weights. If set, also run cigarette miss/low-conf analysis.")
    parser.add_argument("--imgsz", type=int, default=416, help="Prediction image size for optional analysis.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Prediction IoU threshold.")
    parser.add_argument("--match-iou", type=float, default=0.2, help="IoU threshold used to match cigarette boxes.")
    parser.add_argument("--max-images", type=int, default=150, help="Max images for optional prediction analysis.")
    parser.add_argument(
        "--manifest-output",
        help="Optional TXT output listing unique missed/low-confidence cigarette image names for review.",
    )
    parser.add_argument(
        "--output",
        default="runs/reports/cigarette_analysis.json",
        help="JSON report output path.",
    )
    return parser.parse_args()


def resolve_split_dirs(data_path: Path, split: str) -> tuple[Path, Path]:
    config = load_yaml(data_path)
    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    image_dir = dataset_root / config[split]
    if not image_dir.exists():
        raise FileNotFoundError(f"Split image dir not found: {image_dir}")
    label_dir = dataset_root / "labels" / split
    if not label_dir.exists():
        raise FileNotFoundError(f"Split label dir not found: {label_dir}")
    return image_dir, label_dir


def parse_yolo_boxes(label_path: Path) -> list[dict[str, float | int]]:
    if not label_path.exists():
        return []
    rows: list[dict[str, float | int]] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, x, y, w, h = parts
        rows.append(
            {
                "class_id": int(cls_id),
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
            }
        )
    return rows


def normalized_to_xyxy(box: dict[str, float | int], width: int, height: int) -> list[float]:
    x = float(box["x"]) * width
    y = float(box["y"]) * height
    w = float(box["w"]) * width
    h = float(box["h"]) * height
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dataset_summary(image_dir: Path, label_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_infos: list[dict[str, Any]] = []
    cig_areas: list[float] = []
    cig_widths: list[float] = []
    cig_heights: list[float] = []
    size_buckets = Counter()
    images_with_cigarette = 0

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as img:
            width, height = img.size
        boxes = parse_yolo_boxes(label_path)
        cigarette_boxes = [box for box in boxes if int(box["class_id"]) == 0]
        if cigarette_boxes:
            images_with_cigarette += 1
        for box in cigarette_boxes:
            area_ratio = float(box["w"]) * float(box["h"])
            cig_areas.append(area_ratio)
            cig_widths.append(float(box["w"]))
            cig_heights.append(float(box["h"]))
            if area_ratio < 0.0005:
                size_buckets["tiny(<0.0005)"] += 1
            elif area_ratio < 0.001:
                size_buckets["very_small(<0.001)"] += 1
            elif area_ratio < 0.0025:
                size_buckets["small(<0.0025)"] += 1
            else:
                size_buckets["normal(>=0.0025)"] += 1
        image_infos.append(
            {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "width": width,
                "height": height,
                "cigarette_gt_count": len(cigarette_boxes),
                "cigarette_boxes": cigarette_boxes,
            }
        )

    summary = {
        "image_count": len(image_infos),
        "images_with_cigarette": images_with_cigarette,
        "cigarette_box_count": len(cig_areas),
        "cigarette_area_ratio_mean": round(statistics.fmean(cig_areas), 6) if cig_areas else 0.0,
        "cigarette_area_ratio_median": round(statistics.median(cig_areas), 6) if cig_areas else 0.0,
        "cigarette_width_ratio_mean": round(statistics.fmean(cig_widths), 6) if cig_widths else 0.0,
        "cigarette_height_ratio_mean": round(statistics.fmean(cig_heights), 6) if cig_heights else 0.0,
        "size_buckets": dict(size_buckets),
    }
    return image_infos, summary


def prediction_summary(
    image_infos: list[dict[str, Any]],
    *,
    weights: Path,
    imgsz: int,
    conf: float,
    iou: float,
    match_iou: float,
    max_images: int,
) -> dict[str, Any]:
    model = build_model(str(weights))
    missed: list[dict[str, Any]] = []
    low_conf: list[dict[str, Any]] = []
    false_positive: list[dict[str, Any]] = []
    analyzed = 0

    for info in image_infos:
        if analyzed >= max_images:
            break
        image_path = Path(info["image_path"])
        gt_boxes = [
            normalized_to_xyxy(box, int(info["width"]), int(info["height"]))
            for box in info["cigarette_boxes"]
        ]
        if not gt_boxes:
            continue
        analyzed += 1
        result = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device="cpu",
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        pred_boxes: list[list[float]] = []
        pred_conf: list[float] = []
        if boxes is not None and boxes.cls is not None:
            xyxy = boxes.xyxy.tolist() if boxes.xyxy is not None else []
            for idx, cls_id in enumerate(boxes.cls.tolist()):
                if int(cls_id) != 0 or idx >= len(xyxy):
                    continue
                pred_boxes.append([float(value) for value in xyxy[idx]])
                pred_conf.append(float(boxes.conf.tolist()[idx]))

        matched_preds: set[int] = set()
        matched_gts = 0
        for gt_idx, gt_box in enumerate(gt_boxes):
            best_idx = -1
            best_iou = 0.0
            for pred_idx, pred_box in enumerate(pred_boxes):
                candidate_iou = iou_xyxy(gt_box, pred_box)
                if candidate_iou > best_iou:
                    best_iou = candidate_iou
                    best_idx = pred_idx
            if best_idx >= 0 and best_iou >= match_iou:
                matched_gts += 1
                matched_preds.add(best_idx)
                if pred_conf[best_idx] < 0.4:
                    low_conf.append(
                        {
                            "image": image_path.name,
                            "gt_index": gt_idx,
                            "confidence": round(pred_conf[best_idx], 4),
                            "match_iou": round(best_iou, 4),
                        }
                    )
            else:
                missed.append({"image": image_path.name, "gt_index": gt_idx})

        for pred_idx, score in enumerate(pred_conf):
            if pred_idx not in matched_preds and score >= 0.45:
                false_positive.append({"image": image_path.name, "confidence": round(score, 4)})

    return {
        "weights": str(weights),
        "analyzed_images_with_cigarette": analyzed,
        "missed_cigarette_gt": len(missed),
        "low_confidence_matches": len(low_conf),
        "high_confidence_false_positive": len(false_positive),
        "missed_examples": missed[:20],
        "low_confidence_examples": low_conf[:20],
        "false_positive_examples": false_positive[:20],
        "priority_review_images": sorted(
            {
                item["image"]
                for item in [*missed, *low_conf]
            }
        )[:100],
    }


def recommendations(report: dict[str, Any]) -> list[str]:
    summary = report.get("dataset_summary", {})
    prediction = report.get("prediction_analysis", {})
    tips: list[str] = []
    tiny = int(summary.get("size_buckets", {}).get("tiny(<0.0005)", 0))
    very_small = int(summary.get("size_buckets", {}).get("very_small(<0.001)", 0))
    if tiny or very_small:
        tips.append("Try a larger imgsz such as 512 or 640 while keeping a CPU-friendly batch size.")
    if prediction:
        missed = int(prediction.get("missed_cigarette_gt", 0))
        low_conf = int(prediction.get("low_confidence_matches", 0))
        false_positive = int(prediction.get("high_confidence_false_positive", 0))
        if missed > false_positive:
            tips.append("The current failure mode is mostly missed cigarettes rather than false positives, so prioritize recall before tightening confidence thresholds.")
        if low_conf:
            tips.append("Review low-confidence matched samples to check for blur, occlusion, or undersized labels.")
        if prediction.get("priority_review_images"):
            tips.append("Use the priority_review_images manifest for label review and hard-case curation before the next training run.")
    return tips


def dump_manifest(path: Path, report: dict[str, Any]) -> None:
    prediction = report.get("prediction_analysis", {})
    image_names = prediction.get("priority_review_images", [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(image_names), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_path = ensure_exists(args.data, "Dataset config")
    image_dir, label_dir = resolve_split_dirs(data_path, args.split)
    image_infos, summary = dataset_summary(image_dir, label_dir)

    report: dict[str, Any] = {
        "data": str(data_path),
        "split": args.split,
        "dataset_summary": summary,
    }

    if args.weights:
        report["prediction_analysis"] = prediction_summary(
            image_infos,
            weights=ensure_exists(args.weights, "Analysis weights"),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            match_iou=args.match_iou,
            max_images=args.max_images,
        )
    report["recommendations"] = recommendations(report)

    dump_json(args.output, report)
    if args.manifest_output:
        dump_manifest(Path(args.manifest_output), report)
    print(f"Cigarette analysis report saved to: {args.output}")


if __name__ == "__main__":
    main()
