from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.enhanced_inference import Detection, EnhancedInference, compute_iou
from scripts.yolo_utils import ensure_exists, load_yaml

DEFAULT_WEIGHTS = ROOT / "runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt"
DEFAULT_DATA = ROOT / "configs/data_smoking_balanced.yaml"
CLASS_NAMES = {0: "cigarette", 1: "smoking_person", 2: "smoke"}


def load_yolo_labels(path: Path, width: int, height: int) -> list[Detection]:
    if not path.exists():
        return []
    items: list[Detection] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        items.append(Detection(class_id, CLASS_NAMES.get(class_id, str(class_id)), 1.0, x1, y1, x2, y2))
    return items


def evaluate_image(preds: list[Detection], gts: list[Detection], iou_threshold: float) -> dict[int, tuple[int, int, int]]:
    stats: dict[int, list[int]] = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}  # tp, fp, fn
    gt_used: set[int] = set()
    for pred in sorted(preds, key=lambda d: d.confidence, reverse=True):
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gts):
            if idx in gt_used or gt.class_id != pred.class_id:
                continue
            score = compute_iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            stats[pred.class_id][0] += 1
            gt_used.add(best_idx)
        else:
            stats[pred.class_id][1] += 1
    for idx, gt in enumerate(gts):
        if idx not in gt_used:
            stats[gt.class_id][2] += 1
    return {k: tuple(v) for k, v in stats.items()}


def summarize(stats_by_class: dict[int, list[int]]) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0
    for class_id, name in CLASS_NAMES.items():
        tp, fp, fn = stats_by_class[class_id]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"overall": {"tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": precision, "recall": recall, "f1": f1}, "per_class": per_class}


def collect_test_images(data_yaml: Path, split: str) -> tuple[Path, list[Path]]:
    cfg = load_yaml(data_yaml)
    dataset_root = Path(cfg.get("path", ""))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    image_dir = dataset_root / "images" / split
    ensure_exists(image_dir, f"{split} image dir")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    return dataset_root, images


def evaluate_modes(images: list[Path], dataset_root: Path, modes: list[str], weights: Path, imgsz: int, conf: float, iou: float, match_iou: float) -> dict[str, Any]:
    engines = {mode: EnhancedInference(weights, mode=mode, imgsz=imgsz, conf=conf, iou=iou).load() for mode in modes}
    totals = {mode: {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]} for mode in modes}
    times = {mode: [] for mode in modes}
    for idx, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = dataset_root / "labels" / "test" / f"{image_path.stem}.txt"
        gts = load_yolo_labels(label_path, width, height)
        print(f"[{idx}/{len(images)}] {image_path.name} gt={len(gts)}")
        for mode, engine in engines.items():
            start = time.time()
            preds = engine.predict(image)
            times[mode].append(time.time() - start)
            per_image = evaluate_image(preds, gts, match_iou)
            for class_id, values in per_image.items():
                for i, value in enumerate(values):
                    totals[mode][class_id][i] += value
    results: dict[str, Any] = {}
    for mode in modes:
        metrics = summarize(totals[mode])
        metrics["avg_seconds_per_image"] = sum(times[mode]) / len(times[mode]) if times[mode] else 0.0
        metrics["total_seconds"] = sum(times[mode])
        results[mode] = metrics
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate normal/TTA/SAHI enhanced inference on a sample of the test split.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--split", default="test")
    parser.add_argument("--modes", nargs="+", default=["normal", "sahi", "tta+sahi"])
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--output", default="output/enhanced_eval/enhanced_eval_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = ensure_exists(args.weights, "weights")
    data_yaml = ensure_exists(args.data, "data yaml")
    dataset_root, images = collect_test_images(data_yaml, args.split)
    if args.sample > 0 and args.sample < len(images):
        random.seed(args.seed)
        images = sorted(random.sample(images, args.sample))
    print(f"Evaluating {len(images)} images, modes={args.modes}, weights={weights}")
    results = evaluate_modes(images, dataset_root, args.modes, weights, args.imgsz, args.conf, args.iou, args.match_iou)
    print("\nmode        P       R       F1      cig_R   person_R smoke_R sec/img")
    for mode in args.modes:
        r = results[mode]
        o = r["overall"]
        pc = r["per_class"]
        print(f"{mode:<11} {o['precision']:.4f} {o['recall']:.4f} {o['f1']:.4f} {pc['cigarette']['recall']:.4f} {pc['smoking_person']['recall']:.4f} {pc['smoke']['recall']:.4f} {r['avg_seconds_per_image']:.2f}")
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"weights": str(weights), "data": str(data_yaml), "split": args.split, "sample": len(images), "modes": args.modes, "results": results}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
