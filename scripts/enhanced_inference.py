from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_utils import build_model, ensure_exists

DEFAULT_WEIGHTS = ROOT / "runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt"
CLASS_NAMES = {0: "cigarette", 1: "smoking_person", 2: "smoke"}
CLASS_CONF_THRESHOLDS = {0: 0.12, 1: 0.22, 2: 0.28}
BOX_COLORS = {0: (15, 73, 217), 1: (214, 111, 29), 2: (68, 158, 47)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def shifted(self, dx: float, dy: float) -> "Detection":
        return Detection(self.class_id, self.class_name, self.confidence, self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def clipped(self, width: int, height: int) -> "Detection":
        return Detection(
            self.class_id,
            self.class_name,
            self.confidence,
            max(0.0, min(self.x1, width - 1.0)),
            max(0.0, min(self.y1, height - 1.0)),
            max(0.0, min(self.x2, width - 1.0)),
            max(0.0, min(self.y2, height - 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in (self.x1, self.y1, self.x2, self.y2)],
        }


def compute_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms_detections(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    kept: list[Detection] = []
    by_class: dict[int, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)
    for class_dets in by_class.values():
        pending = sorted(class_dets, key=lambda d: d.confidence, reverse=True)
        while pending:
            best = pending.pop(0)
            kept.append(best)
            pending = [d for d in pending if compute_iou(best, d) < iou_threshold]
    return sorted(kept, key=lambda d: d.confidence, reverse=True)


def apply_class_threshold(detections: list[Detection]) -> list[Detection]:
    return [d for d in detections if d.confidence >= CLASS_CONF_THRESHOLDS.get(d.class_id, 0.25)]


def extract_detections(result: Any, scale_back: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return []
    xyxy = boxes.xyxy.tolist() if boxes.xyxy is not None else []
    output: list[Detection] = []
    for idx, (cls_id, conf) in enumerate(zip(boxes.cls.tolist(), boxes.conf.tolist())):
        class_id = int(cls_id)
        if class_id not in CLASS_NAMES or idx >= len(xyxy):
            continue
        x1, y1, x2, y2 = [float(v) / scale_back for v in xyxy[idx]]
        output.append(Detection(class_id, CLASS_NAMES[class_id], float(conf), x1 + dx, y1 + dy, x2 + dx, y2 + dy))
    return output


def draw_detections(image: np.ndarray, detections: list[Detection], title: str = "") -> np.ndarray:
    canvas = image.copy()
    if title:
        cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 620), 34), (20, 20, 20), -1)
        cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in (det.x1, det.y1, det.x2, det.y2)]
        color = BOX_COLORS.get(det.class_id, (40, 40, 40))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        text_y = max(y1 - 8, 20)
        cv2.putText(canvas, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


class EnhancedInference:
    """Inference engine with normal, TTA, SAHI, and TTA+SAHI modes."""

    def __init__(self, weights_path: str | Path = DEFAULT_WEIGHTS, mode: str = "tta+sahi", imgsz: int = 640, conf: float = 0.10, iou: float = 0.45) -> None:
        self.weights_path = ensure_exists(weights_path, "weights")
        self.mode = mode.lower()
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.model: Any | None = None

    def load(self) -> "EnhancedInference":
        self.model = build_model(str(self.weights_path))
        return self

    def _predict_once(self, image: np.ndarray) -> list[Detection]:
        if self.model is None:
            self.load()
        results = self.model.predict(source=image, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device="cpu", verbose=False)
        detections = extract_detections(results[0]) if results else []
        return nms_detections(apply_class_threshold(detections), self.iou)

    def predict(self, image: np.ndarray) -> list[Detection]:
        if self.mode == "normal":
            return self._predict_once(image)
        if self.mode == "tta":
            return self._predict_tta(image)
        if self.mode == "sahi":
            return self._predict_sahi(image)
        if self.mode == "tta+sahi":
            return nms_detections(self._predict_tta(image) + self._predict_sahi(image), 0.5)
        raise ValueError(f"unsupported mode: {self.mode}")

    def _predict_tta(self) -> list[Detection]:
        raise RuntimeError("internal misuse")

    def _predict_tta(self, image: np.ndarray) -> list[Detection]:
        if self.model is None:
            self.load()
        height, width = image.shape[:2]
        all_dets: list[Detection] = []
        for scale in (0.8, 1.0, 1.2):
            if scale == 1.0:
                scaled = image
            else:
                scaled = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
            for flip in (False, True):
                inp = cv2.flip(scaled, 1) if flip else scaled
                results = self.model.predict(source=inp, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device="cpu", verbose=False)
                dets = extract_detections(results[0], scale_back=scale) if results else []
                if flip:
                    scaled_w = scaled.shape[1] / scale
                    dets = [Detection(d.class_id, d.class_name, d.confidence, scaled_w - d.x2, d.y1, scaled_w - d.x1, d.y2) for d in dets]
                all_dets.extend(d.clipped(width, height) for d in dets)
        return nms_detections(apply_class_threshold(all_dets), 0.5)

    def _slice_windows(self, width: int, height: int, slice_size: int = 640, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
        if width <= slice_size and height <= slice_size:
            return [(0, 0, width, height)]
        step = max(1, int(slice_size * (1 - overlap)))
        xs = list(range(0, max(width - slice_size, 0) + 1, step))
        ys = list(range(0, max(height - slice_size, 0) + 1, step))
        if xs[-1] != max(width - slice_size, 0):
            xs.append(max(width - slice_size, 0))
        if ys[-1] != max(height - slice_size, 0):
            ys.append(max(height - slice_size, 0))
        return [(x, y, min(x + slice_size, width), min(y + slice_size, height)) for y in ys for x in xs]

    def _predict_sahi(self, image: np.ndarray) -> list[Detection]:
        if self.model is None:
            self.load()
        height, width = image.shape[:2]
        all_dets = self._predict_once(image)
        for x1, y1, x2, y2 in self._slice_windows(width, height):
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            results = self.model.predict(source=crop, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device="cpu", verbose=False)
            dets = extract_detections(results[0], dx=x1, dy=y1) if results else []
            all_dets.extend(d.clipped(width, height) for d in dets)
        return nms_detections(apply_class_threshold(all_dets), 0.5)


def collect_images(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    raise FileNotFoundError(f"invalid image source: {source}")


def compare_modes(image: np.ndarray, weights_path: str | Path = DEFAULT_WEIGHTS, imgsz: int = 640, conf: float = 0.10, iou: float = 0.45) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for mode in ("normal", "tta", "sahi", "tta+sahi"):
        engine = EnhancedInference(weights_path, mode=mode, imgsz=imgsz, conf=conf, iou=iou).load()
        start = time.time()
        detections = engine.predict(image)
        elapsed = time.time() - start
        counts: dict[str, int] = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        results[mode] = {
            "total_detections": len(detections),
            "class_counts": counts,
            "elapsed_seconds": round(elapsed, 3),
            "detections": detections,
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced inference with TTA and manual SAHI slicing.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--mode", default="tta+sahi", choices=["normal", "tta", "sahi", "tta+sahi", "compare"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", default="output/enhanced_inference")
    parser.add_argument("--save-compare", action="store_true")
    parser.add_argument("--json", action="store_true", help="save detection json files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = collect_images(Path(args.source))
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = ensure_exists(args.weights, "weights")

    if args.mode == "compare":
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"[skip] failed to read {path}")
                continue
            results = compare_modes(image, weights, args.imgsz, args.conf, args.iou)
            print(f"\n{path.name}")
            print("mode         total  cigarette  person  smoke  seconds")
            for mode, info in results.items():
                cc = info["class_counts"]
                print(f"{mode:<12} {info['total_detections']:<6} {cc.get('cigarette',0):<10} {cc.get('smoking_person',0):<7} {cc.get('smoke',0):<6} {info['elapsed_seconds']}")
            if args.save_compare:
                h, w = image.shape[:2]
                grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
                for mode, (oy, ox) in zip(("normal", "tta", "sahi", "tta+sahi"), ((0, 0), (0, w), (h, 0), (h, w))):
                    dets = results[mode]["detections"]
                    grid[oy:oy + h, ox:ox + w] = draw_detections(image, dets, f"{mode} n={len(dets)}")
                out = output_dir / f"compare_{path.stem}.jpg"
                cv2.imwrite(str(out), grid)
                print(f"saved {out}")
        return

    engine = EnhancedInference(weights, mode=args.mode, imgsz=args.imgsz, conf=args.conf, iou=args.iou).load()
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[skip] failed to read {path}")
            continue
        start = time.time()
        detections = engine.predict(image)
        elapsed = time.time() - start
        vis = draw_detections(image, detections, f"{args.mode} n={len(detections)}")
        out_image = output_dir / f"{args.mode}_{path.name}"
        cv2.imwrite(str(out_image), vis)
        if args.json:
            (output_dir / f"{args.mode}_{path.stem}.json").write_text(json.dumps([d.to_dict() for d in detections], ensure_ascii=False, indent=2), encoding="utf-8")
        counts: dict[str, int] = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        print(f"{path.name}: {len(detections)} dets {counts} {elapsed:.2f}s -> {out_image}")


if __name__ == "__main__":
    main()
