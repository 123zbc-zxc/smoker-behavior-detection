from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from scripts.yolo_utils import build_model, dump_json, ensure_exists, resolve_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO prediction on images, folders, videos, or streams.")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    parser.add_argument("--source", required=True, help="Input image, folder, video, or stream source.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    parser.add_argument("--device", help="Inference device, e.g. cpu or 0.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--project", default="runs/predict", help="Output project directory.")
    parser.add_argument("--name", default="smoking_demo", help="Output run name.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format prediction labels.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidences with labels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(str(ensure_exists(args.weights, "Prediction weights")))
    source = args.source
    if not (
        source.isdigit()
        or "://" in source
        or source.startswith("rtsp:")
        or source.startswith("rtmp:")
    ):
        ensure_exists(source, "Prediction source")
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(resolve_output_dir(args.project)),
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )

    save_dir = Path(results[0].save_dir) if results else Path(args.project) / args.name
    dump_json(
        save_dir / "predict_summary.json",
        {
            "weights": args.weights,
            "source": args.source,
            "save_dir": str(save_dir),
            "conf": args.conf,
            "iou": args.iou,
        },
    )
    print(f"Prediction artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
