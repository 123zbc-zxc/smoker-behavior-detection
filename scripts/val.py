from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from scripts.yolo_utils import (
    build_model,
    collect_box_metrics,
    dump_json,
    ensure_exists,
    resolve_output_dir,
    validate_data_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained YOLO model on the smoking dataset.")
    parser.add_argument("--weights", required=True, help="Path to .pt model weights.")
    parser.add_argument("--data", default="configs/data_smoking.yaml", help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=416, help="Validation image size.")
    parser.add_argument("--device", help="Validation device, e.g. cpu or 0.")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="Dataset split to evaluate.")
    parser.add_argument("--project", default="runs/val", help="Validation output directory.")
    parser.add_argument("--name", default="smoking_eval", help="Validation run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_data_config(args.data)
    model = build_model(str(ensure_exists(args.weights, "Validation weights")))
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        split=args.split,
        project=str(resolve_output_dir(args.project)),
        name=args.name,
    )

    save_dir = Path(getattr(metrics, "save_dir", "runs/val"))
    dump_json(
        save_dir / f"{args.split}_summary.json",
        {
            "weights": args.weights,
            "data": args.data,
            "split": args.split,
            "save_dir": str(save_dir),
            "metrics": collect_box_metrics(metrics),
        },
    )
    print(f"Validation artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
