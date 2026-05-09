from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from scripts.yolo_utils import (
    build_model,
    dump_json,
    ensure_exists,
    latest_checkpoint_for_run,
    load_yaml,
    normalize_project_args,
    validate_data_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 baseline or lightweight attention variants.")
    parser.add_argument("--config", default="configs/train_yolov8n.yaml", help="Training config YAML path.")
    parser.add_argument("--device", help="Training device, e.g. cpu or 0.")
    parser.add_argument("--epochs", type=int, help="Override training epochs.")
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--imgsz", type=int, help="Override image size.")
    parser.add_argument("--patience", type=int, help="Override early-stop patience.")
    parser.add_argument("--workers", type=int, help="Override dataloader workers.")
    parser.add_argument("--project", help="Override Ultralytics project directory.")
    parser.add_argument("--name", help="Override Ultralytics run name.")
    parser.add_argument("--fraction", type=float, help="Train on a fraction of the dataset for debugging.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing run directory.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint for this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = normalize_project_args(load_yaml(args.config), args)
    model_path = str(ensure_exists(config.pop("model"), "Model config"))
    weights_path = config.pop("weights", None)
    if weights_path and not args.resume:
        weights_path = str(ensure_exists(weights_path, "Pretrained weights"))
    data_path = config.get("data")
    if not data_path:
        raise ValueError(f"Training config `{args.config}` is missing `data`.")
    validate_data_config(data_path)
    if args.resume:
        latest_checkpoint_for_run(config.get("project", "runs/train"), config.get("name", "exp"))

    model = build_model(model_path, None if args.resume else weights_path)
    results = model.train(resume=args.resume, **config)

    run_dir = Path(getattr(results, "save_dir", config.get("project", "runs/train")))
    dump_json(
        run_dir / "train_summary.json",
        {
            "config": args.config,
            "model": model_path,
            "weights": weights_path,
            "save_dir": str(run_dir),
            "resume": args.resume,
            "fraction": config.get("fraction", 1.0),
        },
    )
    print(f"Training artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
