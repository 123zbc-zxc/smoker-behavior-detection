from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from scripts.yolo_utils import build_model, dump_json, ensure_exists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained YOLO weights to ONNX for CPU deployment.")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    parser.add_argument("--imgsz", type=int, default=416, help="Export image size.")
    parser.add_argument("--device", default="cpu", help="Export device.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph after export.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch/input shapes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(str(ensure_exists(args.weights, "Export weights")))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        device=args.device,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
    )
    exported_path = Path(exported)
    dump_json(
        exported_path.with_suffix(".export.json"),
        {
            "weights": args.weights,
            "onnx_path": str(exported_path),
            "imgsz": args.imgsz,
            "device": args.device,
            "opset": args.opset,
            "simplify": args.simplify,
            "dynamic": args.dynamic,
        },
    )
    print(f"Exported ONNX model: {exported_path}")


if __name__ == "__main__":
    main()
