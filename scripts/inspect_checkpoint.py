from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.yolo_utils import dump_json, ensure_exists


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect YOLO checkpoint metadata before resuming training.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint such as last.pt or best.pt.")
    parser.add_argument("--output", help="Optional JSON output path for the parsed checkpoint metadata.")
    return parser.parse_args()


def summarize_checkpoint(checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload is not a dict: {checkpoint_path}")

    train_args = payload.get("train_args") or {}
    summary = {
        "checkpoint": str(checkpoint_path),
        "epoch": int(payload.get("epoch", -1)),
        "date": payload.get("date", ""),
        "best_fitness": optional_float(payload.get("best_fitness")),
        "version": payload.get("version", ""),
        "train_args": {
            key: train_args.get(key)
            for key in (
                "data",
                "epochs",
                "imgsz",
                "batch",
                "device",
                "workers",
                "cache",
                "project",
                "name",
                "optimizer",
            )
            if key in train_args
        },
    }
    return summary


def main() -> None:
    args = parse_args()
    checkpoint_path = ensure_exists(args.checkpoint, "Checkpoint")
    summary = summarize_checkpoint(checkpoint_path)
    if args.output:
        dump_json(args.output, summary)
    print(summary)


if __name__ == "__main__":
    main()
