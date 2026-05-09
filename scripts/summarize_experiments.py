from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_utils import dump_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and ECA experiment outputs.")
    parser.add_argument("--baseline", required=True, help="Path to the baseline run directory.")
    parser.add_argument("--improved", required=True, help="Path to the improved/ECA run directory.")
    parser.add_argument(
        "--output",
        default="runs/reports/experiment_comparison.json",
        help="Comparison report output path.",
    )
    return parser.parse_args()


def read_last_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows[-1]


def extract_metrics(run_dir: Path) -> dict[str, float | str]:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results.csv in {run_dir}")

    row = read_last_row(results_csv)
    wanted = {
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "fitness",
    }
    metrics: dict[str, float | str] = {"run_dir": str(run_dir)}
    for key, value in row.items():
        if key in wanted:
            metrics[key] = float(value)
    return metrics


def load_val_summary(run_dir: Path) -> dict[str, float | dict[str, float]] | None:
    candidates = [
        run_dir / "test_summary.json",
        ROOT / "runs" / "val" / run_dir.name / "test_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return payload.get("metrics")
    return None


def main() -> None:
    args = parse_args()
    baseline = extract_metrics(Path(args.baseline))
    improved = extract_metrics(Path(args.improved))
    baseline_val = load_val_summary(Path(args.baseline))
    improved_val = load_val_summary(Path(args.improved))

    report = {
        "baseline": baseline,
        "improved": improved,
        "delta": {
            key: float(improved[key]) - float(baseline[key])
            for key in baseline
            if key in improved and key != "run_dir"
        },
    }
    if baseline_val:
        report["baseline_val"] = baseline_val
    if improved_val:
        report["improved_val"] = improved_val
    if baseline_val and improved_val:
        baseline_per_class = baseline_val.get("per_class", {})
        improved_per_class = improved_val.get("per_class", {})
        report["per_class_delta"] = {
            class_name: {
                metric_name: float(improved_per_class[class_name][metric_name]) - float(metrics[metric_name])
                for metric_name in metrics
            }
            for class_name, metrics in baseline_per_class.items()
            if class_name in improved_per_class
        }
    dump_json(args.output, report)
    print(f"Comparison report saved to: {Path(args.output)}")


if __name__ == "__main__":
    main()
