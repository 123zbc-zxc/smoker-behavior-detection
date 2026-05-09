from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_utils import dump_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize validation summaries with a cigarette-first ranking."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more validation summary JSON files produced by scripts/val.py.",
    )
    parser.add_argument(
        "--analysis-report",
        help="Optional cigarette analysis JSON used to append recommendations and review counts.",
    )
    parser.add_argument(
        "--output",
        default="runs/reports/cigarette_experiment_summary.json",
        help="Summary output path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_name(path: Path, payload: dict[str, Any]) -> str:
    weights = payload.get("weights", "")
    if weights:
        return Path(weights).parents[1].name if len(Path(weights).parents) >= 2 else Path(weights).stem
    return path.parent.name


def pick_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    per_class = metrics.get("per_class", {})
    cigarette = per_class.get("cigarette", {})
    return {
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "map50": metrics.get("map50"),
        "map50_95": metrics.get("map50_95"),
        "cigarette_precision": cigarette.get("precision"),
        "cigarette_recall": cigarette.get("recall"),
        "cigarette_map50": cigarette.get("map50"),
        "cigarette_map50_95": cigarette.get("map50_95"),
    }


def format_metric(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    args = parse_args()
    entries: list[dict[str, Any]] = []
    for item in args.inputs:
        path = Path(item)
        payload = load_json(path)
        entry = {
            "name": candidate_name(path, payload),
            "summary_path": str(path),
            "weights": payload.get("weights"),
            "split": payload.get("split"),
        }
        entry.update(pick_metrics(payload))
        entries.append(entry)

    ranked = sorted(
        entries,
        key=lambda item: (
            item.get("cigarette_map50") if item.get("cigarette_map50") is not None else -1.0,
            item.get("cigarette_recall") if item.get("cigarette_recall") is not None else -1.0,
            item.get("map50") if item.get("map50") is not None else -1.0,
        ),
        reverse=True,
    )

    report: dict[str, Any] = {
        "rank_basis": ["cigarette_map50", "cigarette_recall", "map50"],
        "experiments": ranked,
        "best": ranked[0] if ranked else None,
        "dashboard_experiments": {
            item["name"]: {
                "precision": format_metric(item.get("precision")),
                "recall": format_metric(item.get("recall")),
                "map50": format_metric(item.get("map50")),
                "map50_95": format_metric(item.get("map50_95")),
                "cigarette_precision": format_metric(item.get("cigarette_precision")),
                "cigarette_recall": format_metric(item.get("cigarette_recall")),
                "cigarette_map50": format_metric(item.get("cigarette_map50")),
                "cigarette_map50_95": format_metric(item.get("cigarette_map50_95")),
                "weights_name": Path(item.get("weights") or "").name or "-",
                "summary_path": item.get("summary_path"),
            }
            for item in ranked
        },
    }
    if args.analysis_report:
        analysis = load_json(Path(args.analysis_report))
        report["analysis_report"] = {
            "path": args.analysis_report,
            "priority_review_image_count": len(
                analysis.get("prediction_analysis", {}).get("priority_review_images", [])
            ),
            "recommendations": analysis.get("recommendations", []),
        }

    dump_json(args.output, report)
    print(f"Cigarette experiment summary saved to: {args.output}")


if __name__ == "__main__":
    main()
