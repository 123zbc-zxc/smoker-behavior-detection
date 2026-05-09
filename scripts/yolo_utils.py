from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {yaml_path}")
    return data


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def require_ultralytics() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run `pip install -r requirements.txt` first."
        ) from exc
    return YOLO


def register_custom_modules() -> None:
    from ultralytics.nn import tasks
    from models.modules import ECA, SEAttention

    tasks.ECA = ECA
    tasks.SEAttention = SEAttention


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_output_dir(path: str | Path) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = project_root() / output
    return output


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root() / candidate
    return candidate


def ensure_exists(path: str | Path, label: str) -> Path:
    candidate = resolve_repo_path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    return candidate


def validate_data_config(path: str | Path) -> dict[str, Any]:
    config_path = ensure_exists(path, "Dataset config")
    config = load_yaml(config_path)
    dataset_root = config_path.parent
    root_from_config = config.get("path")
    if root_from_config:
        candidate_root = Path(root_from_config)
        if not candidate_root.is_absolute():
            candidate_root = (project_root() / candidate_root).resolve()
        dataset_root = candidate_root
    for split in ("train", "val", "test"):
        value = config.get(split)
        if not value:
            continue
        split_values = value if isinstance(value, list) else [value]
        for item in split_values:
            split_path = Path(item)
            if not split_path.is_absolute():
                split_path = (dataset_root / split_path).resolve()
            if not split_path.exists():
                raise FileNotFoundError(f"Dataset split `{split}` not found: {split_path}")
    names = config.get("names")
    if names is None:
        raise ValueError(f"Dataset config missing `names`: {config_path}")
    return config


def latest_checkpoint_for_run(project: str | Path, run_name: str) -> Path:
    run_dir = resolve_output_dir(project) / run_name
    checkpoint = run_dir / "weights" / "last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {checkpoint}. Run training once before using --resume."
        )
    return checkpoint


def collect_box_metrics(metrics: Any) -> dict[str, Any]:
    if not hasattr(metrics, "box"):
        return {}

    box = metrics.box
    names = getattr(metrics, "names", {})
    class_results = getattr(box, "class_result", None)
    per_class: dict[str, dict[str, float]] = {}

    if callable(class_results):
        for class_id, class_name in names.items():
            values = class_results(class_id)
            if not values:
                continue
            per_class[str(class_name)] = {
                "precision": float(values[0]),
                "recall": float(values[1]),
                "map50": float(values[2]),
                "map50_95": float(values[3]),
            }

    return {
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
        "per_class": per_class,
    }


def build_model(model_path: str, weights_path: str | None = None) -> Any:
    YOLO = require_ultralytics()
    register_custom_modules()

    model = YOLO(model_path)
    if weights_path:
        model = model.load(weights_path)
    return model


def normalize_project_args(config: dict[str, Any], args: Any) -> dict[str, Any]:
    merged = dict(config)
    for key in (
        "project",
        "name",
        "device",
        "imgsz",
        "batch",
        "epochs",
        "patience",
        "workers",
        "fraction",
        "exist_ok",
    ):
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value
    if "project" in merged:
        merged["project"] = str(resolve_output_dir(merged["project"]))
    return merged
