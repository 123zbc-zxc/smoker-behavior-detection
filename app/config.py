from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEMO_CONFIG_PATH = ROOT / "configs" / "web_demo.json"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "web_demo"
DEFAULT_SQLITE_URL = f"sqlite:///{(DEFAULT_OUTPUT_ROOT / 'smoker_behavior.db').as_posix()}"


def load_demo_config() -> dict[str, Any]:
    if not DEMO_CONFIG_PATH.exists():
        return {}
    return json.loads(DEMO_CONFIG_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RuntimeConfig:
    database_url: str
    output_root: Path
    uploads_root: Path
    results_root: Path
    image_upload_dir: Path
    video_upload_dir: Path
    image_result_dir: Path
    video_result_dir: Path


def build_runtime_config() -> RuntimeConfig:
    output_root = Path(os.getenv("SMOKER_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
    uploads_root = output_root / "uploads"
    results_root = output_root / "results"
    return RuntimeConfig(
        database_url=os.getenv("SMOKER_DB_URL", DEFAULT_SQLITE_URL),
        output_root=output_root,
        uploads_root=uploads_root,
        results_root=results_root,
        image_upload_dir=uploads_root / "images",
        video_upload_dir=uploads_root / "videos",
        image_result_dir=results_root / "images",
        video_result_dir=results_root / "videos",
    )


def ensure_runtime_dirs(config: RuntimeConfig) -> None:
    for path in (
        config.output_root,
        config.uploads_root,
        config.results_root,
        config.image_upload_dir,
        config.video_upload_dir,
        config.image_result_dir,
        config.video_result_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
