from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_utils import resolve_output_dir

DEFAULT_EXPORT_FILES = (
    "weights/last.pt",
    "weights/best.pt",
    "results.csv",
    "args.yaml",
    "results.png",
    "train_batch0.jpg",
    "train_batch1.jpg",
    "train_batch2.jpg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle YOLO training artifacts into a single zip file.")
    parser.add_argument("--run-dir", required=True, help="Training run directory such as runs/train/yolov8n_colab_640.")
    parser.add_argument(
        "--output",
        default="runs/exports/training_bundle.zip",
        help="Zip output path. Relative paths resolve from the project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_output_dir(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Training run directory not found: {run_dir}")

    output_path = resolve_output_dir(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in DEFAULT_EXPORT_FILES:
            source = run_dir / relative
            if source.exists():
                archive.write(source, arcname=source.name)
                copied.append(relative)

        if not copied:
            raise FileNotFoundError(f"No exportable artifacts found under: {run_dir}")

        manifest = {
            "run_dir": str(run_dir),
            "exported_files": copied,
        }
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"Exported training bundle to: {output_path}")


if __name__ == "__main__":
    main()
