from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_distillation_dataset import build_distillation_dataset, parse_target_classes
from scripts.yolo_utils import (
    build_model,
    dump_json,
    ensure_exists,
    latest_checkpoint_for_run,
    load_yaml,
    normalize_project_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CPU-friendly student model on a teacher-guided distillation dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/train_yolov8n_se_balanced.yaml",
        help="Base student training config YAML path.",
    )
    parser.add_argument(
        "--teacher-targets",
        required=True,
        help="Teacher targets JSON exported by export_teacher_targets.py.",
    )
    parser.add_argument(
        "--distill-output-root",
        default="datasets/final/smoke_bal_distill",
        help="Output YOLO dataset root for merged teacher-guided labels.",
    )
    parser.add_argument(
        "--distill-report",
        default="datasets/reports/distillation_dataset_report.json",
        help="Report output path for the generated distillation dataset.",
    )
    parser.add_argument(
        "--distill-splits",
        nargs="+",
        default=["train"],
        choices=("train", "val", "test"),
        help="Dataset splits that should receive teacher pseudo labels.",
    )
    parser.add_argument(
        "--target-classes",
        default="0",
        help="Comma-separated class ids to accept from the teacher. Default keeps only cigarette class 0.",
    )
    parser.add_argument(
        "--pseudo-label-conf",
        type=float,
        default=0.40,
        help="Minimum teacher confidence before a pseudo label can be merged.",
    )
    parser.add_argument(
        "--duplicate-iou",
        type=float,
        default=0.60,
        help="Skip teacher boxes that overlap too much with existing labels of the same class.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0,
        help="Optional minimum normalized box area ratio required for pseudo labels.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to materialize images into the generated distillation dataset.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build the distillation dataset and skip model training.",
    )
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
    source_data = config.get("data")
    if not source_data:
        raise ValueError(f"Training config `{args.config}` is missing `data`.")

    distill = build_distillation_dataset(
        data_path=source_data,
        teacher_targets_path=args.teacher_targets,
        output_root=args.distill_output_root,
        distill_splits=list(args.distill_splits),
        target_classes=parse_target_classes(args.target_classes),
        pseudo_label_conf=args.pseudo_label_conf,
        duplicate_iou=args.duplicate_iou,
        min_area_ratio=args.min_area_ratio,
        copy_mode=args.copy_mode,
        output_yaml_path=None,
        report_path=args.distill_report,
    )

    if args.prepare_only:
        dump_json(
            ROOT / "runs" / "reports" / "distilled_student_prepare_only.json",
            {
                "config": args.config,
                "teacher_targets": args.teacher_targets,
                "distillation_dataset": distill,
            },
        )
        print(f"Distillation dataset prepared at: {distill['output_root']}")
        print(f"Generated dataset YAML: {distill['output_yaml']}")
        return

    config["data"] = distill["output_yaml"]
    model_path = str(ensure_exists(config.pop("model"), "Model config"))
    weights_path = config.pop("weights", None)
    if weights_path and not args.resume:
        weights_path = str(ensure_exists(weights_path, "Pretrained weights"))
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
            "distillation": {
                "teacher_targets": str(ensure_exists(args.teacher_targets, "Teacher targets")),
                "generated_dataset_root": distill["output_root"],
                "generated_dataset_yaml": distill["output_yaml"],
                "distillation_report": distill["report_path"],
                "target_classes": sorted(parse_target_classes(args.target_classes)),
                "pseudo_label_conf": args.pseudo_label_conf,
                "duplicate_iou": args.duplicate_iou,
                "min_area_ratio": args.min_area_ratio,
            },
        },
    )
    print(f"Distilled student training artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
