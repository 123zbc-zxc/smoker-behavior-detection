# Repository Guidelines

## Project Structure & Module Ownership
This project centers on a YOLOv8 smoker-behavior pipeline. Keep training and inference entry
scripts in `scripts/` (`train.py`, `val.py`, `predict.py`, `export_onnx.py`, `run_web_demo.py`),
generic dataset utilities in `scripts/` (`remap_labels.py`, `split_dataset.py`, `check_dataset.py`),
analysis and comparison helpers in `scripts/` (`summarize_experiments.py`, `analyze_cigarette_detection.py`),
shared helpers in `scripts/yolo_utils.py` and `scripts/dataset_utils.py`, model definitions in
`models/` and `models/modules/`, desktop and web UI code in `app/`, and experiment/demo configs
in `configs/`. Put FastAPI routes in `app/web_demo.py`, reusable inference helpers in `app/utils/`,
database/runtime config in `app/config.py`, SQLAlchemy models/session helpers in `app/db_models.py`
and `app/db.py`, and browser assets in `app/ui/templates/` and `app/ui/static/`.

Dataset assets follow the active staged layout:
- `datasets/raw/` for original downloads
- `datasets/interim/` for converted/remapped data
- `datasets/final/` for production YOLO datasets
- `datasets/reports/` for JSON checks and preparation reports

`datasets/processed/` is not part of the current active workflow and should stay unused unless a
future change explicitly documents a new role for it.

## Active Build Expectations
Contributors should actively complete missing code instead of leaving placeholders when the
requirement is clear. If you touch data preparation, training, or inference logic, also update
the adjacent config, output path, report file, and README notes. Prefer extending existing modules
over creating one-off scripts, and keep each script single-purpose: conversion, remapping,
splitting, cleaning, validation, training, export, or prediction.

## Coding Style & Naming Conventions
Use Python with 4-space indentation, UTF-8 text, type hints, and `pathlib.Path` for paths.
Follow existing naming: files/modules in `snake_case`, constants in `UPPER_SNAKE_CASE`, functions
as short verbs such as `remap_label_text` or `collect_yolo_pairs`. Write small reusable functions,
avoid hard-coded absolute paths, and keep side effects inside `main()`.

## Build, Test, and Verification Commands
- `pip install -r requirements.txt` - install runtime dependencies.
- `$env:INCLUDE_SMOKE_LEGACY='0'; python scripts/build_partial_final_dataset.py` - rebuild the full merged dataset without the legacy fire/smoke-derived subset.
- `python scripts/convert_voc_to_yolo.py` - convert AI Studio VOC labels.
- `python scripts/prepare_roboflow_smoking.py` - unpack and remap Roboflow labels.
- `python scripts/prepare_added_datasets.py` - prepare added cigarette/smoke ZIP datasets into `datasets/interim/`.
- `python scripts/remap_labels.py --source-root <src_yolo_root> --output-root <dst_yolo_root> --mapping-file <mapping.json>` - remap YOLO class ids into a new dataset root.
- `python scripts/split_dataset.py --source-root <src_yolo_root> --output-root <dst_yolo_root> --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15` - split a flat or merged YOLO dataset into train/val/test folders.
- `python scripts/build_balanced_dataset.py` - create the mildly balanced CPU-friendly dataset in `datasets/final/smoke_bal`.
- `python scripts/train.py --config configs/train_yolov8n.yaml` - train the baseline YOLOv8n run using the paper hyperparameters.
- `python scripts/train.py --config configs/train_yolov8n_eca.yaml` - train the ECA-enhanced experiment from the custom model YAML.
- `python scripts/train.py --config configs/train_yolov8n_balanced.yaml` - train a faster baseline on the balanced dataset.
- `python scripts/train.py --config configs/train_yolov8n_eca_balanced.yaml` - train the ECA model on the balanced dataset.
- `python scripts/train.py --config configs/train_yolov8n_se_balanced.yaml` - train the SE-attention comparison model on the balanced dataset.
- `python scripts/train.py --config configs/train_yolov8n_cigarette_focus.yaml` - train the CPU-friendly cigarette-focus variant with larger input size.
- `python scripts/train.py --config configs/train_yolov8n_cigarette_focus_640.yaml` - train the more aggressive 640-input cigarette-focus experiment for higher recall.
- `python scripts/train.py --config configs/train_yolov8n_se_cigarette_focus.yaml` - train the SE-attention cigarette-focus variant for a CPU-oriented small-object comparison.
- `python scripts/train.py --config configs/train_yolov8n_colab_640_hard.yaml` - run the stronger Colab 640px hard-case fine-tuning experiment after label review; copy the current champion to `weights_imported/best.pt` first.
- `python scripts/inspect_checkpoint.py --checkpoint <last.pt>` - inspect a saved YOLO checkpoint to confirm epoch/date/train args before resuming.
- `python scripts/resume_training.py --checkpoint <last.pt> --backup-dir <backup_dir>` - resume from an explicit checkpoint such as a Google Drive backup and periodically sync `last.pt` / `best.pt`.
- `python scripts/export_training_bundle.py --run-dir runs/train/<run> --output runs/exports/<run>.zip` - bundle `last.pt`, `best.pt`, `results.csv`, and optional plots/images for download from Colab.
- `python scripts/train_distilled_student.py --config configs/train_yolov8n_se_balanced.yaml --teacher-targets runs/reports/teacher_targets.json --name yolov8n_se_distilled --exist-ok` - build a teacher-guided distillation dataset and train the student model on the merged labels.
- `python scripts/val.py --weights runs/train/<run>/weights/best.pt --split test` - validate a trained checkpoint.
- `python scripts/predict.py --weights runs/train/<run>/weights/best.pt --source assets` - run inference on local media.
- `python scripts/export_onnx.py --weights runs/train/<run>/weights/best.pt` - export a trained checkpoint to ONNX.
- `python scripts/export_teacher_targets.py --weights runs/train/<teacher>/weights/best.pt --data configs/data_smoking_balanced.yaml --split train --output runs/reports/teacher_targets.json --output-label-dir tmp/teacher_pseudo_labels` - export teacher detections that can be reused for student distillation or pseudo-label auditing.
- `python scripts/build_distillation_dataset.py --data configs/data_smoking_balanced.yaml --teacher-targets runs/reports/teacher_targets.json --output-root datasets/final/smoke_bal_distill` - merge filtered teacher pseudo labels into the source YOLO dataset to form a student distillation dataset.
- `python scripts/summarize_experiments.py --baseline runs/train/<baseline> --improved runs/train/<eca>` - compare final baseline/ECA experiment metrics.
- `python scripts/analyze_cigarette_detection.py --data configs/data_smoking_balanced.yaml --split test --weights runs/train/<run>/weights/best.pt --manifest-output runs/reports/cigarette_priority_review.txt` - analyze cigarette small-object box distribution and export a priority review manifest.
- `python scripts/build_cigarette_review_pack.py --manifest runs/reports/cigarette_priority_review.txt --data configs/data_smoking_balanced.yaml --output-dir tmp/cigarette_priority_review` - build a human-review pack with copied images, labels, previews, CSV/JSON summaries, an index HTML page, and grouped outputs for `suspicious_small_box`, `ranked_smallest_candidate`, `multi_cigarette`, `low_resolution`, `slender_cigarette`, and `other_priority`.
- `python scripts/summarize_cigarette_experiments.py --inputs <baseline_summary_json> <eca_summary_json> <se_summary_json> <distilled_summary_json> --analysis-report runs/reports/cigarette_analysis.json` - rank experiments by cigarette metrics, emit dashboard-ready comparison fields, and attach review recommendations.
- `python scripts/search_video_temporal_params.py --limit 20 --output-dir runs/video_temporal/threshold_search_20260503` - search CPU-friendly class confidence thresholds and temporal smoothing parameters on HMDB51 smoke videos.
- `python scripts/enhanced_inference.py --source <image_or_dir> --mode tta+sahi --output output/enhanced_inference_demo --json` - run optional TTA + manual SAHI slicing enhanced inference for offline image demos.
- `python scripts/eval_enhanced.py --sample 50 --modes normal sahi tta+sahi --output output/enhanced_eval/verify_sample50.json` - quantify the optional enhanced inference trade-off on a sampled test split.
- `python scripts/check_dataset.py` - validate `datasets/final/smoking_yolo_3cls_full`.
- `python scripts/check_dataset.py --dataset-root datasets/final/smoke_bal --report datasets/reports/balanced_dataset_check_report.json` - validate the balanced dataset.
- `python scripts/audit_yolo_dataset.py --dataset-root datasets/final/smoke_bal` - run a deeper YOLO label-quality audit and export suspicious-label CSV/JSON/Markdown reports plus preview images.
- `python scripts/inventory_dataset_assets.py --output-dir datasets/reports/dataset_asset_inventory_20260504` - inventory all dataset assets under `datasets/raw`, `datasets/interim`, and `datasets/final`, including image/video/archive counts, detected YOLO dataset roots, label counts, and class distributions.
- `python scripts/build_custom_video_frame_candidates.py --video-root datasets/raw/custom_smoking_videos` - extract hard-case frame candidates from local custom videos with pseudo labels and preview HTML for focused review.
- `python scripts/prepare_roboflow_cigarette_smoke_detection.py --overwrite` - normalize the Roboflow Cigarette Smoke Detection v4 ZIP into a short-path YOLO 3-class-compatible dataset; source classes 0/1 map to project class 0 `cigarette`, while polygon-only cold-breath/sunlight lines are dropped and kept as negative images.
- `python generate_figures.py` - regenerate paper figures.
- `python app/main.py` - launch the PyQt5 demo for image/video detection.
- `$env:SMOKER_DB_URL='postgresql+psycopg://<user>:<password>@127.0.0.1:5432/<db>'; python scripts/run_web_demo.py --reload` - launch the FastAPI browser admin with PostgreSQL persistence.
- `python scripts/system_smoke_test.py` - run health, dashboard, image detection, records, and async video-task checks for the web admin, including temporal-smoothing summary fields.

After changes, run the smallest relevant check first, then rerun the full dataset validation if
outputs changed.

## Code Checking Tools
Use multiple checks before submitting changes:
- `python -m py_compile <python files>` for syntax
- `python scripts/check_dataset.py` for data integrity
- targeted smoke runs of modified scripts

If you change the web demo, run `python scripts/run_web_demo.py` once and confirm `/api/health`
responds, or run `python scripts/system_smoke_test.py` to cover the main browser-admin endpoints.
If `pytest` is introduced, place tests in `tests/` with names like `test_check_dataset.py`.

## Commit & PR Guidelines
Use short imperative commits, for example `Add Roboflow label remap validation`. PRs should
include changed paths, commands executed, key outputs or screenshots, and any dataset/report files
intentionally regenerated. Do not commit raw archives, large weights, or noisy files from `runs/`
unless they are required release artifacts.

## Agent-Specific Notes
When updating this repository, proactively keep `AGENTS.md` aligned with the actual workflow. If
you add a module, command, check, or coding rule, update this guide in the same change.

## Rule
Answer the user in Chinese unless they ask otherwise.
