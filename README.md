# Smoker Behavior Detection Based on Deep Learning

## Project Goal
This repository packages the graduation-project workflow for smoker behavior detection:
multi-source dataset preparation, YOLOv8n baseline training, YOLOv8n+ECA comparison,
desktop inference, ONNX export, cigarette-risk analysis, and a FastAPI management console
for persisted image/video detection.

## Current State
- The core engineering chain is already available: dataset conversion, remapping, merging,
  balancing, validation, training, evaluation, export, desktop demo, and web admin.
- The active experiment path currently uses the balanced dataset `datasets/final/smoke_bal`
  and the hard fine-tuned Colab 640px checkpoint
  `runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt`.
- The FastAPI demo now stores image detection records, video tasks, and model/settings metadata
  through SQLAlchemy-backed storage. Set `SMOKER_DB_URL` to your PostgreSQL URL for normal runs.
- The full merged dataset `datasets/final/smoking_yolo_3cls_full` is kept as the main
  archive of the cleaned 3-class training corpus.
- `datasets/processed/` is currently a historical empty directory and is not part of the
  active pipeline; use `raw/`, `interim/`, `final/`, and `reports/` instead.

## Repository Layout
- `configs/` - dataset, training, and demo configuration files
- `datasets/` - raw, interim, final, and report artifacts
- `scripts/` - data preparation, training, validation, prediction, export, reporting, and utilities
- `models/` - custom YOLO model definitions and modules
- `app/` - PyQt5 desktop app, FastAPI web admin, SQLAlchemy models, and browser assets
- `runs/` - training, validation, prediction, and report outputs
- `docs/` - thesis draft materials and checklists
- `figures/` - thesis figures and architecture diagrams

## Main Commands

### Dataset Preparation
- `python scripts/convert_voc_to_yolo.py`
- `python scripts/prepare_roboflow_smoking.py`
- `python scripts/prepare_added_datasets.py`
- `$env:INCLUDE_SMOKE_LEGACY='0'; python scripts/build_partial_final_dataset.py`
- `python scripts/build_balanced_dataset.py`
- `python scripts/check_dataset.py`
- `python scripts/check_dataset.py --dataset-root datasets/final/smoke_bal --report datasets/reports/balanced_dataset_check_report.json`
- `python scripts/audit_yolo_dataset.py --dataset-root datasets/final/smoke_bal` - export a deeper suspicious-label audit with CSV/JSON/Markdown reports and preview images
- `python scripts/inventory_dataset_assets.py --output-dir datasets/reports/dataset_asset_inventory_20260504` - inventory all dataset assets under `datasets/raw`, `datasets/interim`, and `datasets/final`
- `python scripts/build_custom_video_frame_candidates.py --video-root datasets/raw/custom_smoking_videos` - extract custom-video hard-case frame candidates with pseudo labels and preview pages
- `python scripts/prepare_roboflow_cigarette_smoke_detection.py --overwrite` - normalize the Roboflow Cigarette Smoke Detection v4 ZIP into a short-path YOLO 3-class-compatible external dataset

### Generic Dataset Utilities
- `python scripts/remap_labels.py --source-root <src_yolo_root> --output-root <dst_yolo_root> --mapping-file <mapping.json>`
- `python scripts/split_dataset.py --source-root <src_yolo_root> --output-root <dst_yolo_root> --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15`

### Training
- `python scripts/train.py --config configs/train_yolov8n.yaml`
- `python scripts/train.py --config configs/train_yolov8n_eca.yaml`
- `python scripts/train.py --config configs/train_yolov8n_balanced.yaml`
- `python scripts/train.py --config configs/train_yolov8n_eca_balanced.yaml`
- `python scripts/train.py --config configs/train_yolov8n_se_balanced.yaml`
- `python scripts/train.py --config configs/train_yolov8n_cigarette_focus.yaml`
- `python scripts/train.py --config configs/train_yolov8n_cigarette_focus_640.yaml`
- `python scripts/train.py --config configs/train_yolov8n_se_cigarette_focus.yaml`
- `python scripts/train.py --config configs/train_yolov8n_colab_640_plus_external.yaml` - continue fine-tuning with the balanced dataset plus the normalized Roboflow cigarette-focused external dataset
- `python scripts/train_distilled_student.py --config configs/train_yolov8n_se_balanced.yaml --teacher-targets runs/reports/teacher_targets.json --name yolov8n_se_distilled --exist-ok` - build a teacher-guided distillation dataset and train the student on it
- `python scripts/train.py --config configs/train_yolov8n_balanced.yaml --epochs 30 --name yolov8n_balanced_30 --exist-ok`
- `python scripts/train.py --config configs/train_yolov8n_eca_balanced.yaml --epochs 30 --name yolov8n_eca_balanced_304 --exist-ok`
- `python scripts/inspect_checkpoint.py --checkpoint <last.pt>` - inspect checkpoint epoch/date/train args before resuming
- `python scripts/resume_training.py --checkpoint <last.pt> --backup-dir <backup_dir>` - resume from an explicit checkpoint and optionally sync `last.pt` / `best.pt` backups
- `python scripts/export_training_bundle.py --run-dir runs/train/<run> --output runs/exports/<run>.zip` - bundle `last.pt`, `best.pt`, `results.csv`, and related artifacts for download

### Evaluation and Export
- `python scripts/val.py --weights runs/train/yolov8n_balanced_30/weights/best.pt --split test --name smoking_eval`
- `python scripts/predict.py --weights runs/train/yolov8n_balanced_30/weights/best.pt --source assets`
- `python scripts/export_onnx.py --weights runs/train/yolov8n_balanced_30/weights/best.pt`
- `python scripts/export_teacher_targets.py --weights runs/train/<teacher>/weights/best.pt --data configs/data_smoking_balanced.yaml --split train --output runs/reports/teacher_targets.json --output-label-dir tmp/teacher_pseudo_labels` - export teacher detections for student distillation or pseudo-label review
- `python scripts/build_distillation_dataset.py --data configs/data_smoking_balanced.yaml --teacher-targets runs/reports/teacher_targets.json --output-root datasets/final/smoke_bal_distill` - merge ground-truth labels with filtered teacher pseudo labels to create a distillation dataset
- `python scripts/summarize_experiments.py --baseline runs/train/yolov8n_balanced_30 --improved runs/train/yolov8n_eca_balanced_304`
- `python scripts/analyze_cigarette_detection.py --data configs/data_smoking_balanced.yaml --split test --weights runs/train/yolov8n_balanced_30/weights/best.pt --manifest-output runs/reports/cigarette_priority_review.txt`
- `python scripts/build_cigarette_review_pack.py --manifest runs/reports/cigarette_priority_review.txt --data configs/data_smoking_balanced.yaml --output-dir tmp/cigarette_priority_review` - build a review pack and auto-split `suspicious_small_box`, `ranked_smallest_candidate`, `multi_cigarette`, `low_resolution`, `slender_cigarette`, and `other_priority` groups for faster manual relabel review
- `python scripts/summarize_cigarette_experiments.py --inputs runs/val/baseline_eval/test_summary.json runs/val/eca_eval/test_summary.json runs/val/se_eval/test_summary.json runs/val/distilled_eval/test_summary.json --analysis-report runs/reports/cigarette_analysis.json` - build a cigarette-first comparison report for baseline / ECA / SE / distilled runs; the Web Dashboard will auto-read `runs/reports/cigarette_experiment_summary.json` if present
- `python scripts/search_video_temporal_params.py --limit 20 --output-dir runs/video_temporal/threshold_search_20260503` - search CPU-friendly class confidence thresholds and temporal smoothing parameters on HMDB51 smoke videos
- `python scripts/enhanced_inference.py --source <image_or_dir> --mode tta+sahi --output output/enhanced_inference_demo --json` - run optional TTA + manual SAHI slicing enhanced inference for offline image demos
- `python scripts/eval_enhanced.py --sample 50 --modes normal sahi tta+sahi --output output/enhanced_eval/verify_sample50.json` - evaluate the optional enhanced inference recall/precision/speed trade-off

### Demo
- `python app/main.py`
- `$env:SMOKER_DB_URL='postgresql+psycopg://<user>:<password>@127.0.0.1:5432/<database>'; python scripts/run_web_demo.py --reload`
- `python scripts/run_web_demo.py --reload`
- `python scripts/system_smoke_test.py` - includes video temporal-smoothing summary checks

Open the web admin in a browser at `http://127.0.0.1:8000`.

## Active Assets and Defaults
- Full merged dataset: `datasets/final/smoking_yolo_3cls_full`
- Balanced experiment dataset: `datasets/final/smoke_bal`
- Baseline config: `configs/train_yolov8n_balanced.yaml`
- ECA comparison config: `configs/train_yolov8n_eca_balanced.yaml`
- SE comparison config: `configs/train_yolov8n_se_balanced.yaml`
- Cigarette-focus config: `configs/train_yolov8n_cigarette_focus.yaml`
- Cigarette-focus 640 config: `configs/train_yolov8n_cigarette_focus_640.yaml`
- Colab dataset config: `configs/data_smoking_balanced_colab.yaml`
- Colab GPU config: `configs/train_yolov8n_colab_gpu.yaml`
- Colab hard-case 640 config: `configs/train_yolov8n_colab_640_hard.yaml`
- External Roboflow cigarette config: `configs/data_smoking_balanced_plus_rfcsd4.yaml`
- Colab external-data fine-tune config: `configs/train_yolov8n_colab_640_plus_external.yaml`
- SE cigarette-focus config: `configs/train_yolov8n_se_cigarette_focus.yaml`
- Web demo config: `configs/web_demo.json`
- Runtime DB env var: `SMOKER_DB_URL`
- Web demo artifact root: `output/web_demo`
- Web demo default weights search order:
  1. `runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt`
  2. `runs/imported/smoker_weights_20260429/best.pt`
  3. `runs/imported/smoker_weights_20260429/last.pt`
  4. `runs/train/yolov8n_balanced_512/weights/best.pt`
  5. `runs/train/yolov8n_balanced_30/weights/best.pt`
  6. `yolov8n.pt`

## Installed Experiment Summary
Current balanced-dataset experiment metrics used in the thesis:

- Hard fine-tune 640 best: Precision `0.541`, Recall `0.690`, mAP@0.5 `0.560`, mAP@0.5:0.95 `0.356`;
  cigarette mAP@0.5 `0.497`, cigarette recall `0.747`
- Old Colab 640 best: Precision `0.539`, Recall `0.686`, mAP@0.5 `0.561`, mAP@0.5:0.95 `0.360`;
  cigarette mAP@0.5 `0.492`, cigarette recall `0.721`
- baseline: Precision `0.526`, Recall `0.625`, mAP@0.5 `0.520`, mAP@0.5:0.95 `0.323`
- ECA: Precision `0.513`, Recall `0.572`, mAP@0.5 `0.494`, mAP@0.5:0.95 `0.299`

At the current stage, the hard fine-tuned checkpoint is the Web default because it improves
cigarette recall and full-video temporal stability. The old Colab checkpoint remains a comparison
baseline because its mAP@0.5:0.95 is slightly higher.
