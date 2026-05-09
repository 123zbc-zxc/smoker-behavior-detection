# Dataset Workspace Notes

## Purpose
This directory stores all dataset assets for the smoker behavior detection project.

## Active Layout
- `raw/`: original downloaded datasets, kept as-is
- `interim/`: converted, remapped, or partially cleaned datasets
- `final/`: final YOLO datasets used by experiments and demos
- `reports/`: statistics, validation outputs, and preparation reports

`processed/` currently remains an empty historical directory and is not used by the active workflow.

## Raw Sources
- `raw/aistudio/`: main smoking dataset from AI Studio
- `raw/roboflow_smoking_drinking/`: Roboflow SmokingAndDrinking dataset
- `raw/kaggle_behavior_25k/`: Kaggle smoking/eating/sleeping/phone dataset
- `raw/kaggle_smoking_drinking_yolo/`: Kaggle smoking and drinking dataset for YOLO
- `raw/dfire/`: D-Fire smoke/fire dataset

## Interim Datasets
- `interim/aistudio_yolo/`: AI Studio converted to YOLO format
- `interim/roboflow_remap/`: Roboflow classes remapped to the 3-class schema
- `interim/cigarette_yolo/`: added cigarette-only dataset remapped into class `0`
- `interim/smoke_yolo/`: added smoke-only dataset remapped into class `2`
- `interim/smoke_legacy_yolo/`: legacy smoke/fire subset filtered to smoke-only boxes

## Final Datasets
- `final/smoking_yolo_3cls_full/`: full merged 3-class dataset preserved after cleaning and merging
- `final/smoke_bal/`: mildly balanced dataset used by current CPU-friendly experiments
- `final/smoking_yolo_3cls/` and `final/smoking_yolo_3cls_balanced/`: older historical outputs kept for reference only

## Class Mapping
Final training labels use this class mapping:
- `0 cigarette`
- `1 smoking_person`
- `2 smoke`

## Useful Commands
- Rebuild the full merged dataset without the legacy smoke subset:
  - `$env:INCLUDE_SMOKE_LEGACY='0'; python scripts/build_partial_final_dataset.py`
- Build the balanced dataset:
  - `python scripts/build_balanced_dataset.py`
- Validate the current balanced dataset:
  - `python scripts/check_dataset.py --dataset-root datasets/final/smoke_bal --report datasets/reports/balanced_dataset_check_report.json`
- Remap labels into a fresh YOLO dataset root:
  - `python scripts/remap_labels.py --source-root <src_yolo_root> --output-root <dst_yolo_root> --mapping-file <mapping.json>`
- Split a YOLO dataset into `train/val/test`:
  - `python scripts/split_dataset.py --source-root <src_yolo_root> --output-root <dst_yolo_root> --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15`
