# Project Structure Notes

This file tracks the current engineering structure of the smoker behavior detection project.

## Active code paths
- `scripts/`: dataset preparation, remapping, splitting, validation, training, export, report comparison, and cigarette-risk analysis
- `models/`: custom YOLO model definitions, including the ECA-enhanced YAML and module implementation
- `app/`: PyQt5 desktop demo, FastAPI web admin, SQLAlchemy runtime/config helpers, and browser assets
- `configs/`: dataset configs, training configs, and web demo metadata

## Active data paths
- `datasets/raw/`: downloaded source datasets
- `datasets/interim/`: converted and remapped intermediate datasets
- `datasets/final/smoking_yolo_3cls_full/`: full merged training dataset
- `datasets/final/smoke_bal/`: mildly balanced experiment dataset
- `datasets/reports/`: dataset build and validation reports

## Supporting assets
- `runs/`: experiment outputs and checkpoints
- `output/web_demo/`: persisted web-admin uploads, results, and local fallback database artifacts
- `docs/`: thesis draft materials and checklists
- `figures/`: thesis figures and generated visual assets

## Notes
- `datasets/processed/` currently has no active role in the workflow.
- `smoking_yolo_3cls/` and `smoking_yolo_3cls_balanced/` are kept as older outputs and should not
  replace `smoking_yolo_3cls_full/` or `smoke_bal/` without updating configs and docs.
