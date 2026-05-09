# Colab GPU Training Strategy

This plan assumes each Google account provides about 3 GPU hours per day. Use accounts to run different experiment lines, not the same checkpoint at the same time.

## Current Champion

- Default deployment checkpoint: `runs/imported/smoker_weights_20260429/best.pt`
- Test metrics: mAP50 `0.561`, mAP50-95 `0.360`
- Cigarette metrics: precision `0.445`, recall `0.721`, mAP50 `0.492`, mAP50-95 `0.299`
- Current decision: use this checkpoint for Web Demo until a new model beats it on full test validation.

## Daily 3-Hour GPU Window

Run this sequence whenever a Colab account gets a GPU:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!rm -rf /content/smoker_detection
!unzip -q /content/drive/MyDrive/smoker_detection/smoker_cloud_pack.zip -d /content/smoker_detection
%cd /content/smoker_detection
!pip install ultralytics==8.3.145 -q
!pip install -r requirements.txt -q
```

Check GPU:

```python
!nvidia-smi
```

Before training, always inspect the checkpoint:

```python
!python scripts/inspect_checkpoint.py \
  --checkpoint /content/drive/MyDrive/smoker_detection_backups/last.pt
```

Resume safely:

```python
!python scripts/resume_training.py \
  --checkpoint /content/drive/MyDrive/smoker_detection_backups/last.pt \
  --device 0 \
  --workers 2 \
  --cache disk \
  --backup-dir /content/drive/MyDrive/smoker_detection_backups \
  --sync-last-every 2 \
  --snapshot-every 5
```

Export artifacts before the session ends:

```python
!python scripts/export_training_bundle.py \
  --run-dir runs/train/yolov8n_colab_640 \
  --output runs/exports/yolov8n_colab_640_bundle.zip
```

## Account Allocation

- Account A: finished the `yolov8n_colab_640` line to epoch 80; the epoch-54 `best.pt` remains the champion.
- Account B: run `configs/train_yolov8n_colab_640_hard.yaml` as a short fine-tuning line from the current champion checkpoint.
- Account C or later: only start after A/B results are known; use it for distillation or one more targeted 640 run.

Do not run the same `last.pt` from two accounts at once. It will create conflicting Drive backups and make the checkpoint lineage unclear.

## Validation Gates

Only promote a model if it beats the current champion on full test validation:

```python
!python scripts/val.py \
  --weights runs/train/<run>/weights/best.pt \
  --data configs/data_smoking_balanced_colab.yaml \
  --imgsz 640 \
  --batch 16 \
  --split test \
  --device 0 \
  --name <run>_test_eval
```

Promotion thresholds:

- Overall `map50` must be greater than `0.561`.
- Cigarette `map50` must be greater than `0.492`.
- Cigarette recall should stay at or above `0.721`.
- If cigarette recall improves but precision collapses, keep it as a research checkpoint, not the Web Demo default.

## Next Accuracy Work

Use local CPU time for data quality, not long training:

```powershell
& ".venv\Scripts\python.exe" scripts\analyze_cigarette_detection.py --data configs\data_smoking_balanced.yaml --split train --weights runs\imported\smoker_weights_20260429\best.pt --imgsz 640 --conf 0.25 --max-images 500 --manifest-output runs\reports\train_hard_cigarette_priority_20260502.txt --output runs\reports\train_hard_cigarette_analysis_20260502.json
& ".venv\Scripts\python.exe" scripts\build_cigarette_review_pack.py --manifest runs\reports\train_hard_cigarette_priority_20260502.txt --data configs\data_smoking_balanced.yaml --output-dir tmp\cigarette_priority_review_train_hard_20260502 --rank-smallest-count 30 --small-side-px-threshold 16 --tiny-area-threshold 0.001
```

Review the generated hard cases first. Prioritize missed cigarettes, low-confidence cigarette matches, tiny or slender cigarette boxes, blurry or occluded cases, and incorrect or missing cigarette labels.

After cleaning labels or adding hard cases, rebuild the upload package and train the 640 hard config again. This is more likely to improve accuracy than simply adding more epochs to the current checkpoint.

For the next Colab line, copy the current champion to the path used by the hard config:

```python
from pathlib import Path
import shutil

project = Path("/content/smoker_detection")
weights_dir = project / "weights_imported"
weights_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2("/content/drive/MyDrive/smoker_detection_backups/best.pt", weights_dir / "best.pt")
```

Then run:

```python
%cd /content/smoker_detection
!python scripts/train.py --config configs/train_yolov8n_colab_640_hard.yaml --exist-ok
```

This hard line intentionally uses fewer epochs and a lower learning rate than the original continuation run. Its purpose is targeted fine-tuning from the champion, not retraining from `yolov8n.pt`.
