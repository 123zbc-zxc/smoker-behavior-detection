# Google Colab 断点续训指南

适用于本项目在 Google Colab GPU 环境下的快速恢复训练场景，目标是：

- 重新连接 Colab 后快速恢复到上次进度
- 用 Google Drive 保存 `last.pt` / `best.pt`
- 训练完成后一次性导出 `last.pt`、`best.pt`、`results.csv`

## 1. 推荐目录约定

- Drive 项目压缩包：`/content/drive/MyDrive/smoker_detection/smoker_cloud_pack.zip`
- Colab 解压目录：`/content/smoker_detection`
- Drive 权重备份目录：`/content/drive/MyDrive/smoker_detection_backups`

项目内已经提供：

- `configs/data_smoking_balanced_colab.yaml`
- `configs/train_yolov8n_colab_gpu.yaml`
- `scripts/inspect_checkpoint.py`
- `scripts/resume_training.py`
- `scripts/export_training_bundle.py`

其中 `configs/data_smoking_balanced_colab.yaml` 默认假设项目解压到 `/content/smoker_detection`。

## 2. 重新连接 Colab 后的最短流程

### 2.1 挂载 Drive 并解压项目

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

### 2.2 检查断点是不是最新

```python
!python scripts/inspect_checkpoint.py \
  --checkpoint /content/drive/MyDrive/smoker_detection_backups/last.pt
```

重点看：

- `epoch`
- `date`
- `train_args`

如果 `epoch` 明显低于预期，先到 Drive 检查是否有更高编号的 `last_eXX.pt`。

### 2.3 直接恢复训练

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

这个脚本会：

- 读取并打印 checkpoint 元数据
- 从该 `last.pt` 恢复训练
- 每 2 轮同步一次 `last.pt`
- 每 5 轮额外保留一个 `last_e{epoch}.pt`
- 持续同步 `best.pt`

### 2.4 训练完成后导出结果

```python
!python scripts/export_training_bundle.py \
  --run-dir runs/train/yolov8n_colab_640 \
  --output runs/exports/yolov8n_colab_640_bundle.zip
```

如果恢复后的实际目录不是 `runs/train/yolov8n_colab_640`，先执行：

```python
!find runs/train -path "*/weights/last.pt"
```

再把 `--run-dir` 改成真实路径。

导出文件默认包含：

- `weights/last.pt`
- `weights/best.pt`
- `results.csv`
- `args.yaml`
- `results.png`
- `train_batch0.jpg`
- `train_batch1.jpg`
- `train_batch2.jpg`

## 3. 训练结束后的验证

优先使用 `best.pt` 验证测试集：

```python
!python scripts/val.py \
  --weights runs/train/yolov8n_colab_640/weights/best.pt \
  --data configs/data_smoking_balanced_colab.yaml \
  --imgsz 640 \
  --batch 16 \
  --split test \
  --device 0 \
  --name colab_640_final_eval
```

## 4. 常见问题

### Q1. 重新连接后为什么不能直接 `--resume`

因为 Colab 重连后本地运行目录会消失，而 `scripts/train.py --resume` 默认只会找当前工作区的
`runs/train/<name>/weights/last.pt`。如果断点保存在 Drive，就应该改用：

```python
python scripts/resume_training.py --checkpoint <Drive里的last.pt>
```

### Q2. `configs/data_smoking_balanced_colab.yaml` 路径不对怎么办

当前文件默认写死为：

```text
/content/smoker_detection/datasets/final/smoke_bal
```

如果你把项目解压到了别的目录，直接修改该 YAML 里的 `path:` 即可。

### Q3. 为什么恢复训练时不建议改 `name`

断点续训的目标是沿用 checkpoint 里的原训练参数继续跑完。恢复时再改 `name` 或 `project`，
容易让 Ultralytics 的恢复逻辑和实际输出目录不一致。
