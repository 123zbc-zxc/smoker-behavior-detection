# 项目记忆文档：基于深度学习的吸烟者行为检测系统

更新时间：2026-04-18
项目路径：`D:\Smoker Behavior Detection Based on Deep Learning`

## 1. 当前工程状态

该仓库已经不再是“只有论文材料的半成品”，而是一个已打通主要链路的毕业设计工程仓库。

### 已确认可用的主链路
- 数据准备：
  - `scripts/convert_voc_to_yolo.py`
  - `scripts/prepare_roboflow_smoking.py`
  - `scripts/prepare_added_datasets.py`
  - `scripts/build_partial_final_dataset.py`
  - `scripts/build_balanced_dataset.py`
  - `scripts/check_dataset.py`
- 通用数据工具：
  - `scripts/remap_labels.py`
  - `scripts/split_dataset.py`
- 训练与评估：
  - `scripts/train.py`
  - `scripts/val.py`
  - `scripts/predict.py`
  - `scripts/export_onnx.py`
  - `scripts/summarize_experiments.py`
- 演示系统：
  - `app/main.py`（PyQt5 本地演示）
  - `app/web_demo.py` + `scripts/run_web_demo.py`（FastAPI 答辩演示）
  - `scripts/system_smoke_test.py`（Web Demo 冒烟测试）

### 当前推荐使用的资产
- 全量融合数据集：`datasets/final/smoking_yolo_3cls_full`
- 当前实验主用数据集：`datasets/final/smoke_bal`
- baseline 主权重：`runs/train/yolov8n_balanced_30/weights/best.pt`
- ECA 对比权重：`runs/train/yolov8n_eca_balanced_304/weights/best.pt`
- Web Demo 配置：`configs/web_demo.json`

## 2. 当前保留的实验结论

当前论文采用的平衡数据集实验结果为：

- baseline：
  - Precision `0.526`
  - Recall `0.625`
  - mAP@0.5 `0.520`
  - mAP@0.5:0.95 `0.323`
- ECA：
  - Precision `0.513`
  - Recall `0.572`
  - mAP@0.5 `0.494`
  - mAP@0.5:0.95 `0.299`

当前阶段 baseline 仍是更稳妥的论文主结果，ECA 作为完整对比实验保留。

## 3. 数据与目录共识

### 统一类别定义
建议并已实际使用的 3 类定义为：

```text
0 cigarette
1 smoking_person
2 smoke
```

### 当前有效的数据阶段
- `datasets/raw/`：原始下载数据
- `datasets/interim/`：转换、映射、局部清洗后的中间数据
- `datasets/final/`：最终 YOLO 数据集
- `datasets/reports/`：构建和校验报告

说明：
- `datasets/processed/` 当前为空的历史目录，不属于现行流程。
- `datasets/final/smoking_yolo_3cls/` 与 `datasets/final/smoking_yolo_3cls_balanced/` 视为历史输出，不作为默认主入口。

## 4. 论文与工程的关系

### 工程主线
用于训练、验证、预测、导出和演示：
- 数据集整理
- YOLOv8n baseline 训练
- YOLOv8n+ECA 对比实验
- ONNX 导出
- PyQt5 本地演示
- FastAPI 网页演示

### 论文交付主线
用于答辩与文档整理：
- `generate_figures.py`
- `scripts/build_standardized_thesis_docx.py`
- `scripts/sync_final_docx_from_markdown.py`
- `scripts/fix_final_docx.py`
- `docs/paper/` 下的草稿与检查清单

结论：后续维护时应明确区分“检测工程主线”和“论文材料主线”，避免互相污染。

## 5. 当前仍需持续关注的问题

- 训练结果仍以 CPU 环境下的 30 轮实验为主，若后续补做更完整实验，需要同步更新 README、配置说明和论文口径。
- 历史目录和历史数据集仍然保留，后续如决定清理，需要先确认不会影响论文附件、截图和已有脚本。
- 文档脚本较多，后续如果再增加入口脚本，应同步更新 `AGENTS.md`、`README.md` 和本文件。
