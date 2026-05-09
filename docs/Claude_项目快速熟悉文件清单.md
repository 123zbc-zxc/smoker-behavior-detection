# Claude 项目快速熟悉文件清单

项目路径：

```text
D:\Smoker Behavior Detection Based on Deep Learning
```

本文档用于让 Claude 快速熟悉项目，同时避免递归读取数据集、权重、图片、视频、训练产物等大文件，减少 token 浪费。

## 一、项目一句话概述

本项目是一个基于 YOLOv8n 的吸烟相关目标检测系统。系统不是做“吸烟/不吸烟”二分类，而是检测三类可见目标：

```text
0 cigarette
1 smoking_person
2 smoke
```

最终 Web 默认模型为：

```text
runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt
```

该模型含义是：

```text
YOLOv8n 640px hard fine-tune
```

它不是新网络结构，而是在 YOLOv8n 基础上，针对困难样本和香烟小目标继续微调得到的最终权重。

## 二、Claude 第一轮必须阅读的文件

以下文件用于快速建立项目整体认知：

```text
D:\Smoker Behavior Detection Based on Deep Learning\AGENTS.md
D:\Smoker Behavior Detection Based on Deep Learning\README.md
D:\Smoker Behavior Detection Based on Deep Learning\PROJECT_DEEP_DIVE.md
D:\Smoker Behavior Detection Based on Deep Learning\PROJECT_MEMORY.md
D:\Smoker Behavior Detection Based on Deep Learning\PROJECT_STRUCTURE.md
```

用途说明：

| 文件 | 作用 |
|---|---|
| `AGENTS.md` | 项目规则、目录约定、训练命令、系统运行方式 |
| `README.md` | 项目整体说明 |
| `PROJECT_DEEP_DIVE.md` | 项目深入总结 |
| `PROJECT_MEMORY.md` | 项目记忆和关键结论 |
| `PROJECT_STRUCTURE.md` | 项目结构说明，避免误扫目录 |

## 三、Web 系统核心代码

如果需要理解图片检测、视频检测、数据库、记录管理和阈值调整，优先阅读：

```text
D:\Smoker Behavior Detection Based on Deep Learning\app\web_demo.py
D:\Smoker Behavior Detection Based on Deep Learning\app\utils\web_inference.py
D:\Smoker Behavior Detection Based on Deep Learning\app\config.py
D:\Smoker Behavior Detection Based on Deep Learning\app\db.py
D:\Smoker Behavior Detection Based on Deep Learning\app\db_models.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\run_web_demo.py
```

用途说明：

| 文件 | 作用 |
|---|---|
| `app\web_demo.py` | FastAPI Web 系统主入口，包含路由、上传、检测和记录接口 |
| `app\utils\web_inference.py` | 核心推理逻辑，包含模型加载、图片检测、视频检测、时序平滑 |
| `app\config.py` | 运行配置，包含 SQLite/PostgreSQL 切换和输出目录 |
| `app\db.py` | 数据库初始化、Session 管理和默认配置 |
| `app\db_models.py` | 数据库表结构 |
| `scripts\run_web_demo.py` | Web 系统启动脚本 |

特别重要：

```text
D:\Smoker Behavior Detection Based on Deep Learning\app\utils\web_inference.py
```

该文件中包含项目核心推理逻辑、类别阈值、视频时序平滑参数和默认权重路径。

## 四、前端展示相关文件

如果需要理解 Web 页面、上传图片/视频、阈值调整和记录管理，阅读：

```text
D:\Smoker Behavior Detection Based on Deep Learning\app\ui\templates\index.html
D:\Smoker Behavior Detection Based on Deep Learning\app\ui\templates\video_report.html
D:\Smoker Behavior Detection Based on Deep Learning\app\ui\static\js\app.js
D:\Smoker Behavior Detection Based on Deep Learning\app\ui\static\css\site.css
```

用途说明：

| 文件 | 作用 |
|---|---|
| `index.html` | Web 主页面结构 |
| `video_report.html` | 视频检测报告页面 |
| `app.js` | 前端交互逻辑 |
| `site.css` | 页面样式 |

## 五、训练、验证和推理核心脚本

如果需要理解模型训练、验证和推理流程，阅读：

```text
D:\Smoker Behavior Detection Based on Deep Learning\scripts\train.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\val.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\predict.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\export_onnx.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\inspect_checkpoint.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\resume_training.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\export_training_bundle.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\yolo_utils.py
```

用途说明：

| 文件 | 作用 |
|---|---|
| `train.py` | 训练入口，读取 YAML 配置并调用 Ultralytics |
| `val.py` | 验证/测试入口 |
| `predict.py` | 本地图片/视频推理 |
| `export_onnx.py` | 导出 ONNX |
| `inspect_checkpoint.py` | 检查权重信息 |
| `resume_training.py` | 从 checkpoint 继续训练 |
| `export_training_bundle.py` | 打包训练结果 |
| `yolo_utils.py` | YOLO 相关通用工具 |

## 六、数据集构建与标签处理脚本

如果需要理解数据来源、类别映射、标签清洗和数据集构建，阅读：

```text
D:\Smoker Behavior Detection Based on Deep Learning\scripts\convert_voc_to_yolo.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\prepare_roboflow_smoking.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\prepare_roboflow_cigarette_smoke_detection.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\prepare_added_datasets.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\remap_labels.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\split_dataset.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\build_balanced_dataset.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\build_partial_final_dataset.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\check_dataset.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\audit_yolo_dataset.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\dataset_utils.py
```

用途说明：

| 文件 | 作用 |
|---|---|
| `convert_voc_to_yolo.py` | AI Studio VOC 标注转 YOLO |
| `prepare_roboflow_smoking.py` | Roboflow Smoking/Drinking 数据准备 |
| `prepare_roboflow_cigarette_smoke_detection.py` | Roboflow Cigarette Smoke Detection 数据准备 |
| `prepare_added_datasets.py` | 本地补充数据准备 |
| `remap_labels.py` | 类别编号重映射 |
| `split_dataset.py` | train/val/test 划分 |
| `build_balanced_dataset.py` | 构建 `smoke_bal` 平衡数据集 |
| `build_partial_final_dataset.py` | 构建合并数据集 |
| `check_dataset.py` | 数据完整性检查 |
| `audit_yolo_dataset.py` | 标签质量审计 |
| `dataset_utils.py` | 数据集通用工具函数 |

## 七、模型结构与注意力模块

如果需要理解 ECA、SE 和结构对比实验，阅读：

```text
D:\Smoker Behavior Detection Based on Deep Learning\models\yolov8n_eca.yaml
D:\Smoker Behavior Detection Based on Deep Learning\models\yolov8n_se.yaml
D:\Smoker Behavior Detection Based on Deep Learning\models\modules\eca.py
D:\Smoker Behavior Detection Based on Deep Learning\models\modules\se.py
D:\Smoker Behavior Detection Based on Deep Learning\models\modules\__init__.py
```

说明：

```text
ECA/SE 是结构对比实验，不是最终默认模型。
```

最终默认模型仍是：

```text
YOLOv8n 640px hard fine-tune
```

## 八、关键配置文件

如果需要理解数据配置、训练配置和最终模型配置，阅读：

```text
D:\Smoker Behavior Detection Based on Deep Learning\configs\data_smoking.yaml
D:\Smoker Behavior Detection Based on Deep Learning\configs\data_smoking_balanced.yaml
D:\Smoker Behavior Detection Based on Deep Learning\configs\data_smoking_balanced_plus_rfcsd4.yaml
D:\Smoker Behavior Detection Based on Deep Learning\configs\train_yolov8n.yaml
D:\Smoker Behavior Detection Based on Deep Learning\configs\train_yolov8n_balanced.yaml
D:\Smoker Behavior Detection Based on Deep Learning\configs\train_yolov8n_eca_balanced.yaml
D:\Smoker Behavior Detection Based on Deep Learning\configs\train_yolov8n_se_balanced.yaml
D:\Smoker Behavior Detection Based on Deep Learning\configs\train_yolov8n_colab_640_hard.yaml
D:\Smoker Behavior Detection Based on Deep Learning\configs\train_yolov8n_colab_640_plus_external.yaml
D:\Smoker Behavior Detection Based on Deep Learning\configs\web_demo.json
```

重点文件：

| 文件 | 作用 |
|---|---|
| `data_smoking_balanced.yaml` | 主实验数据集配置 |
| `data_smoking_balanced_plus_rfcsd4.yaml` | 加外部数据的负结果实验配置 |
| `train_yolov8n_colab_640_hard.yaml` | 最终 hard fine-tune 训练配置 |
| `train_yolov8n_colab_640_plus_external.yaml` | plus_external 对比实验配置 |
| `web_demo.json` | Web 系统配置 |

## 九、实验分析脚本

如果需要理解为什么最终选择 hard fine-tune，以及为什么 plus_external 是负结果，阅读：

```text
D:\Smoker Behavior Detection Based on Deep Learning\scripts\summarize_experiments.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\summarize_cigarette_experiments.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\analyze_cigarette_detection.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\search_video_temporal_params.py
D:\Smoker Behavior Detection Based on Deep Learning\scripts\evaluate_hmdb51_smoke_temporal.py
```

用途说明：

| 文件 | 作用 |
|---|---|
| `summarize_experiments.py` | 对比实验结果 |
| `summarize_cigarette_experiments.py` | 香烟检测专项对比 |
| `analyze_cigarette_detection.py` | 香烟小目标分析 |
| `search_video_temporal_params.py` | 视频时序平滑参数搜索 |
| `evaluate_hmdb51_smoke_temporal.py` | 视频时序评估 |

## 十、论文和答辩相关总结文件

如果任务是修改论文、制作 PPT 或准备答辩，优先阅读：

```text
D:\Smoker Behavior Detection Based on Deep Learning\docs\项目知识熟悉手册_答辩版_20260506.md
D:\Smoker Behavior Detection Based on Deep Learning\docs\project_overall_report_20260504.md
D:\Smoker Behavior Detection Based on Deep Learning\docs\thesis_model_system_conclusions_20260503.md
D:\Smoker Behavior Detection Based on Deep Learning\docs\attention_mechanism_failure_analysis_20260505.md
D:\Smoker Behavior Detection Based on Deep Learning\docs\smoke_smokingperson_analysis_20260505.md
D:\Smoker Behavior Detection Based on Deep Learning\docs\VERIFICATION_REPORT_20260505.md
D:\Smoker Behavior Detection Based on Deep Learning\第5章_注意力机制集成与负向结果分析.md
D:\Smoker Behavior Detection Based on Deep Learning\论文章节生成进度报告.md
```

用途说明：

| 文件 | 作用 |
|---|---|
| `项目知识熟悉手册_答辩版_20260506.md` | 答辩项目知识手册 |
| `project_overall_report_20260504.md` | 项目整体报告 |
| `thesis_model_system_conclusions_20260503.md` | 模型和系统结论 |
| `attention_mechanism_failure_analysis_20260505.md` | ECA/SE 没提升的解释 |
| `smoke_smokingperson_analysis_20260505.md` | smoke/smoking_person 分析 |
| `VERIFICATION_REPORT_20260505.md` | 验证报告 |
| `第5章_注意力机制集成与负向结果分析.md` | 论文第五章相关材料 |
| `论文章节生成进度报告.md` | 论文修改进度 |

## 十一、最终模型路径

只需要告诉 Claude 路径，不要让 Claude 读取 `.pt` 文件。

```text
D:\Smoker Behavior Detection Based on Deep Learning\runs\imported\yolov8n_colab_640_hard_candidate_20260502\train\weights\best.pt
```

说明：

```text
最终 Web 默认模型是 YOLOv8n 640px hard fine-tune。
它不是新结构，而是在 YOLOv8n 基础上针对困难样本和香烟小目标继续微调得到的最终权重。
```

## 十二、可以知道存在，但不要递归读取的目录

以下目录很大，只在确实需要具体路径时查看，不要递归读取：

```text
D:\Smoker Behavior Detection Based on Deep Learning\datasets
D:\Smoker Behavior Detection Based on Deep Learning\runs
D:\Smoker Behavior Detection Based on Deep Learning\output
D:\Smoker Behavior Detection Based on Deep Learning\tmp
D:\Smoker Behavior Detection Based on Deep Learning\backup
D:\Smoker Behavior Detection Based on Deep Learning\smoker_cloud_pack
D:\Smoker Behavior Detection Based on Deep Learning\smokingVSnotsmoking
```

原因：

| 目录 | 是否读取 | 原因 |
|---|---|---|
| `datasets` | 不递归读 | 图片、标签、缓存太多 |
| `runs` | 不递归读 | 训练结果、权重、图表很多 |
| `output` | 不递归读 | Web 输出、PPT、检测结果多 |
| `tmp` | 不读 | 临时文件多，容易误导 |
| `backup` | 不读 | 备份内容重复 |
| `smoker_cloud_pack` | 不读 | 打包文件，重复项目内容 |
| `smokingVSnotsmoking` | 默认不读 | 可能是早期或旁支数据 |

## 十三、禁止读取的文件类型

为了节省 token，Claude 不应读取以下文件类型：

```text
*.zip
*.pt
*.onnx
*.jpg
*.jpeg
*.png
*.mp4
*.avi
*.cache
*.db
*.pyc
__pycache__/
.venv/
.idea/
```

尤其不要读取：

```text
D:\Smoker Behavior Detection Based on Deep Learning\Cigarette.yolov8.zip
D:\Smoker Behavior Detection Based on Deep Learning\smoke.yolov8.zip
D:\Smoker Behavior Detection Based on Deep Learning\smoker_cloud_pack.zip
D:\Smoker Behavior Detection Based on Deep Learning\smoker_cloud_pack_ok.zip
D:\Smoker Behavior Detection Based on Deep Learning\smoker_weights.zip
D:\Smoker Behavior Detection Based on Deep Learning\yolov8n.pt
```

## 十四、如果只给 Claude 10 个文件

如果想最省 token，第一轮只给 Claude 读这 10 个：

```text
D:\Smoker Behavior Detection Based on Deep Learning\AGENTS.md
D:\Smoker Behavior Detection Based on Deep Learning\README.md
D:\Smoker Behavior Detection Based on Deep Learning\PROJECT_DEEP_DIVE.md
D:\Smoker Behavior Detection Based on Deep Learning\app\web_demo.py
D:\Smoker Behavior Detection Based on Deep Learning\app\utils\web_inference.py
D:\Smoker Behavior Detection Based on Deep Learning\app\db_models.py
D:\Smoker Behavior Detection Based on Deep Learning\configs\train_yolov8n_colab_640_hard.yaml
D:\Smoker Behavior Detection Based on Deep Learning\configs\data_smoking_balanced.yaml
D:\Smoker Behavior Detection Based on Deep Learning\docs\项目知识熟悉手册_答辩版_20260506.md
D:\Smoker Behavior Detection Based on Deep Learning\docs\attention_mechanism_failure_analysis_20260505.md
```

这 10 个文件足够 Claude 快速理解：

```text
1. 项目做什么
2. 系统怎么运行
3. 模型怎么推理
4. 数据库怎么保存记录
5. 最终模型是什么
6. 数据集怎么配置
7. 为什么 ECA/SE 和 plus_external 不是最终方案
```

## 十五、如果 Claude 要改论文

论文任务时再额外提供以下文件，不要一开始就给：

```text
D:\Smoker Behavior Detection Based on Deep Learning\定稿_智能224-202212903403_陈刚_林一锋_完整重写版_自动目录格式最终版.docx
D:\Smoker Behavior Detection Based on Deep Learning\docs\paper\graduation_final_draft.md
D:\Smoker Behavior Detection Based on Deep Learning\docs\paper\finalization_checklist.md
D:\Smoker Behavior Detection Based on Deep Learning\第5章_注意力机制集成与负向结果分析.md
```

说明：

```text
.docx 文件只有在要改论文格式或正文时再给。
如果只是熟悉项目，先不要让 Claude 读取 .docx。
```

## 十六、可以直接复制给 Claude 的提示词

```text
你现在需要快速熟悉一个毕业设计项目，但请严格控制读取范围，不要递归扫描整个项目。

项目路径：
D:\Smoker Behavior Detection Based on Deep Learning

请优先阅读以下文件：
1. AGENTS.md
2. README.md
3. PROJECT_DEEP_DIVE.md
4. PROJECT_MEMORY.md
5. PROJECT_STRUCTURE.md
6. app\web_demo.py
7. app\utils\web_inference.py
8. app\config.py
9. app\db.py
10. app\db_models.py
11. configs\data_smoking_balanced.yaml
12. configs\train_yolov8n_colab_640_hard.yaml
13. docs\项目知识熟悉手册_答辩版_20260506.md
14. docs\attention_mechanism_failure_analysis_20260505.md

请不要读取以下目录或文件类型：
datasets/
runs/
output/
tmp/
backup/
.venv/
__pycache__/
*.zip
*.pt
*.onnx
*.jpg
*.png
*.mp4
*.avi
*.cache
*.db
*.pyc

项目核心结论：
本项目是基于 YOLOv8n 的吸烟相关目标检测系统，不是吸烟/不吸烟二分类。最终检测三类目标：
0 cigarette
1 smoking_person
2 smoke

最终 Web 默认模型是：
runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt

该模型含义是 YOLOv8n 640px hard fine-tune，即在 YOLOv8n 基础上针对困难样本和香烟小目标继续微调得到的最终权重，不是新网络结构。

请先输出：
1. 项目整体功能概述
2. 数据集构建流程
3. 模型训练与最终模型选择逻辑
4. Web 系统模块结构
5. 答辩时最容易被问的问题
6. 你还需要我补充哪些文件
```

## 十七、给 Claude 的阅读原则

```text
1. 不要递归扫描整个项目。
2. 不要读取数据集图片、视频、权重和压缩包。
3. 先读 README、AGENTS、PROJECT_DEEP_DIVE，再读 app 和 configs。
4. 如果要理解模型，优先读配置和总结文档，不要读 .pt。
5. 如果要理解论文，先读 Markdown 总结，最后再处理 docx。
6. 如果要理解系统演示，重点读 app\web_demo.py、app\utils\web_inference.py 和前端文件。
```

