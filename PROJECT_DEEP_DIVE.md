# 吸烟行为检测项目深度解析

## 项目概览

**项目名称：** 基于深度学习的吸烟行为检测系统  
**项目类型：** 毕业设计项目  
**技术栈：** YOLOv8n + 注意力机制 + PyQt5 + FastAPI  
**项目路径：** `D:\Smoker Behavior Detection Based on Deep Learning`

---

## 一、项目目标与范围

### 核心目标
构建一个完整的吸烟行为检测系统，从数据准备、模型训练、性能评估到应用部署的全流程工程化实现。

### 检测目标（3类）
- **Class 0: cigarette（香烟）** - 小目标检测，最具挑战性
- **Class 1: smoking_person（吸烟者）** - 人物级别检测
- **Class 2: smoke（烟雾）** - 烟雾检测

### 系统功能
1. 多源数据集整合与预处理
2. YOLOv8n基线模型训练
3. 注意力机制（ECA/SE）对比实验
4. ONNX模型导出与优化
5. PyQt5桌面端推理演示
6. FastAPI Web管理控制台
7. 视频时序平滑检测
8. 香烟检测专项分析

---

## 二、项目结构详解

### 目录组织

```
项目根目录/
├── scripts/          # 40+ Python脚本：数据处理、训练、验证、预测、导出、分析
├── models/           # 自定义YOLO模型定义 + 注意力模块（ECA, SE）
├── app/              # 应用层：PyQt5桌面端 + FastAPI Web端
│   ├── main.py       # PyQt5桌面推理演示
│   ├── web_demo.py   # FastAPI Web管理控制台
│   ├── utils/        # 推理服务、数据库模型等
│   └── static/       # 前端静态资源
├── configs/          # YAML/JSON配置文件（数据集、训练、Web配置）
├── datasets/         # 数据集（原始、中间、最终、报告）
│   ├── raw/          # 原始数据源
│   ├── interim/      # 中间处理结果
│   └── final/        # 最终训练数据集
├── runs/             # 训练输出、权重、验证结果、实验报告
├── docs/             # 论文草稿和检查清单
└── figures/          # 生成的论文图表和架构图
```

---

## 三、数据管线

### 数据来源（5个数据集）
1. **AI Studio** - 百度AI Studio吸烟数据集
2. **Roboflow SmokingAndDrinking** - 吸烟饮酒检测数据集
3. **Kaggle smoking/eating/sleeping/phone** - 多行为数据集
4. **Kaggle smoking-drinking YOLO** - YOLO格式吸烟数据集
5. **D-Fire smoke/fire** - 烟雾火灾数据集

### 数据处理流程
```
原始数据 → VOC转YOLO格式 → 标签重映射 → 数据集合并 → 类别平衡 → 数据验证
```

### 关键处理脚本
- `convert_voc_to_yolo.py` - VOC格式转YOLO
- `prepare_roboflow_smoking.py` - Roboflow数据处理
- `prepare_added_datasets.py` - 额外数据集整合
- `build_balanced_dataset.py` - 类别均衡采样
- `check_dataset.py` - 数据集完整性验证

### 最终数据集
- **完整数据集：** `datasets/final/smoking_yolo_3cls_full/` - 全量合并3类数据
- **均衡数据集：** `datasets/final/smoke_bal/` - 均衡采样版本（CPU友好）

---

## 四、模型架构

### 基线模型：YOLOv8n
- Ultralytics YOLOv8 nano变体
- 参数量少，适合CPU部署
- 预训练权重初始化

### 改进方案：注意力机制

#### ECA（Efficient Channel Attention）
- **核心思想：** 使用自适应核大小的1D卷积代替全连接层
- **核大小计算：** `kernel = max(3, int(abs((log2(channels) + b) / gamma)))`
- **插入位置：** 检测头中两个上采样-拼接融合点之后（768和384通道）
- **优势：** 参数增量极小，针对性增强小目标特征

#### SE（Squeeze-and-Excitation）
- **核心思想：** 全局平均池化 → FC瓶颈 → Sigmoid门控
- **实现特点：** 首次前传时动态构建层结构，适配可变输入通道
- **对比基准：** 参数量略大于ECA

### 自定义模块注册
通过 `models/yolo_utils.py` 中的 `register_custom_modules()` 函数，将自定义注意力模块注册到Ultralytics框架，实现YAML定义架构的无缝集成。

### 架构配置文件
- `models/yolov8n_eca.yaml` - ECA增强模型
- `models/yolov8n_se.yaml` - SE增强模型

---

## 五、训练策略

### 默认训练配置
```yaml
model: yolov8n.pt（预训练权重）
dataset: configs/data_smoking_balanced.yaml
epochs: 80（最多200，视实验而定）
batch: 8
imgsz: 416（基线）/ 640（精细调优）
optimizer: AdamW
lr0: 0.001
scheduler: cosine
early_stopping_patience: 20
device: CPU（workers=0, cache=false）
```

### 训练入口
```bash
python scripts/train.py --config configs/train_yolov8n_balanced.yaml
```

### 训练流程
1. 加载YAML配置 → 2. 验证数据集路径 → 3. 注册自定义模块 → 4. 构建YOLO模型 → 5. 调用model.train() → 6. 输出训练摘要JSON

### 断点续训
`latest_checkpoint_for_run()` 函数定位 `weights/last.pt`，支持从中断处恢复训练。

### 知识蒸馏
`train_distilled_student.py` 实现教师-学生框架，`export_teacher_targets.py` 生成伪标签。

---

## 六、实验结果

### 核心模型对比

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------|-----------|--------|----------|---------------|
| YOLOv8n 基线 | 0.526 | 0.625 | 0.520 | 0.323 |
| YOLOv8n + ECA | 0.513 | 0.572 | 0.494 | 0.299 |
| YOLOv8n 640px硬微调 | **0.541** | **0.690** | **0.560** | **0.356** |

### 香烟检测专项
- 640px微调模型的香烟 mAP@0.5: 0.497
- 香烟 Recall: 0.747（召回率偏向设计）

### 视频时序验证
- 测试集：109个HMDB51烟雾视频
- 事件命中视频：99个
- 平滑命中帧：7054
- 稳定轨迹：429

---

## 七、应用层架构

### PyQt5 桌面端
- **入口：** `app/main.py`
- **功能：** 模型加载、图片/视频检测、实时帧显示
- **技术：** QTimer驱动视频播放，results列表展示检测结果

### FastAPI Web 管理控制台
- **入口：** `scripts/run_web_demo.py`
- **地址：** `http://127.0.0.1:8000`
- **技术栈：** FastAPI + SQLAlchemy + Jinja2 + 静态文件服务

#### 数据库模型
- `ModelRegistry` - 模型元数据注册
- `AppSetting` - 全局配置
- `ImageDetection` - 图片检测记录
- `VideoTask` - 视频处理任务

#### 数据库支持
- 主数据库：PostgreSQL（通过 `SMOKER_DB_URL` 环境变量配置）
- 备选：SQLite（`output/web_demo/` 本地存储）

### 模型加载优先级（Web端）
1. `runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt`
2. `runs/imported/smoker_weights_20260429/best.pt`
3. `runs/train/yolov8n_balanced_512/weights/best.pt`
4. `runs/train/yolov8n_balanced_30/weights/best.pt`
5. YOLOv8n预训练权重（兜底）

---

## 八、视频时序平滑算法

### 核心参数
```python
TEMPORAL_MATCH_IOU = 0.25      # 帧间匹配IoU阈值
TEMPORAL_STABLE_HITS = 3       # 稳定所需连续命中帧数
TEMPORAL_BRIDGE_FRAMES = 2     # 允许桥接的丢失帧数
TRACK_STALE_FRAMES = 5         # 轨迹过期帧数
CONFIDENCE_SMOOTH_ALPHA = 0.7  # 置信度指数平滑系数
```

### 算法流程
1. 对每帧检测结果，用IoU贪心匹配关联已有轨迹
2. 匹配到的轨迹更新位置，指数平滑置信度（0.6×旧 + 0.4×新）
3. 连续3次命中后轨迹变为"稳定"，开始渲染输出
4. 稳定轨迹最多桥接2帧丢失检测
5. 超过5帧未见的轨迹被剪枝

### 类别置信度阈值
- cigarette: 0.12（低阈值，召回优先）
- smoking_person: 0.22
- smoke: 0.28

---

## 九、关键依赖

| 类别 | 库 |
|------|-----|
| 深度学习 | ultralytics, torch, torchvision |
| 数据处理 | opencv-python, pillow, numpy, pyyaml |
| 桌面UI | PyQt5 |
| Web框架 | FastAPI, uvicorn, Jinja2 |
| 数据库 | SQLAlchemy, psycopg |
| 模型导出 | onnx, onnxruntime |
| 可视化 | matplotlib |
| 视频处理 | imageio-ffmpeg |

---

## 十、扩展指南

### 添加新注意力模块
1. 在 `models/modules/` 下创建模块文件
2. 在 `models/yolo_utils.py` 的 `register_custom_modules()` 中注册
3. 创建新的YAML架构文件引用自定义模块
4. 配置新的训练YAML运行实验

### 添加新数据源
1. 在 `scripts/` 中编写数据准备脚本
2. 转换为YOLO格式，统一标签映射
3. 使用 `build_balanced_dataset.py` 合并并均衡
4. 运行 `check_dataset.py` 验证完整性

### 调优视频检测
修改 `app/utils/web_inference.py` 中的常量参数即可调整时序平滑策略。

### 部署新模型
1. 训练完成后权重自动保存到 `runs/train/`
2. 在Web端通过 `ModelRegistry` 注册新模型
3. 或手动放入 `runs/imported/` 并更新配置

---

## 十一、已知局限性

1. **小目标定位困难：** 香烟目标极小，检测精度受限
2. **遮挡敏感：** 手部遮挡影响检测
3. **验证偏差：** 视频验证集偏向正样本烟雾视频
4. **CPU限制：** 当前配置面向CPU部署，训练速度较慢
5. **无版本锁定：** requirements.txt未固定依赖版本

---

*本文档生成于 2026-05-04，基于项目实际代码和配置文件的深度分析。*
