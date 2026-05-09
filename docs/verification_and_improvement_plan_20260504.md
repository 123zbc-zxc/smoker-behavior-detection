# 数据验证报告 & 识别率提升最优方案

**文档日期：** 2026-05-04  
**基础文档：** `docs/project_overall_report_20260504.md`  
**验证方法：** 逐项交叉比对源JSON报告文件与实际文件系统

---

## 第一部分：数据真实性验证

### 验证结论：全部通过（31/31项 100%）

报告中所有数据均可追溯到源文件，无虚报、错报或数据不一致问题。

### 验证明细

#### 1. 源文件存在性验证

| 引用文件 | 状态 |
|----------|------|
| `runs/reports/final_model_system_test_report_20260503.json` | VERIFIED |
| `runs/reports/project_full_audit_20260503.json` | VERIFIED |
| `runs/reports/system_smoke_test.json` | VERIFIED |
| `configs/web_demo.json` | VERIFIED |

#### 2. 模型测试集指标验证

| 指标 | 报告值 | 源文件精确值 | 状态 |
|------|--------|--------------|------|
| hard_finetune Precision | 0.5405 | 0.5405130262470422 | VERIFIED |
| hard_finetune Recall | 0.6900 | 0.6899805529856463 | VERIFIED |
| hard_finetune mAP@0.5 | 0.5604 | 0.5604166396681468 | VERIFIED |
| hard_finetune mAP@0.5:0.95 | 0.3559 | 0.35594531083523145 | VERIFIED |
| hard_finetune cigarette Recall | 0.7472 | 0.7472118959107806 | VERIFIED |
| old_champion Precision | 0.5387 | 0.5387117452911295 | VERIFIED |
| old_champion Recall | 0.6859 | 0.6858748894882419 | VERIFIED |
| old_champion mAP@0.5 | 0.5613 | 0.5612740629141322 | VERIFIED |
| old_champion mAP@0.5:0.95 | 0.3597 | 0.3597388332341315 | VERIFIED |

#### 3. 数据集统计验证

| 数据集分割 | 报告值 | 实际值 | 状态 |
|-----------|--------|--------|------|
| train | 14416 | 14416 | VERIFIED |
| val | 1895 | 1895 | VERIFIED |
| test | 1925 | 1925 | VERIFIED |

#### 4. 视频时序验证数据

| 指标 | 报告值(hard) | 实际值 | 状态 |
|------|-------------|--------|------|
| temporal_event_hit_videos | 99 | 99 | VERIFIED |
| smoothed_hit_frames | 7054 | 7054 | VERIFIED |
| stable_track_count | 429 | 429 | VERIFIED |
| old_champion event_hit | 97 | 97 | VERIFIED |
| old_champion smoothed | 6720 | 6720 | VERIFIED |
| old_champion tracks | 388 | 388 | VERIFIED |

#### 5. 物理文件验证

| 文件/资源 | 状态 |
|-----------|------|
| hard_finetune 权重 best.pt | VERIFIED |
| old_champion 权重 best.pt | VERIFIED |
| 4个自定义视频文件 | VERIFIED（元数据完全匹配） |
| Web Demo配置 conf/iou/imgsz | VERIFIED |
| models_count=7 | VERIFIED |

### 验证结论

报告数据100%真实可靠，四舍五入均正确，可直接用于论文引用和答辩材料。

---

## 第二部分：当前瓶颈深度分析

### 瓶颈一：数据质量与小目标问题（主要瓶颈）

**现象：**
- cigarette 类存在大量极小目标（151个tiny目标面积<0.0005，482个small目标面积<0.0025）
- 500张测试图中有390个漏检cigarette，其中183个为低置信度匹配
- 标注边界一致性不足（5个数据源标注标准不统一）

**影响：**
- cigarette精确率仅0.4456，说明存在大量误检框
- mAP@0.5:0.95仅0.2989，表明定位精度差——模型"找到了但框不准"

**证据：**
- ECA注意力模块专为小目标设计，却反而比基线差（mAP@0.5: 0.494 < 0.520），说明架构改进无法弥补数据信号不足

### 瓶颈二：模型容量不足

**现象：**
- YOLOv8n是最小变体，参数量有限
- 三类目标尺度差异极大（cigarette极小，smoking_person全身，smoke不定形）
- smoking_person表现最弱（P=0.433, R=0.597, mAP@0.5=0.396）

**影响：**
- nano模型特征表达能力不足以同时处理跨尺度目标
- 从未测试过yolov8s/yolov8m等更大模型

### 瓶颈三：训练策略局限

**现象：**
- CPU训练限制了batch size（8）和迭代速度
- hard_finetune提升recall但mAP@0.5:0.95下降，存在召回-精度权衡失调
- 缺少类别加权loss、焦点loss等针对性策略

**影响：**
- 模型偏向记忆简单模式，对难样本泛化不足
- 训练轮次不够（GPU训练可轻松跑300+epoch）

### 瓶颈四：验证体系不完整

**现象：**
- 视频验证仅使用正样本（HMDB51 smoke视频）
- 缺乏非吸烟视频的误检率（False Positive Rate）测试
- 无法定量回答"系统会不会误报"

---

## 第三部分：提高识别率最优方案

### 方案总览（按性价比排序）

| 优先级 | 方案 | 预期mAP@0.5提升 | 实施难度 | 耗时 |
|--------|------|-----------------|----------|------|
| P0 | 升级模型规模 + GPU训练 | +5~10% | 低 | 1-2天 |
| P1 | 数据质量提升（标注清洗+难样本挖掘） | +3~8% | 中 | 2-3天 |
| P2 | 多尺度训练 + 高级增强策略 | +2~5% | 低 | 1天 |
| P3 | 类别解耦策略 | +2~4% | 中 | 1-2天 |
| P4 | 模型融合/集成 | +1~3% | 中 | 1天 |
| P5 | 补充负样本验证 | 不直接提升指标 | 低 | 半天 |

---

### P0：升级模型规模 + GPU训练（最高优先级）

**核心思路：** 当前YOLOv8n容量不足是硬性限制，升级到YOLOv8s或YOLOv8m是最直接的提升手段。

**具体操作：**
```yaml
# configs/train_yolov8s_gpu.yaml
model: yolov8s.pt          # 从nano升级到small（参数量×3.3）
imgsz: 640                 # 固定640分辨率
batch: 32                  # GPU可支持更大batch
epochs: 200                # 更充分训练
optimizer: AdamW
lr0: 0.001
lrf: 0.01                  # cosine退火到1%
patience: 30               # 更多耐心
device: 0                  # GPU
close_mosaic: 15           # 后15个epoch关闭mosaic精细化
```

**预期效果：**
- yolov8s在COCO上比yolov8n高约7个mAP点，迁移到本任务预期+5~10%
- GPU加速10-20倍，允许更多实验迭代
- 640px + 大batch稳定训练

**风险控制：**
- 保留当前hard_finetune作为对照组
- 如果GPU资源有限，可用Google Colab T4（免费版够用）

---

### P1：数据质量提升

**核心思路：** 数据是检测模型的天花板。标注不一致和小目标信号弱是根本问题。

**具体操作：**

**(a) 标注一致性清洗**
```python
# 思路：用当前模型做pseudo-label，与原标注对比
# 1. 对train set做推理，找出模型高置信预测 vs 无标注的矛盾样本
# 2. 找出标注框 vs 模型预测IoU极低的样本（标注可能有误）
# 3. 人工复审Top-200困难样本
```

**(b) 难样本挖掘（Hard Example Mining）**
```python
# 对test/val集合中漏检的390个cigarette案例分类：
# - 遮挡类：手指遮挡香烟 → 增加遮挡增强
# - 极小类：面积<0.0005 → 提高输入分辨率或裁切放大训练
# - 模糊类：低分辨率/运动模糊 → 增加blur增强
# - 异形类：雪茄/电子烟 → 补充特定数据
```

**(c) 小目标专项增强**
```yaml
# 增加copy-paste增强，把cigarette粘贴到更多背景中
copy_paste: 0.15          # 从0.05提升到0.15
mixup: 0.1                # 引入mixup
scale: 0.7                # 扩大尺度变化范围
```

---

### P2：多尺度训练 + 高级增强

**核心思路：** 利用多尺度训练让模型适应不同大小目标。

**具体操作：**
```yaml
# 多尺度训练配置
imgsz: 640
multi_scale: true          # 启用0.5x-1.5x随机缩放
# 或者固定两阶段：
# Stage 1: imgsz=416, epochs=100 (快速收敛)
# Stage 2: imgsz=640, epochs=100, lr=1e-4 (精细化)

# 高级增强
mosaic: 0.7                # 更激进的mosaic
erasing: 0.15              # 随机擦除模拟遮挡
perspective: 0.001         # 轻微透视变换
```

**Tile推理优化（不改训练，改推理）：**
```python
# 对高分辨率图片做滑窗推理，专注小目标
# SAHI (Slicing Aided Hyper Inference) 方案
from sahi import AutoDetectionModel, get_sliced_prediction
# 切片大小640，重叠0.2，对cigarette类效果显著
```

---

### P3：类别解耦策略

**核心思路：** 三类目标特性差异太大，统一训练存在梯度冲突。

**方案A：类别加权Loss**
```python
# 在ultralytics训练中修改class weights
# smoking_person最弱，提升其loss权重
cls_weights: [1.0, 1.5, 0.8]  # cigarette, smoking_person, smoke
# 或使用Focal Loss增强难分类样本的梯度
```

**方案B：双模型策略**
```
模型1：专注cigarette+smoke（小目标+烟雾）→ imgsz=640
模型2：专注smoking_person（人体检测）→ imgsz=416
最终：后处理合并两个模型输出
```

**方案C：课程学习（Curriculum Learning）**
```
Phase 1: 只用smoke类（最简单）训练warm-up
Phase 2: 加入smoking_person
Phase 3: 加入cigarette难样本
```

---

### P4：模型融合

**核心思路：** 多个模型投票/融合可提升稳定性。

**WBF (Weighted Boxes Fusion)：**
```python
# 融合hard_finetune + old_champion + 新训yolov8s的预测
from ensemble_boxes import weighted_boxes_fusion
# 三模型融合通常可在单模型基础上+1~3% mAP
```

**TTA (Test Time Augmentation)：**
```python
# Ultralytics内置TTA
model.predict(source, augment=True)  # 自动多尺度+翻转
# 通常+1~2% mAP，但推理变慢3x
```

---

### P5：补充负样本验证

**核心思路：** 虽然不直接提升指标，但对论文可信度和系统实用性至关重要。

**操作：**
```
1. 收集20-30个非吸烟场景视频（办公室、街道、餐厅等）
2. 用当前模型推理，统计False Positive Rate
3. 设定合理阈值使FPR < 5%
4. 论文中报告"非吸烟场景误检率"指标
```

---

## 第四部分：推荐实施路线图

### 最优策略组合（2-3天见效）

```
Day 1: [P0] Colab GPU训练YOLOv8s 640px 200epoch
        同时 [P5] 收集负样本视频准备测试

Day 2: [P2] 对YOLOv8s模型加入多尺度训练 + copy_paste增强
        [P1a] 用模型推理结果做pseudo-label对比，筛选矛盾样本

Day 3: [P3] 尝试class_weights调整smoking_person权重
        [P4] 融合YOLOv8s + hard_finetune做WBF集成
        评估最终指标
```

### 保守策略（仅答辩需要，1天内）

```
1. 直接用SAHI切片推理提升cigarette检测（不改模型，0.5天）
2. TTA推理模式启用（一行代码，即时生效）
3. 补充5-10个负样本视频做误检报告（增强论文可信度）
```

---

## 第五部分：预期目标

| 指标 | 当前值 | 保守目标 | 理想目标 |
|------|--------|----------|----------|
| 总体 mAP@0.5 | 0.560 | 0.620 | 0.680+ |
| 总体 mAP@0.5:0.95 | 0.356 | 0.400 | 0.450+ |
| cigarette Recall | 0.747 | 0.780 | 0.820+ |
| smoking_person mAP@0.5 | 0.396 | 0.450 | 0.520+ |
| 视频事件命中率 | 99/109 | 103/109 | 107/109 |

---

*本方案基于项目实际数据验证和瓶颈分析制定，所有建议均可在现有代码框架内实施。*
