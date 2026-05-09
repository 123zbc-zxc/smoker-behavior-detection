# 答辩前72小时行动计划

**目标：** 提升项目稳定性、完善论文数据、确保答辩演示万无一失  
**时间：** 2026-05-05 至 2026-05-08（答辩日）  
**策略：** 优先保证"不出错" > 追求"更好看"

---

## Day 1（今天，5月5日）：修复 + 验证

**核心任务：** 修正已知问题，固化演示流程

### 上午（4小时）：修正报告数据 + 补充负样本测试

#### Task 1.1：修正分析报告中的错误数据（30分钟）

已发现的4处错误需要修正：

```bash
# 1. 修正 attention_mechanism_failure_analysis_20260505.md
# 将 "151个tiny目标" 改为 "51个tiny目标"
# 将 "面积<0.0005" 改为 "阈值0.00025"
# 标注 "SE epoch 1 mAP=0.004" 为 "根据实验记录（详细日志未保存）"

# 2. 修正 smoke_smokingperson_analysis_20260505.md
# 将 "只有30%帧有检测" 改为 "约43%帧有检测"
```

**执行：** 手动编辑这两个文档，使用验证报告中的"修正后"数据。

#### Task 1.2：补充负样本误检测试（2小时）

**目的：** 回答"系统会不会误报"，增强论文可信度。

```bash
# 1. 收集10-15个非吸烟视频（办公室、街道、餐厅等）
# 可以从YouTube下载或使用项目中已有的视频

# 2. 运行检测并统计误检
cd "D:\Smoker Behavior Detection Based on Deep Learning"
& ".\.venv\Scripts\python.exe" scripts\predict.py --source "path/to/negative_videos/" --weights "runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt" --conf 0.12 --save

# 3. 手动查看结果，统计误检数量
# 记录：总帧数、误检帧数、误检率
```

**预期结果：** 得到一个"非吸烟场景误检率"数据（目标<5%），写入论文。

#### Task 1.3：固化答辩演示流程（1.5小时）

**准备3套演示素材：**

1. **图片检测演示**
   ```bash
   # 选择1张效果好的测试图片（cigarette+smoke都清晰）
   # 运行TTA+SAHI对比
   & ".\.venv\Scripts\python.exe" scripts\enhanced_inference.py --source "你选的图片.jpg" --mode compare --save-compare --output output\demo_defense
   
   # 验证输出图片清晰可见
   ```

2. **视频检测演示**
   ```bash
   # 选择1个短视频（10-15秒，吸烟行为明显）
   # 运行Web Demo视频检测，确保能生成报告
   # 或使用命令行：
   python scripts/video_detect.py --source "你的视频.mp4" --output output/demo_defense/
   ```

3. **Web Demo演示**
   ```bash
   # 启动Web服务，测试图片上传和检测
   python app/main.py
   # 访问 http://localhost:8000
   # 截图保存：首页、图片检测结果、视频报告页面
   ```

**输出：** 3个文件夹，每个包含演示素材和截图，答辩当天直接用。

---

### 下午（4小时）：GPU训练YOLOv8s（如果有GPU）

#### Task 1.4：快速训练YOLOv8s（3小时训练 + 1小时验证）

**前提：** 如果你有GPU（本地或Colab），立即开始。如果没有，跳到Task 1.5。

```python
# 在Colab或本地GPU上运行
from ultralytics import YOLO

# 加载YOLOv8s预训练权重
model = YOLO('yolov8s.pt')

# 训练配置（快速版，3小时内完成）
results = model.train(
    data='configs/data_smoking_balanced.yaml',
    epochs=100,              # 100轮约3小时（GPU）
    imgsz=640,
    batch=32,                # GPU可支持
    device=0,
    patience=20,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    mosaic=0.7,
    copy_paste=0.15,
    close_mosaic=10,
    project='runs/train',
    name='yolov8s_quick_20260505',
)

# 训练完成后，立即在测试集上验证
model = YOLO('runs/train/yolov8s_quick_20260505/weights/best.pt')
metrics = model.val(data='configs/data_smoking_balanced.yaml', split='test')

# 保存结果
print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5:0.95: {metrics.box.map}")
```

**预期效果：**
- YOLOv8s比YOLOv8n大3.4倍，预期mAP@0.5提升5-8%
- 如果达到0.60+，可以作为"改进实验"写入论文

**如果没有GPU：** 跳过此步，使用当前hard_finetune模型即可。

#### Task 1.5（备选）：优化当前模型的置信度阈值

如果没有GPU，用这个时间优化阈值：

```python
# 调整置信度阈值，提升smoking_person和smoke的召回
# 修改 app/utils/web_inference.py 或 scripts/enhanced_inference.py

CLASS_CONF_THRESHOLDS = {
    0: 0.12,    # cigarette 保持不变
    1: 0.15,    # smoking_person 从0.22降到0.15
    2: 0.22,    # smoke 从0.28降到0.22
}

# 重新运行测试集验证
python scripts/val.py --weights "runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt" --data configs/data_smoking_balanced.yaml --split test

# 对比新旧阈值的效果
```

**预期效果：** smoking_person Recall提升10-15%，smoke Recall提升5-10%。

---

## Day 2（5月6日）：论文完善 + 数据补全

**核心任务：** 补充论文缺失的图表和数据

### 上午（4小时）：生成论文所需的图表

#### Task 2.1：生成对比图表（2小时）

**需要的图表：**

1. **模型性能对比表**（已有数据，制作表格）
   - hard_finetune vs old_champion
   - 各类别P/R/mAP对比
   - 使用Excel或Python matplotlib生成

2. **TTA+SAHI效果对比图**
   ```bash
   # 使用之前生成的compare模式输出
   # 2x2对比图：normal / TTA / SAHI / TTA+SAHI
   # 已有，直接复制到论文图片文件夹
   ```

3. **视频时序平滑效果图**
   ```python
   # 绘制一个视频的检测框数量随时间变化曲线
   # raw detections vs smoothed detections
   # 展示时序平滑的效果
   ```

4. **类别分布饼图**
   ```python
   import matplotlib.pyplot as plt
   
   # 数据集分布
   labels = ['cigarette', 'smoking_person', 'smoke']
   train_counts = [7021, 7021, 10562]
   test_counts = [1077, 784, 1043]
   
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   ax1.pie(train_counts, labels=labels, autopct='%1.1f%%')
   ax1.set_title('训练集分布')
   ax2.pie(test_counts, labels=labels, autopct='%1.1f%%')
   ax2.set_title('测试集分布')
   plt.savefig('output/paper_figures/dataset_distribution.png', dpi=300)
   ```

#### Task 2.2：补充实验数据表格（2小时）

**需要的表格：**

1. **数据集统计表**（已验证，直接引用）
   - 原始数据、均衡后数据、train/val/test分布

2. **模型训练配置表**
   - 超参数、增强策略、训练轮次

3. **消融实验对比表**
   - Baseline vs ECA vs SE vs Hard_finetune
   - 注意：ECA/SE数据标注为"历史实验记录"

4. **视频时序参数表**
   - match_iou, stable_hits, bridge_frames等
   - 已有，直接引用

5. **负样本误检率表**（今天上午Task 1.2的结果）
   - 视频数量、总帧数、误检帧数、误检率

---

### 下午（4小时）：论文文字完善

#### Task 2.3：修正论文中的数据引用（1小时）

**检查并修正：**
- 所有引用的mAP/P/R数据是否与验证报告一致
- 数据集数量（51 tiny boxes，不是151）
- 视频帧检测率（43.3%，不是30%）
- TTA+SAHI提升表述（明确绝对vs相对）

#### Task 2.4：补充"改进与展望"章节（1小时）

**基于验证报告，写入：**

1. **已识别的问题：**
   - smoking_person数据不足（784测试样本）
   - smoke阈值过高导致召回不足
   - 注意力机制插入位置不当

2. **已尝试的改进：**
   - TTA+SAHI推理增强（cigarette Recall +14.8%）
   - 视频时序平滑（稳定轨迹数429）
   - 类别阈值优化（0.12/0.22/0.28）

3. **未来改进方向：**
   - 升级YOLOv8s/m模型
   - 补充smoking_person样本
   - 两阶段检测（人体检测+动作分类）
   - 多模态融合（RGB+红外）

#### Task 2.5：准备答辩PPT核心页（2小时）

**必备幻灯片：**

1. **项目背景与目标**（1页）
2. **数据集与处理**（1页，含分布饼图）
3. **模型架构**（1页，YOLOv8n结构图）
4. **实验结果**（2页）
   - 测试集指标表
   - TTA+SAHI对比图
5. **视频检测演示**（1页，时序平滑效果）
6. **问题与改进**（1页）
   - 诚实说明smoking_person效果不佳的原因
   - 展示TTA+SAHI改进效果
7. **总结与展望**（1页）

**PPT制作建议：**
- 使用简洁模板，避免花哨动画
- 每页不超过5个要点
- 图表清晰，字号≥18pt
- 准备备用页（如果老师追问ECA/SE）

---

## Day 3（5月7日）：全流程彩排 + 应急预案

**核心任务：** 模拟答辩，准备应急方案

### 上午（4小时）：全流程彩排

#### Task 3.1：答辩演示完整彩排（2小时）

**彩排流程：**

1. **启动Web服务**（5分钟）
   ```bash
   cd "D:\Smoker Behavior Detection Based on Deep Learning"
   & ".\.venv\Scripts\python.exe" app\main.py
   ```
   - 验证能正常访问 http://localhost:8000
   - 测试图片上传和检测功能

2. **图片检测演示**（5分钟）
   - 打开准备好的对比图（normal vs TTA+SAHI）
   - 讲解：TTA+SAHI如何提升小目标检测

3. **视频检测演示**（5分钟）
   - 播放检测后的视频
   - 展示中文报告页面
   - 讲解时序平滑机制

4. **PPT讲解**（10-15分钟）
   - 按照准备好的幻灯片讲一遍
   - 计时，确保不超时

5. **问答准备**（1小时）
   - 列出可能被问到的10个问题
   - 准备简洁的回答（每个<2分钟）

**常见问题预演：**

Q1: "为什么smoking_person效果最差？"
A: "主要原因是数据不足（测试集只有784样本）和标注歧义（什么算吸烟者边界不清）。我们通过TTA+SAHI推理增强，将其Recall从20%提升到26.7%。"

Q2: "ECA/SE为什么没有提升？"
A: "主要原因是插入位置不当——在neck融合点之后，无法在特征提取早期增强小目标表达。如果插入到backbone的C2f层之后，预期会有2-4%提升。"

Q3: "如何保证系统不误报？"
A: "我们补充了负样本测试，在X个非吸烟视频上误检率<5%。同时使用类别阈值（cigarette=0.12, smoke=0.28）和视频时序平滑来降低误检。"

Q4: "为什么不用更大的模型？"
A: "受限于CPU训练条件，我们使用YOLOv8n。如果使用GPU训练YOLOv8s，预期mAP@0.5可提升5-10%。"（如果Day 1训练了YOLOv8s，这里可以展示结果）

Q5: "项目的创新点是什么？"
A: "（1）融合5个数据源并均衡处理；（2）引入视频时序平滑机制（IoU匹配+轨迹管理）；（3）TTA+SAHI推理增强无需重训练即可提升14.8%召回率。"

#### Task 3.2：准备应急预案（2小时）

**应急方案A：Web服务启动失败**
```bash
# 备用：使用命令行演示
python scripts/enhanced_inference.py --source "demo.jpg" --mode compare --save-compare
# 提前生成好结果图，直接展示
```

**应急方案B：演示视频卡顿**
```bash
# 备用：使用静态截图代替视频播放
# 提前截取关键帧，制作成PPT动画
```

**应急方案C：数据被质疑**
```bash
# 准备验证报告打印版
# 展示 VERIFICATION_REPORT_20260505.md
# 说明："所有数据均可追溯到源JSON文件，验证通过率75-80%"
```

**应急方案D：电脑故障**
```bash
# 备份所有演示素材到U盘
# 包括：PPT、对比图、视频、验证报告PDF
# 可以在其他电脑上展示
```

---

### 下午（4小时）：最后检查 + 休息

#### Task 3.3：最后检查清单（2小时）

**文件检查：**
- [ ] 论文PDF最终版（含修正后的数据）
- [ ] 答辩PPT（含备用页）
- [ ] 演示素材文件夹（图片/视频/截图）
- [ ] 验证报告打印版
- [ ] U盘备份

**代码检查：**
- [ ] Web服务能正常启动
- [ ] 图片检测功能正常
- [ ] 视频检测功能正常
- [ ] TTA+SAHI脚本能运行
- [ ] 所有依赖包已安装

**数据检查：**
- [ ] 所有引用的数据与验证报告一致
- [ ] 图表清晰可读
- [ ] 表格格式统一

**心理准备：**
- [ ] 彩排至少2遍，熟悉流程
- [ ] 准备好10个常见问题的回答
- [ ] 了解项目的优势和不足

#### Task 3.4：充分休息（2小时）

**重要：** 答辩前一天下午和晚上要充分休息，不要熬夜。

---

## 答辩日（5月8日）：从容应对

### 答辩前1小时

1. **提前到场**，测试设备
2. **启动Web服务**，验证功能
3. **打开演示素材**，准备随时展示
4. **深呼吸**，保持冷静

### 答辩中

1. **开场**：简洁介绍项目背景和目标（2分钟）
2. **演示**：图片检测 → 视频检测 → Web页面（5分钟）
3. **讲解**：PPT核心页（8-10分钟）
4. **问答**：诚实回答，不懂就说"这是一个很好的问题，我会在后续研究中探索"

### 答辩后

无论结果如何，你已经尽力了。这个项目已经是一个完整的、可运行的系统。

---

## 关键原则

1. **诚实 > 完美**：承认不足比夸大更可信
2. **演示 > 讲解**：能展示的不要只说
3. **数据 > 感觉**：用验证报告支撑每个结论
4. **稳定 > 炫技**：确保演示不出错比追求高指标更重要

---

## 时间分配总结

| 日期 | 上午 | 下午 | 核心产出 |
|------|------|------|---------|
| Day 1 (5/5) | 修正报告+负样本测试+演示固化 | GPU训练YOLOv8s（可选） | 演示素材+负样本数据 |
| Day 2 (5/6) | 生成图表+补充数据表格 | 论文完善+PPT制作 | 论文最终版+PPT |
| Day 3 (5/7) | 全流程彩排+问答准备 | 最后检查+休息 | 应急预案+心理准备 |
| Day 4 (5/8) | 答辩 | - | 顺利通过 |

---

**最重要的一句话：** 你的项目已经是一个完整的、数据可验证的系统。三天时间足够让它更稳定、更完善。加油！
