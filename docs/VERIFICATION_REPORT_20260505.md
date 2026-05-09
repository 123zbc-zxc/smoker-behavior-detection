# 分析报告验证结果与勘误

**验证日期：** 2026-05-05  
**验证方法：** 深入代码和数据文件交叉验证

---

## 验证总结

**总体准确率：** 75-80%  
**验证状态：** 核心结论正确，部分具体数据需修正

---

## 一、已验证正确的核心结论

### 1. Smoke 和 Smoking_Person 分析报告

✅ **完全验证通过的数据：**

| 数据项 | 报告值 | 验证源文件 | 状态 |
|--------|--------|-----------|------|
| 测试集cigarette数量 | 1077 | final_model_system_test_report_20260503.json | ✓ |
| 测试集smoking_person数量 | 784 | final_model_system_test_report_20260503.json | ✓ |
| 测试集smoke数量 | 1043 | final_model_system_test_report_20260503.json | ✓ |
| 原始smoking_person样本 | 7041 | balanced_dataset_report.json | ✓ |
| 原始smoke样本 | 25579 | balanced_dataset_report.json | ✓ |
| cigarette置信度阈值 | 0.12 | final_model_system_test_report_20260503.json | ✓ |
| smoking_person置信度阈值 | 0.22 | final_model_system_test_report_20260503.json | ✓ |
| smoke置信度阈值 | 0.28 | final_model_system_test_report_20260503.json | ✓ |
| smoking_person Precision | 0.4331 | final_model_system_test_report_20260503.json | ✓ |
| smoking_person Recall | 0.5969 | final_model_system_test_report_20260503.json | ✓ |
| smoking_person mAP@0.5 | 0.3962 | final_model_system_test_report_20260503.json | ✓ |
| smoke Precision | 0.7429 | final_model_system_test_report_20260503.json | ✓ |
| smoke Recall | 0.7258 | final_model_system_test_report_20260503.json | ✓ |
| smoke mAP@0.5 | 0.7883 | final_model_system_test_report_20260503.json | ✓ |
| 困难样本空标签数 | 70 | hardcase_labeling_workspace_20260503/annotation_summary.json | ✓ |

### 2. TTA+SAHI 验证报告

✅ **完全验证通过的数据：**

| 指标 | normal | tta+sahi | 提升 | 验证源 |
|------|--------|----------|------|--------|
| cigarette Recall | 0.7105 | 0.8158 | +14.8% | TTA_SAHI_QUICKSTART.md | ✓ |
| smoking_person Recall | 0.2000 | 0.2667 | +33.4% | TTA_SAHI_QUICKSTART.md | ✓ |
| 总体 Recall | 0.6049 | 0.6790 | +12.2% | TTA_SAHI_QUICKSTART.md | ✓ |

### 3. 注意力机制失效分析

✅ **验证通过的技术分析：**
- ECA/SE代码实现正确（已检查源码）
- ECA插入位置在neck融合点之后（已验证YAML文件）
- SE使用延迟初始化（已验证代码）
- 插入位置不适合小目标检测（架构分析正确）

---

## 二、需要修正的数据

### 修正1：极小目标数量

**报告原文：**
> "151个tiny目标面积<0.0005"

**验证结果：**
- 实际数据：**51个tiny boxes**，阈值为**0.00025**（不是0.0005）
- 来源：`datasets/reports/smoke_bal_audit_20260504/dataset_audit_summary.json`

**修正后：**
> "51个tiny目标面积<0.00025（相当于640px图中约10px²）"

---

### 修正2：视频帧检测率

**报告原文：**
> "视频检测中只有30%的帧有检测"

**验证结果：**
- 实际数据：**43.3%**（smoothed_hit_frame_ratio = 0.433）
- 来源：`runs/reports/final_model_system_test_report_20260503.json`

**修正后：**
> "视频检测中约43%的帧有检测（smoothed_hit_frame_ratio=0.433）"

---

### 修正3：TTA+SAHI提升百分比表述

**报告原文：**
> "cigarette Recall提升10.5%"

**验证结果：**
- 绝对提升：0.8158 - 0.7105 = 0.1053（10.53个百分点）
- 相对提升：(0.8158 - 0.7105) / 0.7105 = 14.8%

**修正后：**
> "cigarette Recall从0.7105提升到0.8158（绝对提升10.5个百分点，相对提升14.8%）"

---

## 三、无法验证的声明（需要标注为推测）

### 1. SE模块第1轮epoch的mAP=0.004

**报告原文：**
> "SE在第1轮就几乎完全失效（mAP=0.004）"

**验证结果：**
- ❌ **无法验证** - 未找到SE模型的训练日志
- 可能的日志位置都不存在或未保存

**建议修正：**
> "根据项目记录，SE模型在训练早期表现不佳（具体训练日志未保存），推测可能与延迟初始化导致的优化器参数不匹配有关。"

---

### 2. ECA mAP@0.5 = 0.494 vs 基线 0.520

**报告原文：**
> "YOLOv8n + ECA: mAP@0.5 = 0.494"

**验证结果：**
- ❌ **无法直接验证** - `final_model_system_test_report_20260503.json`只包含hard_finetune vs old_champion对比
- 未找到ECA模型的独立测试结果JSON

**建议修正：**
> "根据项目实验记录，ECA模型的mAP@0.5约为0.494，低于基线的0.520（注：具体测试报告未在当前审计文件中找到）。"

---

## 四、架构分析的验证

### ECA/SE插入位置验证

✅ **完全正确** - 已验证 `models/yolov8n_eca.yaml`：

```yaml
# Line 21-23
- [[-1, 6], 1, Concat, [1]]      # 拼接
- [-1, 1, ECA, [768]]             # ECA在拼接后
- [-1, 3, C2f, [512]]

# Line 26-28
- [[-1, 4], 1, Concat, [1]]
- [-1, 1, ECA, [384]]
- [-1, 3, C2f, [256]]
```

**结论：** 报告中关于"插入位置在neck融合点之后，不适合小目标检测"的分析是正确的。

---

## 五、数据来源清单

所有已验证数据的源文件：

1. `runs/reports/final_model_system_test_report_20260503.json` - 模型性能指标
2. `datasets/reports/balanced_dataset_report.json` - 数据集分布
3. `datasets/reports/smoke_bal_audit_20260504/dataset_audit_summary.json` - 数据集审计
4. `datasets/interim/hardcase_labeling_workspace_20260503/annotation_summary.json` - 困难样本统计
5. `models/yolov8n_eca.yaml` - ECA模型架构
6. `models/modules/eca.py` - ECA实现代码
7. `models/modules/se.py` - SE实现代码
8. `docs/TTA_SAHI_QUICKSTART.md` - TTA+SAHI验证结果

---

## 六、总体评估

### 核心结论的可靠性

| 报告 | 核心结论 | 可靠性 |
|------|---------|--------|
| Smoke/Smoking_Person分析 | smoking_person数据不足、smoke阈值过高 | **高** - 所有关键数据已验证 |
| TTA+SAHI效果 | 提升recall但降低precision | **高** - 实测数据完整 |
| 注意力机制失效 | 插入位置不当、模型容量不足 | **中高** - 架构分析正确，部分训练数据缺失 |

### 需要修正的内容

1. **立即修正：**
   - 极小目标数量：151 → 51
   - 视频帧检测率：30% → 43.3%
   - TTA+SAHI提升表述：明确绝对vs相对提升

2. **需要标注为推测：**
   - SE epoch 1 mAP=0.004（无训练日志）
   - ECA mAP@0.5 = 0.494（无独立测试报告）

3. **保持不变：**
   - 所有smoke/smoking_person性能数据
   - 数据集分布统计
   - 置信度阈值
   - TTA+SAHI实测数据
   - 架构分析结论

---

## 七、建议

### 对于答辩和论文

1. **可以放心引用的数据：**
   - 所有测试集性能指标（P/R/mAP）
   - 数据集分布统计
   - TTA+SAHI提升效果
   - 架构分析结论

2. **需要谨慎表述的内容：**
   - ECA/SE具体训练过程（缺少完整日志）
   - 使用"根据实验记录"而非"训练日志显示"

3. **建议补充的验证：**
   - 如果可能，重新运行ECA/SE模型的val.py获取完整指标
   - 或明确标注"历史实验数据，详细日志未保存"

---

## 八、修正后的关键数据表

### 数据集特征（已验证）

| 特征 | 数值 | 来源 |
|------|------|------|
| 训练集图片数 | 14416 | balanced_dataset_report.json |
| 测试集图片数 | 1925 | final_model_system_test_report_20260503.json |
| cigarette测试样本 | 1077 | final_model_system_test_report_20260503.json |
| smoking_person测试样本 | 784 | final_model_system_test_report_20260503.json |
| smoke测试样本 | 1043 | final_model_system_test_report_20260503.json |
| **tiny目标数量** | **51** | smoke_bal_audit_20260504/dataset_audit_summary.json |
| **tiny阈值** | **0.00025** | smoke_bal_audit_20260504/dataset_audit_summary.json |

### 模型性能（已验证）

| 类别 | Precision | Recall | mAP@0.5 | 置信度阈值 |
|------|-----------|--------|----------|-----------|
| cigarette | 0.4456 | 0.7472 | 0.4967 | 0.12 |
| smoking_person | 0.4331 | 0.5969 | 0.3962 | 0.22 |
| smoke | 0.7429 | 0.7258 | 0.7883 | 0.28 |

### 视频检测（已验证）

| 指标 | 数值 | 来源 |
|------|------|------|
| 测试视频数 | 109 | final_model_system_test_report_20260503.json |
| 事件命中视频 | 99 | final_model_system_test_report_20260503.json |
| **平滑命中帧比例** | **0.433 (43.3%)** | final_model_system_test_report_20260503.json |
| 稳定轨迹数 | 429 | final_model_system_test_report_20260503.json |

---

**验证结论：** 报告的核心分析和结论是可靠的，但部分具体数据需要修正。建议在答辩前使用修正后的数据，并对无法验证的声明标注为"根据实验记录"而非"训练日志显示"。
