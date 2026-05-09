# TTA + SAHI 增强推理快速指南

## 适用场景

TTA + SAHI 是推理端增强方案，不重新训练、不改权重。它适合答辩演示、图片检测和离线分析，尤其适合香烟这类小目标；不适合实时视频，因为速度明显变慢。

## 已验证结论

本机已在 `smoke_bal` test split 抽样 50 张图片上验证：

| 模式 | Precision | Recall | F1 | cigarette Recall | smoking_person Recall | smoke Recall | 平均耗时/图 |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal | 0.5698 | 0.6049 | 0.5868 | 0.7105 | 0.2000 | 0.6786 | 0.08s |
| sahi | 0.5506 | 0.6049 | 0.5765 | 0.7105 | 0.2000 | 0.6786 | 0.17s |
| tta+sahi | 0.4741 | 0.6790 | 0.5584 | 0.8158 | 0.2667 | 0.7143 | 0.61s |

结论：`tta+sahi` 明显提高 Recall，尤其 cigarette Recall 从 0.7105 提高到 0.8158；代价是 Precision 和速度下降。因此它适合“找得更全”的答辩演示，不适合作为默认实时视频方案。

## 单张图片对比

```powershell
cd "D:\Smoker Behavior Detection Based on Deep Learning"
& ".\.venv\Scripts\python.exe" scripts\enhanced_inference.py --source "datasets\final\smoke_bal\images\test\cg_000011_jpg.rf.cgWt4mjkpgoh49SYIuSm.jpg" --mode compare --save-compare --output output\enhanced_inference_verify
```

输出对比图在：

```text
output/enhanced_inference_verify/
```

## 答辩演示推荐命令

```powershell
cd "D:\Smoker Behavior Detection Based on Deep Learning"
& ".\.venv\Scripts\python.exe" scripts\enhanced_inference.py --source "你的图片.jpg" --mode tta+sahi --output output\enhanced_inference_demo --json
```

如果误检太多，改用：

```powershell
& ".\.venv\Scripts\python.exe" scripts\enhanced_inference.py --source "你的图片.jpg" --mode sahi --conf 0.12 --output output\enhanced_inference_demo --json
```

## 小样本定量验证

```powershell
cd "D:\Smoker Behavior Detection Based on Deep Learning"
& ".\.venv\Scripts\python.exe" scripts\eval_enhanced.py --sample 50 --modes normal sahi tta+sahi --output output\enhanced_eval\verify_sample50.json
```

## 论文/答辩表述

可以说：

> 在不重新训练模型的情况下，系统补充了 TTA 与 SAHI 切片推理作为增强推理模式。抽样测试显示，TTA+SAHI 能提高整体 Recall 和 cigarette 类别 Recall，说明切片推理对小目标香烟有帮助。但该方法会降低 Precision 并增加推理耗时，因此项目中仅作为离线增强和答辩演示模式，Web 视频实时检测仍使用原始推理加时序平滑方案。

不能说：

- TTA+SAHI 一定全面提高模型性能。
- TTA+SAHI 能把精准率提高到 90%。
- TTA+SAHI 适合实时视频。
