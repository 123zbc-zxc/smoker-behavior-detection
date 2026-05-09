# 数据资产盘点报告

- 扫描根目录: `D:\Smoker Behavior Detection Based on Deep Learning\datasets`
- 图片文件总数: `91242`
- 视频文件总数: `113`
- 压缩包总数: `5`
- 检测到的数据集根目录: `13`
- YOLO 有效框总数: `123528`

## 分阶段资产

- `raw`: 图片 `834`, 视频 `113`, 压缩包 `3`, 数据集根 `1`
- `interim`: 图片 `29996`, 视频 `0`, 压缩包 `2`, 数据集根 `8`
- `final`: 图片 `60412`, 视频 `0`, 压缩包 `0`, 数据集根 `4`

## 主要 YOLO 数据集

- `datasets\final\smoking_yolo_3cls_full`: 图片 `28915`, 标签 `28915`, 有效框 `39641`, 空标签 `0`, 无效行 `0`, 类别 `{'2:2': 7041, '0:0': 7021, '1:1': 25579}`
- `datasets\interim\roboflow_remap`: 图片 `18269`, 标签 `18269`, 有效框 `25579`, 空标签 `48`, 无效行 `6`, 类别 `{'1:1': 25579}`
- `datasets\final\smoke_bal`: 图片 `18236`, 标签 `18236`, 有效框 `24624`, 空标签 `0`, 无效行 `0`, 类别 `{'2:2': 7041, '0:0': 7021, '1:1': 10562}`
- `datasets\final\smoking_yolo_3cls_balanced`: 图片 `11940`, 标签 `11940`, 有效框 `15762`, 空标签 `0`, 无效行 `0`, 类别 `{'2:2': 7041, '0:0': 7021, '1:1': 1700}`
- `datasets\interim\cigarette_yolo`: 图片 `4663`, 标签 `4663`, 有效框 `7021`, 空标签 `0`, 无效行 `0`, 类别 `{'0:0': 7021}`
- `datasets\interim\smoke_yolo`: 图片 `4898`, 标签 `4898`, 有效框 `5770`, 空标签 `0`, 无效行 `0`, 类别 `{'2:2': 5770}`
- `datasets\final\smoking_yolo_3cls`: 图片 `1321`, 标签 `1711`, 有效框 `3309`, 空标签 `0`, 无效行 `0`, 类别 `{'1:1': 3067, '2:2': 242}`
- `datasets\interim\aistudio_yolo`: 图片 `783`, 标签 `783`, 有效框 `816`, 空标签 `0`, 无效行 `0`, 类别 `{'2:2': 816}`
- `datasets\interim\smoke_legacy_yolo`: 图片 `356`, 标签 `356`, 有效框 `455`, 空标签 `0`, 无效行 `0`, 类别 `{'2:2': 455}`
- `datasets\interim\roboflow_cigarette_smoke_detection_v4_yolo3cls`: 图片 `413`, 标签 `413`, 有效框 `388`, 空标签 `25`, 无效行 `0`, 类别 `{'0:cigarette': 388}`
- `datasets\interim\hardcase_labeling_workspace_20260503`: 图片 `120`, 标签 `120`, 有效框 `89`, 空标签 `70`, 无效行 `0`, 类别 `{'0:cigarette': 71, '2:smoke': 18}`
- `datasets\raw\roboflow_cigarette_smoke_detection_v4_20260504\extracted`: 图片 `51`, 标签 `46`, 有效框 `41`, 空标签 `3`, 无效行 `2`, 类别 `{'0:Cigar in hand': 20, '1:Cigar near mouth': 21}`
- `datasets\interim\hmdb51_smoke_hardcase_pseudo`: 图片 `90`, 标签 `180`, 有效框 `33`, 空标签 `128`, 无效行 `33`, 类别 `{'0:cigarette': 33}`

## 重复图片估计

- 未启用图片 hash。需要去重时运行 `--hash-images`。

## 结论

- 当前项目不是没有数据，而是存在多阶段重复资产、原始归档、训练子集和候选伪标签混在一起的问题。
- 下一轮 Google 训练应优先使用 `datasets/final/smoke_bal` 加确认过的 `datasets/interim/*_yolo3cls`，不要把 raw 目录直接混入训练。
- `hmdb51_smoke` 和自定义视频主要用于视频时序验证/抽帧候选，不是天然 YOLO 检测训练集。
