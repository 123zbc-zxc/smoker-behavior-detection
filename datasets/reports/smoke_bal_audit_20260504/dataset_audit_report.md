# YOLO 数据集体检报告

- 数据集: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal`
- 图片总数: `18236`
- 标签文件总数: `18236`
- 标注框总数: `24624`
- 可疑项总数: `4913`

## 类别分布

- `0 cigarette`: `7021`
- `1 smoking_person`: `10562`
- `2 smoke`: `7041`

## 可疑项类型统计

- `small_smoking_person_box`: `3421`
- `huge_cigarette_box`: `919`
- `small_cigarette_review`: `225`
- `huge_box`: `163`
- `small_smoke_region`: `65`
- `tiny_box`: `51`
- `extreme_aspect_ratio`: `47`
- `box_crosses_image_edge`: `11`
- `overlapping_duplicate_box`: `8`
- `unsupported_image_format`: `2`
- `duplicate_label_line`: `1`

## 优先检查样本

- `error` `unsupported_image_format` val  line : `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\val\cg_124_jpg.rf.hUiRambsegxeZgxyT0HR.jpg`
- `error` `unsupported_image_format` test  line : `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\test\cg_107_jpg.rf.vx6aTUN0KitYbQUpHtIP.jpg`
- `warning` `tiny_box` train cigarette line 1: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\cg_255_jpg.rf.xgEhOtYoBl4n1QxECv4t.jpg`
- `warning` `tiny_box` train smoking_person line 1: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_255_jpg.rf.92c863bf62431a8d834c3d60ded9d3d7.jpg`
- `warning` `small_smoking_person_box` train smoking_person line 1: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_255_jpg.rf.92c863bf62431a8d834c3d60ded9d3d7.jpg`
- `warning` `tiny_box` train cigarette line 1: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\cg_235_jpg.rf.yThmKujLoqoKts5aI1yf.jpg`
- `warning` `tiny_box` train cigarette line 2: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\cg_out115_png_jpg.rf.76NC8O3j7hd19PMwnT74.jpg`
- `warning` `tiny_box` test cigarette line 3: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\test\cg_431_jpg.rf.3h6O77ddnWjNZAUVMq0Y.jpg`
- `warning` `tiny_box` train smoking_person line 3: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_431_jpg.rf.16be9598a98eeb6de5c2f7400b7dab66.jpg`
- `warning` `small_smoking_person_box` train smoking_person line 3: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_431_jpg.rf.16be9598a98eeb6de5c2f7400b7dab66.jpg`
- `warning` `tiny_box` train cigarette line 2: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\cg_255_jpg.rf.xgEhOtYoBl4n1QxECv4t.jpg`
- `warning` `tiny_box` train smoking_person line 2: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_255_jpg.rf.92c863bf62431a8d834c3d60ded9d3d7.jpg`
- `warning` `small_smoking_person_box` train smoking_person line 2: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_255_jpg.rf.92c863bf62431a8d834c3d60ded9d3d7.jpg`
- `warning` `tiny_box` train smoking_person line 2: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_out115_png_jpg.rf.35314bf45c6a408b718603e0fc75141f.jpg`
- `warning` `small_smoking_person_box` train smoking_person line 2: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_out115_png_jpg.rf.35314bf45c6a408b718603e0fc75141f.jpg`
- `warning` `tiny_box` train cigarette line 3: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\cg_361_jpg.rf.MQVW0fyWnoEifFcCoWEI.jpg`
- `warning` `tiny_box` train smoking_person line 3: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_361_jpg.rf.87b42533da10de675a879a43ffdb44d8.jpg`
- `warning` `small_smoking_person_box` train smoking_person line 3: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_361_jpg.rf.87b42533da10de675a879a43ffdb44d8.jpg`
- `warning` `tiny_box` train smoking_person line 3: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_361_jpg.rf.af43775372b952c3a00644be5be54cb2.jpg`
- `warning` `small_smoking_person_box` train smoking_person line 3: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_361_jpg.rf.af43775372b952c3a00644be5be54cb2.jpg`
- `warning` `tiny_box` train smoking_person line 1: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_bb0646_jpg.rf.923184dd8910e71de2ee34ad0bc84986.jpg`
- `warning` `small_smoking_person_box` train smoking_person line 1: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_bb0646_jpg.rf.923184dd8910e71de2ee34ad0bc84986.jpg`
- `warning` `tiny_box` train smoking_person line 1: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_bb0410_jpg.rf.71bfd5778bdaf42f252aecd730ca71c7.jpg`
- `warning` `small_smoking_person_box` train smoking_person line 1: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_bb0410_jpg.rf.71bfd5778bdaf42f252aecd730ca71c7.jpg`
- `warning` `tiny_box` train smoking_person line 3: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_out220_png_jpg.rf.95def9483783e4a5db44af9ffb1a2815.jpg`
- `warning` `small_smoking_person_box` train smoking_person line 3: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\rf_out220_png_jpg.rf.95def9483783e4a5db44af9ffb1a2815.jpg`
- `warning` `tiny_box` test smoking_person line 1: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\test\rf_c00223_jpg.rf.0ba3e7d6f587b503d6847fffbb892448.jpg`
- `warning` `small_smoking_person_box` test smoking_person line 1: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\test\rf_c00223_jpg.rf.0ba3e7d6f587b503d6847fffbb892448.jpg`
- `warning` `tiny_box` train cigarette line 1: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\cg_smoking_0361_jpg.rf.QZWOQM2xGUo5p2o0V6qf.jpg`
- `warning` `tiny_box` train cigarette line 3: `D:\Smoker Behavior Detection Based on Deep Learning\datasets\final\smoke_bal\images\train\cg_449_jpg.rf.OGWzLBZ8cbm06pclTsHP.jpg`

## 处理建议

- 先处理 `error` 和 `warning`，尤其是缺失文件、越界框、重复框、异常大/异常小框。
- `small_cigarette_review` 不一定是错标；香烟本来就是小目标，但这类样本最影响召回率，适合优先人工抽查。
- 不建议把全部可疑项都手工修完；答辩阶段优先修 80-150 张代表性 hard-case 即可。
