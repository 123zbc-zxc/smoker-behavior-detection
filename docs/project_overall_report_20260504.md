# 基于深度学习的吸烟行为检测系统项目总报告

**报告日期：** 2026-05-04  
**项目路径：** `D:\Smoker Behavior Detection Based on Deep Learning`  
**报告依据：** 本报告只引用项目中已经生成的真实配置、测试结果和审计文件，主要依据 `runs/reports/final_model_system_test_report_20260503.json`、`runs/reports/project_full_audit_20260503.json`、`runs/reports/system_smoke_test.json`、`configs/web_demo.json` 和自定义视频清单。

---

## 1. 项目总体结论

本项目已经达到“毕业设计演示系统 + 研究型原型”的完成度，具备完整的数据处理、模型训练、模型评估、图片检测、视频检测、时序平滑、Web 管理后台和测试报告输出能力。当前系统不应描述为生产级商用系统，但可以作为毕业论文、答辩演示和后续优化研究的完整原型。

当前项目状态来自审计文件：`graduation_demo_and_research_prototype_ready_not_production_ready`。

| 模块 | 当前完成度 |
|---|---:|
| 数据集处理流程 | 85-90% |
| 模型训练与评估 | 75-80% |
| Web Demo 系统 | 80-85% |
| 视频时序验证 | 75-80% |
| 论文/答辩材料 | 75-85% |
| 生产级可用性 | 55-60% |

总体判断：项目主链路已经打通，当前最适合的方向不是继续盲目训练，而是固定当前最佳模型、强化视频时序解释、完善论文表述和答辩演示材料。

---

## 2. 项目目标与系统功能

本项目目标是构建一个基于 YOLOv8 的吸烟行为检测系统，围绕三类吸烟相关目标进行检测：

| 类别 | 含义 | 当前作用 |
|---|---|---|
| `cigarette` | 香烟、雪茄等吸烟器物 | 最直接的吸烟证据，也是当前系统重点优化类别 |
| `smoking_person` | 正在吸烟或明显与吸烟行为相关的人 | 行为主体证据，但当前检测稳定性弱于 cigarette 和 smoke |
| `smoke` | 可见烟雾 | 辅助证据，受边界模糊、背景雾气、光照影响较大 |

系统已经具备以下功能：

1. 多来源数据集整理、转换、清洗、划分和校验。
2. YOLOv8n baseline、ECA、SE、hard fine-tune 等实验对比。
3. 图片检测，并将检测记录写入数据库。
4. 视频异步检测，支持任务队列、进度查看、输出视频和 JSON 报告。
5. 视频时序平滑，统计 raw detections、smoothed detections、stable tracks、temporal event hit 等指标。
6. Web 管理页面支持模型切换、参数配置、检测记录查看和视频报告查看。
7. 支持 PostgreSQL 持久化；测试环境也支持 SQLite。

---

## 3. 数据集与资源状态

### 3.1 主训练/测试数据集

当前 Web 配置中的主数据集为 `smoke_bal`：

| split | 图像数 |
|---|---:|
| train | 14416 |
| val | 1895 |
| test | 1925 |

类别为：`cigarette, smoking_person, smoke`。

根据项目审计结果，`datasets/final/smoke_bal` 结构有效，图像与标签匹配，没有发现非法 YOLO 标签。该数据集是当前项目最主要的平衡训练与评估数据集。

### 3.2 视频测试资源

项目已加入 HMDB51 smoke 视频数据用于时序验证，并建立了自定义视频测试区：`datasets/raw/custom_smoking_videos`。

当前自定义视频数量：4。

| 视频文件 | 帧数 | FPS | 分辨率 | 时长(s) | 编码 | OpenCV 可读 |
|---|---:|---:|---|---:|---|---|
| hello_shu_xiansheng_wang_baoqiang_smoking_aigei_20260503.mp4 | 378 | 30 | 1280x720 | 12.6 | h264 | true |
| friends_smoking_in_cafe_aigei_20260503.mp4 | 892 | 25 | 1280x720 | 35.68 | h264 | true |
| elderly_cigar_coughing_aigei_20260503.mp4 | 157 | 24 | 1280x720 | 6.5417 | h264 | true |
| wechat_video_20260504_103718_692.mp4 | 424 | 30 | 576x1280 | 14.1333 | hevc | true |

这些视频属于正样本或疑似正样本视频，适合用于演示和吸烟事件检测验证，但不能直接作为“误检率下降”的证明。误检率还需要非吸烟视频作为负样本测试。

---

## 4. 当前最佳模型

### 4.1 最终固定模型

当前 Web 默认模型已经固定为：`YOLOv8n 640px Hard Fine-tune (20260502)`。

权重路径：`runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt`

旧冠军对照模型：`runs/imported/smoker_weights_20260429/best.pt`

选择 hard_finetune 作为默认模型的原因是：它在 cigarette 召回率和完整视频时序稳定性上更适合当前项目目标。需要注意，hard_finetune 并不是所有指标全面超过旧冠军，尤其 mAP@0.5:0.95 略低，因此论文中必须客观表述。

### 4.2 测试集指标对比

| 指标 | hard_finetune | old_champion | 差值 hard-old |
|---|---:|---:|---:|
| Precision | 0.5405 | 0.5387 | 0.0018 |
| Recall | 0.6900 | 0.6859 | 0.0041 |
| mAP@0.5 | 0.5604 | 0.5613 | -0.0009 |
| mAP@0.5:0.95 | 0.3559 | 0.3597 | -0.0038 |
| cigarette Recall | 0.7472 | 0.7212 | 0.0260 |
| cigarette mAP@0.5 | 0.4967 | 0.4920 | 0.0048 |

结论：hard_finetune 的 cigarette Recall 从 `0.7212` 提高到 `0.7472`，更适合“尽可能发现吸烟相关目标”的需求。但 mAP@0.5:0.95 从 `0.3597` 降到 `0.3559`，说明它不是全指标最优。

### 4.3 三类目标表现

hard_finetune 各类别表现：

| 类别 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| cigarette | 0.4456 | 0.7472 | 0.4967 | 0.2989 |
| smoking_person | 0.4331 | 0.5969 | 0.3962 | 0.2494 |
| smoke | 0.7429 | 0.7258 | 0.7883 | 0.5195 |

从结果看，`smoking_person` 是三类中较弱的类别。原因主要包括：人体框标注标准更难统一，吸烟人经常只露出脸部或上半身，且模型容易把局部嘴部/手部区域与 smoking_person 混淆。因此系统最终判断吸烟事件时，不应只依赖 `smoking_person` 框，而应综合 `cigarette`、`smoke` 和视频时序稳定性。

---

## 5. 视频时序验证结果

### 5.1 109 个 HMDB51 smoke 视频验证

| 指标 | hard_finetune | old_champion | 差值 hard-old |
|---|---:|---:|---:|
| 视频数量 | 109 | 109 | - |
| 总帧数 | 16291 | 16291 | - |
| temporal_event_hit_videos | 99 | 97 | 2 |
| raw_hit_frames | 7156 | 6893 | 263 |
| smoothed_hit_frames | 7054 | 6720 | 334 |
| smoothed_hit_frame_ratio | 0.433 | 0.4125 | 0.0205 |
| stable_track_count | 429 | 388 | 41 |

结论：hard_finetune 在 109 个 HMDB51 smoke 视频上比旧冠军多命中 2 个视频事件，平滑后命中帧增加 334 帧，稳定轨迹数从 388 增加到 429。这说明 hard_finetune 更适合作为当前 Web 视频检测默认模型。

### 5.2 时序机制说明

视频检测不是简单地逐帧显示所有框，而是加入了时序规则：

- 相邻帧使用 IoU 进行轨迹关联。
- 目标连续命中达到 `stable_hits=3` 后才认为形成稳定轨迹。
- 对短时遮挡或短时漏检允许 `bridge_frames=2` 的补帧。
- 输出 raw detections 和 smoothed detections，便于比较单帧检测与时序平滑的差异。

该机制的意义是：降低视频中检测框闪烁，让吸烟行为判断更接近“事件检测”，而不是只看单帧框。

---

## 6. 阈值与运行参数

当前推荐运行参数来自视频阈值与时序参数搜索：

| 参数 | 当前值 |
|---|---:|
| 默认 conf | 0.12 |
| 默认 IoU | 0.45 |
| imgsz | 640 |
| cigarette 阈值 | 0.12 |
| smoking_person 阈值 | 0.22 |
| smoke 阈值 | 0.28 |
| match_iou | 0.25 |
| stable_hits | 3 |
| bridge_frames | 2 |
| stale_frames | 5 |

参数搜索结果：

- 推荐参数组：`recall_plus`
- 测试视频数量：20
- temporal_event_hit：17/20
- temporal_event_hit_rate：0.85
- smoothed_hit_frame_ratio：0.2997
- stable_track_count：69

解释：当前参数偏召回，主要目的是保留小目标 cigarette 候选框，再通过视频时序规则过滤短时噪声。该参数适合视频检测和项目演示。对于单张图片展示，如果误检过多，可以临时把 conf 调到 0.25 或 0.30，使画面更干净。

---

## 7. Web Demo 与数据库状态

系统 Smoke Test 最近一次结果：

| 项目 | 结果 |
|---|---|
| health status | ok |
| database backend | sqlite |
| weights_exists | true |
| index_status | 200 |
| video_task_status | completed |
| video_report_status | 200 |
| invalid_upload_status | 400 |
| models_count | 7 |
| default_conf | 0.12 |
| default_iou | 0.45 |
| default_imgsz | 640 |

说明：Smoke Test 使用的是测试环境数据库，报告中显示 `sqlite` 是正常的。用户实际运行时可以通过 `SMOKER_DB_URL` 使用 PostgreSQL。此前 Web 页面已经能识别 PostgreSQL，并显示数据库为 `postgresql`。

Web Demo 主要能力：

1. 首页展示当前模型、数据库、实验对比和推荐运行参数。
2. 图片检测支持上传、推理、结果图回显、目标列表和数据库记录。
3. 视频检测支持异步任务、进度轮询、输出视频、中文报告和 JSON 报告。
4. 模型页支持多个权重切换，默认使用 hard_finetune。
5. 参数页支持管理默认 conf、IoU、imgsz 和上传大小限制。

---

## 8. 当前项目优势

1. **工程链路完整**：从数据处理、训练、评估到 Web 展示均已具备可运行脚本。
2. **模型选择有依据**：不是主观选择模型，而是依据 test split、cigarette Recall 和 HMDB51 视频时序结果固定 hard_finetune。
3. **视频时序验证是亮点**：相比普通图片检测项目，本项目增加了连续命中帧、稳定轨迹、时序命中等视频层指标。
4. **Web Demo 可展示性强**：具备图片检测、视频检测、数据库记录、模型切换和中文报告页面，适合答辩演示。
5. **结果表述较稳健**：报告明确指出 hard_finetune 并非所有指标全面领先，也说明正样本视频搜索不能证明误检率下降，避免夸大。

---

## 9. 当前项目问题与风险

### 9.1 模型层面

- mAP@0.5 约为 0.56，仍有明显提升空间。
- mAP@0.5:0.95 约为 0.356，说明定位精度仍不够强。
- `smoking_person` 类别较弱，Recall 约为 `0.5969`，容易出现不框人、局部误框或类别混淆。
- cigarette 是小目标，低清晰度、遮挡、手指接触、雪茄/电子烟等场景仍可能漏检或框偏大。
- 低 conf 参数会提高召回，但也会增加重复框和误检。

### 9.2 数据层面

- 数据集结构有效，但标注质量仍可能不一致。
- 手工 hard-case 修框方案已取消，因此短期内不再通过人工修框提升数据质量。
- 视频时序验证主要使用正样本，缺少系统化非吸烟视频误检测试。

### 9.3 系统层面

- Web Demo 已适合演示，但不是生产级系统。
- PostgreSQL 可以使用，但仍建议答辩前再做一次本机 PostgreSQL 启动验证。
- 仓库中仍有较多原始压缩包、runs 输出、tmp 文件和大权重文件，最终提交或打包前需要清理。

---

## 10. 论文与答辩建议表述

推荐表述：

> 本项目最终采用 YOLOv8n 640px hard fine-tune 模型作为默认检测模型。该模型在测试集上提升了 cigarette 类别召回率，并在 HMDB51 smoke 视频验证中取得更好的时序命中和稳定轨迹表现。由于吸烟行为在视频中具有连续性，系统进一步引入基于 IoU 关联的时序平滑策略，通过连续命中、短时补帧和稳定轨迹判断减少单帧检测闪烁，使检测结果更适合视频场景下的吸烟事件判断。

需要避免的表述：

- 不要说“模型精准率达到 90%”。当前没有这个证据。
- 不要说“hard 模型所有指标全面优于旧模型”。它的 mAP@0.5:0.95 略低。
- 不要说“阈值搜索证明误检率下降”。该搜索基于正样本视频，只能说明视频命中和连续性。
- 不要说“系统已经达到生产级应用”。目前更准确的是毕业项目演示系统和研究型原型。

---

## 11. 后续工作建议

按性价比排序：

1. **答辩演示流程固化**：准备 1 张图片、1 段短视频、1 个 Web 报告页面，保证稳定演示。
2. **补充非吸烟视频误检测试**：找 10-20 个无吸烟视频，测试 false positive，增强报告可信度。
3. **整理项目目录**：最终提交前清理无关压缩包、临时文件和重复 runs 输出。
4. **补充论文截图**：加入 Web 首页、图片检测结果、视频报告、模型对比表、时序参数表。
5. **如果继续优化模型**：不要盲目加 epoch，优先做少量高质量 hard-case 标注或引入负样本误检测试。

---

## 12. 最终结论

本项目已经形成完整的吸烟行为检测系统原型。当前最佳策略是使用 `YOLOv8n 640px Hard Fine-tune (20260502)` 作为默认模型，结合 `conf=0.12`、`IoU=0.45`、类别阈值和视频时序平滑进行检测。系统在 cigarette 召回和视频时序稳定性方面具备明确改进依据，但仍存在小目标定位、smoking_person 类别不稳定、负样本误检验证不足等问题。

因此，项目当前最适合定位为：**面向毕业设计答辩的可运行吸烟行为检测与视频时序验证系统**。
