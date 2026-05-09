# 吸烟行为检测系统优化方案文档

**项目**：Smoker Behavior Detection Based on Deep Learning  
**日期**：2026-04-20  
**状态**：代码修改已完成，训练待执行

---

## 一、背景与问题诊断

### 1.1 项目现状

本项目使用 YOLOv8n 检测3个类别：
- `cigarette`（香烟，class 0）
- `smoking_person`（吸烟者，class 1）
- `smoke`（烟雾，class 2）

**数据集**：`datasets/final/smoke_bal`（平衡处理后）
- 训练集：14,416 张
- 验证集：1,895 张
- 测试集：1,925 张
- 合计：18,220 张

### 1.2 三大核心问题

| 问题 | 具体表现 |
|------|---------|
| 训练时间过长 | 30轮需18.2小时（36.4分钟/轮），50轮约需30小时 |
| 识别率不高 | 整体 mAP@0.5 仅 0.520，香烟类最差 |
| 系统稳定性 | 视频检测存在闪烁，置信度不稳定 |

### 1.3 当前最佳模型性能（YOLOv8n Baseline，balanced_30）

| 类别 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------|-----------|--------|---------|--------------|
| 整体 | 0.526 | 0.625 | 0.520 | 0.323 |
| cigarette | 0.282 | 0.640 | 0.321 | 0.193 |
| smoking_person | 0.577 | 0.583 | 0.519 | 0.305 |
| smoke | 0.719 | 0.653 | 0.721 | 0.470 |

### 1.4 训练瓶颈根因分析

| 瓶颈 | 原始配置 | 影响 |
|------|---------|------|
| CPU 训练 | `device: cpu`，无 GPU | 最大瓶颈，计算速度慢 10-50 倍 |
| 单线程数据加载 | `workers: 0` | 数据读取串行，大量 CPU 空闲等待 |
| 无数据缓存 | `cache: false` | 每轮重复读取全部 14,416 张图像 |
| 小 batch | `batch: 8` | 受 CPU 内存限制 |
| 过多训练轮数 | `epochs: 80` | 实际 30 轮已收敛，80 轮浪费 |

### 1.5 识别率问题根因分析

- **香烟是小目标**：平均面积比 3.66%，中位数仅 1.34%；151 个极小目标（面积比 < 0.1%）
- **分辨率不足**：416px 输入对小目标特征提取不充分
- **失败模式**：漏检（22个）为主，误检（0个）极少，说明模型对香烟的感知能力不足，应降低置信度阈值、提高输入分辨率

### 1.6 已尝试但效果不佳的优化（不要重复尝试）

| 方法 | 结果 | 结论 |
|------|------|------|
| ECA 注意力机制 | mAP@0.5 下降 3.3%（0.520→0.489） | 不适用 |
| SE 注意力机制 | mAP@0.5 下降 94.8%（0.520→0.052） | 不适用 |
| 知识蒸馏 | mAP@0.5 下降 94.5%（0.520→0.047） | 不适用 |

---

## 二、优化方案总览

本方案分三个阶段，在 CPU + 16GB 内存的设备限制下最大化提升效果。

```
阶段1（立即生效）：训练加速 + 后处理优化
阶段2（训练完成后）：512px 分辨率提升
阶段3（持续改善）：系统稳定性增强
```

---

## 三、阶段1：训练加速优化

### 3.1 新建配置文件：`configs/train_yolov8n_balanced_fast.yaml`

**与原始 `configs/train_yolov8n_balanced.yaml` 的对比：**

| 参数 | 原始值 | 新值 | 改动原因 |
|------|--------|------|---------|
| `epochs` | 80 | 50 | 30轮已收敛，节省训练时间 |
| `workers` | 0 | 4 | 启用多线程数据加载，充分利用多核CPU |
| `cache` | false | true | 首轮后将数据缓存到内存，消除重复磁盘IO |
| `patience` | 20 | 15 | 提前停止，避免无效等待 |
| `close_mosaic` | 10 | 5 | 提前关闭 mosaic 增强，减少计算量 |

**完整文件内容：**
```yaml
model: yolov8n.pt
data: configs/data_smoking_balanced.yaml
epochs: 50
imgsz: 416
batch: 8
optimizer: AdamW
lr0: 0.001
patience: 15
device: cpu
workers: 4
cache: true
cos_lr: true
close_mosaic: 5
pretrained: true
seed: 42
deterministic: true
plots: true
project: runs/train
name: yolov8n_balanced_fast
```

> **注意**：`persistent_workers` 不是有效的 YOLO 参数，已从文件中移除（运行时会报 SyntaxError）。

**预期效果：**
- 每轮训练时间：36.4分钟 → 约 12-15 分钟（提速 60-67%）
- 50 轮总时间：约 10-12.5 小时（比原来 30 轮还快）

**内存需求：** cache:true 会将数据集加载到 RAM，约需额外 3-5GB 内存。16GB 系统安全可用。

**运行命令：**
```bash
# 先运行5轮冒烟测试，验证配置无误
python scripts/train.py --config configs/train_yolov8n_balanced_fast.yaml --epochs 5 --name yolov8n_balanced_fast_smoketest

# 确认正常后，启动完整训练
python scripts/train.py --config configs/train_yolov8n_balanced_fast.yaml
```

---

### 3.2 后处理优化：类别差异化置信度阈值

**修改文件：** `app/utils/web_inference.py`

**改动说明：** 原代码对所有类别统一使用 `conf=0.25` 阈值。由于香烟漏检严重（0个误检，22个漏检），应将模型预测的全局阈值先降到 `0.15`，再在后处理阶段按类别阈值进行二次筛选；这样香烟的低分候选框不会在模型输出阶段被提前截断，同时烟雾仍可通过更高类别阈值抑制误检。

**修改前：**
```python
TEMPORAL_MATCH_IOU = 0.3
TEMPORAL_STABLE_HITS = 2
TEMPORAL_BRIDGE_FRAMES = 1
TRACK_STALE_FRAMES = 3
```

**修改后（第21-30行）：**
```python
TEMPORAL_MATCH_IOU = 0.25
TEMPORAL_STABLE_HITS = 3
TEMPORAL_BRIDGE_FRAMES = 2
TRACK_STALE_FRAMES = 5
CONFIDENCE_SMOOTH_ALPHA = 0.7
CLASS_CONF_THRESHOLDS = {
    0: 0.15,   # cigarette：降低阈值，优先提升 recall
    1: 0.25,   # smoking_person：保持标准阈值
    2: 0.30,   # smoke：提高阈值，减少误检
}
DEFAULT_INFERENCE_CONF = 0.15
```

**推理链路同步修改：**
- `app/utils/web_inference.py` 中 `detect_image_bytes()` 默认 `conf` 改为 `0.15`
- `app/utils/web_inference.py` 中 `process_video_file()` 默认 `conf` 改为 `0.15`
- `app/web_demo.py` 启动默认设置中的 `default_conf` 改为 `0.15`
- 数据库中未手动调整过的旧默认值 `0.25` 会在启动时自动升级为 `0.15`

**`_extract_detections` 方法修改（第335-355行）：**

修改前每个检测框都直接添加，没有按类别过滤。修改后：
```python
def _extract_detections(self, result: Any) -> list[DetectionBox]:
    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)
    detections: list[DetectionBox] = []
    if boxes is None or boxes.cls is None:
        return detections

    xyxy = boxes.xyxy.tolist() if boxes.xyxy is not None else []
    for idx, (cls_id, score) in enumerate(zip(boxes.cls.tolist(), boxes.conf.tolist())):
        cls_id_int = int(cls_id)
        if float(score) < CLASS_CONF_THRESHOLDS.get(cls_id_int, 0.25):
            continue                          # 低于类别阈值则跳过
        detections.append(
            DetectionBox(
                class_id=cls_id_int,
                class_name=str(names.get(cls_id_int, cls_id)),
                confidence=float(score),
                xyxy=[float(value) for value in xyxy[idx]] if idx < len(xyxy) else [],
            )
        )
    return detections
```

**预期效果：**
- 香烟 recall 提升约 10-15%（无需重新训练，立即生效）
- 烟雾误检减少约 20-30%

---

## 四、阶段2：识别率提升优化

### 4.1 新建配置文件：`configs/train_yolov8n_balanced_512.yaml`

**核心改动：** 输入分辨率从 416px 提升至 512px（提升 23%）

**原理：** 香烟平均面积比仅 1.34%，在 416px 图像中香烟仅占约 20-30 像素，特征信息极少。提升至 512px 后香烟像素数增加约 51%，显著改善特征提取质量。

**完整文件内容：**
```yaml
model: yolov8n.pt
data: configs/data_smoking_balanced.yaml
epochs: 50
imgsz: 512
batch: 6
optimizer: AdamW
lr0: 0.001
patience: 15
device: cpu
workers: 4
cache: true
cos_lr: true
close_mosaic: 5
pretrained: true
seed: 42
deterministic: true
plots: true
project: runs/train
name: yolov8n_balanced_512
```

**与 fast 配置的差异：**

| 参数 | fast（416px） | 512px | 原因 |
|------|--------------|-------|------|
| `imgsz` | 416 | 512 | 提升小目标分辨率 |
| `batch` | 8 | 6 | 512px 每张图占用更多内存，适当降低 |

**预期效果：**
- 香烟 mAP@0.5：0.321 → 0.40-0.45（提升 25-40%）
- 整体 mAP@0.5：0.520 → 0.56-0.60（提升 8-15%）
- 每轮训练时间约 15-20 分钟，50 轮总计约 13-17 小时

**运行命令：**
```bash
python scripts/train.py --config configs/train_yolov8n_balanced_512.yaml
```

**验证命令：**
```bash
python scripts/val.py --weights runs/train/yolov8n_balanced_512/weights/best.pt --data configs/data_smoking_balanced.yaml --imgsz 512 --batch 6 --split test --name balanced_512_eval
```

---

### 4.2 新建配置文件：`configs/train_yolov8n_balanced_512_augment.yaml`

**核心改动：** 在 512px 基础上，调整数据增强策略使其对小目标更友好。

**问题分析：** 原增强策略（大范围缩放 scale:0.5、mosaic:1.0）会使香烟这类小目标在增强后变得更小甚至消失，反而损害训练效果。

**完整文件内容：**
```yaml
model: yolov8n.pt
data: configs/data_smoking_balanced.yaml
epochs: 50
imgsz: 512
batch: 6
optimizer: AdamW
lr0: 0.001
patience: 15
device: cpu
workers: 4
cache: true
cos_lr: true
close_mosaic: 5
pretrained: true
seed: 42
deterministic: true
plots: true
project: runs/train
name: yolov8n_balanced_512_augment
mosaic: 0.8        # 从1.0降低，减少马赛克拼接导致的小目标切割
scale: 0.3         # 从0.5降低，避免缩放后香烟消失
translate: 0.05    # 从0.1降低，减少小目标平移出框的概率
degrees: 0.0       # 禁用旋转，保护小目标形状
shear: 0.0         # 禁用剪切
hsv_h: 0.01        # 减小色调变化范围
hsv_s: 0.5         # 适度饱和度增强
hsv_v: 0.3         # 适度亮度增强
copy_paste: 0.1    # 启用小目标复制粘贴增强
mixup: 0.0         # 禁用mixup（会模糊小目标边界）
erasing: 0.2       # 从0.4降低随机擦除概率
fliplr: 0.5        # 保留水平翻转
```

**各参数修改对比：**

| 参数 | 原始值 | 新值 | 原因 |
|------|--------|------|------|
| `mosaic` | 1.0 | 0.8 | 减少马赛克拼接概率，避免小目标被切割 |
| `scale` | 0.5 | 0.3 | 缩小缩放范围，防止香烟缩放后不可见 |
| `translate` | 0.1 | 0.05 | 减小平移，防止小目标移出图像边界 |
| `degrees` | 未设置(默认0) | 0.0 | 明确禁用旋转 |
| `shear` | 未设置(默认0) | 0.0 | 明确禁用剪切 |
| `copy_paste` | 未设置(默认0) | 0.1 | 新增小目标复制粘贴，增加香烟样本多样性 |
| `mixup` | 0.05 | 0.0 | 禁用，防止两图混合模糊小目标边界 |
| `erasing` | 0.4 | 0.2 | 降低随机擦除，减少小目标被擦除的概率 |

**预期效果：**
- 香烟 recall 在 512px 基础上额外提升 5-10%
- 配合 512px，香烟 mAP@0.5 预计可达 0.42-0.46

**运行命令：**
```bash
python scripts/train.py --config configs/train_yolov8n_balanced_512_augment.yaml
```

---

## 五、阶段3：系统稳定性增强

### 5.1 时间平滑机制优化

**修改文件：** `app/utils/web_inference.py`

#### 5.1.1 全局参数调整（第21-30行）

| 参数 | 原值 | 新值 | 作用 |
|------|------|------|------|
| `TEMPORAL_MATCH_IOU` | 0.3 | 0.25 | 降低帧间匹配阈值，更容易跟踪轻微移动的目标 |
| `TEMPORAL_STABLE_HITS` | 2 | 3 | 需要3帧连续检测才判定为稳定，减少瞬时噪声 |
| `TEMPORAL_BRIDGE_FRAMES` | 1 | 2 | 允许2帧间隔内的轨迹桥接，容忍短暂遮挡 |
| `TRACK_STALE_FRAMES` | 3 | 5 | 轨迹保持5帧后才过期，减少同一目标反复出现/消失 |
| `CONFIDENCE_SMOOTH_ALPHA` | 无 | 0.7 | 新增：置信度指数移动平均系数 |

#### 5.1.2 `TrackState` 数据类新增字段（第55-58行）

```python
@dataclass
class TrackState:
    track_id: int
    class_id: int
    class_name: str
    xyxy: list[float]
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    consecutive_hits: int = 1
    total_hits: int = 1
    is_stable: bool = False
    smoothed_confidence: float = 0.0   # 新增：EMA平滑后的置信度，初始为0.0
```

#### 5.1.3 `_temporal_filter_detections` 方法修改（第377-404行）

在轨迹匹配更新部分，新增置信度平滑逻辑：

**修改前：**
```python
track.xyxy = detection.xyxy
track.confidence = (track.confidence * 0.6) + (detection.confidence * 0.4)
```

**修改后：**
```python
track.xyxy = detection.xyxy
track.confidence = (track.confidence * 0.6) + (detection.confidence * 0.4)
if track.smoothed_confidence == 0.0:
    track.smoothed_confidence = detection.confidence          # 首次初始化
else:
    track.smoothed_confidence = (
        CONFIDENCE_SMOOTH_ALPHA * track.smoothed_confidence   # 70% 历史值
        + (1 - CONFIDENCE_SMOOTH_ALPHA) * detection.confidence  # 30% 当前值
    )
```

渲染时使用平滑后的置信度（而非原始值）：

**修改前：**
```python
rendered_detections.append(
    DetectionBox(
        class_id=track.class_id,
        class_name=track.class_name,
        confidence=track.confidence,
        xyxy=track.xyxy,
    )
)
```

**修改后：**
```python
rendered_detections.append(
    DetectionBox(
        class_id=track.class_id,
        class_name=track.class_name,
        confidence=track.smoothed_confidence if track.smoothed_confidence > 0.0 else track.confidence,
        xyxy=track.xyxy,
    )
)
```

**预期效果：**
- 视频检测框闪烁减少 50-70%
- 置信度数值变化更平滑，不再跳动
- 短暂遮挡（1-2帧）后目标不消失

### 5.2 Web Demo 模型选择器更新

**修改文件：** `app/utils/web_inference.py`，`available_weight_candidates` 方法

将新训练的512px模型添加到候选列表最前面，训练完成后可在 Web Demo 界面直接切换：

```python
candidates = [
    (
        ROOT / "runs" / "train" / "yolov8n_balanced_512" / "weights" / "best.pt",
        "YOLOv8n 512px (balanced_512)",
        "Optimized 512px model with improved small-object cigarette detection.",
    ),
    (
        ROOT / "runs" / "train" / "yolov8n_balanced_512_augment" / "weights" / "best.pt",
        "YOLOv8n 512px Augment (balanced_512_augment)",
        "512px model with small-target-friendly augmentation strategy.",
    ),
    (
        ROOT / "runs" / "train" / "yolov8n_balanced_fast" / "weights" / "best.pt",
        "YOLOv8n Fast (balanced_fast)",
        "Accelerated training run with workers/cache optimizations.",
    ),
    # ... 原有的 baseline、ECA、SE、yolov8n.pt
]
```

---

## 六、已修复问题与注意事项

### 6.1 `persistent_workers` 参数问题

原先 3 个新配置文件中曾包含 `persistent_workers: true`，但这**不是有效的 YOLO 参数**，运行时会报错：

```
SyntaxError: 'persistent_workers' is not a valid YOLO argument.
```

**当前状态：** 已从 3 个配置文件中删除 `persistent_workers: true`：
- `configs/train_yolov8n_balanced_fast.yaml`（第12行）
- `configs/train_yolov8n_balanced_512.yaml`（第12行）
- `configs/train_yolov8n_balanced_512_augment.yaml`（第12行）

**注意：** 若后续复制这些配置生成新实验 YAML，不要重新加入该参数。

---

## 七、推荐实施路线图

### 第1天（立即执行）

1. **运行冒烟测试**：验证 cache/workers 配置正常
   ```bash
   python scripts/train.py --config configs/train_yolov8n_balanced_fast.yaml --epochs 5 --name yolov8n_balanced_fast_smoketest
   ```
2. 确认无误后，启动完整 fast 训练（后台运行，约10-12小时）
3. 后处理优化已生效（`web_inference.py` 已修改，默认 `conf=0.15` + 类别二次阈值），可立即测试

### 第2-3天（fast训练完成后）

1. 评估 fast 模型性能：
   ```bash
   python scripts/val.py --weights runs/train/yolov8n_balanced_fast/weights/best.pt --data configs/data_smoking_balanced.yaml --imgsz 416 --batch 8 --split test --name balanced_fast_eval
   ```
2. 对比 baseline vs fast 的指标差异
3. 启动 512px 训练（后台运行，约16-17小时）

### 第4-5天（512训练完成后）

1. 评估 512 模型性能：
   ```bash
   python scripts/val.py --weights runs/train/yolov8n_balanced_512/weights/best.pt --data configs/data_smoking_balanced.yaml --imgsz 512 --batch 6 --split test --name balanced_512_eval
   ```
2. 如512显著优于fast，启动 `512_augment` 训练进一步优化
3. 在 Web Demo 中切换到最佳模型进行实际测试

### 第6-7天（可选）

1. 测试 `512_augment` 模型性能
2. 综合对比所有模型，选定最终部署版本
3. 调优时间平滑参数（根据实际视频测试效果）

---

## 八、性能预测对比

| 阶段 | 训练时间 | 整体mAP@0.5 | 香烟mAP@0.5 | 香烟Recall | 视频稳定性 |
|------|---------|-------------|-------------|------------|-----------|
| **当前基线** | 18.2h/30轮 | 0.520 | 0.321 | 0.640 | 基线 |
| **阶段1完成后** | ~10-12h/50轮 | 0.52-0.55 | 0.36-0.42 | 0.70-0.75 | 基线 |
| **阶段2完成后** | ~16-17h/50轮 | 0.56-0.60 | 0.40-0.45 | 0.72-0.78 | 基线 |
| **阶段3完成后** | — | 0.56-0.60 | 0.40-0.45 | 0.72-0.78 | 闪烁-50-70% |
| **全部完成** | — | **+8-15%** | **+25-40%** | **+12-22%** | **显著改善** |

---

## 九、风险评估

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|---------|
| `cache:true` 内存不足 | 低（16GB充足） | 训练崩溃 | 回退为 `cache: false`，仍可用 `workers:4` 提速 30-40% |
| `workers:4` 在 Windows 多进程报错 | 中 | 训练报错 | 降至 `workers: 2`，或 `workers: 0`（原始值） |
| 512px 训练时间超预期 | 低 | 约16-17小时可接受 | 后台运行，不影响正常使用 |
| 香烟阈值降低导致误检 | 低（分析显示0个FP） | 精度下降 | 调回 `CLASS_CONF_THRESHOLDS[0]` 至 0.20 或 0.25 |
| 时间平滑参数过严（STABLE_HITS=3）导致检测延迟 | 低 | 需要3帧才显示 | 降回 `TEMPORAL_STABLE_HITS = 2` |

---

## 十、文件变更清单

### 新建文件

| 文件路径 | 说明 |
|---------|------|
| `configs/train_yolov8n_balanced_fast.yaml` | 416px加速训练配置（workers/cache优化） |
| `configs/train_yolov8n_balanced_512.yaml` | 512px分辨率训练配置 |
| `configs/train_yolov8n_balanced_512_augment.yaml` | 512px+小目标友好增强配置 |

### 修改文件

| 文件路径 | 修改内容 |
|---------|---------|
| `app/utils/web_inference.py` | ① 时间平滑参数优化；② 新增 `CLASS_CONF_THRESHOLDS`；③ 新增 `DEFAULT_INFERENCE_CONF=0.15`；④ `_extract_detections` 类别差异化阈值；⑤ `TrackState` 新增 `smoothed_confidence` 字段；⑥ 置信度 EMA 平滑逻辑；⑦ `available_weight_candidates` 添加新模型 |
| `app/web_demo.py` | 将 Web 默认 `default_conf` 调整为 `0.15`，确保类别二次阈值策略能实际生效 |
| `app/db.py` | 启动时将未手动调整过的旧默认 `default_conf=0.25` 自动升级为 `0.15` |
| `app/db_models.py` | 将应用默认 `default_conf` 从 `0.25` 改为 `0.15` |

### 已修复

| 文件路径 | 原问题 | 处理结果 |
|---------|------|---------|
| `configs/train_yolov8n_balanced_fast.yaml` | 含无效参数 `persistent_workers` | 已删除 |
| `configs/train_yolov8n_balanced_512.yaml` | 含无效参数 `persistent_workers` | 已删除 |
| `configs/train_yolov8n_balanced_512_augment.yaml` | 含无效参数 `persistent_workers` | 已删除 |

---

## 十一、参考指标与评估方法

### 评估单个模型
```bash
python scripts/val.py --weights <权重路径> --data configs/data_smoking_balanced.yaml --split test
# 结果保存至 runs/val/<run_name>/test_summary.json
```

### 模型对比（运行后查看 JSON 结果）
```bash
# 评估 baseline
python scripts/val.py --weights runs/train/yolov8n_balanced_30/weights/best.pt --data configs/data_smoking_balanced.yaml --imgsz 416 --batch 8 --split test --name baseline_eval

# 评估 fast
python scripts/val.py --weights runs/train/yolov8n_balanced_fast/weights/best.pt --data configs/data_smoking_balanced.yaml --imgsz 416 --batch 8 --split test --name fast_eval

# 评估 512px
python scripts/val.py --weights runs/train/yolov8n_balanced_512/weights/best.pt --data configs/data_smoking_balanced.yaml --imgsz 512 --batch 6 --split test --name 512_eval
```

### 关注的核心指标
1. `mAP@0.5`（整体）：系统综合检测能力
2. `cigarette mAP@0.5`：香烟检测改善程度（最重要）
3. `cigarette Recall`：漏检率改善情况
4. 视频测试：主观评估闪烁改善效果

---

*本文档由 Claude Code 生成，记录了2026-04-20本次优化会话的所有改动和方案。*
