from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_utils import build_model, dump_json, ensure_exists


CLASS_NAMES = {
    0: "cigarette",
    1: "smoking_person",
    2: "smoke",
}
BOX_COLORS = {
    0: (15, 73, 217),
    1: (214, 111, 29),
    2: (68, 158, 47),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LabelImg-ready hard-case annotation workspace.")
    parser.add_argument(
        "--hmdb-pack",
        default="datasets/interim/hmdb51_smoke_hardcase_pseudo",
        help="Existing HMDB51 pseudo-label pack.",
    )
    parser.add_argument(
        "--custom-video",
        default="datasets/raw/custom_smoking_videos/hello_shu_xiansheng_wang_baoqiang_smoking_aigei_20260503.mp4",
        help="Custom smoking video to sample frames from.",
    )
    parser.add_argument(
        "--weights",
        default="runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt",
        help="Weights used to generate pseudo labels for custom-video frames.",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/interim/hardcase_labeling_workspace_20260503",
        help="Output LabelImg-ready workspace.",
    )
    parser.add_argument("--custom-frame-count", type=int, default=30, help="Number of custom-video frames to sample.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def yolo_line(cls_id: int, xyxy: list[float], width: int, height: int, conf: float | None = None) -> str:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = abs(x2 - x1) / width
    bh = abs(y2 - y1) / height
    base = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    if conf is not None:
        return f"{base} {conf:.6f}"
    return base


def draw_preview(image_path: Path, detections: list[dict[str, Any]], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    for det in detections:
        cls_id = int(det["class_id"])
        score = float(det.get("confidence", 0.0))
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        color = BOX_COLORS.get(cls_id, (30, 30, 30))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{CLASS_NAMES.get(cls_id, cls_id)} {score:.2f}"
        cv2.putText(image, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "source_group",
        "source_video",
        "frame_index",
        "timestamp_sec",
        "image_path",
        "label_path",
        "preview_path",
        "pseudo_box_count",
        "max_confidence",
        "needs_review",
        "review_status",
        "issue_tags",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def copy_hmdb_pack(hmdb_pack: Path, output_root: Path) -> list[dict[str, Any]]:
    manifest_path = ensure_exists(hmdb_pack / "annotation_manifest.csv", "HMDB annotation manifest")
    rows = read_csv(manifest_path)
    images_dir = output_root / "images" / "train"
    labels_dir = output_root / "labels" / "train"
    label_conf_dir = output_root / "labels_pseudo_conf" / "train"
    previews_dir = output_root / "previews"
    out_rows: list[dict[str, Any]] = []

    for row in rows:
        src_image = ensure_exists(hmdb_pack / row["image_path"], "HMDB image")
        src_label = hmdb_pack / row["label_path"]
        src_preview = hmdb_pack / row["preview_path"]
        sample_id = f"hmdb51_{row['sample_id']}"
        dst_image = images_dir / f"{sample_id}.jpg"
        dst_label = labels_dir / f"{sample_id}.txt"
        dst_preview = previews_dir / f"{sample_id}.jpg"
        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_label, dst_label)
        if src_preview.exists():
            shutil.copy2(src_preview, dst_preview)
        else:
            shutil.copy2(src_image, dst_preview)
        out_rows.append(
            {
                "sample_id": sample_id,
                "source_group": row.get("group", "hmdb51"),
                "source_video": row.get("video_name", ""),
                "frame_index": row.get("frame_index", ""),
                "timestamp_sec": row.get("timestamp_sec", ""),
                "image_path": dst_image.relative_to(output_root).as_posix(),
                "label_path": dst_label.relative_to(output_root).as_posix(),
                "preview_path": dst_preview.relative_to(output_root).as_posix(),
                "pseudo_box_count": row.get("pseudo_box_count", "0"),
                "max_confidence": row.get("max_confidence", "0"),
                "needs_review": "yes",
                "review_status": "",
                "issue_tags": "",
                "notes": "",
            }
        )
    return out_rows


def sample_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0:
        return []
    if count <= 1:
        return [max(1, total_frames // 2)]
    return [max(1, min(total_frames, round(1 + idx * (total_frames - 1) / (count - 1)))) for idx in range(count)]


def add_custom_video_frames(
    video_path: Path,
    output_root: Path,
    weights: Path,
    *,
    frame_count: int,
    imgsz: int,
    conf: float,
    iou: float,
) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open custom video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    indices = sample_indices(total_frames, frame_count)
    model = build_model(str(weights))
    images_dir = output_root / "images" / "train"
    labels_dir = output_root / "labels" / "train"
    label_conf_dir = output_root / "labels_pseudo_conf" / "train"
    previews_dir = output_root / "previews"
    out_rows: list[dict[str, Any]] = []

    for idx, frame_index in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index - 1))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        height, width = frame.shape[:2]
        sample_id = f"custom_hello_shu_xiansheng_f{frame_index:05d}"
        image_path = images_dir / f"{sample_id}.jpg"
        label_path = labels_dir / f"{sample_id}.txt"
        label_conf_path = label_conf_dir / f"{sample_id}.txt"
        preview_path = previews_dir / f"{sample_id}.jpg"
        cv2.imwrite(str(image_path), frame)

        result = model.predict(str(image_path), imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)[0]
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.cls is not None:
            xyxy = boxes.xyxy.tolist() if boxes.xyxy is not None else []
            confs = boxes.conf.tolist() if boxes.conf is not None else []
            for det_idx, cls_value in enumerate(boxes.cls.tolist()):
                if det_idx >= len(xyxy):
                    continue
                cls_id = int(cls_value)
                if cls_id not in CLASS_NAMES:
                    continue
                score = float(confs[det_idx]) if det_idx < len(confs) else 0.0
                detections.append({"class_id": cls_id, "confidence": score, "xyxy": [float(v) for v in xyxy[det_idx]]})

        label_path.write_text(
            "\n".join(yolo_line(int(det["class_id"]), det["xyxy"], width, height) for det in detections),
            encoding="utf-8",
        )
        label_conf_path.write_text(
            "\n".join(yolo_line(int(det["class_id"]), det["xyxy"], width, height, float(det["confidence"])) for det in detections),
            encoding="utf-8",
        )
        draw_preview(image_path, detections, preview_path)
        out_rows.append(
            {
                "sample_id": sample_id,
                "source_group": "custom_hello_shu_xiansheng",
                "source_video": video_path.name,
                "frame_index": frame_index,
                "timestamp_sec": round(frame_index / fps, 3) if fps else "",
                "image_path": image_path.relative_to(output_root).as_posix(),
                "label_path": label_path.relative_to(output_root).as_posix(),
                "preview_path": preview_path.relative_to(output_root).as_posix(),
                "pseudo_box_count": len(detections),
                "max_confidence": round(max((float(det["confidence"]) for det in detections), default=0.0), 4),
                "needs_review": "yes",
                "review_status": "",
                "issue_tags": "",
                "notes": "",
            }
        )
        print(f"[custom {idx}/{len(indices)}] {sample_id}: {len(detections)} pseudo boxes")
    cap.release()
    return out_rows


def write_workspace_files(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    (output_root / "classes.txt").write_text("cigarette\nsmoking_person\nsmoke\n", encoding="utf-8")
    (output_root / "data.yaml").write_text(
        """# Hard-case annotation workspace. Review and correct labels before training.
path: .
train: images/train
val: images/train
test: images/train
names:
  0: cigarette
  1: smoking_person
  2: smoke
""",
        encoding="utf-8",
    )
    readme = f"""# Hard-case Labeling Workspace

This directory is for manual box correction before the next hard-case training round.

## What to edit

- Images: `images/train`
- YOLO labels to correct: `labels/train`
- Class list: `classes.txt`
- Visual preview: `index.html`
- Tracking sheet: `annotation_manifest.csv`

## Label rules

- `0 cigarette`: box only the cigarette body; do not include hand, mouth, or face.
- `1 smoking_person`: box the person clearly related to smoking behavior.
- `2 smoke`: box visible smoke regions only; do not label background haze.

## Important

Pseudo labels are only a starting point. Delete false boxes, add missing boxes, and adjust loose boxes before training.

Source HMDB pack: `{args.hmdb_pack}`
Custom video: `{args.custom_video}`
Weights for pseudo labels: `{args.weights}`
"""
    (output_root / "README_LABELING.md").write_text(readme, encoding="utf-8")


def write_index_html(output_root: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
            <article class="card {html.escape(str(row['source_group']))}">
              <img src="{html.escape(str(row['preview_path']))}" alt="{html.escape(str(row['sample_id']))}">
              <div class="meta">
                <h3>{html.escape(str(row['sample_id']))}</h3>
                <p><strong>source:</strong> {html.escape(str(row['source_group']))}</p>
                <p><strong>video:</strong> {html.escape(str(row['source_video']))}</p>
                <p><strong>frame:</strong> {html.escape(str(row['frame_index']))} / {html.escape(str(row['timestamp_sec']))}s</p>
                <p><strong>pseudo boxes:</strong> {html.escape(str(row['pseudo_box_count']))} | <strong>max conf:</strong> {html.escape(str(row['max_confidence']))}</p>
                <p><a href="{html.escape(str(row['image_path']))}">image</a> | <a href="{html.escape(str(row['label_path']))}">label</a></p>
              </div>
            </article>
            """
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hard-case 修框工作台</title>
<style>
body {{ margin: 24px; font-family: "Microsoft YaHei", Arial, sans-serif; background:#f6f1e8; color:#1f2933; }}
.hero {{ background:#fff; border:1px solid #ded4c2; border-radius:16px; padding:18px; margin-bottom:18px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:16px; }}
.card {{ background:white; border:1px solid #d7cfbf; border-radius:14px; overflow:hidden; box-shadow:0 8px 20px rgba(0,0,0,.06); }}
.card img {{ width:100%; display:block; }}
.meta {{ padding:12px; }}
.meta h3 {{ margin:0 0 8px; font-size:14px; word-break:break-all; }}
.meta p {{ margin:5px 0; line-height:1.45; }}
.custom_hello_shu_xiansheng {{ border-top:5px solid #0f766e; }}
.top_loss,.lost_temporal_hit {{ border-top:5px solid #d9480f; }}
.top_gain,.new_temporal_hit {{ border-top:5px solid #2f9e44; }}
code {{ background:#f1eee7; padding:2px 5px; border-radius:5px; }}
</style>
</head>
<body>
<section class="hero">
  <h1>Hard-case 修框工作台</h1>
  <p>共 {len(rows)} 张图。打开 <code>images/train</code> 和 <code>labels/train</code> 用 LabelImg 修 YOLO 框。</p>
  <p>必须人工检查：补漏检框、删除误检框、调整偏大的框。不要直接把伪标签当真值训练。</p>
</section>
<div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    hmdb_pack = ensure_exists(args.hmdb_pack, "HMDB pseudo-label pack")
    custom_video = ensure_exists(args.custom_video, "Custom video")
    weights = ensure_exists(args.weights, "Pseudo-label weights")
    output_root = Path(args.output_root)
    for directory in (
        output_root / "images" / "train",
        output_root / "labels" / "train",
        output_root / "labels_pseudo_conf" / "train",
        output_root / "previews",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    rows = copy_hmdb_pack(hmdb_pack, output_root)
    rows.extend(
        add_custom_video_frames(
            custom_video,
            output_root,
            weights,
            frame_count=args.custom_frame_count,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )
    )
    write_workspace_files(output_root, rows, args)
    write_csv(output_root / "annotation_manifest.csv", rows)
    write_index_html(output_root, rows)
    dump_json(
        output_root / "annotation_summary.json",
        {
            "output_root": str(output_root),
            "image_count": len(rows),
            "pseudo_labeled_images": sum(1 for row in rows if int(row.get("pseudo_box_count", 0)) > 0),
            "empty_label_images": sum(1 for row in rows if int(row.get("pseudo_box_count", 0)) == 0),
            "pseudo_box_count": sum(int(row.get("pseudo_box_count", 0)) for row in rows),
            "class_names": CLASS_NAMES,
            "warning": "Review and correct labels before training.",
        },
    )
    print(f"Hard-case labeling workspace saved to: {output_root}")


if __name__ == "__main__":
    main()
