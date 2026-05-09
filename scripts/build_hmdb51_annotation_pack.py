from __future__ import annotations

import argparse
import csv
import html
import json
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
    parser = argparse.ArgumentParser(description="Build a YOLO annotation pack from HMDB51 review frames with pseudo labels.")
    parser.add_argument("--review-dir", default="tmp/hmdb51_smoke_frame_review_20260502", help="Frame review pack directory.")
    parser.add_argument(
        "--weights",
        default="runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt",
        help="Weights used to generate pseudo labels.",
    )
    parser.add_argument("--output-root", default="datasets/interim/hmdb51_smoke_hardcase_pseudo", help="Output YOLO pack root.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.12, help="Pseudo-label confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--cigarette-only", action="store_true", default=True, help="Keep only cigarette pseudo boxes.")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        score = float(det["confidence"])
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        color = BOX_COLORS.get(cls_id, (30, 30, 30))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{CLASS_NAMES.get(cls_id, cls_id)} {score:.2f}"
        cv2.putText(image, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def write_data_yaml(output_root: Path) -> None:
    text = """# HMDB51 smoke hard-case pseudo-label annotation pack.
# Review and correct labels before using this dataset for training.
path: .
train: images/train
val: images/train
test: images/train
names:
  0: cigarette
  1: smoking_person
  2: smoke
"""
    (output_root / "data.yaml").write_text(text, encoding="utf-8")


def write_index_html(output_root: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        preview = Path(row["preview_path"]).as_posix()
        image = Path(row["image_path"]).as_posix()
        label = Path(row["label_path"]).as_posix()
        cards.append(
            f"""
            <article class="card {html.escape(row['group'])}">
              <img src="{html.escape(preview)}" alt="{html.escape(row['sample_id'])}">
              <div class="meta">
                <h3>{html.escape(row['sample_id'])}</h3>
                <p><strong>group:</strong> {html.escape(row['group'])}</p>
                <p><strong>video:</strong> {html.escape(row['video_name'])}</p>
                <p><strong>frame:</strong> {html.escape(str(row['frame_index']))}</p>
                <p><strong>pseudo boxes:</strong> {row['pseudo_box_count']} | <strong>max conf:</strong> {row['max_confidence']}</p>
                <p><a href="{html.escape(image)}">image</a> | <a href="{html.escape(label)}">label</a></p>
              </div>
            </article>
            """
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HMDB51 Pseudo Label Review</title>
<style>
body {{ margin: 24px; font-family: "Microsoft YaHei", Arial, sans-serif; background:#f6f1e8; color:#1f2933; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
.card {{ background:white; border:1px solid #d7cfbf; border-radius:14px; overflow:hidden; box-shadow:0 8px 20px rgba(0,0,0,.06); }}
.card img {{ width:100%; display:block; }}
.meta {{ padding:12px; }}
.meta h3 {{ margin:0 0 8px; font-size:14px; word-break:break-all; }}
.meta p {{ margin:5px 0; }}
.top_gain,.new_temporal_hit {{ border-top:5px solid #2f9e44; }}
.top_loss,.lost_temporal_hit {{ border-top:5px solid #d9480f; }}
.warning {{ background:#fff7e6; border:1px solid #f0c36d; padding:12px 14px; border-radius:12px; margin:16px 0; }}
</style>
</head>
<body>
<h1>HMDB51 Pseudo Label Review</h1>
<div class="warning">这些标签是模型伪标签，只能作为修框起点。训练前必须人工检查并修正 YOLO txt。</div>
<div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    review_dir = ensure_exists(args.review_dir, "Review directory")
    manifest_path = ensure_exists(review_dir / "frame_review_manifest.csv", "Frame review manifest")
    weights = ensure_exists(args.weights, "Pseudo-label weights")
    output_root = Path(args.output_root)
    images_dir = output_root / "images" / "train"
    labels_dir = output_root / "labels" / "train"
    previews_dir = output_root / "previews"
    for directory in (images_dir, labels_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model = build_model(str(weights))
    rows = read_manifest(manifest_path)
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        source_image = review_dir / row["frame_path"]
        source_image = ensure_exists(source_image, "Review frame")
        sample_id = row["sample_id"]
        dst_image = images_dir / f"hmdb51_{sample_id}.jpg"
        dst_label = labels_dir / f"hmdb51_{sample_id}.txt"
        dst_preview = previews_dir / f"hmdb51_{sample_id}.jpg"
        image = cv2.imread(str(source_image))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {source_image}")
        height, width = image.shape[:2]
        cv2.imwrite(str(dst_image), image)

        result = model.predict(str(source_image), imgsz=args.imgsz, conf=args.conf, iou=args.iou, device="cpu", verbose=False)[0]
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.cls is not None:
            xyxy = boxes.xyxy.tolist() if boxes.xyxy is not None else []
            confs = boxes.conf.tolist() if boxes.conf is not None else []
            for det_idx, cls_value in enumerate(boxes.cls.tolist()):
                cls_id = int(cls_value)
                if args.cigarette_only and cls_id != 0:
                    continue
                if det_idx >= len(xyxy):
                    continue
                score = float(confs[det_idx]) if det_idx < len(confs) else 0.0
                detections.append({"class_id": cls_id, "confidence": score, "xyxy": [float(v) for v in xyxy[det_idx]]})

        dst_label.write_text(
            "\n".join(yolo_line(int(det["class_id"]), det["xyxy"], width, height) for det in detections),
            encoding="utf-8",
        )
        (labels_dir / f"hmdb51_{sample_id}.pseudo_conf.txt").write_text(
            "\n".join(yolo_line(int(det["class_id"]), det["xyxy"], width, height, float(det["confidence"])) for det in detections),
            encoding="utf-8",
        )
        draw_preview(dst_image, detections, dst_preview)
        out_row = {
            **row,
            "image_path": dst_image.relative_to(output_root).as_posix(),
            "label_path": dst_label.relative_to(output_root).as_posix(),
            "preview_path": dst_preview.relative_to(output_root).as_posix(),
            "pseudo_box_count": len(detections),
            "max_confidence": round(max((float(det["confidence"]) for det in detections), default=0.0), 4),
            "needs_human_review": "yes",
        }
        out_rows.append(out_row)
        print(f"[{idx}/{len(rows)}] {sample_id}: {len(detections)} pseudo boxes")

    write_data_yaml(output_root)
    with (output_root / "annotation_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(out_rows[0].keys()) if out_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    write_index_html(output_root, out_rows)
    dump_json(
        output_root / "annotation_summary.json",
        {
            "review_dir": str(review_dir),
            "weights": str(weights),
            "output_root": str(output_root),
            "image_count": len(out_rows),
            "pseudo_labeled_images": sum(1 for row in out_rows if int(row["pseudo_box_count"]) > 0),
            "empty_label_images": sum(1 for row in out_rows if int(row["pseudo_box_count"]) == 0),
            "pseudo_box_count": sum(int(row["pseudo_box_count"]) for row in out_rows),
            "warning": "Pseudo labels must be reviewed and corrected before training.",
        },
    )
    print(f"Annotation pack saved to: {output_root}")


if __name__ == "__main__":
    main()
