from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_utils import build_model, ensure_exists


CLASS_NAMES = {0: "cigarette", 1: "smoking_person", 2: "smoke"}
BOX_COLORS = {0: (38, 70, 225), 1: (214, 120, 28), 2: (61, 150, 57)}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Build custom-video hard-case frame candidates with pseudo labels.")
    parser.add_argument("--video-root", default="datasets/raw/custom_smoking_videos", help="Folder containing custom videos.")
    parser.add_argument(
        "--weights",
        default="runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt",
        help="Model weights used for pseudo-label and preview generation.",
    )
    parser.add_argument(
        "--output-root",
        default=f"datasets/interim/custom_video_frame_candidates_{stamp}",
        help="Output candidate pack directory.",
    )
    parser.add_argument("--frames-per-video", type=int, default=36, help="Max frames evaluated per video before filtering.")
    parser.add_argument("--keep-per-video", type=int, default=24, help="Max candidate frames kept per video.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.08, help="Low inference threshold to expose weak detections.")
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu", help="Use cpu by default; pass 0 if a CUDA GPU is available.")
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def list_videos(video_root: Path) -> list[Path]:
    return sorted(p for p in video_root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def sample_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0:
        return []
    if count >= total_frames:
        return list(range(total_frames))
    if count <= 1:
        return [total_frames // 2]
    return sorted({round(idx * (total_frames - 1) / (count - 1)) for idx in range(count)})


def safe_stem(path: Path) -> str:
    allowed = []
    for char in path.stem:
        if char.isalnum() and ord(char) < 128:
            allowed.append(char.lower())
        elif char in ("-", "_"):
            allowed.append(char)
        else:
            allowed.append("_")
    stem = "".join(allowed).strip("_")
    return stem or "video"


def yolo_line(cls_id: int, xyxy: list[float], width: int, height: int, conf: float | None = None) -> str:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)
    bw = abs(x2 - x1) / max(width, 1)
    bh = abs(y2 - y1) / max(height, 1)
    base = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    if conf is None:
        return base
    return f"{base} {conf:.6f}"


def draw_preview(frame: Any, detections: list[dict[str, Any]], output_path: Path) -> None:
    preview = frame.copy()
    for det in detections:
        cls_id = int(det["class_id"])
        score = float(det["confidence"])
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        color = BOX_COLORS.get(cls_id, (40, 40, 40))
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls_id, cls_id)} {score:.2f}"
        cv2.putText(preview, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)


def detect_frame(model: Any, frame: Any, imgsz: int, conf: float, iou: float, device: str) -> list[dict[str, Any]]:
    result = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
    boxes = getattr(result, "boxes", None)
    detections: list[dict[str, Any]] = []
    if boxes is None or boxes.cls is None:
        return detections
    xyxy = boxes.xyxy.tolist() if boxes.xyxy is not None else []
    confs = boxes.conf.tolist() if boxes.conf is not None else []
    for idx, cls_value in enumerate(boxes.cls.tolist()):
        if idx >= len(xyxy):
            continue
        cls_id = int(cls_value)
        if cls_id not in CLASS_NAMES:
            continue
        detections.append(
            {
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "confidence": float(confs[idx]) if idx < len(confs) else 0.0,
                "xyxy": [float(v) for v in xyxy[idx]],
            }
        )
    return detections


def candidate_score(detections: list[dict[str, Any]]) -> tuple[int, float]:
    if not detections:
        return 0, 0.0
    max_conf = max(float(det["confidence"]) for det in detections)
    has_smoking_person = any(int(det["class_id"]) == 1 for det in detections)
    has_smoke = any(int(det["class_id"]) == 2 for det in detections)
    weak = 0.0 < max_conf < 0.25
    class_bonus = 2 if has_smoking_person else 1 if has_smoke else 0
    weak_bonus = 2 if weak else 0
    return class_bonus + weak_bonus + min(len(detections), 3), max_conf


def tags_for(detections: list[dict[str, Any]]) -> list[str]:
    if not detections:
        return ["no_detection_candidate"]
    tags: list[str] = []
    max_conf = max(float(det["confidence"]) for det in detections)
    if max_conf < 0.25:
        tags.append("low_confidence")
    if any(int(det["class_id"]) == 0 for det in detections):
        tags.append("has_cigarette")
    if any(int(det["class_id"]) == 1 for det in detections):
        tags.append("has_smoking_person")
    if any(int(det["class_id"]) == 2 for det in detections):
        tags.append("has_smoke")
    if len(detections) >= 3:
        tags.append("multi_detection")
    return tags


def process_video(video_path: Path, model: Any, output_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    indices = sample_indices(total_frames, args.frames_per_video)
    stem = safe_stem(video_path)
    rows: list[dict[str, Any]] = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        detections = detect_frame(model, frame, args.imgsz, args.conf, args.iou, args.device)
        score, max_conf = candidate_score(detections)
        rows.append(
            {
                "video_path": str(video_path),
                "video_name": video_path.name,
                "video_stem": stem,
                "frame_index": frame_index,
                "timestamp_sec": round(frame_index / fps, 3) if fps else "",
                "width": width,
                "height": height,
                "fps": round(fps, 3) if fps else "",
                "total_frames": total_frames,
                "detections": detections,
                "detection_count": len(detections),
                "max_confidence": round(max_conf, 4),
                "candidate_score": score,
                "issue_tags": ";".join(tags_for(detections)),
                "frame": frame,
            }
        )
    cap.release()

    no_det = [r for r in rows if int(r["detection_count"]) == 0]
    detected = [r for r in rows if int(r["detection_count"]) > 0]
    selected = sorted(detected, key=lambda r: (-int(r["candidate_score"]), float(r["max_confidence"])))[: args.keep_per_video]
    remaining_slots = max(0, args.keep_per_video - len(selected))
    selected.extend(no_det[: min(remaining_slots, 4)])
    selected = sorted(selected, key=lambda r: int(r["frame_index"]))

    image_dir = output_root / "images"
    label_dir = output_root / "labels_pseudo"
    label_conf_dir = output_root / "labels_pseudo_conf"
    preview_dir = output_root / "previews"
    kept_rows: list[dict[str, Any]] = []
    for row in selected:
        frame = row.pop("frame")
        sample_id = f"{row['video_stem']}_f{int(row['frame_index']):06d}"
        image_path = image_dir / f"{sample_id}.jpg"
        label_path = label_dir / f"{sample_id}.txt"
        label_conf_path = label_conf_dir / f"{sample_id}.txt"
        preview_path = preview_dir / f"{sample_id}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_conf_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), frame)
        label_path.write_text(
            "\n".join(yolo_line(int(det["class_id"]), det["xyxy"], int(row["width"]), int(row["height"])) for det in row["detections"]),
            encoding="utf-8",
        )
        label_conf_path.write_text(
            "\n".join(
                yolo_line(
                    int(det["class_id"]),
                    det["xyxy"],
                    int(row["width"]),
                    int(row["height"]),
                    float(det["confidence"]),
                )
                for det in row["detections"]
            ),
            encoding="utf-8",
        )
        draw_preview(frame, row["detections"], preview_path)
        kept = dict(row)
        kept.update(
            {
                "sample_id": sample_id,
                "image_path": image_path.relative_to(output_root).as_posix(),
                "label_path": label_path.relative_to(output_root).as_posix(),
                "label_conf_path": label_conf_path.relative_to(output_root).as_posix(),
                "preview_path": preview_path.relative_to(output_root).as_posix(),
                "classes_detected": ";".join(sorted({det["class_name"] for det in row["detections"]})),
            }
        )
        kept.pop("detections", None)
        kept_rows.append(kept)
    print(f"{video_path.name}: evaluated {len(rows)} frames, kept {len(kept_rows)} candidates")
    return kept_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "video_name",
        "frame_index",
        "timestamp_sec",
        "width",
        "height",
        "fps",
        "total_frames",
        "detection_count",
        "max_confidence",
        "classes_detected",
        "candidate_score",
        "issue_tags",
        "image_path",
        "label_path",
        "label_conf_path",
        "preview_path",
        "review_status",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            writer.writerow(out)


def write_index(output_root: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
            <article class="card">
              <img src="{html.escape(str(row['preview_path']))}" alt="{html.escape(str(row['sample_id']))}">
              <div class="meta">
                <h3>{html.escape(str(row['sample_id']))}</h3>
                <p><b>视频:</b> {html.escape(str(row['video_name']))}</p>
                <p><b>帧:</b> {row['frame_index']} / {row['timestamp_sec']}s</p>
                <p><b>检测:</b> {row['detection_count']} | <b>最高置信度:</b> {row['max_confidence']}</p>
                <p><b>类别:</b> {html.escape(str(row['classes_detected']))}</p>
                <p><b>标签:</b> {html.escape(str(row['issue_tags']))}</p>
                <p><a href="{html.escape(str(row['image_path']))}">原图</a> | <a href="{html.escape(str(row['label_path']))}">伪标签</a></p>
              </div>
            </article>
            """
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>自定义视频抽帧候选包</title>
<style>
body {{ margin: 24px; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f7f3ea; color: #1f2933; }}
.hero {{ background: #fffdf7; border: 1px solid #ded6c8; border-radius: 16px; padding: 18px; margin-bottom: 18px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }}
.card {{ background: white; border: 1px solid #ded6c8; border-radius: 14px; overflow: hidden; box-shadow: 0 10px 24px rgba(0,0,0,.07); }}
.card img {{ width: 100%; display: block; }}
.meta {{ padding: 12px; }}
.meta h3 {{ margin: 0 0 8px; font-size: 14px; word-break: break-all; }}
.meta p {{ margin: 5px 0; line-height: 1.45; }}
code {{ background: #eee6d8; padding: 2px 5px; border-radius: 5px; }}
</style>
</head>
<body>
<section class="hero">
  <h1>自定义视频抽帧候选包</h1>
  <p>共 {len(rows)} 张候选帧。红框为 cigarette，蓝框为 smoking_person，绿框为 smoke。</p>
  <p>这些标签是模型伪标签，只能作为 hard-case 复查候选，不应未经人工检查直接加入训练集。</p>
</section>
<div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    video_root = ensure_exists(args.video_root, "Custom video folder")
    weights = ensure_exists(args.weights, "Weights")
    output_root = repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    videos = list_videos(video_root)
    if not videos:
        raise FileNotFoundError(f"No supported videos found in: {video_root}")

    model = build_model(str(weights))
    all_rows: list[dict[str, Any]] = []
    for video_path in videos:
        all_rows.extend(process_video(video_path, model, output_root, args))

    (output_root / "classes.txt").write_text("cigarette\nsmoking_person\nsmoke\n", encoding="utf-8")
    write_csv(output_root / "frame_candidates.csv", all_rows)
    summary = {
        "video_root": str(video_root),
        "weights": str(weights),
        "output_root": str(output_root),
        "video_count": len(videos),
        "candidate_frame_count": len(all_rows),
        "frames_per_video": args.frames_per_video,
        "keep_per_video": args.keep_per_video,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "warning": "Pseudo labels require human review before training.",
    }
    (output_root / "candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_index(output_root, all_rows)
    shutil.make_archive(str(output_root), "zip", output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Candidate pack saved to: {output_root}")
    print(f"Zip saved to: {output_root}.zip")


if __name__ == "__main__":
    main()
