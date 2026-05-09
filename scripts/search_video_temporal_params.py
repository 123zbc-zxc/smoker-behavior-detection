from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_utils import build_model, dump_json, ensure_exists

VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
CLASS_NAMES = {0: "cigarette", 1: "smoking_person", 2: "smoke"}


@dataclass(frozen=True)
class ParamSet:
    name: str
    class_conf: dict[int, float]
    match_iou: float
    stable_hits: int
    bridge_frames: int
    stale_frames: int


@dataclass
class DetectionBox:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass
class TrackState:
    track_id: int
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    last_seen_frame: int
    consecutive_hits: int = 1
    total_hits: int = 1
    is_stable: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search CPU-friendly confidence and temporal smoothing parameters on smoke videos."
    )
    parser.add_argument("--video-dir", default="datasets/raw/hmdb51_smoke")
    parser.add_argument(
        "--weights",
        default="runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt",
    )
    parser.add_argument("--output-dir", default="runs/video_temporal/threshold_search_20260503")
    parser.add_argument("--limit", type=int, default=20, help="Use 0 for all videos.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base-conf", type=float, default=0.10, help="Lowest YOLO predict confidence used to cache candidates.")
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def default_param_sets() -> list[ParamSet]:
    return [
        ParamSet("web_default", {0: 0.15, 1: 0.25, 2: 0.30}, 0.25, 3, 2, 5),
        ParamSet("recall_plus", {0: 0.12, 1: 0.22, 2: 0.28}, 0.25, 3, 2, 5),
        ParamSet("recall_bridge", {0: 0.13, 1: 0.24, 2: 0.30}, 0.25, 3, 3, 6),
        ParamSet("stable_strict", {0: 0.15, 1: 0.25, 2: 0.30}, 0.25, 4, 2, 5),
        ParamSet("precision_guard", {0: 0.18, 1: 0.28, 2: 0.35}, 0.25, 3, 1, 4),
        ParamSet("loose_match", {0: 0.15, 1: 0.25, 2: 0.30}, 0.20, 3, 2, 5),
    ]


def collect_videos(video_dir: Path, limit: int) -> list[Path]:
    videos = sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    return videos[:limit] if limit > 0 else videos


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def extract_result_boxes(result: Any) -> list[DetectionBox]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return []
    xyxy = boxes.xyxy.tolist() if boxes.xyxy is not None else []
    out: list[DetectionBox] = []
    for idx, (cls_id, score) in enumerate(zip(boxes.cls.tolist(), boxes.conf.tolist())):
        cls_int = int(cls_id)
        if cls_int not in CLASS_NAMES:
            continue
        coords = xyxy[idx] if idx < len(xyxy) else []
        if len(coords) != 4:
            continue
        out.append(DetectionBox(cls_int, float(score), tuple(float(v) for v in coords)))
    return out


def predict_video(model: Any, video_path: Path, *, imgsz: int, base_conf: float, nms_iou: float, device: str) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
    frame_detections: list[list[DetectionBox]] = []
    peak_conf = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            results = model.predict(source=frame, imgsz=imgsz, conf=base_conf, iou=nms_iou, device=device, verbose=False)
            boxes = extract_result_boxes(results[0]) if results else []
            for det in boxes:
                peak_conf = max(peak_conf, det.confidence)
            frame_detections.append(boxes)
    finally:
        capture.release()
    return {
        "video": str(video_path),
        "video_name": video_path.name,
        "total_frames": total_frames or len(frame_detections),
        "processed_frames": len(frame_detections),
        "frames": frame_detections,
        "peak_confidence": round(peak_conf, 4),
    }


def filter_boxes(boxes: list[DetectionBox], params: ParamSet) -> list[DetectionBox]:
    return [box for box in boxes if box.confidence >= params.class_conf.get(box.class_id, 0.25)]


def temporal_filter(
    tracks: list[TrackState],
    detections: list[DetectionBox],
    frame_index: int,
    next_track_id: int,
    stable_track_ids: set[int],
    params: ParamSet,
) -> tuple[list[DetectionBox], int, int]:
    candidate_pairs: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        for det_index, det in enumerate(detections):
            if track.class_id != det.class_id:
                continue
            score = iou_xyxy(track.xyxy, det.xyxy)
            if score >= params.match_iou:
                candidate_pairs.append((score, track_index, det_index))

    matches: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    for _, track_index, det_index in sorted(candidate_pairs, reverse=True):
        if track_index in used_tracks or det_index in used_dets:
            continue
        used_tracks.add(track_index)
        used_dets.add(det_index)
        matches.append((track_index, det_index))

    rendered: list[DetectionBox] = []
    bridged = 0
    for track_index, det_index in matches:
        track = tracks[track_index]
        det = detections[det_index]
        gap = frame_index - track.last_seen_frame
        track.xyxy = det.xyxy
        track.confidence = (track.confidence * 0.6) + (det.confidence * 0.4)
        track.last_seen_frame = frame_index
        track.total_hits += 1
        track.consecutive_hits = track.consecutive_hits + 1 if gap <= 1 else 1
        if not track.is_stable and track.consecutive_hits >= params.stable_hits:
            track.is_stable = True
            stable_track_ids.add(track.track_id)
        if track.is_stable:
            rendered.append(det)

    for det_index, det in enumerate(detections):
        if det_index in used_dets:
            continue
        tracks.append(TrackState(next_track_id, det.class_id, det.xyxy, det.confidence, frame_index))
        next_track_id += 1

    active: list[TrackState] = []
    for track_index, track in enumerate(tracks):
        gap = frame_index - track.last_seen_frame
        if track_index not in used_tracks and track.is_stable and 0 < gap <= params.bridge_frames:
            rendered.append(DetectionBox(track.class_id, track.confidence, track.xyxy))
            bridged += 1
        if gap <= params.stale_frames:
            active.append(track)
    tracks[:] = active
    return rendered, next_track_id, bridged


def max_consecutive(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def evaluate_cached_video(cached: dict[str, Any], params: ParamSet) -> dict[str, Any]:
    tracks: list[TrackState] = []
    next_track_id = 1
    stable_track_ids: set[int] = set()
    raw_hits: list[bool] = []
    smoothed_hits: list[bool] = []
    raw_counts = Counter[str]()
    smoothed_counts = Counter[str]()
    raw_num = 0
    smoothed_num = 0
    bridged = 0
    peak_conf = 0.0

    for frame_index, boxes in enumerate(cached["frames"], start=1):
        filtered = filter_boxes(boxes, params)
        rendered, next_track_id, bridge_count = temporal_filter(
            tracks, filtered, frame_index, next_track_id, stable_track_ids, params
        )
        raw_hits.append(bool(filtered))
        smoothed_hits.append(bool(rendered))
        raw_num += len(filtered)
        smoothed_num += len(rendered)
        bridged += bridge_count
        for box in filtered:
            raw_counts[CLASS_NAMES[box.class_id]] += 1
            peak_conf = max(peak_conf, box.confidence)
        for box in rendered:
            smoothed_counts[CLASS_NAMES[box.class_id]] += 1

    frames = int(cached["processed_frames"])
    raw_hit_frames = sum(raw_hits)
    smooth_hit_frames = sum(smoothed_hits)
    return {
        "video_name": cached["video_name"],
        "processed_frames": frames,
        "raw_num_detections": raw_num,
        "smoothed_num_detections": smoothed_num,
        "raw_hit_frames": raw_hit_frames,
        "smoothed_hit_frames": smooth_hit_frames,
        "raw_hit_frame_ratio": round(raw_hit_frames / frames, 4) if frames else 0.0,
        "smoothed_hit_frame_ratio": round(smooth_hit_frames / frames, 4) if frames else 0.0,
        "raw_max_consecutive_hit_frames": max_consecutive(raw_hits),
        "smoothed_max_consecutive_hit_frames": max_consecutive(smoothed_hits),
        "stable_track_count": len(stable_track_ids),
        "flicker_suppressed_count": bridged,
        "temporal_event_hit": bool(stable_track_ids),
        "peak_confidence": round(peak_conf, 4),
        "raw_per_class_counts": dict(raw_counts),
        "per_class_counts": dict(smoothed_counts),
    }


def aggregate(rows: list[dict[str, Any]], params: ParamSet) -> dict[str, Any]:
    video_count = len(rows)
    frames = sum(int(row["processed_frames"]) for row in rows)
    raw_hit_frames = sum(int(row["raw_hit_frames"]) for row in rows)
    smooth_hit_frames = sum(int(row["smoothed_hit_frames"]) for row in rows)
    stable_tracks = sum(int(row["stable_track_count"]) for row in rows)
    temporal_hits = sum(1 for row in rows if row["temporal_event_hit"])
    raw_detections = sum(int(row["raw_num_detections"]) for row in rows)
    smoothed_detections = sum(int(row["smoothed_num_detections"]) for row in rows)
    raw_density = raw_detections / frames if frames else 0.0
    smooth_density = smoothed_detections / frames if frames else 0.0
    hit_rate = temporal_hits / video_count if video_count else 0.0
    smooth_ratio = smooth_hit_frames / frames if frames else 0.0
    avg_run = sum(int(row["smoothed_max_consecutive_hit_frames"]) for row in rows) / video_count if video_count else 0.0
    # Positive-video proxy objective: prioritize event coverage and stable continuity, then penalize excessive raw density.
    score = (hit_rate * 100.0) + (smooth_ratio * 30.0) + (avg_run * 0.2) + ((stable_tracks / max(video_count, 1)) * 2.0) - (raw_density * 0.5)
    return {
        "param_set": params.name,
        "class_conf": {CLASS_NAMES[k]: v for k, v in params.class_conf.items()},
        "match_iou": params.match_iou,
        "stable_hits": params.stable_hits,
        "bridge_frames": params.bridge_frames,
        "stale_frames": params.stale_frames,
        "video_count": video_count,
        "total_frames": frames,
        "temporal_event_hit_videos": temporal_hits,
        "temporal_event_hit_rate": round(hit_rate, 4),
        "raw_hit_frames": raw_hit_frames,
        "raw_hit_frame_ratio": round(raw_hit_frames / frames, 4) if frames else 0.0,
        "smoothed_hit_frames": smooth_hit_frames,
        "smoothed_hit_frame_ratio": round(smooth_ratio, 4),
        "raw_num_detections": raw_detections,
        "smoothed_num_detections": smoothed_detections,
        "raw_detection_density": round(raw_density, 4),
        "smoothed_detection_density": round(smooth_density, 4),
        "stable_track_count": stable_tracks,
        "avg_stable_tracks_per_video": round(stable_tracks / video_count, 4) if video_count else 0.0,
        "avg_smoothed_max_consecutive_hit_frames": round(avg_run, 2),
        "flicker_suppressed_count": sum(int(row["flicker_suppressed_count"]) for row in rows),
        "score": round(score, 4),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "param_set",
        "score",
        "video_count",
        "temporal_event_hit_videos",
        "temporal_event_hit_rate",
        "smoothed_hit_frame_ratio",
        "raw_detection_density",
        "stable_track_count",
        "avg_smoothed_max_consecutive_hit_frames",
        "class_conf",
        "match_iou",
        "stable_hits",
        "bridge_frames",
        "stale_frames",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["class_conf"] = json.dumps(item["class_conf"], ensure_ascii=False)
            writer.writerow({field: item.get(field, "") for field in fields})


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    best = report["recommended"]
    lines = [
        "# ?????????????",
        "",
        f"- ???????`{report['video_dir']}`",
        f"- ???????{report['video_count']}",
        f"- ???`{report['weights']}`",
        f"- imgsz?{report['imgsz']}?NMS IoU?{report['nms_iou']}????{report['device']}",
        "- ???????? HMDB51 smoke ???????????????????????????????????????",
        "",
        "## ????",
        "",
        f"- ????{best['param_set']}",
        f"- ?????{best['class_conf']}",
        f"- match_iou?{best['match_iou']}?stable_hits?{best['stable_hits']}?bridge_frames?{best['bridge_frames']}?stale_frames?{best['stale_frames']}",
        f"- temporal_event_hit?{best['temporal_event_hit_videos']}/{best['video_count']} ({best['temporal_event_hit_rate']})",
        f"- smoothed_hit_frame_ratio?{best['smoothed_hit_frame_ratio']}",
        f"- stable_track_count?{best['stable_track_count']}",
        "",
        "## ??",
        "",
        "| ?? | ??? | Score | ????? | ??????? | ????? | ?????? |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(report["ranking"], start=1):
        lines.append(
            f"| {idx} | {row['param_set']} | {row['score']} | {row['temporal_event_hit_rate']} | "
            f"{row['smoothed_hit_frame_ratio']} | {row['stable_track_count']} | {row['raw_detection_density']} |"
        )
    lines.extend([
        "",
        "## ????",
        "",
        "- ???????????????????????????",
        "- ????????????????????????????????????",
        "- ??????????????????????????",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    video_dir = ensure_exists(args.video_dir, "Video directory")
    weights = ensure_exists(args.weights, "Weights")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = collect_videos(video_dir, args.limit)
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir}")

    print(f"Loading model: {weights}")
    model = build_model(str(weights))
    cached_videos: list[dict[str, Any]] = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] caching predictions: {video.name}")
        cached_videos.append(
            predict_video(model, video, imgsz=args.imgsz, base_conf=args.base_conf, nms_iou=args.nms_iou, device=args.device)
        )

    results: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []
    for params in default_param_sets():
        rows = [evaluate_cached_video(cached, params) for cached in cached_videos]
        agg = aggregate(rows, params)
        results[params.name] = {"aggregate": agg, "videos": rows}
        ranking.append(agg)
    ranking.sort(key=lambda row: row["score"], reverse=True)

    report = {
        "video_dir": str(video_dir),
        "weights": str(weights),
        "output_dir": str(output_dir),
        "video_count": len(videos),
        "imgsz": args.imgsz,
        "base_conf": args.base_conf,
        "nms_iou": args.nms_iou,
        "device": args.device,
        "search_note": "Positive-video proxy search for temporal event coverage and stability; not a false-positive benchmark.",
        "recommended": ranking[0],
        "ranking": ranking,
        "results": results,
    }
    dump_json(output_dir / "video_threshold_temporal_search.json", report)
    write_csv(output_dir / "video_threshold_temporal_search.csv", ranking)
    write_markdown(output_dir / "video_threshold_temporal_search.md", report)
    print(json.dumps({"recommended": ranking[0], "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
