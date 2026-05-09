from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.web_inference import DetectionService
from scripts.yolo_utils import dump_json, ensure_exists


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal video detection evaluation on HMDB51 smoke videos.")
    parser.add_argument("--video-dir", default="datasets/raw/hmdb51_smoke", help="Directory containing HMDB51 smoke videos.")
    parser.add_argument(
        "--old-weights",
        default="runs/imported/smoker_weights_20260429/best.pt",
        help="Current champion checkpoint.",
    )
    parser.add_argument(
        "--hard-weights",
        default="runs/imported/yolov8n_colab_640_hard_candidate_20260502/train/weights/best.pt",
        help="Hard fine-tuned checkpoint.",
    )
    parser.add_argument("--output-dir", default="runs/video_temporal/hmdb51_smoke_20260502", help="Report output dir.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of videos. 0 means all videos.")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated output videos.")
    return parser.parse_args()


def collect_videos(video_dir: Path, limit: int) -> list[Path]:
    videos = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv", ".webm"}
    )
    if limit > 0:
        return videos[:limit]
    return videos


def max_consecutive_hit_frames(frame_hits: list[bool]) -> int:
    best = 0
    current = 0
    for hit in frame_hits:
        if hit:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def hit_segments(frame_hits: list[bool]) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    start: int | None = None
    for idx, hit in enumerate(frame_hits, start=1):
        if hit and start is None:
            start = idx
        if not hit and start is not None:
            segments.append({"start_frame": start, "end_frame": idx - 1, "length": idx - start})
            start = None
    if start is not None:
        segments.append({"start_frame": start, "end_frame": len(frame_hits), "length": len(frame_hits) - start + 1})
    return segments


def miss_segments(frame_hits: list[bool]) -> list[dict[str, int]]:
    return hit_segments([not hit for hit in frame_hits])


def summarize_segments(segments: list[dict[str, int]]) -> dict[str, Any]:
    if not segments:
        return {"count": 0, "max_length": 0, "segments": []}
    return {
        "count": len(segments),
        "max_length": max(seg["length"] for seg in segments),
        "segments": segments[:20],
    }


def evaluate_video(service: DetectionService, video_path: Path, output_path: Path, *, conf: float, iou: float) -> dict[str, Any]:
    frame_raw_hits: list[bool] = []
    frame_smoothed_hits: list[bool] = []

    def frame_callback(frame_index: int, raw_count: int, smoothed_count: int) -> None:
        frame_raw_hits.append(raw_count > 0)
        frame_smoothed_hits.append(smoothed_count > 0)

    summary = service.process_video_file(
        video_path,
        output_path,
        conf=conf,
        iou=iou,
        frame_callback=frame_callback,
    )
    total_frames = int(summary.get("processed_frames") or len(frame_raw_hits))
    raw_hit_frames = sum(frame_raw_hits)
    smoothed_hit_frames = sum(frame_smoothed_hits)
    raw_segments = hit_segments(frame_raw_hits)
    smoothed_segments = hit_segments(frame_smoothed_hits)
    raw_misses = miss_segments(frame_raw_hits)
    smoothed_misses = miss_segments(frame_smoothed_hits)

    return {
        **summary,
        "video": str(video_path),
        "raw_hit_frames": raw_hit_frames,
        "raw_miss_frames": max(total_frames - raw_hit_frames, 0),
        "raw_hit_frame_ratio": round(raw_hit_frames / total_frames, 4) if total_frames else 0.0,
        "raw_max_consecutive_hit_frames": max_consecutive_hit_frames(frame_raw_hits),
        "raw_hit_segments": summarize_segments(raw_segments),
        "raw_miss_segments": summarize_segments(raw_misses),
        "smoothed_hit_frames": smoothed_hit_frames,
        "smoothed_miss_frames": max(total_frames - smoothed_hit_frames, 0),
        "smoothed_hit_frame_ratio": round(smoothed_hit_frames / total_frames, 4) if total_frames else 0.0,
        "smoothed_max_consecutive_hit_frames": max_consecutive_hit_frames(frame_smoothed_hits),
        "smoothed_hit_segments": summarize_segments(smoothed_segments),
        "smoothed_miss_segments": summarize_segments(smoothed_misses),
        "temporal_stability_gain_frames": smoothed_hit_frames - raw_hit_frames,
    }


def aggregate_model_results(model_name: str, video_results: list[dict[str, Any]]) -> dict[str, Any]:
    video_count = len(video_results)
    total_frames = sum(int(item.get("processed_frames", 0)) for item in video_results)
    raw_hit_frames = sum(int(item.get("raw_hit_frames", 0)) for item in video_results)
    smoothed_hit_frames = sum(int(item.get("smoothed_hit_frames", 0)) for item in video_results)
    temporal_hits = sum(1 for item in video_results if item.get("temporal_event_hit"))
    raw_any_hits = sum(1 for item in video_results if int(item.get("raw_hit_frames", 0)) > 0)
    smoothed_any_hits = sum(1 for item in video_results if int(item.get("smoothed_hit_frames", 0)) > 0)
    return {
        "model": model_name,
        "video_count": video_count,
        "total_frames": total_frames,
        "raw_any_hit_videos": raw_any_hits,
        "smoothed_any_hit_videos": smoothed_any_hits,
        "temporal_event_hit_videos": temporal_hits,
        "raw_hit_frames": raw_hit_frames,
        "raw_hit_frame_ratio": round(raw_hit_frames / total_frames, 4) if total_frames else 0.0,
        "smoothed_hit_frames": smoothed_hit_frames,
        "smoothed_hit_frame_ratio": round(smoothed_hit_frames / total_frames, 4) if total_frames else 0.0,
        "flicker_suppressed_count": sum(int(item.get("flicker_suppressed_count", 0)) for item in video_results),
        "stable_track_count": sum(int(item.get("stable_track_count", 0)) for item in video_results),
        "avg_raw_max_consecutive_hit_frames": round(
            sum(int(item.get("raw_max_consecutive_hit_frames", 0)) for item in video_results) / video_count,
            2,
        ) if video_count else 0.0,
        "avg_smoothed_max_consecutive_hit_frames": round(
            sum(int(item.get("smoothed_max_consecutive_hit_frames", 0)) for item in video_results) / video_count,
            2,
        ) if video_count else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "video_name",
        "processed_frames",
        "raw_hit_frames",
        "raw_hit_frame_ratio",
        "raw_max_consecutive_hit_frames",
        "raw_miss_frames",
        "raw_num_detections",
        "smoothed_hit_frames",
        "smoothed_hit_frame_ratio",
        "smoothed_max_consecutive_hit_frames",
        "smoothed_miss_frames",
        "smoothed_num_detections",
        "stable_track_count",
        "flicker_suppressed_count",
        "temporal_event_hit",
        "peak_confidence",
        "output_video_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    video_dir = ensure_exists(args.video_dir, "Video directory")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = collect_videos(video_dir, args.limit)
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir}")

    models = [
        ModelSpec("old_champion", ensure_exists(args.old_weights, "Old champion weights")),
        ModelSpec("hard_finetune", ensure_exists(args.hard_weights, "Hard fine-tune weights")),
    ]

    all_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "video_dir": str(video_dir),
        "output_dir": str(output_dir),
        "video_count": len(videos),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "models": {},
    }

    for model in models:
        print(f"Evaluating {model.name}: {model.weights}")
        service = DetectionService(weights_path=model.weights, imgsz=args.imgsz)
        model_dir = output_dir / model.name
        model_dir.mkdir(parents=True, exist_ok=True)
        video_results: list[dict[str, Any]] = []
        for index, video in enumerate(videos, start=1):
            print(f"[{model.name}] {index}/{len(videos)} {video.name}")
            output_video = model_dir / f"{video.stem}.mp4"
            if not args.save_annotated:
                output_video = output_dir / "_tmp_no_keep.mp4"
            item = evaluate_video(service, video, output_video, conf=args.conf, iou=args.iou)
            if not args.save_annotated and output_video.exists():
                output_video.unlink()
                item["output_video_path"] = ""
            item["model"] = model.name
            item["video_name"] = video.name
            video_results.append(item)
            all_rows.append(item)
        report["models"][model.name] = {
            "weights": str(model.weights),
            "aggregate": aggregate_model_results(model.name, video_results),
            "videos": video_results,
        }

    old = report["models"]["old_champion"]["aggregate"]
    hard = report["models"]["hard_finetune"]["aggregate"]
    report["comparison"] = {
        "hard_minus_old_raw_hit_frames": hard["raw_hit_frames"] - old["raw_hit_frames"],
        "hard_minus_old_smoothed_hit_frames": hard["smoothed_hit_frames"] - old["smoothed_hit_frames"],
        "hard_minus_old_temporal_event_hit_videos": hard["temporal_event_hit_videos"] - old["temporal_event_hit_videos"],
        "hard_minus_old_raw_hit_ratio": round(hard["raw_hit_frame_ratio"] - old["raw_hit_frame_ratio"], 4),
        "hard_minus_old_smoothed_hit_ratio": round(hard["smoothed_hit_frame_ratio"] - old["smoothed_hit_frame_ratio"], 4),
    }

    dump_json(output_dir / "hmdb51_smoke_temporal_report.json", report)
    write_csv(output_dir / "hmdb51_smoke_temporal_per_video.csv", all_rows)
    print(f"Report saved to: {output_dir / 'hmdb51_smoke_temporal_report.json'}")
    print(json.dumps({
        "old": old,
        "hard": hard,
        "comparison": report["comparison"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
