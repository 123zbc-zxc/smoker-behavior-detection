from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_utils import dump_json, ensure_exists


@dataclass(frozen=True)
class VideoPick:
    video_name: str
    group: str
    delta_smoothed: int
    frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frame review pack from HMDB51 temporal comparison videos.")
    parser.add_argument("--report-dir", default="runs/video_temporal/hmdb51_smoke_20260502", help="Temporal report directory.")
    parser.add_argument("--video-dir", default="datasets/raw/hmdb51_smoke", help="HMDB51 smoke video directory.")
    parser.add_argument("--output-dir", default="tmp/hmdb51_smoke_frame_review_20260502", help="Output review pack directory.")
    parser.add_argument("--top-gains", type=int, default=10, help="Number of top gain videos to sample.")
    parser.add_argument("--top-losses", type=int, default=10, help="Number of top loss videos to sample.")
    parser.add_argument("--new-hits", type=int, default=6, help="Number of new temporal-hit videos to sample.")
    parser.add_argument("--lost-hits", type=int, default=4, help="Number of lost temporal-hit videos to sample.")
    parser.add_argument("--frames-per-video", type=int, default=3, help="Frames sampled per selected video.")
    return parser.parse_args()


def load_picks(summary_path: Path, args: argparse.Namespace) -> list[VideoPick]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    groups = [
        ("top_gain", data.get("top_smoothed_gains", [])[: args.top_gains]),
        ("top_loss", data.get("top_smoothed_losses", [])[: args.top_losses]),
        ("new_temporal_hit", data.get("new_temporal_hits", [])[: args.new_hits]),
        ("lost_temporal_hit", data.get("lost_temporal_hits", [])[: args.lost_hits]),
    ]
    picks: list[VideoPick] = []
    seen: set[tuple[str, str]] = set()
    for group_name, rows in groups:
        for row in rows:
            key = (group_name, row["video_name"])
            if key in seen:
                continue
            seen.add(key)
            picks.append(
                VideoPick(
                    video_name=row["video_name"],
                    group=group_name,
                    delta_smoothed=int(row.get("delta_smoothed_hit_frames", 0)),
                    frames=int(row.get("frames", 0)),
                )
            )
    return picks


def frame_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0:
        return []
    if count <= 1:
        return [max(1, total_frames // 2)]
    anchors = [0.2, 0.5, 0.8]
    if count != 3:
        anchors = [(idx + 1) / (count + 1) for idx in range(count)]
    return sorted({min(total_frames, max(1, int(round(total_frames * anchor)))) for anchor in anchors})


def write_frame(video_path: Path, frame_index: int, output_path: Path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index - 1, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    height, width = frame.shape[:2]
    timestamp = frame_index / fps if fps else 0.0
    return width, height, timestamp


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "group",
        "video_name",
        "frame_index",
        "timestamp_sec",
        "width",
        "height",
        "delta_smoothed_hit_frames",
        "review_label",
        "issue_tags",
        "notes",
        "frame_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        rel = Path(row["frame_path"]).as_posix()
        cards.append(
            f"""
            <article class="card {html.escape(row['group'])}">
              <img src="{html.escape(rel)}" alt="{html.escape(row['sample_id'])}">
              <div class="meta">
                <h3>{html.escape(row['sample_id'])}</h3>
                <p><strong>group:</strong> {html.escape(row['group'])}</p>
                <p><strong>video:</strong> {html.escape(row['video_name'])}</p>
                <p><strong>frame:</strong> {row['frame_index']} / {row['timestamp_sec']:.2f}s</p>
                <p><strong>delta smoothed:</strong> {row['delta_smoothed_hit_frames']}</p>
                <p><strong>人工复查建议:</strong> 香烟太小 / 烟雾遮挡 / 低清晰度 / 手嘴背景误检 / 漏标 / 可加入训练</p>
              </div>
            </article>
            """
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HMDB51 Smoke Frame Review</title>
  <style>
    body {{ margin: 24px; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f6f1e8; color: #1d252b; }}
    h1 {{ margin-bottom: 8px; }}
    .hint {{ background: #fff; border: 1px solid #ded4c2; border-radius: 14px; padding: 14px 16px; margin: 16px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #d8d0c2; border-radius: 14px; overflow: hidden; box-shadow: 0 8px 22px rgba(0,0,0,.06); }}
    .top_gain, .new_temporal_hit {{ border-top: 5px solid #2f9e44; }}
    .top_loss, .lost_temporal_hit {{ border-top: 5px solid #d9480f; }}
    img {{ width: 100%; display: block; background: #ddd; }}
    .meta {{ padding: 12px; }}
    .meta h3 {{ margin: 0 0 8px; font-size: 15px; word-break: break-all; }}
    .meta p {{ margin: 5px 0; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>HMDB51 Smoke Frame Review</h1>
  <div class="hint">
    <p>复查目标：确认 hard 模型提升/退化的原因，并筛选可人工修框加入 hard-case 训练的帧。</p>
    <p>建议在 CSV 中填写 review_label、issue_tags、notes。issue_tags 可用：small_cigarette, smoke_occlusion, low_resolution, hand_mouth_false_positive, missed_cigarette, usable_for_training, discard。</p>
  </div>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_dir = ensure_exists(args.report_dir, "Temporal report directory")
    video_dir = ensure_exists(args.video_dir, "Video directory")
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    picks = load_picks(report_dir / "hmdb51_smoke_temporal_delta_summary.json", args)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for pick in picks:
        video_path = video_dir / pick.video_name
        if not video_path.exists():
            failures.append({"video": pick.video_name, "error": "missing video"})
            continue
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or pick.frames)
        cap.release()
        for frame_index in frame_indices(total_frames, args.frames_per_video):
            sample_id = f"{pick.group}__{video_path.stem}__f{frame_index:05d}"
            rel_frame_path = Path("frames") / pick.group / f"{sample_id}.jpg"
            out_frame = output_dir / rel_frame_path
            try:
                width, height, timestamp = write_frame(video_path, frame_index, out_frame)
            except Exception as exc:
                failures.append({"video": pick.video_name, "frame": str(frame_index), "error": repr(exc)})
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "group": pick.group,
                    "video_name": pick.video_name,
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp, 3),
                    "width": width,
                    "height": height,
                    "delta_smoothed_hit_frames": pick.delta_smoothed,
                    "review_label": "",
                    "issue_tags": "",
                    "notes": "",
                    "frame_path": rel_frame_path.as_posix(),
                }
            )

    write_csv(output_dir / "frame_review_manifest.csv", rows)
    write_html(output_dir / "index.html", rows)
    dump_json(
        output_dir / "frame_review_summary.json",
        {
            "report_dir": str(report_dir),
            "video_dir": str(video_dir),
            "output_dir": str(output_dir),
            "selected_video_entries": len(picks),
            "frame_count": len(rows),
            "failures": failures,
            "groups": {group: sum(1 for row in rows if row["group"] == group) for group in sorted({row["group"] for row in rows})},
        },
    )
    print(f"Frame review pack saved to: {output_dir}")
    print(f"Frames: {len(rows)}, failures: {len(failures)}")


if __name__ == "__main__":
    main()
