from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "SMOKER_DB_URL",
    f"sqlite:///{(ROOT / 'tmp' / 'system_smoke_test.db').as_posix()}",
)

from fastapi.testclient import TestClient

from app.web_demo import app
from scripts.yolo_utils import dump_json


MOJIBAKE_MARKERS = (
    "????",
    "\ufffd",
    "\u935a",  # common mojibake marker seen when UTF-8 is decoded incorrectly
    "\u59ab\u20ac",
    "\u9418\u8235",
)

EXPECTED_INDEX_PHRASES = (
    "\u5438\u70df\u884c\u4e3a\u68c0\u6d4b\u540e\u53f0",
    "\u89c6\u9891\u4efb\u52a1",
    "\u4efb\u52a1\u961f\u5217",
    "\u8f93\u51fa",
)

EXPECTED_APP_JS_SNIPPETS = (
    "查看中文报告",
    "播放视频",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke checks for the FastAPI smoker-behavior admin.")
    parser.add_argument(
        "--sample-image",
        default="datasets/final/smoke_bal/images/val/ai_smoke_a188.jpg",
        help="Local sample image used for image-detection checks.",
    )
    parser.add_argument(
        "--output",
        default="runs/reports/system_smoke_test.json",
        help="JSON output path for the system smoke-test report.",
    )
    return parser.parse_args()


def assert_ok(response, label: str) -> dict[str, Any]:
    if response.status_code not in (200, 202):
        raise RuntimeError(f"{label} failed with status {response.status_code}: {response.text}")
    return response.json()


def assert_no_mojibake(text: str, label: str) -> None:
    markers = [marker for marker in MOJIBAKE_MARKERS if marker in text]
    if markers:
        raise RuntimeError(f"{label} contains mojibake markers: {markers}")


def assert_contains_all(text: str, expected: tuple[str, ...], label: str) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise RuntimeError(f"{label} is missing expected frontend text: {missing}")


def build_sample_video(image_path: Path) -> Path:
    output = ROOT / "tmp" / "system_smoke_test_video.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Failed to load smoke-test image: {image_path}")
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (width, height))
    for idx in range(6):
        canvas = frame.copy()
        cv2.putText(
            canvas,
            f"frame-{idx + 1}",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()
    return output


def poll_task(client: TestClient, task_id: int, timeout_sec: float = 45.0) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        payload = assert_ok(client.get(f"/api/tasks/video/{task_id}"), "Video task detail")
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.75)
    raise TimeoutError(f"Timed out waiting for video task {task_id}")


def main() -> None:
    args = parse_args()
    image_path = ROOT / args.sample_image
    if not image_path.exists():
        raise FileNotFoundError(f"Sample image not found: {image_path}")
    video_path = build_sample_video(image_path)

    with TestClient(app) as client:
        health = assert_ok(client.get("/api/health"), "Health API")
        dashboard = assert_ok(client.get("/api/dashboard"), "Dashboard API")
        models = assert_ok(client.get("/api/models"), "Models API")
        settings = assert_ok(client.get("/api/settings"), "Settings API")

        index_status = client.get("/").status_code
        if index_status != 200:
            raise RuntimeError(f"Index page failed with status {index_status}")
        index_html = client.get("/").text
        assert_no_mojibake(index_html, "Index page")
        assert_contains_all(index_html, EXPECTED_INDEX_PHRASES, "Index page")
        if "js/app.js?v=" not in index_html:
            raise RuntimeError("Index page does not cache-bust app.js; browsers may keep mojibake link text.")
        app_js = client.get("/static/js/app.js").text
        assert_no_mojibake(app_js, "Frontend app.js")
        assert_contains_all(app_js, EXPECTED_APP_JS_SNIPPETS, "Frontend app.js")

        with image_path.open("rb") as handle:
            detect = assert_ok(
                client.post(
                    "/api/detect/image",
                    files={"file": (image_path.name, handle, "image/jpeg")},
                    data={"conf": "0.25", "iou": "0.45"},
                ),
                "Detection API",
            )

        records = assert_ok(client.get("/api/records?limit=5"), "Records API")
        record_detail = assert_ok(client.get(f"/api/records/{detect['record_id']}"), "Record detail API")

        with video_path.open("rb") as handle:
            task_create = assert_ok(
                client.post(
                    "/api/tasks/video",
                    files={"file": (video_path.name, handle, "video/mp4")},
                    data={"conf": "0.25", "iou": "0.45"},
                ),
                "Video task create API",
            )
        video_detail = poll_task(client, task_create["task"]["id"])
        video_report_response = client.get(f"/reports/video/{task_create['task']['id']}")
        video_report_status = video_report_response.status_code
        if video_report_status != 200:
            raise RuntimeError(f"Video report page failed with status {video_report_status}")
        assert_no_mojibake(video_report_response.text, "Video report page")

        invalid = client.post(
            "/api/detect/image",
            files={"file": ("invalid.txt", b"not-an-image", "text/plain")},
        )

        report = {
            "health": health,
            "dashboard_keys": sorted(dashboard.keys()),
            "models_count": len(models.get("items", [])),
            "settings": settings,
            "index_status": index_status,
            "sample_image": str(image_path),
            "image_record_id": detect.get("record_id"),
            "detection_count": detect.get("num_detections", 0),
            "records_count": len(records.get("items", [])),
            "record_detail_boxes": len(record_detail.get("boxes", [])),
            "video_task_status": video_detail.get("status"),
            "video_task_progress": video_detail.get("progress"),
            "video_task_output": video_detail.get("output_video_url"),
            "video_report_status": video_report_status,
            "video_task_summary": {
                "num_detections": video_detail.get("summary", {}).get("num_detections"),
                "raw_num_detections": video_detail.get("summary", {}).get("raw_num_detections"),
                "smoothed_num_detections": video_detail.get("summary", {}).get("smoothed_num_detections"),
                "stable_track_count": video_detail.get("summary", {}).get("stable_track_count"),
                "flicker_suppressed_count": video_detail.get("summary", {}).get("flicker_suppressed_count"),
                "temporal_event_hit": video_detail.get("summary", {}).get("temporal_event_hit"),
            },
            "invalid_upload_status": invalid.status_code,
            "invalid_upload_detail": invalid.json().get("detail"),
        }

    dump_json(args.output, report)
    print(f"System smoke-test report saved to: {Path(args.output)}")


if __name__ == "__main__":
    main()
