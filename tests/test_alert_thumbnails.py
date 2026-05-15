from pathlib import Path
import sys
import unittest
from uuid import uuid4

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import session_scope
from app.db_models import AlertEvent
from app.web_demo import ALERT_THUMBNAIL_DIR, app, create_alert_thumbnail_from_video


class AlertThumbnailTests(unittest.TestCase):
    def test_create_alert_thumbnail_from_video_crops_bbox(self) -> None:
        run_id = uuid4().hex
        video_path = ALERT_THUMBNAIL_DIR / f"test_source_{run_id}.mp4"
        thumb_dir = ALERT_THUMBNAIL_DIR / "test_thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: video_path.unlink(missing_ok=True))

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[30:90, 50:110] = (0, 255, 0)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120))
        writer.write(frame)
        writer.release()

        output = create_alert_thumbnail_from_video(
            source_video_path=video_path,
            event={"start_frame": 1, "bbox": [50.0, 30.0, 110.0, 90.0]},
            output_dir=thumb_dir,
            prefix=run_id,
        )

        self.assertTrue(output.exists())
        image = cv2.imread(str(output))
        self.assertIsNotNone(image)
        self.assertLess(image.shape[0], 120)
        self.assertLess(image.shape[1], 160)
        output.unlink(missing_ok=True)

    def test_alert_events_api_returns_thumbnail_url(self) -> None:
        thumb_path = ALERT_THUMBNAIL_DIR / f"api_thumb_{uuid4().hex}.jpg"
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        thumb_path.write_bytes(b"fake jpg bytes")
        event_id = None
        try:
            with session_scope() as session:
                event = AlertEvent(
                    video_task_id=None,
                    alert_type="smoking",
                    severity="confirmed",
                    score=88.0,
                    start_frame=1,
                    end_frame=3,
                    duration_seconds=0.12,
                    bbox_x1=10.0,
                    bbox_y1=20.0,
                    bbox_x2=80.0,
                    bbox_y2=140.0,
                    thumbnail_path=str(thumb_path),
                    status="pending",
                )
                session.add(event)
                session.flush()
                event_id = event.id

            with TestClient(app) as client:
                payload = client.get("/api/alerts/events").json()

            event_payload = next(item for item in payload["events"] if item["id"] == event_id)
            self.assertEqual(event_payload["thumbnail_url"], f"/artifacts/alerts/{thumb_path.name}")
        finally:
            if event_id is not None:
                with session_scope() as session:
                    event = session.get(AlertEvent, event_id)
                    if event is not None:
                        session.delete(event)
            thumb_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
