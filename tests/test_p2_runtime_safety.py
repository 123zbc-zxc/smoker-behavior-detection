from pathlib import Path
from types import SimpleNamespace
import asyncio
import base64
import os
import warnings
import sys
import unittest

from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.web_inference import DEFAULT_INFERENCE_CONF
from app.db_models import utcnow
from app.web_demo import create_request_detection_service, read_upload_bytes_limited, resolve_detection_thresholds, service


class RuntimeSafetyTests(unittest.TestCase):
    def test_confidence_and_iou_are_validated_server_side(self) -> None:
        settings = SimpleNamespace(default_conf=DEFAULT_INFERENCE_CONF, default_iou=0.35)

        with self.assertRaises(HTTPException) as low_conf:
            resolve_detection_thresholds(conf=0.0, iou=0.35, settings=settings)
        self.assertEqual(low_conf.exception.status_code, 400)

        with self.assertRaises(HTTPException) as bad_iou:
            resolve_detection_thresholds(conf=DEFAULT_INFERENCE_CONF, iou=1.5, settings=settings)
        self.assertEqual(bad_iou.exception.status_code, 400)

    def test_missing_form_values_use_persisted_settings(self) -> None:
        settings = SimpleNamespace(default_conf=0.2, default_iou=0.4)

        conf, iou = resolve_detection_thresholds(conf=None, iou=None, settings=settings)

        self.assertEqual(conf, 0.2)
        self.assertEqual(iou, 0.4)

    def test_request_detection_service_does_not_reuse_global_singleton(self) -> None:
        request_service = create_request_detection_service(
            weights_path=str(service.weights_path),
            imgsz=640,
            max_upload_mb=12,
        )

        self.assertIsNot(request_service, service)
        self.assertEqual(request_service.weights_path, service.weights_path)
        self.assertEqual(request_service.imgsz, 640)
        self.assertEqual(request_service.max_upload_bytes, 12 * 1024 * 1024)

    def test_image_endpoint_rejects_invalid_conf_before_inference(self) -> None:
        with TestClient(__import__("app.web_demo", fromlist=["app"]).app) as client:
            response = client.post(
                "/api/detect/image",
                files={"file": ("sample.jpg", b"not really an image", "image/jpeg")},
                data={"conf": "0", "iou": "0.35"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("conf must be between", response.json()["detail"])

    def test_video_endpoint_rejects_invalid_iou_as_400(self) -> None:
        with TestClient(__import__("app.web_demo", fromlist=["app"]).app) as client:
            response = client.post(
                "/api/tasks/video",
                files={"file": ("sample.mp4", b"not really a video", "video/mp4")},
                data={"conf": "0.12", "iou": "2"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("iou must be between", response.json()["detail"])

    def test_default_alert_rule_cannot_be_deleted(self) -> None:
        with TestClient(__import__("app.web_demo", fromlist=["app"]).app) as client:
            rules = client.get("/api/alerts/rules").json()
            default_rule = next(rule for rule in rules if rule["name"] == "\u9ed8\u8ba4\u5438\u70df\u544a\u8b66\u89c4\u5219")
            response = client.delete(f"/api/alerts/rules/{default_rule['id']}")

        self.assertEqual(response.status_code, 400)
        self.assertIn("Default alert rule", response.json()["detail"])

    def test_default_alert_rule_cannot_be_disabled(self) -> None:
        with TestClient(__import__("app.web_demo", fromlist=["app"]).app) as client:
            rules = client.get("/api/alerts/rules").json()
            default_rule = next(rule for rule in rules if rule["name"] == "\u9ed8\u8ba4\u5438\u70df\u544a\u8b66\u89c4\u5219")
            payload = {
                "name": default_rule["name"],
                "enabled": False,
                "score_threshold": default_rule["score_threshold"],
                "min_duration_frames": default_rule["min_duration_frames"],
                "cooldown_seconds": default_rule["cooldown_seconds"],
                "monitor_zones": default_rule["monitor_zones"],
                "ignore_zones": default_rule["ignore_zones"],
                "notification_channels": default_rule["notification_channels"],
            }
            response = client.put(f"/api/alerts/rules/{default_rule['id']}", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn("Default alert rule", response.json()["detail"])

    def test_optional_basic_auth_protects_admin_when_enabled(self) -> None:
        os.environ["SMOKER_ADMIN_PASSWORD"] = "secret"
        os.environ["SMOKER_ADMIN_USER"] = "admin"
        self.addCleanup(lambda: os.environ.pop("SMOKER_ADMIN_PASSWORD", None))
        self.addCleanup(lambda: os.environ.pop("SMOKER_ADMIN_USER", None))

        with TestClient(__import__("app.web_demo", fromlist=["app"]).app) as client:
            unauthorized = client.get("/api/health")
            token = base64.b64encode(b"admin:secret").decode("ascii")
            authorized = client.get("/api/health", headers={"Authorization": f"Basic {token}"})

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)

    def test_api_docs_are_disabled_by_default(self) -> None:
        with TestClient(__import__("app.web_demo", fromlist=["app"]).app) as client:
            self.assertEqual(client.get("/docs").status_code, 404)
            self.assertEqual(client.get("/openapi.json").status_code, 404)

    def test_chunked_upload_reader_rejects_oversized_payload(self) -> None:
        class FakeUpload:
            def __init__(self) -> None:
                self.calls = 0

            async def read(self, size: int = -1) -> bytes:
                self.calls += 1
                if self.calls == 1:
                    return b"a" * 600_000
                if self.calls == 2:
                    return b"b" * 600_000
                return b""

        async def run_case() -> None:
            with self.assertRaises(HTTPException) as ctx:
                await read_upload_bytes_limited(FakeUpload(), max_upload_mb=1, label="video")
            self.assertEqual(ctx.exception.status_code, 413)

        asyncio.run(run_case())

    def test_utcnow_has_no_datetime_utcnow_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            utcnow()

        self.assertFalse([item for item in caught if "utcnow" in str(item.message)])


if __name__ == "__main__":
    unittest.main()
