from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.alert_manager import AlertManager
from app.config import build_runtime_config
from app.db import bootstrap_alert_rules
from app.db_models import AlertRule, Base
from app.web_demo import app, artifact_url, safe_unlink_runtime_file


class WebSecurityRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.client.__enter__()
        cls.runtime = build_runtime_config()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_sqlite_database_is_not_exposed_as_artifact(self) -> None:
        response = self.client.get("/artifacts/smoker_behavior.db")

        self.assertEqual(response.status_code, 404)

    def test_artifact_url_only_exposes_result_files(self) -> None:
        result_path = self.runtime.image_result_dir / "safe.jpg"
        upload_path = self.runtime.image_upload_dir / "source.jpg"

        self.assertEqual(artifact_url(str(result_path)), "/artifacts/images/safe.jpg")
        self.assertEqual(artifact_url(str(upload_path)), "")

    def test_safe_unlink_rejects_paths_outside_allowed_runtime_dirs(self) -> None:
        outside = Path(tempfile.gettempdir()) / "smoker_should_not_delete.txt"
        outside.write_text("keep", encoding="utf-8")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))

        deleted = safe_unlink_runtime_file(outside, allowed_roots=[self.runtime.image_upload_dir])

        self.assertFalse(deleted)
        self.assertTrue(outside.exists())


class AlertRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def test_bootstrap_alert_rules_renames_mojibake_default_rule(self) -> None:
        with self.Session() as session:
            session.add(AlertRule(name="Ĭ�����̸澯����", enabled=False))
            session.commit()

            rule = bootstrap_alert_rules(session)
            session.commit()

            self.assertEqual(rule.name, "默认吸烟告警规则")
            self.assertTrue(rule.enabled)
            self.assertEqual(session.query(AlertRule).count(), 1)

    def test_duplicate_alert_uses_current_session_pending_event(self) -> None:
        with self.Session() as session:
            rule = AlertRule(name="默认吸烟告警规则", cooldown_seconds=60)
            session.add(rule)
            session.flush()
            manager = AlertManager(self.Session)
            manager.create_alert_event(
                session,
                video_task_id=None,
                score=80.0,
                severity="confirmed",
                start_frame=1,
                end_frame=3,
                duration_seconds=0.12,
                bbox=(10.0, 10.0, 100.0, 100.0),
            )

            duplicate = manager.is_duplicate_alert(
                (12.0, 12.0, 98.0, 98.0),
                rule,
                session=session,
            )

            self.assertTrue(duplicate)


if __name__ == "__main__":
    unittest.main()
