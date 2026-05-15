from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.alert_manager import AlertManager
from app.utils.smoking_event_scorer import SPATIAL_BONUS_VALUE, calculate_smoking_score
from app.utils.web_inference import DetectionBox, DetectionService


class SmokingEventScorerTests(unittest.TestCase):
    def test_cigarette_center_inside_person_box_gets_spatial_bonus(self) -> None:
        detections = [
            SimpleNamespace(class_id=0, confidence=0.8, xyxy=[90.0, 90.0, 110.0, 100.0]),
            SimpleNamespace(class_id=1, confidence=0.8, xyxy=[50.0, 50.0, 250.0, 400.0]),
        ]

        score = calculate_smoking_score(detections, consecutive_hits=1)

        self.assertEqual(score.spatial_bonus, SPATIAL_BONUS_VALUE)


class WebInferencePostProcessingTests(unittest.TestCase):
    def test_isolated_smoking_person_below_threshold_is_removed_after_penalty(self) -> None:
        service = DetectionService.__new__(DetectionService)
        detections = [
            DetectionBox(
                class_id=1,
                class_name="smoking_person",
                confidence=0.16,
                xyxy=[10.0, 10.0, 90.0, 180.0],
            )
        ]

        filtered = service._apply_post_processing_rules(detections)

        self.assertEqual(filtered, [])

    def test_contiguous_smoking_frames_are_merged_into_one_event_with_bbox(self) -> None:
        service = DetectionService.__new__(DetectionService)
        events: list[dict] = []
        detections = [
            DetectionBox(1, "smoking_person", 0.8, [10.0, 20.0, 100.0, 220.0]),
            DetectionBox(0, "cigarette", 0.7, [40.0, 80.0, 55.0, 88.0]),
        ]

        service._append_smoking_event(
            events,
            frame_index=10,
            fps=25.0,
            detections=detections,
            score=72.0,
            classification="confirmed",
            evidence_classes=[0, 1],
        )
        service._append_smoking_event(
            events,
            frame_index=11,
            fps=25.0,
            detections=detections,
            score=75.0,
            classification="confirmed",
            evidence_classes=[0, 1],
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_frame"], 10)
        self.assertEqual(events[0]["end_frame"], 11)
        self.assertEqual(events[0]["duration_frames"], 2)
        self.assertEqual(events[0]["bbox"], [10.0, 20.0, 100.0, 220.0])

    def test_stable_cigarette_track_is_promoted_to_alert_event_even_when_frame_score_is_ignore(self) -> None:
        service = DetectionService.__new__(DetectionService)
        events: list[dict] = []
        detections = [
            DetectionBox(0, "cigarette", 0.16, [100.0, 120.0, 180.0, 150.0]),
        ]
        frame_score = calculate_smoking_score(detections, consecutive_hits=10)
        self.assertEqual(frame_score.classification, "ignore")

        emitted = service._append_temporal_alert_event_if_needed(
            events,
            frame_index=10,
            fps=25.0,
            detections=detections,
            frame_score=frame_score,
            consecutive_hits=10,
        )

        self.assertTrue(emitted)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["classification"], "suspected")
        self.assertEqual(events[0]["score"], 70.0)
        self.assertEqual(events[0]["bbox"], [100.0, 120.0, 180.0, 150.0])


class AlertManagerTests(unittest.TestCase):
    def test_rule_min_duration_is_enforced_before_duplicate_check(self) -> None:
        manager = AlertManager(lambda: self.fail("duplicate check should not run"))
        rule = SimpleNamespace(
            enabled=True,
            score_threshold=70.0,
            min_duration_frames=3,
            monitor_zones=None,
            ignore_zones=None,
        )

        should_trigger = manager.should_trigger_alert(
            80.0,
            (0.0, 0.0, 100.0, 100.0),
            frame_index=10,
            fps=25.0,
            rule=rule,
            duration_frames=2,
        )

        self.assertFalse(should_trigger)


if __name__ == "__main__":
    unittest.main()
