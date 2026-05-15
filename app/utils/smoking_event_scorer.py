from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --- Scoring Constants ---

BASE_SCORES: dict[frozenset[int], int] = {
    frozenset({0, 1, 2}): 90,   # cigarette + smoking_person + smoke
    frozenset({0, 1}): 75,      # cigarette + smoking_person
    frozenset({0, 2}): 60,      # cigarette + smoke
    frozenset({1, 2}): 50,      # smoking_person + smoke
    frozenset({0}): 40,         # cigarette only
    frozenset({1}): 30,         # smoking_person only
    frozenset({2}): 25,         # smoke only
}

SPATIAL_BONUS_IOU_THRESHOLD = 0.1
SPATIAL_BONUS_VALUE = 0.15
PERSON_BOX_EXPAND_RATIO = 0.15

CLASSIFICATION_THRESHOLDS = {
    "confirmed": 70,
    "suspected": 50,
    "low_confidence": 30,
}

TEMPORAL_BONUS_PER_FRAME = 0.1
TEMPORAL_BONUS_MAX = 1.0


@dataclass
class SmokingEventScore:
    base_score: float = 0.0
    confidence_weight: float = 0.0
    spatial_bonus: float = 0.0
    temporal_bonus: float = 0.0
    final_score: float = 0.0
    classification: str = "ignore"
    evidence_classes: list[int] = field(default_factory=list)


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _expand_box(box: list[float], ratio: float) -> list[float]:
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return [
        box[0] - width * ratio,
        box[1] - height * ratio,
        box[2] + width * ratio,
        box[3] + height * ratio,
    ]


def _point_in_box(point: tuple[float, float], box: list[float]) -> bool:
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _has_cigarette_person_spatial_relation(cigarette_box: list[float], person_box: list[float]) -> bool:
    if len(cigarette_box) != 4 or len(person_box) != 4:
        return False
    if _compute_iou(cigarette_box, person_box) > SPATIAL_BONUS_IOU_THRESHOLD:
        return True
    cigarette_center = _box_center(cigarette_box)
    expanded_person = _expand_box(person_box, PERSON_BOX_EXPAND_RATIO)
    return _point_in_box(cigarette_center, expanded_person)


def calculate_smoking_score(
    detections: list[Any],
    consecutive_hits: int = 1,
) -> SmokingEventScore:
    if not detections:
        return SmokingEventScore()

    present_classes = frozenset(d.class_id for d in detections)
    evidence_classes = sorted(present_classes)

    base_score = 0
    for combo, score in sorted(BASE_SCORES.items(), key=lambda x: -x[1]):
        if combo <= present_classes:
            base_score = score
            break

    confidences = [d.confidence for d in detections]
    confidence_weight = sum(confidences) / len(confidences) if confidences else 0.0

    spatial_bonus = 0.0
    cigarette_boxes = [d.xyxy for d in detections if d.class_id == 0 and d.xyxy]
    person_boxes = [d.xyxy for d in detections if d.class_id == 1 and d.xyxy]
    for cig in cigarette_boxes:
        for person in person_boxes:
            if _has_cigarette_person_spatial_relation(cig, person):
                spatial_bonus = SPATIAL_BONUS_VALUE
                break
        if spatial_bonus > 0:
            break

    weighted_score = base_score * confidence_weight * (1 + spatial_bonus)

    temporal_bonus = min(consecutive_hits * TEMPORAL_BONUS_PER_FRAME, TEMPORAL_BONUS_MAX)
    final_score = min(weighted_score * (1 + temporal_bonus), 100.0)

    classification = classify_smoking_event(final_score)

    return SmokingEventScore(
        base_score=base_score,
        confidence_weight=confidence_weight,
        spatial_bonus=spatial_bonus,
        temporal_bonus=temporal_bonus,
        final_score=final_score,
        classification=classification,
        evidence_classes=evidence_classes,
    )


def classify_smoking_event(score: float) -> str:
    if score >= CLASSIFICATION_THRESHOLDS["confirmed"]:
        return "confirmed"
    elif score >= CLASSIFICATION_THRESHOLDS["suspected"]:
        return "suspected"
    elif score >= CLASSIFICATION_THRESHOLDS["low_confidence"]:
        return "low_confidence"
    return "ignore"
