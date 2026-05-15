from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.db_models import AlertEvent, AlertRule, utcnow


class AlertManager:

    def __init__(self, session_factory: Any) -> None:
        self._session_factory = session_factory

    def should_trigger_alert(
        self,
        score: float,
        bbox: tuple[float, float, float, float],
        frame_index: int,
        fps: float,
        rule: AlertRule,
        duration_frames: int = 1,
        session: Session | None = None,
    ) -> bool:
        if not rule.enabled:
            return False
        if score < rule.score_threshold:
            return False
        if duration_frames < rule.min_duration_frames:
            return False
        if self.is_duplicate_alert(bbox, rule, session=session):
            return False
        if rule.monitor_zones or rule.ignore_zones:
            if not self.is_in_monitor_zone(bbox, rule.monitor_zones, rule.ignore_zones):
                return False
        return True

    def create_alert_event(
        self,
        session: Session,
        *,
        video_task_id: int | None,
        score: float,
        severity: str,
        start_frame: int,
        end_frame: int,
        duration_seconds: float,
        bbox: tuple[float, float, float, float],
        thumbnail_path: str = "",
    ) -> AlertEvent:
        event = AlertEvent(
            video_task_id=video_task_id,
            alert_type="smoking",
            severity=severity,
            score=score,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_seconds=duration_seconds,
            bbox_x1=bbox[0],
            bbox_y1=bbox[1],
            bbox_x2=bbox[2],
            bbox_y2=bbox[3],
            thumbnail_path=thumbnail_path,
            status="pending",
        )
        session.add(event)
        session.flush()
        return event

    def is_duplicate_alert(
        self,
        bbox: tuple[float, float, float, float],
        rule: AlertRule,
        session: Session | None = None,
    ) -> bool:
        cooldown = rule.cooldown_seconds
        if cooldown <= 0:
            return False

        owns_session = session is None
        session = session or self._session_factory()
        try:
            cutoff = utcnow() - timedelta(seconds=cooldown)

            recent = (
                session.query(AlertEvent)
                .filter(AlertEvent.created_at >= cutoff)
                .filter(AlertEvent.alert_type == "smoking")
                .all()
            )
            for existing in recent:
                existing_bbox = (existing.bbox_x1, existing.bbox_y1, existing.bbox_x2, existing.bbox_y2)
                if self._bbox_iou(bbox, existing_bbox) > 0.5:
                    return True
            return False
        finally:
            if owns_session:
                session.close()

    def is_in_monitor_zone(
        self,
        bbox: tuple[float, float, float, float],
        monitor_zones: list | None,
        ignore_zones: list | None,
    ) -> bool:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        if ignore_zones:
            for zone in ignore_zones:
                if self._point_in_polygon(cx, cy, zone):
                    return False

        if monitor_zones:
            for zone in monitor_zones:
                if self._point_in_polygon(cx, cy, zone):
                    return True
            return False

        return True

    @staticmethod
    def _bbox_iou(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
