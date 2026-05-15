from __future__ import annotations

import base64
import shutil
import subprocess
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from scripts.yolo_utils import build_model, ensure_exists, project_root
from app.utils.smoking_event_scorer import calculate_smoking_score


ROOT = project_root()
DEFAULT_MAX_UPLOAD_MB = 10
DEFAULT_IMAGE_SIZE = 640
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TEMPORAL_MATCH_IOU = 0.25
TEMPORAL_STABLE_HITS = 3
TEMPORAL_BRIDGE_FRAMES = 2
TRACK_STALE_FRAMES = 5
CONFIDENCE_SMOOTH_ALPHA = 0.7
TEMPORAL_ALERT_SCORE_FLOOR = 70.0
CLASS_CONF_THRESHOLDS = {
    0: 0.12,
    1: 0.15,
    2: 0.28,
}
DEFAULT_INFERENCE_CONF = min(CLASS_CONF_THRESHOLDS.values())
PROJECT_CLASS_NAMES = {
    0: "cigarette",
    1: "smoking_person",
    2: "smoke",
}
BOX_COLORS = {
    0: (15, 73, 217),
    1: (214, 111, 29),
    2: (68, 158, 47),
}


@dataclass
class DetectionBox:
    class_id: int
    class_name: str
    confidence: float
    xyxy: list[float]


@dataclass
class TrackState:
    track_id: int
    class_id: int
    class_name: str
    xyxy: list[float]
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    consecutive_hits: int = 1
    total_hits: int = 1
    is_stable: bool = False
    smoothed_confidence: float = 0.0


class DetectionService:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        imgsz: int = DEFAULT_IMAGE_SIZE,
        max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB,
    ) -> None:
        self.weights_path = self._resolve_weights(weights_path)
        self.imgsz = imgsz
        self.max_upload_bytes = max_upload_mb * 1024 * 1024
        self._model: Any | None = None
        self._loaded_weights_path: Path | None = None
        self._predict_lock = threading.Lock()

    def _resolve_weights(self, weights_path: str | Path | None) -> Path:
        if weights_path:
            return ensure_exists(weights_path, "Web demo weights")
        candidates = self.available_weight_candidates()
        if not candidates:
            raise FileNotFoundError("No available weight file was found for the web demo.")
        return candidates[0]["path"]

    def available_weight_candidates(self) -> list[dict[str, Any]]:
        candidates = [
            (
                ROOT / "runs" / "imported" / "yolov8n_colab_640_hard_candidate_20260502" / "train" / "weights" / "best.pt",
                "YOLOv8n 640px Hard Fine-tune (20260502)",
                "Current default: higher cigarette recall and better full-video temporal stability on HMDB51 smoke.",
            ),
            (
                ROOT / "runs" / "imported" / "smoker_weights_20260429" / "best.pt",
                "YOLOv8n 640px Colab Best (imported_20260429)",
                "Former champion kept for metric comparison; hard fine-tune is now the web default.",
            ),
            (
                ROOT / "runs" / "imported" / "smoker_weights_20260429" / "last.pt",
                "YOLOv8n 640px Colab Last (imported_20260429)",
                "Latest epoch-71 checkpoint retained for comparison; best.pt remains the default.",
            ),
            (
                ROOT / "runs" / "train" / "yolov8n_balanced_512" / "weights" / "best.pt",
                "YOLOv8n 512px (balanced_512)",
                "Optimized 512px model with improved small-object cigarette detection.",
            ),
            (
                ROOT / "runs" / "train" / "yolov8n_balanced_512_augment" / "weights" / "best.pt",
                "YOLOv8n 512px Augment (balanced_512_augment)",
                "512px model with small-target-friendly augmentation strategy.",
            ),
            (
                ROOT / "runs" / "train" / "yolov8n_balanced_fast" / "weights" / "best.pt",
                "YOLOv8n Fast (balanced_fast)",
                "Accelerated training run with workers/cache optimizations.",
            ),
            (
                ROOT / "runs" / "train" / "yolov8n_balanced_30" / "weights" / "best.pt",
                "YOLOv8n Baseline (balanced_30)",
                "Current strongest validated CPU baseline for the thesis demo.",
            ),
            (
                ROOT / "runs" / "train" / "yolov8n_eca_balanced_304" / "weights" / "best.pt",
                "YOLOv8n + ECA (balanced_304)",
                "Attention comparison checkpoint retained for quick switching in the web demo.",
            ),
            (
                ROOT / "runs" / "train" / "yolov8n_se_balanced" / "weights" / "best.pt",
                "YOLOv8n + SE (balanced)",
                "SE-attention comparison checkpoint for CPU-oriented experiments.",
            ),
            (
                ROOT / "yolov8n.pt",
                "Ultralytics YOLOv8n",
                "Official fallback weights for debugging and cold-start checks.",
            ),
        ]
        items: list[dict[str, Any]] = []
        for path, name, note in candidates:
            if path.exists():
                items.append(
                    {
                        "name": name,
                        "weights_path": str(path),
                        "note": note,
                        "device": "cpu",
                        "is_available": True,
                        "path": path,
                    }
                )
        return items

    def use_runtime_options(
        self,
        *,
        weights_path: str | Path | None = None,
        imgsz: int | None = None,
        max_upload_mb: int | None = None,
    ) -> None:
        if weights_path is not None:
            resolved = ensure_exists(weights_path, "Web demo weights")
            if self._loaded_weights_path is not None and resolved != self._loaded_weights_path:
                self._model = None
            self.weights_path = resolved
        if imgsz is not None:
            self.imgsz = imgsz
        if max_upload_mb is not None:
            self.max_upload_bytes = max_upload_mb * 1024 * 1024

    def load_model(self) -> Any:
        if self._model is None or self._loaded_weights_path != self.weights_path:
            self._model = build_model(str(self.weights_path))
            self._loaded_weights_path = self.weights_path
        return self._model

    def model_info(self) -> dict[str, Any]:
        return {
            "weights_path": str(self.weights_path),
            "imgsz": self.imgsz,
            "device": "cpu",
            "loaded": self._model is not None,
            "max_upload_mb": self.max_upload_bytes // (1024 * 1024),
            "temporal_stable_hits": TEMPORAL_STABLE_HITS,
            "temporal_bridge_frames": TEMPORAL_BRIDGE_FRAMES,
        }

    def validate_upload(
        self,
        filename: str | None,
        payload: bytes,
        *,
        allowed_suffixes: set[str],
        label: str,
    ) -> None:
        if not payload:
            raise ValueError(f"Uploaded {label} is empty.")
        if len(payload) > self.max_upload_bytes:
            raise ValueError(
                f"Uploaded {label} exceeds {self.max_upload_bytes // (1024 * 1024)} MB limit."
            )
        if filename:
            suffix = Path(filename).suffix.lower()
            if suffix and suffix not in allowed_suffixes:
                raise ValueError(
                    f"Unsupported {label} suffix `{suffix}`. Allowed: {sorted(allowed_suffixes)}."
                )

    def detect_image_bytes(
        self,
        image_bytes: bytes,
        filename: str | None = None,
        *,
        conf: float = DEFAULT_INFERENCE_CONF,
        iou: float = 0.35,
        weights_path: str | Path | None = None,
        imgsz: int | None = None,
    ) -> dict[str, Any]:
        self.validate_upload(filename, image_bytes, allowed_suffixes=ALLOWED_IMAGE_SUFFIXES, label="image")
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode uploaded image.")

        if weights_path is not None or imgsz is not None:
            self.use_runtime_options(weights_path=weights_path, imgsz=imgsz)

        result = self._predict_frame(frame, conf=conf, iou=iou)
        plotted = result.plot()
        ok, encoded = cv2.imencode(".jpg", plotted)
        if not ok:
            raise RuntimeError("Failed to encode plotted result image.")

        detections = self._extract_detections(result)
        detections = self._apply_post_processing_rules(detections)
        encoded_bytes = encoded.tobytes()
        event_score = calculate_smoking_score(detections, consecutive_hits=1)
        return {
            "model": self.model_info(),
            "detections": [self._box_to_dict(det) for det in detections],
            "num_detections": len(detections),
            "annotated_image_base64": base64.b64encode(encoded_bytes).decode("ascii"),
            "_annotated_image_bytes": encoded_bytes,
            "smoking_event": {
                "score": round(event_score.final_score, 1),
                "classification": event_score.classification,
                "evidence_classes": event_score.evidence_classes,
            },
        }

    def process_video_file(
        self,
        source_path: str | Path,
        output_path: str | Path,
        *,
        conf: float = DEFAULT_INFERENCE_CONF,
        iou: float = 0.35,
        weights_path: str | Path | None = None,
        imgsz: int | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
        frame_callback: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        source = ensure_exists(source_path, "Video source")
        if source.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video suffix `{source.suffix}`.")
        if weights_path is not None or imgsz is not None:
            self.use_runtime_options(weights_path=weights_path, imgsz=imgsz)

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {source}")

        total_frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        fps = capture.get(cv2.CAP_PROP_FPS) or 12.0
        width = max(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 1)
        height = max(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        raw_output = output.with_name(f"{output.stem}_raw.mp4")
        writer = cv2.VideoWriter(
            str(raw_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame_index = 0
        raw_num_detections = 0
        smoothed_num_detections = 0
        raw_per_class = Counter[str]()
        smoothed_per_class = Counter[str]()
        peak_confidence = 0.0
        tracks: list[TrackState] = []
        next_track_id = 1
        stable_track_ids: set[int] = set()
        flicker_suppressed_count = 0
        smoking_events: list[dict] = []
        _event_consecutive_hits = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                result = self._predict_frame(frame, conf=conf, iou=iou)
                detections = self._extract_detections(result)
                detections = self._apply_post_processing_rules(detections)
                rendered_detections, next_track_id, bridged = self._temporal_filter_detections(
                    tracks,
                    detections,
                    frame_index,
                    next_track_id,
                    stable_track_ids,
                )
                plotted = self._draw_detections(frame, rendered_detections)
                writer.write(plotted)

                raw_num_detections += len(detections)
                smoothed_num_detections += len(rendered_detections)
                flicker_suppressed_count += bridged

                for det in detections:
                    raw_per_class[det.class_name] += 1
                    peak_confidence = max(peak_confidence, det.confidence)
                for det in rendered_detections:
                    smoothed_per_class[det.class_name] += 1

                if rendered_detections:
                    _event_consecutive_hits += 1
                else:
                    _event_consecutive_hits = 0
                frame_score = calculate_smoking_score(rendered_detections, _event_consecutive_hits)
                self._append_temporal_alert_event_if_needed(
                    smoking_events,
                    frame_index=frame_index,
                    fps=fps,
                    detections=rendered_detections,
                    frame_score=frame_score,
                    consecutive_hits=_event_consecutive_hits,
                )

                if frame_callback:
                    frame_callback(frame_index, len(detections), len(rendered_detections))
                if progress_callback and (frame_index == total_frames or frame_index % 5 == 0):
                    progress_callback(frame_index, total_frames, smoothed_num_detections)
        finally:
            capture.release()
            writer.release()
        browser_video_path = self._finalize_browser_video(raw_output, output)

        return {
            "total_frames": total_frames or frame_index,
            "processed_frames": frame_index,
            "fps": round(float(fps), 4),
            "num_detections": smoothed_num_detections,
            "raw_num_detections": raw_num_detections,
            "smoothed_num_detections": smoothed_num_detections,
            "per_class_counts": dict(smoothed_per_class),
            "raw_per_class_counts": dict(raw_per_class),
            "peak_confidence": round(peak_confidence, 4),
            "stable_track_count": len(stable_track_ids),
            "flicker_suppressed_count": flicker_suppressed_count,
            "temporal_event_hit": bool(stable_track_ids),
            "smoking_events": smoking_events,
            "temporal_parameters": {
                "match_iou": TEMPORAL_MATCH_IOU,
                "stable_hits": TEMPORAL_STABLE_HITS,
                "bridge_frames": TEMPORAL_BRIDGE_FRAMES,
                "track_stale_frames": TRACK_STALE_FRAMES,
            },
            "output_video_path": str(browser_video_path),
            "raw_output_video_path": str(raw_output),
        }

    def _finalize_browser_video(self, raw_output: Path, output: Path) -> Path:
        ffmpeg = self._find_ffmpeg_executable()
        if not ffmpeg:
            if raw_output != output:
                raw_output.replace(output)
            return output
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(raw_output),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output
        except (OSError, subprocess.CalledProcessError):
            if raw_output != output:
                raw_output.replace(output)
            return output

    @staticmethod
    def _find_ffmpeg_executable() -> str | None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        try:
            import imageio_ffmpeg  # type: ignore[import-not-found]
        except ImportError:
            return None
        return imageio_ffmpeg.get_ffmpeg_exe()

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "weights_path": str(self.weights_path),
            "weights_exists": self.weights_path.exists(),
            "model_loaded": self._model is not None,
            "imgsz": self.imgsz,
            "max_upload_mb": self.max_upload_bytes // (1024 * 1024),
        }

    def _predict_frame(self, frame: np.ndarray, *, conf: float, iou: float) -> Any:
        model = self.load_model()
        with self._predict_lock:
            results = model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=conf,
                iou=iou,
                device="cpu",
                verbose=False,
            )
        if not results:
            raise RuntimeError("Model returned no prediction results.")
        return results[0]

    def _extract_detections(self, result: Any) -> list[DetectionBox]:
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        detections: list[DetectionBox] = []
        if boxes is None or boxes.cls is None:
            return detections

        xyxy = boxes.xyxy.tolist() if boxes.xyxy is not None else []
        for idx, (cls_id, score) in enumerate(zip(boxes.cls.tolist(), boxes.conf.tolist())):
            cls_id_int = int(cls_id)
            if float(score) < CLASS_CONF_THRESHOLDS.get(cls_id_int, 0.25):
                continue
            detections.append(
                DetectionBox(
                    class_id=cls_id_int,
                    class_name=PROJECT_CLASS_NAMES.get(cls_id_int, str(names.get(cls_id_int, cls_id))),
                    confidence=float(score),
                    xyxy=[float(value) for value in xyxy[idx]] if idx < len(xyxy) else [],
                )
            )
        return detections

    def _apply_post_processing_rules(self, detections: list[DetectionBox]) -> list[DetectionBox]:
        """Apply lightweight class-specific filters before temporal smoothing."""
        filtered: list[DetectionBox] = []

        for det in [d for d in detections if d.class_id == 0 and len(d.xyxy) == 4]:
            x1, y1, x2, y2 = det.xyxy
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            if 1.5 <= aspect_ratio <= 8.0:
                filtered.append(det)

        for det in [d for d in detections if d.class_id == 2 and len(d.xyxy) == 4]:
            x1, y1, x2, y2 = det.xyxy
            area = (x2 - x1) * (y2 - y1)
            if area >= 100:
                filtered.append(det)

        cigarette_boxes = [d.xyxy for d in detections if d.class_id == 0 and len(d.xyxy) == 4]
        smoke_boxes = [d.xyxy for d in detections if d.class_id == 2 and len(d.xyxy) == 4]
        for det in [d for d in detections if d.class_id == 1]:
            if len(det.xyxy) != 4:
                continue
            has_nearby = False
            x1, y1, x2, y2 = det.xyxy
            w, h = x2 - x1, y2 - y1
            expanded = [x1 - w * 0.25, y1 - h * 0.25, x2 + w * 0.25, y2 + h * 0.25]
            for other_box in cigarette_boxes + smoke_boxes:
                if self._boxes_overlap(expanded, other_box):
                    has_nearby = True
                    break
            if has_nearby:
                filtered.append(det)
            else:
                penalized = DetectionBox(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence * 0.6,
                    xyxy=det.xyxy,
                )
                if penalized.confidence >= CLASS_CONF_THRESHOLDS.get(penalized.class_id, 0.25):
                    filtered.append(penalized)

        return filtered

    def _append_smoking_event(
        self,
        events: list[dict[str, Any]],
        *,
        frame_index: int,
        fps: float,
        detections: list[DetectionBox],
        score: float,
        classification: str,
        evidence_classes: list[int],
    ) -> None:
        bbox = self._representative_event_bbox(detections)
        if not bbox:
            return

        safe_fps = max(float(fps), 1.0)
        if events and events[-1]["end_frame"] == frame_index - 1:
            event = events[-1]
            event["frame"] = frame_index
            event["end_frame"] = frame_index
            event["duration_frames"] = event["end_frame"] - event["start_frame"] + 1
            event["duration_seconds"] = round(event["duration_frames"] / safe_fps, 3)
            event["end_seconds"] = round(frame_index / safe_fps, 3)
            event["score"] = max(event["score"], score)
            event["classification"] = self._stronger_classification(event["classification"], classification)
            event["evidence_classes"] = sorted(set(event["evidence_classes"]) | set(evidence_classes))
            event["bbox"] = self._union_bbox([event["bbox"], bbox])
            return

        start_seconds = round(frame_index / safe_fps, 3)
        events.append(
            {
                "frame": frame_index,
                "start_frame": frame_index,
                "end_frame": frame_index,
                "duration_frames": 1,
                "duration_seconds": round(1 / safe_fps, 3),
                "start_seconds": start_seconds,
                "end_seconds": start_seconds,
                "score": score,
                "classification": classification,
                "evidence_classes": evidence_classes,
                "bbox": bbox,
            }
        )

    def _append_temporal_alert_event_if_needed(
        self,
        events: list[dict[str, Any]],
        *,
        frame_index: int,
        fps: float,
        detections: list[DetectionBox],
        frame_score: Any,
        consecutive_hits: int,
    ) -> bool:
        direct_event = frame_score.classification in ("confirmed", "suspected")
        stable_cigarette_event = consecutive_hits >= TEMPORAL_STABLE_HITS and 0 in set(
            frame_score.evidence_classes
        )
        if not direct_event and not stable_cigarette_event:
            return False

        score = round(frame_score.final_score, 1)
        classification = frame_score.classification
        if stable_cigarette_event and score < TEMPORAL_ALERT_SCORE_FLOOR:
            # A multi-frame stable cigarette track is stronger evidence than
            # a single-frame confidence score, so promote it to an alertable
            # suspected event for the dashboard rule engine.
            score = TEMPORAL_ALERT_SCORE_FLOOR
            classification = "suspected"

        before_count = len(events)
        before_end_frame = events[-1]["end_frame"] if events else None
        self._append_smoking_event(
            events,
            frame_index=frame_index,
            fps=fps,
            detections=detections,
            score=score,
            classification=classification,
            evidence_classes=frame_score.evidence_classes,
        )
        return len(events) > before_count or (bool(events) and events[-1]["end_frame"] != before_end_frame)

    @staticmethod
    def _representative_event_bbox(detections: list[DetectionBox]) -> list[float]:
        person_boxes = [d.xyxy for d in detections if d.class_id == 1 and len(d.xyxy) == 4]
        if person_boxes:
            return DetectionService._union_bbox(person_boxes)
        boxes = [d.xyxy for d in detections if d.class_id in {0, 2} and len(d.xyxy) == 4]
        return DetectionService._union_bbox(boxes)

    @staticmethod
    def _union_bbox(boxes: list[list[float]]) -> list[float]:
        valid = [box for box in boxes if len(box) == 4]
        if not valid:
            return []
        return [
            min(box[0] for box in valid),
            min(box[1] for box in valid),
            max(box[2] for box in valid),
            max(box[3] for box in valid),
        ]

    @staticmethod
    def _stronger_classification(left: str, right: str) -> str:
        rank = {"ignore": 0, "low_confidence": 1, "suspected": 2, "confirmed": 3}
        return left if rank.get(left, 0) >= rank.get(right, 0) else right

    @staticmethod
    def _boxes_overlap(a: list[float], b: list[float]) -> bool:
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    def _box_to_dict(self, det: DetectionBox) -> dict[str, Any]:
        return {
            "class_id": det.class_id,
            "class_name": det.class_name,
            "confidence": det.confidence,
            "xyxy": det.xyxy,
        }

    def _temporal_filter_detections(
        self,
        tracks: list[TrackState],
        detections: list[DetectionBox],
        frame_index: int,
        next_track_id: int,
        stable_track_ids: set[int],
    ) -> tuple[list[DetectionBox], int, int]:
        matches, unmatched_track_indices, unmatched_detection_indices = self._match_tracks(tracks, detections)
        rendered_detections: list[DetectionBox] = []
        bridged_count = 0

        for track_index, detection_index in matches:
            track = tracks[track_index]
            detection = detections[detection_index]
            gap = frame_index - track.last_seen_frame
            track.xyxy = detection.xyxy
            track.confidence = (track.confidence * 0.6) + (detection.confidence * 0.4)
            if track.smoothed_confidence == 0.0:
                track.smoothed_confidence = detection.confidence
            else:
                track.smoothed_confidence = (
                    CONFIDENCE_SMOOTH_ALPHA * track.smoothed_confidence
                    + (1 - CONFIDENCE_SMOOTH_ALPHA) * detection.confidence
                )
            track.last_seen_frame = frame_index
            track.total_hits += 1
            track.consecutive_hits = track.consecutive_hits + 1 if gap <= 1 else 1
            if not track.is_stable and track.consecutive_hits >= TEMPORAL_STABLE_HITS:
                track.is_stable = True
                stable_track_ids.add(track.track_id)
            if track.is_stable:
                rendered_detections.append(
                    DetectionBox(
                        class_id=track.class_id,
                        class_name=track.class_name,
                        confidence=track.smoothed_confidence if track.smoothed_confidence > 0.0 else track.confidence,
                        xyxy=track.xyxy,
                    )
                )

        for detection_index in unmatched_detection_indices:
            detection = detections[detection_index]
            tracks.append(
                TrackState(
                    track_id=next_track_id,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    xyxy=detection.xyxy,
                    confidence=detection.confidence,
                    first_seen_frame=frame_index,
                    last_seen_frame=frame_index,
                )
            )
            next_track_id += 1

        active_tracks: list[TrackState] = []
        for track_index, track in enumerate(tracks):
            gap = frame_index - track.last_seen_frame
            if track_index in unmatched_track_indices and track.is_stable and 0 < gap <= TEMPORAL_BRIDGE_FRAMES:
                rendered_detections.append(
                    DetectionBox(
                        class_id=track.class_id,
                        class_name=track.class_name,
                        confidence=track.confidence,
                        xyxy=track.xyxy,
                    )
                )
                bridged_count += 1
            if gap <= TRACK_STALE_FRAMES:
                active_tracks.append(track)
        tracks[:] = active_tracks

        return rendered_detections, next_track_id, bridged_count

    def _match_tracks(
        self,
        tracks: list[TrackState],
        detections: list[DetectionBox],
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        candidate_pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(detections):
                if track.class_id != detection.class_id:
                    continue
                match_iou = self._iou_xyxy(track.xyxy, detection.xyxy)
                if match_iou >= TEMPORAL_MATCH_IOU:
                    candidate_pairs.append((match_iou, track_index, detection_index))

        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_index, detection_index in sorted(candidate_pairs, reverse=True):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            matches.append((track_index, detection_index))

        unmatched_track_indices = set(range(len(tracks))) - used_tracks
        unmatched_detection_indices = set(range(len(detections))) - used_detections
        return matches, unmatched_track_indices, unmatched_detection_indices

    def _draw_detections(self, frame: np.ndarray, detections: list[DetectionBox]) -> np.ndarray:
        canvas = frame.copy()
        for det in detections:
            if len(det.xyxy) != 4:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in det.xyxy]
            color = BOX_COLORS.get(det.class_id, (40, 40, 40))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            text_y = max(y1 - 10, 20)
            cv2.putText(
                canvas,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return canvas

    @staticmethod
    def _iou_xyxy(a: list[float], b: list[float]) -> float:
        if len(a) != 4 or len(b) != 4:
            return 0.0
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union
