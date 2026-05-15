from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.smoking_event_scorer import SmokingEventScore, calculate_smoking_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate smoking event scorer on images or video.")
    parser.add_argument("--video", help="Path to a video file to test.")
    parser.add_argument("--image", help="Path to an image file to test.")
    parser.add_argument("--weights", default=None, help="Model weights path (uses default if omitted).")
    parser.add_argument("--conf", type=float, default=0.12, help="Confidence threshold.")
    return parser.parse_args()


def test_image(image_path: str, weights: str | None, conf: float) -> None:
    from app.utils.web_inference import DetectionService

    service = DetectionService()
    if weights:
        service.use_runtime_options(weights_path=weights)

    img_bytes = Path(image_path).read_bytes()
    result = service.detect_image_bytes(img_bytes, filename=Path(image_path).name, conf=conf)

    from app.utils.web_inference import DetectionBox
    detections = []
    for box in result.get("detections", []):
        detections.append(DetectionBox(
            class_id=box["class_id"],
            class_name=box["class_name"],
            confidence=box["confidence"],
            xyxy=box["xyxy"],
        ))

    score = calculate_smoking_score(detections, consecutive_hits=1)
    print(f"Image: {image_path}")
    print(f"  Detections: {len(detections)}")
    print(f"  Evidence classes: {score.evidence_classes}")
    print(f"  Base score: {score.base_score}")
    print(f"  Confidence weight: {score.confidence_weight:.3f}")
    print(f"  Spatial bonus: {score.spatial_bonus:.3f}")
    print(f"  Final score: {score.final_score:.1f}")
    print(f"  Classification: {score.classification}")


def test_video(video_path: str, weights: str | None, conf: float) -> None:
    import cv2
    from app.utils.web_inference import DetectionService, DetectionBox

    service = DetectionService()
    if weights:
        service.use_runtime_options(weights_path=weights)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"Video: {video_path} ({total_frames} frames, {fps:.1f} FPS)")
    print("-" * 60)

    frame_idx = 0
    consecutive_hits = 0
    max_score = 0.0
    events_confirmed = 0
    events_suspected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 5 != 0:
            continue

        result = service._predict_frame(frame, conf=conf, iou=0.35)
        detections = service._extract_detections(result)
        detections = service._apply_post_processing_rules(detections)

        if detections:
            consecutive_hits += 1
        else:
            consecutive_hits = 0

        score = calculate_smoking_score(detections, consecutive_hits)

        if score.final_score > max_score:
            max_score = score.final_score

        if score.classification == "confirmed":
            events_confirmed += 1
        elif score.classification == "suspected":
            events_suspected += 1

        if frame_idx % 25 == 0 or score.classification in ("confirmed", "suspected"):
            print(f"  Frame {frame_idx:4d}: score={score.final_score:5.1f} "
                  f"class={score.classification:15s} evidence={score.evidence_classes}")

    cap.release()
    print("-" * 60)
    print(f"Summary:")
    print(f"  Max score: {max_score:.1f}")
    print(f"  Confirmed frames: {events_confirmed}")
    print(f"  Suspected frames: {events_suspected}")
    print(f"  Total sampled frames: {frame_idx // 5}")


def main() -> None:
    args = parse_args()
    if not args.video and not args.image:
        print("ERROR: Provide --video or --image")
        sys.exit(1)

    if args.image:
        test_image(args.image, args.weights, args.conf)
    if args.video:
        test_video(args.video, args.weights, args.conf)


if __name__ == "__main__":
    main()
