from __future__ import annotations

import json
import os
import base64
import binascii
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from app.config import ROOT, build_runtime_config, ensure_runtime_dirs, load_demo_config
from app.db import (
    DEFAULT_ALERT_RULE_NAME,
    bootstrap_alert_rules,
    ensure_default_settings,
    init_database,
    session_scope,
    settings_row,
)
from app.db_models import AppSetting, ImageDetection, ImageDetectionBox, ModelRegistry, VideoTask, AlertEvent, AlertRule, utcnow
from app.utils.web_inference import DEFAULT_INFERENCE_CONF, DetectionService
from app.alert_manager import AlertManager


STATIC_DIR = ROOT / "app" / "ui" / "static"
TEMPLATES_DIR = ROOT / "app" / "ui" / "templates"
RUNTIME = build_runtime_config()
ensure_runtime_dirs(RUNTIME)
ALERT_THUMBNAIL_DIR = RUNTIME.results_root / "alerts"
ALERT_THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Smoker Behavior Detection Web Admin",
    description="FastAPI management console for the smoker behavior detection graduation project.",
    version="2.0.0",
    docs_url="/docs" if os.getenv("SMOKER_ENABLE_API_DOCS") == "1" else None,
    redoc_url="/redoc" if os.getenv("SMOKER_ENABLE_API_DOCS") == "1" else None,
    openapi_url="/openapi.json" if os.getenv("SMOKER_ENABLE_API_DOCS") == "1" else None,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/artifacts", StaticFiles(directory=RUNTIME.results_root), name="artifacts")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
service = DetectionService()
APP_STATE: dict[str, Any] = {"db_ready": False, "db_error": "", "database_url": RUNTIME.database_url}
MIN_IOU = 0.05
MAX_IOU = 0.95
MAX_CONF = 0.99


def _basic_auth_is_valid(header_value: str | None) -> bool:
    expected_password = os.getenv("SMOKER_ADMIN_PASSWORD")
    if not expected_password:
        return True
    if not header_value or not header_value.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header_value.removeprefix("Basic "), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    username, sep, password = decoded.partition(":")
    if not sep:
        return False
    expected_user = os.getenv("SMOKER_ADMIN_USER", "admin")
    return secrets.compare_digest(username, expected_user) and secrets.compare_digest(password, expected_password)


@app.middleware("http")
async def require_optional_basic_auth(request: Request, call_next):
    if _basic_auth_is_valid(request.headers.get("authorization")):
        return await call_next(request)
    return JSONResponse(
        {"detail": "Authentication required."},
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Smoker Demo"'},
    )


class ModelSwitchRequest(BaseModel):
    model_id: int = Field(..., ge=1)


class SettingsUpdateRequest(BaseModel):
    default_model_id: int | None = Field(default=None, ge=1)
    default_conf: float | None = Field(default=None, ge=DEFAULT_INFERENCE_CONF, le=0.99)
    default_iou: float | None = Field(default=None, ge=0.05, le=0.95)
    default_imgsz: int | None = Field(default=None, ge=320, le=960)
    max_upload_mb: int | None = Field(default=None, ge=1, le=512)


def load_cigarette_analysis_report() -> dict[str, Any]:
    candidates = [
        ROOT / "runs" / "reports" / "cigarette_analysis_smoke.json",
        ROOT / "runs" / "reports" / "cigarette_analysis.json",
    ]
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["report_path"] = str(path)
            return payload
    return {}


def load_experiment_suite_report() -> dict[str, Any]:
    path = ROOT / "runs" / "reports" / "cigarette_experiment_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["report_path"] = str(path)
    return payload


def artifact_url(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    try:
        relative = path.resolve().relative_to(RUNTIME.results_root.resolve())
    except ValueError:
        return ""
    return f"/artifacts/{relative.as_posix()}"


def safe_unlink_runtime_file(path: Path | str | None, *, allowed_roots: list[Path]) -> bool:
    if not path:
        return False
    target = Path(path)
    try:
        resolved_target = target.resolve()
    except OSError:
        return False
    allowed = False
    for root in allowed_roots:
        try:
            resolved_target.relative_to(root.resolve())
            allowed = True
            break
        except ValueError:
            continue
    if not allowed or not resolved_target.is_file():
        return False
    resolved_target.unlink(missing_ok=True)
    return True


def resolve_detection_thresholds(
    *,
    conf: float | None,
    iou: float | None,
    settings: AppSetting,
) -> tuple[float, float]:
    conf_value = settings.default_conf if conf is None else conf
    iou_value = settings.default_iou if iou is None else iou

    if not DEFAULT_INFERENCE_CONF <= conf_value <= MAX_CONF:
        raise HTTPException(
            status_code=400,
            detail=f"conf must be between {DEFAULT_INFERENCE_CONF:.2f} and {MAX_CONF:.2f}.",
        )
    if not MIN_IOU <= iou_value <= MAX_IOU:
        raise HTTPException(
            status_code=400,
            detail=f"iou must be between {MIN_IOU:.2f} and {MAX_IOU:.2f}.",
        )
    return float(conf_value), float(iou_value)


def create_request_detection_service(
    *,
    weights_path: str | Path | None,
    imgsz: int,
    max_upload_mb: int,
) -> DetectionService:
    return DetectionService(
        weights_path=weights_path,
        imgsz=imgsz,
        max_upload_mb=max_upload_mb,
    )


async def read_upload_bytes_limited(
    file: UploadFile,
    *,
    max_upload_mb: int,
    label: str,
    chunk_size: int = 1024 * 1024,
) -> bytes:
    limit = max_upload_mb * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded {label} exceeds {max_upload_mb} MB limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _safe_file_stem(value: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return stem.strip("._") or uuid4().hex


def _expanded_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    padding_ratio: float = 0.2,
) -> tuple[int, int, int, int] | None:
    if len(bbox) < 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    if x2 <= x1 or y2 <= y1:
        return None

    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def create_alert_thumbnail_from_video(
    *,
    source_video_path: str | Path,
    event: dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> Path | None:
    """Save a cropped alert thumbnail from the event frame; fall back to the full frame."""
    source_path = Path(source_video_path)
    if not source_path.exists():
        return None

    start_frame = int(event.get("start_frame") or event.get("frame") or 1)
    capture = cv2.VideoCapture(str(source_path))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(start_frame - 1, 0))
        ok, frame = capture.read()
        if not ok or frame is None:
            return None

        height, width = frame.shape[:2]
        crop_box = _expanded_bbox(list(event.get("bbox") or []), width=width, height=height)
        thumbnail = frame
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            thumbnail = frame[y1:y2, x1:x2]

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{_safe_file_stem(prefix)}_f{max(start_frame, 1)}.jpg"
        if not cv2.imwrite(str(output_path), thumbnail):
            return None
        return output_path
    finally:
        capture.release()


def require_database() -> None:
    if not APP_STATE["db_ready"]:
        raise HTTPException(status_code=503, detail=f"Database is not ready: {APP_STATE['db_error']}")


def serialize_model(row: ModelRegistry, settings: AppSetting | None = None) -> dict[str, Any]:
    is_default = bool(settings and settings.default_model_id == row.id)
    return {
        "id": row.id,
        "name": row.name,
        "weights_path": row.weights_path,
        "device": row.device,
        "is_available": row.is_available,
        "is_default": is_default,
        "note": row.note,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def serialize_settings(row: AppSetting) -> dict[str, Any]:
    return {
        "default_model_id": row.default_model_id,
        "default_conf": row.default_conf,
        "default_iou": row.default_iou,
        "default_imgsz": row.default_imgsz,
        "max_upload_mb": row.max_upload_mb,
    }


def serialize_detection(row: ImageDetection, *, with_boxes: bool = False) -> dict[str, Any]:
    payload = {
        "id": row.id,
        "source_name": row.source_name,
        "status": row.status,
        "model_id": row.model_id,
        "model_name": row.model_name,
        "weights_path": row.weights_path,
        "conf": row.conf,
        "iou": row.iou,
        "num_detections": row.num_detections,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "source_image_url": artifact_url(row.source_image_path),
        "annotated_image_url": artifact_url(row.annotated_image_path),
    }
    if with_boxes:
        payload["boxes"] = [
            {
                "id": box.id,
                "class_id": box.class_id,
                "class_name": box.class_name,
                "confidence": box.confidence,
                "xyxy": [box.x1, box.y1, box.x2, box.y2],
            }
            for box in row.boxes
        ]
    return payload


def serialize_video_task(row: VideoTask) -> dict[str, Any]:
    return {
        "id": row.id,
        "task_uuid": row.task_uuid,
        "source_name": row.source_name,
        "status": row.status,
        "model_id": row.model_id,
        "model_name": row.model_name,
        "weights_path": row.weights_path,
        "conf": row.conf,
        "iou": row.iou,
        "progress": row.progress,
        "processed_frames": row.processed_frames,
        "total_frames": row.total_frames,
        "num_detections": row.num_detections,
        "error_message": row.error_message,
        "summary": row.summary_json,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "output_video_url": artifact_url(row.output_video_path),
    }


def selected_model(
    session,
    *,
    model_id: int | None = None,
    settings: AppSetting | None = None,
) -> ModelRegistry | None:
    settings = settings or settings_row(session)
    chosen_id = model_id or settings.default_model_id
    if chosen_id is None:
        return None
    return session.get(ModelRegistry, chosen_id)


def apply_runtime_model(session) -> dict[str, Any]:
    settings = settings_row(session)
    model = selected_model(session, settings=settings)
    if model is not None:
        service.use_runtime_options(
            weights_path=model.weights_path,
            imgsz=settings.default_imgsz,
            max_upload_mb=settings.max_upload_mb,
        )
    else:
        service.use_runtime_options(imgsz=settings.default_imgsz, max_upload_mb=settings.max_upload_mb)
    info = service.model_info()
    info["default_model_id"] = settings.default_model_id
    info["database_backend"] = "postgresql" if "postgresql" in RUNTIME.database_url else "sqlite"
    return info


def queue_video_processing(task_id: int) -> None:
    with session_scope() as session:
        task = session.get(VideoTask, task_id)
        if task is None:
            return
        task.status = "running"
        task.started_at = utcnow()

    def update_progress(processed: int, total: int, detections: int) -> None:
        with session_scope() as progress_session:
            row = progress_session.get(VideoTask, task_id)
            if row is None:
                return
            row.processed_frames = processed
            row.total_frames = total
            row.num_detections = detections
            row.progress = round(processed / max(total, 1), 4)

    try:
        with session_scope() as session:
            settings = settings_row(session)
            task = session.get(VideoTask, task_id)
            if task is None:
                return
            task_service = create_request_detection_service(
                weights_path=task.weights_path,
                imgsz=settings.default_imgsz,
                max_upload_mb=settings.max_upload_mb,
            )
        summary = task_service.process_video_file(
            source_path=task.source_video_path,
            output_path=task.output_video_path,
            conf=task.conf,
            iou=task.iou,
            progress_callback=update_progress,
        )
        with session_scope() as session:
            row = session.get(VideoTask, task_id)
            if row is None:
                return
            row.status = "completed"
            row.progress = 1.0
            row.processed_frames = summary["processed_frames"]
            row.total_frames = summary["total_frames"]
            row.num_detections = summary["num_detections"]
            row.summary_json = summary
            row.finished_at = utcnow()

            smoking_events = summary.get("smoking_events", [])
            if smoking_events:
                from app.db import get_session_factory
                alert_mgr = AlertManager(get_session_factory())
                rule = session.query(AlertRule).filter(AlertRule.enabled == True).first()  # noqa: E712
                if rule:
                    fps = float(summary.get("fps") or 25.0)
                    for evt in smoking_events:
                        start_frame = int(evt.get("start_frame", evt.get("frame", 0)))
                        end_frame = int(evt.get("end_frame", evt.get("frame", start_frame)))
                        duration_frames = int(evt.get("duration_frames", max(end_frame - start_frame + 1, 1)))
                        duration_seconds = float(evt.get("duration_seconds", duration_frames / max(fps, 1.0)))
                        bbox_values = list(evt.get("bbox") or [0, 0, 0, 0])
                        if len(bbox_values) < 4:
                            bbox_values = [0, 0, 0, 0]
                        bbox = tuple(float(value) for value in bbox_values[:4])
                        if alert_mgr.should_trigger_alert(
                            evt["score"],
                            bbox,
                            end_frame,
                            fps,
                            rule,
                            duration_frames=duration_frames,
                            session=session,
                        ):
                            thumbnail_path = create_alert_thumbnail_from_video(
                                source_video_path=task.source_video_path,
                                event={**evt, "start_frame": start_frame, "bbox": list(bbox)},
                                output_dir=ALERT_THUMBNAIL_DIR,
                                prefix=f"task_{task_id}_alert_{start_frame}_{end_frame}",
                            )
                            alert_mgr.create_alert_event(
                                session,
                                video_task_id=task_id,
                                score=evt["score"],
                                severity=evt["classification"],
                                start_frame=start_frame,
                                end_frame=end_frame,
                                duration_seconds=duration_seconds,
                                bbox=bbox,
                                thumbnail_path=str(thumbnail_path) if thumbnail_path else "",
                            )
    except Exception as exc:  # noqa: BLE001
        with session_scope() as session:
            row = session.get(VideoTask, task_id)
            if row is None:
                return
            row.status = "failed"
            row.error_message = str(exc)
            row.finished_at = utcnow()


@app.on_event("startup")
def startup_event() -> None:
    try:
        init_database()
        with session_scope() as session:
            settings = ensure_default_settings(
                session,
                service.available_weight_candidates(),
                default_conf=DEFAULT_INFERENCE_CONF,
                default_iou=0.45,
                default_imgsz=service.imgsz,
                max_upload_mb=10,
            )
            apply_runtime_model(session)
            bootstrap_alert_rules(session)
            APP_STATE["default_model_id"] = settings.default_model_id
        APP_STATE["db_ready"] = True
        APP_STATE["db_error"] = ""
    except Exception as exc:  # noqa: BLE001
        APP_STATE["db_ready"] = False
        APP_STATE["db_error"] = str(exc)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "page_title": "Smoker Behavior Detection Admin",
            "demo_config": load_demo_config(),
        },
    )


@app.get("/alerts", response_class=HTMLResponse)
def alerts_page(request: Request) -> Any:
    return templates.TemplateResponse(request, "alerts.html", {})


@app.get("/reports/video/{task_id}", response_class=HTMLResponse)
def video_report(request: Request, task_id: int) -> Any:
    require_database()
    with session_scope() as session:
        row = session.get(VideoTask, task_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Video task not found.")
        task = serialize_video_task(row)

    summary = task.get("summary") or {}
    return templates.TemplateResponse(
        request,
        "video_report.html",
        {
            "page_title": f"Video Report #{task_id}",
            "task": task,
            "summary": summary,
            "summary_pretty": json.dumps(summary, ensure_ascii=False, indent=2),
            "json_report_url": f"/api/tasks/video/{task_id}",
        },
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    snapshot = service.health_snapshot()
    snapshot.update(
        {
            "status": "ok" if APP_STATE["db_ready"] and snapshot["weights_exists"] else "degraded",
            "database_ready": APP_STATE["db_ready"],
            "database_error": APP_STATE["db_error"],
            "database_backend": "postgresql" if "postgresql" in RUNTIME.database_url else "sqlite",
        }
    )
    return snapshot


@app.get("/api/dashboard")
def dashboard() -> JSONResponse:
    require_database()
    payload = load_demo_config()
    experiment_suite = load_experiment_suite_report()
    with session_scope() as session:
        settings = settings_row(session)
        model_info = apply_runtime_model(session)
        models = session.scalars(select(ModelRegistry).order_by(ModelRegistry.id)).all()
        recent_records = session.scalars(
            select(ImageDetection).order_by(ImageDetection.created_at.desc()).limit(5)
        ).all()
        recent_tasks = session.scalars(select(VideoTask).order_by(VideoTask.created_at.desc()).limit(5)).all()
        total_records = session.scalar(select(func.count(ImageDetection.id))) or 0
        total_tasks = session.scalar(select(func.count(VideoTask.id))) or 0
        completed_tasks = (
            session.scalar(select(func.count(VideoTask.id)).where(VideoTask.status == "completed")) or 0
        )

    payload["model"] = model_info
    payload["settings"] = serialize_settings(settings)
    payload["storage"] = {
        "database_backend": "postgresql" if "postgresql" in RUNTIME.database_url else "sqlite",
        "database_ready": APP_STATE["db_ready"],
        "output_root": str(RUNTIME.output_root),
    }
    payload["models"] = [serialize_model(model, settings) for model in models]
    payload["recent_records"] = [serialize_detection(record) for record in recent_records]
    payload["recent_tasks"] = [serialize_video_task(task) for task in recent_tasks]
    payload["cigarette_analysis"] = load_cigarette_analysis_report()
    if experiment_suite:
        payload["experiments"] = experiment_suite.get("dashboard_experiments", payload.get("experiments", {}))
        payload["experiment_suite"] = experiment_suite
    payload["stats"] = {
        "total_records": total_records,
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "available_models": len(models),
    }
    return JSONResponse(payload)


@app.get("/api/model")
def model_info() -> dict[str, Any]:
    require_database()
    with session_scope() as session:
        return apply_runtime_model(session)


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    require_database()
    with session_scope() as session:
        settings = settings_row(session)
        rows = session.scalars(select(ModelRegistry).order_by(ModelRegistry.id)).all()
        return {
            "items": [serialize_model(row, settings) for row in rows],
            "default_model_id": settings.default_model_id,
        }


@app.post("/api/models/default")
def set_default_model(payload: ModelSwitchRequest) -> dict[str, Any]:
    require_database()
    with session_scope() as session:
        row = session.get(ModelRegistry, payload.model_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Model not found.")
        settings = settings_row(session)
        settings.default_model_id = row.id
        info = apply_runtime_model(session)
        return {"message": "Default model updated.", "model": serialize_model(row, settings), "runtime": info}


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    require_database()
    with session_scope() as session:
        settings = settings_row(session)
        return serialize_settings(settings)


@app.put("/api/settings")
def update_settings(payload: SettingsUpdateRequest) -> dict[str, Any]:
    require_database()
    with session_scope() as session:
        settings = settings_row(session)
        if payload.default_model_id is not None:
            row = session.get(ModelRegistry, payload.default_model_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Model not found.")
            settings.default_model_id = payload.default_model_id
        if payload.default_conf is not None:
            settings.default_conf = payload.default_conf
        if payload.default_iou is not None:
            settings.default_iou = payload.default_iou
        if payload.default_imgsz is not None:
            settings.default_imgsz = payload.default_imgsz
        if payload.max_upload_mb is not None:
            settings.max_upload_mb = payload.max_upload_mb
        runtime = apply_runtime_model(session)
        return {"message": "Settings updated.", "settings": serialize_settings(settings), "runtime": runtime}


@app.get("/api/records")
def list_records(
    limit: int = 20,
    class_name: str | None = None,
    status: str | None = None,
    model_id: int | None = None,
) -> dict[str, Any]:
    require_database()
    with session_scope() as session:
        stmt = select(ImageDetection).options(joinedload(ImageDetection.boxes)).order_by(ImageDetection.created_at.desc())
        if class_name:
            stmt = stmt.join(ImageDetectionBox).where(ImageDetectionBox.class_name == class_name)
        if status:
            stmt = stmt.where(ImageDetection.status == status)
        if model_id:
            stmt = stmt.where(ImageDetection.model_id == model_id)
        rows = session.scalars(stmt.limit(max(1, min(limit, 100)))).unique().all()
        return {"items": [serialize_detection(row) for row in rows]}


@app.get("/api/records/{record_id}")
def record_detail(record_id: int) -> dict[str, Any]:
    require_database()
    with session_scope() as session:
        row = session.scalar(
            select(ImageDetection)
            .options(joinedload(ImageDetection.boxes))
            .where(ImageDetection.id == record_id)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Record not found.")
        return serialize_detection(row, with_boxes=True)


@app.delete("/api/records/{record_id}")
def delete_record(record_id: int) -> dict[str, Any]:
    require_database()
    with session_scope() as session:
        row = session.scalar(
            select(ImageDetection)
            .options(joinedload(ImageDetection.boxes))
            .where(ImageDetection.id == record_id)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Record not found.")
        source_path = Path(row.source_image_path) if row.source_image_path else None
        annotated_path = Path(row.annotated_image_path) if row.annotated_image_path else None
        session.delete(row)
    safe_unlink_runtime_file(source_path, allowed_roots=[RUNTIME.image_upload_dir])
    safe_unlink_runtime_file(annotated_path, allowed_roots=[RUNTIME.image_result_dir])
    return {"message": "Record deleted."}


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float | None = Form(default=None),
    iou: float | None = Form(default=None),
    model_id: int | None = Form(default=None),
) -> JSONResponse:
    require_database()
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        with session_scope() as session:
            settings = settings_row(session)
            model = selected_model(session, model_id=model_id, settings=settings)
            conf_value, iou_value = resolve_detection_thresholds(conf=conf, iou=iou, settings=settings)
            imgsz_value = settings.default_imgsz
            max_upload_mb = settings.max_upload_mb
            image_bytes = await read_upload_bytes_limited(file, max_upload_mb=max_upload_mb, label="image")

            request_service = create_request_detection_service(
                weights_path=model.weights_path if model else None,
                imgsz=imgsz_value,
                max_upload_mb=max_upload_mb,
            )
            payload = request_service.detect_image_bytes(
                image_bytes,
                filename=file.filename,
                conf=conf_value,
                iou=iou_value,
            )

            run_id = uuid4().hex
            suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
            source_path = RUNTIME.image_upload_dir / f"{run_id}{suffix}"
            annotated_path = RUNTIME.image_result_dir / f"{run_id}.jpg"
            write_bytes(source_path, image_bytes)
            annotated_bytes = payload.pop("_annotated_image_bytes")
            write_bytes(annotated_path, annotated_bytes)

            record = ImageDetection(
                source_name=file.filename or source_path.name,
                status="completed",
                model_id=model.id if model else None,
                model_name=model.name if model else "",
                weights_path=model.weights_path if model else str(request_service.weights_path),
                conf=conf_value,
                iou=iou_value,
                source_image_path=str(source_path),
                annotated_image_path=str(annotated_path),
                num_detections=payload["num_detections"],
            )
            session.add(record)
            session.flush()

            for item in payload["detections"]:
                coords = item.get("xyxy") or [0, 0, 0, 0]
                session.add(
                    ImageDetectionBox(
                        detection_id=record.id,
                        class_id=item["class_id"],
                        class_name=item["class_name"],
                        confidence=item["confidence"],
                        x1=coords[0],
                        y1=coords[1],
                        x2=coords[2],
                        y2=coords[3],
                    )
                )

            payload["record_id"] = record.id
            payload["annotated_image_url"] = artifact_url(str(annotated_path))
            payload["source_image_url"] = artifact_url(str(source_path))
            payload["model_meta"] = serialize_model(model, settings) if model else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(payload)


@app.get("/api/tasks/video")
def list_video_tasks(limit: int = 20, status: str | None = None) -> dict[str, Any]:
    require_database()
    with session_scope() as session:
        stmt = select(VideoTask).order_by(VideoTask.created_at.desc())
        if status:
            stmt = stmt.where(VideoTask.status == status)
        rows = session.scalars(stmt.limit(max(1, min(limit, 100)))).all()
        return {"items": [serialize_video_task(row) for row in rows]}


@app.get("/api/tasks/video/{task_id}")
def video_task_detail(task_id: int) -> dict[str, Any]:
    require_database()
    with session_scope() as session:
        row = session.get(VideoTask, task_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Video task not found.")
        return serialize_video_task(row)


@app.post("/api/tasks/video")
async def create_video_task(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf: float | None = Form(default=None),
    iou: float | None = Form(default=None),
    model_id: int | None = Form(default=None),
) -> JSONResponse:
    require_database()

    try:
        with session_scope() as session:
            settings = settings_row(session)
            model = selected_model(session, model_id=model_id, settings=settings)
            conf_value, iou_value = resolve_detection_thresholds(conf=conf, iou=iou, settings=settings)
            video_bytes = await read_upload_bytes_limited(file, max_upload_mb=settings.max_upload_mb, label="video")
            request_service = create_request_detection_service(
                weights_path=model.weights_path if model else None,
                imgsz=settings.default_imgsz,
                max_upload_mb=settings.max_upload_mb,
            )
            request_service.validate_upload(
                file.filename,
                video_bytes,
                allowed_suffixes={".mp4", ".avi", ".mov", ".mkv", ".webm"},
                label="video",
            )

            run_id = uuid4().hex
            suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
            source_path = RUNTIME.video_upload_dir / f"{run_id}{suffix}"
            output_path = RUNTIME.video_result_dir / f"{run_id}.mp4"
            write_bytes(source_path, video_bytes)

            task = VideoTask(
                task_uuid=run_id,
                source_name=file.filename or source_path.name,
                status="queued",
                model_id=model.id if model else None,
                model_name=model.name if model else "",
                weights_path=model.weights_path if model else str(request_service.weights_path),
                conf=conf_value,
                iou=iou_value,
                source_video_path=str(source_path),
                output_video_path=str(output_path),
                summary_json={"message": "Task accepted."},
            )
            session.add(task)
            session.flush()
            task_payload = serialize_video_task(task)
            background_tasks.add_task(queue_video_processing, task.id)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({"message": "Video task queued.", "task": task_payload}, status_code=202)


# --- Alert API Endpoints ---


@app.get("/api/alerts/rules")
def get_alert_rules() -> JSONResponse:
    with session_scope() as session:
        rules = session.query(AlertRule).all()
        return JSONResponse([
            {
                "id": r.id,
                "name": r.name,
                "enabled": r.enabled,
                "score_threshold": r.score_threshold,
                "min_duration_frames": r.min_duration_frames,
                "cooldown_seconds": r.cooldown_seconds,
                "monitor_zones": r.monitor_zones,
                "ignore_zones": r.ignore_zones,
                "notification_channels": r.notification_channels,
            }
            for r in rules
        ])


class AlertRuleCreate(BaseModel):
    name: str
    enabled: bool = True
    score_threshold: float = Field(default=70.0, ge=0, le=100)
    min_duration_frames: int = Field(default=3, ge=1)
    cooldown_seconds: int = Field(default=60, ge=0)
    monitor_zones: list | None = None
    ignore_zones: list | None = None
    notification_channels: list[str] = Field(default_factory=lambda: ["log", "database"])


@app.post("/api/alerts/rules", status_code=201)
def create_alert_rule(body: AlertRuleCreate) -> JSONResponse:
    with session_scope() as session:
        rule = AlertRule(
            name=body.name,
            enabled=body.enabled,
            score_threshold=body.score_threshold,
            min_duration_frames=body.min_duration_frames,
            cooldown_seconds=body.cooldown_seconds,
            monitor_zones=body.monitor_zones,
            ignore_zones=body.ignore_zones,
            notification_channels=body.notification_channels,
        )
        session.add(rule)
        session.flush()
        return JSONResponse({"id": rule.id, "name": rule.name}, status_code=201)


@app.put("/api/alerts/rules/{rule_id}")
def update_alert_rule(rule_id: int, body: AlertRuleCreate) -> JSONResponse:
    with session_scope() as session:
        rule = session.get(AlertRule, rule_id)
        if rule is None:
            raise HTTPException(status_code=404, detail="Rule not found")
        if rule.name == DEFAULT_ALERT_RULE_NAME and not body.enabled:
            raise HTTPException(status_code=400, detail="Default alert rule cannot be disabled.")
        rule.name = body.name
        rule.enabled = body.enabled
        rule.score_threshold = body.score_threshold
        rule.min_duration_frames = body.min_duration_frames
        rule.cooldown_seconds = body.cooldown_seconds
        rule.monitor_zones = body.monitor_zones
        rule.ignore_zones = body.ignore_zones
        rule.notification_channels = body.notification_channels
        return JSONResponse({"id": rule.id, "name": rule.name})


@app.delete("/api/alerts/rules/{rule_id}", status_code=204)
def delete_alert_rule(rule_id: int) -> None:
    with session_scope() as session:
        rule = session.get(AlertRule, rule_id)
        if rule is None:
            raise HTTPException(status_code=404, detail="Rule not found")
        if rule.name == DEFAULT_ALERT_RULE_NAME:
            raise HTTPException(status_code=400, detail="Default alert rule cannot be deleted.")
        session.delete(rule)


@app.get("/api/alerts/events")
def get_alert_events(status: str | None = None, limit: int = 50, offset: int = 0) -> JSONResponse:
    with session_scope() as session:
        query = session.query(AlertEvent).order_by(AlertEvent.created_at.desc())
        if status:
            query = query.filter(AlertEvent.status == status)
        total = query.count()
        events = query.offset(offset).limit(limit).all()
        return JSONResponse({
            "events": [
                {
                    "id": e.id,
                    "video_task_id": e.video_task_id,
                    "alert_type": e.alert_type,
                    "severity": e.severity,
                    "score": e.score,
                    "start_frame": e.start_frame,
                    "end_frame": e.end_frame,
                    "duration_seconds": e.duration_seconds,
                    "bbox": [e.bbox_x1, e.bbox_y1, e.bbox_x2, e.bbox_y2],
                    "thumbnail_url": artifact_url(e.thumbnail_path),
                    "status": e.status,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in events
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        })


@app.put("/api/alerts/events/{event_id}/confirm")
def confirm_alert_event(event_id: int) -> JSONResponse:
    with session_scope() as session:
        event = session.get(AlertEvent, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Alert event not found")
        event.status = "confirmed"
        event.confirmed_at = utcnow()
        return JSONResponse({"id": event.id, "status": event.status})


@app.put("/api/alerts/events/{event_id}/dismiss")
def dismiss_alert_event(event_id: int) -> JSONResponse:
    with session_scope() as session:
        event = session.get(AlertEvent, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Alert event not found")
        event.status = "dismissed"
        return JSONResponse({"id": event.id, "status": event.status})


@app.get("/api/alerts/stats")
def get_alert_stats() -> JSONResponse:
    with session_scope() as session:
        total = session.query(func.count(AlertEvent.id)).scalar() or 0
        confirmed = session.query(func.count(AlertEvent.id)).filter(AlertEvent.status == "confirmed").scalar() or 0
        dismissed = session.query(func.count(AlertEvent.id)).filter(AlertEvent.status == "dismissed").scalar() or 0
        pending = session.query(func.count(AlertEvent.id)).filter(AlertEvent.status == "pending").scalar() or 0
        fpr = dismissed / total if total > 0 else 0.0
        return JSONResponse({
            "total_events": total,
            "confirmed": confirmed,
            "dismissed": dismissed,
            "pending": pending,
            "false_positive_rate": round(fpr, 4),
        })
