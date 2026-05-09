from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.utcnow()


class Base(DeclarativeBase):
    pass


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    weights_path: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    device: Mapped[str] = mapped_column(String(32), default="cpu", nullable=False)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    image_detections: Mapped[list["ImageDetection"]] = relationship(back_populates="model")
    video_tasks: Mapped[list["VideoTask"]] = relationship(back_populates="model")


class AppSetting(Base):
    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    default_model_id: Mapped[int | None] = mapped_column(ForeignKey("model_registry.id"))
    default_conf: Mapped[float] = mapped_column(Float, default=0.15, nullable=False)
    default_iou: Mapped[float] = mapped_column(Float, default=0.45, nullable=False)
    default_imgsz: Mapped[int] = mapped_column(Integer, default=416, nullable=False)
    max_upload_mb: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    default_model: Mapped[ModelRegistry | None] = relationship()


class ImageDetection(Base):
    __tablename__ = "image_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="completed", nullable=False)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("model_registry.id"))
    model_name: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    weights_path: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    conf: Mapped[float] = mapped_column(Float, nullable=False)
    iou: Mapped[float] = mapped_column(Float, nullable=False)
    source_image_path: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    annotated_image_path: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    num_detections: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    model: Mapped[ModelRegistry | None] = relationship(back_populates="image_detections")
    boxes: Mapped[list["ImageDetectionBox"]] = relationship(
        back_populates="detection",
        cascade="all, delete-orphan",
    )


class ImageDetectionBox(Base):
    __tablename__ = "image_detection_boxes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("image_detections.id"), nullable=False)
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(120), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)

    detection: Mapped[ImageDetection] = relationship(back_populates="boxes")


class VideoTask(Base):
    __tablename__ = "video_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_uuid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("model_registry.id"))
    model_name: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    weights_path: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    conf: Mapped[float] = mapped_column(Float, nullable=False)
    iou: Mapped[float] = mapped_column(Float, nullable=False)
    source_video_path: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    output_video_path: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    processed_frames: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_frames: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_detections: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)

    model: Mapped[ModelRegistry | None] = relationship(back_populates="video_tasks")
