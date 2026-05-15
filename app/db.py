from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import RuntimeConfig, build_runtime_config, ensure_runtime_dirs
from app.db_models import AlertRule, AppSetting, Base, ModelRegistry


_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None
DEFAULT_ALERT_RULE_NAME = "\u9ed8\u8ba4\u5438\u70df\u544a\u8b66\u89c4\u5219"
LEGACY_ALERT_RULE_MARKERS = ("\ufffd", "?", "\u012c", "\u6faf")


def runtime_config() -> RuntimeConfig:
    config = build_runtime_config()
    ensure_runtime_dirs(config)
    return config


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        config = runtime_config()
        _ENGINE = create_engine(config.database_url, future=True, pool_pre_ping=True)
    return _ENGINE


def get_session_factory() -> sessionmaker[Session]:
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        _SESSION_FACTORY = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SESSION_FACTORY


def init_database() -> RuntimeConfig:
    config = runtime_config()
    Base.metadata.create_all(get_engine())
    return config


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def settings_row(session: Session) -> AppSetting:
    row = session.get(AppSetting, 1)
    if row is None:
        row = AppSetting(id=1)
        session.add(row)
        session.flush()
    return row


def bootstrap_models(session: Session, candidates: list[dict[str, Any]]) -> list[ModelRegistry]:
    rows: list[ModelRegistry] = []
    for item in candidates:
        row = session.scalar(select(ModelRegistry).where(ModelRegistry.weights_path == item["weights_path"]))
        if row is None:
            row = ModelRegistry(
                name=item["name"],
                weights_path=item["weights_path"],
                device=item.get("device", "cpu"),
                is_available=item.get("is_available", True),
                note=item.get("note", ""),
            )
            session.add(row)
        else:
            row.name = item["name"]
            row.device = item.get("device", row.device)
            row.is_available = item.get("is_available", True)
            row.note = item.get("note", row.note)
        rows.append(row)
    session.flush()
    return rows


def ensure_default_settings(
    session: Session,
    candidates: list[dict[str, Any]],
    *,
    default_conf: float,
    default_iou: float,
    default_imgsz: int,
    max_upload_mb: int,
) -> AppSetting:
    rows = bootstrap_models(session, candidates)
    settings = settings_row(session)
    if rows:
        current_default = session.get(ModelRegistry, settings.default_model_id) if settings.default_model_id else None
        preferred_default = rows[0]
        legacy_default = current_default is not None and (
            "smoker_weights_20260429" in current_default.weights_path
            or "balanced_30" in current_default.weights_path
            or "eca_balanced_304" in current_default.weights_path
        )
        if settings.default_model_id is None or legacy_default:
            settings.default_model_id = preferred_default.id
    # Upgrade untouched legacy rows from the old 0.25 default so class-aware
    # thresholds can expose lower-confidence cigarette candidates.
    if settings.default_conf in (0.15, 0.25):
        settings.default_conf = default_conf
    else:
        settings.default_conf = settings.default_conf or default_conf
    settings.default_iou = settings.default_iou or default_iou
    if settings.default_imgsz in (0, 416):
        settings.default_imgsz = default_imgsz
    else:
        settings.default_imgsz = settings.default_imgsz or default_imgsz
    settings.max_upload_mb = settings.max_upload_mb or max_upload_mb
    session.flush()
    return settings


def bootstrap_alert_rules(session: Session) -> AlertRule:
    existing = session.scalar(select(AlertRule).where(AlertRule.name == DEFAULT_ALERT_RULE_NAME))
    if existing is None:
        for row in session.scalars(select(AlertRule).order_by(AlertRule.id)).all():
            if any(marker in row.name for marker in LEGACY_ALERT_RULE_MARKERS):
                existing = row
                break
    if existing is not None:
        existing.name = DEFAULT_ALERT_RULE_NAME
        existing.enabled = True
        session.flush()
        return existing

    rule = AlertRule(
        name=DEFAULT_ALERT_RULE_NAME,
        enabled=True,
        score_threshold=70.0,
        min_duration_frames=3,
        cooldown_seconds=60,
        monitor_zones=None,
        ignore_zones=None,
        notification_channels=["log", "database"],
    )
    session.add(rule)
    session.flush()
    return rule
