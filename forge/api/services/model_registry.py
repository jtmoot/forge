"""Model registry service — versioned models with stage lifecycle management.

Manages registered models, versioning from experiment runs, and stage
transitions (development → staging → production → archived) with
validation rules and audit logging.
"""

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from forge.api.models.database import (
    ModelStageHistory,
    ModelVersion,
    RegisteredModel,
    Run,
)
from forge.api.services.metrics import MODEL_REGISTRY_VERSIONS

logger = logging.getLogger(__name__)


def _refresh_stage_gauges(db: Session) -> None:
    """Update Prometheus gauge with current version counts per stage."""
    stage_counts = (
        db.query(ModelVersion.stage, func.count(ModelVersion.id))
        .group_by(ModelVersion.stage)
        .all()
    )
    # Reset all known stages to 0, then set actual counts
    for stage in ("development", "staging", "production", "archived"):
        MODEL_REGISTRY_VERSIONS.labels(stage=stage).set(0)
    for stage, count in stage_counts:
        MODEL_REGISTRY_VERSIONS.labels(stage=stage).set(count)


# Stage transition rules
VALID_TRANSITIONS = {
    "development": {"staging", "archived"},
    "staging": {"production", "archived"},
    "production": {"archived"},
    "archived": set(),  # terminal state
}

# Promotion thresholds for staging → production
PRODUCTION_THRESHOLDS = {
    "accuracy": 0.50,
    "inference_latency_ms": 500.0,
}


# ---------------------------------------------------------------------------
# Register model
# ---------------------------------------------------------------------------


def register_model(
    db: Session,
    name: str,
    description: str | None = None,
) -> RegisteredModel:
    """Create a new registered model entry.

    Raises ValueError if a model with that name already exists.
    """
    existing = db.query(RegisteredModel).filter(RegisteredModel.name == name).first()
    if existing:
        raise ValueError(f"Model '{name}' already registered (id={existing.id})")

    model = RegisteredModel(name=name, description=description)
    db.add(model)
    db.commit()
    db.refresh(model)
    logger.info("Registered model '%s' (id=%s)", name, model.id)
    return model


# ---------------------------------------------------------------------------
# Register version
# ---------------------------------------------------------------------------


def register_version(
    db: Session,
    model_name: str,
    run_id: UUID,
    tags: dict | None = None,
) -> ModelVersion:
    """Create a new model version from a completed experiment run.

    Copies metrics from the run into a frozen metrics_snapshot.
    Defaults to 'development' stage.
    """
    model = db.query(RegisteredModel).filter(RegisteredModel.name == model_name).first()
    if model is None:
        raise ValueError(f"Model '{model_name}' not found. Register it first.")

    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise ValueError(f"Run {run_id} not found")
    if run.status != "completed":
        raise ValueError(f"Run {run_id} is not completed (status={run.status})")

    # Auto-increment version
    max_version = (
        db.query(func.max(ModelVersion.version))
        .filter(ModelVersion.model_id == model.id)
        .scalar()
    )
    next_version = (max_version or 0) + 1

    # Freeze metrics
    metrics_snapshot = {
        "accuracy": run.accuracy,
        "precision": run.precision_score,
        "recall": run.recall,
        "f1": run.f1,
        "inference_latency_ms": run.inference_latency_ms,
        "inference_latency_p95_ms": run.inference_latency_p95_ms,
        "peak_memory_mb": run.peak_memory_mb,
        "model_size_mb": run.model_size_mb,
        "throughput_samples_per_sec": run.throughput_samples_per_sec,
        "efficiency_score": run.efficiency_score,
        "training_time_seconds": run.training_time_seconds,
    }

    mv = ModelVersion(
        model_id=model.id,
        version=next_version,
        run_id=run_id,
        stage="development",
        stage_changed_at=datetime.now(UTC),
        stage_changed_by="api",
        s3_artifact_path=run.s3_artifact_path,
        model_size_mb=run.model_size_mb,
        metrics_snapshot=metrics_snapshot,
        tags=tags,
    )
    db.add(mv)

    # Update model timestamp
    model.updated_at = datetime.now(UTC)

    db.commit()
    db.refresh(mv)
    _refresh_stage_gauges(db)
    logger.info("Registered version %d for model '%s' from run %s", next_version, model_name, run_id)
    return mv


# ---------------------------------------------------------------------------
# Stage transitions
# ---------------------------------------------------------------------------


def transition_stage(
    db: Session,
    model_version_id: UUID,
    new_stage: str,
    reason: str | None = None,
) -> ModelVersion:
    """Transition a model version to a new stage with validation.

    Rules:
    - development → staging (always allowed)
    - staging → production (only if metrics meet thresholds)
    - production → archived (always allowed)
    - any → archived (always allowed)
    - No skipping stages (dev → production not allowed)
    - Promoting to production auto-archives the current production version
    """
    mv = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
    if mv is None:
        raise ValueError(f"ModelVersion {model_version_id} not found")

    current_stage = mv.stage
    if new_stage not in VALID_TRANSITIONS.get(current_stage, set()):
        raise ValueError(
            f"Invalid transition: {current_stage} → {new_stage}. "
            f"Allowed: {VALID_TRANSITIONS.get(current_stage, set()) or 'none (terminal state)'}"
        )

    # Validate production promotion thresholds
    if new_stage == "production":
        metrics = mv.metrics_snapshot or {}
        violations = []
        accuracy = metrics.get("accuracy")
        if accuracy is not None and accuracy < PRODUCTION_THRESHOLDS["accuracy"]:
            violations.append(
                f"Accuracy {accuracy:.4f} below threshold {PRODUCTION_THRESHOLDS['accuracy']}"
            )
        latency = metrics.get("inference_latency_ms")
        if latency is not None and latency > PRODUCTION_THRESHOLDS["inference_latency_ms"]:
            violations.append(
                f"Latency {latency:.2f}ms exceeds threshold {PRODUCTION_THRESHOLDS['inference_latency_ms']}ms"
            )
        if violations:
            raise ValueError(
                f"Cannot promote to production: {'; '.join(violations)}"
            )

        # Auto-archive current production version
        current_prod = (
            db.query(ModelVersion)
            .filter(
                ModelVersion.model_id == mv.model_id,
                ModelVersion.stage == "production",
                ModelVersion.id != mv.id,
            )
            .first()
        )
        if current_prod:
            _record_transition(db, current_prod, "production", "archived",
                               f"Auto-archived: replaced by version {mv.version}")
            current_prod.stage = "archived"
            current_prod.stage_changed_at = datetime.now(UTC)
            current_prod.stage_changed_by = "api"

    # Record transition
    _record_transition(db, mv, current_stage, new_stage, reason)

    mv.stage = new_stage
    mv.stage_changed_at = datetime.now(UTC)
    mv.stage_changed_by = "api"

    # Update parent model timestamp
    model = db.query(RegisteredModel).filter(RegisteredModel.id == mv.model_id).first()
    if model:
        model.updated_at = datetime.now(UTC)

    db.commit()
    db.refresh(mv)
    _refresh_stage_gauges(db)
    logger.info("Transitioned version %s: %s → %s", model_version_id, current_stage, new_stage)
    return mv


def _record_transition(
    db: Session,
    mv: ModelVersion,
    from_stage: str,
    to_stage: str,
    reason: str | None,
) -> None:
    """Insert a stage history record."""
    history = ModelStageHistory(
        model_version_id=mv.id,
        from_stage=from_stage,
        to_stage=to_stage,
        reason=reason,
    )
    db.add(history)


# ---------------------------------------------------------------------------
# Compare versions
# ---------------------------------------------------------------------------


def compare_versions(
    db: Session,
    version_id_a: UUID,
    version_id_b: UUID,
) -> dict:
    """Side-by-side comparison of two model versions with deltas."""
    mv_a = db.query(ModelVersion).filter(ModelVersion.id == version_id_a).first()
    mv_b = db.query(ModelVersion).filter(ModelVersion.id == version_id_b).first()

    if mv_a is None:
        raise ValueError(f"ModelVersion {version_id_a} not found")
    if mv_b is None:
        raise ValueError(f"ModelVersion {version_id_b} not found")

    metrics_a = mv_a.metrics_snapshot or {}
    metrics_b = mv_b.metrics_snapshot or {}

    comparison: dict = {}
    all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
    for key in sorted(all_keys):
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        entry: dict = {"version_a": val_a, "version_b": val_b}
        if val_a is not None and val_b is not None and isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            delta = val_b - val_a
            pct_change = (delta / val_a * 100) if val_a != 0 else None
            entry["delta"] = round(delta, 6)
            entry["pct_change"] = round(pct_change, 2) if pct_change is not None else None
        comparison[key] = entry

    return {
        "version_a": {"id": str(mv_a.id), "version": mv_a.version, "stage": mv_a.stage},
        "version_b": {"id": str(mv_b.id), "version": mv_b.version, "stage": mv_b.stage},
        "metrics": comparison,
    }


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_production_model(
    db: Session,
    model_name: str,
) -> ModelVersion | None:
    """Return the current production version for a model, or None."""
    model = db.query(RegisteredModel).filter(RegisteredModel.name == model_name).first()
    if model is None:
        return None
    return (
        db.query(ModelVersion)
        .filter(ModelVersion.model_id == model.id, ModelVersion.stage == "production")
        .first()
    )


def get_model_history(
    db: Session,
    model_name: str,
) -> tuple[RegisteredModel, list[ModelVersion], list[ModelStageHistory]]:
    """Get full model info: registered model, all versions, and stage history."""
    model = db.query(RegisteredModel).filter(RegisteredModel.name == model_name).first()
    if model is None:
        raise ValueError(f"Model '{model_name}' not found")

    versions = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_id == model.id)
        .order_by(ModelVersion.version.desc())
        .all()
    )

    version_ids = [v.id for v in versions]
    history = (
        db.query(ModelStageHistory)
        .filter(ModelStageHistory.model_version_id.in_(version_ids))
        .order_by(ModelStageHistory.changed_at.desc())
        .all()
    ) if version_ids else []

    return model, versions, history


def list_models(db: Session) -> list[dict]:
    """List all registered models with production version summary."""
    models = db.query(RegisteredModel).order_by(RegisteredModel.name).all()
    results = []
    for m in models:
        version_count = db.query(ModelVersion).filter(ModelVersion.model_id == m.id).count()
        prod = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_id == m.id, ModelVersion.stage == "production")
            .first()
        )
        results.append({
            "id": str(m.id),
            "name": m.name,
            "description": m.description,
            "version_count": version_count,
            "production_version": prod.version if prod else None,
            "production_accuracy": (prod.metrics_snapshot or {}).get("accuracy") if prod else None,
            "created_at": m.created_at,
            "updated_at": m.updated_at,
        })
    return results
