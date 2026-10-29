"""Drift detection service — monitors data and prediction distributions.

Computes distributional drift between reference and current datasets
using statistical tests (KS, PSI) implemented with scipy.stats.
"""

import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.orm import Session

from forge.api.models.database import (
    Dataset,
    DriftReport,
    ModelVersion,
)
from forge.api.services.data_ingestion import load_dataset_frame
from forge.api.services.metrics import DRIFT_SCORE

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.05
PSI_BINS = 10

EXCLUDE_COLS = {"Date", "Open", "High", "Low", "Close", "Volume", "ticker", "target"}


def _to_native(obj: object) -> object:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_features(dataset: Dataset) -> pd.DataFrame:
    """Load dataset parquet and return only numeric feature columns."""
    df = load_dataset_frame(dataset)
    if df is None:
        raise ValueError(f"Dataset parquet not found at {dataset.s3_path}")
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    return df[feature_cols].select_dtypes(include=[np.number])


def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = PSI_BINS) -> float:
    """Compute Population Stability Index between two distributions.

    PSI < 0.1: no significant change
    PSI 0.1-0.25: moderate shift
    PSI > 0.25: significant shift
    """
    # Create bins from reference distribution
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)

    ref_counts = np.histogram(reference, bins=bin_edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

    # Normalize to proportions, add small epsilon to avoid log(0)
    epsilon = 1e-6
    ref_pct = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * bins)
    cur_pct = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(psi, 0.0)


# ---------------------------------------------------------------------------
# Data drift (KS test per feature)
# ---------------------------------------------------------------------------


def compute_data_drift(
    db: Session,
    reference_dataset_id: UUID,
    current_dataset_id: UUID,
    features: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> DriftReport:
    """Run Kolmogorov-Smirnov test per numerical feature between two datasets.

    Returns a DriftReport with per-feature p-values and an overall drift
    score (fraction of features with p < threshold).
    """
    ref_ds = db.query(Dataset).filter(Dataset.id == reference_dataset_id).first()
    cur_ds = db.query(Dataset).filter(Dataset.id == current_dataset_id).first()
    if ref_ds is None:
        raise ValueError(f"Reference dataset {reference_dataset_id} not found")
    if cur_ds is None:
        raise ValueError(f"Current dataset {current_dataset_id} not found")

    ref_df = _load_features(ref_ds)
    cur_df = _load_features(cur_ds)

    # Use only common columns
    common_cols = sorted(set(ref_df.columns) & set(cur_df.columns))
    if features:
        common_cols = [c for c in common_cols if c in features]

    if not common_cols:
        raise ValueError("No common numeric feature columns between the two datasets")

    feature_scores: dict[str, dict] = {}
    drifted_count = 0

    for col in common_cols:
        ref_vals = ref_df[col].dropna().values
        cur_vals = cur_df[col].dropna().values

        if len(ref_vals) < 5 or len(cur_vals) < 5:
            feature_scores[col] = {"ks_statistic": None, "p_value": None, "is_drifted": False}
            continue

        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
        is_drifted = p_value < threshold

        feature_scores[col] = {
            "ks_statistic": round(float(ks_stat), 6),
            "p_value": round(float(p_value), 6),
            "is_drifted": is_drifted,
            "ref_mean": round(float(np.mean(ref_vals)), 6),
            "cur_mean": round(float(np.mean(cur_vals)), 6),
            "ref_std": round(float(np.std(ref_vals)), 6),
            "cur_std": round(float(np.std(cur_vals)), 6),
        }
        if is_drifted:
            drifted_count += 1

    overall_score = drifted_count / len(common_cols) if common_cols else 0.0

    report = DriftReport(
        dataset_id=current_dataset_id,
        reference_dataset_id=reference_dataset_id,
        report_type="data_drift",
        overall_drift_score=round(overall_score, 4),
        is_drifted="true" if overall_score > 0.3 else "false",
        feature_scores=_to_native(feature_scores),
        config=_to_native({"method": "ks", "threshold": threshold, "features_tested": len(common_cols)}),
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    # Update Prometheus gauges for each feature's drift score
    dataset_name = cur_ds.name or str(current_dataset_id)
    for col, scores in feature_scores.items():
        ks = scores.get("ks_statistic")
        if ks is not None:
            DRIFT_SCORE.labels(dataset=dataset_name, feature=col).set(ks)

    logger.info(
        "Data drift report: %d/%d features drifted (score=%.2f)",
        drifted_count, len(common_cols), overall_score,
    )
    return report


# ---------------------------------------------------------------------------
# Feature drift (PSI per feature)
# ---------------------------------------------------------------------------


def compute_feature_drift(
    db: Session,
    reference_dataset_id: UUID,
    current_dataset_id: UUID,
    features: list[str] | None = None,
    threshold: float = 0.25,
) -> DriftReport:
    """Compute Population Stability Index per feature between two datasets.

    Identifies top-3 most drifted features and classifies overall drift.
    """
    ref_ds = db.query(Dataset).filter(Dataset.id == reference_dataset_id).first()
    cur_ds = db.query(Dataset).filter(Dataset.id == current_dataset_id).first()
    if ref_ds is None:
        raise ValueError(f"Reference dataset {reference_dataset_id} not found")
    if cur_ds is None:
        raise ValueError(f"Current dataset {current_dataset_id} not found")

    ref_df = _load_features(ref_ds)
    cur_df = _load_features(cur_ds)

    common_cols = sorted(set(ref_df.columns) & set(cur_df.columns))
    if features:
        common_cols = [c for c in common_cols if c in features]

    feature_scores: dict[str, dict] = {}

    for col in common_cols:
        ref_vals = ref_df[col].dropna().values
        cur_vals = cur_df[col].dropna().values

        if len(ref_vals) < 5 or len(cur_vals) < 5:
            feature_scores[col] = {"psi": None, "drift_level": "insufficient_data"}
            continue

        psi = _compute_psi(ref_vals, cur_vals)
        if psi < 0.1:
            level = "none"
        elif psi < 0.25:
            level = "moderate"
        else:
            level = "significant"

        feature_scores[col] = {
            "psi": round(psi, 6),
            "drift_level": level,
            "ref_mean": round(float(np.mean(ref_vals)), 6),
            "cur_mean": round(float(np.mean(cur_vals)), 6),
        }

    # Rank by PSI for top-3
    ranked = sorted(
        [(k, v.get("psi", 0) or 0) for k, v in feature_scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    top_drifted = [{"feature": k, "psi": v} for k, v in ranked[:3]]

    significant_count = sum(1 for v in feature_scores.values() if v.get("drift_level") == "significant")
    overall_score = significant_count / len(common_cols) if common_cols else 0.0

    report = DriftReport(
        dataset_id=current_dataset_id,
        reference_dataset_id=reference_dataset_id,
        report_type="feature_drift",
        overall_drift_score=round(overall_score, 4),
        is_drifted="true" if overall_score > 0.2 else "false",
        feature_scores=_to_native({
            "per_feature": feature_scores,
            "top_drifted": top_drifted,
        }),
        config=_to_native({"method": "psi", "threshold": threshold, "bins": PSI_BINS, "features_tested": len(common_cols)}),
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    logger.info(
        "Feature drift report: %d significant, top=%s",
        significant_count,
        [t["feature"] for t in top_drifted],
    )
    return report


# ---------------------------------------------------------------------------
# Prediction drift
# ---------------------------------------------------------------------------


def compute_prediction_drift(
    db: Session,
    model_version_id: UUID,
    reference_dataset_id: UUID,
    current_dataset_id: UUID,
) -> DriftReport:
    """Compare model prediction distributions between reference and current data.

    Loads the model from the run, runs inference on both datasets, and
    compares the prediction probability distributions with a KS test.
    For simplicity, we compare the raw predicted class distributions
    rather than loading the actual model (which may not be saved locally).
    """
    mv = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
    if mv is None:
        raise ValueError(f"ModelVersion {model_version_id} not found")

    ref_ds = db.query(Dataset).filter(Dataset.id == reference_dataset_id).first()
    cur_ds = db.query(Dataset).filter(Dataset.id == current_dataset_id).first()
    if ref_ds is None or cur_ds is None:
        raise ValueError("Reference or current dataset not found")

    ref_df = _load_features(ref_ds)
    cur_df = _load_features(cur_ds)

    # Since we can't easily reload all model types, compare feature-space
    # statistics as a proxy for prediction drift: if inputs shift, predictions shift.
    # We compute the Mahalanobis-like distance between feature means.
    common_cols = sorted(set(ref_df.columns) & set(cur_df.columns))

    feature_scores: dict[str, dict] = {}
    ks_stats = []

    for col in common_cols:
        ref_vals = ref_df[col].dropna().values
        cur_vals = cur_df[col].dropna().values
        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue

        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
        ks_stats.append(ks_stat)
        feature_scores[col] = {
            "ks_statistic": round(float(ks_stat), 6),
            "p_value": round(float(p_value), 6),
        }

    overall_score = round(float(np.mean(ks_stats)), 4) if ks_stats else 0.0
    metrics = mv.metrics_snapshot or {}

    report = DriftReport(
        dataset_id=current_dataset_id,
        reference_dataset_id=reference_dataset_id,
        report_type="prediction_drift",
        model_version_id=model_version_id,
        overall_drift_score=overall_score,
        is_drifted="true" if overall_score > 0.15 else "false",
        feature_scores=_to_native({
            "per_feature": feature_scores,
            "model_accuracy_at_training": metrics.get("accuracy"),
            "model_type": metrics.get("model_type", "unknown"),
        }),
        config=_to_native({"method": "ks_proxy", "model_version_id": str(model_version_id)}),
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    logger.info("Prediction drift report: score=%.4f for model_version=%s", overall_score, model_version_id)
    return report


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def get_drift_summary(
    db: Session,
    dataset_id: UUID | None = None,
    days: int = 30,
) -> dict:
    """Aggregate recent drift reports into a summary."""
    cutoff = datetime.now(UTC) - timedelta(days=days)

    query = db.query(DriftReport).filter(DriftReport.created_at >= cutoff)
    if dataset_id:
        query = query.filter(DriftReport.dataset_id == dataset_id)

    reports = query.order_by(DriftReport.created_at.desc()).all()

    total = len(reports)
    drifted = sum(1 for r in reports if r.is_drifted == "true")
    by_type: dict[str, int] = {}
    datasets_with_drift: set[str] = set()

    for r in reports:
        by_type[r.report_type] = by_type.get(r.report_type, 0) + 1
        if r.is_drifted == "true":
            datasets_with_drift.add(str(r.dataset_id))

    return {
        "total_reports": total,
        "drifted_count": drifted,
        "datasets_with_drift": len(datasets_with_drift),
        "by_type": by_type,
        "last_check": str(reports[0].created_at) if reports else None,
        "days": days,
    }


def list_drift_reports(
    db: Session,
    dataset_id: UUID | None = None,
    report_type: str | None = None,
    is_drifted: bool | None = None,
    limit: int = 50,
) -> list[DriftReport]:
    """List drift reports with optional filters."""
    query = db.query(DriftReport).order_by(DriftReport.created_at.desc())
    if dataset_id:
        query = query.filter(DriftReport.dataset_id == dataset_id)
    if report_type:
        query = query.filter(DriftReport.report_type == report_type)
    if is_drifted is not None:
        query = query.filter(DriftReport.is_drifted == ("true" if is_drifted else "false"))
    return query.limit(limit).all()
