"""Feature store service — versioned, reproducible feature engineering.

Manages feature set definitions (configs), computes features on demand
against datasets, and stores results for reuse by experiments.
"""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from forge.api.models.database import (
    Dataset,
    FeatureSet,
    FeatureStoreRegistry,
)
from forge.api.services.data_ingestion import load_dataset_frame
from forge.api.services.feature_eng import (
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
    fft_spectral_features,
    rolling_autocorrelation,
)

logger = logging.getLogger(__name__)

FEATURE_DATA_DIR = Path(os.getenv("FORGE_FEATURE_DIR", "data/features"))


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


def register_feature_set(
    db: Session,
    name: str,
    feature_config: dict,
    description: str | None = None,
) -> FeatureSet:
    """Register a new feature set version.

    Auto-increments the version number for the given name.
    Returns the newly created FeatureSet record.
    """
    max_version = (
        db.query(func.max(FeatureSet.version))
        .filter(FeatureSet.name == name)
        .scalar()
    )
    next_version = (max_version or 0) + 1

    # Derive expected output columns from the config
    feature_columns = _derive_columns_from_config(feature_config)

    fs = FeatureSet(
        name=name,
        version=next_version,
        description=description,
        feature_config=feature_config,
        feature_columns=feature_columns,
        is_active="true",
    )
    db.add(fs)
    db.commit()
    db.refresh(fs)
    logger.info("Registered feature set '%s' v%d (id=%s)", name, next_version, fs.id)
    return fs


def _derive_columns_from_config(config: dict) -> list[str]:
    """Derive the expected output column names from a feature config."""
    columns: list[str] = []

    technical = config.get("technical", {})
    if "rsi" in technical:
        columns.append("rsi")
    if "macd" in technical:
        columns.extend(["macd_line", "macd_signal", "macd_histogram"])
    if "bbands" in technical:
        columns.extend(["bb_upper", "bb_middle", "bb_lower"])

    signal = config.get("signal", {})
    if "fft" in signal:
        n = signal["fft"].get("n_components", 3)
        columns.extend([f"dominant_freq_{i+1}" for i in range(n)])
        columns.extend(["spectral_entropy", "snr"])
    if "autocorrelation" in signal:
        lags = signal["autocorrelation"].get("lags", [1, 5, 10, 21])
        columns.extend([f"autocorr_lag_{lag}" for lag in lags])

    price = config.get("price", {})
    if "returns" in price:
        periods = price["returns"] if isinstance(price["returns"], list) else [1]
        for p in periods:
            columns.append(f"returns_{p}d")
    if "volatility" in price:
        window = price["volatility"].get("window", 21)
        columns.append(f"volatility_{window}d")

    return columns


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def compute_features(
    db: Session,
    feature_set_id: UUID,
    dataset_id: UUID,
) -> FeatureStoreRegistry:
    """Compute features for a dataset using the specified feature set config.

    Reads the raw dataset parquet, applies feature functions based on
    the config, saves the result to a new parquet file, and updates
    the registry entry.
    """
    fs = db.query(FeatureSet).filter(FeatureSet.id == feature_set_id).first()
    if fs is None:
        raise ValueError(f"FeatureSet {feature_set_id} not found")

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    # Check if already computed
    existing = (
        db.query(FeatureStoreRegistry)
        .filter(
            FeatureStoreRegistry.feature_set_id == feature_set_id,
            FeatureStoreRegistry.dataset_id == dataset_id,
            FeatureStoreRegistry.status == "ready",
        )
        .first()
    )
    if existing:
        logger.info("Features already computed for fs=%s ds=%s", feature_set_id, dataset_id)
        return existing

    # Create registry entry
    registry = FeatureStoreRegistry(
        feature_set_id=feature_set_id,
        dataset_id=dataset_id,
        status="computing",
    )
    db.add(registry)
    db.commit()
    db.refresh(registry)

    try:
        # Load raw dataset
        raw_df = load_dataset_frame(dataset)
        if raw_df is None:
            raise ValueError(f"Dataset parquet not found at {dataset.s3_path}")
        close = raw_df["Close"].values.astype(np.float64)

        result_df = raw_df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        if "ticker" in raw_df.columns:
            result_df["ticker"] = raw_df["ticker"]

        config = fs.feature_config

        # Technical indicators
        technical = config.get("technical", {})
        if "rsi" in technical:
            period = technical["rsi"].get("period", 14)
            result_df["rsi"] = compute_rsi(close, period=period)

        if "macd" in technical:
            macd_cfg = technical["macd"]
            macd_result = compute_macd(
                close,
                fast_period=macd_cfg.get("fast", 12),
                slow_period=macd_cfg.get("slow", 26),
                signal_period=macd_cfg.get("signal", 9),
            )
            result_df["macd_line"] = macd_result["macd_line"]
            result_df["macd_signal"] = macd_result["macd_signal"]
            result_df["macd_histogram"] = macd_result["macd_histogram"]

        if "bbands" in technical:
            bb_cfg = technical["bbands"]
            bb_result = compute_bollinger_bands(
                close,
                period=bb_cfg.get("period", 20),
                num_std=bb_cfg.get("num_std", 2.0),
            )
            result_df["bb_upper"] = bb_result["bb_upper"]
            result_df["bb_middle"] = bb_result["bb_middle"]
            result_df["bb_lower"] = bb_result["bb_lower"]

        # Signal processing features
        signal_cfg = config.get("signal", {})
        if "fft" in signal_cfg:
            n_components = signal_cfg["fft"].get("n_components", 3)
            fft_features = fft_spectral_features(close)
            # Map top N frequencies
            for i in range(n_components):
                key = f"dominant_freq_{i+1}"
                result_df[key] = fft_features.get(key, None)
            result_df["spectral_entropy"] = fft_features.get("spectral_entropy")
            result_df["snr"] = fft_features.get("snr")

        if "autocorrelation" in signal_cfg:
            lags = signal_cfg["autocorrelation"].get("lags", [1, 5, 10, 21])
            autocorr = rolling_autocorrelation(close, lags=lags)
            for key, value in autocorr.items():
                result_df[key] = value

        # Price-derived features
        price_cfg = config.get("price", {})
        if "returns" in price_cfg:
            periods = price_cfg["returns"] if isinstance(price_cfg["returns"], list) else [1]
            for p in periods:
                result_df[f"returns_{p}d"] = raw_df["Close"].pct_change(periods=p)

        if "volatility" in price_cfg:
            window = price_cfg["volatility"].get("window", 21)
            result_df[f"volatility_{window}d"] = raw_df["Close"].pct_change().rolling(window).std()

        # Save to parquet
        FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        storage_path = str(FEATURE_DATA_DIR / f"{registry.id}.parquet")
        result_df.to_parquet(storage_path, index=False)

        # Update registry
        registry.storage_path = storage_path
        registry.row_count = len(result_df)
        registry.computed_at = datetime.now(UTC)
        registry.status = "ready"
        db.commit()
        db.refresh(registry)

        logger.info(
            "Computed features: fs=%s ds=%s -> %d rows, %d columns",
            feature_set_id, dataset_id, len(result_df), len(result_df.columns),
        )
        return registry

    except Exception:
        registry.status = "failed"
        db.commit()
        logger.exception("Feature computation failed for fs=%s ds=%s", feature_set_id, dataset_id)
        raise


# ---------------------------------------------------------------------------
# Get features
# ---------------------------------------------------------------------------


def get_features(
    db: Session,
    feature_set_id: UUID,
    dataset_id: UUID,
) -> pd.DataFrame:
    """Return the computed feature DataFrame for a feature set + dataset pair.

    Raises ValueError if features haven't been computed yet.
    """
    registry = (
        db.query(FeatureStoreRegistry)
        .filter(
            FeatureStoreRegistry.feature_set_id == feature_set_id,
            FeatureStoreRegistry.dataset_id == dataset_id,
            FeatureStoreRegistry.status == "ready",
        )
        .first()
    )
    if registry is None:
        raise ValueError(
            f"No computed features found for feature_set={feature_set_id}, "
            f"dataset={dataset_id}. Run compute_features first."
        )

    if not registry.storage_path or not Path(registry.storage_path).exists():
        raise ValueError(f"Feature parquet not found at {registry.storage_path}")

    return pd.read_parquet(registry.storage_path)


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def list_feature_sets(
    db: Session,
    name: str | None = None,
) -> list[FeatureSet]:
    """List all feature sets, optionally filtered by name."""
    query = db.query(FeatureSet).order_by(FeatureSet.name, FeatureSet.version.desc())
    if name:
        query = query.filter(FeatureSet.name == name)
    return query.all()


def get_feature_set_detail(
    db: Session,
    feature_set_id: UUID,
) -> tuple[FeatureSet, list[FeatureStoreRegistry]]:
    """Get a feature set with all its registry entries."""
    fs = db.query(FeatureSet).filter(FeatureSet.id == feature_set_id).first()
    if fs is None:
        raise ValueError(f"FeatureSet {feature_set_id} not found")

    registry_entries = (
        db.query(FeatureStoreRegistry)
        .filter(FeatureStoreRegistry.feature_set_id == feature_set_id)
        .order_by(FeatureStoreRegistry.computed_at.desc())
        .all()
    )
    return fs, registry_entries


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def compare_feature_sets(
    db: Session,
    id_a: UUID,
    id_b: UUID,
) -> dict:
    """Diff two feature set configs, showing added/removed/changed features.

    Returns a dict with keys: added, removed, changed, and summary info
    about each feature set.
    """
    fs_a = db.query(FeatureSet).filter(FeatureSet.id == id_a).first()
    fs_b = db.query(FeatureSet).filter(FeatureSet.id == id_b).first()

    if fs_a is None:
        raise ValueError(f"FeatureSet {id_a} not found")
    if fs_b is None:
        raise ValueError(f"FeatureSet {id_b} not found")

    config_a = fs_a.feature_config or {}
    config_b = fs_b.feature_config or {}

    cols_a = set(fs_a.feature_columns or [])
    cols_b = set(fs_b.feature_columns or [])

    # Config-level diff
    all_keys = set(_flatten_config(config_a).keys()) | set(_flatten_config(config_b).keys())
    flat_a = _flatten_config(config_a)
    flat_b = _flatten_config(config_b)

    added = {k: flat_b[k] for k in all_keys if k in flat_b and k not in flat_a}
    removed = {k: flat_a[k] for k in all_keys if k in flat_a and k not in flat_b}
    changed = {
        k: {"from": flat_a[k], "to": flat_b[k]}
        for k in all_keys
        if k in flat_a and k in flat_b and flat_a[k] != flat_b[k]
    }

    return {
        "feature_set_a": {"id": str(fs_a.id), "name": fs_a.name, "version": fs_a.version},
        "feature_set_b": {"id": str(fs_b.id), "name": fs_b.name, "version": fs_b.version},
        "columns_added": sorted(cols_b - cols_a),
        "columns_removed": sorted(cols_a - cols_b),
        "config_added": added,
        "config_removed": removed,
        "config_changed": changed,
    }


def _flatten_config(config: dict, prefix: str = "") -> dict:
    """Flatten a nested config dict into dot-separated keys for comparison."""
    result: dict = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_config(value, full_key))
        else:
            result[full_key] = value
    return result
