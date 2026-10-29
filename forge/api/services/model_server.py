"""Model serving service — loads production models from artifacts and runs inference.

A registered model's production version is rebuilt from its stored artifact
(PyTorch state_dict or joblib pickle plus a ``meta.json`` sidecar), warmed up,
and kept in an in-process cache keyed by model name. Promoting a different
version evicts the cache entry so the next request loads the new version
without a restart.
"""

import asyncio
import io
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

import joblib
import numpy as np
import torch
from sqlalchemy.orm import Session
from torch import nn

from forge.api.models.database import ModelVersion, RegisteredModel, Run
from forge.api.services.batcher import BatchConfig, BatcherPool, MicroBatcher
from forge.api.services.metrics import (
    SERVING_CACHE_HITS_TOTAL,
    SERVING_MODEL_LOADS_TOTAL,
    SERVING_REQUEST_SECONDS,
)
from forge.api.services.s3_client import ArtifactNotFoundError, download_model_artifact
from forge.api.services.training import (
    DEFAULT_CONFIGS,
    SEQUENCE_MODEL_TYPES,
    TimeSeriesCNNLSTM,
    TimeSeriesLSTM,
    TimeSeriesTCN,
    TimeSeriesTransformer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors — mapped to HTTP status codes by the serving router
# ---------------------------------------------------------------------------


class ServingError(Exception):
    """Base class for serving failures."""


class ModelNotFoundError(ServingError):
    """The model name is not registered, or it has no production version."""


class ModelArtifactError(ServingError):
    """The production version has no usable artifact (missing, unreadable, or unbuildable)."""


class InvalidInputError(ServingError):
    """The request inputs don't match the model's expected shape."""


# ---------------------------------------------------------------------------
# Rebuilding models from artifacts
# ---------------------------------------------------------------------------

_ARCHITECTURES: dict[str, Callable[[int, dict], nn.Module]] = {
    "lstm": lambda size, hp: TimeSeriesLSTM(
        size, hp["hidden_size"], hp["num_layers"], hp["dropout"]
    ),
    "tcn": lambda size, hp: TimeSeriesTCN(
        size, hp["num_channels"], hp["num_layers"], hp["kernel_size"], hp["dropout"]
    ),
    "cnn_lstm": lambda size, hp: TimeSeriesCNNLSTM(
        size, hp["cnn_filters"], hp["cnn_kernel_size"], hp["lstm_hidden"], hp["lstm_layers"], hp["dropout"]
    ),
    "transformer": lambda size, hp: TimeSeriesTransformer(
        size, hp["d_model"], hp["nhead"], hp["num_layers"], hp["dim_feedforward"], hp["dropout"]
    ),
}

_TABULAR_MODEL_TYPES = {"xgboost", "random_forest"}


def build_model(artifact_bytes: bytes, metadata: dict) -> object:
    """Rebuild a trained model from its serialized artifact.

    Args:
        artifact_bytes: Raw artifact (torch state_dict or joblib pickle).
        metadata: Artifact metadata; PyTorch models need ``input_size`` and
            the training ``hyperparameters`` to reconstruct the architecture.

    Returns:
        The model, in eval mode for PyTorch.

    Raises:
        ModelArtifactError: If the model type is unknown or the artifact can't be loaded.
    """
    model_type = metadata.get("model_type")
    if model_type in _ARCHITECTURES:
        input_size = metadata.get("input_size")
        if input_size is None:
            raise ModelArtifactError(
                f"{model_type} artifact has no input_size metadata; retrain to make it servable"
            )
        hyperparams = {**DEFAULT_CONFIGS[model_type], **(metadata.get("hyperparameters") or {})}
        model = _ARCHITECTURES[model_type](int(input_size), hyperparams)
        try:
            state_dict = torch.load(io.BytesIO(artifact_bytes), map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
        except (RuntimeError, KeyError, EOFError) as exc:
            raise ModelArtifactError(f"Cannot load {model_type} weights: {exc}") from exc
        return model.eval()

    if model_type in _TABULAR_MODEL_TYPES:
        try:
            return joblib.load(io.BytesIO(artifact_bytes))
        except (EOFError, ValueError, ImportError) as exc:
            raise ModelArtifactError(f"Cannot unpickle {model_type} artifact: {exc}") from exc

    raise ModelArtifactError(f"Unsupported model type for serving: {model_type}")


# ---------------------------------------------------------------------------
# Servable model + inference
# ---------------------------------------------------------------------------


@dataclass
class ServableModel:
    """A loaded, warmed-up model ready to answer prediction requests."""

    name: str
    version: int
    run_id: UUID
    model_type: str
    backend: str
    model: object
    input_shape: tuple[int, ...]
    feature_names: list[str] = field(default_factory=list)
    loaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    load_seconds: float = 0.0


@dataclass
class InferenceResult:
    """Output of one prediction request."""

    probabilities: list[float]
    labels: list[int]
    latency_ms: float


def _backend_for(model: object) -> str:
    """Name the inference backend used for a model (a metrics label)."""
    if isinstance(model, nn.Module):
        return "pytorch"
    if type(model).__module__.startswith("xgboost"):
        return "xgboost"
    return "sklearn"


def predict_proba(servable: ServableModel, inputs: np.ndarray) -> np.ndarray:
    """Return the positive-class probability for each input row or sequence."""
    if isinstance(servable.model, nn.Module):
        with torch.inference_mode():
            output = servable.model(torch.from_numpy(inputs))
        return output.reshape(-1).numpy()
    return servable.model.predict_proba(inputs)[:, 1]


def create_servable(
    name: str,
    version: int,
    run_id: UUID,
    artifact_bytes: bytes,
    metadata: dict,
) -> ServableModel:
    """Build a model from its artifact, warm it up, and wrap it for serving.

    Raises:
        ModelArtifactError: If the model can't be rebuilt or its input size is unknown.
    """
    start = time.perf_counter()
    model = build_model(artifact_bytes, metadata)
    model_type = metadata["model_type"]

    input_size = metadata.get("input_size") or getattr(model, "n_features_in_", None)
    if input_size is None:
        raise ModelArtifactError(f"Cannot determine input size for {name} v{version}")
    if model_type in SEQUENCE_MODEL_TYPES:
        window = metadata.get("window_size") or DEFAULT_CONFIGS[model_type]["window_size"]
        input_shape: tuple[int, ...] = (int(window), int(input_size))
    else:
        input_shape = (int(input_size),)

    servable = ServableModel(
        name=name,
        version=version,
        run_id=run_id,
        model_type=model_type,
        backend=_backend_for(model),
        model=model,
        input_shape=input_shape,
        feature_names=list(metadata.get("feature_names") or []),
    )
    # Warm-up pass so the first real request doesn't pay one-time init costs
    predict_proba(servable, np.zeros((1, *input_shape), dtype=np.float32))
    servable.load_seconds = time.perf_counter() - start
    return servable


def prepare_inputs(servable: ServableModel, raw_inputs: list) -> np.ndarray:
    """Convert a request payload to a float32 array and validate its shape.

    Raises:
        InvalidInputError: If the payload is ragged, non-finite, or the wrong shape.
    """
    try:
        inputs = np.asarray(raw_inputs, dtype=np.float32)
    except ValueError as exc:
        raise InvalidInputError(f"Inputs must be a rectangular array of numbers: {exc}") from exc

    if inputs.ndim != len(servable.input_shape) + 1 or inputs.shape[1:] != servable.input_shape:
        raise InvalidInputError(
            f"Got input shape {list(inputs.shape)}, expected [batch, "
            f"{', '.join(str(dim) for dim in servable.input_shape)}]"
        )
    if not np.isfinite(inputs).all():
        raise InvalidInputError("Inputs contain NaN or infinite values")
    return inputs


def run_inference(servable: ServableModel, raw_inputs: list) -> InferenceResult:
    """Validate inputs, run the model synchronously, and record serving latency."""
    inputs = prepare_inputs(servable, raw_inputs)
    start = time.perf_counter()
    probabilities = predict_proba(servable, inputs)
    return _record_result(servable, probabilities, time.perf_counter() - start)


async def infer(servable: ServableModel, raw_inputs: list) -> InferenceResult:
    """Validate inputs and predict, through the model's micro-batcher when enabled.

    The recorded latency is server-side end to end: queue wait plus forward pass.
    """
    inputs = prepare_inputs(servable, raw_inputs)
    config = get_batch_config(servable.name)
    start = time.perf_counter()
    if config.enabled:
        batcher = batcher_pool.get_or_create(
            (servable.name, servable.version, servable.backend),
            lambda: MicroBatcher(
                lambda batch: predict_proba(servable, batch), config, label=servable.name
            ),
        )
        probabilities = await batcher.submit(inputs)
    else:
        probabilities = await asyncio.to_thread(predict_proba, servable, inputs)
    return _record_result(servable, probabilities, time.perf_counter() - start)


def _record_result(
    servable: ServableModel, probabilities: np.ndarray, elapsed: float
) -> InferenceResult:
    """Observe serving latency and package probabilities with thresholded labels."""
    SERVING_REQUEST_SECONDS.labels(
        model=servable.name, version=str(servable.version), backend=servable.backend
    ).observe(elapsed)
    probs = [float(p) for p in probabilities]
    return InferenceResult(
        probabilities=probs,
        labels=[int(p >= 0.5) for p in probs],
        latency_ms=elapsed * 1000.0,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class ModelCache:
    """Thread-safe cache of loaded production models, keyed by model name.

    Concurrent requests for a model that isn't loaded yet wait on a per-model
    lock, so the artifact is downloaded and rebuilt only once.
    """

    def __init__(self) -> None:
        self._models: dict[str, ServableModel] = {}
        self._load_locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get_or_load(self, name: str, loader: Callable[[], ServableModel]) -> ServableModel:
        """Return the cached model, loading it with ``loader`` on a miss."""
        cached = self._models.get(name)
        if cached is not None:
            SERVING_CACHE_HITS_TOTAL.labels(model=name).inc()
            return cached

        with self._guard:
            load_lock = self._load_locks.setdefault(name, threading.Lock())
        with load_lock:
            cached = self._models.get(name)
            if cached is not None:
                SERVING_CACHE_HITS_TOTAL.labels(model=name).inc()
                return cached
            try:
                servable = loader()
            except ServingError:
                SERVING_MODEL_LOADS_TOTAL.labels(model=name, status="failed").inc()
                raise
            SERVING_MODEL_LOADS_TOTAL.labels(model=name, status="loaded").inc()
            self._models[name] = servable
            logger.info(
                "Loaded %s v%d for serving (%s, %.2fs)",
                name, servable.version, servable.backend, servable.load_seconds,
            )
            return servable

    def evict(self, name: str) -> bool:
        """Drop a model from the cache. Returns True if it was loaded."""
        with self._guard:
            removed = self._models.pop(name, None)
        if removed is not None:
            logger.info("Evicted %s v%d from serving cache", name, removed.version)
        return removed is not None

    def loaded(self) -> list[ServableModel]:
        """Return all currently loaded models."""
        return list(self._models.values())


model_cache = ModelCache()


# ---------------------------------------------------------------------------
# Micro-batching configuration
# ---------------------------------------------------------------------------

DEFAULT_BATCH_CONFIG = BatchConfig(
    max_batch_size=int(os.getenv("FORGE_SERVE_MAX_BATCH", "32")),
    max_wait_ms=float(os.getenv("FORGE_SERVE_MAX_WAIT_MS", "2")),
)
_batch_configs: dict[str, BatchConfig] = {}
batcher_pool = BatcherPool()


def get_batch_config(name: str) -> BatchConfig:
    """Return the micro-batching policy for a model (the default unless overridden)."""
    return _batch_configs.get(name, DEFAULT_BATCH_CONFIG)


def set_batch_config(name: str, config: BatchConfig) -> BatchConfig:
    """Override a model's batching policy; running batchers drain and are replaced."""
    _batch_configs[name] = config
    batcher_pool.retire_model(name)
    return config


def evict_model(name: str) -> None:
    """Hot swap: drop the cached model and retire its batchers (queued requests still finish)."""
    model_cache.evict(name)
    batcher_pool.retire_model(name)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def load_production_model(db: Session, name: str) -> ServableModel:
    """Resolve a model's production version and build it from its artifact.

    Raises:
        ModelNotFoundError: If the model isn't registered or has no production version.
        ModelArtifactError: If the version has no stored artifact or it can't be loaded.
    """
    registered = db.query(RegisteredModel).filter(RegisteredModel.name == name).first()
    if registered is None:
        raise ModelNotFoundError(f"Model '{name}' is not registered")

    version = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_id == registered.id, ModelVersion.stage == "production")
        .first()
    )
    if version is None:
        raise ModelNotFoundError(f"Model '{name}' has no production version")

    run = db.query(Run).filter(Run.id == version.run_id).first()
    artifact_path = version.s3_artifact_path or (run.s3_artifact_path if run else None)
    if artifact_path is None:
        raise ModelArtifactError(
            f"'{name}' v{version.version} has no stored artifact; retrain and promote a new version"
        )

    try:
        artifact_bytes, metadata = download_model_artifact(artifact_path)
    except ArtifactNotFoundError as exc:
        raise ModelArtifactError(str(exc)) from exc

    if metadata is None:
        # Artifacts saved before metadata sidecars existed: tabular models can
        # still be served because the estimator records its input width.
        metadata = {
            "model_type": run.model_type if run else None,
            "hyperparameters": (run.hyperparameters if run else None) or {},
        }
    return create_servable(name, version.version, version.run_id, artifact_bytes, metadata)


def get_production_servable(db: Session, name: str) -> ServableModel:
    """Return the cached production model for ``name``, loading it if needed."""
    return model_cache.get_or_load(name, lambda: load_production_model(db, name))
