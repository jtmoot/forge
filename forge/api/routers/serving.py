"""Serving router — online predictions from production-stage registry models."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from forge.api.models.database import get_db
from forge.api.models.schemas import (
    BatchConfigRequest,
    BatchConfigResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
    ServedModelListResponse,
    ServedModelResponse,
    ServingRuntime,
)
from forge.api.services.batcher import BatchConfig
from forge.api.services.model_server import (
    InvalidInputError,
    ModelArtifactError,
    ModelNotFoundError,
    ServableModel,
    ServingError,
    get_batch_config,
    get_production_servable,
    infer,
    model_cache,
    set_batch_config,
)
from forge.api.services.runtime import runtime_info

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/serve", tags=["serving"])

_STATUS_FOR_ERROR: dict[type[ServingError], int] = {
    ModelNotFoundError: status.HTTP_404_NOT_FOUND,
    ModelArtifactError: status.HTTP_409_CONFLICT,
    InvalidInputError: status.HTTP_422_UNPROCESSABLE_ENTITY,
}


def _to_http_error(exc: ServingError) -> HTTPException:
    """Map a serving error to the matching HTTP status."""
    return HTTPException(
        status_code=_STATUS_FOR_ERROR.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR),
        detail=str(exc),
    )


def _batching(name: str, config: BatchConfig) -> BatchConfigResponse:
    """Build the API description of a batching policy."""
    return BatchConfigResponse(
        model_name=name,
        max_batch_size=config.max_batch_size,
        max_wait_ms=config.max_wait_ms,
        enabled=config.enabled,
    )


def _describe(servable: ServableModel) -> ServedModelResponse:
    """Build the API description of a loaded model."""
    return ServedModelResponse(
        model_name=servable.name,
        version=servable.version,
        run_id=servable.run_id,
        model_type=servable.model_type,
        backend=servable.backend,
        input_shape=list(servable.input_shape),
        feature_names=servable.feature_names,
        loaded_at=servable.loaded_at,
        load_seconds=servable.load_seconds,
        batching=_batching(servable.name, get_batch_config(servable.name)),
    )


@router.get("", response_model=ServedModelListResponse)
def list_served_models() -> ServedModelListResponse:
    """List models currently loaded in the serving cache, and the runtime serving them."""
    return ServedModelListResponse(
        models=[_describe(s) for s in model_cache.loaded()],
        runtime=ServingRuntime(**runtime_info()),
    )


@router.get("/{model_name}", response_model=ServedModelResponse)
def get_served_model(
    model_name: str,
    db: Session = Depends(get_db),
) -> ServedModelResponse:
    """Load (if needed) and describe the production version of a model.

    Also useful as a warm-up call before sending traffic.
    """
    try:
        servable = get_production_servable(db, model_name)
    except ServingError as exc:
        raise _to_http_error(exc) from exc
    return _describe(servable)


@router.get("/{model_name}/batching", response_model=BatchConfigResponse)
def get_batching(model_name: str) -> BatchConfigResponse:
    """Return a model's micro-batching policy."""
    return _batching(model_name, get_batch_config(model_name))


@router.put("/{model_name}/batching", response_model=BatchConfigResponse)
def update_batching(model_name: str, request: BatchConfigRequest) -> BatchConfigResponse:
    """Change a model's micro-batching policy; in-flight requests finish under the old one."""
    config = set_batch_config(
        model_name, BatchConfig(request.max_batch_size, request.max_wait_ms)
    )
    return _batching(model_name, config)


@router.post("/{model_name}/predict", response_model=PredictResponse)
async def predict(
    model_name: str,
    request: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """Run inference with the model's current production version.

    Concurrent requests are micro-batched per the model's batching policy.
    """
    try:
        servable = await run_in_threadpool(get_production_servable, db, model_name)
        result = await infer(servable, request.inputs)
    except ServingError as exc:
        raise _to_http_error(exc) from exc

    return PredictResponse(
        model_name=servable.name,
        version=servable.version,
        model_type=servable.model_type,
        backend=servable.backend,
        predictions=[
            Prediction(probability=prob, label=label)
            for prob, label in zip(result.probabilities, result.labels, strict=True)
        ],
        latency_ms=result.latency_ms,
    )
