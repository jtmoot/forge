"""Serving router — online predictions from production-stage registry models."""

import logging
from dataclasses import asdict
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from forge.api.models.database import get_db
from forge.api.models.schemas import (
    BackendParity,
    BatchConfigRequest,
    BatchConfigResponse,
    BenchmarkListResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkSummary,
    Prediction,
    PredictRequest,
    PredictResponse,
    ServedModelListResponse,
    ServedModelResponse,
    ServingRecommendationResponse,
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
from forge.api.services.serving_bench import (
    DEFAULT_GRID,
    BenchmarkError,
    BenchmarkGrid,
    execute_benchmark,
    get_benchmark,
    latest_completed_benchmark,
    list_benchmarks,
    recommend_config,
    start_benchmark,
)

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
        backends=list(servable.backends),
        parity=[BackendParity(**asdict(result)) for result in servable.parity.values()],
        input_shape=list(servable.input_shape),
        feature_names=servable.feature_names,
        loaded_at=servable.loaded_at,
        load_seconds=servable.load_seconds,
        onnx_build_seconds=servable.onnx_build_seconds,
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
    backend: str | None = Query(
        default=None,
        description="Inference backend, e.g. onnxruntime or onnxruntime-int8; defaults to the native one",
    ),
    db: Session = Depends(get_db),
) -> PredictResponse:
    """Run inference with the model's current production version.

    Concurrent requests are micro-batched per the model's batching policy.
    """
    try:
        servable = await run_in_threadpool(get_production_servable, db, model_name)
        result = await infer(servable, request.inputs, backend)
    except ServingError as exc:
        raise _to_http_error(exc) from exc

    return PredictResponse(
        model_name=servable.name,
        version=servable.version,
        model_type=servable.model_type,
        backend=result.backend,
        predictions=[
            Prediction(probability=prob, label=label)
            for prob, label in zip(result.probabilities, result.labels, strict=True)
        ],
        latency_ms=result.latency_ms,
    )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@router.get("/benchmarks/{benchmark_id}", response_model=BenchmarkResponse)
def get_benchmark_endpoint(
    benchmark_id: UUID,
    db: Session = Depends(get_db),
) -> BenchmarkResponse:
    """Return a benchmark and every cell completed so far (poll while it runs)."""
    benchmark = get_benchmark(db, benchmark_id)
    if benchmark is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Benchmark not found")
    return BenchmarkResponse.model_validate(benchmark)


@router.post(
    "/{model_name}/benchmarks",
    response_model=BenchmarkResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_benchmark(
    model_name: str,
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> BenchmarkResponse:
    """Start a load test of the production model across backends, batching and concurrency.

    Runs in the background (about a minute for the default 48-cell grid);
    poll GET /api/serve/benchmarks/{id}. Only one benchmark runs at a time.
    """
    try:
        grid = BenchmarkGrid(
            backends=tuple(request.backends) if request.backends else DEFAULT_GRID.backends,
            batch_configs=(
                tuple(BatchConfig(c.max_batch_size, c.max_wait_ms) for c in request.batch_configs)
                if request.batch_configs
                else DEFAULT_GRID.batch_configs
            ),
            concurrency=tuple(request.concurrency) if request.concurrency else DEFAULT_GRID.concurrency,
            requests_per_cell=request.requests_per_cell,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    try:
        servable = await run_in_threadpool(get_production_servable, db, model_name)
        grid.check_size(len(servable.backends))
        benchmark = await run_in_threadpool(start_benchmark, db, model_name, servable.version, grid)
    except ServingError as exc:
        raise _to_http_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except BenchmarkError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    background_tasks.add_task(execute_benchmark, benchmark.id, grid)
    return BenchmarkResponse.model_validate(benchmark)


@router.get("/{model_name}/benchmarks", response_model=BenchmarkListResponse)
def list_benchmarks_endpoint(
    model_name: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> BenchmarkListResponse:
    """List a model's recent benchmarks, newest first."""
    return BenchmarkListResponse(
        benchmarks=[
            BenchmarkSummary(
                id=b.id,
                model_name=b.model_name,
                model_version=b.model_version,
                status=b.status,
                cells=len(b.results or []),
                started_at=b.started_at,
                completed_at=b.completed_at,
            )
            for b in list_benchmarks(db, model_name, limit)
        ]
    )


@router.get("/{model_name}/recommendation", response_model=ServingRecommendationResponse)
def recommend_serving_config(
    model_name: str,
    slo_p99_ms: float = Query(..., gt=0, description="p99 latency target in milliseconds"),
    db: Session = Depends(get_db),
) -> ServingRecommendationResponse:
    """Highest-throughput backend/batching/concurrency meeting a p99 SLO, from the latest benchmark."""
    benchmark = latest_completed_benchmark(db, model_name)
    if benchmark is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No completed benchmark for '{model_name}'; POST /api/serve/{model_name}/benchmarks first",
        )
    recommendation = recommend_config(benchmark.results or [], slo_p99_ms)
    return ServingRecommendationResponse(
        benchmark_id=benchmark.id,
        model_name=benchmark.model_name,
        model_version=benchmark.model_version,
        **recommendation,
    )
