"""Serving load-test harness — latency/throughput across backends, batching and concurrency.

A benchmark replays real feature windows from the served model's own test
split through each inference backend, under each micro-batching policy, at
each concurrency level, using a closed-loop load generator (every worker
sends its next request as soon as the previous one returns). Latency is
measured server-side around the predict call, including batching queue
wait, and excludes network and HTTP/JSON overhead.

Benchmarks use private batchers, so they never change the live model's
batching policy, but they do share the API's CPUs with live traffic.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import numpy as np
from sqlalchemy.orm import Session

from forge.api.models.database import (
    Dataset,
    Experiment,
    Run,
    ServingBenchmark,
    SessionLocal,
)
from forge.api.services.batcher import BatchConfig, MicroBatcher
from forge.api.services.data_ingestion import load_dataset_frame
from forge.api.services.model_server import (
    ServableModel,
    get_production_servable,
    predict_proba,
)
from forge.api.services.runtime import init_worker_thread, runtime_info
from forge.api.services.training import EXCLUDE_COLS, create_target, time_series_split

logger = logging.getLogger(__name__)

MAX_CELLS = 96
MAX_INPUT_SAMPLES = 512

DEFAULT_BATCH_CONFIGS = (
    BatchConfig(max_batch_size=1, max_wait_ms=0),
    BatchConfig(max_batch_size=8, max_wait_ms=1),
    BatchConfig(max_batch_size=32, max_wait_ms=2),
    BatchConfig(max_batch_size=64, max_wait_ms=5),
)
DEFAULT_CONCURRENCY = (1, 4, 16, 64)
DEFAULT_REQUESTS_PER_CELL = 200


class BenchmarkError(Exception):
    """A benchmark can't be started or its inputs can't be built."""


@dataclass(frozen=True)
class BenchmarkGrid:
    """The sweep: backends (None = every servable backend) x batch policies x concurrency."""

    backends: tuple[str, ...] | None
    batch_configs: tuple[BatchConfig, ...]
    concurrency: tuple[int, ...]
    requests_per_cell: int

    def __post_init__(self) -> None:
        if not self.batch_configs or not self.concurrency:
            raise ValueError("A benchmark needs at least one batch config and one concurrency level")
        if any(level < 1 or level > 256 for level in self.concurrency):
            raise ValueError("Concurrency levels must be between 1 and 256")
        if not 20 <= self.requests_per_cell <= 5000:
            raise ValueError("requests_per_cell must be between 20 and 5000")
        if self.backends is not None:
            self.check_size(len(self.backends))

    def cell_count(self, available_backends: int) -> int:
        """Number of grid cells given how many backends will be swept."""
        backend_count = len(self.backends) if self.backends is not None else available_backends
        return backend_count * len(self.batch_configs) * len(self.concurrency)

    def check_size(self, available_backends: int) -> None:
        """Raise if the sweep would run more than MAX_CELLS cells."""
        cells = self.cell_count(available_backends)
        if cells > MAX_CELLS:
            raise ValueError(f"Grid has {cells} cells; the limit is {MAX_CELLS}")

    def to_params(self) -> dict:
        """JSON-serializable description stored with the benchmark."""
        return {
            "backends": list(self.backends) if self.backends is not None else None,
            "batch_configs": [
                {"max_batch_size": c.max_batch_size, "max_wait_ms": c.max_wait_ms}
                for c in self.batch_configs
            ],
            "concurrency": list(self.concurrency),
            "requests_per_cell": self.requests_per_cell,
            "rows_per_request": 1,
            "latency_scope": "server-side: queue wait + forward pass; excludes network and HTTP",
        }


DEFAULT_GRID = BenchmarkGrid(None, DEFAULT_BATCH_CONFIGS, DEFAULT_CONCURRENCY, DEFAULT_REQUESTS_PER_CELL)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def make_windows(features: np.ndarray, input_shape: tuple[int, ...]) -> np.ndarray:
    """Turn time-ordered feature rows into model inputs of ``input_shape``.

    Sequence models get every one-step-sliding window; tabular models get the rows.

    Raises:
        ValueError: If the feature width doesn't match or there are too few rows.
    """
    if features.shape[1] != input_shape[-1]:
        raise ValueError(f"Data has {features.shape[1]} features, model expects {input_shape[-1]}")
    if len(input_shape) == 1:
        return features
    window = input_shape[0]
    if len(features) < window:
        raise ValueError(f"Need at least {window} rows for one window, got {len(features)}")
    return np.lib.stride_tricks.sliding_window_view(features, window, axis=0).transpose(0, 2, 1).copy()


def load_benchmark_inputs(
    db: Session, servable: ServableModel, max_samples: int = MAX_INPUT_SAMPLES
) -> np.ndarray:
    """Build benchmark inputs from the test split of the data the model was trained on.

    Raises:
        BenchmarkError: If the run, experiment or dataset can't be found or loaded.
    """
    run = db.query(Run).filter(Run.id == servable.run_id).first()
    experiment = run and db.query(Experiment).filter(Experiment.id == run.experiment_id).first()
    dataset = experiment and db.query(Dataset).filter(Dataset.id == experiment.dataset_id).first()
    if dataset is None:
        raise BenchmarkError(f"Can't find the dataset behind {servable.name} v{servable.version}")

    if run.feature_set_id is not None:
        from forge.api.services.feature_store import get_features

        frame = get_features(db, run.feature_set_id, dataset.id)
    else:
        frame = load_dataset_frame(dataset)
    if frame is None:
        raise BenchmarkError(f"Dataset {dataset.id} parquet is unavailable locally and in S3")

    _, _, test_frame = time_series_split(create_target(frame))
    feature_names = servable.feature_names or [c for c in test_frame.columns if c not in EXCLUDE_COLS]
    missing = [name for name in feature_names if name not in test_frame.columns]
    if missing:
        raise BenchmarkError(f"Dataset is missing model features: {', '.join(missing)}")

    features = np.nan_to_num(
        test_frame[feature_names].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    try:
        windows = make_windows(features, servable.input_shape)
    except ValueError as exc:
        raise BenchmarkError(str(exc)) from exc
    if len(windows) > max_samples:
        picks = np.random.default_rng(0).choice(len(windows), size=max_samples, replace=False)
        windows = windows[np.sort(picks)]
    return windows.astype(np.float32)


# ---------------------------------------------------------------------------
# Load generation
# ---------------------------------------------------------------------------


def latency_summary(latencies_ms: list[float]) -> dict:
    """p50/p95/p99/mean of a latency sample, in ms (None when empty)."""
    if not latencies_ms:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None}
    sample = np.asarray(latencies_ms)
    p50, p95, p99 = np.percentile(sample, [50, 95, 99])
    return {
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "mean_ms": float(sample.mean()),
    }


async def run_cell(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    backend: str,
    config: BatchConfig,
    concurrency: int,
    requests: int,
) -> dict:
    """Drive one grid cell with a closed-loop generator and summarize it.

    Each request is a single input row/window. A warm-up round of
    ``concurrency`` requests runs first and isn't measured.
    """
    rows_per_call: list[int] = []

    def counting_predict(batch: np.ndarray) -> np.ndarray:
        rows_per_call.append(len(batch))
        return predict_fn(batch)

    batcher = (
        MicroBatcher(
            counting_predict, config, label=f"bench/{backend}", thread_initializer=init_worker_thread
        )
        if config.enabled
        else None
    )

    async def call(index: int) -> None:
        request = inputs[index % len(inputs)][np.newaxis]
        if batcher is not None:
            await batcher.submit(request)
        else:
            await asyncio.to_thread(counting_predict, request)

    latencies_ms: list[float] = []
    errors = 0
    next_index = 0

    async def worker(measured: bool, limit: int) -> None:
        nonlocal next_index, errors
        while next_index < limit:
            index = next_index
            next_index += 1
            start = time.perf_counter()
            try:
                await call(index)
            except Exception:  # a failed request is a data point, not a harness failure
                logger.debug("Benchmark request failed (%s)", backend, exc_info=True)
                if measured:
                    errors += 1
                continue
            if measured:
                latencies_ms.append((time.perf_counter() - start) * 1000.0)

    try:
        warmup = min(concurrency, requests)
        await asyncio.gather(*(worker(False, warmup) for _ in range(concurrency)))
        rows_per_call.clear()
        next_index = 0
        started = time.perf_counter()
        await asyncio.gather(*(worker(True, requests) for _ in range(concurrency)))
        duration = time.perf_counter() - started
    finally:
        if batcher is not None:
            await batcher.close()

    completed = len(latencies_ms)
    return {
        "backend": backend,
        "max_batch_size": config.max_batch_size,
        "max_wait_ms": config.max_wait_ms,
        "concurrency": concurrency,
        "requests": requests,
        "errors": errors,
        "error_rate": errors / requests,
        **latency_summary(latencies_ms),
        "throughput_rps": completed / duration if duration > 0 else 0.0,
        "mean_batch_rows": float(np.mean(rows_per_call)) if rows_per_call else 0.0,
        "duration_s": duration,
    }


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def recommend_config(cells: list[dict], slo_p99_ms: float, top: int = 5) -> dict:
    """Pick the highest-throughput error-free cell whose p99 meets the SLO.

    Returns ``best`` (None if nothing meets the SLO), up to ``top`` ranked
    ``candidates``, and ``lowest_p99`` so a miss still says how close it got.

    Raises:
        ValueError: If the SLO isn't positive.
    """
    if slo_p99_ms <= 0:
        raise ValueError("slo_p99_ms must be positive")
    clean = [c for c in cells if c.get("errors", 0) == 0 and c.get("p99_ms") is not None]
    candidates = sorted(
        (c for c in clean if c["p99_ms"] <= slo_p99_ms),
        key=lambda c: (-c["throughput_rps"], c["p99_ms"]),
    )
    return {
        "slo_p99_ms": slo_p99_ms,
        "meets_slo": bool(candidates),
        "best": candidates[0] if candidates else None,
        "candidates": candidates[:top],
        "lowest_p99": min(clean, key=lambda c: c["p99_ms"]) if clean else None,
        "cells_considered": len(clean),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_active_benchmark: UUID | None = None


def start_benchmark(db: Session, model_name: str, version: int, grid: BenchmarkGrid) -> ServingBenchmark:
    """Create the benchmark row; only one benchmark runs at a time per process.

    Rows left ``running`` by a previous process are marked failed first.

    Raises:
        BenchmarkError: If another benchmark is already running.
    """
    global _active_benchmark
    if _active_benchmark is not None:
        raise BenchmarkError(f"Benchmark {_active_benchmark} is still running")

    interrupted = db.query(ServingBenchmark).filter(ServingBenchmark.status == "running").all()
    for stale in interrupted:
        stale.status = "failed"
        stale.error = "Interrupted by an API restart"
        stale.completed_at = datetime.now(UTC)

    benchmark = ServingBenchmark(
        model_name=model_name,
        model_version=version,
        status="running",
        params=grid.to_params(),
        runtime=runtime_info(),
        results=[],
    )
    db.add(benchmark)
    db.commit()
    db.refresh(benchmark)
    _active_benchmark = benchmark.id
    return benchmark


async def execute_benchmark(benchmark_id: UUID, grid: BenchmarkGrid) -> None:
    """Run every grid cell for a started benchmark, saving results as cells finish."""
    global _active_benchmark
    db = SessionLocal()
    benchmark = db.query(ServingBenchmark).filter(ServingBenchmark.id == benchmark_id).first()
    try:
        servable = await asyncio.to_thread(get_production_servable, db, benchmark.model_name)
        inputs = await asyncio.to_thread(load_benchmark_inputs, db, servable)
        backends = grid.backends if grid.backends is not None else tuple(servable.backends)
        unavailable = [b for b in backends if b not in servable.backends]
        if unavailable:
            raise BenchmarkError(f"Backends not available: {', '.join(unavailable)}")
        grid.check_size(len(backends))

        results: list[dict] = []
        for backend in backends:
            for config in grid.batch_configs:
                for concurrency in grid.concurrency:
                    cell = await run_cell(
                        lambda batch, name=backend: predict_proba(servable, batch, name),
                        inputs,
                        backend,
                        config,
                        concurrency,
                        grid.requests_per_cell,
                    )
                    results.append(cell)
                    benchmark.results = list(results)
                    db.commit()

        benchmark.model_version = servable.version
        benchmark.status = "completed"
        benchmark.params = {**grid.to_params(), "input_samples": len(inputs)}
    except Exception as exc:  # any failure must leave the row failed, not stuck running
        logger.exception("Benchmark %s failed", benchmark_id)
        db.rollback()
        benchmark.status = "failed"
        benchmark.error = str(exc)
    finally:
        benchmark.completed_at = datetime.now(UTC)
        db.commit()
        db.close()
        _active_benchmark = None


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def list_benchmarks(db: Session, model_name: str, limit: int = 20) -> list[ServingBenchmark]:
    """Most recent benchmarks for a model, newest first."""
    return (
        db.query(ServingBenchmark)
        .filter(ServingBenchmark.model_name == model_name)
        .order_by(ServingBenchmark.started_at.desc())
        .limit(limit)
        .all()
    )


def get_benchmark(db: Session, benchmark_id: UUID) -> ServingBenchmark | None:
    """Fetch one benchmark by ID."""
    return db.query(ServingBenchmark).filter(ServingBenchmark.id == benchmark_id).first()


def latest_completed_benchmark(db: Session, model_name: str) -> ServingBenchmark | None:
    """The newest completed benchmark for a model, if any."""
    return (
        db.query(ServingBenchmark)
        .filter(ServingBenchmark.model_name == model_name, ServingBenchmark.status == "completed")
        .order_by(ServingBenchmark.completed_at.desc())
        .first()
    )
