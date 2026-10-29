"""Tests for the find_serving_config agent tool's summary logic."""

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from forge.api.agent.tools import summarize_serving_benchmark


def _cell(backend: str, batch: int, conc: int, p99: float, rps: float) -> dict:
    return {
        "backend": backend,
        "max_batch_size": batch,
        "max_wait_ms": 2.0 if batch > 1 else 0.0,
        "concurrency": conc,
        "requests": 200,
        "errors": 0,
        "p50_ms": p99 / 2,
        "p95_ms": p99 * 0.9,
        "p99_ms": p99,
        "throughput_rps": rps,
        "mean_batch_rows": 4.0 if batch > 1 else 1.0,
    }


def _benchmark(cells: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        model_name="spy_direction_predictor",
        model_version=3,
        completed_at=datetime(2026, 10, 29, tzinfo=UTC),
        runtime={"cpu_limit": 8.0, "torch_threads": 8},
        params={"latency_scope": "server-side"},
        results=cells,
    )


def test_recommends_best_and_compares_to_naive_baseline() -> None:
    cells = [
        _cell("pytorch", 1, 16, p99=45.0, rps=900),
        _cell("onnxruntime", 32, 16, p99=12.0, rps=2700),
        _cell("onnxruntime-int8", 32, 16, p99=9.0, rps=2400),
    ]
    summary = summarize_serving_benchmark(_benchmark(cells), slo_p99_ms=15.0)

    assert summary["meets_slo"] is True
    assert summary["recommended"]["backend"] == "onnxruntime"
    assert summary["baseline"]["backend"] == "pytorch"
    assert summary["throughput_vs_baseline"] == 3.0
    assert summary["model_version"] == 3
    assert summary["runtime"]["cpu_limit"] == 8.0


def test_miss_reports_closest_config() -> None:
    cells = [_cell("pytorch", 1, 64, p99=80.0, rps=1500), _cell("onnxruntime", 32, 64, p99=30.0, rps=3000)]
    summary = summarize_serving_benchmark(_benchmark(cells), slo_p99_ms=10.0)

    assert summary["meets_slo"] is False
    assert summary["recommended"] is None
    assert summary["closest"]["backend"] == "onnxruntime"


def test_baseline_missing_when_native_unbatched_not_run() -> None:
    cells = [_cell("onnxruntime", 32, 16, p99=12.0, rps=2700)]
    summary = summarize_serving_benchmark(_benchmark(cells), slo_p99_ms=15.0)
    assert summary["baseline"] is None
    assert summary["throughput_vs_baseline"] is None
