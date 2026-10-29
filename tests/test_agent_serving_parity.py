"""Tests that serving recommendations carry the ONNX parity (accuracy) check."""

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from forge.api.agent.tools import summarize_serving_benchmark

PARITY = {
    "onnxruntime": {"backend": "onnxruntime", "passed": True, "max_abs_diff": 2.7e-7, "label_agreement": 1.0},
    "onnxruntime-int8": {"backend": "onnxruntime-int8", "passed": True, "max_abs_diff": 5.4e-3, "label_agreement": 0.996},
}


def _cell(backend: str, batch: int, p99: float, rps: float) -> dict:
    return {
        "backend": backend, "max_batch_size": batch, "max_wait_ms": 2.0, "concurrency": 64,
        "errors": 0, "p50_ms": p99 / 2, "p99_ms": p99, "throughput_rps": rps,
    }


def _benchmark(params: dict) -> SimpleNamespace:
    cells = [_cell("pytorch", 1, 105.0, 700), _cell("onnxruntime-int8", 32, 3.8, 20000)]
    return SimpleNamespace(
        id=uuid4(), model_name="m", model_version=3, completed_at=datetime(2026, 10, 29, tzinfo=UTC),
        runtime={}, params=params, results=cells,
    )


def test_parity_from_benchmark_params_for_recommended_backend() -> None:
    summary = summarize_serving_benchmark(_benchmark({"parity": PARITY}), slo_p99_ms=10.0)
    assert summary["recommended_parity"]["label_agreement"] == 0.996
    assert summary["recommended_parity"]["passed"] is True


def test_parity_override_used_when_benchmark_has_none() -> None:
    summary = summarize_serving_benchmark(_benchmark({}), slo_p99_ms=10.0, parity=PARITY)
    assert summary["recommended_parity"]["max_abs_diff"] == 5.4e-3


def test_native_recommendation_needs_no_parity() -> None:
    benchmark = _benchmark({})
    benchmark.results = [_cell("pytorch", 32, 4.0, 5000)]
    summary = summarize_serving_benchmark(benchmark, slo_p99_ms=10.0)
    assert summary["recommended_parity"] == {"backend": "pytorch", "native": True}


def test_missing_parity_is_reported_as_unknown() -> None:
    summary = summarize_serving_benchmark(_benchmark({}), slo_p99_ms=10.0)
    assert summary["recommended_parity"] is None
