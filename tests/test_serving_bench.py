"""Tests for the serving load-test harness (windows, cells, recommendations)."""

import time

import numpy as np
import pytest

from forge.api.services.batcher import BatchConfig
from forge.api.services.serving_bench import (
    BenchmarkGrid,
    latency_summary,
    make_windows,
    recommend_config,
    run_cell,
)


def _cell(backend: str, p99: float, rps: float, errors: int = 0, batch: int = 1) -> dict:
    return {
        "backend": backend,
        "max_batch_size": batch,
        "max_wait_ms": 0.0,
        "concurrency": 16,
        "p99_ms": p99,
        "throughput_rps": rps,
        "errors": errors,
    }


class TestMakeWindows:
    """Turning feature rows into model inputs."""

    def test_sequence_windows_slide_one_step(self) -> None:
        features = np.arange(20, dtype=np.float32).reshape(10, 2)
        windows = make_windows(features, (4, 2))
        assert windows.shape == (7, 4, 2)
        np.testing.assert_array_equal(windows[0], features[0:4])
        np.testing.assert_array_equal(windows[-1], features[6:10])

    def test_tabular_rows_pass_through(self) -> None:
        features = np.ones((5, 3), dtype=np.float32)
        assert make_windows(features, (3,)).shape == (5, 3)

    def test_too_few_rows_for_a_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window"):
            make_windows(np.ones((3, 2), dtype=np.float32), (4, 2))

    def test_feature_width_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="features"):
            make_windows(np.ones((10, 3), dtype=np.float32), (4, 2))


class TestLatencySummary:
    """Percentile summaries."""

    def test_percentiles(self) -> None:
        summary = latency_summary(list(range(1, 101)))
        assert summary["p50_ms"] == pytest.approx(50.5)
        assert summary["p99_ms"] == pytest.approx(99.01)
        assert summary["mean_ms"] == pytest.approx(50.5)

    def test_empty_is_none(self) -> None:
        assert latency_summary([])["p99_ms"] is None


class TestBenchmarkGrid:
    """Grid validation."""

    def test_rejects_empty_concurrency(self) -> None:
        with pytest.raises(ValueError):
            BenchmarkGrid(backends=None, batch_configs=(BatchConfig(1, 0),), concurrency=(), requests_per_cell=50)

    def test_rejects_too_many_cells(self) -> None:
        configs = tuple(BatchConfig(size, 1) for size in range(1, 30))
        with pytest.raises(ValueError, match="cells"):
            BenchmarkGrid(backends=("a", "b", "c"), batch_configs=configs, concurrency=(1, 2, 4), requests_per_cell=50)

    def test_cell_count(self) -> None:
        grid = BenchmarkGrid(
            backends=("pytorch", "onnxruntime"),
            batch_configs=(BatchConfig(1, 0), BatchConfig(32, 2)),
            concurrency=(1, 16),
            requests_per_cell=50,
        )
        assert grid.cell_count(available_backends=3) == 8
        assert BenchmarkGrid(None, grid.batch_configs, grid.concurrency, 50).cell_count(3) == 12


@pytest.mark.asyncio
class TestRunCell:
    """Closed-loop load generation for one grid cell."""

    async def test_counts_requests_and_reports_latency(self) -> None:
        inputs = np.ones((8, 3), dtype=np.float32)
        cell = await run_cell(
            lambda batch: batch.sum(axis=1), inputs, "fake", BatchConfig(1, 0), concurrency=4, requests=40
        )
        assert cell["requests"] == 40
        assert cell["errors"] == 0
        assert cell["p99_ms"] >= cell["p50_ms"] > 0
        assert cell["throughput_rps"] > 0
        assert cell["mean_batch_rows"] == pytest.approx(1.0)

    async def test_batching_merges_concurrent_requests(self) -> None:
        def slow_model(batch: np.ndarray) -> np.ndarray:
            time.sleep(0.005)
            return batch.sum(axis=1)

        cell = await run_cell(
            slow_model, np.ones((8, 3), dtype=np.float32), "fake", BatchConfig(16, 2), concurrency=16, requests=64
        )
        assert cell["mean_batch_rows"] > 2
        assert cell["errors"] == 0

    async def test_errors_are_counted_not_raised(self) -> None:
        calls = {"n": 0}

        def flaky(batch: np.ndarray) -> np.ndarray:
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("boom")
            return batch.sum(axis=1)

        cell = await run_cell(flaky, np.ones((4, 3), dtype=np.float32), "fake", BatchConfig(1, 0), 1, 20)
        assert cell["requests"] == 20
        assert cell["errors"] > 0
        assert 0 < cell["error_rate"] < 1


class TestRecommendConfig:
    """Choosing the highest-throughput config that meets a p99 SLO."""

    def test_picks_highest_throughput_under_slo(self) -> None:
        cells = [
            _cell("pytorch", p99=30.0, rps=4000),
            _cell("onnxruntime", p99=12.0, rps=2500, batch=32),
            _cell("onnxruntime-int8", p99=9.0, rps=2000),
        ]
        rec = recommend_config(cells, slo_p99_ms=15.0)
        assert rec["meets_slo"] is True
        assert rec["best"]["backend"] == "onnxruntime"
        assert [c["backend"] for c in rec["candidates"]] == ["onnxruntime", "onnxruntime-int8"]

    def test_cells_with_errors_are_excluded(self) -> None:
        cells = [_cell("a", p99=5.0, rps=9000, errors=3), _cell("b", p99=6.0, rps=1000)]
        assert recommend_config(cells, 10.0)["best"]["backend"] == "b"

    def test_no_cell_meets_slo_returns_lowest_p99(self) -> None:
        cells = [_cell("a", p99=40.0, rps=100), _cell("b", p99=25.0, rps=50)]
        rec = recommend_config(cells, 10.0)
        assert rec["meets_slo"] is False
        assert rec["best"] is None
        assert rec["lowest_p99"]["backend"] == "b"

    def test_rejects_non_positive_slo(self) -> None:
        with pytest.raises(ValueError):
            recommend_config([], 0)
