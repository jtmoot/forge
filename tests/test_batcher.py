"""Tests for the dynamic micro-batcher."""

import asyncio
import threading
import time

import numpy as np
import pytest

from forge.api.services.batcher import BatchConfig, BatcherPool, MicroBatcher


class RecordingModel:
    """Fake predict_fn: returns row sums and records each batch it receives."""

    def __init__(self, delay_seconds: float = 0.0, fail_on_call: int | None = None) -> None:
        self.batch_sizes: list[int] = []
        self.delay_seconds = delay_seconds
        self.fail_on_call = fail_on_call

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        self.batch_sizes.append(len(batch))
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.fail_on_call is not None and len(self.batch_sizes) == self.fail_on_call:
            raise RuntimeError("model exploded")
        return batch.reshape(len(batch), -1).sum(axis=1)


def _rows(start: int, count: int, width: int = 3) -> np.ndarray:
    return np.arange(start, start + count * width, dtype=np.float32).reshape(count, width)


class TestBatchConfig:
    """Config validation and the on/off switch."""

    def test_batch_size_one_disables_batching(self) -> None:
        assert BatchConfig(max_batch_size=1, max_wait_ms=5).enabled is False
        assert BatchConfig(max_batch_size=8, max_wait_ms=5).enabled is True

    @pytest.mark.parametrize(("size", "wait"), [(0, 1.0), (4, -1.0), (4, 5000.0)])
    def test_rejects_invalid_values(self, size: int, wait: float) -> None:
        with pytest.raises(ValueError):
            BatchConfig(max_batch_size=size, max_wait_ms=wait)


@pytest.mark.asyncio
class TestMicroBatcher:
    """Batch formation, result routing, and failure handling."""

    async def test_concurrent_requests_get_their_own_rows(self) -> None:
        model = RecordingModel()
        batcher = MicroBatcher(model, BatchConfig(max_batch_size=64, max_wait_ms=20))
        requests = [_rows(i * 100, count=1 + i % 3) for i in range(20)]

        results = await asyncio.gather(*(batcher.submit(r) for r in requests))

        for request, result in zip(requests, results, strict=True):
            np.testing.assert_allclose(result, request.sum(axis=1))
        assert sum(model.batch_sizes) == sum(len(r) for r in requests)
        assert len(model.batch_sizes) < len(requests)
        await batcher.close()

    async def test_respects_max_batch_size(self) -> None:
        model = RecordingModel()
        batcher = MicroBatcher(model, BatchConfig(max_batch_size=4, max_wait_ms=20))

        await asyncio.gather(*(batcher.submit(_rows(i, 1)) for i in range(10)))

        assert max(model.batch_sizes) <= 4
        assert sum(model.batch_sizes) == 10
        await batcher.close()

    async def test_lone_request_flushes_after_max_wait(self) -> None:
        model = RecordingModel()
        batcher = MicroBatcher(model, BatchConfig(max_batch_size=64, max_wait_ms=30))

        start = time.perf_counter()
        result = await asyncio.wait_for(batcher.submit(_rows(0, 1)), timeout=1.0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        np.testing.assert_allclose(result, [3.0])
        assert model.batch_sizes == [1]
        assert 25 <= elapsed_ms < 500
        await batcher.close()

    async def test_full_batch_flushes_without_waiting(self) -> None:
        model = RecordingModel()
        batcher = MicroBatcher(model, BatchConfig(max_batch_size=4, max_wait_ms=1000))

        start = time.perf_counter()
        await asyncio.gather(*(batcher.submit(_rows(i, 1)) for i in range(4)))

        assert (time.perf_counter() - start) < 0.5
        assert model.batch_sizes == [4]
        await batcher.close()

    async def test_oversized_request_runs_alone_unsplit(self) -> None:
        model = RecordingModel()
        batcher = MicroBatcher(model, BatchConfig(max_batch_size=4, max_wait_ms=10))

        big, small = await asyncio.gather(batcher.submit(_rows(0, 10)), batcher.submit(_rows(500, 1)))

        assert len(big) == 10 and len(small) == 1
        assert 10 in model.batch_sizes
        await batcher.close()

    async def test_error_rejects_whole_batch_and_worker_survives(self) -> None:
        model = RecordingModel(fail_on_call=1)
        batcher = MicroBatcher(model, BatchConfig(max_batch_size=64, max_wait_ms=20))

        outcomes = await asyncio.gather(
            *(batcher.submit(_rows(i, 1)) for i in range(5)), return_exceptions=True
        )
        assert all(isinstance(o, RuntimeError) for o in outcomes)

        recovered = await asyncio.wait_for(batcher.submit(_rows(0, 1)), timeout=1.0)
        np.testing.assert_allclose(recovered, [3.0])
        await batcher.close()

    async def test_cancelled_caller_does_not_break_batch(self) -> None:
        model = RecordingModel(delay_seconds=0.05)
        batcher = MicroBatcher(model, BatchConfig(max_batch_size=64, max_wait_ms=20))

        doomed = asyncio.create_task(batcher.submit(_rows(0, 1)))
        survivor = asyncio.create_task(batcher.submit(_rows(100, 1)))
        await asyncio.sleep(0.03)
        doomed.cancel()

        result = await asyncio.wait_for(survivor, timeout=1.0)
        np.testing.assert_allclose(result, _rows(100, 1).sum(axis=1))
        await batcher.close()

    async def test_retire_drains_queued_requests_then_rejects_new_ones(self) -> None:
        model = RecordingModel(delay_seconds=0.02)
        batcher = MicroBatcher(model, BatchConfig(max_batch_size=2, max_wait_ms=5))

        in_flight = [asyncio.create_task(batcher.submit(_rows(i, 1))) for i in range(6)]
        await asyncio.sleep(0)
        batcher.retire()

        results = await asyncio.wait_for(asyncio.gather(*in_flight), timeout=2.0)
        assert len(results) == 6
        with pytest.raises(RuntimeError, match="retired"):
            await batcher.submit(_rows(0, 1))

    async def test_retire_is_safe_from_another_thread(self) -> None:
        model = RecordingModel()
        batcher = MicroBatcher(model, BatchConfig(max_batch_size=8, max_wait_ms=5))
        await batcher.submit(_rows(0, 1))

        thread = threading.Thread(target=batcher.retire)
        thread.start()
        thread.join()
        await asyncio.sleep(0.05)

        assert batcher.retired is True


@pytest.mark.asyncio
class TestBatcherPool:
    """One batcher per key, retirement by model name."""

    async def test_reuses_batcher_per_key(self) -> None:
        pool = BatcherPool()
        config = BatchConfig(max_batch_size=8, max_wait_ms=5)
        first = pool.get_or_create(("m", 1, "pytorch"), lambda: MicroBatcher(RecordingModel(), config))
        second = pool.get_or_create(("m", 1, "pytorch"), lambda: MicroBatcher(RecordingModel(), config))
        assert first is second
        pool.retire_model("m")

    async def test_retire_model_only_affects_that_model(self) -> None:
        pool = BatcherPool()
        config = BatchConfig(max_batch_size=8, max_wait_ms=5)
        model_a = pool.get_or_create(("a", 1, "pytorch"), lambda: MicroBatcher(RecordingModel(), config))
        model_b = pool.get_or_create(("b", 1, "pytorch"), lambda: MicroBatcher(RecordingModel(), config))

        assert pool.retire_model("a") == 1
        await asyncio.sleep(0.01)

        assert model_a.retired and not model_b.retired
        fresh = pool.get_or_create(("a", 1, "pytorch"), lambda: MicroBatcher(RecordingModel(), config))
        assert fresh is not model_a
        pool.retire_model("b")
        pool.retire_model("a")
