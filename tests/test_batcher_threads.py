"""Tests that the micro-batcher runs its thread initializer on the inference thread."""

import threading

import numpy as np
import pytest

from forge.api.services.batcher import BatchConfig, MicroBatcher


@pytest.mark.asyncio
async def test_thread_initializer_runs_on_inference_thread() -> None:
    initialized: list[str] = []
    predicted_on: list[str] = []

    def initializer() -> None:
        initialized.append(threading.current_thread().name)

    def predict(batch: np.ndarray) -> np.ndarray:
        predicted_on.append(threading.current_thread().name)
        return batch.sum(axis=1)

    batcher = MicroBatcher(
        predict, BatchConfig(max_batch_size=8, max_wait_ms=1), label="t", thread_initializer=initializer
    )
    await batcher.submit(np.ones((2, 3), dtype=np.float32))
    await batcher.submit(np.ones((1, 3), dtype=np.float32))
    await batcher.close()

    assert len(initialized) == 1
    assert set(predicted_on) == {initialized[0]}
    assert initialized[0].startswith("batcher-t")
