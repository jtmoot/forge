"""Dynamic micro-batching for model serving.

Concurrent prediction requests for the same model are queued and flushed as
one forward pass when either ``max_batch_size`` rows are waiting or the
oldest request has waited ``max_wait_ms``. Each caller gets back exactly its
own rows. Larger batches raise throughput; the wait window is what they cost
in tail latency, and the Serving page measures that trade-off.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from forge.api.services.metrics import (
    SERVING_BATCH_SIZE,
    SERVING_QUEUE_DEPTH,
    SERVING_QUEUE_WAIT_SECONDS,
)

logger = logging.getLogger(__name__)

MAX_WAIT_LIMIT_MS = 1000.0


@dataclass(frozen=True)
class BatchConfig:
    """Flush policy for a micro-batcher. ``max_batch_size=1`` turns batching off."""

    max_batch_size: int
    max_wait_ms: float

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if not 0 <= self.max_wait_ms <= MAX_WAIT_LIMIT_MS:
            raise ValueError(f"max_wait_ms must be between 0 and {MAX_WAIT_LIMIT_MS}")

    @property
    def enabled(self) -> bool:
        """Whether requests should go through a batcher at all."""
        return self.max_batch_size > 1


@dataclass
class _QueuedRequest:
    """One caller's rows and the future its result is delivered on."""

    inputs: np.ndarray
    future: asyncio.Future
    enqueued_at: float


_STOP = object()


class MicroBatcher:
    """Queue requests for one model/backend and run them in shared forward passes.

    Forward passes run one at a time on a dedicated thread, so the event loop
    stays free to accept requests while a batch is computing — those requests
    form the next batch.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        config: BatchConfig,
        label: str = "default",
        thread_initializer: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self.label = label
        self.retired = False
        self._predict_fn = predict_fn
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"batcher-{label}", initializer=thread_initializer
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self._carry: _QueuedRequest | None = None

    async def submit(self, inputs: np.ndarray) -> np.ndarray:
        """Queue ``inputs`` (a batch of rows) and wait for their predictions."""
        if self.retired:
            raise RuntimeError(f"Batcher {self.label} is retired")
        self._ensure_worker()
        future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_QueuedRequest(inputs, future, time.perf_counter()))
        SERVING_QUEUE_DEPTH.labels(model=self.label).set(self._queue.qsize())
        return await future

    def retire(self) -> None:
        """Stop accepting requests; already-queued requests still complete.

        Safe to call from any thread (the registry calls it on promotion).
        """
        if self.retired:
            return
        self.retired = True
        if self._loop is None or self._worker is None or self._worker.done():
            self._executor.shutdown(wait=False)
            return
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, _STOP)
        except RuntimeError:
            # Event loop already closed; nothing is left to drain.
            self._executor.shutdown(wait=False)

    async def close(self) -> None:
        """Retire the batcher and wait for queued requests to finish."""
        self.retire()
        if self._worker is not None and not self._worker.done():
            await asyncio.wait_for(asyncio.shield(self._worker), timeout=5.0)

    def _ensure_worker(self) -> None:
        """Start the batching loop on the current event loop if it isn't running."""
        loop = asyncio.get_running_loop()
        if self._worker is not None and not self._worker.done() and self._loop is loop:
            return
        self._loop = loop
        self._queue = asyncio.Queue()
        self._carry = None
        self._worker = loop.create_task(self._run(), name=f"batcher-{self.label}")

    async def _run(self) -> None:
        """Collect requests into batches until retired."""
        loop = asyncio.get_running_loop()
        stopping = False
        while not stopping:
            first = self._carry if self._carry is not None else await self._queue.get()
            self._carry = None
            if first is _STOP:
                break

            batch = [first]
            rows = len(first.inputs)
            deadline = loop.time() + self.config.max_wait_ms / 1000.0
            while rows < self.config.max_batch_size:
                if not self._queue.empty():
                    queued = self._queue.get_nowait()
                else:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        queued = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    except TimeoutError:
                        break
                if queued is _STOP:
                    stopping = True
                    break
                if rows + len(queued.inputs) > self.config.max_batch_size:
                    self._carry = queued
                    break
                batch.append(queued)
                rows += len(queued.inputs)

            SERVING_QUEUE_DEPTH.labels(model=self.label).set(self._queue.qsize())
            await self._execute(batch, loop)

        if self._carry is not None and self._carry is not _STOP:
            await self._execute([self._carry], loop)
            self._carry = None
        self._executor.shutdown(wait=False)

    async def _execute(self, batch: list[_QueuedRequest], loop: asyncio.AbstractEventLoop) -> None:
        """Run one forward pass for the batch and route each caller its rows."""
        live = [request for request in batch if not request.future.done()]
        if not live:
            return

        started = time.perf_counter()
        for request in live:
            SERVING_QUEUE_WAIT_SECONDS.labels(model=self.label).observe(started - request.enqueued_at)

        try:
            stacked = np.concatenate([request.inputs for request in live])
            SERVING_BATCH_SIZE.labels(model=self.label).observe(len(stacked))
            outputs = await loop.run_in_executor(self._executor, self._predict_fn, stacked)
        except Exception as exc:  # any model failure must reach every caller, not hang them
            logger.warning(
                "Batch of %d requests failed for %s", len(live), self.label, exc_info=True
            )
            for request in live:
                if not request.future.done():
                    request.future.set_exception(exc)
            return

        offset = 0
        for request in live:
            count = len(request.inputs)
            if not request.future.done():
                request.future.set_result(outputs[offset : offset + count])
            offset += count


class BatcherPool:
    """One MicroBatcher per key (model name, version, backend), retired by model name."""

    def __init__(self) -> None:
        self._batchers: dict[Hashable, MicroBatcher] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self, key: tuple[str, int, str], factory: Callable[[], MicroBatcher]
    ) -> MicroBatcher:
        """Return the live batcher for ``key``, creating it with ``factory`` if needed."""
        with self._lock:
            batcher = self._batchers.get(key)
            if batcher is None or batcher.retired:
                batcher = factory()
                self._batchers[key] = batcher
            return batcher

    def retire_model(self, name: str) -> int:
        """Retire every batcher for a model; returns how many were retired."""
        with self._lock:
            keys = [key for key in self._batchers if key[0] == name]
            retiring = [self._batchers.pop(key) for key in keys]
        for batcher in retiring:
            batcher.retire()
        return len(retiring)
