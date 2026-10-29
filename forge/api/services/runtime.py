"""Runtime sizing — match torch's thread pool to the container's CPU quota.

os.cpu_count() reports the host's cores, not the container's cgroup CPU
quota. Torch sizes its intra-op pool from the former, so on a throttled
container it oversubscribes the quota and training/inference slow down by
orders of magnitude. This module reads the quota and sizes the pool to it.

torch.set_num_threads() only applies to the calling thread's OpenMP team on
Linux; worker threads (FastAPI's threadpool, asyncio.to_thread, the batcher)
otherwise start a team sized to every visible core. export_thread_env() sets
OMP_NUM_THREADS before torch is first imported so every thread inherits the
right size. This module must not import torch at module level for that reason.
"""

import logging
import math
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def container_cpu_limit(
    cgroup_v2_file: Path = _CGROUP_V2_CPU_MAX,
    cgroup_v1_quota_file: Path = _CGROUP_V1_QUOTA,
    cgroup_v1_period_file: Path = _CGROUP_V1_PERIOD,
) -> float | None:
    """Return the container's CPU quota in cores, or None if it's unlimited/unknown."""
    try:
        quota_text, period_text = cgroup_v2_file.read_text().split()[:2]
        if quota_text == "max":
            return None
        return int(quota_text) / int(period_text)
    except (OSError, ValueError):
        pass

    try:
        quota = int(cgroup_v1_quota_file.read_text().strip())
        period = int(cgroup_v1_period_file.read_text().strip())
    except (OSError, ValueError):
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period


def resolve_torch_threads(cpu_limit: float | None, cpu_count: int, override: str | None) -> int:
    """Pick the torch intra-op thread count.

    A valid positive FORGE_TORCH_THREADS override wins; otherwise the CPU
    quota rounded down (at least 1, at most cpu_count); otherwise cpu_count.
    """
    if override:
        try:
            requested = int(override)
        except ValueError:
            requested = 0
        if requested > 0:
            return requested
        logger.warning("Ignoring invalid FORGE_TORCH_THREADS=%r", override)

    if cpu_limit is None:
        return max(1, cpu_count)
    return max(1, min(cpu_count, math.floor(cpu_limit)))


def export_thread_env() -> int:
    """Set OMP_NUM_THREADS / MKL_NUM_THREADS from the CPU quota unless already set.

    Must run before torch (or any OpenMP library) is first imported. Returns
    the thread count now in effect for OpenMP.
    """
    threads = resolve_torch_threads(
        container_cpu_limit(), os.cpu_count() or 1, os.getenv("FORGE_TORCH_THREADS")
    )
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
    return int(os.environ["OMP_NUM_THREADS"])


def init_worker_thread() -> None:
    """Thread-pool initializer: size this thread's OpenMP team like the main thread's."""
    import torch

    torch.set_num_threads(torch.get_num_threads())


def configure_torch_threads() -> dict:
    """Size torch's thread pool to the container and return the runtime facts used."""
    import torch

    cpu_count = os.cpu_count() or 1
    cpu_limit = container_cpu_limit()
    threads = resolve_torch_threads(cpu_limit, cpu_count, os.getenv("FORGE_TORCH_THREADS"))
    torch.set_num_threads(threads)
    logger.warning(
        "Runtime: cpu_count=%d cpu_limit=%s torch_threads=%d",
        cpu_count, cpu_limit, threads,
    )
    return runtime_info()


def runtime_info() -> dict:
    """Describe the compute this process runs on (recorded with serving results)."""
    import torch

    return {
        "cpu_count": os.cpu_count() or 1,
        "cpu_limit": container_cpu_limit(),
        "torch_threads": torch.get_num_threads(),
    }
