"""Tests for exporting OpenMP thread settings and worker-thread initialization."""

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from forge.api.services import runtime


class TestExportThreadEnv:
    """OMP_NUM_THREADS / MKL_NUM_THREADS export."""

    def test_sets_env_from_cpu_quota(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
        monkeypatch.delenv("FORGE_TORCH_THREADS", raising=False)
        monkeypatch.setattr(runtime, "container_cpu_limit", lambda: 3.0)
        monkeypatch.setattr(runtime.os, "cpu_count", lambda: 48)

        assert runtime.export_thread_env() == 3
        assert runtime.os.environ["OMP_NUM_THREADS"] == "3"
        assert runtime.os.environ["MKL_NUM_THREADS"] == "3"

    def test_respects_explicit_omp_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
        monkeypatch.setattr(runtime, "container_cpu_limit", lambda: 8.0)

        assert runtime.export_thread_env() == 1
        assert runtime.os.environ["MKL_NUM_THREADS"] == "1"

    def test_runtime_module_does_not_import_torch(self) -> None:
        code = (
            "import sys; import forge.api.services.runtime; "
            "sys.exit(1 if 'torch' in sys.modules else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, check=False)
        assert result.returncode == 0, result.stderr.decode()

    def test_package_import_exports_env_before_torch(self) -> None:
        code = (
            "import os; os.environ.pop('OMP_NUM_THREADS', None); "
            "import forge.api, sys; "
            "assert 'OMP_NUM_THREADS' in os.environ; "
            "sys.exit(1 if 'torch' in sys.modules else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, check=False)
        assert result.returncode == 0, result.stderr.decode()


def test_init_worker_thread_matches_main_thread_count() -> None:
    import torch

    original = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        seen: list[int] = []
        with ThreadPoolExecutor(1, initializer=runtime.init_worker_thread) as pool:
            seen.append(pool.submit(torch.get_num_threads).result())
        assert seen == [2]
        assert threading.current_thread() is threading.main_thread()
    finally:
        torch.set_num_threads(original)
