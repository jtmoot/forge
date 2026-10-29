"""Tests for container CPU detection and torch thread sizing."""

from pathlib import Path

import pytest

from forge.api.services.runtime import container_cpu_limit, resolve_torch_threads


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


class TestContainerCpuLimit:
    """Parsing cgroup v2 and v1 CPU quota files."""

    def test_cgroup_v2_quota(self, tmp_path: Path) -> None:
        cpu_max = _write(tmp_path / "cpu.max", "200000 100000\n")
        assert container_cpu_limit(cgroup_v2_file=cpu_max) == 2.0

    def test_cgroup_v2_fractional_quota(self, tmp_path: Path) -> None:
        cpu_max = _write(tmp_path / "cpu.max", "50000 100000\n")
        assert container_cpu_limit(cgroup_v2_file=cpu_max) == 0.5

    def test_cgroup_v2_unlimited(self, tmp_path: Path) -> None:
        cpu_max = _write(tmp_path / "cpu.max", "max 100000\n")
        missing = tmp_path / "nope"
        assert container_cpu_limit(cpu_max, missing, missing) is None

    def test_cgroup_v1_quota(self, tmp_path: Path) -> None:
        quota = _write(tmp_path / "cpu.cfs_quota_us", "400000\n")
        period = _write(tmp_path / "cpu.cfs_period_us", "100000\n")
        assert container_cpu_limit(tmp_path / "missing", quota, period) == 4.0

    def test_cgroup_v1_unlimited(self, tmp_path: Path) -> None:
        quota = _write(tmp_path / "cpu.cfs_quota_us", "-1\n")
        period = _write(tmp_path / "cpu.cfs_period_us", "100000\n")
        assert container_cpu_limit(tmp_path / "missing", quota, period) is None

    def test_no_cgroup_files(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"
        assert container_cpu_limit(missing, missing, missing) is None


class TestResolveTorchThreads:
    """Choosing the torch intra-op thread count."""

    def test_env_override_wins(self) -> None:
        assert resolve_torch_threads(cpu_limit=8.0, cpu_count=64, override="3") == 3

    def test_uses_cpu_limit_rounded_down(self) -> None:
        assert resolve_torch_threads(cpu_limit=2.5, cpu_count=64, override=None) == 2

    def test_fractional_limit_floors_at_one(self) -> None:
        assert resolve_torch_threads(cpu_limit=0.5, cpu_count=64, override=None) == 1

    def test_limit_never_exceeds_cpu_count(self) -> None:
        assert resolve_torch_threads(cpu_limit=16.0, cpu_count=4, override=None) == 4

    def test_no_limit_uses_cpu_count(self) -> None:
        assert resolve_torch_threads(cpu_limit=None, cpu_count=6, override=None) == 6

    @pytest.mark.parametrize("bad", ["0", "-2", "many"])
    def test_invalid_override_ignored(self, bad: str) -> None:
        assert resolve_torch_threads(cpu_limit=2.0, cpu_count=8, override=bad) == 2
