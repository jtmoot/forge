"""Tests for running migrations at API startup."""

from pathlib import Path

import pytest

from forge.api.services.migrations import MIGRATE_SCRIPT, run_migrations


def test_default_script_path_points_at_repo_script() -> None:
    assert MIGRATE_SCRIPT.name == "migrate.py"
    assert MIGRATE_SCRIPT.is_file()


def test_runs_script_and_reports_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_SKIP_MIGRATIONS", raising=False)
    marker = tmp_path / "ran"
    script = tmp_path / "migrate.py"
    script.write_text(f"open({str(marker)!r}, 'w').write('yes')\nprint('[migrate] ok')\n")

    assert run_migrations(script) is True
    assert marker.read_text() == "yes"


def test_nonzero_exit_reports_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_SKIP_MIGRATIONS", raising=False)
    script = tmp_path / "migrate.py"
    script.write_text("import sys\nsys.exit(3)\n")
    assert run_migrations(script) is False


def test_missing_script_reports_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_SKIP_MIGRATIONS", raising=False)
    assert run_migrations(tmp_path / "nope.py") is False


def test_skip_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_SKIP_MIGRATIONS", "1")
    assert run_migrations(tmp_path / "nope.py") is True
