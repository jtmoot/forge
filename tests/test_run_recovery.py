"""Tests for deciding experiment status after orphaned runs are recovered."""

import pytest

from forge.api.services.run_recovery import experiment_status_after_recovery


@pytest.mark.parametrize(
    ("run_statuses", "expected"),
    [
        (["completed", "completed"], "completed"),
        (["completed", "failed"], "failed"),
        (["failed", "pending"], "failed"),
        (["completed", "pending"], "partial"),
        (["pending", "pending"], "partial"),
        ([], "failed"),
    ],
)
def test_experiment_status_after_recovery(run_statuses: list[str], expected: str) -> None:
    assert experiment_status_after_recovery(run_statuses) == expected


def test_running_status_is_not_expected_after_recovery() -> None:
    with pytest.raises(ValueError, match="running"):
        experiment_status_after_recovery(["running", "completed"])
