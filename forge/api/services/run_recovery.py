"""Startup recovery for training runs orphaned by a restart.

Runs execute in FastAPI background tasks inside the API process. If the
process restarts mid-training (e.g. a redeploy), the run row stays
``running`` and its experiment stays ``running`` forever. The API runs a
single worker, so at startup nothing can still be executing: any ``running``
run is orphaned and is marked failed.
"""

import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from forge.api.models.database import Experiment, Run

logger = logging.getLogger(__name__)


def experiment_status_after_recovery(run_statuses: list[str]) -> str:
    """Return an experiment's status once none of its runs can still be running.

    Mirrors the status rules used when background training finishes.

    Raises:
        ValueError: If a run is still marked running.
    """
    statuses = set(run_statuses)
    if "running" in statuses:
        raise ValueError("Recover running runs before deciding experiment status")
    if statuses == {"completed"}:
        return "completed"
    if "pending" in statuses and "failed" not in statuses:
        return "partial"
    return "failed"


def recover_orphaned_runs(db: Session) -> int:
    """Mark orphaned ``running`` runs failed and settle their experiments.

    Returns:
        Number of runs marked failed.
    """
    now = datetime.now(UTC)
    orphaned = db.query(Run).filter(Run.status == "running").all()
    for run in orphaned:
        run.status = "failed"
        run.completed_at = now

    stuck_experiments = db.query(Experiment).filter(Experiment.status == "running").all()
    for experiment in stuck_experiments:
        run_statuses = [
            status
            for (status,) in db.query(Run.status).filter(Run.experiment_id == experiment.id)
        ]
        experiment.status = experiment_status_after_recovery(run_statuses)
        experiment.updated_at = now

    db.commit()
    if orphaned or stuck_experiments:
        logger.warning(
            "Recovered %d orphaned runs and %d stuck experiments",
            len(orphaned), len(stuck_experiments),
        )
    return len(orphaned)
