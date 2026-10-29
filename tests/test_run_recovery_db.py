"""Integration test: orphaned-run recovery against a real PostgreSQL database.

Uses DATABASE_URL (the docker-compose / CI pgvector service) and is skipped
when no database is reachable.
"""

from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from forge.api.models.database import DATABASE_URL, Base, Dataset, Experiment, Run
from forge.api.services.run_recovery import recover_orphaned_runs


@pytest.fixture
def db() -> Iterator[Session]:
    """A session on a fresh schema, matching the app's autoflush=False sessions."""
    engine = create_engine(DATABASE_URL)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except OperationalError:
        pytest.skip("PostgreSQL is not reachable")
    Base.metadata.create_all(engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def _experiment(db: Session, run_statuses: list[str]) -> Experiment:
    dataset = Dataset(id=uuid4(), name=f"ds-{uuid4()}", source="yfinance")
    experiment = Experiment(id=uuid4(), name="exp", dataset_id=dataset.id, status="running")
    db.add_all([dataset, experiment])
    db.flush()
    for status in run_statuses:
        db.add(Run(id=uuid4(), experiment_id=experiment.id, model_type="lstm", hyperparameters={}, status=status))
    db.commit()
    return experiment


def test_recovers_running_run_and_settles_experiment(db: Session) -> None:
    experiment = _experiment(db, ["completed", "running"])

    assert recover_orphaned_runs(db) == 1

    db.refresh(experiment)
    assert experiment.status == "failed"
    statuses = sorted(r.status for r in db.query(Run).filter(Run.experiment_id == experiment.id))
    assert statuses == ["completed", "failed"]


def test_running_experiment_with_only_pending_runs_becomes_partial(db: Session) -> None:
    experiment = _experiment(db, ["completed", "pending"])

    assert recover_orphaned_runs(db) == 0

    db.refresh(experiment)
    assert experiment.status == "partial"
