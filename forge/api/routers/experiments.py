"""Experiments router — create, list, detail, and trigger training runs."""

import logging
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from forge.api.models.database import (
    Dataset,
    Experiment,
    FeatureSet,
    Run,
    RunEnvironment,
    SessionLocal,
    get_db,
)
from forge.api.models.schemas import (
    EnvironmentDiffResponse,
    ExperimentCreateRequest,
    ExperimentDetailResponse,
    ExperimentListResponse,
    ExperimentResponse,
    ReproduceResponse,
    ReproducibilityReport,
    RunEnvironmentResponse,
    RunResponse,
)
from forge.api.services.reproducibility import diff_environments, verify_reproducibility
from forge.api.services.training import run_experiment_run

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.post(
    "",
    response_model=ExperimentDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_experiment(
    request: ExperimentCreateRequest,
    db: Session = Depends(get_db),
) -> ExperimentDetailResponse:
    """Create an experiment with one or more run configurations."""
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {request.dataset_id} not found",
        )

    experiment = Experiment(
        name=request.name,
        description=request.description,
        dataset_id=request.dataset_id,
        status="pending",
    )
    db.add(experiment)
    db.flush()

    runs = []
    for idx, run_config in enumerate(request.runs):
        run = Run(
            experiment_id=experiment.id,
            run_name=run_config.run_name or f"{run_config.model_type}_{idx}",
            model_type=run_config.model_type,
            hyperparameters=run_config.hyperparameters,
            feature_engineering=run_config.feature_engineering,
            feature_set_id=run_config.feature_set_id,
            status="pending",
        )
        db.add(run)
        runs.append(run)

    db.commit()
    db.refresh(experiment)
    for run in runs:
        db.refresh(run)

    return ExperimentDetailResponse(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        dataset_id=experiment.dataset_id,
        status=experiment.status,
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
        runs=[RunResponse.model_validate(r) for r in runs],
    )


@router.get("", response_model=ExperimentListResponse)
def list_experiments(
    db: Session = Depends(get_db),
) -> ExperimentListResponse:
    """List all experiments ordered by creation time."""
    experiments = (
        db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    )
    return ExperimentListResponse(
        experiments=[ExperimentResponse.model_validate(e) for e in experiments],
        count=len(experiments),
    )


@router.get("/compare-environments", response_model=EnvironmentDiffResponse)
def compare_environments(
    run_a: UUID,
    run_b: UUID,
    db: Session = Depends(get_db),
) -> EnvironmentDiffResponse:
    """Diff two run environments side-by-side.

    Shows which packages were added, removed, or changed between the
    two runs, plus field-level diffs and a reproducibility report.
    """
    env_a = db.query(RunEnvironment).filter(RunEnvironment.run_id == run_a).first()
    env_b = db.query(RunEnvironment).filter(RunEnvironment.run_id == run_b).first()

    if env_a is None and env_b is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No environment data found for either run",
        )

    diff = diff_environments(env_a, env_b)
    repro = verify_reproducibility(run_a, run_b, db)

    return EnvironmentDiffResponse(
        run_a=RunEnvironmentResponse.model_validate(env_a) if env_a else None,
        run_b=RunEnvironmentResponse.model_validate(env_b) if env_b else None,
        packages_added=diff["packages_added"],
        packages_removed=diff["packages_removed"],
        packages_changed=diff["packages_changed"],
        field_diffs=diff["field_diffs"],
        environments_identical=diff["environments_identical"],
        reproducibility=ReproducibilityReport(**repro),
    )


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
def get_experiment(
    experiment_id: UUID,
    db: Session = Depends(get_db),
) -> ExperimentDetailResponse:
    """Get experiment details including all runs with metrics."""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    runs = (
        db.query(Run)
        .filter(Run.experiment_id == experiment_id)
        .order_by(Run.created_at)
        .all()
    )

    return ExperimentDetailResponse(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        dataset_id=experiment.dataset_id,
        status=experiment.status,
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
        runs=[RunResponse.model_validate(r) for r in runs],
    )


def _execute_runs_in_background(experiment_id: UUID, run_ids: list[UUID]) -> None:
    """Execute pending runs in a background thread with its own DB session.

    This function is invoked via FastAPI BackgroundTasks so the HTTP response
    returns immediately while training proceeds asynchronously.
    """
    db = SessionLocal()
    try:
        for run_id in run_ids:
            try:
                run_experiment_run(run_id, db)
            except Exception:
                logger.exception("Run %s failed", run_id)

        # Update experiment status based on run outcomes
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if experiment is None:
            return

        all_runs = db.query(Run).filter(Run.experiment_id == experiment_id).all()
        statuses = {r.status for r in all_runs}

        if statuses == {"completed"}:
            experiment.status = "completed"
        elif "failed" in statuses:
            experiment.status = "failed"
        elif "pending" in statuses:
            experiment.status = "partial"
        experiment.updated_at = datetime.now(UTC)
        db.commit()
    finally:
        db.close()


@router.post("/{experiment_id}/run", response_model=ExperimentDetailResponse)
def trigger_experiment_runs(
    experiment_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ExperimentDetailResponse:
    """Trigger all pending runs for an experiment.

    Training is executed in the background so this endpoint returns immediately
    with the experiment in 'running' status. Poll GET /{id} for completion.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    all_runs = db.query(Run).filter(Run.experiment_id == experiment_id).all()
    pending_runs = [r for r in all_runs if r.status == "pending"]

    if not pending_runs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pending runs to execute",
        )

    # Update experiment status and return immediately
    experiment.status = "running"
    experiment.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(experiment)

    # Schedule training in the background with its own DB session
    pending_run_ids = [r.id for r in pending_runs]
    background_tasks.add_task(_execute_runs_in_background, experiment_id, pending_run_ids)

    return ExperimentDetailResponse(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        dataset_id=experiment.dataset_id,
        status=experiment.status,
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
        runs=[RunResponse.model_validate(r) for r in all_runs],
    )


# ---------------------------------------------------------------------------
# Reproducibility endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/{experiment_id}/runs/{run_id}/environment",
    response_model=RunEnvironmentResponse,
)
def get_run_environment(
    experiment_id: UUID,
    run_id: UUID,
    db: Session = Depends(get_db),
) -> RunEnvironmentResponse:
    """Get the full environment snapshot for a specific run."""
    run = db.query(Run).filter(Run.id == run_id, Run.experiment_id == experiment_id).first()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found")

    env = db.query(RunEnvironment).filter(RunEnvironment.run_id == run_id).first()
    if env is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No environment captured for run {run_id}",
        )

    return RunEnvironmentResponse.model_validate(env)


@router.get(
    "/{experiment_id}/runs/{run_id}/reproduce",
    response_model=ReproduceResponse,
)
def get_reproduce_spec(
    experiment_id: UUID,
    run_id: UUID,
    db: Session = Depends(get_db),
) -> ReproduceResponse:
    """Generate a reproduction specification for a training run.

    Returns the git SHA, a shell command to reproduce, data version hash,
    feature set info, and any warnings about reproducibility.
    """
    run = db.query(Run).filter(Run.id == run_id, Run.experiment_id == experiment_id).first()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found")

    env = db.query(RunEnvironment).filter(RunEnvironment.run_id == run_id).first()

    warnings: list[str] = []
    git_sha = env.git_sha if env else None
    random_seed = env.random_seed if env else 42
    env_hash = env.env_hash if env else None

    if env and env.git_dirty:
        warnings.append("git_dirty was True — uncommitted changes may affect results")
    if not git_sha:
        warnings.append("No git SHA captured — cannot pin exact code version")

    # Look up feature set name if present
    feature_set_name: str | None = None
    if run.feature_set_id:
        fs = db.query(FeatureSet).filter(FeatureSet.id == run.feature_set_id).first()
        feature_set_name = f"{fs.name}_v{fs.version}" if fs else str(run.feature_set_id)

    # Build reproduction command
    checkout = f"git checkout {git_sha}" if git_sha else "# git SHA not available"
    hyperparams_json = str(run.hyperparameters or {}).replace("'", '"')
    command = (
        f"{checkout} && "
        f"pip install -r requirements.txt && "
        f"python -m forge.train "
        f"--experiment-id {experiment_id} "
        f"--run-config '{hyperparams_json}' "
        f"--model-type {run.model_type} "
        f"--seed {random_seed}"
    )

    return ReproduceResponse(
        git_sha=git_sha,
        command=command,
        data_version=f"sha256:{run.data_version_hash}" if run.data_version_hash else None,
        feature_set=feature_set_name,
        environment_hash=f"sha256:{env_hash}" if env_hash else None,
        random_seed=random_seed,
        warnings=warnings,
    )


