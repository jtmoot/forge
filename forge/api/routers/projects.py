"""Router for project aggregation and activity feed endpoints."""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func, text
from sqlalchemy.orm import Session

from forge.api.models.database import (
    DriftReport,
    Experiment,
    GitEvent,
    OpsLog,
    Run,
    get_db,
)
from forge.api.models.schemas import (
    ActivityFeedItem,
    ActivityFeedResponse,
    DashboardSummaryResponse,
    ExperimentResponse,
    GitEventResponse,
    OpsLogResponse,
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectSummary,
)
from forge.api.services.anomaly import flag_anomalies

router = APIRouter(prefix="/api", tags=["projects"])


def _compute_health(error_count_24h: int, warn_count_24h: int) -> str:
    """Derive health status from recent error/warning counts.

    green  = no errors or warnings in last 24h
    yellow = warnings but no errors in last 24h
    red    = any errors in last 24h
    """
    if error_count_24h > 0:
        return "red"
    if warn_count_24h > 0:
        return "yellow"
    return "green"


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(db: Session = Depends(get_db)) -> ProjectListResponse:
    """Return unique projects with aggregated 7-day stats and health status.

    Uses two GROUP BY queries (one for ops_logs, one for git_events) instead
    of N+1 per-project queries, then merges the results in Python.
    """
    now = datetime.now(UTC)
    seven_days_ago = now - timedelta(days=7)
    twenty_four_hours_ago = now - timedelta(hours=24)

    # -- Single aggregate query over ops_logs --------------------------------
    ops_stats = (
        db.query(
            OpsLog.project_name,
            func.coalesce(
                func.sum(case((OpsLog.created_at >= seven_days_ago, OpsLog.cost_usd), else_=0.0)),
                0.0,
            ).label("total_cost_7d"),
            func.count(case(
                (
                    (OpsLog.log_level == "ERROR") & (OpsLog.created_at >= seven_days_ago),
                    OpsLog.id,
                ),
            )).label("error_count_7d"),
            func.count(case(
                (
                    OpsLog.log_level.in_(["ERROR", "CRITICAL"]) & (OpsLog.created_at >= twenty_four_hours_ago),
                    OpsLog.id,
                ),
            )).label("error_count_24h"),
            func.count(case(
                (
                    (OpsLog.log_level == "WARN") & (OpsLog.created_at >= twenty_four_hours_ago),
                    OpsLog.id,
                ),
            )).label("warn_count_24h"),
            func.max(OpsLog.created_at).label("last_ops_activity"),
        )
        .group_by(OpsLog.project_name)
        .all()
    )
    ops_by_name = {row.project_name: row for row in ops_stats}

    # -- Single aggregate query over git_events ------------------------------
    git_stats = (
        db.query(
            GitEvent.repo,
            func.count(case(
                (GitEvent.created_at >= seven_days_ago, GitEvent.id),
            )).label("commit_count_7d"),
            func.max(GitEvent.created_at).label("last_git_activity"),
        )
        .group_by(GitEvent.repo)
        .all()
    )
    git_by_name = {row.repo: row for row in git_stats}

    # -- Merge results -------------------------------------------------------
    all_names = sorted(set(ops_by_name.keys()) | set(git_by_name.keys()))

    results: list[ProjectSummary] = []
    for name in all_names:
        ops = ops_by_name.get(name)
        git = git_by_name.get(name)

        commit_count = git.commit_count_7d if git else 0
        total_cost = float(ops.total_cost_7d) if ops else 0.0
        error_count_7d = ops.error_count_7d if ops else 0
        error_count_24h = ops.error_count_24h if ops else 0
        warn_count_24h = ops.warn_count_24h if ops else 0

        timestamps = [
            t for t in [
                ops.last_ops_activity if ops else None,
                git.last_git_activity if git else None,
            ] if t is not None
        ]
        last_activity = max(timestamps) if timestamps else None

        results.append(
            ProjectSummary(
                name=name,
                commit_count_7d=commit_count,
                total_cost_7d=total_cost,
                error_count_7d=error_count_7d,
                last_activity=last_activity,
                health=_compute_health(error_count_24h, warn_count_24h),
            )
        )

    return ProjectListResponse(projects=results, count=len(results))


@router.get("/projects/{name}", response_model=ProjectDetailResponse)
async def get_project_detail(
    name: str,
    db: Session = Depends(get_db),
) -> ProjectDetailResponse:
    """Return project detail: recent logs, git events, and linked experiments."""
    # Recent ops logs (limit 50)
    logs_query = (
        db.query(OpsLog)
        .filter(OpsLog.project_name == name)
        .order_by(OpsLog.created_at.desc())
        .limit(50)
        .all()
    )

    # Flag anomalies on the cost data
    costs = [log.cost_usd or 0.0 for log in logs_query]
    anomaly_flags = flag_anomalies(costs) if len(costs) >= 3 else [False] * len(costs)

    recent_logs = [
        OpsLogResponse(
            id=log.id,
            project_name=log.project_name,
            log_level=log.log_level,
            message=log.message,
            metadata=log.metadata_,
            source=log.source,
            cost_usd=log.cost_usd,
            is_anomaly=anomaly_flags[i],
            created_at=log.created_at,
        )
        for i, log in enumerate(logs_query)
    ]

    # Recent git events (limit 50)
    git_events_query = (
        db.query(GitEvent)
        .filter(GitEvent.repo == name)
        .order_by(GitEvent.created_at.desc())
        .limit(50)
        .all()
    )
    git_events = [
        GitEventResponse.model_validate(evt) for evt in git_events_query
    ]

    # Linked experiments: match via experiment name OR dataset name containing
    # the project name (loose link since there's no direct FK).
    from forge.api.models.database import Dataset
    escaped_name = name.replace("%", r"\%").replace("_", r"\_")
    linked_experiments_query = (
        db.query(Experiment)
        .outerjoin(Dataset, Experiment.dataset_id == Dataset.id)
        .filter(
            Experiment.name.ilike(f"%{escaped_name}%")
            | Dataset.name.ilike(f"%{escaped_name}%")
        )
        .order_by(Experiment.created_at.desc())
        .limit(20)
        .all()
    )
    # If no name-based matches, show all recent experiments as fallback
    if not linked_experiments_query:
        linked_experiments_query = (
            db.query(Experiment)
            .order_by(Experiment.created_at.desc())
            .limit(10)
            .all()
        )
    linked_experiments = [
        ExperimentResponse.model_validate(exp) for exp in linked_experiments_query
    ]

    return ProjectDetailResponse(
        name=name,
        recent_logs=recent_logs,
        git_events=git_events,
        linked_experiments=linked_experiments,
    )


@router.get("/activity/feed", response_model=ActivityFeedResponse)
async def get_activity_feed(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> ActivityFeedResponse:
    """Return interleaved timeline of git commits, ops logs, and experiment completions.

    Uses a SQL UNION to sort and limit across all three sources in the database,
    then hydrates the matching rows into ActivityFeedItem objects.
    """
    # SQL UNION: get the top N IDs+timestamps across all sources in one query
    union_sql = text("""
        (SELECT id::text, created_at AS ts, 'git_commit' AS item_type
         FROM git_events ORDER BY created_at DESC LIMIT :lim)
        UNION ALL
        (SELECT id::text, created_at AS ts, 'ops_log' AS item_type
         FROM ops_logs ORDER BY created_at DESC LIMIT :lim)
        UNION ALL
        (SELECT r.id::text, COALESCE(r.completed_at, r.created_at) AS ts,
                'experiment_completion' AS item_type
         FROM runs r WHERE r.status = 'completed'
         ORDER BY ts DESC LIMIT :lim)
        ORDER BY ts DESC
        LIMIT :lim
    """)
    rows = db.execute(union_sql, {"lim": limit}).fetchall()

    # Collect IDs by type for batch fetching
    git_ids = [row.id for row in rows if row.item_type == "git_commit"]
    ops_ids = [row.id for row in rows if row.item_type == "ops_log"]
    run_ids = [row.id for row in rows if row.item_type == "experiment_completion"]

    # Batch-fetch the actual objects (3 queries max, regardless of limit)
    git_map = {}
    if git_ids:
        for evt in db.query(GitEvent).filter(GitEvent.id.in_(git_ids)).all():
            git_map[str(evt.id)] = evt

    ops_map = {}
    if ops_ids:
        for log in db.query(OpsLog).filter(OpsLog.id.in_(ops_ids)).all():
            ops_map[str(log.id)] = log

    run_map = {}
    if run_ids:
        for run, exp_name in (
            db.query(Run, Experiment.name)
            .join(Experiment, Run.experiment_id == Experiment.id)
            .filter(Run.id.in_(run_ids))
            .all()
        ):
            run_map[str(run.id)] = (run, exp_name)

    # Hydrate in the order returned by the UNION (already sorted by DB)
    items: list[ActivityFeedItem] = []
    for row in rows:
        if row.item_type == "git_commit" and row.id in git_map:
            evt = git_map[row.id]
            items.append(ActivityFeedItem(
                type="git_commit",
                timestamp=evt.created_at,
                project=evt.repo,
                summary=f"[{evt.branch or 'unknown'}] {evt.commit_message or 'No message'}",
                detail={
                    "author": evt.author,
                    "commit_sha": evt.commit_sha,
                    "files_changed": evt.files_changed,
                    "additions": evt.additions,
                    "deletions": evt.deletions,
                },
            ))
        elif row.item_type == "ops_log" and row.id in ops_map:
            log = ops_map[row.id]
            items.append(ActivityFeedItem(
                type="ops_log",
                timestamp=log.created_at,
                project=log.project_name,
                summary=f"[{log.log_level}] {log.message or ''}",
                detail={
                    "source": log.source,
                    "cost_usd": log.cost_usd,
                },
            ))
        elif row.item_type == "experiment_completion" and row.id in run_map:
            run, exp_name = run_map[row.id]
            summary = (
                f"Run '{run.run_name or run.model_type}' completed — accuracy {run.accuracy:.1%}"
                if run.accuracy
                else f"Run '{run.run_name or run.model_type}' completed"
            )
            items.append(ActivityFeedItem(
                type="experiment_completion",
                timestamp=run.completed_at or run.created_at,
                project=exp_name,
                summary=summary,
                detail={
                    "model_type": run.model_type,
                    "accuracy": run.accuracy,
                    "efficiency_score": run.efficiency_score,
                    "run_id": str(run.id),
                },
            ))

    return ActivityFeedResponse(items=items, count=len(items))


@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
async def get_dashboard_summary(
    db: Session = Depends(get_db),
) -> DashboardSummaryResponse:
    """Aggregate stats for the dashboard home: total projects, active experiments, alerts, weekly cost."""
    now = datetime.now(UTC)
    seven_days_ago = now - timedelta(days=7)
    twenty_four_hours_ago = now - timedelta(hours=24)

    # Distinct project names from ops_logs + git_events
    ops_names = {r[0] for r in db.query(OpsLog.project_name).distinct().all()}
    git_names = {r[0] for r in db.query(GitEvent.repo).distinct().all()}
    total_projects = len(ops_names | git_names)

    # Active experiments (status = running)
    active_experiments = (
        db.query(func.count(Experiment.id))
        .filter(Experiment.status == "running")
        .scalar()
    ) or 0

    # Ops alerts: error + critical count last 24h
    ops_alerts = (
        db.query(func.count(OpsLog.id))
        .filter(
            OpsLog.log_level.in_(["ERROR", "CRITICAL"]),
            OpsLog.created_at >= twenty_four_hours_ago,
        )
        .scalar()
    ) or 0

    # Weekly LLM cost
    weekly_cost = (
        db.query(func.coalesce(func.sum(OpsLog.cost_usd), 0.0))
        .filter(OpsLog.created_at >= seven_days_ago)
        .scalar()
    ) or 0.0

    # Drift alerts: drifted reports in last 7 days
    drift_alerts = (
        db.query(func.count(DriftReport.id))
        .filter(
            DriftReport.is_drifted == "true",
            DriftReport.created_at >= seven_days_ago,
        )
        .scalar()
    ) or 0

    return DashboardSummaryResponse(
        total_projects=total_projects,
        active_experiments=active_experiments,
        ops_alerts_24h=ops_alerts,
        weekly_llm_cost=float(weekly_cost),
        drift_alerts_7d=drift_alerts,
    )
