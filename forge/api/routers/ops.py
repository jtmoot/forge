"""Ops monitoring endpoints — log ingestion, querying, and summary."""

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func as sql_func
from sqlalchemy.orm import Session

from forge.api.models.database import OpsLog, get_db
from forge.api.models.schemas import (
    OpsLogCreateRequest,
    OpsLogListResponse,
    OpsLogResponse,
    OpsLogSummaryResponse,
)
from forge.api.services.anomaly import flag_anomalies
from forge.api.services.metrics import LLM_COST_DOLLARS, OPS_ERRORS_TOTAL

router = APIRouter(prefix="/api/ops", tags=["ops"])


def _ops_log_to_response(log: OpsLog, is_anomaly: bool = False) -> OpsLogResponse:
    """Convert an OpsLog ORM object to a response, handling the metadata_ alias."""
    return OpsLogResponse(
        id=log.id,
        project_name=log.project_name,
        log_level=log.log_level,
        message=log.message,
        metadata=log.metadata_,
        source=log.source,
        cost_usd=log.cost_usd,
        is_anomaly=is_anomaly,
        created_at=log.created_at,
    )


@router.post("/logs", response_model=OpsLogResponse, status_code=201)
async def create_log(
    request: OpsLogCreateRequest, db: Session = Depends(get_db)
) -> OpsLogResponse:
    """Ingest a single operational log entry."""
    log_entry = OpsLog(
        project_name=request.project_name,
        log_level=request.log_level,
        message=request.message,
        metadata_=request.metadata,
        source=request.source,
        cost_usd=request.cost_usd,
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)

    # Prometheus: track errors and LLM costs
    if request.log_level in ("ERROR", "CRITICAL"):
        OPS_ERRORS_TOTAL.labels(project=request.project_name).inc()
    if request.cost_usd and request.cost_usd > 0:
        LLM_COST_DOLLARS.inc(request.cost_usd)

    return _ops_log_to_response(log_entry)


@router.get("/logs", response_model=OpsLogListResponse)
async def query_logs(
    db: Session = Depends(get_db),
    project: str | None = Query(None, description="Filter by project name"),
    level: str | None = Query(None, description="Filter by log level"),
    start_date: datetime | None = Query(None, description="Filter from date"),
    end_date: datetime | None = Query(None, description="Filter to date"),
) -> OpsLogListResponse:
    """Query ops logs with optional filters. Flags anomalous entries via rolling z-score."""
    query = db.query(OpsLog)

    if project:
        query = query.filter(OpsLog.project_name == project)
    if level:
        query = query.filter(OpsLog.log_level == level)
    if start_date:
        query = query.filter(OpsLog.created_at >= start_date)
    if end_date:
        query = query.filter(OpsLog.created_at <= end_date)

    logs = query.order_by(OpsLog.created_at.asc()).all()

    # Run anomaly detection on cost_usd values
    cost_values = [log.cost_usd if log.cost_usd is not None else 0.0 for log in logs]
    anomaly_flags = flag_anomalies(cost_values) if cost_values else []

    log_responses = []
    for i, log in enumerate(logs):
        is_anomaly = anomaly_flags[i] if i < len(anomaly_flags) else False
        log_responses.append(_ops_log_to_response(log, is_anomaly=is_anomaly))

    return OpsLogListResponse(logs=log_responses, count=len(log_responses))


@router.get("/summary", response_model=OpsLogSummaryResponse)
async def get_summary(db: Session = Depends(get_db)) -> OpsLogSummaryResponse:
    """Aggregate stats across all ops logs."""
    total_logs = db.query(sql_func.count(OpsLog.id)).scalar() or 0

    error_count = (
        db.query(sql_func.count(OpsLog.id))
        .filter(OpsLog.log_level.in_(["ERROR", "CRITICAL"]))
        .scalar()
        or 0
    )

    total_cost = (
        db.query(sql_func.coalesce(sql_func.sum(OpsLog.cost_usd), 0.0)).scalar()
    )

    # Events grouped by project
    project_rows = (
        db.query(OpsLog.project_name, sql_func.count(OpsLog.id))
        .group_by(OpsLog.project_name)
        .all()
    )
    events_by_project = {row[0]: row[1] for row in project_rows}

    # Events grouped by level
    level_rows = (
        db.query(OpsLog.log_level, sql_func.count(OpsLog.id))
        .group_by(OpsLog.log_level)
        .all()
    )
    events_by_level = {row[0]: row[1] for row in level_rows}

    return OpsLogSummaryResponse(
        total_logs=total_logs,
        error_count=error_count,
        total_cost_usd=float(total_cost),
        events_by_project=events_by_project,
        events_by_level=events_by_level,
    )
