"""Model registry router — register models, manage versions, transition stages."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from forge.api.models.database import get_db
from forge.api.models.schemas import (
    ModelDetailResponse,
    ModelListItem,
    ModelListResponse,
    ModelRegisterRequest,
    ModelStageHistoryResponse,
    ModelStageTransitionRequest,
    ModelVersionCompareResponse,
    ModelVersionCreateRequest,
    ModelVersionResponse,
    RegisteredModelResponse,
)
from forge.api.services.model_registry import (
    compare_versions,
    get_model_history,
    get_production_model,
    list_models,
    register_model,
    register_version,
    transition_stage,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post(
    "/register",
    response_model=RegisteredModelResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_model(
    request: ModelRegisterRequest,
    db: Session = Depends(get_db),
) -> RegisteredModelResponse:
    """Register a new model name."""
    try:
        model = register_model(db, request.name, request.description)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return RegisteredModelResponse.model_validate(model)


@router.get("", response_model=ModelListResponse)
def list_models_endpoint(
    db: Session = Depends(get_db),
) -> ModelListResponse:
    """List registered models with production version summary."""
    models = list_models(db)
    return ModelListResponse(
        models=[ModelListItem(**m) for m in models],
        count=len(models),
    )


@router.get("/{name}", response_model=ModelDetailResponse)
def get_model_detail(
    name: str,
    db: Session = Depends(get_db),
) -> ModelDetailResponse:
    """Get model detail with all versions and stage history."""
    try:
        model, versions, history = get_model_history(db, name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ModelDetailResponse(
        **RegisteredModelResponse.model_validate(model).model_dump(),
        versions=[ModelVersionResponse.model_validate(v) for v in versions],
        stage_history=[ModelStageHistoryResponse.model_validate(h) for h in history],
    )


@router.post(
    "/{name}/versions",
    response_model=ModelVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_version(
    name: str,
    request: ModelVersionCreateRequest,
    db: Session = Depends(get_db),
) -> ModelVersionResponse:
    """Register a new version from a completed experiment run."""
    try:
        mv = register_version(db, name, request.run_id, request.tags)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return ModelVersionResponse.model_validate(mv)


@router.patch(
    "/{name}/versions/{version}/stage",
    response_model=ModelVersionResponse,
)
def transition_version_stage(
    name: str,
    version: int,
    request: ModelStageTransitionRequest,
    db: Session = Depends(get_db),
) -> ModelVersionResponse:
    """Transition a model version to a new stage."""
    # Look up by model name + version number
    try:
        _model, versions, _ = get_model_history(db, name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    mv = next((v for v in versions if v.version == version), None)
    if mv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version} not found for model '{name}'",
        )

    try:
        updated = transition_stage(db, mv.id, request.stage, request.reason)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return ModelVersionResponse.model_validate(updated)


@router.get("/{name}/compare", response_model=ModelVersionCompareResponse)
def compare_model_versions(
    name: str,
    a: int = Query(..., description="Version number A"),
    b: int = Query(..., description="Version number B"),
    db: Session = Depends(get_db),
) -> ModelVersionCompareResponse:
    """Compare two versions of a model side-by-side."""
    try:
        _model, versions, _ = get_model_history(db, name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    mv_a = next((v for v in versions if v.version == a), None)
    mv_b = next((v for v in versions if v.version == b), None)
    if mv_a is None or mv_b is None:
        missing = a if mv_a is None else b
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {missing} not found for model '{name}'",
        )

    result = compare_versions(db, mv_a.id, mv_b.id)
    return ModelVersionCompareResponse(**result)


@router.get("/{name}/production", response_model=ModelVersionResponse | None)
def get_production(
    name: str,
    db: Session = Depends(get_db),
) -> ModelVersionResponse | None:
    """Get the current production version of a model."""
    prod = get_production_model(db, name)
    if prod is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No production version for model '{name}'",
        )
    return ModelVersionResponse.model_validate(prod)
