"""Pydantic models for request/response validation."""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""

    status: str


# ---------------------------------------------------------------------------
# Dataset schemas
# ---------------------------------------------------------------------------


class DatasetIngestRequest(BaseModel):
    """Request body for POST /api/datasets/ingest."""

    name: str = Field(..., min_length=1, max_length=255)
    source: str = Field(default="yfinance", pattern="^(yfinance|fred|csv_upload)$")
    tickers: list[str] = Field(..., min_length=1)
    start_date: date
    end_date: date


class DatasetResponse(BaseModel):
    """Response model for a single dataset."""

    id: UUID
    name: str
    source: str
    tickers: list[str] | None
    start_date: date | None
    end_date: date | None
    num_records: int | None
    feature_columns: list[str] | None
    created_at: datetime | None

    model_config = {"from_attributes": True}


class DatasetDetailResponse(DatasetResponse):
    """Dataset response with feature summary statistics."""

    feature_summary: dict | None = None


class DatasetListResponse(BaseModel):
    """Response model for listing datasets."""

    datasets: list[DatasetResponse]
    count: int


# ---------------------------------------------------------------------------
# Experiment / Run schemas
# ---------------------------------------------------------------------------


class RunConfigRequest(BaseModel):
    """Configuration for a single training run within an experiment."""

    run_name: str | None = None
    model_type: str = Field(..., pattern="^(xgboost|random_forest|lstm|tcn|cnn_lstm|transformer)$")
    hyperparameters: dict = Field(default_factory=dict)
    feature_engineering: dict | None = None
    feature_set_id: UUID | None = None


class ExperimentCreateRequest(BaseModel):
    """Request body for POST /api/experiments."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    dataset_id: UUID
    runs: list[RunConfigRequest] = Field(..., min_length=1)


class RunResponse(BaseModel):
    """Response model for a single run with metrics and profiling."""

    id: UUID
    experiment_id: UUID
    run_name: str | None
    model_type: str
    hyperparameters: dict
    feature_engineering: dict | None

    feature_set_id: str | None = None

    # ML metrics
    accuracy: float | None = None
    precision_score: float | None = None
    recall: float | None = None
    f1: float | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    test_loss: float | None = None
    custom_metrics: dict | None = None

    # Hardware profiling
    inference_latency_ms: float | None = None
    inference_latency_p95_ms: float | None = None
    peak_memory_mb: float | None = None
    model_size_mb: float | None = None
    throughput_samples_per_sec: float | None = None
    training_time_seconds: float | None = None
    efficiency_score: float | None = None

    # Reproducibility
    data_version_hash: str | None = None

    # External integrations
    wandb_run_id: str | None = None
    s3_artifact_path: str | None = None

    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class ExperimentResponse(BaseModel):
    """Response model for an experiment (without runs)."""

    id: UUID
    name: str
    description: str | None
    dataset_id: UUID | None
    status: str
    created_at: datetime | None
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class ExperimentDetailResponse(ExperimentResponse):
    """Experiment response with all associated runs."""

    runs: list[RunResponse] = []


class ExperimentListResponse(BaseModel):
    """Response model for listing experiments."""

    experiments: list[ExperimentResponse]
    count: int


# ---------------------------------------------------------------------------
# Ops Log schemas
# ---------------------------------------------------------------------------


class OpsLogCreateRequest(BaseModel):
    """Request body for POST /api/ops/logs."""

    project_name: str = Field(..., min_length=1, max_length=100)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARN|ERROR|CRITICAL)$")
    message: str = Field(..., min_length=1)
    metadata: dict | None = None
    source: str | None = None
    cost_usd: float | None = None


class OpsLogResponse(BaseModel):
    """Response model for a single ops log entry."""

    id: UUID
    project_name: str
    log_level: str | None
    message: str | None
    metadata: dict | None = None
    source: str | None = None
    cost_usd: float | None = None
    is_anomaly: bool = False
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class OpsLogListResponse(BaseModel):
    """Response model for listing ops logs."""

    logs: list[OpsLogResponse]
    count: int


class OpsLogSummaryResponse(BaseModel):
    """Aggregate stats for ops logs."""

    total_logs: int
    error_count: int
    total_cost_usd: float
    events_by_project: dict[str, int]
    events_by_level: dict[str, int]


# ---------------------------------------------------------------------------
# GitHub Webhook schemas
# ---------------------------------------------------------------------------


class GitHubCommit(BaseModel):
    """A single commit within a GitHub push event."""

    id: str
    message: str | None = None
    author: dict | None = None
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    modified: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class GitHubPushPayload(BaseModel):
    """GitHub push webhook payload (subset of fields we care about)."""

    ref: str
    repository: dict | None = None
    commits: list[GitHubCommit] = Field(default_factory=list)
    pusher: dict | None = None

    model_config = {"extra": "allow"}


class GitEventResponse(BaseModel):
    """Response model for a stored git event."""

    id: UUID
    repo: str
    event_type: str | None
    branch: str | None
    commit_sha: str | None
    commit_message: str | None
    author: str | None
    files_changed: int | None
    additions: int | None
    deletions: int | None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class GitEventListResponse(BaseModel):
    """Response model for listing git events."""

    events: list[GitEventResponse]
    count: int


# ---------------------------------------------------------------------------
# Agent / Analysis schemas
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Project / Activity Feed schemas
# ---------------------------------------------------------------------------


class ProjectSummary(BaseModel):
    """Summary stats for a single project (virtual aggregation over ops_logs + git_events)."""

    name: str
    commit_count_7d: int = 0
    total_cost_7d: float = 0.0
    error_count_7d: int = 0
    last_activity: datetime | None = None
    health: str = "green"  # green / yellow / red


class ProjectListResponse(BaseModel):
    """Response model for GET /api/projects."""

    projects: list[ProjectSummary]
    count: int


class ProjectDetailResponse(BaseModel):
    """Detailed view of a single project with recent logs, git events, and linked experiments."""

    name: str
    recent_logs: list["OpsLogResponse"]
    git_events: list["GitEventResponse"]
    linked_experiments: list["ExperimentResponse"]


class ActivityFeedItem(BaseModel):
    """A single item in the interleaved activity timeline."""

    type: str  # "git_commit" | "ops_log" | "experiment_completion"
    timestamp: datetime
    project: str | None = None
    summary: str
    detail: dict | None = None


class ActivityFeedResponse(BaseModel):
    """Response model for GET /api/activity/feed."""

    items: list[ActivityFeedItem]
    count: int


class DashboardSummaryResponse(BaseModel):
    """Aggregate stats for the dashboard home page."""

    total_projects: int
    active_experiments: int
    ops_alerts_24h: int
    weekly_llm_cost: float
    drift_alerts_7d: int = 0


# ---------------------------------------------------------------------------
# Agent / Analysis schemas
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Feature Store schemas
# ---------------------------------------------------------------------------


class FeatureSetCreateRequest(BaseModel):
    """Request body for POST /api/features/register."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    feature_config: dict = Field(..., description="Declarative feature engineering config")


class FeatureComputeRequest(BaseModel):
    """Request body for POST /api/features/{id}/compute."""

    dataset_id: UUID


class FeatureRegistryResponse(BaseModel):
    """Response model for a feature store registry entry."""

    id: UUID
    feature_set_id: UUID
    dataset_id: UUID
    storage_path: str | None = None
    row_count: int | None = None
    computed_at: datetime | None = None
    status: str

    model_config = {"from_attributes": True}


class FeatureSetResponse(BaseModel):
    """Response model for a single feature set."""

    id: UUID
    name: str
    version: int
    description: str | None = None
    feature_config: dict
    feature_columns: list[str] | None = None
    created_at: datetime | None = None
    is_active: str | None = None

    model_config = {"from_attributes": True}


class FeatureSetDetailResponse(FeatureSetResponse):
    """Feature set with registry entries showing which datasets it's been computed for."""

    registry_entries: list[FeatureRegistryResponse] = []


class FeatureSetListResponse(BaseModel):
    """Response model for listing feature sets."""

    feature_sets: list[FeatureSetResponse]
    count: int


class FeatureSetCompareResponse(BaseModel):
    """Diff between two feature sets."""

    feature_set_a: dict
    feature_set_b: dict
    columns_added: list[str]
    columns_removed: list[str]
    config_added: dict
    config_removed: dict
    config_changed: dict


# ---------------------------------------------------------------------------
# Model Registry schemas
# ---------------------------------------------------------------------------


class ModelRegisterRequest(BaseModel):
    """Request body for POST /api/models/register."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class ModelVersionCreateRequest(BaseModel):
    """Request body for POST /api/models/{name}/versions."""

    run_id: UUID
    tags: dict | None = None


class ModelStageTransitionRequest(BaseModel):
    """Request body for PATCH /api/models/{name}/versions/{version}/stage."""

    stage: str = Field(..., pattern="^(staging|production|archived)$")
    reason: str | None = None


class ModelStageHistoryResponse(BaseModel):
    """Response model for a stage transition history entry."""

    id: UUID
    model_version_id: UUID
    from_stage: str
    to_stage: str
    changed_at: datetime | None = None
    reason: str | None = None

    model_config = {"from_attributes": True}


class ModelVersionResponse(BaseModel):
    """Response model for a single model version."""

    id: UUID
    model_id: UUID
    version: int
    run_id: UUID
    stage: str
    stage_changed_at: datetime | None = None
    stage_changed_by: str | None = None
    s3_artifact_path: str | None = None
    model_size_mb: float | None = None
    metrics_snapshot: dict | None = None
    tags: dict | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class RegisteredModelResponse(BaseModel):
    """Response model for a registered model."""

    id: UUID
    name: str
    description: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class ModelDetailResponse(RegisteredModelResponse):
    """Model with all versions and stage history."""

    versions: list[ModelVersionResponse] = []
    stage_history: list[ModelStageHistoryResponse] = []


class ModelListItem(BaseModel):
    """Summary of a registered model for listing."""

    id: str
    name: str
    description: str | None = None
    version_count: int
    production_version: int | None = None
    production_accuracy: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ModelListResponse(BaseModel):
    """Response model for listing registered models."""

    models: list[ModelListItem]
    count: int


class ModelVersionCompareResponse(BaseModel):
    """Side-by-side version comparison with deltas."""

    version_a: dict
    version_b: dict
    metrics: dict


# ---------------------------------------------------------------------------
# Drift Detection schemas
# ---------------------------------------------------------------------------


class DriftDetectRequest(BaseModel):
    """Request body for POST /api/drift/detect."""

    reference_dataset_id: UUID
    current_dataset_id: UUID
    report_type: str = Field(..., pattern="^(data_drift|prediction_drift|feature_drift)$")
    model_version_id: UUID | None = None
    features: list[str] | None = None
    threshold: float | None = None


class DriftReportResponse(BaseModel):
    """Response model for a drift report."""

    id: UUID
    dataset_id: UUID
    reference_dataset_id: UUID
    report_type: str
    model_version_id: UUID | None = None
    overall_drift_score: float | None = None
    is_drifted: str | None = None
    feature_scores: dict | None = None
    config: dict | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class DriftReportListResponse(BaseModel):
    """Response model for listing drift reports."""

    reports: list[DriftReportResponse]
    count: int


class DriftSummaryResponse(BaseModel):
    """High-level drift summary."""

    total_reports: int
    drifted_count: int
    datasets_with_drift: int
    by_type: dict[str, int]
    last_check: str | None = None
    days: int


# ---------------------------------------------------------------------------
# Reproducibility schemas
# ---------------------------------------------------------------------------


class RunEnvironmentResponse(BaseModel):
    """Response model for a run's captured environment snapshot."""

    id: UUID
    run_id: UUID
    git_sha: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    python_version: str | None = None
    package_versions: dict[str, str] | None = None
    docker_image_tag: str | None = None
    random_seed: int | None = None
    env_hash: str | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class ReproduceResponse(BaseModel):
    """Reproduction specification for a training run."""

    git_sha: str | None = None
    command: str
    data_version: str | None = None
    feature_set: str | None = None
    environment_hash: str | None = None
    random_seed: int | None = None
    warnings: list[str] = Field(default_factory=list)


class ReproducibilityReport(BaseModel):
    """Report comparing reproducibility factors between two runs."""

    verdict: str
    factors: dict[str, dict]
    warnings: list[str] = Field(default_factory=list)


class EnvironmentDiffResponse(BaseModel):
    """Diff between two run environments."""

    run_a: RunEnvironmentResponse | None = None
    run_b: RunEnvironmentResponse | None = None
    packages_added: dict[str, str] = Field(default_factory=dict)
    packages_removed: dict[str, str] = Field(default_factory=dict)
    packages_changed: dict[str, dict] = Field(default_factory=dict)
    field_diffs: dict[str, dict] = Field(default_factory=dict)
    environments_identical: bool = False
    reproducibility: ReproducibilityReport | None = None


class AgentQueryRequest(BaseModel):
    """Request body for POST /api/agent/query."""

    question: str = Field(..., min_length=1, max_length=2000)


class IntermediateResult(BaseModel):
    """A single tool call result from the agent's reasoning chain."""

    tool: str
    result_preview: str


class AgentQueryResponse(BaseModel):
    """Response from the analysis agent including reasoning transparency."""

    answer: str
    tools_used: list[str] = Field(default_factory=list)
    intermediate_results: list[IntermediateResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Serving schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for POST /api/serve/{model_name}/predict.

    Tabular models take rows of features ``[batch, n_features]``; sequence
    models take windows ``[batch, window_size, n_features]``.
    """

    inputs: list[list[float] | list[list[float]]] = Field(..., min_length=1, max_length=1024)


class Prediction(BaseModel):
    """One model output: positive-class probability and thresholded label."""

    probability: float
    label: int


class PredictResponse(BaseModel):
    """Predictions from the model's current production version.

    ``backend`` is the backend that actually ran the request.
    """

    model_name: str
    version: int
    model_type: str
    backend: str
    predictions: list[Prediction]
    latency_ms: float


class BatchConfigRequest(BaseModel):
    """Request body for PUT /api/serve/{model_name}/batching. max_batch_size=1 disables batching."""

    max_batch_size: int = Field(..., ge=1, le=1024)
    max_wait_ms: float = Field(..., ge=0, le=1000)


class BatchConfigResponse(BaseModel):
    """A model's micro-batching policy."""

    model_name: str
    max_batch_size: int
    max_wait_ms: float
    enabled: bool


class BackendParity(BaseModel):
    """Parity of an ONNX backend against the model's native backend.

    ``skipped`` means the backend couldn't be built (``reason`` says why);
    otherwise ``passed`` reports whether it met the gate and is served.
    """

    backend: str
    passed: bool
    skipped: bool
    max_abs_diff: float | None = None
    label_agreement: float | None = None
    max_abs_diff_limit: float | None = None
    min_label_agreement: float | None = None
    samples: int
    reason: str | None = None


class ServedModelResponse(BaseModel):
    """A model loaded into the serving cache.

    ``backend`` is the native (default) backend; ``backends`` lists every
    backend a predict request can select.
    """

    model_name: str
    version: int
    run_id: UUID
    model_type: str
    backend: str
    backends: list[str]
    parity: list[BackendParity]
    input_shape: list[int]
    feature_names: list[str]
    loaded_at: datetime
    load_seconds: float
    onnx_build_seconds: float
    batching: BatchConfigResponse


class ServingRuntime(BaseModel):
    """Compute the API process runs on."""

    cpu_count: int
    cpu_limit: float | None
    torch_threads: int


class ServedModelListResponse(BaseModel):
    """All models currently loaded for serving, plus the runtime serving them."""

    models: list[ServedModelResponse]
    runtime: ServingRuntime
