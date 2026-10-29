"""SQLAlchemy database engine, session factory, models, and FastAPI dependency."""

import os
import uuid
from collections.abc import Generator

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.sql import func

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://forge:forge@localhost:5432/forge"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """Run a simple query to verify the database is reachable."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True


# ---------------------------------------------------------------------------
# Models — match SPEC.md Section 4 exactly
# ---------------------------------------------------------------------------


class Dataset(Base):
    """Financial datasets ingested from APIs."""

    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    name = Column(String(255), nullable=False)
    source = Column(String(50), nullable=False)
    tickers = Column(ARRAY(Text))
    start_date = Column(Date)
    end_date = Column(Date)
    num_records = Column(Integer)
    feature_columns = Column(ARRAY(Text))
    s3_path = Column(String(500))
    created_at = Column(DateTime, server_default=func.now())


class Experiment(Base):
    """Experiment definitions grouping multiple training runs."""

    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"))
    status = Column(String(20), server_default="pending")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Run(Base):
    """Individual training runs within an experiment."""

    __tablename__ = "runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"))
    run_name = Column(String(255))
    model_type = Column(String(50), nullable=False)
    hyperparameters = Column(JSONB, nullable=False)
    feature_engineering = Column(JSONB)

    feature_set_id = Column(UUID(as_uuid=True), ForeignKey("feature_sets.id"), nullable=True)

    # ML Metrics
    train_loss = Column(Float)
    val_loss = Column(Float)
    test_loss = Column(Float)
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall = Column(Float)
    f1 = Column(Float)
    custom_metrics = Column(JSONB)

    # Hardware-Aware Profiling
    inference_latency_ms = Column(Float)
    inference_latency_p95_ms = Column(Float)
    peak_memory_mb = Column(Float)
    model_size_mb = Column(Float)
    throughput_samples_per_sec = Column(Float)
    flops_estimate = Column(BigInteger)
    training_time_seconds = Column(Float)

    # Deployment tradeoff score
    efficiency_score = Column(Float)

    # Reproducibility
    data_version_hash = Column(String(64))

    wandb_run_id = Column(String(100))
    s3_artifact_path = Column(String(500))
    status = Column(String(20), server_default="pending")
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())


class RunEnvironment(Base):
    """Captured environment snapshot for reproducibility of a training run."""

    __tablename__ = "run_environments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False, unique=True)
    git_sha = Column(String(40))
    git_branch = Column(String(100))
    git_dirty = Column(Boolean, default=False)
    python_version = Column(String(20))
    package_versions = Column(JSONB)
    docker_image_tag = Column(String(255))
    random_seed = Column(Integer)
    env_hash = Column(String(64))
    created_at = Column(DateTime, server_default=func.now())


class FeatureSet(Base):
    """Versioned feature set definitions for reproducible feature engineering."""

    __tablename__ = "feature_sets"
    __table_args__ = (
        Index("uq_feature_set_name_version", "name", "version", unique=True),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    name = Column(String(255), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    description = Column(Text)
    feature_config = Column(JSONB, nullable=False)
    feature_columns = Column(JSONB)
    created_at = Column(DateTime, server_default=func.now())
    is_active = Column(String(5), server_default="true")


class FeatureStoreRegistry(Base):
    """Tracks computed feature sets per dataset."""

    __tablename__ = "feature_store_registry"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    feature_set_id = Column(UUID(as_uuid=True), ForeignKey("feature_sets.id"), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    storage_path = Column(String(500))
    row_count = Column(Integer)
    computed_at = Column(DateTime)
    status = Column(String(20), server_default="pending")


class RegisteredModel(Base):
    """Top-level model registry entry — groups versions under a unique name."""

    __tablename__ = "registered_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ModelVersion(Base):
    """A specific version of a registered model, linked to a training run."""

    __tablename__ = "model_versions"
    __table_args__ = (
        Index("uq_model_version", "model_id", "version", unique=True),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    model_id = Column(UUID(as_uuid=True), ForeignKey("registered_models.id"), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False)
    stage = Column(String(20), server_default="development")
    stage_changed_at = Column(DateTime, server_default=func.now())
    stage_changed_by = Column(String(50), server_default="api")
    s3_artifact_path = Column(String(500))
    model_size_mb = Column(Float)
    metrics_snapshot = Column(JSONB)
    tags = Column(JSONB)
    created_at = Column(DateTime, server_default=func.now())


class ModelStageHistory(Base):
    """Audit trail of model stage transitions."""

    __tablename__ = "model_stage_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    model_version_id = Column(UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=False)
    from_stage = Column(String(20), nullable=False)
    to_stage = Column(String(20), nullable=False)
    changed_at = Column(DateTime, server_default=func.now())
    reason = Column(Text)


class DriftReport(Base):
    """Drift detection reports comparing reference vs current data distributions."""

    __tablename__ = "drift_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    reference_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    report_type = Column(String(30), nullable=False)  # data_drift, prediction_drift, feature_drift
    model_version_id = Column(UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=True)
    overall_drift_score = Column(Float)
    is_drifted = Column(String(5), server_default="false")
    feature_scores = Column(JSONB)
    config = Column(JSONB)
    created_at = Column(DateTime, server_default=func.now())


class ServingBenchmark(Base):
    """Load-test results for a served model: latency/throughput per backend x batching x concurrency."""

    __tablename__ = "serving_benchmarks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    model_name = Column(String(255), nullable=False, index=True)
    model_version = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, server_default="running")  # running, completed, failed
    params = Column(JSONB)  # grid: backends, batch configs, concurrency levels, requests per cell
    runtime = Column(JSONB)  # cpu_count, cpu_limit, torch_threads at run time
    results = Column(JSONB)  # list of per-cell results, appended as cells finish
    error = Column(Text)
    started_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime)


class OpsLog(Base):
    """Ops monitoring: agent/project logs."""

    __tablename__ = "ops_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    project_name = Column(String(100), nullable=False)
    log_level = Column(String(10))
    message = Column(Text)
    metadata_ = Column("metadata", JSONB)
    source = Column(String(100))
    cost_usd = Column(Float)
    created_at = Column(DateTime, server_default=func.now())


class GitEvent(Base):
    """Git events from webhooks."""

    __tablename__ = "git_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    repo = Column(String(255), nullable=False)
    event_type = Column(String(50))
    branch = Column(String(100))
    commit_sha = Column(String(40))
    commit_message = Column(Text)
    author = Column(String(100))
    files_changed = Column(Integer)
    additions = Column(Integer)
    deletions = Column(Integer)
    payload = Column(JSONB)
    created_at = Column(DateTime, server_default=func.now())


class ExperimentEmbedding(Base):
    """pgvector embeddings for semantic search over experiments."""

    __tablename__ = "experiment_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"))
    content_type = Column(String(50))
    content_text = Column(Text)
    embedding = Column(Vector(1536))
    created_at = Column(DateTime, server_default=func.now())


# IVFFlat index for vector similarity search
ix_experiment_embeddings_embedding = Index(
    "ix_experiment_embeddings_embedding",
    ExperimentEmbedding.embedding,
    postgresql_using="ivfflat",
    postgresql_with={"lists": 100},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)
