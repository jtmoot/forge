"""Prometheus custom metrics for Forge platform.

Defines all forge_* metrics used for observability dashboards.
Metrics are registered globally via prometheus_client and updated
by services throughout the codebase.
"""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Experiment metrics
# ---------------------------------------------------------------------------

EXPERIMENTS_TOTAL = Counter(
    "forge_experiments_total",
    "Total experiment runs",
    ["model_type", "status"],
)

TRAINING_DURATION_SECONDS = Histogram(
    "forge_training_duration_seconds",
    "Training time per run",
    ["model_type"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "forge_inference_latency_seconds",
    "Inference latency from profiler",
    ["model_type"],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)

# ---------------------------------------------------------------------------
# Data ingestion metrics
# ---------------------------------------------------------------------------

DATA_INGESTION_ROWS_TOTAL = Counter(
    "forge_data_ingestion_rows_total",
    "Total rows ingested",
    ["ticker"],
)

# ---------------------------------------------------------------------------
# Drift metrics
# ---------------------------------------------------------------------------

DRIFT_SCORE = Gauge(
    "forge_drift_score",
    "Latest drift score",
    ["dataset", "feature"],
)

# ---------------------------------------------------------------------------
# Model registry metrics
# ---------------------------------------------------------------------------

MODEL_REGISTRY_VERSIONS = Gauge(
    "forge_model_registry_versions",
    "Count of model versions per stage",
    ["stage"],
)

# ---------------------------------------------------------------------------
# Serving metrics
# ---------------------------------------------------------------------------

SERVING_REQUEST_SECONDS = Histogram(
    "forge_serving_request_seconds",
    "Model inference time per serving request",
    ["model", "version", "backend"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

SERVING_MODEL_LOADS_TOTAL = Counter(
    "forge_serving_model_loads_total",
    "Model loads into the serving cache",
    ["model", "status"],
)

SERVING_CACHE_HITS_TOTAL = Counter(
    "forge_serving_cache_hits_total",
    "Serving requests answered by an already-loaded model",
    ["model"],
)

# ---------------------------------------------------------------------------
# Ops metrics
# ---------------------------------------------------------------------------

OPS_ERRORS_TOTAL = Counter(
    "forge_ops_errors_total",
    "Ops log errors",
    ["project"],
)

# ---------------------------------------------------------------------------
# LLM cost metrics
# ---------------------------------------------------------------------------

LLM_COST_DOLLARS = Counter(
    "forge_llm_cost_dollars",
    "Cumulative LLM/agent cost",
)
