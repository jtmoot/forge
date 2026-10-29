"""Metrics summary endpoint for the frontend dashboard mini-card."""

from fastapi import APIRouter
from prometheus_client import REGISTRY

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


def _get_sample_value(metric_name: str, labels: dict | None = None) -> float:
    """Read the current value of a Prometheus metric from the registry.

    For counters and gauges, returns the _total or current value.
    Returns 0.0 if the metric is not found.
    """
    for metric in REGISTRY.collect():
        if metric.name == metric_name or metric.name == f"{metric_name}_total":
            for sample in metric.samples:
                if labels is None:
                    return sample.value
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return sample.value
    return 0.0


def _sum_metric(metric_name: str) -> float:
    """Sum all label combinations for a given metric."""
    total = 0.0
    for metric in REGISTRY.collect():
        if metric.name == metric_name or metric.name == f"{metric_name}_total":
            for sample in metric.samples:
                if sample.name.endswith("_total") or sample.name == metric_name:
                    total += sample.value
    return total


@router.get("/summary")
async def metrics_summary() -> dict:
    """Return current request rate and error rate for the dashboard card.

    Reads from the prometheus-fastapi-instrumentator metrics that are
    automatically collected for all endpoints.
    """
    total_requests = 0.0
    total_errors = 0.0

    for metric in REGISTRY.collect():
        # The instrumentator uses http_requests_total by default
        if metric.name in ("http_requests", "http_requests_total"):
            for sample in metric.samples:
                if sample.name.endswith("_total") or sample.name == "http_requests":
                    count = sample.value
                    total_requests += count
                    status = sample.labels.get("status", "")
                    if status.startswith(("4", "5")):
                        total_errors += count

    error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0

    return {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate_pct": round(error_rate, 2),
        "experiments_total": _sum_metric("forge_experiments_total"),
        "llm_cost_dollars": _get_sample_value("forge_llm_cost_dollars"),
    }
