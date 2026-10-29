"""LangChain tool implementations for the Forge analysis agent.

Five tools that query experiments, runs, ops logs, and embeddings
in the PostgreSQL database and return formatted results.
"""

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from uuid import UUID

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from forge.api.models.database import Experiment, OpsLog, Run, ServingBenchmark
from forge.api.services.embeddings import search_similar_runs
from forge.api.services.model_server import model_cache
from forge.api.services.serving_bench import recommend_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _format_run_row(run: Run) -> dict:
    """Convert a Run ORM object into a serialisable dict of key fields."""
    return {
        "run_id": str(run.id),
        "experiment_id": str(run.experiment_id),
        "run_name": run.run_name,
        "model_type": run.model_type,
        "accuracy": run.accuracy,
        "f1": run.f1,
        "precision": run.precision_score,
        "recall": run.recall,
        "inference_latency_ms": run.inference_latency_ms,
        "inference_latency_p95_ms": run.inference_latency_p95_ms,
        "peak_memory_mb": run.peak_memory_mb,
        "model_size_mb": run.model_size_mb,
        "throughput_samples_per_sec": run.throughput_samples_per_sec,
        "efficiency_score": run.efficiency_score,
        "training_time_seconds": run.training_time_seconds,
        "status": run.status,
        "hyperparameters": run.hyperparameters,
    }


# ---------------------------------------------------------------------------
# Tool factories — each returns a closure bound to a db session
# ---------------------------------------------------------------------------


def make_query_experiments_tool(db: Session):
    """Create the query_experiments tool bound to a DB session."""

    @tool
    def query_experiments(query: str) -> str:
        """Search experiments and their runs. The query can be a model type
        (e.g. 'xgboost'), experiment name keyword, or 'all' to list everything.
        Returns experiment names with their runs and key metrics."""
        experiments = db.query(Experiment).all()

        results = []
        for exp in experiments:
            if query.lower() not in ("all", "") and query.lower() not in (exp.name or "").lower():
                # Check if query matches any run model_type in this experiment
                runs = db.query(Run).filter(Run.experiment_id == exp.id).all()
                matching_runs = [r for r in runs if query.lower() in (r.model_type or "").lower()]
                if not matching_runs:
                    continue
            else:
                runs = db.query(Run).filter(Run.experiment_id == exp.id).all()
                matching_runs = runs

            exp_info = {
                "experiment_id": str(exp.id),
                "name": exp.name,
                "description": exp.description,
                "status": exp.status,
                "runs": [_format_run_row(r) for r in matching_runs],
            }
            results.append(exp_info)

        if not results:
            return f"No experiments found matching '{query}'."

        return json.dumps(results, indent=2, default=str)

    return query_experiments


def make_compare_runs_tool(db: Session):
    """Create the compare_runs tool bound to a DB session."""

    @tool
    def compare_runs(run_id_1: str, run_id_2: str) -> str:
        """Compare two runs side-by-side. Takes two run UUIDs and returns
        a detailed comparison of their metrics, profiling, and configuration."""
        try:
            run1 = db.query(Run).filter(Run.id == UUID(run_id_1)).first()
            run2 = db.query(Run).filter(Run.id == UUID(run_id_2)).first()
        except ValueError:
            return "Error: invalid run ID format. Provide valid UUIDs."

        if run1 is None:
            return f"Error: run {run_id_1} not found."
        if run2 is None:
            return f"Error: run {run_id_2} not found."

        r1 = _format_run_row(run1)
        r2 = _format_run_row(run2)

        comparison = {
            "run_1": r1,
            "run_2": r2,
            "differences": {},
        }

        # Highlight metric differences
        metric_keys = [
            "accuracy", "f1", "precision", "recall",
            "inference_latency_ms", "peak_memory_mb",
            "efficiency_score", "throughput_samples_per_sec",
        ]
        for key in metric_keys:
            v1 = r1.get(key)
            v2 = r2.get(key)
            if v1 is not None and v2 is not None:
                diff = v1 - v2
                winner = "run_1" if (diff > 0 if key != "inference_latency_ms" else diff < 0) else "run_2"
                comparison["differences"][key] = {
                    "run_1": v1,
                    "run_2": v2,
                    "delta": round(diff, 6),
                    "better": winner,
                }

        return json.dumps(comparison, indent=2, default=str)

    return compare_runs


def make_search_similar_tool(db: Session):
    """Create the search_similar tool bound to a DB session."""

    @tool
    def search_similar(query: str) -> str:
        """Semantic search over experiment run summaries using pgvector embeddings.
        Returns the most similar runs to the natural language query."""
        results = search_similar_runs(query, db, limit=5)
        if not results:
            return "No embeddings found. Embeddings may not have been generated yet (requires OPENAI_API_KEY)."
        return json.dumps(results, indent=2, default=str)

    return search_similar


def make_get_ops_summary_tool(db: Session):
    """Create the get_ops_summary tool bound to a DB session."""

    @tool
    def get_ops_summary(hours: int = 24) -> str:
        """Summarize ops activity for the given time period (default: last 24 hours).
        Returns log counts by project and level, total cost, and any anomalies."""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)

        logs = db.query(OpsLog).filter(OpsLog.created_at >= cutoff).all()

        if not logs:
            return f"No ops logs found in the last {hours} hours."

        total_cost = sum(log.cost_usd or 0 for log in logs)
        by_project: dict[str, int] = {}
        by_level: dict[str, int] = {}
        errors = []

        for log in logs:
            by_project[log.project_name] = by_project.get(log.project_name, 0) + 1
            level = log.log_level or "UNKNOWN"
            by_level[level] = by_level.get(level, 0) + 1
            if level in ("ERROR", "CRITICAL"):
                errors.append({
                    "project": log.project_name,
                    "message": log.message,
                    "cost_usd": log.cost_usd,
                    "created_at": str(log.created_at),
                })

        # Simple anomaly detection on costs
        costs = [log.cost_usd for log in logs if log.cost_usd is not None]
        anomalous_costs = []
        if len(costs) > 5:
            import numpy as np
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            if std_cost > 0:
                for log in logs:
                    if log.cost_usd is not None:
                        zscore = abs((log.cost_usd - mean_cost) / std_cost)
                        if zscore > 2.5:
                            anomalous_costs.append({
                                "project": log.project_name,
                                "cost_usd": log.cost_usd,
                                "z_score": round(zscore, 2),
                                "message": log.message,
                            })

        summary = {
            "time_period_hours": hours,
            "total_logs": len(logs),
            "total_cost_usd": round(total_cost, 4),
            "by_project": by_project,
            "by_level": by_level,
            "error_details": errors[:10],
            "cost_anomalies": anomalous_costs,
        }

        return json.dumps(summary, indent=2, default=str)

    return get_ops_summary


def make_compute_efficiency_frontier_tool(db: Session):
    """Create the compute_efficiency_frontier tool bound to a DB session."""

    @tool
    def compute_efficiency_frontier(top_n: int = 10) -> str:
        """Find Pareto-optimal runs on the accuracy vs. latency frontier.
        A run is Pareto-optimal if no other run is both more accurate AND
        faster. Returns the frontier runs sorted by accuracy descending."""
        runs = (
            db.query(Run)
            .filter(
                Run.status == "completed",
                Run.accuracy.isnot(None),
                Run.inference_latency_ms.isnot(None),
            )
            .all()
        )

        if not runs:
            return "No completed runs with both accuracy and latency data found."

        # Build Pareto frontier: keep runs where no other run dominates
        run_data = [
            {
                "run_id": str(r.id),
                "run_name": r.run_name,
                "model_type": r.model_type,
                "accuracy": r.accuracy,
                "inference_latency_ms": r.inference_latency_ms,
                "efficiency_score": r.efficiency_score,
                "peak_memory_mb": r.peak_memory_mb,
            }
            for r in runs
        ]

        frontier = []
        for candidate in run_data:
            dominated = False
            for other in run_data:
                if other is candidate:
                    continue
                # other dominates candidate if it's better or equal on both axes
                # and strictly better on at least one
                better_acc = other["accuracy"] >= candidate["accuracy"]
                better_lat = other["inference_latency_ms"] <= candidate["inference_latency_ms"]
                strictly_better = (
                    other["accuracy"] > candidate["accuracy"]
                    or other["inference_latency_ms"] < candidate["inference_latency_ms"]
                )
                if better_acc and better_lat and strictly_better:
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate)

        frontier.sort(key=lambda r: r["accuracy"], reverse=True)
        frontier = frontier[:top_n]

        result = {
            "total_completed_runs": len(run_data),
            "frontier_size": len(frontier),
            "pareto_frontier": frontier,
        }

        return json.dumps(result, indent=2, default=str)

    return compute_efficiency_frontier


def make_query_model_registry_tool(db: Session):
    """Create the query_model_registry tool bound to a DB session."""

    @tool
    def query_model_registry(query: str) -> str:
        """Query the model registry for registered models, versions, and stages.
        Use for questions like 'what's in production?', 'compare model versions',
        'what models are ready for promotion?', or 'show model history'.
        The query can be a model name, 'all' to list everything, or 'production'
        to show only production models."""
        from forge.api.models.database import ModelVersion, RegisteredModel
        from forge.api.services.model_registry import get_model_history, list_models

        if query.lower() in ("all", "list", ""):
            models = list_models(db)
            if not models:
                return "No models registered in the registry."
            return json.dumps(models, indent=2, default=str)

        if query.lower() == "production":
            models = list_models(db)
            prod_models = [m for m in models if m["production_version"] is not None]
            if not prod_models:
                return "No models currently in production."
            return json.dumps(prod_models, indent=2, default=str)

        if query.lower() in ("staging", "ready for promotion"):
            versions = (
                db.query(ModelVersion)
                .filter(ModelVersion.stage == "staging")
                .all()
            )
            if not versions:
                return "No models currently in staging."
            result = []
            for v in versions:
                model = db.query(RegisteredModel).filter(RegisteredModel.id == v.model_id).first()
                result.append({
                    "model_name": model.name if model else "unknown",
                    "version": v.version,
                    "stage": v.stage,
                    "metrics": v.metrics_snapshot,
                })
            return json.dumps(result, indent=2, default=str)

        # Try as model name
        try:
            model, versions, history = get_model_history(db, query)
            result = {
                "model_name": model.name,
                "description": model.description,
                "versions": [
                    {
                        "version": v.version,
                        "stage": v.stage,
                        "metrics": v.metrics_snapshot,
                        "run_id": str(v.run_id),
                        "created_at": str(v.created_at),
                    }
                    for v in versions
                ],
                "recent_transitions": [
                    {
                        "version_id": str(h.model_version_id),
                        "from": h.from_stage,
                        "to": h.to_stage,
                        "reason": h.reason,
                        "at": str(h.changed_at),
                    }
                    for h in history[:10]
                ],
            }
            return json.dumps(result, indent=2, default=str)
        except ValueError:
            return f"No model found with name '{query}'. Use 'all' to list registered models."

    return query_model_registry


def make_check_drift_tool(db: Session):
    """Create the check_drift tool bound to a DB session."""

    @tool
    def check_drift(query: str) -> str:
        """Check for data or model drift. Use for questions like 'is SPY data drifting?',
        'which features are most unstable?', or 'should I retrain the production model?'.
        Pass a dataset name, 'summary', or 'all' to get drift reports."""
        from forge.api.models.database import Dataset
        from forge.api.services.drift_detection import (
            get_drift_summary,
            list_drift_reports,
        )

        if query.lower() in ("summary", "all", ""):
            summary = get_drift_summary(db)
            if summary["total_reports"] == 0:
                return "No drift reports found. Run drift detection first via POST /api/drift/detect."
            return json.dumps(summary, indent=2, default=str)

        # Try finding reports for a dataset by name
        datasets = db.query(Dataset).filter(Dataset.name.ilike(f"%{query}%")).all()
        if not datasets:
            return f"No datasets found matching '{query}'. Try 'summary' for an overview."

        results = []
        for ds in datasets:
            reports = list_drift_reports(db, dataset_id=ds.id, limit=5)
            for r in reports:
                feature_scores = r.feature_scores or {}
                # Find top drifted features
                top_drifted = feature_scores.get("top_drifted", [])
                if not top_drifted and isinstance(feature_scores, dict):
                    # For data_drift reports, extract from per-feature scores
                    per_feat = {k: v for k, v in feature_scores.items() if isinstance(v, dict) and v.get("is_drifted")}
                    top_drifted = [{"feature": k, "p_value": v.get("p_value")} for k, v in sorted(per_feat.items(), key=lambda x: x[1].get("p_value", 1))[:3]]

                results.append({
                    "dataset": ds.name,
                    "report_type": r.report_type,
                    "overall_score": r.overall_drift_score,
                    "is_drifted": r.is_drifted,
                    "top_drifted_features": top_drifted,
                    "created_at": str(r.created_at),
                })

        if not results:
            return f"No drift reports found for datasets matching '{query}'. Run drift detection first."

        return json.dumps(results, indent=2, default=str)

    return check_drift


_NATIVE_BACKENDS = {"pytorch", "xgboost", "sklearn"}
_CELL_FIELDS = ("backend", "max_batch_size", "max_wait_ms", "concurrency", "p50_ms", "p99_ms", "throughput_rps")


def _brief_cell(cell: dict | None) -> dict | None:
    """Keep the fields an answer needs from a benchmark cell."""
    if cell is None:
        return None
    return {field: cell.get(field) for field in _CELL_FIELDS}


def _recommended_parity(best: dict | None, parity: dict | None) -> dict | None:
    """ONNX parity (accuracy vs the native model) for the recommended backend."""
    if best is None:
        return None
    if best["backend"] in _NATIVE_BACKENDS:
        return {"backend": best["backend"], "native": True}
    return (parity or {}).get(best["backend"])


def summarize_serving_benchmark(
    benchmark: ServingBenchmark, slo_p99_ms: float, parity: dict | None = None
) -> dict:
    """Summarize a benchmark against a p99 SLO, with a naive-serving baseline for context.

    The baseline is the native backend with batching off at the recommended
    config's concurrency, so the answer can say how much the optimizations buy.
    Parity comes from the benchmark's params, else from ``parity`` (the loaded model's).
    """
    cells = benchmark.results or []
    recommendation = recommend_config(cells, slo_p99_ms, top=3)
    best = recommendation["best"]

    baseline = None
    if best is not None:
        baseline = next(
            (
                c for c in cells
                if c["backend"] in _NATIVE_BACKENDS
                and c["max_batch_size"] == 1
                and c["concurrency"] == best["concurrency"]
            ),
            None,
        )
    speedup = (
        round(best["throughput_rps"] / baseline["throughput_rps"], 2)
        if best is not None and baseline is not None and baseline["throughput_rps"] > 0
        else None
    )
    return {
        "model_name": benchmark.model_name,
        "model_version": benchmark.model_version,
        "benchmark_id": str(benchmark.id),
        "benchmarked_at": benchmark.completed_at,
        "runtime": benchmark.runtime,
        "latency_scope": (benchmark.params or {}).get("latency_scope"),
        "slo_p99_ms": slo_p99_ms,
        "meets_slo": recommendation["meets_slo"],
        "recommended": _brief_cell(best),
        "other_candidates": [_brief_cell(c) for c in recommendation["candidates"][1:]],
        "closest": _brief_cell(recommendation["lowest_p99"]) if best is None else None,
        "baseline": _brief_cell(baseline),
        "throughput_vs_baseline": speedup,
        "cells_considered": recommendation["cells_considered"],
        "recommended_parity": _recommended_parity(
            best, (benchmark.params or {}).get("parity") or parity
        ),
    }


def make_find_serving_config_tool(db: Session):
    """Create the find_serving_config tool bound to a DB session."""

    @tool
    def find_serving_config(slo_p99_ms: float, model_name: str = "") -> str:
        """Find the serving config (backend, micro-batching, concurrency) with the highest
        throughput whose p99 latency meets an SLO, from the latest load-test benchmark.

        Args:
            slo_p99_ms: p99 latency target in milliseconds (e.g. 20).
            model_name: Served model name; empty means the most recently benchmarked model.
        """
        if slo_p99_ms <= 0:
            return json.dumps({"error": "slo_p99_ms must be positive"})
        query = db.query(ServingBenchmark).filter(ServingBenchmark.status == "completed")
        if model_name:
            query = query.filter(ServingBenchmark.model_name == model_name)
        benchmark = query.order_by(ServingBenchmark.completed_at.desc()).first()
        if benchmark is None:
            target = f"'{model_name}'" if model_name else "any model"
            return json.dumps({
                "error": f"No completed serving benchmark for {target}",
                "hint": "Run POST /api/serve/{model_name}/benchmarks first",
            })
        loaded = next(
            (
                s for s in model_cache.loaded()
                if s.name == benchmark.model_name and s.version == benchmark.model_version
            ),
            None,
        )
        loaded_parity = {name: asdict(result) for name, result in loaded.parity.items()} if loaded else None
        return json.dumps(
            summarize_serving_benchmark(benchmark, slo_p99_ms, loaded_parity), indent=2, default=str
        )

    return find_serving_config


def build_tools(db: Session) -> list:
    """Build all agent tools bound to the given database session."""
    return [
        make_query_experiments_tool(db),
        make_compare_runs_tool(db),
        make_search_similar_tool(db),
        make_get_ops_summary_tool(db),
        make_compute_efficiency_frontier_tool(db),
        make_query_model_registry_tool(db),
        make_check_drift_tool(db),
        make_find_serving_config_tool(db),
    ]
