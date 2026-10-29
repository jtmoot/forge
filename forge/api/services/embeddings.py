"""Embedding generation and semantic search using OpenAI + pgvector.

Generates text embeddings for experiment run summaries and stores them
in the experiment_embeddings table. Provides cosine-similarity search
over stored embeddings for the LangGraph analysis agent.
"""

import logging
import os

from sqlalchemy import text
from sqlalchemy.orm import Session

from forge.api.models.database import Experiment, ExperimentEmbedding, Run

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIM = 1536


def _get_openai_client():
    """Return an OpenAI client, or None if the API key is not set.

    Reads OPENAI_API_KEY lazily so that key rotation or late-set env vars
    are picked up without restarting the process.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — embeddings disabled")
        return None
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def generate_embedding(content: str) -> list[float] | None:
    """Generate a 1536-dim embedding for the given text using OpenAI.

    Returns None if OPENAI_API_KEY is not configured.
    """
    client = _get_openai_client()
    if client is None:
        return None

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=content,
    )
    return response.data[0].embedding


def build_run_summary(run: Run, experiment: Experiment) -> str:
    """Build a human-readable text summary of a completed run for embedding.

    Combines experiment name, model type, hyperparameters, and key metrics
    into a single string suitable for semantic search.
    """
    parts = [
        f"Experiment: {experiment.name}",
        f"Description: {experiment.description or 'N/A'}",
        f"Model: {run.model_type}",
        f"Run: {run.run_name or run.id}",
    ]

    if run.hyperparameters:
        hp_str = ", ".join(f"{k}={v}" for k, v in run.hyperparameters.items())
        parts.append(f"Hyperparameters: {hp_str}")

    if run.accuracy is not None:
        parts.append(f"Accuracy: {run.accuracy:.4f}")
    if run.f1 is not None:
        parts.append(f"F1: {run.f1:.4f}")
    if run.precision_score is not None:
        parts.append(f"Precision: {run.precision_score:.4f}")
    if run.recall is not None:
        parts.append(f"Recall: {run.recall:.4f}")

    if run.inference_latency_ms is not None:
        parts.append(f"Inference latency: {run.inference_latency_ms:.2f}ms")
    if run.peak_memory_mb is not None:
        parts.append(f"Peak memory: {run.peak_memory_mb:.2f}MB")
    if run.efficiency_score is not None:
        parts.append(f"Efficiency score: {run.efficiency_score:.4f}")
    if run.throughput_samples_per_sec is not None:
        parts.append(f"Throughput: {run.throughput_samples_per_sec:.1f} samples/sec")

    parts.append(f"Status: {run.status}")

    return "\n".join(parts)


def embed_run(run_id, db: Session) -> ExperimentEmbedding | None:
    """Generate and store an embedding for a completed run.

    Looks up the run and its parent experiment, builds a text summary,
    generates an embedding via OpenAI, and inserts it into the
    experiment_embeddings table.

    Returns the created ExperimentEmbedding row, or None if embeddings
    are disabled (no API key).
    """
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        logger.error("Cannot embed run %s — not found", run_id)
        return None

    experiment = db.query(Experiment).filter(Experiment.id == run.experiment_id).first()
    if experiment is None:
        logger.error("Cannot embed run %s — experiment not found", run_id)
        return None

    summary = build_run_summary(run, experiment)
    embedding_vector = generate_embedding(summary)
    if embedding_vector is None:
        return None

    # Upsert: remove old embedding for this run if it exists
    db.query(ExperimentEmbedding).filter(
        ExperimentEmbedding.run_id == run_id,
        ExperimentEmbedding.content_type == "run_summary",
    ).delete()

    row = ExperimentEmbedding(
        run_id=run_id,
        content_type="run_summary",
        content_text=summary,
        embedding=embedding_vector,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info("Stored embedding for run %s (%d chars)", run_id, len(summary))
    return row


def search_similar_runs(
    query: str,
    db: Session,
    limit: int = 5,
) -> list[dict]:
    """Semantic search over run embeddings using pgvector cosine distance.

    Returns the top-N most similar runs with their content text and
    similarity scores.
    """
    query_embedding = generate_embedding(query)
    if query_embedding is None:
        return []

    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    sql = text("""
        SELECT
            ee.run_id,
            ee.content_text,
            1 - (ee.embedding <=> :embedding::vector) AS similarity
        FROM experiment_embeddings ee
        ORDER BY ee.embedding <=> :embedding::vector
        LIMIT :limit
    """)

    rows = db.execute(sql, {"embedding": embedding_str, "limit": limit}).fetchall()

    return [
        {
            "run_id": str(row.run_id),
            "content_text": row.content_text,
            "similarity": round(float(row.similarity), 4),
        }
        for row in rows
    ]
