"""FastAPI application entry point."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.exc import SQLAlchemyError

from forge.api.models.database import SessionLocal
from forge.api.routers import (
    analysis,
    datasets,
    drift,
    experiments,
    feature_store,
    health,
    metrics,
    model_registry,
    ops,
    projects,
    serving,
    webhooks,
)
from forge.api.services.run_recovery import recover_orphaned_runs
from forge.api.services.runtime import configure_torch_threads

logger = logging.getLogger(__name__)

# Size torch's thread pool to the container CPU quota before any model runs
configure_torch_threads()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup: fail training runs orphaned by the previous process."""
    db = SessionLocal()
    try:
        recover_orphaned_runs(db)
    except SQLAlchemyError:
        logger.exception("Could not recover orphaned runs at startup")
    finally:
        db.close()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Forge",
    description="ML Experimentation & Agent Operations Platform",
    version="0.1.0",
)

_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler — ensures CORS headers are sent even on unhandled 500s.
# Without this, CORSMiddleware never wraps the raw exception response and browsers
# block the response entirely due to missing Access-Control-Allow-Origin.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a JSON 500 response so CORS middleware can attach headers."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# Prometheus auto-instrumentation: request count, latency histogram, error rate,
# in-progress requests — exposed at GET /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(ops.router)
app.include_router(webhooks.router)
app.include_router(projects.router)
app.include_router(analysis.router)
app.include_router(feature_store.router)
app.include_router(model_registry.router)
app.include_router(serving.router)
app.include_router(drift.router)
app.include_router(metrics.router)

# Railway deployment: read PORT from environment for dynamic port binding.
port = int(os.getenv("PORT", "8000"))
