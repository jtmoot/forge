"""Run database migrations from the API process at startup.

The Docker image's CMD runs scripts/migrate.py before uvicorn, but a
platform start-command override (Railway's dashboard setting) replaces the
CMD and silently skips it — new tables then never get created. Running the
same idempotent script from the app's startup makes migrations independent
of how the process is launched.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MIGRATE_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "migrate.py"
MIGRATION_TIMEOUT_SECONDS = 120


def run_migrations(script: Path = MIGRATE_SCRIPT) -> bool:
    """Run the migration script; returns True if it exited cleanly.

    Skipped (returns True) when FORGE_SKIP_MIGRATIONS=1. The script itself
    logs and tolerates failures, so the API still starts if migrations fail.
    """
    if os.getenv("FORGE_SKIP_MIGRATIONS") == "1":
        logger.info("FORGE_SKIP_MIGRATIONS=1 — skipping startup migrations")
        return True
    if not script.is_file():
        logger.warning("Migration script not found at %s", script)
        return False
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=MIGRATION_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error("Migrations timed out after %ds", MIGRATION_TIMEOUT_SECONDS)
        return False
    for line in (result.stdout + result.stderr).splitlines():
        logger.warning("%s", line)
    return result.returncode == 0
