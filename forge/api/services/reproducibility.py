"""Reproducibility service — environment capture, data hashing, and seed management.

Ensures every experiment run can be fully reproduced by pinning code version,
data version, environment, and random seeds.
"""

import hashlib
import logging
import os
import platform
import random
import subprocess
from uuid import UUID

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from forge.api.models.database import Run, RunEnvironment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _run_git_command(args: list[str]) -> str | None:
    """Run a git command and return stdout, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_package_versions() -> dict[str, str]:
    """Get installed package versions using importlib.metadata.

    Returns a dict mapping package name to version string.
    """
    from importlib.metadata import distributions

    packages: dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        version = dist.metadata["Version"]
        if name and version:
            packages[name.lower()] = version
    return packages


def _compute_env_hash(package_versions: dict[str, str]) -> str:
    """Compute SHA-256 hash of sorted package versions for fast equality check."""
    sorted_packages = sorted(package_versions.items())
    content = "\n".join(f"{name}=={version}" for name, version in sorted_packages)
    return hashlib.sha256(content.encode()).hexdigest()


def capture_environment(random_seed: int = 42) -> dict:
    """Capture full environment snapshot for reproducibility.

    Returns a dict with git info, Python version, package versions,
    docker image tag, random seed, and environment hash.

    Args:
        random_seed: The random seed that will be used for this run.

    Returns:
        Dict with keys: git_sha, git_branch, git_dirty, python_version,
        package_versions, docker_image_tag, random_seed, env_hash.
    """
    git_sha = _run_git_command(["rev-parse", "HEAD"])
    git_branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])

    git_dirty = False
    dirty_check = _run_git_command(["status", "--porcelain"])
    if dirty_check is not None:
        git_dirty = len(dirty_check) > 0

    python_version = platform.python_version()
    package_versions = _get_package_versions()
    env_hash = _compute_env_hash(package_versions)

    docker_image_tag = os.getenv("DOCKER_IMAGE_TAG")

    return {
        "git_sha": git_sha,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "python_version": python_version,
        "package_versions": package_versions,
        "docker_image_tag": docker_image_tag,
        "random_seed": random_seed,
        "env_hash": env_hash,
    }


# ---------------------------------------------------------------------------
# Data hashing
# ---------------------------------------------------------------------------


def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 hash of a DataFrame's content.

    Incorporates column names, dtypes, and cell values so that any change
    in the data produces a different hash, regardless of serialization format.

    Args:
        df: The input DataFrame to hash.

    Returns:
        64-character hex SHA-256 digest.
    """
    hasher = hashlib.sha256()

    # Hash column names and dtypes
    col_info = "|".join(f"{col}:{dtype}" for col, dtype in zip(df.columns, df.dtypes))
    hasher.update(col_info.encode())

    # Hash row content using pandas hash utility
    row_hashes = pd.util.hash_pandas_object(df, index=False)
    hasher.update(row_hashes.values.tobytes())

    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int = 42) -> int:
    """Set random seeds for all RNG sources used in the training pipeline.

    Seeds: Python random, NumPy, and PyTorch (if available).
    Also sets PYTHONHASHSEED and enables PyTorch deterministic mode.

    Args:
        seed: The seed value to use.

    Returns:
        The seed that was set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info("All random seeds set to %d", seed)
    return seed


# ---------------------------------------------------------------------------
# Store environment snapshot
# ---------------------------------------------------------------------------


def store_environment(db: Session, run_id: UUID, env: dict) -> RunEnvironment:
    """Persist a captured environment snapshot linked to a run.

    Args:
        db: Database session.
        run_id: The run to associate the environment with.
        env: Dict from capture_environment().

    Returns:
        The created RunEnvironment record.
    """
    run_env = RunEnvironment(
        run_id=run_id,
        git_sha=env["git_sha"],
        git_branch=env["git_branch"],
        git_dirty=env["git_dirty"],
        python_version=env["python_version"],
        package_versions=env["package_versions"],
        docker_image_tag=env.get("docker_image_tag"),
        random_seed=env["random_seed"],
        env_hash=env["env_hash"],
    )
    db.add(run_env)
    db.flush()
    return run_env


# ---------------------------------------------------------------------------
# Reproducibility verification
# ---------------------------------------------------------------------------


def verify_reproducibility(run_id_a: UUID, run_id_b: UUID, db: Session) -> dict:
    """Compare two runs to assess reproducibility.

    Checks whether git SHA, environment hash, data version hash, random seed,
    and feature set ID match between the two runs. If all match but metrics
    differ, flags the result as non-deterministic.

    Args:
        run_id_a: First run ID.
        run_id_b: Second run ID.
        db: Database session.

    Returns:
        Dict with per-factor match status and an overall reproducibility verdict.
    """
    run_a = db.query(Run).filter(Run.id == run_id_a).first()
    run_b = db.query(Run).filter(Run.id == run_id_b).first()
    if run_a is None:
        raise ValueError(f"Run {run_id_a} not found")
    if run_b is None:
        raise ValueError(f"Run {run_id_b} not found")

    env_a = db.query(RunEnvironment).filter(RunEnvironment.run_id == run_id_a).first()
    env_b = db.query(RunEnvironment).filter(RunEnvironment.run_id == run_id_b).first()

    factors: dict[str, dict] = {}

    # Git SHA
    sha_a = env_a.git_sha if env_a else None
    sha_b = env_b.git_sha if env_b else None
    factors["git_sha"] = {
        "run_a": sha_a,
        "run_b": sha_b,
        "match": sha_a == sha_b and sha_a is not None,
    }

    # Environment hash
    hash_a = env_a.env_hash if env_a else None
    hash_b = env_b.env_hash if env_b else None
    factors["env_hash"] = {
        "run_a": hash_a,
        "run_b": hash_b,
        "match": hash_a == hash_b and hash_a is not None,
    }

    # Data version hash
    factors["data_version_hash"] = {
        "run_a": run_a.data_version_hash,
        "run_b": run_b.data_version_hash,
        "match": (
            run_a.data_version_hash == run_b.data_version_hash
            and run_a.data_version_hash is not None
        ),
    }

    # Random seed
    seed_a = env_a.random_seed if env_a else None
    seed_b = env_b.random_seed if env_b else None
    factors["random_seed"] = {
        "run_a": seed_a,
        "run_b": seed_b,
        "match": seed_a == seed_b and seed_a is not None,
    }

    # Feature set ID
    fs_a = str(run_a.feature_set_id) if run_a.feature_set_id else None
    fs_b = str(run_b.feature_set_id) if run_b.feature_set_id else None
    factors["feature_set_id"] = {
        "run_a": fs_a,
        "run_b": fs_b,
        "match": fs_a == fs_b,
    }

    all_match = all(f["match"] for f in factors.values())

    # Check if metrics differ despite matching environments
    warnings: list[str] = []
    if (
        all_match
        and run_a.accuracy is not None
        and run_b.accuracy is not None
        and abs(run_a.accuracy - run_b.accuracy) > 1e-6
    ):
        warnings.append(
            "All reproducibility factors match but metrics differ — "
            "this is expected for some GPU operations (non-deterministic CUDA kernels)"
        )

    # Dirty git warnings
    if env_a and env_a.git_dirty:
        warnings.append("Run A had uncommitted changes (git dirty)")
    if env_b and env_b.git_dirty:
        warnings.append("Run B had uncommitted changes (git dirty)")

    verdict = "reproducible" if all_match else "not_reproducible"
    if all_match and warnings:
        verdict = "reproducible_with_warnings"

    return {
        "verdict": verdict,
        "factors": factors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Package diff between environments
# ---------------------------------------------------------------------------


def diff_environments(env_a: RunEnvironment | None, env_b: RunEnvironment | None) -> dict:
    """Compute detailed diff between two run environments.

    Returns added, removed, and changed packages plus other field diffs.

    Args:
        env_a: First environment (may be None).
        env_b: Second environment (may be None).

    Returns:
        Dict with packages_added, packages_removed, packages_changed,
        and field-level diffs.
    """
    pkgs_a = (env_a.package_versions or {}) if env_a else {}
    pkgs_b = (env_b.package_versions or {}) if env_b else {}

    all_keys = set(pkgs_a.keys()) | set(pkgs_b.keys())

    packages_added: dict[str, str] = {}
    packages_removed: dict[str, str] = {}
    packages_changed: dict[str, dict] = {}

    for key in sorted(all_keys):
        in_a = key in pkgs_a
        in_b = key in pkgs_b
        if in_a and not in_b:
            packages_removed[key] = pkgs_a[key]
        elif in_b and not in_a:
            packages_added[key] = pkgs_b[key]
        elif pkgs_a[key] != pkgs_b[key]:
            packages_changed[key] = {"run_a": pkgs_a[key], "run_b": pkgs_b[key]}

    field_diffs: dict[str, dict] = {}
    for field in ("git_sha", "git_branch", "python_version", "random_seed", "env_hash"):
        val_a = getattr(env_a, field, None) if env_a else None
        val_b = getattr(env_b, field, None) if env_b else None
        if val_a != val_b:
            field_diffs[field] = {"run_a": val_a, "run_b": val_b}

    return {
        "packages_added": packages_added,
        "packages_removed": packages_removed,
        "packages_changed": packages_changed,
        "field_diffs": field_diffs,
        "environments_identical": (
            len(packages_added) == 0
            and len(packages_removed) == 0
            and len(packages_changed) == 0
            and len(field_diffs) == 0
        ),
    }
