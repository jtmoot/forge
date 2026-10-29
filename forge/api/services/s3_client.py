"""AWS S3 artifact storage service.

Handles uploading model artifacts and datasets to S3, reading model
artifacts back for serving, and generating presigned URLs for download.
Gracefully degrades if AWS credentials are not configured — model
artifacts fall back to local disk, other operations return None.
"""

import io
import json
import logging
import os
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import UUID

import joblib
import torch

logger = logging.getLogger(__name__)

METADATA_FILENAME = "meta.json"
_LOCAL_ARTIFACT_DIR = Path(os.getenv("FORGE_ARTIFACT_DIR", "data/artifacts"))

# ---------------------------------------------------------------------------
# Check availability at import time
# ---------------------------------------------------------------------------

_S3_AVAILABLE = False
_s3_client = None
_S3_BUCKET: str | None = None

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        _S3_BUCKET = os.getenv("AWS_S3_BUCKET", "forge-artifacts")

        # Support custom endpoint for localstack or S3-compatible storage
        endpoint_url = os.getenv("AWS_ENDPOINT_URL")
        region = os.getenv("AWS_REGION", "us-east-1")

        session_kwargs: dict = {"region_name": region}
        client_kwargs: dict = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        _s3_client = boto3.client("s3", **session_kwargs, **client_kwargs)
        _S3_AVAILABLE = True
        logger.info("S3 client initialized — bucket=%s", _S3_BUCKET)
    else:
        logger.info("AWS credentials not set — S3 storage disabled")
except ImportError:
    logger.info("boto3 package not installed — S3 storage disabled")


def is_enabled() -> bool:
    """Return True if S3 storage is available and configured."""
    return _S3_AVAILABLE


# ---------------------------------------------------------------------------
# Upload operations
# ---------------------------------------------------------------------------


def upload_model_artifact(
    model: object,
    run_id: UUID,
    model_type: str,
    metadata: dict | None = None,
) -> str | None:
    """Serialize and store a trained model, with an optional metadata sidecar.

    Stores to S3 when configured, otherwise to the local artifact directory
    (FORGE_ARTIFACT_DIR) so models stay servable in local development. The
    metadata dict is written as ``meta.json`` next to the artifact; the
    serving layer needs it to rebuild PyTorch architectures.

    Args:
        model: Trained model (sklearn, xgboost, or PyTorch nn.Module).
        run_id: The run UUID, used in the artifact key path.
        model_type: Model type identifier (e.g. 'xgboost', 'lstm').
        metadata: JSON-serializable model metadata (input size, features, ...).

    Returns:
        The artifact path (s3://bucket/key or file:///abs/path), or None if
        the upload failed.
    """
    buffer = io.BytesIO()
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), buffer)
        extension = "pt"
    else:
        joblib.dump(model, buffer)
        extension = "joblib"

    key = f"models/{run_id}/{model_type}.{extension}"
    meta_key = f"models/{run_id}/{METADATA_FILENAME}"
    meta_bytes = json.dumps(metadata).encode() if metadata is not None else None

    if not _S3_AVAILABLE:
        local_path = _LOCAL_ARTIFACT_DIR.resolve() / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(buffer.getvalue())
        if meta_bytes is not None:
            (local_path.parent / METADATA_FILENAME).write_bytes(meta_bytes)
        logger.info("S3 disabled — stored model artifact locally at %s", local_path)
        return local_path.as_uri()

    try:
        _s3_client.put_object(
            Bucket=_S3_BUCKET,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )
        if meta_bytes is not None:
            _s3_client.put_object(
                Bucket=_S3_BUCKET,
                Key=meta_key,
                Body=meta_bytes,
                ContentType="application/json",
            )
        s3_path = f"s3://{_S3_BUCKET}/{key}"
        logger.info("Uploaded model artifact to %s", s3_path)
        return s3_path
    except (ClientError, NoCredentialsError):
        logger.exception("Failed to upload model artifact for run %s", run_id)
        return None


# ---------------------------------------------------------------------------
# Download operations
# ---------------------------------------------------------------------------


class ArtifactNotFoundError(Exception):
    """Raised when a model artifact cannot be read from its stored path."""


def download_model_artifact(artifact_path: str) -> tuple[bytes, dict | None]:
    """Read a model artifact and its metadata sidecar (if present).

    Args:
        artifact_path: Path returned by upload_model_artifact
            (s3://bucket/key or file:///abs/path).

    Returns:
        Tuple of (artifact bytes, metadata dict or None when no sidecar exists).

    Raises:
        ArtifactNotFoundError: If the artifact itself cannot be read.
    """
    if artifact_path.startswith("file://"):
        local_path = Path(unquote(urlparse(artifact_path).path))
        if not local_path.is_file():
            raise ArtifactNotFoundError(f"Local artifact not found: {local_path}")
        meta_path = local_path.parent / METADATA_FILENAME
        metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else None
        return local_path.read_bytes(), metadata

    if not artifact_path.startswith("s3://"):
        raise ArtifactNotFoundError(f"Unsupported artifact path: {artifact_path}")
    if not _S3_AVAILABLE:
        raise ArtifactNotFoundError(f"S3 is not configured; cannot read {artifact_path}")

    bucket, _, key = artifact_path.removeprefix("s3://").partition("/")
    try:
        artifact_bytes = _s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except (ClientError, NoCredentialsError) as exc:
        raise ArtifactNotFoundError(f"Cannot read {artifact_path}: {exc}") from exc

    meta_key = f"{key.rpartition('/')[0]}/{METADATA_FILENAME}"
    try:
        meta_body = _s3_client.get_object(Bucket=bucket, Key=meta_key)["Body"].read()
        metadata = json.loads(meta_body)
    except ClientError:
        metadata = None
    return artifact_bytes, metadata


def upload_dataset_artifact(
    file_path: str,
    dataset_id: UUID,
) -> str | None:
    """Upload a dataset file (e.g. parquet) to S3.

    Args:
        file_path: Local path to the dataset file.
        dataset_id: The dataset UUID, used in the S3 key path.

    Returns:
        The S3 path, or None if S3 is disabled or the upload failed.
    """
    if not _S3_AVAILABLE:
        return None

    filename = os.path.basename(file_path)
    s3_key = f"datasets/{dataset_id}/{filename}"

    try:
        _s3_client.upload_file(file_path, _S3_BUCKET, s3_key)
        s3_path = f"s3://{_S3_BUCKET}/{s3_key}"
        logger.info("Uploaded dataset artifact to %s", s3_path)
        return s3_path
    except (ClientError, NoCredentialsError):
        logger.exception("Failed to upload dataset artifact for dataset %s", dataset_id)
        return None


def download_dataset_artifact(dataset_id: UUID, filename: str, destination: Path) -> bool:
    """Restore a dataset file uploaded by upload_dataset_artifact to local disk.

    Args:
        dataset_id: The dataset UUID used in the S3 key path.
        filename: The uploaded file's name (e.g. '<id>.parquet').
        destination: Local path to write the file to.

    Returns:
        True if the file was downloaded, False if S3 is disabled or it isn't there.
    """
    if not _S3_AVAILABLE:
        return False

    s3_key = f"datasets/{dataset_id}/{filename}"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _s3_client.download_file(_S3_BUCKET, s3_key, str(destination))
        logger.info("Restored dataset %s from s3://%s/%s", dataset_id, _S3_BUCKET, s3_key)
        return True
    except (ClientError, NoCredentialsError):
        logger.warning("Dataset %s not available in S3 at %s", dataset_id, s3_key)
        return False


def generate_presigned_url(
    s3_path: str,
    expiry_seconds: int = 3600,
) -> str | None:
    """Generate a presigned download URL for an S3 object.

    Args:
        s3_path: Full S3 path (s3://bucket/key).
        expiry_seconds: URL expiry time in seconds (default 1 hour).

    Returns:
        The presigned URL string, or None if S3 is disabled or generation failed.
    """
    if not _S3_AVAILABLE:
        return None

    # Parse s3://bucket/key format
    path = s3_path.removeprefix("s3://")
    bucket, _, key = path.partition("/")

    try:
        url = _s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiry_seconds,
        )
        return url
    except (ClientError, NoCredentialsError):
        logger.exception("Failed to generate presigned URL for %s", s3_path)
        return None
