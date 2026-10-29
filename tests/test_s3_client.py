"""Tests for model artifact storage (local-disk fallback path)."""

from pathlib import Path
from uuid import uuid4

import pytest
from sklearn.tree import DecisionTreeClassifier

from forge.api.services import s3_client
from forge.api.services.s3_client import (
    ArtifactNotFoundError,
    download_model_artifact,
    upload_model_artifact,
)


@pytest.fixture
def local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force the local-disk fallback into a temporary directory."""
    monkeypatch.setattr(s3_client, "_S3_AVAILABLE", False)
    monkeypatch.setattr(s3_client, "_LOCAL_ARTIFACT_DIR", tmp_path)
    return tmp_path


def test_local_roundtrip_with_metadata(local_storage: Path) -> None:
    model = DecisionTreeClassifier().fit([[0, 0], [1, 1]], [0, 1])
    metadata = {"model_type": "random_forest", "input_size": 2, "feature_names": ["a", "b"]}
    run_id = uuid4()

    path = upload_model_artifact(model, run_id, "random_forest", metadata)

    assert path is not None and path.startswith("file://")
    artifact_bytes, loaded_metadata = download_model_artifact(path)
    assert len(artifact_bytes) > 0
    assert loaded_metadata == metadata
    assert (local_storage / "models" / str(run_id) / "meta.json").is_file()


def test_local_roundtrip_without_metadata(local_storage: Path) -> None:
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    path = upload_model_artifact(model, uuid4(), "random_forest")
    _, loaded_metadata = download_model_artifact(path)
    assert loaded_metadata is None


def test_missing_local_artifact_raises(local_storage: Path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        download_model_artifact((local_storage / "nope.joblib").as_uri())


def test_unsupported_scheme_raises() -> None:
    with pytest.raises(ArtifactNotFoundError):
        download_model_artifact("gs://bucket/key")
