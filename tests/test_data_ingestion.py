"""Tests for dataset loading with S3 restore of missing local parquet files."""

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pandas as pd
import pytest

from forge.api.services import data_ingestion
from forge.api.services.data_ingestion import load_dataset_frame


@pytest.fixture
def frame() -> pd.DataFrame:
    """Tiny dataset frame."""
    return pd.DataFrame({"Close": [1.0, 2.0, 3.0], "rsi_14": [50.0, 55.0, 60.0]})


def _dataset(path: Path | None) -> SimpleNamespace:
    return SimpleNamespace(id=uuid4(), s3_path=str(path) if path else None)


def test_reads_local_parquet(tmp_path: Path, frame: pd.DataFrame) -> None:
    path = tmp_path / "ds.parquet"
    frame.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(load_dataset_frame(_dataset(path)), frame)


def test_restores_missing_file_from_s3(
    tmp_path: Path, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gone" / "ds.parquet"
    requested: list[tuple[UUID, str, Path]] = []

    def fake_download(dataset_id: UUID, filename: str, destination: Path) -> bool:
        requested.append((dataset_id, filename, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)
        return True

    monkeypatch.setattr(data_ingestion, "download_dataset_artifact", fake_download)
    dataset = _dataset(path)

    pd.testing.assert_frame_equal(load_dataset_frame(dataset), frame)
    assert requested == [(dataset.id, "ds.parquet", path)]


def test_returns_none_when_missing_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_ingestion, "download_dataset_artifact", lambda *_: False)
    assert load_dataset_frame(_dataset(tmp_path / "missing.parquet")) is None


def test_returns_none_without_path() -> None:
    assert load_dataset_frame(_dataset(None)) is None
