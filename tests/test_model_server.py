"""Tests for the model serving service (artifact rebuild, inference, cache)."""

import io
import threading
import time
from uuid import uuid4

import joblib
import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from forge.api.services.model_server import (
    InvalidInputError,
    ModelArtifactError,
    ModelCache,
    ServableModel,
    build_model,
    create_servable,
    prepare_inputs,
    run_inference,
)
from forge.api.services.training import TimeSeriesLSTM, TimeSeriesTransformer

N_FEATURES = 6
WINDOW = 10


def _torch_bytes(model: torch.nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def _joblib_bytes(model: object) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def _seq_metadata(model_type: str, hyperparameters: dict) -> dict:
    return {
        "model_type": model_type,
        "input_size": N_FEATURES,
        "window_size": WINDOW,
        "feature_names": [f"f{i}" for i in range(N_FEATURES)],
        "hyperparameters": {"window_size": WINDOW, **hyperparameters},
    }


@pytest.fixture
def tabular_data() -> tuple[np.ndarray, np.ndarray]:
    """Small separable tabular dataset."""
    rng = np.random.RandomState(0)
    x = rng.randn(200, N_FEATURES).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    return x, y


@pytest.fixture
def lstm_servable() -> tuple[ServableModel, TimeSeriesLSTM]:
    """An LSTM servable built from serialized weights, plus the original model."""
    torch.manual_seed(0)
    original = TimeSeriesLSTM(N_FEATURES, hidden_size=16, num_layers=2, dropout=0.0).eval()
    metadata = _seq_metadata("lstm", {"hidden_size": 16, "num_layers": 2, "dropout": 0.0})
    servable = create_servable("spy", 3, uuid4(), _torch_bytes(original), metadata)
    return servable, original


class TestBuildModel:
    """Rebuilding models from stored artifacts."""

    def test_lstm_roundtrip_matches_original(
        self, lstm_servable: tuple[ServableModel, TimeSeriesLSTM]
    ) -> None:
        servable, original = lstm_servable
        batch = np.random.RandomState(1).randn(4, WINDOW, N_FEATURES).astype(np.float32)
        with torch.no_grad():
            expected = original(torch.tensor(batch)).squeeze(-1).numpy()
        result = run_inference(servable, batch.tolist())
        np.testing.assert_allclose(result.probabilities, expected, rtol=1e-5, atol=1e-6)

    def test_transformer_roundtrip(self) -> None:
        torch.manual_seed(0)
        hp = {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 32, "dropout": 0.0}
        original = TimeSeriesTransformer(N_FEATURES, **hp).eval()
        model = build_model(_torch_bytes(original), _seq_metadata("transformer", hp))
        batch = torch.randn(2, WINDOW, N_FEATURES)
        with torch.no_grad():
            torch.testing.assert_close(model(batch), original(batch))

    def test_xgboost_roundtrip(self, tabular_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = tabular_data
        original = XGBClassifier(n_estimators=10, max_depth=2, verbosity=0).fit(x, y)
        metadata = {"model_type": "xgboost", "input_size": N_FEATURES, "window_size": None}
        servable = create_servable("xgb", 1, uuid4(), _joblib_bytes(original), metadata)
        result = run_inference(servable, x[:5].tolist())
        np.testing.assert_allclose(result.probabilities, original.predict_proba(x[:5])[:, 1], rtol=1e-6)
        assert servable.backend == "xgboost"

    def test_sklearn_without_metadata_uses_n_features_in(
        self, tabular_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = tabular_data
        original = RandomForestClassifier(n_estimators=5, random_state=0).fit(x, y)
        servable = create_servable(
            "rf", 1, uuid4(), _joblib_bytes(original), {"model_type": "random_forest"}
        )
        assert servable.input_shape == (N_FEATURES,)
        assert servable.backend == "sklearn"

    def test_torch_without_input_size_raises(self) -> None:
        original = TimeSeriesLSTM(N_FEATURES, hidden_size=8)
        with pytest.raises(ModelArtifactError):
            build_model(_torch_bytes(original), {"model_type": "lstm", "hyperparameters": {}})

    def test_unknown_model_type_raises(self) -> None:
        with pytest.raises(ModelArtifactError):
            build_model(b"", {"model_type": "mystery", "input_size": 3})


class TestInputValidation:
    """Request payload shape checks."""

    def test_sequence_input_shape(self, lstm_servable: tuple[ServableModel, TimeSeriesLSTM]) -> None:
        servable, _ = lstm_servable
        assert servable.input_shape == (WINDOW, N_FEATURES)
        batch = np.zeros((3, WINDOW, N_FEATURES)).tolist()
        assert prepare_inputs(servable, batch).shape == (3, WINDOW, N_FEATURES)

    def test_wrong_feature_count_rejected(
        self, lstm_servable: tuple[ServableModel, TimeSeriesLSTM]
    ) -> None:
        servable, _ = lstm_servable
        with pytest.raises(InvalidInputError, match="expected"):
            prepare_inputs(servable, np.zeros((1, WINDOW, N_FEATURES + 1)).tolist())

    def test_tabular_rows_sent_to_sequence_model_rejected(
        self, lstm_servable: tuple[ServableModel, TimeSeriesLSTM]
    ) -> None:
        servable, _ = lstm_servable
        with pytest.raises(InvalidInputError):
            prepare_inputs(servable, np.zeros((2, N_FEATURES)).tolist())

    def test_ragged_input_rejected(self, lstm_servable: tuple[ServableModel, TimeSeriesLSTM]) -> None:
        servable, _ = lstm_servable
        with pytest.raises(InvalidInputError):
            prepare_inputs(servable, [[1.0, 2.0], [1.0]])

    def test_non_finite_values_rejected(
        self, lstm_servable: tuple[ServableModel, TimeSeriesLSTM]
    ) -> None:
        servable, _ = lstm_servable
        batch = np.zeros((1, WINDOW, N_FEATURES))
        batch[0, 0, 0] = np.nan
        with pytest.raises(InvalidInputError):
            prepare_inputs(servable, batch.tolist())


class TestRunInference:
    """End-to-end inference output."""

    def test_labels_threshold_probabilities(
        self, lstm_servable: tuple[ServableModel, TimeSeriesLSTM]
    ) -> None:
        servable, _ = lstm_servable
        batch = np.random.RandomState(2).randn(8, WINDOW, N_FEATURES).tolist()
        result = run_inference(servable, batch)
        assert len(result.probabilities) == 8
        assert all(0.0 <= p <= 1.0 for p in result.probabilities)
        assert result.labels == [int(p >= 0.5) for p in result.probabilities]
        assert result.latency_ms > 0


class TestModelCache:
    """Load-once caching and hot-swap eviction."""

    def _fake_servable(self, version: int) -> ServableModel:
        x = np.random.RandomState(0).randn(20, 3)
        model = RandomForestClassifier(n_estimators=2, random_state=0).fit(x, (x[:, 0] > 0))
        return create_servable("m", version, uuid4(), _joblib_bytes(model), {"model_type": "random_forest"})

    def test_loader_called_once(self) -> None:
        cache = ModelCache()
        calls = []

        def loader() -> ServableModel:
            calls.append(1)
            return self._fake_servable(1)

        first = cache.get_or_load("m", loader)
        second = cache.get_or_load("m", loader)
        assert first is second
        assert len(calls) == 1

    def test_evict_forces_reload_of_new_version(self) -> None:
        cache = ModelCache()
        versions = iter([1, 2])
        cache.get_or_load("m", lambda: self._fake_servable(next(versions)))
        assert cache.evict("m") is True
        reloaded = cache.get_or_load("m", lambda: self._fake_servable(next(versions)))
        assert reloaded.version == 2
        assert cache.evict("missing") is False

    def test_concurrent_requests_load_once(self) -> None:
        cache = ModelCache()
        calls = []
        servable = self._fake_servable(1)

        def slow_loader() -> ServableModel:
            calls.append(1)
            time.sleep(0.05)
            return servable

        results: list[ServableModel] = []
        threads = [
            threading.Thread(target=lambda: results.append(cache.get_or_load("m", slow_loader)))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(calls) == 1
        assert all(result is servable for result in results)

    def test_failed_load_is_not_cached(self) -> None:
        cache = ModelCache()

        def failing_loader() -> ServableModel:
            raise ModelArtifactError("boom")

        with pytest.raises(ModelArtifactError):
            cache.get_or_load("m", failing_loader)
        assert cache.get_or_load("m", lambda: self._fake_servable(1)).version == 1
        assert [s.name for s in cache.loaded()] == ["m"]
