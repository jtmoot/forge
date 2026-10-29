"""Tests for ONNX Runtime serving backends (export, int8, parity gating, selection)."""

import io
from uuid import uuid4

import joblib
import numpy as np
import onnx
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from forge.api.models.database import get_db
from forge.api.routers import serving
from forge.api.services.model_server import (
    InvalidInputError,
    ServableModel,
    create_servable,
    predict_proba,
    run_inference,
)
from forge.api.services.onnx_backend import (
    FP32_MAX_ABS_DIFF,
    INT8_MAX_ABS_DIFF,
    INT8_MIN_LABEL_AGREEMENT,
    ONNX_FP32_BACKEND,
    ONNX_INT8_BACKEND,
    OnnxExportError,
    OnnxSession,
    QuantizationNotApplicableError,
    check_parity,
    export_to_onnx,
    parity_inputs,
    quantize_int8,
)
from forge.api.services.training import (
    TimeSeriesCNNLSTM,
    TimeSeriesLSTM,
    TimeSeriesTCN,
    TimeSeriesTransformer,
)

N_FEATURES = 6
WINDOW = 10

TINY_SEQUENCE_MODELS: dict[str, tuple[type[torch.nn.Module], dict]] = {
    "lstm": (TimeSeriesLSTM, {"hidden_size": 16, "num_layers": 2, "dropout": 0.0}),
    "tcn": (TimeSeriesTCN, {"num_channels": 8, "num_layers": 3, "kernel_size": 3, "dropout": 0.0}),
    "cnn_lstm": (
        TimeSeriesCNNLSTM,
        {"cnn_filters": 8, "cnn_kernel_size": 3, "lstm_hidden": 8, "lstm_layers": 1, "dropout": 0.0},
    ),
    "transformer": (
        TimeSeriesTransformer,
        {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 32, "dropout": 0.0},
    ),
}


def _torch_bytes(model: torch.nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def _joblib_bytes(model: object) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def _sequence_servable(model_type: str) -> ServableModel:
    """Build a tiny sequence-model servable through the real load path."""
    model_class, hyperparameters = TINY_SEQUENCE_MODELS[model_type]
    torch.manual_seed(0)
    original = model_class(N_FEATURES, **hyperparameters).eval()
    metadata = {
        "model_type": model_type,
        "input_size": N_FEATURES,
        "window_size": WINDOW,
        "hyperparameters": {"window_size": WINDOW, **hyperparameters},
    }
    return create_servable(model_type, 1, uuid4(), _torch_bytes(original), metadata)


@pytest.fixture(scope="module")
def tabular_data() -> tuple[np.ndarray, np.ndarray]:
    """Tabular data with features on very different scales, like raw market features."""
    rng = np.random.RandomState(0)
    scales = np.array([1.0, 10.0, 100.0, 1e3, 1e6, 0.01], dtype=np.float32)
    x = (rng.randn(400, N_FEATURES) * scales).astype(np.float32)
    y = (x[:, 0] + x[:, 4] / 1e6 > 0).astype(int)
    return x, y


@pytest.fixture(scope="module")
def lstm_servable() -> ServableModel:
    """A tiny LSTM servable shared by the selection tests."""
    return _sequence_servable("lstm")


class _ScaledSession(OnnxSession):
    """A real ONNX session whose outputs are deliberately perturbed."""

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        return super().predict_proba(inputs) * 0.5


class TestSequenceExport:
    """Export + parity for every sequence architecture."""

    @pytest.mark.parametrize("model_type", sorted(TINY_SEQUENCE_MODELS))
    def test_fp32_and_int8_backends_pass_parity(self, model_type: str) -> None:
        servable = _sequence_servable(model_type)

        assert servable.backend == "pytorch"
        assert set(servable.backends) == {"pytorch", ONNX_FP32_BACKEND, ONNX_INT8_BACKEND}
        fp32 = servable.parity[ONNX_FP32_BACKEND]
        assert fp32.passed and not fp32.skipped
        assert fp32.max_abs_diff <= FP32_MAX_ABS_DIFF
        int8 = servable.parity[ONNX_INT8_BACKEND]
        assert int8.passed
        assert int8.max_abs_diff <= INT8_MAX_ABS_DIFF
        assert int8.label_agreement >= INT8_MIN_LABEL_AGREEMENT

    @pytest.mark.parametrize("model_type", sorted(TINY_SEQUENCE_MODELS))
    def test_onnx_matches_native_on_fresh_inputs(self, model_type: str) -> None:
        servable = _sequence_servable(model_type)
        for batch_size in (1, 7):
            inputs = np.random.RandomState(batch_size).randn(batch_size, WINDOW, N_FEATURES)
            inputs = inputs.astype(np.float32)
            native = predict_proba(servable, inputs)
            onnx_fp32 = predict_proba(servable, inputs, ONNX_FP32_BACKEND)
            assert onnx_fp32.shape == (batch_size,)
            np.testing.assert_allclose(onnx_fp32, native, atol=FP32_MAX_ABS_DIFF)

    def test_exported_graph_has_dynamic_batch(self) -> None:
        model = TimeSeriesLSTM(N_FEATURES, hidden_size=8, num_layers=1, dropout=0.0).eval()
        session = OnnxSession(export_to_onnx(model, "lstm", (WINDOW, N_FEATURES)))
        batch_dim = session.session.get_inputs()[0].shape[0]
        assert isinstance(batch_dim, str)
        assert session.predict_proba(np.zeros((33, WINDOW, N_FEATURES), np.float32)).shape == (33,)

    def test_int8_quantizes_lstm_weights(self) -> None:
        model = TimeSeriesLSTM(N_FEATURES, hidden_size=16, num_layers=2, dropout=0.0).eval()
        quantized = quantize_int8(export_to_onnx(model, "lstm", (WINDOW, N_FEATURES)))
        op_types = {node.op_type for node in onnx.load_from_string(quantized).graph.node}
        assert "DynamicQuantizeLSTM" in op_types
        assert "LSTM" not in op_types


class TestTabularExport:
    """Export + parity for tree ensembles."""

    def test_xgboost_onnx_parity(self, tabular_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = tabular_data
        model = XGBClassifier(n_estimators=30, max_depth=4, verbosity=0).fit(x, y)
        metadata = {"model_type": "xgboost", "input_size": N_FEATURES}
        servable = create_servable("xgb", 1, uuid4(), _joblib_bytes(model), metadata)

        assert set(servable.backends) == {"xgboost", ONNX_FP32_BACKEND}
        assert servable.parity[ONNX_FP32_BACKEND].passed
        np.testing.assert_allclose(
            predict_proba(servable, x[:50], ONNX_FP32_BACKEND),
            model.predict_proba(x[:50])[:, 1],
            atol=FP32_MAX_ABS_DIFF,
        )

    def test_random_forest_onnx_parity(self, tabular_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = tabular_data
        model = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=0).fit(x, y)
        servable = create_servable(
            "rf", 1, uuid4(), _joblib_bytes(model), {"model_type": "random_forest"}
        )

        assert set(servable.backends) == {"sklearn", ONNX_FP32_BACKEND}
        assert servable.parity[ONNX_FP32_BACKEND].passed
        np.testing.assert_allclose(
            predict_proba(servable, x[:50], ONNX_FP32_BACKEND),
            model.predict_proba(x[:50])[:, 1],
            atol=FP32_MAX_ABS_DIFF,
        )

    def test_int8_skipped_for_tree_ensembles(self, tabular_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = tabular_data
        model = RandomForestClassifier(n_estimators=5, random_state=0).fit(x, y)
        with pytest.raises(QuantizationNotApplicableError):
            quantize_int8(export_to_onnx(model, "random_forest", (N_FEATURES,)))

        servable = create_servable(
            "rf", 1, uuid4(), _joblib_bytes(model), {"model_type": "random_forest"}
        )
        int8 = servable.parity[ONNX_INT8_BACKEND]
        assert int8.skipped and not int8.passed
        assert "int8 not applicable" in int8.reason
        assert ONNX_INT8_BACKEND not in servable.backends

    def test_parity_inputs_span_tree_splits(self, tabular_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = tabular_data
        model = XGBClassifier(n_estimators=10, max_depth=3, verbosity=0).fit(x, y)
        inputs = parity_inputs(model, "xgboost", (N_FEATURES,))
        # The large-scale feature is sampled on its own scale, not N(0, 1)
        assert np.abs(inputs[:, 4]).max() > 1e4
        assert len(np.unique(model.apply(inputs)[:, 0])) > 1

    def test_unsupported_model_type_raises(self) -> None:
        with pytest.raises(OnnxExportError):
            export_to_onnx(object(), "mystery", (N_FEATURES,))


class TestParityGate:
    """Backends that fail parity are never served."""

    def test_perturbed_session_is_not_served(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("forge.api.services.onnx_backend.OnnxSession", _ScaledSession)
        servable = _sequence_servable("tcn")

        assert set(servable.backends) == {"pytorch"}
        for backend in (ONNX_FP32_BACKEND, ONNX_INT8_BACKEND):
            result = servable.parity[backend]
            assert not result.passed and not result.skipped
            assert "max_abs_diff" in result.reason
        with pytest.raises(InvalidInputError, match="max_abs_diff"):
            predict_proba(servable, np.zeros((1, WINDOW, N_FEATURES), np.float32), ONNX_FP32_BACKEND)

    def test_label_flips_fail_int8_gate(self) -> None:
        inputs = np.zeros((100, 1), np.float32)
        native = np.full(100, 0.49, np.float32)
        flipped = native.copy()
        flipped[:2] = 0.51

        result = check_parity(
            ONNX_INT8_BACKEND, lambda batch: native[: len(batch)], lambda batch: flipped[: len(batch)],
            inputs, INT8_MAX_ABS_DIFF, INT8_MIN_LABEL_AGREEMENT,
        )
        assert result.max_abs_diff < INT8_MAX_ABS_DIFF
        assert not result.passed
        assert "label_agreement" in result.reason

    def test_export_failure_keeps_native_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def failing_export(model: object, model_type: str, input_shape: tuple[int, ...]) -> bytes:
            raise OnnxExportError("exporter exploded")

        monkeypatch.setattr("forge.api.services.onnx_backend.export_to_onnx", failing_export)
        servable = _sequence_servable("lstm")

        assert set(servable.backends) == {"pytorch"}
        assert servable.parity[ONNX_FP32_BACKEND].skipped
        assert servable.parity[ONNX_FP32_BACKEND].reason == "exporter exploded"


class TestBackendSelection:
    """Choosing a backend per request."""

    def test_default_is_native(self, lstm_servable: ServableModel) -> None:
        result = run_inference(lstm_servable, np.zeros((2, WINDOW, N_FEATURES)).tolist())
        assert result.backend == "pytorch"

    @pytest.mark.parametrize("backend", [ONNX_FP32_BACKEND, ONNX_INT8_BACKEND])
    def test_onnx_backend_selected(self, lstm_servable: ServableModel, backend: str) -> None:
        batch = np.random.RandomState(3).randn(4, WINDOW, N_FEATURES).tolist()
        native = run_inference(lstm_servable, batch)
        result = run_inference(lstm_servable, batch, backend)
        assert result.backend == backend
        np.testing.assert_allclose(result.probabilities, native.probabilities, atol=INT8_MAX_ABS_DIFF)

    def test_unknown_backend_rejected(self, lstm_servable: ServableModel) -> None:
        batch = np.zeros((1, WINDOW, N_FEATURES)).tolist()
        with pytest.raises(InvalidInputError, match="tensorrt"):
            run_inference(lstm_servable, batch, "tensorrt")
        with pytest.raises(InvalidInputError):
            predict_proba(lstm_servable, np.zeros((1, WINDOW, N_FEATURES), np.float32), "tensorrt")

    def test_session_threads_follow_torch(self) -> None:
        model = TimeSeriesTCN(N_FEATURES, num_channels=4, num_layers=1).eval()
        session = OnnxSession(export_to_onnx(model, "tcn", (WINDOW, N_FEATURES)))
        assert session.num_threads == torch.get_num_threads()


class TestServingApi:
    """Backend selection and parity reporting over HTTP."""

    @pytest.fixture
    def client(self, lstm_servable: ServableModel, monkeypatch: pytest.MonkeyPatch) -> TestClient:
        monkeypatch.setattr(serving, "get_production_servable", lambda db, name: lstm_servable)
        app = FastAPI()
        app.include_router(serving.router)
        app.dependency_overrides[get_db] = lambda: None
        return TestClient(app)

    def test_predict_with_backend_query_param(self, client: TestClient) -> None:
        payload = {"inputs": np.zeros((2, WINDOW, N_FEATURES)).tolist()}
        default = client.post("/api/serve/lstm/predict", json=payload)
        int8 = client.post(f"/api/serve/lstm/predict?backend={ONNX_INT8_BACKEND}", json=payload)

        assert default.status_code == 200 and default.json()["backend"] == "pytorch"
        assert int8.status_code == 200 and int8.json()["backend"] == ONNX_INT8_BACKEND
        assert len(int8.json()["predictions"]) == 2

    def test_unknown_backend_is_422(self, client: TestClient) -> None:
        payload = {"inputs": np.zeros((1, WINDOW, N_FEATURES)).tolist()}
        response = client.post("/api/serve/lstm/predict?backend=tensorrt", json=payload)
        assert response.status_code == 422
        assert "tensorrt" in response.json()["detail"]

    def test_describe_lists_backends_and_parity(self, client: TestClient) -> None:
        body = client.get("/api/serve/lstm").json()
        assert body["backend"] == "pytorch"
        assert body["backends"] == ["pytorch", ONNX_FP32_BACKEND, ONNX_INT8_BACKEND]
        parity = {entry["backend"]: entry for entry in body["parity"]}
        assert parity[ONNX_FP32_BACKEND]["passed"] is True
        assert parity[ONNX_INT8_BACKEND]["max_abs_diff_limit"] == INT8_MAX_ABS_DIFF
        assert parity[ONNX_INT8_BACKEND]["min_label_agreement"] == INT8_MIN_LABEL_AGREEMENT
        assert body["onnx_build_seconds"] > 0
