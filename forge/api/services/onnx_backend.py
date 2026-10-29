"""ONNX Runtime inference backends — export, int8 quantization, and parity gating.

A served model can answer requests through its native framework or through
ONNX Runtime. At load time the native model is exported to ONNX, optionally
dynamically quantized to int8, and each ONNX session is checked against the
native model on sample inputs. Only sessions that pass the parity gate are
served; the rest are reported with the reason they were held back.

Every exported graph has the same contract: one float32 input ``inputs``
shaped ``[batch, *input_shape]`` and one output ``probabilities`` shaped
``[batch, 1]`` holding the positive-class probability.
"""

import logging
import tempfile
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxmltools
import onnxruntime as ort
import torch
from onnx import helper
from onnxmltools.convert.common.data_types import FloatTensorType as XgbFloatTensorType
from onnxruntime.capi import onnxruntime_pybind11_state as ort_errors
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType
from torch import nn

from forge.api.services.training import SEQUENCE_MODEL_TYPES

logger = logging.getLogger(__name__)

# The dynamo exporter logs a warning per export for every torchvision op it
# can't register; Forge models never use torchvision ops.
logging.getLogger("torch.onnx._internal.exporter._registration").setLevel(logging.ERROR)

ONNX_FP32_BACKEND = "onnxruntime"
ONNX_INT8_BACKEND = "onnxruntime-int8"

INPUT_NAME = "inputs"
OUTPUT_NAME = "probabilities"

# Parity gates. fp32 ONNX must reproduce the native model up to float
# rounding. Dynamic int8 quantization changes the arithmetic, so it is held
# to a classification gate instead: probabilities stay close and predicted
# labels (p >= 0.5) almost never flip.
FP32_MAX_ABS_DIFF = 1e-4
INT8_MAX_ABS_DIFF = 0.05
INT8_MIN_LABEL_AGREEMENT = 0.99
PARITY_SAMPLES = 256
PARITY_SEED = 1234

# Opset for the positive-class Slice added to converted tree ensembles
_DEFAULT_DOMAIN_OPSET = 17

# Ops dynamic quantization rewrites to integer kernels. A graph with none of
# them (tree ensembles) has nothing to quantize.
_QUANTIZABLE_OPS = {"MatMul", "Gemm", "Conv", "LSTM", "GRU", "Attention"}

# Errors ONNX Runtime raises when a graph can't be loaded or run
_ORT_ERRORS: tuple[type[Exception], ...] = (
    ort_errors.Fail,
    ort_errors.InvalidArgument,
    ort_errors.InvalidGraph,
    ort_errors.InvalidProtobuf,
    ort_errors.NotImplemented,
    ort_errors.RuntimeException,
)


class OnnxExportError(Exception):
    """The model couldn't be converted to an ONNX graph or session."""


class QuantizationNotApplicableError(Exception):
    """The graph has no ops that dynamic int8 quantization can rewrite."""


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _export_torch(model: nn.Module, input_shape: tuple[int, ...]) -> onnx.ModelProto:
    """Export a sequence model with the torch.export-based (dynamo) exporter.

    The example batch has 2 rows: torch.export specializes dimensions of size
    0 and 1, so a batch-1 example would freeze the batch axis.
    """
    example = torch.zeros((2, *input_shape), dtype=torch.float32)
    with warnings.catch_warnings():
        # Tracing LSTMs emits size-check and deprecation warnings that don't
        # affect the exported graph; parity checks verify the result.
        warnings.simplefilter("ignore")
        program = torch.onnx.export(
            model.eval(),
            (example,),
            dynamo=True,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_shapes=({0: torch.export.Dim("batch")},),
            verbose=False,
        )
    return program.model_proto


def _select_positive_class(model_proto: onnx.ModelProto) -> onnx.ModelProto:
    """Rewire a classifier graph to output only the positive-class probability.

    Converted tree ensembles output ``label`` and ``probabilities`` ``[batch, 2]``;
    this slices column 1 into a single ``[batch, 1]`` output.
    """
    graph = model_proto.graph
    class_probs = next(out.name for out in graph.output if out.name == OUTPUT_NAME)
    renamed = f"{class_probs}_all_classes"
    for node in graph.node:
        node.output[:] = [renamed if name == class_probs else name for name in node.output]

    slice_args = {"starts": [1], "ends": [2], "axes": [1]}
    for arg_name, values in slice_args.items():
        graph.initializer.append(
            helper.make_tensor(f"positive_class_{arg_name}", onnx.TensorProto.INT64, [1], values)
        )
    graph.node.append(
        helper.make_node(
            "Slice",
            [renamed, *(f"positive_class_{arg_name}" for arg_name in slice_args)],
            [OUTPUT_NAME],
            name="positive_class",
        )
    )
    # onnxmltools' xgboost graphs only import the ai.onnx.ml domain
    if not any(opset.domain in ("", "ai.onnx") for opset in model_proto.opset_import):
        model_proto.opset_import.append(helper.make_opsetid("", _DEFAULT_DOMAIN_OPSET))
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info(OUTPUT_NAME, onnx.TensorProto.FLOAT, ["batch", 1])
    )
    onnx.checker.check_model(model_proto)
    return model_proto


def _export_xgboost(model: object, n_features: int) -> onnx.ModelProto:
    """Convert an XGBClassifier with onnxmltools."""
    return onnxmltools.convert_xgboost(
        model, initial_types=[(INPUT_NAME, XgbFloatTensorType([None, n_features]))]
    )


def _export_random_forest(model: object, n_features: int) -> onnx.ModelProto:
    """Convert a scikit-learn classifier with skl2onnx (tensor output, no ZipMap)."""
    return convert_sklearn(
        model,
        initial_types=[(INPUT_NAME, SklFloatTensorType([None, n_features]))],
        options={id(model): {"zipmap": False}},
    )


def export_to_onnx(model: object, model_type: str, input_shape: tuple[int, ...]) -> bytes:
    """Export a trained model to a serialized ONNX graph.

    Args:
        model: The native model (torch module in eval mode, XGBClassifier, or
            RandomForestClassifier).
        model_type: Forge model type, e.g. ``lstm`` or ``xgboost``.
        input_shape: Per-example input shape (``(window, n_features)`` or ``(n_features,)``).

    Returns:
        ONNX model bytes with a dynamic batch axis and a ``[batch, 1]``
        positive-class probability output.

    Raises:
        OnnxExportError: If the model type has no exporter or conversion fails.
    """
    try:
        if model_type in SEQUENCE_MODEL_TYPES:
            model_proto = _export_torch(model, input_shape)
        elif model_type == "xgboost":
            model_proto = _select_positive_class(_export_xgboost(model, input_shape[0]))
        elif model_type == "random_forest":
            model_proto = _select_positive_class(_export_random_forest(model, input_shape[0]))
        else:
            raise OnnxExportError(f"No ONNX exporter for model type '{model_type}'")
    except (
        torch.onnx.OnnxExporterError,
        onnx.checker.ValidationError,
        RuntimeError,
        ValueError,
        TypeError,
        NotImplementedError,
        StopIteration,
    ) as exc:
        raise OnnxExportError(f"Cannot export {model_type} to ONNX: {exc}") from exc
    return model_proto.SerializeToString()


# ---------------------------------------------------------------------------
# Int8 quantization
# ---------------------------------------------------------------------------


def quantize_int8(onnx_bytes: bytes) -> bytes:
    """Dynamically quantize a graph's weights to int8.

    Weights of MatMul/Gemm/Conv/LSTM ops are stored as int8 and activations
    are quantized per batch at run time, so no calibration data is needed.
    The graph is first run through ONNX Runtime's quantization pre-processing
    (shape inference plus constant folding); without it the dynamo exporter's
    LSTM weight re-ordering isn't folded and the LSTM layers stay fp32.

    Raises:
        QuantizationNotApplicableError: If the graph has no quantizable ops
            (tree ensembles) or quantization rewrote none of them.
        OnnxExportError: If pre-processing or quantization fails.
    """
    model_proto = onnx.load_from_string(onnx_bytes)
    original_ops = {node.op_type for node in model_proto.graph.node}
    if not original_ops & _QUANTIZABLE_OPS:
        raise QuantizationNotApplicableError(
            "graph has no MatMul/Gemm/Conv/LSTM weights to quantize "
            f"(ops: {', '.join(sorted(original_ops))})"
        )

    with tempfile.TemporaryDirectory() as workdir:
        source_path = Path(workdir) / "model.onnx"
        preprocessed_path = Path(workdir) / "preprocessed.onnx"
        quantized_path = Path(workdir) / "quantized.onnx"
        source_path.write_bytes(onnx_bytes)
        try:
            quant_pre_process(str(source_path), str(preprocessed_path))
            quantize_dynamic(str(preprocessed_path), str(quantized_path), weight_type=QuantType.QInt8)
        except (
            *_ORT_ERRORS,
            onnx.shape_inference.InferenceError,
            RuntimeError,
            ValueError,
            AssertionError,
        ) as exc:
            raise OnnxExportError(f"int8 quantization failed: {exc}") from exc
        quantized_bytes = quantized_path.read_bytes()

    quantized_ops = {node.op_type for node in onnx.load_from_string(quantized_bytes).graph.node}
    if not quantized_ops - original_ops:
        raise QuantizationNotApplicableError("quantization left every op in fp32")
    return quantized_bytes


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class OnnxSession:
    """An ONNX Runtime CPU session that returns positive-class probabilities.

    Intra-op threads match torch's pool, which runtime.py sizes to the
    container's CPU quota, so both backends get the same compute.
    """

    def __init__(self, onnx_bytes: bytes) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = torch.get_num_threads()
        options.inter_op_num_threads = 1
        # By default idle ORT worker threads spin-wait between requests. That
        # cuts batch-1 latency but burns ~3 cores per 4 threads while idle,
        # which eats the container's CPU quota and starves the other backends.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        try:
            self.session = ort.InferenceSession(
                onnx_bytes, sess_options=options, providers=["CPUExecutionProvider"]
            )
        except _ORT_ERRORS as exc:
            raise OnnxExportError(f"ONNX Runtime cannot load the graph: {exc}") from exc
        self.num_threads = options.intra_op_num_threads

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        """Return the positive-class probability per row: float32 in, 1-D out."""
        batch = np.ascontiguousarray(inputs, dtype=np.float32)
        (probabilities,) = self.session.run([OUTPUT_NAME], {INPUT_NAME: batch})
        return probabilities.reshape(-1)


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@dataclass
class ParityResult:
    """Outcome of comparing an ONNX backend against the native model.

    A skipped backend was never built (export failed, or int8 doesn't apply
    to the model); ``reason`` says why. A failed backend was built but its
    outputs fell outside the gate; ``reason`` names the gate it missed.
    """

    backend: str
    passed: bool
    skipped: bool = False
    max_abs_diff: float | None = None
    label_agreement: float | None = None
    max_abs_diff_limit: float | None = None
    min_label_agreement: float | None = None
    samples: int = 0
    reason: str | None = None


def _skipped(backend: str, reason: str) -> ParityResult:
    """Parity entry for a backend that couldn't be built."""
    return ParityResult(backend=backend, passed=False, skipped=True, reason=reason)


def _tree_split_ranges(model: object, n_features: int) -> dict[int, tuple[float, float]]:
    """Return the (min, max) split threshold per feature of a tree ensemble."""
    feature_ids: list[np.ndarray] = []
    thresholds: list[np.ndarray] = []
    if hasattr(model, "estimators_"):
        for estimator in model.estimators_:
            tree = estimator.tree_
            is_split = tree.feature >= 0
            feature_ids.append(tree.feature[is_split])
            thresholds.append(tree.threshold[is_split])
    elif hasattr(model, "get_booster"):
        booster = model.get_booster()
        splits = booster.trees_to_dataframe()
        splits = splits[splits["Feature"] != "Leaf"]
        names = booster.feature_names or [f"f{index}" for index in range(n_features)]
        index_of = {name: index for index, name in enumerate(names)}
        feature_ids.append(splits["Feature"].map(index_of).to_numpy(dtype=np.int64))
        thresholds.append(splits["Split"].to_numpy(dtype=np.float64))
    if not feature_ids:
        return {}

    all_features = np.concatenate(feature_ids)
    all_thresholds = np.concatenate(thresholds)
    return {
        int(feature): (
            float(all_thresholds[all_features == feature].min()),
            float(all_thresholds[all_features == feature].max()),
        )
        for feature in np.unique(all_features)
    }


def parity_inputs(
    model: object, model_type: str, input_shape: tuple[int, ...], samples: int = PARITY_SAMPLES
) -> np.ndarray:
    """Generate deterministic sample inputs for the parity check.

    No real data is available at load time. Sequence models get standard
    normal windows. Tree ensembles get each feature drawn uniformly across
    the range of its split thresholds (plus a 10% margin) so the samples
    reach many leaves; standard normal inputs on raw features such as volume
    would all land on the same side of every split and make parity vacuous.
    """
    rng = np.random.default_rng(PARITY_SEED)
    inputs = rng.standard_normal((samples, *input_shape)).astype(np.float32)
    if model_type in SEQUENCE_MODEL_TYPES:
        return inputs
    for feature, (low, high) in _tree_split_ranges(model, input_shape[0]).items():
        margin = max(0.1 * (high - low), 1e-3 * max(abs(low), abs(high), 1.0))
        inputs[:, feature] = rng.uniform(low - margin, high + margin, samples).astype(np.float32)
    return inputs


def check_parity(
    backend: str,
    native_predict: Callable[[np.ndarray], np.ndarray],
    candidate_predict: Callable[[np.ndarray], np.ndarray],
    sample_inputs: np.ndarray,
    max_abs_diff_limit: float,
    min_label_agreement: float | None = None,
) -> ParityResult:
    """Compare a candidate backend with the native model on sample inputs.

    Gates (constants in this module):

    - ``onnxruntime`` (fp32): max |p_native - p_onnx| <= 1e-4. No label gate,
      since a 1e-4 difference can legitimately flip a probability sitting on
      0.5, and anything bigger already fails the probability gate.
    - ``onnxruntime-int8``: max |p_native - p_int8| <= 0.05 AND label
      agreement (p >= 0.5) >= 99%. With 256 samples, at most 2 labels may flip.

    The full sample batch and a single-row batch are both checked, so an
    export that froze the batch size fails parity.
    """
    native = np.concatenate([native_predict(sample_inputs[:1]), native_predict(sample_inputs)])
    candidate = np.concatenate(
        [candidate_predict(sample_inputs[:1]), candidate_predict(sample_inputs)]
    )
    result = ParityResult(
        backend=backend,
        passed=False,
        max_abs_diff_limit=max_abs_diff_limit,
        min_label_agreement=min_label_agreement,
        samples=len(sample_inputs),
    )
    if native.shape != candidate.shape:
        result.reason = f"output shape {list(candidate.shape)} != native {list(native.shape)}"
        return result
    if not np.isfinite(candidate).all():
        result.reason = "output contains NaN or infinite values"
        return result

    result.max_abs_diff = float(np.max(np.abs(native.astype(np.float64) - candidate.astype(np.float64))))
    result.label_agreement = float(np.mean((native >= 0.5) == (candidate >= 0.5)))
    if result.max_abs_diff > max_abs_diff_limit:
        result.reason = f"max_abs_diff {result.max_abs_diff:.3g} > {max_abs_diff_limit:g}"
    elif min_label_agreement is not None and result.label_agreement < min_label_agreement:
        result.reason = f"label_agreement {result.label_agreement:.3f} < {min_label_agreement:g}"
    else:
        result.passed = True
    return result


# ---------------------------------------------------------------------------
# Load-time entry point
# ---------------------------------------------------------------------------


@dataclass
class OnnxBackends:
    """ONNX sessions that passed parity, plus the parity report for every candidate."""

    sessions: dict[str, OnnxSession]
    parity: dict[str, ParityResult]
    export_seconds: float = 0.0
    total_seconds: float = 0.0


def build_onnx_backends(
    model: object,
    model_type: str,
    input_shape: tuple[int, ...],
    native_predict: Callable[[np.ndarray], np.ndarray],
) -> OnnxBackends:
    """Export, quantize, and parity-check ONNX backends for a loaded model.

    Never raises for an unexportable model: the native backend keeps serving
    and the failure is recorded in the parity report. A backend that fails
    parity is left out of ``sessions``.

    Args:
        model: The native model.
        model_type: Forge model type.
        input_shape: Per-example input shape.
        native_predict: The native backend's positive-class predictor.
    """
    start = time.perf_counter()
    backends = OnnxBackends(sessions={}, parity={})

    try:
        fp32_bytes = export_to_onnx(model, model_type, input_shape)
    except OnnxExportError as exc:
        logger.warning("ONNX backends unavailable for %s: %s", model_type, exc)
        for backend in (ONNX_FP32_BACKEND, ONNX_INT8_BACKEND):
            backends.parity[backend] = _skipped(backend, str(exc))
        backends.total_seconds = time.perf_counter() - start
        return backends

    int8_skip: ParityResult | None = None

    candidates: dict[str, tuple[bytes, float, float | None]] = {
        ONNX_FP32_BACKEND: (fp32_bytes, FP32_MAX_ABS_DIFF, None),
    }
    try:
        candidates[ONNX_INT8_BACKEND] = (
            quantize_int8(fp32_bytes), INT8_MAX_ABS_DIFF, INT8_MIN_LABEL_AGREEMENT,
        )
    except QuantizationNotApplicableError as exc:
        int8_skip = _skipped(ONNX_INT8_BACKEND, f"int8 not applicable: {exc}")
    except OnnxExportError as exc:
        int8_skip = _skipped(ONNX_INT8_BACKEND, str(exc))
    backends.export_seconds = time.perf_counter() - start

    sample_inputs = parity_inputs(model, model_type, input_shape)
    for backend, (onnx_bytes, max_diff, min_agreement) in candidates.items():
        try:
            session = OnnxSession(onnx_bytes)
            result = check_parity(
                backend, native_predict, session.predict_proba, sample_inputs, max_diff, min_agreement
            )
        except (OnnxExportError, *_ORT_ERRORS) as exc:
            result = _skipped(backend, str(exc))
        backends.parity[backend] = result
        if result.passed:
            backends.sessions[backend] = session
        else:
            logger.warning("Not serving %s for %s: %s", backend, model_type, result.reason)
    if int8_skip is not None:
        backends.parity[ONNX_INT8_BACKEND] = int8_skip

    backends.total_seconds = time.perf_counter() - start
    logger.info(
        "ONNX backends for %s: export %.2fs, total %.2fs, serving %s",
        model_type, backends.export_seconds, backends.total_seconds, sorted(backends.sessions) or "none",
    )
    return backends
