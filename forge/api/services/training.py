"""Model training service — trains XGBoost, Random Forest, and LSTM models.

Handles time-series aware splitting, model training, evaluation,
hardware profiling, and storing results in the database.
"""

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sqlalchemy.orm import Session
from torch import nn
from xgboost import XGBClassifier

from forge.api.models.database import Dataset, Experiment, Run
from forge.api.services.embeddings import embed_run
from forge.api.services.metrics import (
    EXPERIMENTS_TOTAL,
    INFERENCE_LATENCY_SECONDS,
    TRAINING_DURATION_SECONDS,
)
from forge.api.services.profiler import profile_model
from forge.api.services.reproducibility import (
    capture_environment,
    compute_data_hash,
    set_all_seeds,
    store_environment,
)
from forge.api.services.s3_client import upload_model_artifact
from forge.api.services.wandb_tracker import WandbTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time-series split — chronological, no future leakage
# ---------------------------------------------------------------------------


def time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into train / val / test sets.

    The data must already be sorted by time (oldest first). This function
    simply slices — no shuffling — to prevent future data leakage.

    Args:
        df: Time-ordered DataFrame.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Target engineering — binary classification: price goes up (1) or down (0)
# ---------------------------------------------------------------------------


def create_target(df: pd.DataFrame, column: str = "Close") -> pd.DataFrame:
    """Add a binary target column: 1 if next-day close > today's close, else 0.

    Drops the last row (which has no future to predict).
    """
    df = df.copy()
    df["target"] = (df[column].shift(-1) > df[column]).astype(int)
    df = df.iloc[:-1]  # last row has no target
    return df


# ---------------------------------------------------------------------------
# Feature / target extraction helpers
# ---------------------------------------------------------------------------

EXCLUDE_COLS = {"Date", "Open", "High", "Low", "Close", "Volume", "ticker", "target"}


def extract_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y from a prepared DataFrame."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    x = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    # Replace NaN/Inf with 0 for model safety
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y


# ---------------------------------------------------------------------------
# LSTM model definition
# ---------------------------------------------------------------------------


class TimeSeriesLSTM(nn.Module):
    """2-layer LSTM for binary classification on financial time-series."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x shape (batch, seq_len, input_size) → (batch, 1)."""
        _lstm_out, (hidden, _cell) = self.lstm(x)
        last_hidden = hidden[-1]
        output = self.sigmoid(self.fc(last_hidden))
        return output


# ---------------------------------------------------------------------------
# TCN — Temporal Convolutional Network with causal dilated convolutions
# ---------------------------------------------------------------------------


class CausalConv1dBlock(nn.Module):
    """Single causal convolution block with dilation, residual connection, and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal trimming: (batch, channels, seq_len) → same shape."""
        out = self.conv(x)
        # Trim future timesteps to enforce causality
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)
        return out + self.residual(x)


class TimeSeriesTCN(nn.Module):
    """Temporal Convolutional Network for binary classification.

    Uses exponentially increasing dilation rates (1, 2, 4, 8...) so the
    receptive field covers the full input window without deep stacking.
    """

    def __init__(
        self,
        input_size: int,
        num_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else num_channels
            layers.append(CausalConv1dBlock(in_ch, num_channels, kernel_size, dilation=2**i, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: (batch, seq_len, features) → (batch, 1)."""
        # Conv1d expects (batch, channels, seq_len)
        out = x.transpose(1, 2)
        out = self.network(out)
        # Global average pooling over time dimension
        out = out.mean(dim=2)
        return self.sigmoid(self.fc(out))


# ---------------------------------------------------------------------------
# CNN-LSTM — Conv1D feature extractor → LSTM sequence model
# ---------------------------------------------------------------------------


class TimeSeriesCNNLSTM(nn.Module):
    """Hybrid Conv1D + LSTM for binary classification.

    Conv1D layers extract local patterns (e.g., candlestick formations),
    then LSTM captures temporal dependencies across the extracted features.
    """

    def __init__(
        self,
        input_size: int,
        cnn_filters: int = 64,
        cnn_kernel_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: (batch, seq_len, features) → (batch, 1)."""
        # CNN expects (batch, channels, seq_len)
        cnn_out = self.cnn(x.transpose(1, 2))
        # Back to (batch, seq_len, channels) for LSTM
        lstm_in = cnn_out.transpose(1, 2)
        _, (hidden, _) = self.lstm(lstm_in)
        return self.sigmoid(self.fc(hidden[-1]))


# ---------------------------------------------------------------------------
# Transformer encoder for time-series classification
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence ordering."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2]) if d_model % 2 else torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding: (batch, seq_len, d_model) → same shape."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder for binary classification on time-series.

    Projects input features to d_model dimensions, adds sinusoidal
    positional encoding, then applies multi-head self-attention layers.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: (batch, seq_len, features) → (batch, 1)."""
        out = self.input_proj(x)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)
        # Mean pooling over the sequence dimension
        out = out.mean(dim=1)
        return self.sigmoid(self.fc(out))


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hyperparams: dict,
) -> XGBClassifier:
    """Train an XGBoost classifier with early stopping on validation set."""
    params = {
        "n_estimators": hyperparams.get("n_estimators", 500),
        "max_depth": hyperparams.get("max_depth", 6),
        "learning_rate": hyperparams.get("learning_rate", 0.1),
        "eval_metric": "logloss",
        "early_stopping_rounds": hyperparams.get("early_stopping_rounds", 20),
        "verbosity": 0,
    }
    model = XGBClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def train_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hyperparams: dict,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    params = {
        "n_estimators": hyperparams.get("n_estimators", 200),
        "max_depth": hyperparams.get("max_depth", 10),
        "min_samples_split": hyperparams.get("min_samples_split", 5),
        "n_jobs": -1,
        "random_state": 42,
    }
    model = RandomForestClassifier(**params)
    model.fit(x_train, y_train)
    return model


def _build_lstm_sequences(
    x: np.ndarray, y: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flat feature arrays into overlapping sequences for LSTM input.

    Produces arrays of shape (n_samples - window_size, window_size, n_features)
    and (n_samples - window_size,).
    """
    sequences, targets = [], []
    for i in range(window_size, len(x)):
        sequences.append(x[i - window_size : i])
        targets.append(y[i])
    return np.array(sequences), np.array(targets)


def _train_pytorch_model(
    model: nn.Module,
    x_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    x_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    hyperparams: dict,
    epoch_callback: Callable[[int, float, float], None] | None = None,
) -> nn.Module:
    """Generic PyTorch training loop for binary classification models.

    Handles mini-batch training with BCE loss, validation, early stopping,
    and optional epoch callbacks for W&B logging. Works for any nn.Module
    that takes (batch, seq_len, features) input and returns (batch, 1) output.
    """
    lr = hyperparams.get("learning_rate", 0.001)
    epochs = hyperparams.get("epochs", 50)
    batch_size = hyperparams.get("batch_size", 32)
    patience = hyperparams.get("patience", 5)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    x_train_t = torch.tensor(x_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
    x_val_t = torch.tensor(x_val_seq, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1)

    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        indices = torch.randperm(len(x_train_t))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(x_train_t), batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = x_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_preds = model(x_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
        model.train()

        avg_train_loss = epoch_loss / max(n_batches, 1)
        logger.info(
            "Epoch %d/%d — train_loss=%.4f val_loss=%.4f",
            epoch + 1, epochs, avg_train_loss, val_loss,
        )

        if epoch_callback is not None:
            epoch_callback(epoch + 1, avg_train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    model.eval()
    return model


def train_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hyperparams: dict,
    epoch_callback: Callable[[int, float, float], None] | None = None,
) -> TimeSeriesLSTM:
    """Train a 2-layer LSTM for binary classification.

    Converts flat features into windowed sequences, trains with BCE loss,
    and uses early stopping based on validation loss.
    """
    window_size = hyperparams.get("window_size", 30)
    hidden_size = hyperparams.get("hidden_size", 128)
    num_layers = hyperparams.get("num_layers", 2)
    dropout = hyperparams.get("dropout", 0.2)

    x_train_seq, y_train_seq = _build_lstm_sequences(x_train, y_train, window_size)
    x_val_seq, y_val_seq = _build_lstm_sequences(x_val, y_val, window_size)

    input_size = x_train_seq.shape[2]
    model = TimeSeriesLSTM(input_size, hidden_size, num_layers, dropout)
    return _train_pytorch_model(model, x_train_seq, y_train_seq, x_val_seq, y_val_seq, hyperparams, epoch_callback)


def train_tcn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hyperparams: dict,
    epoch_callback: Callable[[int, float, float], None] | None = None,
) -> TimeSeriesTCN:
    """Train a Temporal Convolutional Network for binary classification.

    Converts flat features into windowed sequences, then trains with
    causal dilated convolutions and early stopping.
    """
    window_size = hyperparams.get("window_size", 30)
    num_channels = hyperparams.get("num_channels", 64)
    num_layers = hyperparams.get("num_layers", 4)
    kernel_size = hyperparams.get("kernel_size", 3)
    dropout = hyperparams.get("dropout", 0.2)

    x_train_seq, y_train_seq = _build_lstm_sequences(x_train, y_train, window_size)
    x_val_seq, y_val_seq = _build_lstm_sequences(x_val, y_val, window_size)

    input_size = x_train_seq.shape[2]
    model = TimeSeriesTCN(input_size, num_channels, num_layers, kernel_size, dropout)
    return _train_pytorch_model(model, x_train_seq, y_train_seq, x_val_seq, y_val_seq, hyperparams, epoch_callback)


def train_cnn_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hyperparams: dict,
    epoch_callback: Callable[[int, float, float], None] | None = None,
) -> TimeSeriesCNNLSTM:
    """Train a CNN-LSTM hybrid for binary classification.

    Conv1D extracts local patterns, LSTM captures temporal dependencies.
    """
    window_size = hyperparams.get("window_size", 30)
    cnn_filters = hyperparams.get("cnn_filters", 64)
    cnn_kernel_size = hyperparams.get("cnn_kernel_size", 3)
    lstm_hidden = hyperparams.get("lstm_hidden", 64)
    lstm_layers = hyperparams.get("lstm_layers", 1)
    dropout = hyperparams.get("dropout", 0.2)

    x_train_seq, y_train_seq = _build_lstm_sequences(x_train, y_train, window_size)
    x_val_seq, y_val_seq = _build_lstm_sequences(x_val, y_val, window_size)

    input_size = x_train_seq.shape[2]
    model = TimeSeriesCNNLSTM(input_size, cnn_filters, cnn_kernel_size, lstm_hidden, lstm_layers, dropout)
    return _train_pytorch_model(model, x_train_seq, y_train_seq, x_val_seq, y_val_seq, hyperparams, epoch_callback)


def train_transformer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hyperparams: dict,
    epoch_callback: Callable[[int, float, float], None] | None = None,
) -> TimeSeriesTransformer:
    """Train a Transformer encoder for binary classification on time-series.

    Uses sinusoidal positional encoding + multi-head self-attention.
    """
    window_size = hyperparams.get("window_size", 30)
    d_model = hyperparams.get("d_model", 64)
    nhead = hyperparams.get("nhead", 4)
    num_layers = hyperparams.get("num_layers", 2)
    dim_feedforward = hyperparams.get("dim_feedforward", 128)
    dropout = hyperparams.get("dropout", 0.1)

    x_train_seq, y_train_seq = _build_lstm_sequences(x_train, y_train, window_size)
    x_val_seq, y_val_seq = _build_lstm_sequences(x_val, y_val, window_size)

    input_size = x_train_seq.shape[2]
    model = TimeSeriesTransformer(input_size, d_model, nhead, num_layers, dim_feedforward, dropout)
    return _train_pytorch_model(model, x_train_seq, y_train_seq, x_val_seq, y_val_seq, hyperparams, epoch_callback)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: object,
    x_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int | None = None,
) -> dict:
    """Compute classification metrics on the test set.

    Returns dict with accuracy, precision, recall, and f1.
    """
    if isinstance(model, torch.nn.Module):
        x_seq, y_seq = _build_lstm_sequences(x_test, y_test, window_size or 30)
        with torch.no_grad():
            preds_prob = model(torch.tensor(x_seq, dtype=torch.float32))
            preds = (preds_prob.squeeze().numpy() >= 0.5).astype(int)
        y_true = y_seq.astype(int)
    else:
        preds = model.predict(x_test).astype(int)
        y_true = y_test.astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Run orchestrator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Default hyperparameter configs — stored with runs for reproducibility
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[str, dict] = {
    "xgboost": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.1, "early_stopping_rounds": 20},
    "random_forest": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5},
    "lstm": {"window_size": 30, "hidden_size": 128, "num_layers": 2, "dropout": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32},
    "tcn": {"window_size": 30, "num_channels": 64, "num_layers": 4, "kernel_size": 3, "dropout": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32},
    "cnn_lstm": {"window_size": 30, "cnn_filters": 64, "cnn_kernel_size": 3, "lstm_hidden": 64, "lstm_layers": 1, "dropout": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32},
    "transformer": {"window_size": 30, "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1, "learning_rate": 0.001, "epochs": 50, "batch_size": 32},
}

# Model types that use windowed sequence input
SEQUENCE_MODEL_TYPES = {"lstm", "tcn", "cnn_lstm", "transformer"}


def _make_trainers(
    epoch_callback: Callable[[int, float, float], None] | None = None,
) -> dict:
    """Build trainer dispatch dict, threading epoch_callback to PyTorch models."""
    return {
        "xgboost": lambda xt, yt, xv, yv, hp: train_xgboost(xt, yt, xv, yv, hp),
        "random_forest": lambda xt, yt, xv, yv, hp: train_random_forest(xt, yt, hp),
        "lstm": lambda xt, yt, xv, yv, hp: train_lstm(xt, yt, xv, yv, hp, epoch_callback),
        "tcn": lambda xt, yt, xv, yv, hp: train_tcn(xt, yt, xv, yv, hp, epoch_callback),
        "cnn_lstm": lambda xt, yt, xv, yv, hp: train_cnn_lstm(xt, yt, xv, yv, hp, epoch_callback),
        "transformer": lambda xt, yt, xv, yv, hp: train_transformer(xt, yt, xv, yv, hp, epoch_callback),
    }


def run_experiment_run(run_id: UUID, db: Session) -> Run:
    """Execute a single training run: load data → split → train → profile → log → store.

    Integrates W&B tracking and S3 artifact upload. Both are optional —
    if credentials are missing, training still completes normally.

    Args:
        run_id: The UUID of the Run record to execute.
        db: Active database session.

    Returns:
        The updated Run record with metrics and profiling results.
    """
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise ValueError(f"Run {run_id} not found")

    experiment = db.query(Experiment).filter(Experiment.id == run.experiment_id).first()
    if experiment is None:
        raise ValueError(f"Experiment {run.experiment_id} not found")

    dataset = db.query(Dataset).filter(Dataset.id == experiment.dataset_id).first()
    if dataset is None:
        raise ValueError(f"Dataset {experiment.dataset_id} not found")

    # Mark run as started
    run.status = "running"
    run.started_at = datetime.now(UTC)
    db.commit()

    # Reproducibility: set seeds before anything else (including model init)
    hyperparams = run.hyperparameters or {}
    seed = hyperparams.get("random_seed", 42)
    set_all_seeds(seed)

    # Capture and store environment snapshot
    env_snapshot = capture_environment(random_seed=seed)
    store_environment(db, run_id, env_snapshot)
    db.commit()

    # Initialize W&B tracker (no-op if disabled)
    tracker = WandbTracker()
    tracker.init_run(
        project="forge",
        experiment_name=experiment.name,
        model_type=run.model_type,
        hyperparameters=hyperparams,
        tags=[run.model_type, dataset.name],
    )

    try:
        # Load data — from feature store if feature_set_id is set, else raw parquet
        if run.feature_set_id is not None:
            from forge.api.services.feature_store import get_features
            logger.info("Loading features from feature store (fs=%s, ds=%s)", run.feature_set_id, experiment.dataset_id)
            df = get_features(db, run.feature_set_id, experiment.dataset_id)
        else:
            df = pd.read_parquet(dataset.s3_path)
        df = create_target(df)

        # Reproducibility: compute and store data version hash
        run.data_version_hash = compute_data_hash(df)

        # Time-series split
        train_df, val_df, test_df = time_series_split(df)
        x_train, y_train = extract_xy(train_df)
        x_val, y_val = extract_xy(val_df)
        x_test, y_test = extract_xy(test_df)

        # Train — pass epoch callback for W&B logging
        model_type = run.model_type
        trainers = _make_trainers(epoch_callback=tracker.log_epoch_metrics)
        if model_type not in trainers:
            raise ValueError(f"Unsupported model type: {model_type}")

        train_start = time.time()
        model = trainers[model_type](x_train, y_train, x_val, y_val, hyperparams)
        training_time = time.time() - train_start

        # Evaluate
        window_size = hyperparams.get("window_size", 30) if model_type in SEQUENCE_MODEL_TYPES else None
        metrics = evaluate_model(model, x_test, y_test, window_size)

        # Profile
        if model_type in SEQUENCE_MODEL_TYPES:
            sample = x_test[:window_size].reshape(1, window_size, x_test.shape[1])
        else:
            sample = x_test[:1]

        profile = profile_model(model, sample, accuracy=metrics["accuracy"])

        # Log final results to W&B
        profiling_dict = {
            "inference_latency_ms": profile.inference_latency_ms,
            "inference_latency_p95_ms": profile.inference_latency_p95_ms,
            "peak_memory_mb": profile.peak_memory_mb,
            "model_size_mb": profile.model_size_mb,
            "throughput_samples_per_sec": profile.throughput_samples_per_sec,
            "efficiency_score": profile.efficiency_score,
            "training_time_seconds": training_time,
        }
        tracker.log_final_results(metrics, profiling_dict)

        # Upload model artifact to S3
        s3_path = upload_model_artifact(model, run_id, model_type)

        # Finish W&B run and capture run ID
        wandb_run_id = tracker.finish()

        # Store results
        run.train_loss = None  # only tracked per-epoch for LSTM
        run.val_loss = None
        run.test_loss = None
        run.accuracy = metrics["accuracy"]
        run.precision_score = metrics["precision"]
        run.recall = metrics["recall"]
        run.f1 = metrics["f1"]
        run.training_time_seconds = training_time
        run.inference_latency_ms = profile.inference_latency_ms
        run.inference_latency_p95_ms = profile.inference_latency_p95_ms
        run.peak_memory_mb = profile.peak_memory_mb
        run.model_size_mb = profile.model_size_mb
        run.throughput_samples_per_sec = profile.throughput_samples_per_sec
        run.efficiency_score = profile.efficiency_score
        run.wandb_run_id = wandb_run_id
        run.s3_artifact_path = s3_path
        run.status = "completed"
        run.completed_at = datetime.now(UTC)

        db.commit()
        db.refresh(run)
        logger.info("Run %s completed — accuracy=%.4f", run_id, metrics["accuracy"])

        # Prometheus metrics
        EXPERIMENTS_TOTAL.labels(model_type=model_type, status="completed").inc()
        TRAINING_DURATION_SECONDS.labels(model_type=model_type).observe(training_time)
        if profile.inference_latency_ms is not None:
            INFERENCE_LATENCY_SECONDS.labels(model_type=model_type).observe(
                profile.inference_latency_ms / 1000.0
            )

        # Generate and store embedding for semantic search (best-effort)
        try:
            embed_run(run_id, db)
        except Exception:
            logger.warning("Failed to generate embedding for run %s", run_id, exc_info=True)

    except Exception:
        tracker.finish()  # clean up W&B run on failure
        EXPERIMENTS_TOTAL.labels(model_type=run.model_type, status="failed").inc()
        run.status = "failed"
        run.completed_at = datetime.now(UTC)
        db.commit()
        logger.exception("Run %s failed", run_id)
        raise

    return run
