"""Data ingestion service — fetches financial data and computes features."""

import logging
import os
import time
from datetime import date
from pathlib import Path
from uuid import UUID

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from forge.api.models.database import Dataset
from forge.api.services.feature_eng import compute_all_features
from forge.api.services.metrics import DATA_INGESTION_ROWS_TOTAL
from forge.api.services.s3_client import (
    download_dataset_artifact,
    upload_dataset_artifact,
)

DATA_DIR = Path(os.getenv("FORGE_DATA_DIR", "data/datasets"))


def fetch_ohlcv(
    tickers: list[str], start_date: date, end_date: date
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data from yfinance for each ticker.

    Returns a dict mapping ticker symbol to its OHLCV DataFrame.
    """
    max_retries = 3
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = pd.DataFrame()
        for attempt in range(max_retries):
            df = yf.download(
                ticker,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                progress=False,
                auto_adjust=True,
            )
            if not df.empty:
                break
            logger.warning(
                "yfinance returned empty for %s (attempt %d/%d), retrying...",
                ticker, attempt + 1, max_retries,
            )
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

        if df.empty:
            continue
        # yfinance may return MultiIndex columns for single ticker — flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        result[ticker] = df
    return result


def ingest_dataset(
    session: Session,
    name: str,
    tickers: list[str],
    start_date: date,
    end_date: date,
    source: str = "yfinance",
) -> Dataset:
    """Orchestrate data fetch, feature computation, and storage.

    Fetches OHLCV data, computes features, saves as parquet, and records
    metadata in the datasets table.
    """
    ticker_data = fetch_ohlcv(tickers, start_date, end_date)

    if not ticker_data:
        raise ValueError(
            f"No data returned from yfinance for tickers: {tickers} "
            f"between {start_date} and {end_date}"
        )

    # Compute features for each ticker and concatenate
    all_frames: list[pd.DataFrame] = []
    for ticker, df in ticker_data.items():
        featured_df = compute_all_features(df)
        featured_df["ticker"] = ticker
        all_frames.append(featured_df)
        DATA_INGESTION_ROWS_TOTAL.labels(ticker=ticker).inc(len(featured_df))

    combined = pd.concat(all_frames, ignore_index=True)

    # Determine feature columns (everything we added beyond OHLCV + Date + ticker)
    ohlcv_cols = {"Date", "Open", "High", "Low", "Close", "Volume", "ticker"}
    feature_columns = [col for col in combined.columns if col not in ohlcv_cols]

    # Save to parquet
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(
        name=name,
        source=source,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        num_records=len(combined),
        feature_columns=feature_columns,
    )
    session.add(dataset)
    session.flush()  # Get the generated ID

    parquet_path = DATA_DIR / f"{dataset.id}.parquet"
    combined.to_parquet(parquet_path, index=False)
    dataset.s3_path = str(parquet_path)
    # Local disk is ephemeral on Railway; keep a durable copy in S3 when configured
    upload_dataset_artifact(str(parquet_path), dataset.id)

    session.commit()
    session.refresh(dataset)
    return dataset


def get_dataset_by_id(session: Session, dataset_id: UUID) -> Dataset | None:
    """Fetch a single dataset by ID."""
    return session.query(Dataset).filter(Dataset.id == dataset_id).first()


def list_datasets(session: Session) -> list[Dataset]:
    """Fetch all datasets ordered by creation time descending."""
    return (
        session.query(Dataset).order_by(Dataset.created_at.desc()).all()
    )


def load_dataset_frame(dataset: Dataset) -> pd.DataFrame | None:
    """Load a dataset's parquet, restoring it from S3 if the local copy is gone.

    Returns None when the file exists neither locally nor in S3.
    """
    if not dataset.s3_path:
        return None
    local_path = Path(dataset.s3_path)
    if not local_path.exists() and not download_dataset_artifact(
        dataset.id, local_path.name, local_path
    ):
        return None
    return pd.read_parquet(local_path)


def get_feature_summary(dataset: Dataset) -> dict | None:
    """Load the parquet file and compute summary stats for feature columns."""
    df = load_dataset_frame(dataset)
    if df is None:
        return None
    feature_cols = dataset.feature_columns or []

    # Only summarize columns that exist in the parquet and are numeric
    available = [col for col in feature_cols if col in df.columns]
    if not available:
        return None

    summary = df[available].describe().to_dict()
    return summary
