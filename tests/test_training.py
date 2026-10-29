"""Tests for the training service — time-series split and data preparation."""

import numpy as np
import pandas as pd
import pytest

from forge.api.services.training import (
    create_target,
    extract_xy,
    time_series_split,
)


@pytest.fixture
def sample_timeseries_df() -> pd.DataFrame:
    """Create a realistic time-ordered DataFrame with 200 rows."""
    n = 200
    rng = np.random.RandomState(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.randn(n) * 0.5)

    return pd.DataFrame({
        "Date": dates,
        "Open": close - rng.rand(n),
        "High": close + rng.rand(n),
        "Low": close - rng.rand(n) - 0.5,
        "Close": close,
        "Volume": rng.randint(1_000_000, 10_000_000, size=n),
        "rsi": rng.rand(n) * 100,
        "macd_line": rng.randn(n),
    })


class TestTimeSeriesSplit:
    """Tests for chronological train/val/test splitting."""

    def test_split_sizes(self, sample_timeseries_df: pd.DataFrame) -> None:
        train, val, test = time_series_split(sample_timeseries_df)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_timeseries_df)

    def test_default_ratios(self, sample_timeseries_df: pd.DataFrame) -> None:
        n = len(sample_timeseries_df)
        train, val, _test = time_series_split(sample_timeseries_df)
        assert len(train) == int(n * 0.70)
        assert len(val) == int(n * 0.85) - int(n * 0.70)

    def test_no_overlap(self, sample_timeseries_df: pd.DataFrame) -> None:
        train, val, test = time_series_split(sample_timeseries_df)
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_chronological_order_no_leakage(self, sample_timeseries_df: pd.DataFrame) -> None:
        """The last training date must be before the first validation date,
        and the last validation date must be before the first test date."""
        train, val, test = time_series_split(sample_timeseries_df)
        assert train["Date"].iloc[-1] < val["Date"].iloc[0]
        assert val["Date"].iloc[-1] < test["Date"].iloc[0]

    def test_custom_ratios(self, sample_timeseries_df: pd.DataFrame) -> None:
        train, val, test = time_series_split(
            sample_timeseries_df, train_ratio=0.6, val_ratio=0.2
        )
        n = len(sample_timeseries_df)
        assert len(train) == int(n * 0.6)
        assert len(val) == int(n * 0.8) - int(n * 0.6)
        assert len(test) == n - int(n * 0.8)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(columns=["Date", "Close"])
        train, val, test = time_series_split(df)
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0


class TestCreateTarget:
    """Tests for target column creation."""

    def test_target_is_binary(self, sample_timeseries_df: pd.DataFrame) -> None:
        df = create_target(sample_timeseries_df)
        assert set(df["target"].unique()).issubset({0, 1})

    def test_drops_last_row(self, sample_timeseries_df: pd.DataFrame) -> None:
        df = create_target(sample_timeseries_df)
        assert len(df) == len(sample_timeseries_df) - 1

    def test_target_logic(self) -> None:
        df = pd.DataFrame({"Close": [10.0, 12.0, 11.0, 15.0]})
        result = create_target(df)
        # 12>10 → 1, 11<12 → 0, 15>11 → 1
        assert list(result["target"]) == [1, 0, 1]


class TestExtractXY:
    """Tests for feature/target extraction."""

    def test_excludes_non_feature_columns(self, sample_timeseries_df: pd.DataFrame) -> None:
        df = create_target(sample_timeseries_df)
        x, _y = extract_xy(df)
        # Only rsi and macd_line should be features
        assert x.shape[1] == 2

    def test_y_shape(self, sample_timeseries_df: pd.DataFrame) -> None:
        df = create_target(sample_timeseries_df)
        _x, y = extract_xy(df)
        assert y.shape == (len(df),)

    def test_no_nans_in_x(self, sample_timeseries_df: pd.DataFrame) -> None:
        df = create_target(sample_timeseries_df)
        # Inject a NaN to verify it gets replaced
        df.iloc[0, df.columns.get_loc("rsi")] = np.nan
        x, _y = extract_xy(df)
        assert not np.any(np.isnan(x))
