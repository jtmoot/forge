"""Unit tests for anomaly detection service."""


from forge.api.services.anomaly import compute_rolling_zscores, flag_anomalies


class TestComputeRollingZscores:
    """Tests for rolling z-score computation."""

    def test_empty_list_returns_empty(self) -> None:
        """Empty input should return empty output."""
        assert compute_rolling_zscores([]) == []

    def test_single_value_returns_zero(self) -> None:
        """A single value has no history — z-score should be 0."""
        result = compute_rolling_zscores([5.0])
        assert result == [0.0]

    def test_constant_values_all_zero(self) -> None:
        """When all values are identical, std is 0 — all z-scores should be 0."""
        result = compute_rolling_zscores([1.0, 1.0, 1.0, 1.0, 1.0])
        assert all(z == 0.0 for z in result)

    def test_known_spike_produces_high_zscore(self) -> None:
        """A clear outlier after stable values should have a high z-score."""
        # 10 stable values at ~1.0, then a spike at 100.0
        values = [1.0] * 10 + [100.0]
        result = compute_rolling_zscores(values, window=10)
        # The spike (last value) should have a very high z-score
        assert result[-1] > 2.5
        # The stable values should all be 0 (identical history)
        assert all(z == 0.0 for z in result[:10])

    def test_output_length_matches_input(self) -> None:
        """Result list should always be the same length as input."""
        for length in [0, 1, 5, 50]:
            values = [float(i) for i in range(length)]
            result = compute_rolling_zscores(values)
            assert len(result) == length

    def test_window_limits_history(self) -> None:
        """Z-score should only consider the last `window` values."""
        # Spike early, then stable values, then another spike
        # With a small window, the early 100.0 is outside the window for the final value
        values = [100.0] + [1.0] * 10 + [50.0]
        result_small = compute_rolling_zscores(values, window=5)
        result_large = compute_rolling_zscores(values, window=20)
        # With window=5, only the last 5 stable 1.0s are in the window → high z-score
        assert result_small[-1] > 2.0
        # With window=20, the early 100.0 is in the window, inflating std → lower z-score
        assert result_large[-1] > 1.0
        assert result_large[-1] < result_small[-1]

    def test_gradual_increase_not_anomalous(self) -> None:
        """A steady linear increase should not produce extreme z-scores after warmup."""
        values = [float(i) for i in range(30)]
        result = compute_rolling_zscores(values, window=10)
        # Skip early values where limited history causes edge cases (e.g., std=0)
        # After warmup, a linear trend should have bounded z-scores
        assert all(abs(z) < 5.0 for z in result[3:])


class TestFlagAnomalies:
    """Tests for the anomaly flagging wrapper."""

    def test_flags_clear_outlier(self) -> None:
        """A spike in otherwise stable data should be flagged."""
        values = [1.0] * 20 + [100.0]
        flags = flag_anomalies(values, window=20, threshold=2.5)
        assert flags[-1] is True
        # Normal values should not be flagged
        assert not any(flags[:-1])

    def test_no_flags_for_stable_data(self) -> None:
        """Stable data should produce no anomaly flags."""
        values = [5.0] * 30
        flags = flag_anomalies(values)
        assert not any(flags)

    def test_custom_threshold(self) -> None:
        """Lower threshold should flag more values."""
        values = [1.0] * 10 + [3.0]
        flags_strict = flag_anomalies(values, window=10, threshold=5.0)
        flags_loose = flag_anomalies(values, window=10, threshold=0.5)
        # Loose threshold should flag at least as many as strict
        assert sum(flags_loose) >= sum(flags_strict)

    def test_empty_input(self) -> None:
        """Empty input should return empty flags."""
        assert flag_anomalies([]) == []
