"""Tests for src.smc – Smart Money Concepts detection."""

import numpy as np

from src.smc import (
    Direction,
    LiquiditySweep,
    detect_fvg,
    detect_liquidity_sweeps,
    detect_mss,
)


# ---------------------------------------------------------------------------
# Liquidity Sweep
# ---------------------------------------------------------------------------


class TestLiquiditySweep:
    def _make_candles(self, n=60):
        """Create synthetic candle data with a known high/low range."""
        np.random.seed(123)
        close = np.cumsum(np.random.randn(n) * 0.5) + 100
        high = close + np.random.uniform(0.1, 0.5, n)
        low = close - np.random.uniform(0.1, 0.5, n)
        return high, low, close

    def test_no_sweep_in_normal_data(self):
        high, low, close = self._make_candles()
        sweeps = detect_liquidity_sweeps(high, low, close, lookback=50)
        # May or may not find sweeps depending on random seed – just ensure no crash
        assert isinstance(sweeps, list)

    def test_bullish_sweep_detected(self):
        """Wick below recent low, close back inside."""
        n = 60
        high = np.ones(n) * 105
        low = np.ones(n) * 95
        close = np.ones(n) * 100

        # Last candle wicks below the range
        high[-1] = 105
        low[-1] = 93  # below 95 (recent low)
        close[-1] = 95.04  # within 0.05 % of 95

        sweeps = detect_liquidity_sweeps(high, low, close, lookback=50)
        assert len(sweeps) >= 1
        assert any(s.direction == Direction.LONG for s in sweeps)

    def test_bearish_sweep_detected(self):
        """Wick above recent high, close back inside."""
        n = 60
        high = np.ones(n) * 105
        low = np.ones(n) * 95
        close = np.ones(n) * 100

        high[-1] = 107  # above 105
        low[-1] = 95
        close[-1] = 105.04  # within 0.05 %

        sweeps = detect_liquidity_sweeps(high, low, close, lookback=50)
        assert len(sweeps) >= 1
        assert any(s.direction == Direction.SHORT for s in sweeps)

    def test_insufficient_data(self):
        sweeps = detect_liquidity_sweeps(
            np.array([1.0, 2.0]), np.array([0.5, 1.5]), np.array([0.8, 1.8]),
            lookback=50,
        )
        assert sweeps == []


# ---------------------------------------------------------------------------
# MSS
# ---------------------------------------------------------------------------


class TestMSS:
    def test_long_mss_confirmed(self):
        sweep = LiquiditySweep(
            index=59,
            direction=Direction.LONG,
            sweep_level=95,
            close_price=95.04,
            wick_high=105,
            wick_low=93,
        )
        # Midpoint = (105 + 93) / 2 = 99
        ltf_close = np.array([94.0, 95.0, 100.0])  # last close > 99
        mss = detect_mss(sweep, ltf_close)
        assert mss is not None
        assert mss.direction == Direction.LONG

    def test_short_mss_confirmed(self):
        sweep = LiquiditySweep(
            index=59,
            direction=Direction.SHORT,
            sweep_level=105,
            close_price=105.04,
            wick_high=107,
            wick_low=95,
        )
        # Midpoint = (107 + 95) / 2 = 101
        ltf_close = np.array([106.0, 105.0, 99.0])  # last close < 101
        mss = detect_mss(sweep, ltf_close)
        assert mss is not None
        assert mss.direction == Direction.SHORT

    def test_mss_not_confirmed(self):
        sweep = LiquiditySweep(
            index=59,
            direction=Direction.LONG,
            sweep_level=95,
            close_price=95.04,
            wick_high=105,
            wick_low=93,
        )
        ltf_close = np.array([94.0, 95.0, 96.0])  # 96 < 99 midpoint → not confirmed
        mss = detect_mss(sweep, ltf_close)
        assert mss is None


# ---------------------------------------------------------------------------
# FVG
# ---------------------------------------------------------------------------


class TestFVG:
    def test_bullish_fvg(self):
        # candle[i+2].low > candle[i].high  →  bullish gap
        high = np.array([100, 101, 102, 105, 106])
        low = np.array([98, 99, 100, 103, 104])
        close = np.array([99, 100, 101, 104, 105])
        zones = detect_fvg(high, low, close, lookback=10)
        bullish = [z for z in zones if z.direction == Direction.LONG]
        assert len(bullish) >= 1

    def test_bearish_fvg(self):
        # candle[i+2].high < candle[i].low  →  bearish gap
        high = np.array([106, 105, 104, 100, 99])
        low = np.array([104, 103, 102, 98, 97])
        close = np.array([105, 104, 103, 99, 98])
        zones = detect_fvg(high, low, close, lookback=10)
        bearish = [z for z in zones if z.direction == Direction.SHORT]
        assert len(bearish) >= 1

    def test_no_fvg_in_tight_data(self):
        """Overlapping candles should produce no gaps."""
        n = 20
        high = np.ones(n) * 101
        low = np.ones(n) * 99
        close = np.ones(n) * 100
        zones = detect_fvg(high, low, close, lookback=10)
        assert zones == []


# ---------------------------------------------------------------------------
# 2-D array robustness (Issue 1: ValueError ambiguous truth value)
# ---------------------------------------------------------------------------


class TestNonFlatArrayInputs:
    """All detection functions must handle 2-D (non-flat) input arrays
    without raising ``ValueError: truth value of an array``."""

    def _make_2d(self, n=60):
        """Return synthetic candle data wrapped as 2-D column vectors."""
        np.random.seed(42)
        close = np.cumsum(np.random.randn(n) * 0.5) + 100
        high = close + 0.5
        low = close - 0.5
        # Reshape to (n, 1) – simulates data loaded with an extra dimension
        return high.reshape(-1, 1), low.reshape(-1, 1), close.reshape(-1, 1)

    def test_detect_liquidity_sweeps_2d_input(self):
        high, low, close = self._make_2d()
        sweeps = detect_liquidity_sweeps(high, low, close, lookback=50)
        assert isinstance(sweeps, list)

    def test_detect_fvg_2d_input(self):
        high, low, close = self._make_2d(20)
        zones = detect_fvg(high, low, close, lookback=10)
        assert isinstance(zones, list)

    def test_detect_mss_2d_ltf_close(self):
        sweep = LiquiditySweep(
            index=59,
            direction=Direction.LONG,
            sweep_level=95,
            close_price=95.04,
            wick_high=105,
            wick_low=93,
        )
        # Midpoint = 99; pass last_close > 99 as a 2-D array
        ltf_close = np.array([[94.0], [95.0], [100.0]])
        mss = detect_mss(sweep, ltf_close)
        assert mss is not None
        assert mss.direction == Direction.LONG

    def test_detect_liquidity_sweeps_bearish_2d(self):
        """Bearish sweep still detected when arrays are 2-D."""
        n = 60
        high = np.ones((n, 1)) * 105.0
        low = np.ones((n, 1)) * 95.0
        close = np.ones((n, 1)) * 100.0
        high[-1] = 107.0
        low[-1] = 95.0
        close[-1] = 105.04
        sweeps = detect_liquidity_sweeps(high, low, close, lookback=50)
        assert any(s.direction == Direction.SHORT for s in sweeps)


# ---------------------------------------------------------------------------
# Fix 1: Expanded scan_window (5 candles instead of 1)
# ---------------------------------------------------------------------------


class TestExpandedScanWindow:
    """detect_liquidity_sweeps must scan the last scan_window candles, not just
    the very last candle."""

    def test_sweep_detected_on_candle_minus_2(self):
        """A bullish sweep that occurred 2 candles ago must still be detected."""
        n = 60
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0

        # Sweep candle is at index n-3 (i.e. 2 candles before the last)
        high[n - 3] = 105.0
        low[n - 3] = 93.0      # below 95 (recent low)
        close[n - 3] = 95.04   # closed back inside

        # Last two candles are normal
        high[n - 2] = 105.0
        low[n - 2] = 95.0
        close[n - 2] = 100.0
        high[n - 1] = 105.0
        low[n - 1] = 95.0
        close[n - 1] = 100.0

        sweeps = detect_liquidity_sweeps(high, low, close, lookback=50, scan_window=5)
        assert any(s.direction == Direction.LONG for s in sweeps), (
            "Bullish sweep at n-3 should be detected with scan_window=5"
        )

    def test_sweep_missed_with_scan_window_1(self):
        """With scan_window=1, a sweep 2 candles ago is NOT detected."""
        n = 60
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0

        high[n - 3] = 105.0
        low[n - 3] = 93.0
        close[n - 3] = 95.04

        high[n - 2] = 105.0
        low[n - 2] = 95.0
        close[n - 2] = 100.0
        high[n - 1] = 105.0
        low[n - 1] = 95.0
        close[n - 1] = 100.0

        sweeps = detect_liquidity_sweeps(high, low, close, lookback=50, scan_window=1)
        assert not any(s.direction == Direction.LONG for s in sweeps), (
            "Sweep at n-3 should NOT be detected with scan_window=1"
        )

    def test_no_duplicate_sweeps_in_window(self):
        """A single sweep event should not produce duplicate entries."""
        n = 60
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0

        # Only the last candle sweeps
        low[-1] = 93.0
        close[-1] = 95.04

        sweeps = detect_liquidity_sweeps(high, low, close, lookback=50, scan_window=5)
        long_sweeps = [s for s in sweeps if s.direction == Direction.LONG]
        assert len(long_sweeps) == 1, "Same candle should not appear twice"


# ---------------------------------------------------------------------------
# Fix 10: Volume confirmation for SMC sweeps
# ---------------------------------------------------------------------------


class TestVolumeConfirmation:
    """Volume-confirmed sweeps: only count if sweep candle volume >= 1.2× avg."""

    def test_high_volume_sweep_detected(self):
        """Sweep candle with 2× average volume should be detected."""
        n = 60
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0
        volume = np.ones(n) * 1000.0

        low[-1] = 93.0
        close[-1] = 95.04
        volume[-1] = 2500.0   # 2.5× average → passes 1.2× filter

        sweeps = detect_liquidity_sweeps(
            high, low, close, lookback=50, volume=volume, volume_multiplier=1.2
        )
        assert any(s.direction == Direction.LONG for s in sweeps)

    def test_low_volume_sweep_filtered(self):
        """Sweep candle with only 0.8× average volume should be filtered out."""
        n = 60
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0
        volume = np.ones(n) * 1000.0

        low[-1] = 93.0
        close[-1] = 95.04
        volume[-1] = 800.0    # 0.8× average → fails 1.2× filter

        sweeps = detect_liquidity_sweeps(
            high, low, close, lookback=50, volume=volume, volume_multiplier=1.2
        )
        assert not any(s.direction == Direction.LONG for s in sweeps)

    def test_no_volume_data_unchanged_behavior(self):
        """Without volume data, existing sweep detection is unchanged."""
        n = 60
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0

        low[-1] = 93.0
        close[-1] = 95.04

        sweeps = detect_liquidity_sweeps(high, low, close, lookback=50)
        assert any(s.direction == Direction.LONG for s in sweeps)
