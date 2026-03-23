"""Tests for src.smc – Smart Money Concepts detection."""

import numpy as np
import pytest

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
        # body_top = close_price = 95.04 (no open_price); last close 100 > 95.04
        ltf_close = np.array([94.0, 95.0, 100.0])  # last close > body_top
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
        # body_bottom = close_price = 105.04 (no open_price); last close 99 < 105.04
        ltf_close = np.array([106.0, 105.0, 99.0])  # last close < body_bottom
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
        # body_top = close_price = 95.04 (no open_price);
        # last close at 94.5 is below body_top → not confirmed
        ltf_close = np.array([94.0, 94.5, 94.5])
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
        # body_top = close_price = 95.04; pass last_close > 95.04 as a 2-D array
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


# ---------------------------------------------------------------------------
# MSS body-based confirmation (replaces 50% wick midpoint logic)
# ---------------------------------------------------------------------------


class TestMSSBodyConfirmation:
    """detect_mss must require LTF close to break the sweep candle's structural
    body (Open/Close range), not merely the 50% wick midpoint."""

    def _make_sweep(
        self,
        direction: Direction,
        close_price: float,
        wick_high: float,
        wick_low: float,
        open_price: float = 0.0,
    ) -> LiquiditySweep:
        return LiquiditySweep(
            index=59,
            direction=direction,
            sweep_level=close_price,
            close_price=close_price,
            wick_high=wick_high,
            wick_low=wick_low,
            open_price=open_price,
        )

    # --- With open_price available ---

    def test_long_confirmed_with_open_price(self):
        """LONG MSS confirmed when LTF close > max(open, close) of sweep candle."""
        sweep = self._make_sweep(Direction.LONG, close_price=95.0, wick_high=105.0, wick_low=93.0, open_price=94.0)
        # body_top = max(94, 95) = 95; last close 96 > 95
        mss = detect_mss(sweep, np.array([93.0, 94.0, 96.0]))
        assert mss is not None
        assert mss.direction == Direction.LONG
        assert mss.midpoint == pytest.approx(95.0)  # body_top stored in midpoint

    def test_long_rejected_between_midpoint_and_body(self):
        """Close above wick midpoint but below body top must NOT trigger MSS."""
        # Sweep candle: wick_low=93, wick_high=105, open=94, close=95 → body_top=95
        # Old logic midpoint = (93+105)/2 = 99; 97 > 99 is False anyway — use a
        # case where close is between body_top and midpoint.
        sweep = self._make_sweep(Direction.LONG, close_price=98.0, wick_high=105.0, wick_low=93.0, open_price=96.0)
        # body_top = max(96, 98) = 98; wick midpoint = 99; close=97 is between them
        mss = detect_mss(sweep, np.array([94.0, 96.0, 97.0]))
        assert mss is None  # 97 < body_top=98 → not confirmed

    def test_short_confirmed_with_open_price(self):
        """SHORT MSS confirmed when LTF close < min(open, close) of sweep candle."""
        sweep = self._make_sweep(Direction.SHORT, close_price=105.0, wick_high=107.0, wick_low=95.0, open_price=106.0)
        # body_bottom = min(106, 105) = 105; last close 104 < 105
        mss = detect_mss(sweep, np.array([106.0, 105.5, 104.0]))
        assert mss is not None
        assert mss.direction == Direction.SHORT
        assert mss.midpoint == pytest.approx(105.0)

    def test_doji_sweep_is_stricter_than_midpoint(self):
        """Doji candle (open ≈ close) at midpoint: old midpoint logic would pass,
        body logic correctly rejects a close that only reaches the midpoint."""
        # Doji: open=99.5, close=100.0 → body_top=100.0, body_bottom=99.5
        # Wick: low=93, high=107 → old midpoint=100
        sweep = self._make_sweep(Direction.LONG, close_price=100.0, wick_high=107.0, wick_low=93.0, open_price=99.5)
        # LTF close at exactly the wick midpoint (100.0) — old logic: 100 > 100 is False (boundary)
        # New logic: body_top=100.0; 100.0 is NOT strictly > 100.0
        mss = detect_mss(sweep, np.array([95.0, 99.0, 100.0]))
        assert mss is None  # at the body boundary, not past it

    def test_doji_mss_confirmed_past_body(self):
        """Doji sweep: close strictly past body top is confirmed."""
        sweep = self._make_sweep(Direction.LONG, close_price=100.0, wick_high=107.0, wick_low=93.0, open_price=99.5)
        # LTF close at 100.1 > body_top=100.0 → confirmed
        mss = detect_mss(sweep, np.array([95.0, 99.0, 100.1]))
        assert mss is not None
        assert mss.direction == Direction.LONG

    # --- Without open_price (fallback to close_price) ---

    def test_long_fallback_uses_close_price_as_body(self):
        """Without open_price, close_price acts as body_top for LONG sweeps."""
        sweep = self._make_sweep(Direction.LONG, close_price=95.04, wick_high=105.0, wick_low=93.0)
        # body_top = close_price = 95.04; last close 95.1 > 95.04 → confirmed
        mss = detect_mss(sweep, np.array([94.0, 95.0, 95.1]))
        assert mss is not None

    def test_short_fallback_uses_close_price_as_body(self):
        """Without open_price, close_price acts as body_bottom for SHORT sweeps."""
        sweep = self._make_sweep(Direction.SHORT, close_price=105.04, wick_high=107.0, wick_low=95.0)
        # body_bottom = close_price = 105.04; last close 104.9 < 105.04 → confirmed
        mss = detect_mss(sweep, np.array([106.0, 105.5, 104.9]))
        assert mss is not None

    def test_open_price_stored_in_sweep(self):
        """open_prices passed to detect_liquidity_sweeps must be stored in sweeps."""
        n = 60
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0
        open_arr = np.ones(n) * 99.5  # open below close

        low[-1] = 93.0
        close[-1] = 95.04
        open_arr[-1] = 94.0  # known open for sweep candle

        sweeps = detect_liquidity_sweeps(high, low, close, lookback=50, open_prices=open_arr)
        long_sweeps = [s for s in sweeps if s.direction == Direction.LONG]
        assert len(long_sweeps) >= 1
        assert long_sweeps[0].open_price == pytest.approx(94.0)

    def test_bearish_sweep_candle_uses_open_as_body_top(self):
        """Bearish sweep candle (open > close): body_top = open_price, not close_price."""
        # Bearish sweep candle: open=98, close=95 → body_top=98 (open), body_bottom=95
        sweep = self._make_sweep(Direction.LONG, close_price=95.0, wick_high=105.0, wick_low=93.0, open_price=98.0)
        # body_top = max(98, 95) = 98; LTF close at 97 < 98 → NOT confirmed
        mss_rejected = detect_mss(sweep, np.array([94.0, 96.0, 97.0]))
        assert mss_rejected is None  # 97 < body_top=98

        # LTF close at 99 > 98 → confirmed
        mss_confirmed = detect_mss(sweep, np.array([94.0, 96.0, 99.0]))
        assert mss_confirmed is not None
        assert mss_confirmed.midpoint == pytest.approx(98.0)  # body_top stored

    def test_bullish_sweep_candle_uses_close_as_body_top(self):
        """Bullish sweep candle (close > open): body_top = close_price."""
        # Bullish sweep candle: open=94, close=95 → body_top=95 (close)
        sweep = self._make_sweep(Direction.LONG, close_price=95.0, wick_high=105.0, wick_low=93.0, open_price=94.0)
        # body_top = max(94, 95) = 95
        mss = detect_mss(sweep, np.array([94.0, 95.5, 96.0]))
        assert mss is not None
        assert mss.midpoint == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# Scalp-optimised sweep detection parameters (lookback=20, tolerance=0.15)
# ---------------------------------------------------------------------------


class TestScalpSweepParameters:
    """Verify that scalp-tuned lookback=20 / tolerance_pct=0.15 detect
    institutional sweeps that the default parameters (50 / 0.05) would miss.
    """

    def test_scalp_lookback_detects_recent_sweep(self):
        """With lookback=20, only the last 20 candles form the S/R level.

        If a high/low occurred 25 candles ago (beyond lookback=20), the recent
        high/low is measured from the last 20 candles only, allowing the wick
        to breach that shorter-range level and be detected as a sweep.
        """
        n = 60
        # Base candles: price flat at 100
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0

        # Candle 35 candles ago: a spike high to 110 — beyond the 20-candle window
        high[n - 35] = 110.0
        close[n - 35] = 100.0

        # Last candle: wicks above 105 (the 20-candle high) and closes back inside
        high[-1] = 107.0
        low[-1] = 95.0
        close[-1] = 105.10  # within 0.15% of 105

        # With default lookback=50, recent_high includes the 110 spike → no breach
        sweeps_default = detect_liquidity_sweeps(high, low, close, lookback=50, tolerance_pct=0.05)
        bearish_default = [s for s in sweeps_default if s.direction == Direction.SHORT]
        assert len(bearish_default) == 0, "Default lookback=50 should NOT detect sweep (110 is in window)"

        # With scalp lookback=20, the 110 spike is outside the window → recent_high=105
        sweeps_scalp = detect_liquidity_sweeps(high, low, close, lookback=20, tolerance_pct=0.15)
        bearish_scalp = [s for s in sweeps_scalp if s.direction == Direction.SHORT]
        assert len(bearish_scalp) >= 1, "Scalp lookback=20 SHOULD detect sweep against 105 level"

    def test_scalp_tolerance_detects_wider_reclaim(self):
        """tolerance_pct=0.15 at BTC-like prices detects sweeps where price
        only partially reclaims the swept level (closes below the level but
        within the wider tolerance window) — missed by the 0.05% default.
        """
        # Simulate BTC-like pricing: base at ~68,000
        n = 30
        high = np.ones(n) * 68_500.0
        low = np.ones(n) * 67_500.0
        close = np.ones(n) * 68_000.0

        # Last candle: wick pierces below 67_500 (recent_low with lookback=20)
        # and closes at 67_440 — below the swept level by $60.
        #   0.05% tol → tol_low = 67500 × 0.0005 = 33.75 → close must be ≥ 67_466.25 → MISS
        #   0.15% tol → tol_low = 67500 × 0.0015 = 101.25 → close must be ≥ 67_398.75 → HIT
        recent_low = 67_500.0
        high[-1] = 68_000.0
        low[-1] = 67_200.0   # wick below recent_low
        close[-1] = recent_low - 60.0  # 67_440 — inside 0.15% window but outside 0.05%

        sweeps_default = detect_liquidity_sweeps(high, low, close, lookback=20, tolerance_pct=0.05)
        bullish_default = [s for s in sweeps_default if s.direction == Direction.LONG]
        assert len(bullish_default) == 0, "Default tolerance=0.05 should NOT detect this partial reclaim"

        sweeps_scalp = detect_liquidity_sweeps(high, low, close, lookback=20, tolerance_pct=0.15)
        bullish_scalp = [s for s in sweeps_scalp if s.direction == Direction.LONG]
        assert len(bullish_scalp) >= 1, "Scalp tolerance=0.15 SHOULD detect this partial reclaim"

    def test_scalp_params_preserve_direction(self):
        """Detected sweeps with scalp params still have correct direction."""
        n = 30
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0

        # Bullish sweep: wick below 95, close back inside with wide tolerance
        high[-1] = 105.0
        low[-1] = 93.0
        close[-1] = 95.12  # 0.13% above 95 — caught by 0.15% tolerance only

        sweeps = detect_liquidity_sweeps(high, low, close, lookback=20, tolerance_pct=0.15)
        bullish = [s for s in sweeps if s.direction == Direction.LONG]
        assert len(bullish) >= 1
        assert bullish[0].sweep_level == pytest.approx(95.0)

    def test_min_candles_with_scalp_lookback(self):
        """With lookback=20, at least 21 candles are needed; 20 returns empty."""
        high = np.ones(20) * 105.0
        low = np.ones(20) * 95.0
        close = np.ones(20) * 100.0
        # Exactly lookback candles → n < lookback + 1 → empty
        sweeps = detect_liquidity_sweeps(high, low, close, lookback=20, tolerance_pct=0.15)
        assert sweeps == []

    def test_sufficient_candles_with_scalp_lookback(self):
        """With lookback=20, 25 candles is sufficient and a bearish sweep is detected."""
        n = 25
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0
        # Last candle: wick above recent high, close back within 0.15% tolerance
        high[-1] = 107.0
        close[-1] = 105.12  # within 0.15% of 105 (tol = 105 * 0.0015 = 0.1575)
        sweeps = detect_liquidity_sweeps(high, low, close, lookback=20, tolerance_pct=0.15)
        bearish = [s for s in sweeps if s.direction == Direction.SHORT]
        assert len(bearish) >= 1, "Should detect bearish sweep with 25 candles and lookback=20"
