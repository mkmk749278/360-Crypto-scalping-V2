"""Tests for src.order_flow – Open Interest, Liquidations, and CVD Divergence."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.order_flow import (
    LiquidationEvent,
    OISnapshot,
    OITrend,
    OrderFlowStore,
    classify_oi_trend,
    detect_cvd_divergence,
    is_oi_invalidated,
    is_squeeze,
)


# ---------------------------------------------------------------------------
# classify_oi_trend
# ---------------------------------------------------------------------------


class TestClassifyOITrend:
    def _snaps(self, values):
        """Build OISnapshot list from a sequence of OI values."""
        return [
            OISnapshot(timestamp=float(i), open_interest=v)
            for i, v in enumerate(values)
        ]

    def test_falling_oi(self):
        snaps = self._snaps([1000, 998, 995, 990, 985])
        assert classify_oi_trend(snaps) == OITrend.FALLING

    def test_rising_oi(self):
        snaps = self._snaps([1000, 1005, 1010, 1015, 1020])
        assert classify_oi_trend(snaps) == OITrend.RISING

    def test_neutral_oi_tiny_change(self):
        # Change < 0.5 % threshold → NEUTRAL
        snaps = self._snaps([1000, 1001, 1002, 1003, 1004])
        assert classify_oi_trend(snaps) == OITrend.NEUTRAL

    def test_insufficient_snapshots(self):
        assert classify_oi_trend([]) == OITrend.NEUTRAL
        assert classify_oi_trend(self._snaps([1000])) == OITrend.NEUTRAL

    def test_zero_first_oi(self):
        snaps = self._snaps([0, 1000, 1010])
        assert classify_oi_trend(snaps) == OITrend.NEUTRAL

    def test_lookback_smaller_than_list(self):
        # Only the last 3 entries matter
        snaps = self._snaps([1000, 1050, 1100, 995, 990, 985])
        result = classify_oi_trend(snaps, lookback=3)
        assert result == OITrend.FALLING


# ---------------------------------------------------------------------------
# is_squeeze
# ---------------------------------------------------------------------------


class TestIsSqueeze:
    def test_squeeze_confirmed(self):
        assert is_squeeze(OITrend.FALLING, 100_000) is True

    def test_squeeze_no_liquidations(self):
        # OI falling but no liquidation volume → not a squeeze
        assert is_squeeze(OITrend.FALLING, 0.0) is False

    def test_not_squeeze_rising_oi(self):
        assert is_squeeze(OITrend.RISING, 500_000) is False

    def test_not_squeeze_neutral_oi(self):
        assert is_squeeze(OITrend.NEUTRAL, 500_000) is False

    def test_custom_threshold(self):
        assert is_squeeze(OITrend.FALLING, 50_000, liq_threshold_usd=100_000) is False
        assert is_squeeze(OITrend.FALLING, 200_000, liq_threshold_usd=100_000) is True


# ---------------------------------------------------------------------------
# is_oi_invalidated
# ---------------------------------------------------------------------------


class TestIsOIInvalidated:
    def test_rising_invalidates(self):
        # A significant OI rise (> 1%) invalidates the signal
        assert is_oi_invalidated(OITrend.RISING, "LONG", oi_change_pct=0.02) is True
        assert is_oi_invalidated(OITrend.RISING, "SHORT", oi_change_pct=0.015) is True

    def test_rising_below_noise_threshold_does_not_invalidate(self):
        # An OI rise below 1% is treated as noise and does NOT invalidate
        assert is_oi_invalidated(OITrend.RISING, "LONG", oi_change_pct=0.005) is False
        assert is_oi_invalidated(OITrend.RISING, "SHORT", oi_change_pct=0.009) is False

    def test_rising_exactly_at_threshold_invalidates(self):
        # Exactly 1% is at or above the threshold → invalidates
        assert is_oi_invalidated(OITrend.RISING, "LONG", oi_change_pct=0.01) is True

    def test_falling_does_not_invalidate(self):
        assert is_oi_invalidated(OITrend.FALLING, "LONG") is False
        assert is_oi_invalidated(OITrend.FALLING, "SHORT") is False

    def test_neutral_does_not_invalidate(self):
        assert is_oi_invalidated(OITrend.NEUTRAL, "LONG") is False

    def test_no_oi_change_pct_does_not_invalidate(self):
        # Default oi_change_pct=0.0 — below noise threshold, treated as no change
        assert is_oi_invalidated(OITrend.RISING, "LONG") is False


# ---------------------------------------------------------------------------
# detect_cvd_divergence
# ---------------------------------------------------------------------------


class TestDetectCVDDivergence:
    def test_bullish_divergence(self):
        """Price lower low in second half, CVD higher low → BULLISH."""
        # First half: price 100, CVD 0
        # Second half: price dips to 95 (lower low), CVD only dips to 5 (higher low)
        close = np.array([100, 100, 100, 100, 100, 95, 96, 97, 98, 99], dtype=float)
        cvd = np.array([0, 0, 0, 0, 0, 5, 6, 7, 8, 9], dtype=float)
        result = detect_cvd_divergence(close, cvd, lookback=10)
        assert result == "BULLISH"

    def test_bearish_divergence(self):
        """Price higher high in second half, CVD lower high → BEARISH."""
        close = np.array([100, 100, 100, 100, 100, 105, 104, 103, 102, 101], dtype=float)
        cvd = np.array([10, 10, 10, 10, 10, 8, 7, 6, 5, 4], dtype=float)
        result = detect_cvd_divergence(close, cvd, lookback=10)
        assert result == "BEARISH"

    def test_no_divergence_flat(self):
        """Flat price and CVD → no divergence."""
        close = np.ones(20, dtype=float) * 100
        cvd = np.zeros(20, dtype=float)
        result = detect_cvd_divergence(close, cvd, lookback=20)
        assert result is None

    def test_insufficient_data(self):
        close = np.array([100.0, 101.0, 102.0])
        cvd = np.array([0.0, 1.0, 2.0])
        result = detect_cvd_divergence(close, cvd, lookback=20)
        assert result is None

    def test_aligned_move_no_divergence(self):
        """Price lower low AND CVD lower low → no bullish divergence."""
        close = np.array([100, 100, 100, 100, 100, 95, 94, 93, 92, 91], dtype=float)
        cvd = np.array([10, 10, 10, 10, 10, 5, 4, 3, 2, 1], dtype=float)
        result = detect_cvd_divergence(close, cvd, lookback=10)
        assert result is None


# ---------------------------------------------------------------------------
# OrderFlowStore
# ---------------------------------------------------------------------------


class TestOrderFlowStore:
    def _store(self):
        return OrderFlowStore()

    # ---- OI ----

    def test_add_and_get_oi_trend_falling(self):
        store = self._store()
        for oi in [1000, 998, 995, 990, 985]:
            store.add_oi_snapshot("BTCUSDT", float(oi))
        assert store.get_oi_trend("BTCUSDT") == OITrend.FALLING

    def test_get_oi_trend_no_data(self):
        store = self._store()
        assert store.get_oi_trend("ETHUSDT") == OITrend.NEUTRAL

    # ---- Liquidations ----

    def test_add_and_get_liq_volume(self):
        store = self._store()
        evt = LiquidationEvent(
            timestamp=time.monotonic(),
            symbol="BTCUSDT",
            side="SELL",
            qty=1.0,
            price=50_000.0,
        )
        store.add_liquidation(evt)
        vol = store.get_recent_liq_volume_usd("BTCUSDT", window_seconds=60)
        assert vol == pytest.approx(50_000.0)

    def test_liq_volume_outside_window_excluded(self):
        store = self._store()
        # Add a very old event by back-dating the timestamp
        old_evt = LiquidationEvent(
            timestamp=time.monotonic() - 3600,  # 1 hour ago
            symbol="BTCUSDT",
            side="SELL",
            qty=1.0,
            price=50_000.0,
        )
        store.add_liquidation(old_evt)
        vol = store.get_recent_liq_volume_usd("BTCUSDT", window_seconds=300)
        assert vol == pytest.approx(0.0)

    def test_liq_volume_side_filter(self):
        store = self._store()
        now = time.monotonic()
        long_liq = LiquidationEvent(timestamp=now, symbol="BTCUSDT", side="SELL", qty=1.0, price=50_000)
        short_liq = LiquidationEvent(timestamp=now, symbol="BTCUSDT", side="BUY", qty=0.5, price=50_000)
        store.add_liquidation(long_liq)
        store.add_liquidation(short_liq)
        # BUY side only (short liquidations)
        vol_short_liq = store.get_recent_liq_volume_usd("BTCUSDT", side="BUY")
        assert vol_short_liq == pytest.approx(25_000.0)
        # SELL side only (long liquidations)
        vol_long_liq = store.get_recent_liq_volume_usd("BTCUSDT", side="SELL")
        assert vol_long_liq == pytest.approx(50_000.0)

    # ---- CVD ----

    def test_cvd_update_and_history(self):
        store = self._store()
        store.update_cvd_from_tick("BTCUSDT", buy_vol_usd=1000.0, sell_vol_usd=600.0)
        store.snapshot_cvd_at_candle_close("BTCUSDT")
        store.update_cvd_from_tick("BTCUSDT", buy_vol_usd=500.0, sell_vol_usd=800.0)
        store.snapshot_cvd_at_candle_close("BTCUSDT")
        hist = store.get_cvd_history("BTCUSDT")
        assert len(hist) == 2
        assert hist[0] == pytest.approx(400.0)   # 1000-600
        assert hist[1] == pytest.approx(100.0)   # 400 + (500-800)

    def test_cvd_divergence_with_store(self):
        store = self._store()
        # Build a bullish divergence scenario: price lower low, CVD higher low
        prices = [100, 100, 100, 100, 100, 95, 96, 97, 98, 99]
        cvd_vals = [0, 0, 0, 0, 0, 5, 6, 7, 8, 9]
        for v in cvd_vals:
            # Manually set running CVD
            store._running_cvd["BTCUSDT"] = float(v)
            store.snapshot_cvd_at_candle_close("BTCUSDT")

        close_arr = np.array(prices, dtype=float)
        result = store.get_cvd_divergence("BTCUSDT", close_arr, lookback=10)
        assert result == "BULLISH"

    def test_cvd_divergence_no_history(self):
        store = self._store()
        close_arr = np.array([100.0, 101.0, 102.0], dtype=float)
        result = store.get_cvd_divergence("ETHUSDT", close_arr, lookback=10)
        assert result is None
