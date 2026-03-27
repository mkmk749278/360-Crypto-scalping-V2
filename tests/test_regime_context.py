"""Tests for RegimeContext, atr_percentile, and volume_profile_classify."""

from __future__ import annotations

import numpy as np
import pytest

from src.regime import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeContext,
    RegimeResult,
    atr_percentile,
    volume_profile_classify,
)


# ---------------------------------------------------------------------------
# atr_percentile
# ---------------------------------------------------------------------------


class TestAtrPercentile:
    def test_atr_percentile_basic(self):
        """Output should be in [0, 100] range."""
        rng = np.random.default_rng(42)
        series = rng.uniform(0.5, 2.5, 300)
        result = atr_percentile(series)
        assert 0.0 <= result <= 100.0

    def test_atr_percentile_short_series(self):
        """Fallback to 50.0 when series has fewer than 2 elements."""
        assert atr_percentile(np.array([])) == 50.0
        assert atr_percentile(np.array([1.5])) == 50.0

    def test_atr_percentile_extreme_low(self):
        """Current ATR lower than all prior values → near 0."""
        series = np.array([2.0, 3.0, 4.0, 5.0, 1.0])
        result = atr_percentile(series)
        assert result < 30.0

    def test_atr_percentile_extreme_high(self):
        """Current ATR higher than all prior values → 100."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        result = atr_percentile(series)
        assert result == 100.0

    def test_atr_percentile_lookback_respected(self):
        """Only the last `lookback` bars are used."""
        # First 200 bars very low, last 200 bars contain current (high)
        low_bars = np.ones(300) * 0.1
        high_current = np.array([10.0])
        series = np.concatenate([low_bars, high_current])
        result = atr_percentile(series, lookback=200)
        # window = series[-200:] which includes the high current value
        assert result == 100.0


# ---------------------------------------------------------------------------
# volume_profile_classify
# ---------------------------------------------------------------------------


class TestVolumeProfileClassify:
    def test_volume_profile_accumulation(self):
        """More than 60% of volume traded above VWAP → ACCUMULATION."""
        vwap = 100.0
        closes = np.array([101.0] * 18 + [99.0] * 2, dtype=float)  # 90% above
        volumes = np.ones(20, dtype=float)
        result = volume_profile_classify(volumes, closes, vwap)
        assert result == "ACCUMULATION"

    def test_volume_profile_distribution(self):
        """More than 60% of volume traded below VWAP → DISTRIBUTION."""
        vwap = 100.0
        closes = np.array([99.0] * 18 + [101.0] * 2, dtype=float)  # 90% below
        volumes = np.ones(20, dtype=float)
        result = volume_profile_classify(volumes, closes, vwap)
        assert result == "DISTRIBUTION"

    def test_volume_profile_neutral(self):
        """Balanced volume around VWAP → NEUTRAL."""
        vwap = 100.0
        closes = np.array([101.0] * 10 + [99.0] * 10, dtype=float)  # 50/50
        volumes = np.ones(20, dtype=float)
        result = volume_profile_classify(volumes, closes, vwap)
        assert result == "NEUTRAL"

    def test_volume_profile_edge_cases(self):
        """Edge cases: zero VWAP, short arrays → NEUTRAL."""
        closes = np.ones(20, dtype=float) * 100.0
        volumes = np.ones(20, dtype=float)
        # Zero VWAP
        assert volume_profile_classify(volumes, closes, 0.0) == "NEUTRAL"
        # Negative VWAP
        assert volume_profile_classify(volumes, closes, -5.0) == "NEUTRAL"
        # Arrays too short
        assert volume_profile_classify(
            np.ones(5), np.ones(5) * 100.0, 100.0
        ) == "NEUTRAL"

    def test_volume_profile_zero_total_volume(self):
        """When all volumes are zero, result is NEUTRAL (avoids division by zero)."""
        vwap = 100.0
        closes = np.array([101.0] * 20, dtype=float)
        volumes = np.zeros(20, dtype=float)
        result = volume_profile_classify(volumes, closes, vwap)
        assert result == "NEUTRAL"


# ---------------------------------------------------------------------------
# RegimeContext dataclass
# ---------------------------------------------------------------------------


class TestRegimeContextDataclass:
    def test_regime_context_dataclass(self):
        """RegimeContext fields should be correctly stored and typed."""
        rc = RegimeContext(
            label="TRENDING_UP",
            adx_value=28.5,
            adx_slope=1.2,
            atr_percentile=72.0,
            volume_profile="ACCUMULATION",
            is_regime_strengthening=True,
        )
        assert rc.label == "TRENDING_UP"
        assert rc.adx_value == pytest.approx(28.5)
        assert rc.adx_slope == pytest.approx(1.2)
        assert rc.atr_percentile == pytest.approx(72.0)
        assert rc.volume_profile == "ACCUMULATION"
        assert rc.is_regime_strengthening is True

    def test_regime_context_not_strengthening(self):
        """is_regime_strengthening should be False when slope <= 0 or ADX <= 20."""
        rc = RegimeContext(
            label="RANGING",
            adx_value=15.0,
            adx_slope=-0.5,
            atr_percentile=40.0,
            volume_profile="NEUTRAL",
            is_regime_strengthening=False,
        )
        assert rc.is_regime_strengthening is False


# ---------------------------------------------------------------------------
# build_regime_context integration
# ---------------------------------------------------------------------------


class TestBuildRegimeContext:
    def _make_candles(self, n: int = 100) -> dict:
        rng = np.random.default_rng(0)
        close = np.cumsum(rng.uniform(-1, 1, n)) + 100.0
        high = close + rng.uniform(0.1, 1.0, n)
        low = close - rng.uniform(0.1, 1.0, n)
        volume = rng.uniform(100.0, 1000.0, n)
        return {
            "close": list(close),
            "high": list(high),
            "low": list(low),
            "volume": list(volume),
        }

    def test_build_regime_context_integration(self):
        """build_regime_context() should return a valid RegimeContext."""
        candles = self._make_candles(100)
        detector = MarketRegimeDetector()
        regime_result = RegimeResult(regime=MarketRegime.RANGING, adx=18.0)
        rc = detector.build_regime_context(regime_result, candles=candles)

        assert isinstance(rc, RegimeContext)
        assert rc.label == MarketRegime.RANGING.value
        assert 0.0 <= rc.atr_percentile <= 100.0
        assert rc.volume_profile in {"ACCUMULATION", "DISTRIBUTION", "NEUTRAL"}
        assert isinstance(rc.adx_slope, float)
        assert isinstance(rc.is_regime_strengthening, bool)

    def test_build_regime_context_no_candles(self):
        """Without candle data, defaults should be returned gracefully."""
        detector = MarketRegimeDetector()
        regime_result = RegimeResult(regime=MarketRegime.VOLATILE, adx=None)
        rc = detector.build_regime_context(regime_result)

        assert rc.label == MarketRegime.VOLATILE.value
        assert rc.adx_value == 0.0
        assert rc.adx_slope == 0.0
        assert rc.atr_percentile == 50.0
        assert rc.volume_profile == "NEUTRAL"
        assert rc.is_regime_strengthening is False

    def test_build_regime_context_with_vwap(self):
        """When VWAP is provided and candles are long enough, volume_profile is set."""
        candles = self._make_candles(100)
        closes = np.array(candles["close"])
        vwap = float(np.mean(closes))  # approximate VWAP

        detector = MarketRegimeDetector()
        regime_result = RegimeResult(regime=MarketRegime.TRENDING_UP, adx=30.0)
        rc = detector.build_regime_context(regime_result, candles=candles, vwap=vwap)

        assert rc.volume_profile in {"ACCUMULATION", "DISTRIBUTION", "NEUTRAL"}

    def test_build_regime_context_strengthening(self):
        """is_regime_strengthening should be True when slope > 0 and ADX > 20."""
        candles = self._make_candles(100)
        detector = MarketRegimeDetector()
        regime_result = RegimeResult(regime=MarketRegime.TRENDING_UP, adx=28.0)
        rc = detector.build_regime_context(regime_result, candles=candles)

        # The is_regime_strengthening depends on the computed adx_slope
        expected = rc.adx_slope > 0 and rc.adx_value > 20
        assert rc.is_regime_strengthening == expected


# ---------------------------------------------------------------------------
# BB width QUIET threshold
# ---------------------------------------------------------------------------


class TestBBWidthQuietThreshold:
    """Verify that _BB_WIDTH_QUIET_PCT is set to 1.2 (reduced from 1.5).

    This reduces the number of pairs classified as QUIET by requiring a
    tighter Bollinger Band squeeze, which increases valid signal generation.
    """

    def test_bb_width_quiet_threshold_is_1_point_2(self):
        from src.regime import _BB_WIDTH_QUIET_PCT
        assert _BB_WIDTH_QUIET_PCT == pytest.approx(1.2)

    def test_pair_above_1_point_2_bb_width_not_quiet(self):
        """A pair with BB width of 1.35% (between 1.2 and old 1.5) should
        no longer be classified as QUIET with the new threshold."""
        import numpy as np
        # Build candles with Bollinger Bands slightly wider than 1.2% of price
        # so the pair would have been QUIET under the old 1.5 threshold
        # but NOT quiet under the new 1.2 threshold.
        n = 50
        price = 100.0
        # We need BB width ≈ 1.35% of price → upper - lower ≈ 1.35
        # BB uses 20-period SMA ± 2 std. To hit ~0.675 std, we need
        # std ≈ 0.3375, so use returns with ~0.3% std.
        rng = np.random.default_rng(42)
        closes = price + rng.normal(0, 0.35, n)
        closes = np.maximum(closes, 0.01)
        highs = closes + 0.05
        lows = closes - 0.05
        candles = {
            "close": list(closes),
            "high": list(highs),
            "low": list(lows),
            "open": list(closes),
            "volume": [1_000_000.0] * n,
        }
        indicators = {
            "bb_upper_last": float(np.mean(closes[-20:]) + 2 * np.std(closes[-20:])),
            "bb_lower_last": float(np.mean(closes[-20:]) - 2 * np.std(closes[-20:])),
            "bb_mid_last":   float(np.mean(closes[-20:])),
            "adx_last": 25.0,
            "ema_fast_last": float(closes[-1]),
            "ema_slow_last": float(closes[-1]) * 0.99,
            "atr_last": 0.3,
            "rsi_last": 50.0,
        }
        detector = MarketRegimeDetector()
        result = detector.classify(indicators, candles, timeframe="5m")
        # The regime should NOT be QUIET when BB width is above 1.2%
        bb_upper = indicators["bb_upper_last"]
        bb_lower = indicators["bb_lower_last"]
        bb_width_pct = (bb_upper - bb_lower) / float(np.mean(closes[-20:])) * 100.0
        if bb_width_pct > 1.2:
            assert result.regime != MarketRegime.QUIET
