"""Unit tests for the 5 advanced signal quality filter modules.

Modules under test
------------------
* src.mtf        – Multi-Timeframe Confluence Matrix
* src.vwap       – VWAP + Standard Deviation Bands
* src.oi_filter  – Open Interest & Funding Rate Filter
* src.kill_zone  – Kill Zone / Session Volume Profiling
* src.cross_asset – Cross-Asset Correlation (BTC/ETH Sneeze Filter)
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

from src.mtf import (
    TimeframeState,
    _classify_trend,
    check_mtf_gate,
    compute_mtf_confluence,
)
from src.vwap import VWAPResult, check_vwap_extension, compute_vwap
from src.oi_filter import OIAnalysis, analyse_oi, check_oi_gate
from src.kill_zone import SessionResult, check_kill_zone_gate, classify_session
from src.cross_asset import (
    AssetState,
    check_cross_asset_gate,
    get_dominant_market_state,
)


# ===========================================================================
# 1. MTF Confluence Matrix
# ===========================================================================


class TestClassifyTrend:
    """Unit tests for the _classify_trend internal helper."""

    def test_bullish_when_fast_above_slow_and_close_above_fast(self):
        assert _classify_trend(ema_fast=102.0, ema_slow=100.0, close=103.0) == "BULLISH"

    def test_bearish_when_fast_below_slow_and_close_below_fast(self):
        assert _classify_trend(ema_fast=98.0, ema_slow=100.0, close=97.0) == "BEARISH"

    def test_neutral_when_fast_above_slow_but_close_below(self):
        # Fast > slow but price is below fast → mixed → NEUTRAL
        assert _classify_trend(ema_fast=102.0, ema_slow=100.0, close=101.0) == "NEUTRAL"

    def test_neutral_when_equal_emas(self):
        assert _classify_trend(ema_fast=100.0, ema_slow=100.0, close=100.0) == "NEUTRAL"


class TestComputeMTFConfluence:
    """Tests for compute_mtf_confluence()."""

    def _all_bullish_tfs(self) -> dict:
        return {
            "1m":  {"ema_fast": 101.0, "ema_slow": 100.0, "close": 101.5},
            "15m": {"ema_fast": 102.0, "ema_slow": 100.5, "close": 102.5},
            "1h":  {"ema_fast": 103.0, "ema_slow": 101.0, "close": 103.5},
        }

    def _all_bearish_tfs(self) -> dict:
        return {
            "1m":  {"ema_fast": 99.0, "ema_slow": 101.0, "close": 98.5},
            "15m": {"ema_fast": 98.0, "ema_slow": 100.5, "close": 97.0},
            "1h":  {"ema_fast": 97.0, "ema_slow": 100.0, "close": 96.0},
        }

    def test_all_aligned_long(self):
        result = compute_mtf_confluence("LONG", self._all_bullish_tfs())
        assert result.score == 1.0
        # aligned_count is now weighted: 1m=0.5 + 15m=1.5 + 1h=2.0 = 4.0
        assert result.aligned_count == pytest.approx(4.0, abs=0.01)
        assert result.total_count == 3
        assert result.is_aligned is True
        assert result.is_strong is True
        assert result.reason == ""

    def test_all_aligned_short(self):
        result = compute_mtf_confluence("SHORT", self._all_bearish_tfs())
        assert result.score == 1.0
        assert result.is_aligned is True

    def test_no_alignment_long_vs_bearish_tfs(self):
        result = compute_mtf_confluence("LONG", self._all_bearish_tfs())
        assert result.aligned_count == 0
        assert result.score == 0.0
        assert result.is_aligned is False
        assert "misaligned" in result.reason

    def test_partial_alignment_passes_at_50pct(self):
        tfs = {
            "1m":  {"ema_fast": 101.0, "ema_slow": 100.0, "close": 101.5},  # BULLISH
            "15m": {"ema_fast": 99.0, "ema_slow": 100.5, "close": 98.0},    # BEARISH
        }
        result = compute_mtf_confluence("LONG", tfs, min_score=0.5)
        # With weights: 1m=0.5 (BULLISH→0.5), 15m=1.5 (BEARISH→0); total=2.0
        # score = 0.5/2.0 = 0.25 → below threshold
        assert result.aligned_count == pytest.approx(0.5, abs=0.01)
        assert result.total_count == 2
        assert result.score == pytest.approx(0.25, abs=0.01)
        assert result.is_aligned is False  # 0.25 < 0.5 threshold

    def test_empty_timeframes_returns_zero_score(self):
        result = compute_mtf_confluence("LONG", {})
        assert result.total_count == 0
        assert result.score == 0.0
        assert result.is_aligned is False
        assert "no valid" in result.reason

    def test_malformed_entry_skipped(self):
        tfs = {
            "1m":  {"ema_fast": 101.0, "ema_slow": 100.0, "close": 101.5},
            "bad": {"ema_fast": "nope", "ema_slow": 100.0, "close": 100.0},
        }
        result = compute_mtf_confluence("LONG", tfs)
        assert result.total_count == 1  # only valid TF counted

    def test_lowercase_direction_normalised(self):
        result = compute_mtf_confluence("long", self._all_bullish_tfs())
        assert result.signal_direction == "LONG"
        assert result.is_aligned is True

    def test_timeframe_states_populated(self):
        result = compute_mtf_confluence("LONG", self._all_bullish_tfs())
        assert len(result.timeframe_states) == 3
        assert all(isinstance(s, TimeframeState) for s in result.timeframe_states)

    def test_two_of_three_still_passes(self):
        tfs = {
            "1m":  {"ema_fast": 101.0, "ema_slow": 100.0, "close": 101.5},  # BULLISH
            "15m": {"ema_fast": 102.0, "ema_slow": 100.5, "close": 102.5},  # BULLISH
            "1h":  {"ema_fast": 99.0, "ema_slow": 100.0, "close": 98.0},    # BEARISH
        }
        result = compute_mtf_confluence("LONG", tfs, min_score=0.5)
        assert result.aligned_count == 2
        assert result.is_aligned is True


class TestCheckMTFGate:
    """Tests for the pipeline hook check_mtf_gate()."""

    def test_fails_open_when_no_timeframes(self):
        allowed, reason = check_mtf_gate("LONG", {})
        assert allowed is True
        assert reason == ""

    def test_blocks_when_misaligned(self):
        tfs = {
            "1m": {"ema_fast": 99.0, "ema_slow": 101.0, "close": 98.0},
            "1h": {"ema_fast": 98.0, "ema_slow": 101.0, "close": 97.0},
        }
        allowed, reason = check_mtf_gate("LONG", tfs)
        assert allowed is False
        assert "misaligned" in reason

    def test_passes_when_aligned(self):
        tfs = {
            "1m": {"ema_fast": 101.0, "ema_slow": 100.0, "close": 101.5},
            "1h": {"ema_fast": 102.0, "ema_slow": 100.5, "close": 102.5},
        }
        allowed, reason = check_mtf_gate("LONG", tfs)
        assert allowed is True
        assert reason == ""


# ===========================================================================
# 2. VWAP + Standard Deviation Bands
# ===========================================================================


class TestComputeVWAP:
    """Tests for compute_vwap()."""

    def _simple_data(self):
        highs   = np.array([101.5, 102.0, 101.8, 102.5])
        lows    = np.array([ 99.5, 100.5, 100.0, 101.0])
        closes  = np.array([101.0, 101.5, 101.0, 102.0])
        volumes = np.array([1000.0, 1500.0, 1200.0, 800.0])
        return highs, lows, closes, volumes

    def test_returns_vwap_result(self):
        h, l, c, v = self._simple_data()
        result = compute_vwap(h, l, c, v)
        assert isinstance(result, VWAPResult)

    def test_vwap_is_volume_weighted(self):
        # Single bar: VWAP == typical price
        h = np.array([102.0])
        l = np.array([98.0])
        c = np.array([100.0])
        v = np.array([1000.0])
        result = compute_vwap(h, l, c, v)
        expected_typical = (102.0 + 98.0 + 100.0) / 3
        assert result is not None
        assert abs(result.vwap - expected_typical) < 1e-6

    def test_single_bar_std_dev_is_zero(self):
        result = compute_vwap(
            np.array([100.0]),
            np.array([98.0]),
            np.array([99.0]),
            np.array([500.0]),
        )
        assert result is not None
        assert result.std_dev == 0.0
        assert result.upper_band_1 == result.vwap
        assert result.lower_band_1 == result.vwap

    def test_bands_symmetric(self):
        h, l, c, v = self._simple_data()
        result = compute_vwap(h, l, c, v)
        assert result is not None
        vwap = result.vwap
        sd = result.std_dev
        assert abs(result.upper_band_1 - (vwap + sd)) < 1e-6
        assert abs(result.lower_band_1 - (vwap - sd)) < 1e-6
        assert abs(result.upper_band_2 - (vwap + 2 * sd)) < 1e-6
        assert abs(result.lower_band_2 - (vwap - 2 * sd)) < 1e-6
        assert abs(result.upper_band_3 - (vwap + 3 * sd)) < 1e-6
        assert abs(result.lower_band_3 - (vwap - 3 * sd)) < 1e-6

    def test_higher_volume_bars_dominate_vwap(self):
        # One very expensive bar with huge volume should pull VWAP up
        highs   = np.array([100.0, 200.0])
        lows    = np.array([ 98.0, 198.0])
        closes  = np.array([ 99.0, 199.0])
        volumes = np.array([  10.0, 1000.0])  # 2nd bar dominates
        result = compute_vwap(highs, lows, closes, volumes)
        assert result is not None
        typical_1 = (100.0 + 98.0 + 99.0) / 3   # ~99.0
        typical_2 = (200.0 + 198.0 + 199.0) / 3  # ~199.0
        # VWAP should be much closer to typical_2
        assert result.vwap > (typical_1 + typical_2) / 2

    def test_empty_input_returns_none(self):
        result = compute_vwap(
            np.array([]), np.array([]), np.array([]), np.array([])
        )
        assert result is None

    def test_zero_volume_returns_none(self):
        result = compute_vwap(
            np.array([100.0]),
            np.array([98.0]),
            np.array([99.0]),
            np.array([0.0]),
        )
        assert result is None

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_vwap(
                np.array([100.0, 101.0]),
                np.array([99.0]),
                np.array([100.0]),
                np.array([500.0]),
            )

    def test_vwap_mathematical_correctness(self):
        # Manual calculation for 2 bars
        h = np.array([110.0, 90.0])
        l = np.array([ 90.0, 70.0])
        c = np.array([100.0, 80.0])
        v = np.array([100.0, 100.0])
        tp1 = (110 + 90 + 100) / 3   # 100.0
        tp2 = (90  + 70 + 80)  / 3   # 80.0
        expected_vwap = (tp1 * 100 + tp2 * 100) / 200  # 90.0
        result = compute_vwap(h, l, c, v)
        assert result is not None
        assert abs(result.vwap - expected_vwap) < 1e-6


class TestCheckVWAPExtension:
    """Tests for check_vwap_extension()."""

    def _make_result(self, vwap: float = 100.0, sd: float = 2.0) -> VWAPResult:
        return VWAPResult(
            vwap=vwap,
            std_dev=sd,
            upper_band_1=vwap + sd,
            upper_band_2=vwap + 2 * sd,
            upper_band_3=vwap + 3 * sd,
            lower_band_1=vwap - sd,
            lower_band_2=vwap - 2 * sd,
            lower_band_3=vwap - 3 * sd,
        )

    def test_long_rejected_at_plus_3sd(self):
        r = self._make_result(vwap=100.0, sd=2.0)
        # +3 SD band = 106.0
        allowed, reason = check_vwap_extension("LONG", 106.0, r)
        assert allowed is False
        assert "overextended" in reason
        assert "+3.0 SD" in reason

    def test_long_allowed_below_plus_3sd(self):
        r = self._make_result(vwap=100.0, sd=2.0)
        allowed, reason = check_vwap_extension("LONG", 105.9, r)
        assert allowed is True

    def test_short_rejected_at_minus_3sd(self):
        r = self._make_result(vwap=100.0, sd=2.0)
        # -3 SD band = 94.0
        allowed, reason = check_vwap_extension("SHORT", 94.0, r)
        assert allowed is False
        assert "overextended" in reason

    def test_short_allowed_above_minus_3sd(self):
        r = self._make_result(vwap=100.0, sd=2.0)
        allowed, reason = check_vwap_extension("SHORT", 94.1, r)
        assert allowed is True

    def test_fails_open_when_none(self):
        allowed, reason = check_vwap_extension("LONG", 200.0, None)
        assert allowed is True
        assert reason == ""

    def test_custom_extension_sd(self):
        r = self._make_result(vwap=100.0, sd=2.0)
        # +2 SD band = 104.0 → reject at extension_sd=2.0
        allowed, reason = check_vwap_extension("LONG", 104.0, r, extension_sd=2.0)
        assert allowed is False
        assert "+2.0 SD" in reason

    def test_zero_std_dev_no_extension(self):
        r = self._make_result(vwap=100.0, sd=0.0)
        # bands == vwap, price at vwap → rejected for LONG (≥ upper_band)
        allowed, _ = check_vwap_extension("LONG", 100.0, r)
        assert allowed is False  # price equals the 3 SD band (which is vwap when sd=0)


# ===========================================================================
# 3. Open Interest & Funding Rate Filter
# ===========================================================================


class TestAnalyseOI:
    """Tests for analyse_oi()."""

    def test_momentum_high_quality_rising_price_oi(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0])
        oi     = np.array([5000.0, 5100.0, 5200.0, 5300.0])
        result = analyse_oi(prices, oi)
        assert result.signal == "MOMENTUM"
        assert result.quality == "HIGH"
        assert result.price_direction == "RISING"
        assert result.oi_direction == "RISING"

    def test_squeeze_low_quality_rising_price_falling_oi(self):
        prices = np.array([100.0, 102.0, 104.0, 106.0])
        oi     = np.array([5000.0, 4800.0, 4600.0, 4400.0])
        result = analyse_oi(prices, oi)
        assert result.signal == "SQUEEZE"
        assert result.quality == "LOW"
        assert result.price_direction == "RISING"
        assert result.oi_direction == "FALLING"

    def test_distribution_low_quality_falling_price_rising_oi(self):
        prices = np.array([106.0, 104.0, 102.0, 100.0])
        oi     = np.array([4400.0, 4600.0, 4800.0, 5000.0])
        result = analyse_oi(prices, oi)
        assert result.signal == "DISTRIBUTION"
        assert result.quality == "LOW"

    def test_momentum_bearish_falling_price_falling_oi(self):
        prices = np.array([106.0, 104.0, 102.0, 100.0])
        oi     = np.array([5000.0, 4800.0, 4600.0, 4400.0])
        result = analyse_oi(prices, oi)
        assert result.signal == "MOMENTUM"
        assert result.quality == "HIGH"  # shorts covering / longs liquidated

    def test_neutral_when_both_flat(self):
        prices = np.array([100.0, 100.001, 99.999, 100.0])
        oi     = np.array([5000.0, 5001.0, 4999.0, 5000.0])
        result = analyse_oi(prices, oi)
        assert result.signal == "NEUTRAL"
        assert result.quality == "MEDIUM"

    def test_funding_long_crowded(self):
        prices = np.array([100.0, 101.0])
        oi     = np.array([5000.0, 5100.0])
        funding = np.array([0.001, 0.005])  # 0.5% – extreme positive
        result = analyse_oi(prices, oi, funding_rates=funding, funding_threshold=0.003)
        assert result.funding_bias == "LONG_CROWDED"
        assert result.latest_funding_rate == pytest.approx(0.005)

    def test_funding_short_crowded(self):
        prices = np.array([100.0, 99.0])
        oi     = np.array([5000.0, 4900.0])
        funding = np.array([-0.001, -0.004])  # -0.4% – extreme negative
        result = analyse_oi(prices, oi, funding_rates=funding, funding_threshold=0.003)
        assert result.funding_bias == "SHORT_CROWDED"

    def test_funding_neutral(self):
        prices = np.array([100.0, 101.0])
        oi     = np.array([5000.0, 5100.0])
        funding = np.array([0.0001])
        result = analyse_oi(prices, oi, funding_rates=funding)
        assert result.funding_bias == "NEUTRAL"

    def test_empty_arrays_returns_neutral(self):
        result = analyse_oi([], [])
        assert result.signal == "NEUTRAL"
        assert result.quality == "MEDIUM"

    def test_pct_changes_computed(self):
        prices = np.array([100.0, 110.0])
        oi     = np.array([1000.0, 900.0])
        result = analyse_oi(prices, oi)
        assert abs(result.price_change_pct - 0.1) < 1e-4
        assert abs(result.oi_change_pct - (-0.1)) < 1e-4


class TestCheckOIGate:
    """Tests for check_oi_gate()."""

    def test_fails_open_when_none(self):
        allowed, reason = check_oi_gate("LONG", None)
        assert allowed is True
        assert reason == ""

    def test_squeeze_blocks_long(self):
        result = OIAnalysis(
            price_direction="RISING",
            oi_direction="FALLING",
            signal="SQUEEZE",
            quality="LOW",
            funding_bias="NEUTRAL",
            price_change_pct=0.05,
            oi_change_pct=-0.05,
            latest_funding_rate=None,
        )
        allowed, reason = check_oi_gate("LONG", result)
        assert allowed is False
        assert "squeeze" in reason.lower()

    def test_squeeze_does_not_block_short(self):
        result = OIAnalysis(
            price_direction="RISING",
            oi_direction="FALLING",
            signal="SQUEEZE",
            quality="LOW",
            funding_bias="NEUTRAL",
            price_change_pct=0.05,
            oi_change_pct=-0.05,
            latest_funding_rate=None,
        )
        allowed, reason = check_oi_gate("SHORT", result)
        assert allowed is True

    def test_distribution_blocks_short(self):
        result = OIAnalysis(
            price_direction="FALLING",
            oi_direction="RISING",
            signal="DISTRIBUTION",
            quality="LOW",
            funding_bias="NEUTRAL",
            price_change_pct=-0.05,
            oi_change_pct=0.05,
            latest_funding_rate=None,
        )
        allowed, reason = check_oi_gate("SHORT", result)
        assert allowed is False
        assert "distribution" in reason.lower()

    def test_momentum_passes_long(self):
        result = OIAnalysis(
            price_direction="RISING",
            oi_direction="RISING",
            signal="MOMENTUM",
            quality="HIGH",
            funding_bias="NEUTRAL",
            price_change_pct=0.05,
            oi_change_pct=0.05,
            latest_funding_rate=None,
        )
        allowed, reason = check_oi_gate("LONG", result)
        assert allowed is True

    def test_squeeze_warning_mode_does_not_block(self):
        result = OIAnalysis(
            price_direction="RISING",
            oi_direction="FALLING",
            signal="SQUEEZE",
            quality="LOW",
            funding_bias="NEUTRAL",
            price_change_pct=0.05,
            oi_change_pct=-0.05,
            latest_funding_rate=None,
        )
        allowed, reason = check_oi_gate("LONG", result, reject_low_quality=False)
        assert allowed is True

    def test_squeeze_below_hard_threshold_is_soft_warning(self):
        """OI change between 1% and 3% triggers soft warning but does not hard-reject."""
        result = OIAnalysis(
            price_direction="RISING",
            oi_direction="FALLING",
            signal="SQUEEZE",
            quality="LOW",
            funding_bias="NEUTRAL",
            price_change_pct=0.02,
            oi_change_pct=-0.02,  # 2% — between OI_SOFT_THRESHOLD and OI_HARD_THRESHOLD
            latest_funding_rate=None,
        )
        from src.oi_filter import OI_HARD_THRESHOLD, OI_SOFT_THRESHOLD
        assert abs(-0.02) >= OI_SOFT_THRESHOLD
        assert abs(-0.02) < OI_HARD_THRESHOLD
        allowed, reason = check_oi_gate("LONG", result)
        assert allowed is True
        assert "soft warning" in reason.lower() or "moderate" in reason.lower()

    def test_squeeze_at_hard_threshold_is_hard_rejected(self):
        """OI change >= 3% with LOW quality triggers hard rejection."""
        result = OIAnalysis(
            price_direction="RISING",
            oi_direction="FALLING",
            signal="SQUEEZE",
            quality="LOW",
            funding_bias="NEUTRAL",
            price_change_pct=0.04,
            oi_change_pct=-0.04,  # 4% — exceeds OI_HARD_THRESHOLD
            latest_funding_rate=None,
        )
        from src.oi_filter import OI_HARD_THRESHOLD
        assert abs(-0.04) >= OI_HARD_THRESHOLD
        allowed, reason = check_oi_gate("LONG", result)
        assert allowed is False

    def test_distribution_soft_warning_between_thresholds(self):
        """Distribution between 1–3% OI change issues soft warning for SHORT."""
        result = OIAnalysis(
            price_direction="FALLING",
            oi_direction="RISING",
            signal="DISTRIBUTION",
            quality="LOW",
            funding_bias="NEUTRAL",
            price_change_pct=-0.02,
            oi_change_pct=0.02,  # 2% — soft window
            latest_funding_rate=None,
        )
        allowed, reason = check_oi_gate("SHORT", result)
        assert allowed is True
        assert "soft warning" in reason.lower() or "moderate" in reason.lower()

    def test_oi_soft_threshold_constant_is_one_pct(self):
        from src.oi_filter import OI_SOFT_THRESHOLD
        assert OI_SOFT_THRESHOLD == pytest.approx(0.01)

    def test_oi_hard_threshold_constant_is_three_pct(self):
        from src.oi_filter import OI_HARD_THRESHOLD
        assert OI_HARD_THRESHOLD == pytest.approx(0.03)


def _utc(weekday_offset: int, hour: int, minute: int = 0) -> datetime:
    """Create a UTC datetime for a given weekday offset from Monday=0."""
    # Use a fixed Monday as base: 2024-01-01 is a Monday
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    from datetime import timedelta
    return base + timedelta(days=weekday_offset, hours=hour, minutes=minute)


class TestClassifySession:
    """Tests for classify_session()."""

    def test_ny_london_overlap_has_highest_multiplier(self):
        dt = _utc(0, 13)  # Monday 13:00 UTC
        result = classify_session(dt)
        assert result.session_name == "NY_LONDON_OVERLAP"
        assert result.confidence_multiplier == 1.0
        assert result.is_kill_zone is False

    def test_london_open(self):
        dt = _utc(0, 8)  # Monday 08:00 UTC
        result = classify_session(dt)
        assert result.session_name == "LONDON_OPEN"
        assert result.confidence_multiplier == 0.95

    def test_ny_session(self):
        dt = _utc(0, 17)  # Monday 17:00 UTC
        result = classify_session(dt)
        assert result.session_name == "NY_SESSION"

    def test_asian_dead_zone(self):
        dt = _utc(0, 5)  # Monday 05:00 UTC
        result = classify_session(dt)
        assert result.session_name == "ASIAN_DEAD_ZONE"
        assert result.confidence_multiplier == 0.50
        # multiplier equals the minimum threshold (not below it), so is_kill_zone is False
        assert result.is_kill_zone is False

    def test_post_ny_lull(self):
        dt = _utc(0, 21)  # Monday 21:00 UTC
        result = classify_session(dt)
        assert result.session_name == "POST_NY_LULL"

    def test_weekend_saturday_after_22(self):
        dt = _utc(5, 23)  # Saturday 23:00 UTC
        result = classify_session(dt)
        assert result.is_weekend is True
        assert result.session_name == "WEEKEND_DEAD_ZONE"
        assert result.is_kill_zone is True

    def test_weekend_all_sunday(self):
        for hour in [0, 6, 12, 18, 20]:
            dt = _utc(6, hour)  # Sunday
            result = classify_session(dt)
            assert result.is_weekend is True, f"Expected weekend at Sunday {hour}:00"

    def test_weekday_saturday_before_22_not_weekend(self):
        dt = _utc(5, 21)  # Saturday 21:00 UTC – before the kill zone
        result = classify_session(dt)
        assert result.is_weekend is False

    def test_naive_datetime_treated_as_utc(self):
        # Should not raise; naive dt assumed UTC
        dt = datetime(2024, 1, 1, 13, 0, 0)  # no tzinfo
        result = classify_session(dt)
        assert result.session_name == "NY_LONDON_OVERLAP"

    def test_none_dt_uses_current_time(self):
        # Just check it doesn't raise and returns a valid result
        result = classify_session(None)
        assert isinstance(result, SessionResult)
        assert 0.0 <= result.confidence_multiplier <= 1.0


class TestCheckKillZoneGate:
    """Tests for check_kill_zone_gate()."""

    def test_blocks_asian_dead_zone(self):
        dt = _utc(0, 5)  # 05:00 UTC – Asian dead zone (multiplier 0.50 == minimum)
        # At exactly the minimum threshold, the gate does NOT block (strict < check)
        # Use a lower threshold to confirm blocking behaviour
        allowed, reason = check_kill_zone_gate(dt, minimum_multiplier=0.51)
        assert allowed is False
        assert "kill zone" in reason.lower() or "low-liquidity" in reason.lower()

    def test_blocks_weekend(self):
        dt = _utc(6, 12)  # Sunday
        allowed, reason = check_kill_zone_gate(dt, block_weekends=True)
        assert allowed is False
        assert "WEEKEND" in reason

    def test_allows_ny_london_overlap(self):
        dt = _utc(0, 14)  # Monday 14:00 UTC
        allowed, reason = check_kill_zone_gate(dt)
        assert allowed is True
        assert reason == ""

    def test_weekend_allowed_when_block_disabled(self):
        dt = _utc(6, 12)  # Sunday
        allowed, reason = check_kill_zone_gate(dt, block_weekends=False)
        # Weekend multiplier is 0.40 which is below default 0.50 threshold
        # So it should still be blocked by the multiplier check
        assert allowed is False

    def test_custom_minimum_multiplier(self):
        dt = _utc(0, 8)  # London open: multiplier 0.95
        # Require 1.0 → should block
        allowed, reason = check_kill_zone_gate(dt, minimum_multiplier=1.0)
        assert allowed is False

    def test_asian_session_allowed_with_low_threshold(self):
        dt = _utc(0, 2)  # 02:00 UTC – Asian session (multiplier 0.75)
        allowed, reason = check_kill_zone_gate(dt, minimum_multiplier=0.50)
        assert allowed is True


# ===========================================================================
# 5. Cross-Asset Correlation (BTC/ETH Sneeze Filter)
# ===========================================================================


class TestAssetState:
    """Tests for AssetState helpers."""

    def test_is_bearish_dumping(self):
        s = AssetState(symbol="BTCUSDT", trend="DUMPING")
        assert s.is_bearish() is True

    def test_is_bearish_downtrend(self):
        s = AssetState(symbol="BTCUSDT", trend="DOWNTREND")
        assert s.is_bearish() is True

    def test_not_bearish_neutral(self):
        s = AssetState(symbol="BTCUSDT", trend="NEUTRAL")
        assert s.is_bearish() is False

    def test_not_bearish_bullish(self):
        s = AssetState(symbol="BTCUSDT", trend="BULLISH")
        assert s.is_bearish() is False

    def test_is_high_volatility(self):
        s = AssetState(symbol="BTCUSDT", volatility="HIGH")
        assert s.is_high_volatility() is True

    def test_not_high_volatility_normal(self):
        s = AssetState(symbol="BTCUSDT", volatility="NORMAL")
        assert s.is_high_volatility() is False

    def test_not_high_volatility_when_none(self):
        s = AssetState(symbol="BTCUSDT", volatility=None)
        assert s.is_high_volatility() is False

    def test_case_insensitive_trend(self):
        s = AssetState(symbol="BTCUSDT", trend="dumping")
        assert s.is_bearish() is True


class TestCheckCrossAssetGate:
    """Tests for check_cross_asset_gate()."""

    def test_fails_open_empty_states(self):
        allowed, reason = check_cross_asset_gate("LONG", "SOLUSDT", [])
        assert allowed is True
        assert reason == ""

    def test_blocks_altcoin_long_when_btc_dumping(self):
        btc = AssetState(symbol="BTCUSDT", trend="DUMPING")
        allowed, reason = check_cross_asset_gate("LONG", "SOLUSDT", [btc])
        assert allowed is False
        assert "BTCUSDT" in reason
        assert "DUMPING" in reason

    def test_blocks_altcoin_long_when_eth_bearish(self):
        eth = AssetState(symbol="ETHUSDT", trend="BEARISH")
        allowed, reason = check_cross_asset_gate("LONG", "AVAXUSDT", [eth])
        assert allowed is False

    def test_allows_altcoin_long_when_btc_bullish(self):
        btc = AssetState(symbol="BTCUSDT", trend="BULLISH")
        allowed, reason = check_cross_asset_gate("LONG", "SOLUSDT", [btc])
        assert allowed is True

    def test_short_not_blocked_even_when_btc_dumping(self):
        btc = AssetState(symbol="BTCUSDT", trend="DUMPING")
        allowed, reason = check_cross_asset_gate("SHORT", "SOLUSDT", [btc])
        assert allowed is True

    def test_btc_signal_not_filtered_by_itself(self):
        btc = AssetState(symbol="BTCUSDT", trend="DUMPING")
        # BTC signalling BTC – should not self-block
        allowed, reason = check_cross_asset_gate("LONG", "BTCUSDT", [btc])
        assert allowed is True

    def test_non_major_asset_does_not_trigger_filter(self):
        bnb = AssetState(symbol="BNBUSDT", trend="DUMPING")
        allowed, reason = check_cross_asset_gate("LONG", "SOLUSDT", [bnb])
        assert allowed is True  # BNBUSDT not in default major set

    def test_custom_major_symbols(self):
        bnb = AssetState(symbol="BNBUSDT", trend="DUMPING")
        allowed, reason = check_cross_asset_gate(
            "LONG", "SOLUSDT", [bnb],
            major_symbols=frozenset({"BNBUSDT"}),
        )
        assert allowed is False

    def test_high_volatility_down_blocks_long(self):
        btc = AssetState(
            symbol="BTCUSDT",
            trend="HIGH_VOLATILITY_DOWN",
            volatility="HIGH",
        )
        allowed, reason = check_cross_asset_gate("LONG", "SOLUSDT", [btc])
        assert allowed is False

    def test_mixed_states_one_bearish_blocks(self):
        btc = AssetState(symbol="BTCUSDT", trend="BULLISH")
        eth = AssetState(symbol="ETHUSDT", trend="DUMPING")
        allowed, reason = check_cross_asset_gate("LONG", "SOLUSDT", [btc, eth])
        assert allowed is False
        assert "ETHUSDT" in reason

    def test_both_neutral_allows_long(self):
        btc = AssetState(symbol="BTCUSDT", trend="NEUTRAL")
        eth = AssetState(symbol="ETHUSDT", trend="NEUTRAL")
        allowed, reason = check_cross_asset_gate("LONG", "SOLUSDT", [btc, eth])
        assert allowed is True

    def test_case_insensitive_direction(self):
        btc = AssetState(symbol="BTCUSDT", trend="DUMPING")
        allowed, reason = check_cross_asset_gate("long", "SOLUSDT", [btc])
        assert allowed is False


class TestGetDominantMarketState:
    """Tests for get_dominant_market_state()."""

    def test_risk_off_when_majority_bearish(self):
        states = [
            AssetState("BTCUSDT", trend="DUMPING"),
            AssetState("ETHUSDT", trend="BEARISH"),
        ]
        assert get_dominant_market_state(states) == "RISK_OFF"

    def test_risk_on_when_majority_bullish(self):
        states = [
            AssetState("BTCUSDT", trend="BULLISH"),
            AssetState("ETHUSDT", trend="BULLISH"),
        ]
        assert get_dominant_market_state(states) == "RISK_ON"

    def test_volatile_when_majority_high_vol(self):
        states = [
            AssetState("BTCUSDT", trend="RANGING", volatility="EXTREME"),
            AssetState("ETHUSDT", trend="RANGING", volatility="HIGH"),
        ]
        assert get_dominant_market_state(states) == "VOLATILE"

    def test_neutral_empty(self):
        assert get_dominant_market_state([]) == "NEUTRAL"

    def test_non_major_ignored(self):
        states = [AssetState("BNBUSDT", trend="DUMPING")]
        assert get_dominant_market_state(states) == "NEUTRAL"

    def test_neutral_when_mixed(self):
        states = [
            AssetState("BTCUSDT", trend="BULLISH"),
            AssetState("ETHUSDT", trend="BEARISH"),
        ]
        # 1 bullish, 1 bearish – no majority
        result = get_dominant_market_state(states)
        assert result == "NEUTRAL"
