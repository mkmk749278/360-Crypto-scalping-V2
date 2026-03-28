"""Tests for PR05 – Common Channel Gates."""

from __future__ import annotations

import pytest

from src.scanner.common_gates import (
    check_basic_filters,
    check_adx_gate,
    check_range_fade_adx,
    check_rsi_extreme_gate,
    check_regime_compatibility,
    check_probability_gate,
    regime_confidence_adjustment,
    ADX_MIN_SCALP,
    ADX_MAX_RANGE_FADE,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
)


class TestCheckBasicFilters:
    def test_pass_within_limits(self):
        assert check_basic_filters(0.01, 5_000_000, max_spread=0.05, min_volume=1_000_000) is True

    def test_fail_spread_too_high(self):
        assert check_basic_filters(0.10, 5_000_000, max_spread=0.05, min_volume=1_000_000) is False

    def test_fail_volume_too_low(self):
        assert check_basic_filters(0.01, 500_000, max_spread=0.05, min_volume=1_000_000) is False

    def test_fail_both(self):
        assert check_basic_filters(0.20, 0, max_spread=0.05, min_volume=1_000_000) is False

    def test_exactly_at_limits(self):
        # Exactly at limits → pass (not strictly greater/lower)
        assert check_basic_filters(0.05, 1_000_000, max_spread=0.05, min_volume=1_000_000) is True


class TestCheckAdxGate:
    def test_strong_trend_passes(self):
        assert check_adx_gate(30.0, adx_min=ADX_MIN_SCALP) is True

    def test_weak_trend_fails(self):
        assert check_adx_gate(10.0, adx_min=ADX_MIN_SCALP) is False

    def test_none_adx_passes(self):
        assert check_adx_gate(None) is True

    def test_exactly_at_min(self):
        assert check_adx_gate(ADX_MIN_SCALP, adx_min=ADX_MIN_SCALP) is True


class TestCheckRangeFadeAdx:
    def test_below_ceiling_passes(self):
        assert check_range_fade_adx(15.0, adx_max=ADX_MAX_RANGE_FADE) is True

    def test_above_ceiling_fails(self):
        assert check_range_fade_adx(30.0, adx_max=ADX_MAX_RANGE_FADE) is False

    def test_none_passes(self):
        assert check_range_fade_adx(None) is True


class TestCheckRsiExtremeGate:
    def test_long_not_overbought_passes(self):
        assert check_rsi_extreme_gate(60.0, "LONG") is True

    def test_long_overbought_rejected(self):
        assert check_rsi_extreme_gate(RSI_OVERBOUGHT, "LONG") is False

    def test_short_not_oversold_passes(self):
        assert check_rsi_extreme_gate(40.0, "SHORT") is True

    def test_short_oversold_rejected(self):
        assert check_rsi_extreme_gate(RSI_OVERSOLD, "SHORT") is False

    def test_none_rsi_passes(self):
        assert check_rsi_extreme_gate(None, "LONG") is True
        assert check_rsi_extreme_gate(None, "SHORT") is True

    def test_case_insensitive(self):
        assert check_rsi_extreme_gate(80.0, "long") is False
        assert check_rsi_extreme_gate(20.0, "short") is False


class TestCheckRegimeCompatibility:
    def test_vwap_blocked_in_quiet(self):
        assert check_regime_compatibility("QUIET", "360_SCALP_VWAP") is False

    def test_vwap_allowed_in_ranging(self):
        assert check_regime_compatibility("RANGING", "360_SCALP_VWAP") is True

    def test_swing_blocked_in_volatile(self):
        assert check_regime_compatibility("VOLATILE", "360_SWING") is False

    def test_scalp_allowed_in_quiet(self):
        # SCALP is not in the default blocked list for QUIET
        assert check_regime_compatibility("QUIET", "360_SCALP") is True

    def test_unknown_channel_allowed(self):
        assert check_regime_compatibility("QUIET", "360_UNKNOWN") is True

    def test_custom_map_used(self):
        custom_map = {"360_SCALP": ["QUIET"]}
        assert check_regime_compatibility("QUIET", "360_SCALP", regime_incompatible_map=custom_map) is False
        assert check_regime_compatibility("RANGING", "360_SCALP", regime_incompatible_map=custom_map) is True


class TestCheckProbabilityGate:
    def test_high_quality_allowed(self):
        pair_data = {
            "regime": "TRENDING_UP",
            "spread_pct": 0.01,
            "volume_24h_usd": 50_000_000.0,
            "atr_pct": 0.5,
            "hit_rate": 0.7,
        }
        allowed, score = check_probability_gate(pair_data, threshold=70.0)
        assert allowed is True

    def test_zero_threshold_always_allows(self):
        allowed, _ = check_probability_gate({}, threshold=0.0)
        assert allowed is True

    def test_channel_name_in_log(self):
        # Just verify it does not raise
        check_probability_gate({"regime": "QUIET"}, channel="360_SCALP")


class TestRegimeConfidenceAdjustment:
    def test_scalp_in_quiet_penalised(self):
        adj = regime_confidence_adjustment("QUIET", "360_SCALP")
        assert adj < 0

    def test_volatile_regime_penalised(self):
        adj = regime_confidence_adjustment("VOLATILE", "360_SCALP")
        assert adj < 0

    def test_trending_no_penalty(self):
        adj = regime_confidence_adjustment("TRENDING_UP", "360_SCALP")
        assert adj == 0.0

    def test_swing_no_penalty_trending(self):
        adj = regime_confidence_adjustment("TRENDING_UP", "360_SWING")
        assert adj == 0.0
