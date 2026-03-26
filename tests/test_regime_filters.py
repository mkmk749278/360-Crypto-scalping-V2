"""Tests for regime-aware filter thresholds."""

from src.filters import (
    get_rsi_thresholds,
    get_adx_min,
    check_rsi_regime,
    check_adx_regime,
    check_ema_alignment_regime,
)


class TestGetRsiThresholds:
    def test_trending_up_wider_thresholds(self):
        ob, os = get_rsi_thresholds("TRENDING_UP")
        assert ob == 80.0
        assert os == 20.0

    def test_trending_down_wider_thresholds(self):
        ob, os = get_rsi_thresholds("TRENDING_DOWN")
        assert ob == 80.0
        assert os == 20.0

    def test_ranging_tighter_thresholds(self):
        ob, os = get_rsi_thresholds("RANGING")
        assert ob == 70.0
        assert os == 30.0

    def test_volatile_wider_thresholds(self):
        ob, os = get_rsi_thresholds("VOLATILE")
        assert ob == 80.0
        assert os == 20.0

    def test_quiet_tighter_thresholds(self):
        ob, os = get_rsi_thresholds("QUIET")
        assert ob == 70.0
        assert os == 30.0

    def test_empty_regime_defaults(self):
        ob, os = get_rsi_thresholds("")
        assert ob == 75.0
        assert os == 25.0

    def test_unknown_regime_defaults(self):
        ob, os = get_rsi_thresholds("UNKNOWN")
        assert ob == 75.0
        assert os == 25.0


class TestGetAdxMin:
    def test_range_fade_in_ranging(self):
        assert get_adx_min("RANGING", "RANGE_FADE") == 10.0

    def test_range_rejection_in_ranging(self):
        assert get_adx_min("RANGING", "RANGE_REJECTION") == 12.0

    def test_range_fade_in_quiet(self):
        assert get_adx_min("QUIET", "RANGE_FADE") == 8.0

    def test_range_rejection_in_quiet(self):
        assert get_adx_min("QUIET", "RANGE_REJECTION") == 10.0

    def test_trend_continuation_in_trending_up(self):
        assert get_adx_min("TRENDING_UP", "TREND_PULLBACK_CONTINUATION") == 22.0

    def test_breakout_retest_in_trending_up(self):
        assert get_adx_min("TRENDING_UP", "BREAKOUT_RETEST") == 20.0

    def test_momentum_expansion_in_trending_up(self):
        assert get_adx_min("TRENDING_UP", "MOMENTUM_EXPANSION") == 25.0

    def test_trend_continuation_in_trending_down(self):
        assert get_adx_min("TRENDING_DOWN", "TREND_PULLBACK_CONTINUATION") == 22.0

    def test_whale_momentum_in_volatile(self):
        assert get_adx_min("VOLATILE", "WHALE_MOMENTUM") == 15.0

    def test_momentum_expansion_in_volatile(self):
        assert get_adx_min("VOLATILE", "MOMENTUM_EXPANSION") == 20.0

    def test_empty_regime_defaults(self):
        assert get_adx_min("") == 20.0

    def test_empty_regime_with_setup_defaults(self):
        assert get_adx_min("", "RANGE_FADE") == 20.0

    def test_unknown_setup_falls_back_to_regime_default_ranging(self):
        assert get_adx_min("RANGING", "UNKNOWN_SETUP") == 15.0

    def test_unknown_setup_falls_back_to_regime_default_trending(self):
        assert get_adx_min("TRENDING_UP", "UNKNOWN_SETUP") == 20.0

    def test_unknown_setup_falls_back_to_regime_default_volatile(self):
        assert get_adx_min("VOLATILE", "UNKNOWN_SETUP") == 18.0

    def test_unknown_setup_falls_back_to_regime_default_quiet(self):
        assert get_adx_min("QUIET", "UNKNOWN_SETUP") == 12.0

    def test_unknown_regime_and_setup_defaults_to_20(self):
        assert get_adx_min("UNKNOWN", "UNKNOWN") == 20.0


class TestCheckRsiRegime:
    def test_long_passes_in_trending_wider_threshold(self):
        # RSI 76 would FAIL with default 75 threshold, but PASSES in trending (80)
        assert check_rsi_regime(76.0, "LONG", regime="TRENDING_UP") is True

    def test_long_fails_in_ranging_tighter_threshold(self):
        # RSI 72 would PASS with default 75 threshold, but FAILS in ranging (70)
        assert check_rsi_regime(72.0, "LONG", regime="RANGING") is False

    def test_short_passes_in_trending(self):
        # RSI 22 would FAIL with default 25 threshold, but PASSES in trending (20)
        assert check_rsi_regime(22.0, "SHORT", regime="TRENDING_DOWN") is True

    def test_short_fails_in_ranging_tighter_threshold(self):
        # RSI 28 would PASS with default 25 threshold, but FAILS in ranging (30)
        assert check_rsi_regime(28.0, "SHORT", regime="RANGING") is False

    def test_none_rsi_passes(self):
        assert check_rsi_regime(None, "LONG", regime="TRENDING_UP") is True

    def test_none_rsi_passes_any_regime(self):
        assert check_rsi_regime(None, "SHORT", regime="RANGING") is True

    def test_empty_regime_uses_defaults(self):
        # RSI 76 should fail with default 75
        assert check_rsi_regime(76.0, "LONG", regime="") is False

    def test_empty_regime_rsi_below_default_passes(self):
        assert check_rsi_regime(74.0, "LONG", regime="") is True

    def test_volatile_regime_uses_wide_thresholds(self):
        # RSI 78 passes in volatile (threshold 80)
        assert check_rsi_regime(78.0, "LONG", regime="VOLATILE") is True

    def test_quiet_regime_uses_tight_thresholds(self):
        # RSI 71 fails in quiet (threshold 70)
        assert check_rsi_regime(71.0, "LONG", regime="QUIET") is False


class TestCheckAdxRegime:
    def test_low_adx_passes_for_range_fade(self):
        assert check_adx_regime(12.0, regime="RANGING", setup_class="RANGE_FADE") is True

    def test_low_adx_fails_for_trend_continuation(self):
        assert check_adx_regime(12.0, regime="TRENDING_UP", setup_class="TREND_PULLBACK_CONTINUATION") is False

    def test_empty_regime_uses_default_20(self):
        assert check_adx_regime(19.0, regime="") is False
        assert check_adx_regime(21.0, regime="") is True

    def test_none_adx_fails(self):
        assert check_adx_regime(None, regime="RANGING", setup_class="RANGE_FADE") is False

    def test_max_adx_respected(self):
        assert check_adx_regime(50.0, regime="RANGING", setup_class="RANGE_FADE", max_adx=30.0) is False

    def test_whale_momentum_in_volatile_low_threshold(self):
        # ADX 16 passes for WHALE_MOMENTUM in VOLATILE (min=15)
        assert check_adx_regime(16.0, regime="VOLATILE", setup_class="WHALE_MOMENTUM") is True

    def test_whale_momentum_in_volatile_too_low(self):
        # ADX 14 fails for WHALE_MOMENTUM in VOLATILE (min=15)
        assert check_adx_regime(14.0, regime="VOLATILE", setup_class="WHALE_MOMENTUM") is False


class TestCheckEmaAlignmentRegime:
    def test_ranging_regime_always_passes(self):
        # Even misaligned EMAs pass in ranging regime
        assert check_ema_alignment_regime(98.0, 100.0, "LONG", regime="RANGING") is True

    def test_ranging_regime_passes_even_with_none(self):
        assert check_ema_alignment_regime(None, 100.0, "LONG", regime="RANGING") is True

    def test_quiet_regime_always_passes(self):
        assert check_ema_alignment_regime(98.0, 100.0, "LONG", regime="QUIET") is True

    def test_quiet_regime_passes_even_with_none(self):
        assert check_ema_alignment_regime(None, 100.0, "SHORT", regime="QUIET") is True

    def test_trending_up_requires_alignment_long(self):
        assert check_ema_alignment_regime(98.0, 100.0, "LONG", regime="TRENDING_UP") is False
        assert check_ema_alignment_regime(102.0, 100.0, "LONG", regime="TRENDING_UP") is True

    def test_trending_down_requires_alignment_short(self):
        assert check_ema_alignment_regime(102.0, 100.0, "SHORT", regime="TRENDING_DOWN") is False
        assert check_ema_alignment_regime(98.0, 100.0, "SHORT", regime="TRENDING_DOWN") is True

    def test_volatile_moderate_allows_close_emas(self):
        # EMAs very close together (gap < 0.05%) should pass in volatile
        assert check_ema_alignment_regime(100.01, 100.0, "LONG", regime="VOLATILE") is True

    def test_volatile_moderate_rejects_misaligned_when_gap_meaningful(self):
        # Large gap but wrong direction should still fail in MODERATE mode
        assert check_ema_alignment_regime(98.0, 100.0, "LONG", regime="VOLATILE") is False

    def test_volatile_moderate_passes_aligned_when_gap_meaningful(self):
        assert check_ema_alignment_regime(102.0, 100.0, "LONG", regime="VOLATILE") is True

    def test_volatile_moderate_none_ema_fails(self):
        assert check_ema_alignment_regime(None, 100.0, "LONG", regime="VOLATILE") is False
        assert check_ema_alignment_regime(100.0, None, "LONG", regime="VOLATILE") is False

    def test_empty_regime_strict_mode(self):
        assert check_ema_alignment_regime(98.0, 100.0, "LONG", regime="") is False
        assert check_ema_alignment_regime(102.0, 100.0, "LONG", regime="") is True

    def test_none_ema_in_strict_mode(self):
        assert check_ema_alignment_regime(None, 100.0, "LONG", regime="TRENDING_UP") is False

    def test_none_ema_fast_passes_in_relaxed(self):
        assert check_ema_alignment_regime(None, 100.0, "LONG", regime="RANGING") is True

    def test_none_ema_slow_passes_in_relaxed(self):
        assert check_ema_alignment_regime(100.0, None, "SHORT", regime="QUIET") is True
