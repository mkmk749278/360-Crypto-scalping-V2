"""Tests for check_spread_adaptive and regime-aware filter propagation."""

from src.filters import check_spread_adaptive


class TestCheckSpreadAdaptive:
    def test_no_regime_uses_base(self):
        assert check_spread_adaptive(0.01, 0.02) is True
        assert check_spread_adaptive(0.03, 0.02) is False

    def test_volatile_relaxes_threshold(self):
        # 0.025 > 0.02 base, but 0.025 < 0.03 (0.02 * 1.5)
        assert check_spread_adaptive(0.025, 0.02, regime="VOLATILE") is True

    def test_quiet_tightens_threshold(self):
        # 0.015 < 0.02 base, but 0.015 > 0.014 (0.02 * 0.7)
        assert check_spread_adaptive(0.015, 0.02, regime="QUIET") is False
        assert check_spread_adaptive(0.013, 0.02, regime="QUIET") is True

    def test_trending_slightly_relaxes(self):
        # 0.02 * 1.2 = 0.024
        assert check_spread_adaptive(0.022, 0.02, regime="TRENDING_UP") is True
        assert check_spread_adaptive(0.022, 0.02, regime="TRENDING_DOWN") is True

    def test_ranging_uses_base(self):
        assert check_spread_adaptive(0.02, 0.02, regime="RANGING") is True
        assert check_spread_adaptive(0.021, 0.02, regime="RANGING") is False

    def test_atr_bonus_scales_further(self):
        # Base 0.02, VOLATILE → 0.03, ATR 2% → bonus 0.4 → 0.03 * 1.4 = 0.042
        assert check_spread_adaptive(0.04, 0.02, regime="VOLATILE", atr_pct=2.0) is True

    def test_atr_bonus_capped(self):
        # ATR 10% → capped at 0.5 → bonus max +50%
        result_high = check_spread_adaptive(0.06, 0.02, regime="VOLATILE", atr_pct=10.0)
        result_extreme = check_spread_adaptive(0.06, 0.02, regime="VOLATILE", atr_pct=20.0)
        # Both should be the same because of the cap
        assert result_high == result_extreme

    def test_atr_below_threshold_no_bonus(self):
        # ATR < 1.0 → no bonus applied
        assert check_spread_adaptive(0.025, 0.02, regime="VOLATILE", atr_pct=0.5) is True  # 0.03 still
        assert check_spread_adaptive(0.031, 0.02, regime="VOLATILE", atr_pct=0.5) is False  # > 0.03
