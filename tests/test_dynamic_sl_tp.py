"""Tests for dynamic SL/TP ratio computation."""

import pytest
from src.channels.base import compute_dynamic_sl_tp_ratios


class TestComputeDynamicSlTpRatios:
    """Tests for compute_dynamic_sl_tp_ratios()."""

    def test_high_atr_percentile_widens_sl(self):
        """ATR percentile >= 80 should widen SL."""
        sl_mult, tp = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 90.0, "TRENDING_UP", "MIDCAP"
        )
        assert sl_mult > 1.0  # Widened

    def test_low_atr_percentile_tightens_sl(self):
        """ATR percentile <= 20 should tighten SL."""
        sl_mult, tp = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 10.0, "TRENDING_UP", "MIDCAP"
        )
        assert sl_mult < 0.85

    def test_trending_regime_boosts_tp3(self):
        """TRENDING regime should boost the last (runner) TP target."""
        sl_mult, tp = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 2.0], 1.0, 50.0, "TRENDING_UP", "MIDCAP"
        )
        assert tp[2] > 2.0   # Runner TP boosted
        assert tp[0] == 0.5  # TP1 unchanged

    def test_ranging_regime_compresses_all_tp(self):
        """RANGING regime should compress all TP targets."""
        sl_mult, tp = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "RANGING", "MIDCAP"
        )
        assert tp[0] < 0.5
        assert tp[1] < 1.0
        assert tp[2] < 1.5

    def test_quiet_regime_tightens_sl_and_tp(self):
        """QUIET regime compresses SL and TP."""
        sl_mult, tp = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "QUIET", "MIDCAP"
        )
        assert sl_mult < 1.0
        assert all(t < base for t, base in zip(tp, [0.5, 1.0, 1.5]))

    def test_volatile_regime_widens_sl(self):
        """VOLATILE regime should widen SL significantly."""
        sl_mult, tp = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "VOLATILE", "MIDCAP"
        )
        assert sl_mult > 1.3

    def test_altcoin_tier_widens_sl(self):
        """ALTCOIN tier should widen SL to account for manipulation wicks."""
        sl_mult_alt, _ = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "TRENDING_UP", "ALTCOIN"
        )
        sl_mult_major, _ = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "TRENDING_UP", "MAJOR"
        )
        assert sl_mult_alt > sl_mult_major

    def test_major_tier_tightens_sl(self):
        """MAJOR tier should have tighter SL than MIDCAP."""
        sl_mult_major, _ = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "TRENDING_UP", "MAJOR"
        )
        sl_mult_mid, _ = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "TRENDING_UP", "MIDCAP"
        )
        assert sl_mult_major < sl_mult_mid

    def test_base_sl_mult_propagated(self):
        """base_sl_mult from signal_params should be incorporated."""
        sl_mult_2x, _ = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 2.0, 50.0, "TRENDING_UP", "MIDCAP"
        )
        sl_mult_1x, _ = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "TRENDING_UP", "MIDCAP"
        )
        assert sl_mult_2x == pytest.approx(sl_mult_1x * 2.0)

    def test_neutral_conditions_no_change(self):
        """Mid-percentile, no regime, MIDCAP → base ratios unchanged."""
        sl_mult, tp = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "", "MIDCAP"
        )
        assert sl_mult == pytest.approx(1.0)
        assert tp == [0.5, 1.0, 1.5]

    def test_combined_high_vol_volatile_altcoin(self):
        """Maximum widening: high ATR + VOLATILE + ALTCOIN."""
        sl_mult, tp = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 95.0, "VOLATILE", "ALTCOIN"
        )
        # 1.0 * 1.3 (vol) * 1.4 (regime) * 1.20 (tier) = 2.184
        assert sl_mult == pytest.approx(1.3 * 1.4 * 1.20)

    def test_combined_low_vol_quiet_major(self):
        """Maximum tightening: low ATR + QUIET + MAJOR."""
        sl_mult, tp = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 10.0, "QUIET", "MAJOR"
        )
        # 1.0 * 0.8 (vol) * 0.85 (regime) * 0.95 (tier)
        assert sl_mult == pytest.approx(0.8 * 0.85 * 0.95)

    def test_two_element_tp_ratios(self):
        """Should handle 2-element TP ratio lists without error."""
        sl_mult, tp = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0], 1.0, 50.0, "TRENDING_UP", "MIDCAP"
        )
        assert len(tp) == 2
        # No TP3 to boost, so both should be unchanged (1.0 multiplier)
        assert tp[0] == pytest.approx(0.5)
        assert tp[1] == pytest.approx(1.0)

    def test_trending_down_also_boosts_tp3(self):
        """TRENDING_DOWN should also boost runner TP like TRENDING_UP."""
        _, tp_up = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 2.0], 1.0, 50.0, "TRENDING_UP", "MIDCAP"
        )
        _, tp_down = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 2.0], 1.0, 50.0, "TRENDING_DOWN", "MIDCAP"
        )
        assert tp_up[2] == pytest.approx(tp_down[2])
        assert tp_up[2] > 2.0

    def test_volatile_regime_widens_all_tp(self):
        """VOLATILE regime should widen all TP targets."""
        _, tp = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "VOLATILE", "MIDCAP"
        )
        assert tp[0] > 0.5
        assert tp[1] > 1.0
        assert tp[2] > 1.5

    def test_unknown_pair_tier_defaults_to_midcap(self):
        """Unknown tier string should fall back to MIDCAP multiplier (1.0)."""
        sl_mult_unknown, _ = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "TRENDING_UP", "UNKNOWN_TIER"
        )
        sl_mult_mid, _ = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "TRENDING_UP", "MIDCAP"
        )
        assert sl_mult_unknown == pytest.approx(sl_mult_mid)

    def test_unknown_regime_defaults_to_neutral(self):
        """Unknown regime string should produce sl_mult = base_sl_mult (tier × vol only)."""
        sl_mult, tp = compute_dynamic_sl_tp_ratios(
            [0.5, 1.0, 1.5], 1.0, 50.0, "UNKNOWN_REGIME", "MIDCAP"
        )
        # vol_sl_adj=1.0, regime_sl=1.0 (default), tier_sl=1.0 → 1.0
        assert sl_mult == pytest.approx(1.0)
        # TP not scaled by regime (neutral), all multiplied by 1.0
        assert tp == pytest.approx([0.5, 1.0, 1.5])
