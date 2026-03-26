"""Tests for signal_params lookup table and build_channel_signal regime-aware behaviour."""

from __future__ import annotations

import numpy as np

from src.channels.signal_params import _DEFAULT, lookup_signal_params
from src.channels.base import build_channel_signal
from src.smc import Direction
from config import CHANNEL_SCALP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(n=60, base=100.0, trend=0.1):
    close = np.cumsum(np.ones(n) * trend) + base
    high = close + 0.5
    low = close - 0.5
    volume = np.ones(n) * 1000
    return {"open": close - 0.1, "high": high, "low": low, "close": close, "volume": volume}


def _simple_signal(
    close: float = 100.0,
    sl_dist: float = 0.5,
    direction: Direction = Direction.LONG,
    setup_class: str = "",
    regime: str = "",
    atr_val: float = 0.5,
):
    """Helper to build a signal via build_channel_signal with given params."""
    if direction == Direction.LONG:
        sl = close - sl_dist
        tp1 = close + sl_dist * CHANNEL_SCALP.tp_ratios[0]
        tp2 = close + sl_dist * CHANNEL_SCALP.tp_ratios[1]
        tp3 = close + sl_dist * CHANNEL_SCALP.tp_ratios[2]
    else:
        sl = close + sl_dist
        tp1 = close - sl_dist * CHANNEL_SCALP.tp_ratios[0]
        tp2 = close - sl_dist * CHANNEL_SCALP.tp_ratios[1]
        tp3 = close - sl_dist * CHANNEL_SCALP.tp_ratios[2]

    return build_channel_signal(
        config=CHANNEL_SCALP,
        symbol="BTCUSDT",
        direction=direction,
        close=close,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        sl_dist=sl_dist,
        id_prefix="TEST",
        atr_val=atr_val,
        setup_class=setup_class,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# lookup_signal_params tests
# ---------------------------------------------------------------------------

class TestLookupSignalParams:
    def test_exact_match_returns_correct_params(self):
        """Exact (channel, setup_class, regime) key returns the right params."""
        params = lookup_signal_params("360_SCALP", "RANGE_FADE", "RANGING")
        assert params.entry_zone_bias == 0.5
        assert params.dca_enabled is True
        assert params.tp_ratios == (0.5, 0.8, 1.2)
        assert params.validity_minutes == 12

    def test_exact_match_whale_momentum_volatile(self):
        params = lookup_signal_params("360_SCALP", "WHALE_MOMENTUM", "VOLATILE")
        assert params.dca_enabled is False
        assert params.entry_zone_bias == 0.8
        assert params.validity_minutes == 3

    def test_exact_match_swing(self):
        params = lookup_signal_params("360_SWING", "BREAKOUT_RETEST", "TRENDING_UP")
        assert params.tp_ratios == (1.2, 2.0, 3.0)
        assert params.validity_minutes == 30

    def test_fallback_any_regime(self):
        """When regime has no exact match, returns params for any matching regime."""
        # LIQUIDITY_SWEEP_REVERSAL exists for TRENDING_UP/TRENDING_DOWN/RANGING
        # QUIET is not in the table — should fall back to a LIQUIDITY_SWEEP_REVERSAL entry
        params = lookup_signal_params("360_SCALP", "LIQUIDITY_SWEEP_REVERSAL", "QUIET")
        # Should not be the default (which has no tp_ratios override)
        assert params.tp_ratios is not None

    def test_default_for_unknown_combination(self):
        """Completely unknown (channel, setup_class, regime) returns _DEFAULT."""
        params = lookup_signal_params("360_SCALP", "UNKNOWN_SETUP", "UNKNOWN_REGIME")
        assert params == _DEFAULT
        assert params.tp_ratios is None
        assert params.dca_enabled is True
        assert params.entry_zone_bias == 0.7

    def test_channel_prefix_detection_scalp(self):
        """Various SCALP channel name formats resolve to SCALP prefix."""
        p1 = lookup_signal_params("360_SCALP", "RANGE_FADE", "RANGING")
        p2 = lookup_signal_params("SCALP_CVD", "RANGE_FADE", "RANGING")
        assert p1.entry_zone_bias == p2.entry_zone_bias == 0.5

    def test_channel_prefix_detection_swing(self):
        params = lookup_signal_params("360_SWING", "BREAKOUT_RETEST", "TRENDING_UP")
        assert params.validity_minutes == 30

    def test_channel_prefix_detection_spot(self):
        params = lookup_signal_params("360_SPOT", "BREAKOUT_RETEST", "TRENDING_UP")
        assert params.tp_ratios == (1.5, 2.5, 4.0)


# ---------------------------------------------------------------------------
# build_channel_signal regime-aware behaviour tests
# ---------------------------------------------------------------------------

class TestBuildChannelSignalRegimeAware:
    def test_range_fade_ranging_produces_symmetric_entry_zone(self):
        """RANGE_FADE + RANGING → entry_zone_bias=0.5 → symmetric zone around entry."""
        sig = _simple_signal(
            close=100.0,
            sl_dist=0.5,
            direction=Direction.LONG,
            setup_class="RANGE_FADE",
            regime="RANGING",
        )
        assert sig is not None
        assert sig.entry_zone_low is not None
        assert sig.entry_zone_high is not None
        # With bias=0.5, zone should be roughly symmetric around close
        low_dist = abs(sig.entry_zone_low - sig.entry)
        high_dist = abs(sig.entry_zone_high - sig.entry)
        assert abs(low_dist - high_dist) < 1e-6, (
            f"Zone should be symmetric: low_dist={low_dist}, high_dist={high_dist}"
        )

    def test_whale_momentum_volatile_dca_disabled(self):
        """WHALE_MOMENTUM + VOLATILE → dca_enabled=False → DCA zone fields are zero."""
        sig = _simple_signal(
            close=100.0,
            sl_dist=0.5,
            direction=Direction.LONG,
            setup_class="WHALE_MOMENTUM",
            regime="VOLATILE",
        )
        assert sig is not None
        assert sig.dca_zone_lower == 0.0
        assert sig.dca_zone_upper == 0.0

    def test_whale_momentum_volatile_validity_window(self):
        """WHALE_MOMENTUM + VOLATILE → validity_minutes=3."""
        sig = _simple_signal(
            close=100.0,
            sl_dist=0.5,
            direction=Direction.LONG,
            setup_class="WHALE_MOMENTUM",
            regime="VOLATILE",
        )
        assert sig is not None
        assert sig.valid_for_minutes == 3

    def test_backward_compat_empty_setup_regime(self):
        """Empty setup_class and regime → default params → same behaviour as before."""
        sig_default = _simple_signal(
            close=100.0, sl_dist=0.5, direction=Direction.LONG,
            setup_class="", regime="",
        )
        assert sig_default is not None
        # Default bias is 0.7: LONG zone biased below close
        low_dist = abs(sig_default.entry_zone_low - sig_default.entry)
        high_dist = abs(sig_default.entry_zone_high - sig_default.entry)
        assert low_dist > high_dist, "Default LONG zone should be biased below close"

    def test_backward_compat_dca_populated_by_default(self):
        """Default params have dca_enabled=True → DCA fields are populated."""
        sig = _simple_signal(
            close=100.0, sl_dist=0.5, direction=Direction.LONG,
            setup_class="", regime="",
        )
        assert sig is not None
        # At least one DCA field should be non-zero (DCA zone was computed)
        assert sig.dca_zone_lower != 0.0 or sig.dca_zone_upper != 0.0

    def test_range_fade_ranging_dca_enabled(self):
        """RANGE_FADE + RANGING has dca_enabled=True → DCA fields populated."""
        sig = _simple_signal(
            close=100.0, sl_dist=0.5, direction=Direction.LONG,
            setup_class="RANGE_FADE", regime="RANGING",
        )
        assert sig is not None
        assert sig.dca_zone_lower != 0.0 or sig.dca_zone_upper != 0.0

    def test_sl_multiplier_applied(self):
        """WHALE_MOMENTUM VOLATILE has sl_multiplier=1.5 → wider SL than default."""
        close = 100.0
        sl_dist = 0.5

        sig_default = _simple_signal(close=close, sl_dist=sl_dist, direction=Direction.LONG)
        sig_whale = _simple_signal(
            close=close, sl_dist=sl_dist, direction=Direction.LONG,
            setup_class="WHALE_MOMENTUM", regime="VOLATILE",
        )
        assert sig_default is not None
        assert sig_whale is not None
        # WHALE with sl_multiplier=1.5 should have a SL further from close
        default_sl_dist = abs(sig_default.entry - sig_default.stop_loss)
        whale_sl_dist = abs(sig_whale.entry - sig_whale.stop_loss)
        assert whale_sl_dist > default_sl_dist

    def test_tp_ratio_override(self):
        """RANGE_FADE RANGING has custom tp_ratios → TP1 reflects them (legacy path)."""
        from unittest.mock import patch
        close = 100.0
        sl_dist = 0.5

        with patch("src.channels.base.DYNAMIC_SL_TP_ENABLED", False):
            sig = _simple_signal(
                close=close, sl_dist=sl_dist, direction=Direction.LONG,
                setup_class="RANGE_FADE", regime="RANGING",
            )
        assert sig is not None
        # RANGE_FADE RANGING: tp_ratios=(0.5, 0.8, 1.2), sl_multiplier=0.8
        effective_sl_dist = sl_dist * 0.8
        expected_tp1 = close + effective_sl_dist * 0.5
        assert abs(sig.tp1 - expected_tp1) < 1e-6

    def test_setup_class_field_set(self):
        """setup_class is propagated to the signal's setup_class field."""
        sig = _simple_signal(
            close=100.0, sl_dist=0.5, direction=Direction.LONG,
            setup_class="RANGE_FADE", regime="RANGING",
        )
        assert sig is not None
        assert sig.setup_class == "RANGE_FADE"
