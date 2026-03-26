"""Tests for adaptive EMA thresholds (PR_03)."""

from src.filters import check_ema_alignment_adaptive


class TestCheckEmaAlignmentAdaptive:
    """Tests for the ATR-normalised, pair-tier-aware EMA alignment check."""

    # --- Basic direction checks ---
    def test_long_aligned_above_buffer(self):
        """LONG with fast well above slow should pass."""
        assert check_ema_alignment_adaptive(
            101.0, 100.0, "LONG",
            atr_val=0.3, close=100.0, regime="TRENDING_UP", pair_tier="MAJOR"
        )

    def test_short_aligned_below_buffer(self):
        """SHORT with fast well below slow should pass."""
        assert check_ema_alignment_adaptive(
            99.0, 100.0, "SHORT",
            atr_val=0.3, close=100.0, regime="TRENDING_DOWN", pair_tier="MAJOR"
        )

    def test_long_below_buffer_fails(self):
        """LONG with fast barely above slow should fail (within buffer)."""
        # MAJOR tier, TRENDING_UP: min_buffer=0.10, regime_mult=0.8
        # atr_pct = 0.3, buffer_pct = max(0.10, 0.3*0.8*0.5) = max(0.10, 0.12) = 0.12
        # buffer_abs = 100 * 0.12 / 100 = 0.12
        assert not check_ema_alignment_adaptive(
            100.05, 100.0, "LONG",
            atr_val=0.3, close=100.0, regime="TRENDING_UP", pair_tier="MAJOR"
        )

    # --- Regime-aware behavior ---
    def test_ranging_regime_always_passes(self):
        """RANGING regime should always pass (mean-reversion doesn't need alignment)."""
        assert check_ema_alignment_adaptive(
            98.0, 100.0, "LONG",
            atr_val=0.3, close=100.0, regime="RANGING", pair_tier="MIDCAP"
        )

    def test_quiet_regime_always_passes(self):
        assert check_ema_alignment_adaptive(
            98.0, 100.0, "LONG",
            atr_val=0.3, close=100.0, regime="QUIET", pair_tier="MIDCAP"
        )

    def test_volatile_wider_buffer(self):
        """VOLATILE regime should have a wider buffer, filtering marginal alignment."""
        # ALTCOIN tier, VOLATILE: min_buffer=0.30, regime_mult=1.5
        # atr_pct=1.0, buffer_pct = max(0.30, 1.0*1.5*0.5) = max(0.30, 0.75) = 0.75
        # buffer_abs = 100 * 0.75 / 100 = 0.75
        assert not check_ema_alignment_adaptive(
            100.5, 100.0, "LONG",
            atr_val=1.0, close=100.0, regime="VOLATILE", pair_tier="ALTCOIN"
        )

    # --- Pair tier scaling ---
    def test_major_tier_tighter_buffer(self):
        """MAJOR tier should have a tighter minimum buffer floor."""
        # MAJOR, TRENDING_UP: min_buffer=0.10
        # This should pass with a smaller gap than ALTCOIN would require
        assert check_ema_alignment_adaptive(
            100.2, 100.0, "LONG",
            atr_val=0.15, close=100.0, regime="TRENDING_UP", pair_tier="MAJOR"
        )

    def test_altcoin_tier_wider_buffer(self):
        """ALTCOIN tier should require wider EMA separation."""
        # ALTCOIN, TRENDING_UP: min_buffer=0.30, regime_mult=0.8
        # atr_pct=0.15, buffer_pct = max(0.30, 0.15*0.8*0.5) = 0.30
        # buffer_abs = 100 * 0.30 / 100 = 0.30
        assert not check_ema_alignment_adaptive(
            100.2, 100.0, "LONG",
            atr_val=0.15, close=100.0, regime="TRENDING_UP", pair_tier="ALTCOIN"
        )

    # --- None handling ---
    def test_none_ema_fast_fails_in_trending(self):
        assert not check_ema_alignment_adaptive(
            None, 100.0, "LONG",
            atr_val=0.3, close=100.0, regime="TRENDING_UP", pair_tier="MIDCAP"
        )

    def test_none_ema_slow_fails_in_trending(self):
        assert not check_ema_alignment_adaptive(
            100.0, None, "SHORT",
            atr_val=0.3, close=100.0, regime="TRENDING_DOWN", pair_tier="MIDCAP"
        )

    def test_none_ema_passes_in_ranging(self):
        """Missing EMAs should pass in RANGING/QUIET (relaxed mode)."""
        assert check_ema_alignment_adaptive(
            None, 100.0, "LONG",
            atr_val=0.3, close=100.0, regime="RANGING", pair_tier="MIDCAP"
        )

    # --- Edge cases ---
    def test_zero_close_uses_fallback(self):
        """Zero close should use fallback buffer calculation."""
        result = check_ema_alignment_adaptive(
            101.0, 100.0, "LONG",
            atr_val=0.3, close=0.0, regime="TRENDING_UP", pair_tier="MIDCAP"
        )
        # Should still work with fallback
        assert isinstance(result, bool)

    def test_zero_atr_uses_fallback(self):
        result = check_ema_alignment_adaptive(
            101.0, 100.0, "LONG",
            atr_val=0.0, close=100.0, regime="TRENDING_UP", pair_tier="MIDCAP"
        )
        assert isinstance(result, bool)

    def test_invalid_direction_returns_false(self):
        assert not check_ema_alignment_adaptive(
            101.0, 100.0, "INVALID",
            atr_val=0.3, close=100.0, regime="TRENDING_UP", pair_tier="MAJOR"
        )

    def test_empty_regime_uses_default_multiplier(self):
        """Empty regime should use multiplier of 1.0."""
        result = check_ema_alignment_adaptive(
            101.0, 100.0, "LONG",
            atr_val=0.3, close=100.0, regime="", pair_tier="MIDCAP"
        )
        assert isinstance(result, bool)
