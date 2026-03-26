"""Tests for MACD confirmation layer (PR_04)."""

from src.filters import check_macd_confirmation


class TestCheckMacdConfirmation:
    """Tests for the MACD histogram confirmation filter."""

    # --- LONG confirmation ---
    def test_long_passes_when_rising(self):
        """LONG with rising histogram should pass."""
        ok, adj = check_macd_confirmation(0.5, 0.3, "LONG", regime="RANGING", strict=True)
        assert ok is True
        assert adj == 0.0

    def test_long_passes_when_positive(self):
        """LONG with positive histogram (even if falling) should pass."""
        ok, adj = check_macd_confirmation(0.3, 0.5, "LONG", regime="TRENDING_UP", strict=True)
        assert ok is True
        assert adj == 0.0

    def test_long_fails_strict_when_falling_and_negative(self):
        """LONG with falling AND negative histogram should fail in strict mode."""
        ok, adj = check_macd_confirmation(-0.5, -0.3, "LONG", regime="RANGING", strict=True)
        assert ok is False
        assert adj == 0.0

    def test_long_soft_penalty_when_not_confirmed_non_strict(self):
        """LONG with unconfirmed MACD in non-strict should pass with penalty."""
        ok, adj = check_macd_confirmation(-0.5, -0.3, "LONG", regime="TRENDING_UP", strict=False)
        assert ok is True
        assert adj == -5.0

    # --- SHORT confirmation ---
    def test_short_passes_when_falling(self):
        """SHORT with falling histogram should pass."""
        ok, adj = check_macd_confirmation(-0.5, -0.3, "SHORT", regime="TRENDING_DOWN", strict=True)
        assert ok is True
        assert adj == 0.0

    def test_short_passes_when_negative(self):
        """SHORT with negative histogram (even if rising) should pass."""
        ok, adj = check_macd_confirmation(-0.3, -0.5, "SHORT", regime="TRENDING_DOWN", strict=True)
        assert ok is True
        assert adj == 0.0

    def test_short_fails_strict_when_rising_and_positive(self):
        """SHORT with rising AND positive histogram should fail in strict mode."""
        ok, adj = check_macd_confirmation(0.5, 0.3, "SHORT", regime="QUIET", strict=True)
        assert ok is False
        assert adj == 0.0

    def test_short_soft_penalty_when_not_confirmed_non_strict(self):
        """SHORT with unconfirmed MACD in non-strict should pass with penalty."""
        ok, adj = check_macd_confirmation(0.5, 0.3, "SHORT", regime="VOLATILE", strict=False)
        assert ok is True
        assert adj == -5.0

    # --- None handling ---
    def test_none_histogram_last_passes(self):
        """Missing histogram_last should fail open (pass with no penalty)."""
        ok, adj = check_macd_confirmation(None, 0.3, "LONG", regime="TRENDING_UP", strict=True)
        assert ok is True
        assert adj == 0.0

    def test_none_histogram_prev_passes(self):
        """Missing histogram_prev should fail open (pass with no penalty)."""
        ok, adj = check_macd_confirmation(0.5, None, "SHORT", regime="RANGING", strict=True)
        assert ok is True
        assert adj == 0.0

    def test_both_none_passes(self):
        """Both None values should fail open (pass with no penalty)."""
        ok, adj = check_macd_confirmation(None, None, "LONG", regime="VOLATILE", strict=True)
        assert ok is True
        assert adj == 0.0

    # --- Invalid direction ---
    def test_invalid_direction_passes(self):
        """Invalid direction should fail open."""
        ok, adj = check_macd_confirmation(0.5, 0.3, "INVALID", regime="TRENDING_UP", strict=True)
        assert ok is True
        assert adj == 0.0

    # --- Edge cases ---
    def test_zero_histogram_values(self):
        """Zero histogram values should not be considered positive or negative."""
        ok, adj = check_macd_confirmation(0.0, 0.0, "LONG", regime="RANGING", strict=True)
        assert ok is False  # Not rising, not positive

    def test_exactly_equal_values_long(self):
        """Equal values: not rising, so depends on whether positive."""
        ok, adj = check_macd_confirmation(0.5, 0.5, "LONG", regime="RANGING", strict=True)
        assert ok is True  # Not rising but positive (0.5 > 0)

    def test_exactly_equal_negative_values_short(self):
        """Equal negative values: not falling but negative."""
        ok, adj = check_macd_confirmation(-0.5, -0.5, "SHORT", regime="RANGING", strict=True)
        assert ok is True  # Not falling but negative (-0.5 < 0)

    # --- Regime-specific strict behavior ---
    def test_ranging_regime_should_use_strict(self):
        """Validate the expected strict=True behavior for RANGING."""
        # Falling negative histogram for LONG — should reject in strict
        ok, adj = check_macd_confirmation(-0.5, -0.3, "LONG", regime="RANGING", strict=True)
        assert ok is False

    def test_trending_regime_should_use_soft(self):
        """Validate the expected strict=False behavior for TRENDING."""
        ok, adj = check_macd_confirmation(-0.5, -0.3, "LONG", regime="TRENDING_UP", strict=False)
        assert ok is True
        assert adj == -5.0
