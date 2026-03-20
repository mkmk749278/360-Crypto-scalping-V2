"""Tests for src.risk – RiskManager and position sizing helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.risk import RiskManager, calculate_position_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    entry: float = 100.0,
    stop_loss: float = 97.0,
    tp1: float = 106.0,
    direction: str = "LONG",
    symbol: str = "BTCUSDT",
    spread_pct: float = 0.0,
    confidence: float = 70.0,
) -> SimpleNamespace:
    dir_ns = SimpleNamespace(value=direction)
    return SimpleNamespace(
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        direction=dir_ns,
        symbol=symbol,
        spread_pct=spread_pct,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# R:R floor
# ---------------------------------------------------------------------------

class TestRRFloor:
    """calculate_risk must hard-reject trades with R:R < 1.2."""

    def setup_method(self):
        self.rm = RiskManager()

    def test_sufficient_rr_allowed(self):
        """Trade with R:R >= 1.2 must be allowed (ignoring concurrent limits)."""
        # entry=100, sl=97 (3 pts), tp1=106 (6 pts) → R:R = 6/3 = 2.0
        sig = _make_signal(entry=100.0, stop_loss=97.0, tp1=106.0)
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        assert result.allowed is True
        assert result.risk_reward == pytest.approx(2.0)

    def test_exact_rr_floor_allowed(self):
        """Trade with R:R clearly above 1.2 must be allowed."""
        # entry=100, sl=97 (3 pts), tp1=104.0 (4 pts) → R:R = 4/3 ≈ 1.333 > 1.2
        sig = _make_signal(entry=100.0, stop_loss=97.0, tp1=104.0)
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        assert result.allowed is True
        assert result.risk_reward >= 1.2

    def test_insufficient_rr_rejected(self):
        """Trade with R:R < 1.2 must be hard-rejected."""
        # entry=100, sl=97 (3 pts), tp1=103.0 (3 pts) → R:R = 1.0 < 1.2
        sig = _make_signal(entry=100.0, stop_loss=97.0, tp1=103.0)
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        assert result.allowed is False
        assert "R:R" in result.reason
        assert "1.2" in result.reason

    def test_very_bad_rr_rejected(self):
        """Trade with inverted R:R (R:R < 1) must be rejected."""
        # entry=100, sl=95 (5 pts), tp1=103 (3 pts) → R:R = 0.6 < 1.2
        sig = _make_signal(entry=100.0, stop_loss=95.0, tp1=103.0)
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        assert result.allowed is False
        assert "Insufficient R:R" in result.reason

    def test_zero_sl_dist_rr_rejected(self):
        """When SL == entry, R:R = 0 and trade is rejected."""
        sig = _make_signal(entry=100.0, stop_loss=100.0, tp1=106.0)
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        assert result.allowed is False

    def test_rr_rejection_overrides_concurrent_pass(self):
        """R:R rejection must apply even when concurrent limits are not exceeded."""
        sig = _make_signal(entry=100.0, stop_loss=97.0, tp1=102.0)  # R:R ≈ 0.67
        result = self.rm.calculate_risk(sig, {}, 100_000_000, active_signals={})
        assert result.allowed is False
        assert "R:R" in result.reason


# ---------------------------------------------------------------------------
# Spread-based position size penalty
# ---------------------------------------------------------------------------

class TestSpreadPenalty:
    """High-spread pairs must receive a smaller position size."""

    def setup_method(self):
        self.rm = RiskManager()

    def _position_size(self, spread_pct: float) -> float:
        # Good R:R signal (2.0), only spread_pct changes
        sig = _make_signal(entry=100.0, stop_loss=97.0, tp1=106.0, spread_pct=spread_pct)
        result = self.rm.calculate_risk(sig, {}, 100_000_000)
        return result.position_size_pct

    def test_no_spread_full_size(self):
        """Zero spread → no penalty applied."""
        size_no_spread = self._position_size(0.0)
        size_low_spread = self._position_size(0.01)  # below 0.02 threshold
        # Both should be the same (below threshold, no penalty)
        assert size_no_spread == size_low_spread

    def test_high_spread_reduces_size(self):
        """Spread above 0.02 threshold must reduce position size."""
        size_base = self._position_size(0.01)
        size_high_spread = self._position_size(0.05)
        assert size_high_spread < size_base

    def test_very_high_spread_floored_at_50_pct(self):
        """Extremely high spread must be floored at 50% of base position size."""
        size_base = self._position_size(0.0)
        size_extreme = self._position_size(0.50)  # massive spread
        # Floor at 50% of base
        assert size_extreme >= size_base * 0.5 - 0.01  # small floating-point tolerance

    def test_spread_penalty_proportional(self):
        """Higher spread → smaller position (monotonically)."""
        size_low = self._position_size(0.03)
        size_mid = self._position_size(0.06)
        size_high = self._position_size(0.12)
        assert size_low >= size_mid >= size_high


# ---------------------------------------------------------------------------
# calculate_position_size module-level helper
# ---------------------------------------------------------------------------

class TestCalculatePositionSize:
    def test_zero_confidence_returns_zero(self):
        assert calculate_position_size(0.0, 1.0) == 0.0

    def test_high_confidence_low_atr_larger(self):
        size_high = calculate_position_size(90.0, 0.1, entry=100.0)
        size_low = calculate_position_size(50.0, 2.0, entry=100.0)
        assert size_high > size_low

    def test_capped_at_100(self):
        size = calculate_position_size(100.0, 0.001, account_risk_pct=100.0, entry=100.0)
        assert size <= 100.0
