"""Tests for src/confidence_decay.py."""

from __future__ import annotations

import pytest

from src.confidence_decay import (
    _DECAY_RATE,
    _DEFAULT_MAX_FRESHNESS,
    _HARD_PENALTY_MULTIPLIER,
    _MAX_FRESHNESS,
    apply_confidence_decay,
)


# ---------------------------------------------------------------------------
# Basic (no decay) – signal generated "now"
# ---------------------------------------------------------------------------


def test_fresh_signal_no_meaningful_decay():
    """A signal with age=0 should have virtually no decay."""
    result = apply_confidence_decay(
        confidence=80.0,
        signal_generated_at=0.0,
        current_time=0.0,  # age = 0
        channel="360_SCALP",
    )
    assert result == pytest.approx(80.0, abs=0.1)


def test_slight_age_reduces_confidence_slightly():
    """After half the max freshness window the signal is gently reduced."""
    max_fresh = _MAX_FRESHNESS["360_SCALP"]  # 60s
    half_age = max_fresh / 2
    result = apply_confidence_decay(
        confidence=100.0,
        signal_generated_at=0.0,
        current_time=half_age,
        channel="360_SCALP",
    )
    # Expected factor: 1 - (0.5 * 0.15) = 0.925
    expected_factor = 1.0 - (0.5 * _DECAY_RATE)
    assert result == pytest.approx(100.0 * expected_factor, abs=0.01)


# ---------------------------------------------------------------------------
# At exactly max_freshness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", list(_MAX_FRESHNESS.keys()))
def test_decay_at_max_freshness(channel: str):
    """Confidence at exactly max_freshness is reduced by _DECAY_RATE fraction."""
    max_fresh = _MAX_FRESHNESS[channel]
    result = apply_confidence_decay(
        confidence=100.0,
        signal_generated_at=0.0,
        current_time=max_fresh,
        channel=channel,
    )
    expected = 100.0 * (1.0 - _DECAY_RATE)
    assert result == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Hard penalty (age > 2× max_freshness)
# ---------------------------------------------------------------------------


def test_hard_penalty_applied_for_very_stale_signal():
    """Age > 2× max_freshness → hard penalty multiplier."""
    max_fresh = _MAX_FRESHNESS["360_SCALP"]  # 60s
    stale_age = max_fresh * 2.5  # > 2×
    result = apply_confidence_decay(
        confidence=80.0,
        signal_generated_at=0.0,
        current_time=stale_age,
        channel="360_SCALP",
    )
    assert result == pytest.approx(80.0 * _HARD_PENALTY_MULTIPLIER, abs=0.01)


def test_hard_penalty_boundary():
    """Exactly 2× max_freshness is NOT yet in hard-penalty territory (uses linear formula)."""
    max_fresh = _MAX_FRESHNESS["360_SCALP"]  # 60s
    boundary_age = max_fresh * 2.0
    result = apply_confidence_decay(
        confidence=100.0,
        signal_generated_at=0.0,
        current_time=boundary_age,
        channel="360_SCALP",
    )
    # Should use linear formula at exactly 2× max_freshness (condition is strict >)
    linear_factor = max(0.0, 1.0 - (boundary_age / max_fresh) * _DECAY_RATE)
    linear_value = 100.0 * linear_factor
    assert result == pytest.approx(linear_value, abs=0.01)

    # Verify that age strictly above 2× uses the hard penalty path.
    # Use a confidence value where hard_penalty ≠ linear to distinguish them.
    # hard = 80 * 0.70 = 56.0
    # linear at age=500s: factor = max(0, 1 - (500/60)*0.15) = max(0, 1 - 1.25) = 0 → 0.0
    just_over = max_fresh * 10  # clearly > 2× max_freshness
    result_over = apply_confidence_decay(
        confidence=80.0,
        signal_generated_at=0.0,
        current_time=just_over,
        channel="360_SCALP",
    )
    expected_hard = 80.0 * _HARD_PENALTY_MULTIPLIER  # 56.0
    assert result_over == pytest.approx(expected_hard, abs=0.01)


# ---------------------------------------------------------------------------
# Clamp: output stays in [0, 100]
# ---------------------------------------------------------------------------


def test_output_never_exceeds_100():
    result = apply_confidence_decay(
        confidence=99.9,
        signal_generated_at=0.0,
        current_time=0.0,
        channel="360_SWING",
    )
    assert result <= 100.0


def test_output_never_below_zero():
    result = apply_confidence_decay(
        confidence=0.0,
        signal_generated_at=0.0,
        current_time=9999.0,
        channel="360_SCALP",
    )
    assert result >= 0.0


# ---------------------------------------------------------------------------
# Unknown channel falls back to _DEFAULT_MAX_FRESHNESS
# ---------------------------------------------------------------------------


def test_unknown_channel_uses_default():
    """An unrecognised channel should use the default freshness window."""
    max_fresh = _DEFAULT_MAX_FRESHNESS
    result = apply_confidence_decay(
        confidence=100.0,
        signal_generated_at=0.0,
        current_time=max_fresh,
        channel="360_UNKNOWN",
    )
    expected = 100.0 * (1.0 - _DECAY_RATE)
    assert result == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Channel-specific windows: faster channels decay earlier
# ---------------------------------------------------------------------------


def test_scalp_decays_faster_than_swing():
    """360_SCALP should reach hard penalty before 360_SWING does."""
    age = 130.0  # 130 seconds (> 2× SCALP 60s max, but < 2× SWING 600s max)
    scalp_result = apply_confidence_decay(
        confidence=80.0,
        signal_generated_at=0.0,
        current_time=age,
        channel="360_SCALP",  # max 60s → > 2× → hard penalty
    )
    swing_result = apply_confidence_decay(
        confidence=80.0,
        signal_generated_at=0.0,
        current_time=age,
        channel="360_SWING",  # max 600s → barely aged → small decay
    )
    assert scalp_result < swing_result
