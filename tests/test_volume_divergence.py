"""Tests for src/volume_divergence.py."""

from __future__ import annotations

from src.volume_divergence import (
    DECLINE_THRESHOLD,
    MIN_CANDLE_HISTORY,
    SPIKE_THRESHOLD,
    check_volume_divergence_gate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candles(
    primary_tf: str,
    primary_last_vol: float,
    primary_avg_vol: float,
    higher_tf: str,
    higher_last_vol: float,
    higher_avg_vol: float,
    history: int = 10,
) -> dict[str, dict]:
    """Build a synthetic candles dict for two timeframes."""
    def _vols(last: float, avg: float, n: int) -> list[float]:
        return [avg] * n + [last]

    return {
        primary_tf: {"close": [100.0] * (history + 1), "volume": _vols(primary_last_vol, primary_avg_vol, history)},
        higher_tf: {"close": [100.0] * (history + 1), "volume": _vols(higher_last_vol, higher_avg_vol, history)},
    }


# ---------------------------------------------------------------------------
# Normal volumes → allowed
# ---------------------------------------------------------------------------


def test_normal_volumes_allowed():
    """No spike on either side → allowed."""
    candles = _make_candles("5m", 10.0, 10.0, "15m", 10.0, 10.0)
    allowed, reason = check_volume_divergence_gate("LONG", candles, "5m")
    assert allowed is True
    assert reason == ""


def test_spike_without_higher_tf_decline_allowed():
    """Primary spike alone (higher TF not declining) → allowed."""
    candles = _make_candles("5m", 30.0, 10.0, "15m", 10.0, 10.0)  # primary ratio = 3.0, higher = 1.0
    allowed, reason = check_volume_divergence_gate("LONG", candles, "5m")
    assert allowed is True


def test_higher_tf_decline_without_spike_allowed():
    """Higher TF decline alone (no primary spike) → allowed."""
    candles = _make_candles("5m", 10.0, 10.0, "15m", 5.0, 10.0)  # higher ratio = 0.5
    allowed, reason = check_volume_divergence_gate("LONG", candles, "5m")
    assert allowed is True


# ---------------------------------------------------------------------------
# Divergence detected → blocked
# ---------------------------------------------------------------------------


def test_divergence_blocks_signal():
    """Primary spike + higher TF decline → blocked."""
    # primary ratio = 3.0 (> SPIKE_THRESHOLD=2.0), higher ratio = 0.5 (< DECLINE_THRESHOLD=0.7)
    candles = _make_candles("5m", 30.0, 10.0, "15m", 5.0, 10.0)
    allowed, reason = check_volume_divergence_gate("LONG", candles, "5m")
    assert allowed is False
    assert "divergence" in reason.lower() or "spiking" in reason.lower()


def test_divergence_blocked_for_short():
    """Direction should not affect the divergence check."""
    candles = _make_candles("5m", 30.0, 10.0, "15m", 5.0, 10.0)
    allowed, reason = check_volume_divergence_gate("SHORT", candles, "5m")
    assert allowed is False


def test_divergence_blocked_on_1m_5m():
    """Check divergence detection on 1m primary with 5m higher."""
    candles = _make_candles("1m", 25.0, 10.0, "5m", 6.0, 10.0)
    allowed, reason = check_volume_divergence_gate("LONG", candles, "1m")
    assert allowed is False


# ---------------------------------------------------------------------------
# Fail-open: missing data
# ---------------------------------------------------------------------------


def test_no_candle_data_allows():
    """When candles dict is empty, fail open."""
    allowed, reason = check_volume_divergence_gate("LONG", {}, "5m")
    assert allowed is True
    assert reason == ""


def test_missing_primary_tf_allows():
    """When primary TF data is absent, fail open."""
    candles = {"15m": {"close": [100.0] * 15, "volume": [10.0] * 15}}
    allowed, reason = check_volume_divergence_gate("LONG", candles, "5m")
    assert allowed is True


def test_missing_higher_tf_fails_open():
    """When the higher TF has no data, fail open."""
    primary_vols = [10.0] * 10 + [30.0]
    candles = {"5m": {"close": [100.0] * 11, "volume": primary_vols}}
    # 15m is the higher TF but is not in candles
    allowed, reason = check_volume_divergence_gate("LONG", candles, "5m")
    assert allowed is True


def test_insufficient_volume_history_fails_open():
    """When volume array is shorter than MIN_CANDLE_HISTORY, fail open."""
    short_vol = [10.0] * (MIN_CANDLE_HISTORY - 2)
    candles = {
        "5m": {"close": [100.0] * len(short_vol), "volume": short_vol},
        "15m": {"close": [100.0] * len(short_vol), "volume": short_vol},
    }
    allowed, reason = check_volume_divergence_gate("LONG", candles, "5m")
    assert allowed is True


def test_no_higher_tf_for_top_of_hierarchy_fails_open():
    """4h is the top of the hierarchy — no higher TF exists → fail open."""
    candles = _make_candles("4h", 30.0, 10.0, "1d", 5.0, 10.0)  # 1d not in hierarchy
    allowed, reason = check_volume_divergence_gate("LONG", candles, "4h")
    assert allowed is True


# ---------------------------------------------------------------------------
# Threshold boundary values
# ---------------------------------------------------------------------------


def test_exactly_at_spike_threshold_not_blocked():
    """Primary ratio exactly equal to SPIKE_THRESHOLD is NOT above it → allowed."""
    primary_avg = 10.0
    primary_last = primary_avg * SPIKE_THRESHOLD  # exactly 2.0×
    candles = _make_candles("5m", primary_last, primary_avg, "15m", 5.0, 10.0)
    # ratio == 2.0 which is NOT strictly greater than SPIKE_THRESHOLD
    allowed, _ = check_volume_divergence_gate("LONG", candles, "5m")
    assert allowed is True


def test_exactly_at_decline_threshold_not_blocked():
    """Higher TF ratio exactly equal to DECLINE_THRESHOLD is NOT below it → allowed."""
    primary_avg = 10.0
    primary_last = primary_avg * (SPIKE_THRESHOLD + 0.5)  # clearly above spike threshold
    higher_last = 10.0 * DECLINE_THRESHOLD  # exactly at decline threshold
    candles = _make_candles("5m", primary_last, primary_avg, "15m", higher_last, 10.0)
    # h_ratio == DECLINE_THRESHOLD which is NOT strictly less than it
    allowed, _ = check_volume_divergence_gate("LONG", candles, "5m")
    assert allowed is True
