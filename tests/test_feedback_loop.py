"""Tests for src/feedback_loop.py."""

from __future__ import annotations

import time

import pytest

from src.feedback_loop import (
    _EXEC_PENALTY,
    _EXEC_PENALTY_THRESHOLD,
    _MARKET_BOOST,
    _MARKET_BOOST_THRESHOLD,
    _MIN_SAMPLE_SIZE,
    _SETUP_BOOST,
    _SETUP_PENALTY,
    FeedbackLoop,
    TradeOutcome,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outcome(
    channel: str = "360_SCALP",
    setup_class: str = "SWEEP_REVERSAL",
    outcome: str = "TP1",
    r_multiple: float = 1.5,
    execution: float = 15.0,
    market: float = 20.0,
    timestamp: float = 0.0,
) -> TradeOutcome:
    return TradeOutcome(
        symbol="SOLUSDT",
        channel=channel,
        direction="LONG",
        setup_class=setup_class,
        market_state="TRENDING",
        component_scores={
            "market": market,
            "setup": 18.0,
            "execution": execution,
            "risk": 12.0,
            "context": 6.0,
        },
        confidence=72.5,
        r_multiple=r_multiple,
        outcome=outcome,
        hold_duration_seconds=240.0,
        timestamp=timestamp or time.monotonic(),
    )


def _fill_loop(
    loop: FeedbackLoop,
    n: int,
    channel: str,
    setup_class: str,
    outcome_str: str,
) -> None:
    for _ in range(n):
        loop.record_outcome(_outcome(channel=channel, setup_class=setup_class, outcome=outcome_str))


# ---------------------------------------------------------------------------
# Basic outcome recording
# ---------------------------------------------------------------------------


def test_record_outcome_increases_history():
    loop = FeedbackLoop()
    assert len(loop._outcomes) == 0
    loop.record_outcome(_outcome())
    assert len(loop._outcomes) == 1


def test_max_history_evicts_oldest():
    loop = FeedbackLoop(max_history=5)
    for _ in range(10):
        loop.record_outcome(_outcome())
    assert len(loop._outcomes) == 5


# ---------------------------------------------------------------------------
# Win rate computation
# ---------------------------------------------------------------------------


def test_get_setup_win_rate_insufficient_data_returns_neutral():
    loop = FeedbackLoop()
    # < _MIN_SAMPLE_SIZE records → neutral 0.5
    for _ in range(_MIN_SAMPLE_SIZE - 1):
        loop.record_outcome(_outcome(outcome="TP1"))
    rate = loop.get_setup_win_rate("SWEEP_REVERSAL", "360_SCALP")
    assert rate == 0.5


def test_get_setup_win_rate_all_wins():
    loop = FeedbackLoop()
    _fill_loop(loop, _MIN_SAMPLE_SIZE + 5, "360_SCALP", "SWEEP_REVERSAL", "TP1")
    rate = loop.get_setup_win_rate("SWEEP_REVERSAL", "360_SCALP")
    assert rate == pytest.approx(1.0)


def test_get_setup_win_rate_all_losses():
    loop = FeedbackLoop()
    _fill_loop(loop, _MIN_SAMPLE_SIZE + 5, "360_SCALP", "SWEEP_REVERSAL", "SL")
    rate = loop.get_setup_win_rate("SWEEP_REVERSAL", "360_SCALP")
    assert rate == pytest.approx(0.0)


def test_get_setup_win_rate_mixed():
    loop = FeedbackLoop()
    wins = _MIN_SAMPLE_SIZE
    losses = _MIN_SAMPLE_SIZE
    _fill_loop(loop, wins, "360_SCALP", "SWEEP_REVERSAL", "TP1")
    _fill_loop(loop, losses, "360_SCALP", "SWEEP_REVERSAL", "SL")
    rate = loop.get_setup_win_rate("SWEEP_REVERSAL", "360_SCALP")
    assert rate == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Weight adjustments (setup-level)
# ---------------------------------------------------------------------------


def test_setup_penalty_applied_for_low_win_rate():
    loop = FeedbackLoop()
    # Fill with mostly losses so win rate < 40%
    losses = _MIN_SAMPLE_SIZE + 5
    _fill_loop(loop, losses, "360_SCALP", "BAD_SETUP", "SL")
    adj = loop.get_confidence_adjustment({}, "360_SCALP", "BAD_SETUP")
    assert adj <= _SETUP_PENALTY  # should be the penalty value


def test_setup_boost_applied_for_high_win_rate():
    loop = FeedbackLoop()
    # Fill with all wins so win rate > 70%
    _fill_loop(loop, _MIN_SAMPLE_SIZE + 5, "360_SCALP", "GREAT_SETUP", "TP2")
    adj = loop.get_confidence_adjustment({}, "360_SCALP", "GREAT_SETUP")
    assert adj >= _SETUP_BOOST


def test_neutral_win_rate_no_adjustment():
    loop = FeedbackLoop()
    # 50% win rate → no adjustment (between 40% and 70%)
    _fill_loop(loop, _MIN_SAMPLE_SIZE, "360_SCALP", "OK_SETUP", "TP1")
    _fill_loop(loop, _MIN_SAMPLE_SIZE, "360_SCALP", "OK_SETUP", "SL")
    adj = loop.get_confidence_adjustment({}, "360_SCALP", "OK_SETUP")
    assert adj == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Component-level adjustments
# ---------------------------------------------------------------------------


def test_no_adjustment_with_empty_history():
    loop = FeedbackLoop()
    adj = loop.get_confidence_adjustment(
        {"execution": 10.0, "market": 25.0}, "360_SCALP", "SETUP"
    )
    assert adj == pytest.approx(0.0)


def test_exec_penalty_applied_when_history_warrants_it():
    loop = FeedbackLoop()
    # Flood with low-execution losses so _exec_penalty_channels gets "360_SPOT"
    for _ in range(_MIN_SAMPLE_SIZE + 5):
        loop.record_outcome(_outcome(channel="360_SPOT", outcome="SL", execution=10.0))
    # Now a new signal with low execution should receive penalty
    adj = loop.get_confidence_adjustment(
        {"execution": _EXEC_PENALTY_THRESHOLD - 1, "market": 15.0},
        "360_SPOT",
        "",
    )
    assert adj <= _EXEC_PENALTY


def test_market_boost_applied_when_history_warrants_it():
    loop = FeedbackLoop()
    for _ in range(_MIN_SAMPLE_SIZE + 5):
        loop.record_outcome(_outcome(channel="360_SWING", outcome="TP3", market=25.0))
    adj = loop.get_confidence_adjustment(
        {"execution": 15.0, "market": _MARKET_BOOST_THRESHOLD + 1},
        "360_SWING",
        "",
    )
    assert adj >= _MARKET_BOOST


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


def test_adjustment_clamped_to_minus_15():
    loop = FeedbackLoop()
    # Trigger both setup penalty and exec penalty simultaneously
    _fill_loop(loop, _MIN_SAMPLE_SIZE + 5, "360_SPOT", "BAD", "SL")
    for _ in range(_MIN_SAMPLE_SIZE + 5):
        loop.record_outcome(_outcome(channel="360_SPOT", outcome="SL", execution=10.0))
    adj = loop.get_confidence_adjustment(
        {"execution": 5.0, "market": 5.0}, "360_SPOT", "BAD"
    )
    assert adj >= -15.0


def test_adjustment_clamped_to_plus_15():
    loop = FeedbackLoop()
    _fill_loop(loop, _MIN_SAMPLE_SIZE + 5, "360_SCALP", "GREAT", "TP3")
    for _ in range(_MIN_SAMPLE_SIZE + 5):
        loop.record_outcome(_outcome(channel="360_SCALP", outcome="TP3", market=25.0))
    adj = loop.get_confidence_adjustment(
        {"execution": 15.0, "market": 25.0}, "360_SCALP", "GREAT"
    )
    assert adj <= 15.0


# ---------------------------------------------------------------------------
# Time-decay weighting
# ---------------------------------------------------------------------------


def test_time_weight_recent_is_near_one():
    """Very recent outcomes have weight close to 1.0."""
    loop = FeedbackLoop()
    o = _outcome(timestamp=time.monotonic())  # use keyword arg
    # Recent outcome: age ≈ 0 → weight ≈ 1.0
    weight = loop._time_weight(o)
    assert weight == pytest.approx(1.0, abs=0.01)


def test_time_weight_decays_with_age():
    """Older outcome should weigh less than a recent one."""
    loop = FeedbackLoop()
    # Simulate an "old" outcome by setting timestamp to a past time.monotonic()
    new_o = _outcome(timestamp=time.monotonic())
    old_o = _outcome(timestamp=time.monotonic() - 7 * 24 * 3600)  # 7 days ago
    assert loop._time_weight(old_o) < loop._time_weight(new_o)


def test_time_weight_half_life():
    """Outcome exactly at half-life age should have weight ≈ 0.5."""
    from src.feedback_loop import _DECAY_HALF_LIFE_SECONDS
    loop = FeedbackLoop()
    o = _outcome(timestamp=time.monotonic() - _DECAY_HALF_LIFE_SECONDS)
    assert loop._time_weight(o) == pytest.approx(0.5, abs=0.02)


def test_win_rate_dominated_by_recent_wins():
    """Time-decay: 10 old losses + 10 very recent wins → win rate > 0.5."""
    loop = FeedbackLoop()
    # Old losses (simulated ~14 days ago) — each has weight ≈ 0.25
    for _ in range(10):
        old_loss = TradeOutcome(
            symbol="BTCUSDT",
            channel="360_SCALP",
            direction="LONG",
            setup_class="SWEEP_REVERSAL",
            market_state="TRENDING",
            component_scores={"market": 20.0, "setup": 18.0, "execution": 15.0, "risk": 12.0, "context": 6.0},
            confidence=70.0,
            r_multiple=-1.0,
            outcome="SL",
            hold_duration_seconds=120.0,
            timestamp=time.monotonic() - 14 * 24 * 3600,
        )
        loop._outcomes.append(old_loss)
    # Recent wins — each has weight ≈ 1.0
    for _ in range(10):
        new_win = TradeOutcome(
            symbol="BTCUSDT",
            channel="360_SCALP",
            direction="LONG",
            setup_class="SWEEP_REVERSAL",
            market_state="TRENDING",
            component_scores={"market": 20.0, "setup": 18.0, "execution": 15.0, "risk": 12.0, "context": 6.0},
            confidence=70.0,
            r_multiple=2.0,
            outcome="TP2",
            hold_duration_seconds=180.0,
            timestamp=time.monotonic(),
        )
        loop._outcomes.append(new_win)
    loop._recompute_weights()
    rate = loop.get_setup_win_rate("SWEEP_REVERSAL", "360_SCALP")
    # Recent wins (weight≈1 each) dominate old losses (weight≈0.25 each)
    assert rate > 0.5


# ---------------------------------------------------------------------------
# Widened adjustment range
# ---------------------------------------------------------------------------


def test_widened_adj_range_allows_greater_penalty():
    """With _ADJ_MIN=-15 and _SETUP_PENALTY=-8, combined with exec can exceed -10."""
    from src.feedback_loop import _ADJ_MIN, _ADJ_MAX, _SETUP_PENALTY, _SETUP_BOOST
    assert _ADJ_MIN == -15.0
    assert _ADJ_MAX == +15.0
    assert _SETUP_PENALTY == -8.0
    assert _SETUP_BOOST == +5.0


def test_setup_penalty_is_8():
    """_SETUP_PENALTY changed from -5 to -8."""
    assert _SETUP_PENALTY == pytest.approx(-8.0)


def test_setup_boost_is_5():
    """_SETUP_BOOST changed from +3 to +5."""
    assert _SETUP_BOOST == pytest.approx(5.0)
