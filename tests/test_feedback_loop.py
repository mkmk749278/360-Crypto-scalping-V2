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
        timestamp=time.monotonic(),
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


def test_adjustment_clamped_to_minus_10():
    loop = FeedbackLoop()
    # Trigger both setup penalty and exec penalty simultaneously
    _fill_loop(loop, _MIN_SAMPLE_SIZE + 5, "360_TAPE", "BAD", "SL")
    for _ in range(_MIN_SAMPLE_SIZE + 5):
        loop.record_outcome(_outcome(channel="360_TAPE", outcome="SL", execution=10.0))
    adj = loop.get_confidence_adjustment(
        {"execution": 5.0, "market": 5.0}, "360_TAPE", "BAD"
    )
    assert adj >= -10.0


def test_adjustment_clamped_to_plus_10():
    loop = FeedbackLoop()
    _fill_loop(loop, _MIN_SAMPLE_SIZE + 5, "360_SCALP", "GREAT", "TP3")
    for _ in range(_MIN_SAMPLE_SIZE + 5):
        loop.record_outcome(_outcome(channel="360_SCALP", outcome="TP3", market=25.0))
    adj = loop.get_confidence_adjustment(
        {"execution": 15.0, "market": 25.0}, "360_SCALP", "GREAT"
    )
    assert adj <= 10.0
