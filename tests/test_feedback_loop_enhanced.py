"""Tests for enhanced feedback loop (PR: Feedback Loop Integration)."""

from __future__ import annotations

import time

import pytest

from src.feedback_loop import FeedbackLoop, TradeOutcome


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


class TestRewardPunish:
    def test_reward_signal_records_win(self):
        loop = FeedbackLoop()
        loop.reward_signal("BTCUSDT", "360_SCALP", "SWEEP_REVERSAL")
        assert len(loop._outcomes) == 1
        assert loop._outcomes[0].outcome == "TP1"
        assert loop._outcomes[0].r_multiple > 0

    def test_punish_signal_records_loss(self):
        loop = FeedbackLoop()
        loop.punish_signal("ETHUSDT", "360_SWING", "BAD_SETUP")
        assert len(loop._outcomes) == 1
        assert loop._outcomes[0].outcome == "SL"
        assert loop._outcomes[0].r_multiple < 0

    def test_reward_uses_positive_r(self):
        loop = FeedbackLoop()
        loop.reward_signal("BTC", "360_SCALP", "SETUP", r_multiple=-2.0)
        assert loop._outcomes[0].r_multiple == 2.0  # abs applied

    def test_punish_uses_negative_r(self):
        loop = FeedbackLoop()
        loop.punish_signal("BTC", "360_SCALP", "SETUP", r_multiple=1.5)
        assert loop._outcomes[0].r_multiple == -1.5  # negated


class TestFeedbackMetrics:
    def test_empty_metrics(self):
        loop = FeedbackLoop()
        metrics = loop.get_feedback_metrics()
        assert metrics["total_outcomes"] == 0
        assert metrics["overall_win_rate"] == 0.0
        assert metrics["per_channel"] == {}
        assert metrics["per_setup"] == {}

    def test_metrics_with_data(self):
        loop = FeedbackLoop()
        for _ in range(6):
            loop.record_outcome(_outcome(outcome="TP1", r_multiple=1.5))
        for _ in range(4):
            loop.record_outcome(_outcome(outcome="SL", r_multiple=-1.0))

        metrics = loop.get_feedback_metrics()
        assert metrics["total_outcomes"] == 10
        assert metrics["overall_win_rate"] == pytest.approx(0.6)
        assert "360_SCALP" in metrics["per_channel"]
        assert "SWEEP_REVERSAL" in metrics["per_setup"]

    def test_per_channel_win_rate(self):
        loop = FeedbackLoop()
        for _ in range(5):
            loop.record_outcome(_outcome(channel="360_SCALP", outcome="TP1"))
        for _ in range(5):
            loop.record_outcome(_outcome(channel="360_SWING", outcome="SL"))

        metrics = loop.get_feedback_metrics()
        assert metrics["per_channel"]["360_SCALP"]["win_rate"] == 1.0
        assert metrics["per_channel"]["360_SWING"]["win_rate"] == 0.0

    def test_per_setup_metrics(self):
        loop = FeedbackLoop()
        for _ in range(3):
            loop.record_outcome(_outcome(setup_class="GOOD", outcome="TP2", r_multiple=2.0))
        for _ in range(7):
            loop.record_outcome(_outcome(setup_class="BAD", outcome="SL", r_multiple=-1.0))

        metrics = loop.get_feedback_metrics()
        assert metrics["per_setup"]["GOOD"]["win_rate"] == 1.0
        assert metrics["per_setup"]["BAD"]["win_rate"] == 0.0

    def test_avg_r_multiple(self):
        loop = FeedbackLoop()
        loop.record_outcome(_outcome(r_multiple=2.0, outcome="TP1"))
        loop.record_outcome(_outcome(r_multiple=-1.0, outcome="SL"))
        metrics = loop.get_feedback_metrics()
        assert metrics["avg_r_multiple"] == pytest.approx(0.5)


class TestRetrainingData:
    def test_get_retraining_data(self):
        loop = FeedbackLoop()
        loop.record_outcome(_outcome())
        data = loop.get_retraining_data()
        assert len(data) == 1
        assert "symbol" in data[0]
        assert "is_win" in data[0]
        assert data[0]["is_win"] is True

    def test_retraining_data_includes_component_scores(self):
        loop = FeedbackLoop()
        loop.record_outcome(_outcome())
        data = loop.get_retraining_data()
        assert "score_market" in data[0]
        assert "score_execution" in data[0]

    def test_should_retrain_false_when_empty(self):
        loop = FeedbackLoop()
        assert loop.should_retrain(min_new_outcomes=10) is False

    def test_should_retrain_true_when_enough(self):
        loop = FeedbackLoop()
        for _ in range(15):
            loop.record_outcome(_outcome())
        assert loop.should_retrain(min_new_outcomes=10) is True
