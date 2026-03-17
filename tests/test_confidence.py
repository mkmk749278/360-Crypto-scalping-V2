"""Tests for src.confidence – multi-layer confidence scoring."""

import pytest

from src.confidence import (
    ConfidenceInput,
    compute_confidence,
    score_ai_sentiment,
    score_data_sufficiency,
    score_liquidity,
    score_multi_exchange,
    score_smc,
    score_spread,
    score_trend,
)


class TestScoreSMC:
    def test_all_present(self):
        assert score_smc(True, True, True) == 25.0

    def test_none_present(self):
        assert score_smc(False, False, False) == 0.0

    def test_sweep_only(self):
        assert score_smc(True, False, False) == 12.0

    def test_sweep_and_mss(self):
        assert score_smc(True, True, False) == 21.0


class TestScoreTrend:
    def test_all_positive(self):
        assert score_trend(True, True, True) == 20.0

    def test_none(self):
        assert score_trend(False, False, False) == 0.0


class TestScoreAISentiment:
    def test_neutral(self):
        assert score_ai_sentiment(0.0) == pytest.approx(7.5)

    def test_bullish(self):
        assert score_ai_sentiment(1.0) == 15.0

    def test_bearish(self):
        assert score_ai_sentiment(-1.0) == 0.0


class TestScoreLiquidity:
    def test_high_volume(self):
        assert score_liquidity(10_000_000) == 15.0

    def test_zero_volume(self):
        assert score_liquidity(0) == 0.0

    def test_partial(self):
        result = score_liquidity(2_500_000)
        assert 0 < result < 15


class TestScoreSpread:
    def test_zero_spread(self):
        assert score_spread(0.0) == 10.0

    def test_max_spread(self):
        assert score_spread(0.02) == 0.0

    def test_half_spread(self):
        assert score_spread(0.01) == pytest.approx(5.0)


class TestScoreDataSufficiency:
    def test_enough(self):
        assert score_data_sufficiency(500) == 10.0

    def test_partial(self):
        assert score_data_sufficiency(250) == pytest.approx(5.0)


class TestScoreMultiExchange:
    def test_verified_true(self):
        assert score_multi_exchange(True) == 5.0

    def test_verified_false(self):
        assert score_multi_exchange(False) == 0.0

    def test_neutral_none(self):
        assert score_multi_exchange(None) == pytest.approx(2.5)

    def test_default_is_neutral(self):
        assert score_multi_exchange() == pytest.approx(2.5)


class TestComputeConfidence:
    def test_basic(self):
        inp = ConfidenceInput(
            smc_score=25,
            trend_score=20,
            ai_sentiment_score=15,
            liquidity_score=15,
            spread_score=10,
            data_sufficiency=10,
            multi_exchange=5,
        )
        result = compute_confidence(inp)
        assert result.total == 100.0

    def test_cap_for_new_pair(self):
        inp = ConfidenceInput(
            smc_score=25,
            trend_score=20,
            ai_sentiment_score=15,
            liquidity_score=15,
            spread_score=10,
            data_sufficiency=10,
            multi_exchange=5,
            has_enough_history=False,
        )
        result = compute_confidence(inp)
        assert result.total == 50.0
        assert result.capped is True

    def test_blocked_by_correlation(self):
        inp = ConfidenceInput(
            smc_score=25,
            trend_score=20,
            opposing_position_open=True,
        )
        result = compute_confidence(inp)
        assert result.blocked is True

    def test_zero_inputs(self):
        result = compute_confidence(ConfidenceInput())
        assert result.total == 0.0
