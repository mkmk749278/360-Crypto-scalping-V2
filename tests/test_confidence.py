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
        # With no gradient inputs, base scores: sweep=8, mss=9, fvg=2 → 19
        assert score_smc(True, True, True) == 19.0

    def test_none_present(self):
        assert score_smc(False, False, False) == 0.0

    def test_sweep_only_no_depth(self):
        # Base sweep score only (no depth bonus)
        assert score_smc(True, False, False) == 8.0

    def test_sweep_and_mss_no_depth(self):
        assert score_smc(True, True, False) == 17.0

    def test_sweep_with_full_depth_bonus(self):
        # sweep_depth_pct=0.5 → full depth bonus (+4), total = 8 + 4 = 12
        assert score_smc(True, False, False, sweep_depth_pct=0.5) == 12.0

    def test_sweep_with_half_depth_bonus(self):
        # sweep_depth_pct=0.25 → half depth bonus (+2), total = 8 + 2 = 10
        assert score_smc(True, False, False, sweep_depth_pct=0.25) == pytest.approx(10.0)

    def test_fvg_with_atr_ratio_bonus(self):
        # fvg_atr_ratio=1.5 → full size bonus (+2), base 2 + 2 = 4
        assert score_smc(False, False, True, fvg_atr_ratio=1.5) == pytest.approx(4.0)

    def test_all_max_gradient(self):
        # sweep: 8+4=12, mss: 9, fvg: 2+2=4 → 25, capped at 25
        assert score_smc(True, True, True, sweep_depth_pct=0.5, fvg_atr_ratio=1.5) == 25.0

    def test_backward_compat_no_gradient_params(self):
        # Old 3-arg call signature still works
        assert score_smc(True, True, True) == 19.0


class TestScoreTrend:
    def test_all_positive_base_only(self):
        # With no gradient inputs: ema=8, adx base=3, mom base=2 → 13
        assert score_trend(True, True, True) == 13.0

    def test_none(self):
        assert score_trend(False, False, False) == 0.0

    def test_with_adx_at_20_no_bonus(self):
        # ADX=20 → adx_bonus = 0, base only → ema=8, adx=3, mom=2 → 13
        assert score_trend(True, True, True, adx_value=20.0) == pytest.approx(13.0)

    def test_with_adx_at_40_full_bonus(self):
        # ADX=40 → adx_bonus = 4, ema=8, adx=3+4=7, mom=2 → 17
        assert score_trend(True, True, True, adx_value=40.0) == pytest.approx(17.0)

    def test_with_momentum_strength_full_bonus(self):
        # momentum_strength=1.0 → mom_bonus=3, ema=8, adx=3 (no adx_value), mom=2+3=5 → 16
        assert score_trend(True, True, True, momentum_strength=1.0) == pytest.approx(16.0)

    def test_negative_momentum_strength_same_bonus(self):
        # abs(-1.0) = 1.0 → same bonus as +1.0 (useful for SHORT signals)
        assert score_trend(True, True, True, momentum_strength=-1.0) == pytest.approx(16.0)

    def test_all_max_gradient(self):
        # ema=8, adx=3+4=7, mom=2+3=5 → 20
        assert score_trend(True, True, True, adx_value=40.0, momentum_strength=1.0) == pytest.approx(20.0)

    def test_backward_compat_no_gradient_params(self):
        # Old 3-arg call signature still works
        assert score_trend(True, True, True) == 13.0


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
