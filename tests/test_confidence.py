"""Tests for src.confidence – multi-layer confidence scoring."""

import pytest
from datetime import datetime, timezone

from src.confidence import (
    ConfidenceInput,
    compute_confidence,
    get_session_multiplier,
    score_data_sufficiency,
    score_liquidity,
    score_multi_exchange,
    score_order_flow,
    score_smc,
    score_spread,
    score_trend,
)


class TestScoreSMC:
    def test_all_present(self):
        # With no gradient inputs, base scores: sweep=10, mss=11, fvg=2 → 23
        assert score_smc(True, True, True) == 23.0

    def test_none_present(self):
        assert score_smc(False, False, False) == 0.0

    def test_sweep_only_no_depth(self):
        # Base sweep score only (no depth bonus)
        assert score_smc(True, False, False) == 10.0

    def test_sweep_and_mss_no_depth(self):
        assert score_smc(True, True, False) == 21.0

    def test_sweep_with_full_depth_bonus(self):
        # sweep_depth_pct=0.5 → full depth bonus (+5), total = 10 + 5 = 15
        assert score_smc(True, False, False, sweep_depth_pct=0.5) == 15.0

    def test_sweep_with_half_depth_bonus(self):
        # sweep_depth_pct=0.25 → half depth bonus (+2.5), total = 10 + 2.5 = 12.5
        assert score_smc(True, False, False, sweep_depth_pct=0.25) == pytest.approx(12.5)

    def test_fvg_with_atr_ratio_bonus(self):
        # fvg_atr_ratio=1.5 → full size bonus (+2), base 2 + 2 = 4
        assert score_smc(False, False, True, fvg_atr_ratio=1.5) == pytest.approx(4.0)

    def test_all_max_gradient(self):
        # sweep: 10+5=15, mss: 11, fvg: 2+2=4 → 30, capped at 30
        assert score_smc(True, True, True, sweep_depth_pct=0.5, fvg_atr_ratio=1.5) == 30.0

    def test_backward_compat_no_gradient_params(self):
        # Old 3-arg call signature still works
        assert score_smc(True, True, True) == 23.0


class TestScoreTrend:
    def test_all_positive_base_only(self):
        # With no gradient inputs: ema=10, adx base=4, mom base=2 → 16
        assert score_trend(True, True, True) == 16.0

    def test_none(self):
        assert score_trend(False, False, False) == 0.0

    def test_with_adx_at_20_no_bonus(self):
        # ADX=20 → adx_bonus = 0, base only → ema=10, adx=4, mom=2 → 16
        assert score_trend(True, True, True, adx_value=20.0) == pytest.approx(16.0)

    def test_with_adx_at_40_full_bonus(self):
        # ADX=40 → adx_bonus = 5, ema=10, adx=4+5=9, mom=2 → 21
        assert score_trend(True, True, True, adx_value=40.0) == pytest.approx(21.0)

    def test_with_momentum_strength_full_bonus(self):
        # momentum_strength=1.0 → mom_bonus=4, ema=10, adx=4 (no adx_value), mom=2+4=6 → 20
        assert score_trend(True, True, True, momentum_strength=1.0) == pytest.approx(20.0)

    def test_negative_momentum_strength_same_bonus(self):
        # abs(-1.0) = 1.0 → same bonus as +1.0 (useful for SHORT signals)
        assert score_trend(True, True, True, momentum_strength=-1.0) == pytest.approx(20.0)

    def test_all_max_gradient(self):
        # ema=10, adx=4+5=9, mom=2+4=6 → 25
        assert score_trend(True, True, True, adx_value=40.0, momentum_strength=1.0) == pytest.approx(25.0)

    def test_backward_compat_no_gradient_params(self):
        # Old 3-arg call signature still works
        assert score_trend(True, True, True) == 16.0


class TestScoreLiquidity:
    def test_high_volume(self):
        assert score_liquidity(10_000_000) == 20.0

    def test_zero_volume(self):
        assert score_liquidity(0) == 0.0

    def test_partial(self):
        result = score_liquidity(2_500_000)
        assert 0 < result < 20


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
    # Use a fixed EU-session datetime so the session multiplier is always 1.0×,
    # keeping existing behavioural assertions stable regardless of when the
    # tests are run.
    _EU_SESSION = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    def test_basic(self):
        inp = ConfidenceInput(
            smc_score=30,
            trend_score=25,
            liquidity_score=20,
            spread_score=10,
            data_sufficiency=10,
            multi_exchange=5,
        )
        result = compute_confidence(inp, session_now=self._EU_SESSION)
        assert result.total == 100.0

    def test_cap_for_new_pair(self):
        inp = ConfidenceInput(
            smc_score=30,
            trend_score=25,
            liquidity_score=20,
            spread_score=10,
            data_sufficiency=10,
            multi_exchange=5,
            has_enough_history=False,
        )
        result = compute_confidence(inp, session_now=self._EU_SESSION)
        assert result.total == 50.0
        assert result.capped is True

    def test_blocked_by_correlation(self):
        inp = ConfidenceInput(
            smc_score=25,
            trend_score=20,
            opposing_position_open=True,
        )
        result = compute_confidence(inp, session_now=self._EU_SESSION)
        assert result.blocked is True

    def test_zero_inputs(self):
        result = compute_confidence(ConfidenceInput(), session_now=self._EU_SESSION)
        assert result.total == 0.0

    def test_no_ai_sentiment_in_breakdown(self):
        """Breakdown dict must NOT contain an 'ai_sentiment' key."""
        inp = ConfidenceInput(smc_score=10, trend_score=5)
        result = compute_confidence(inp, session_now=self._EU_SESSION)
        assert "ai_sentiment" not in result.breakdown


# ---------------------------------------------------------------------------
# Fix 9: Session-aware confidence multiplier
# ---------------------------------------------------------------------------


class TestGetSessionMultiplier:
    def test_asian_session(self):
        """Hours 0–7 UTC → 0.9× multiplier."""
        for hour in (0, 3, 7):
            t = datetime(2024, 1, 15, hour, 0, 0, tzinfo=timezone.utc)
            assert get_session_multiplier(t) == pytest.approx(0.9), f"hour={hour}"

    def test_eu_session(self):
        """Hours 8–15 UTC → 1.0× multiplier."""
        for hour in (8, 12, 15):
            t = datetime(2024, 1, 15, hour, 0, 0, tzinfo=timezone.utc)
            assert get_session_multiplier(t) == pytest.approx(1.0), f"hour={hour}"

    def test_us_session(self):
        """Hours 16–23 UTC → 1.05× multiplier."""
        for hour in (16, 20, 23):
            t = datetime(2024, 1, 15, hour, 0, 0, tzinfo=timezone.utc)
            assert get_session_multiplier(t) == pytest.approx(1.05), f"hour={hour}"

    def test_compute_confidence_asian_reduces_total(self):
        """Session multiplier 0.9× must reduce total confidence in Asian session."""
        inp = ConfidenceInput(smc_score=20, trend_score=15, liquidity_score=10)
        asian_t = datetime(2024, 1, 15, 3, 0, 0, tzinfo=timezone.utc)
        eu_t = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result_asian = compute_confidence(inp, session_now=asian_t)
        result_eu = compute_confidence(inp, session_now=eu_t)
        assert result_asian.total < result_eu.total

    def test_compute_confidence_us_increases_total(self):
        """Session multiplier 1.05× must increase total confidence in US session."""
        inp = ConfidenceInput(smc_score=20, trend_score=15, liquidity_score=10)
        us_t = datetime(2024, 1, 15, 18, 0, 0, tzinfo=timezone.utc)
        eu_t = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result_us = compute_confidence(inp, session_now=us_t)
        result_eu = compute_confidence(inp, session_now=eu_t)
        assert result_us.total > result_eu.total

    def test_compute_confidence_caps_at_100(self):
        """Even with the 1.05× US multiplier, total must be capped at 100."""
        inp = ConfidenceInput(
            smc_score=30, trend_score=25,
            liquidity_score=20, spread_score=10, data_sufficiency=10, multi_exchange=5,
        )
        us_t = datetime(2024, 1, 15, 18, 0, 0, tzinfo=timezone.utc)
        result = compute_confidence(inp, session_now=us_t)
        assert result.total <= 100.0


# ---------------------------------------------------------------------------
# score_order_flow
# ---------------------------------------------------------------------------


class TestScoreOrderFlow:
    _EU = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    def test_no_data_zero(self):
        assert score_order_flow() == 0.0

    def test_falling_oi_no_liquidations(self):
        # OI falling but zero liq → only the base OI bonus (5)
        assert score_order_flow(oi_trend="FALLING", liq_vol_usd=0.0) == pytest.approx(5.0)

    def test_falling_oi_with_liquidations(self):
        # OI falling + large liq → up to 10 (5 base + up to 5 liq bonus)
        score = score_order_flow(oi_trend="FALLING", liq_vol_usd=500_000.0)
        assert score == pytest.approx(10.0)

    def test_falling_oi_partial_liquidations(self):
        # 250k USD liq = 50% of 500k cap → 2.5 liq bonus, total = 5+2.5 = 7.5
        score = score_order_flow(oi_trend="FALLING", liq_vol_usd=250_000.0)
        assert score == pytest.approx(7.5)

    def test_cvd_divergence_bonus(self):
        # CVD divergence without signal_direction → no bonus (0.0 for safety)
        assert score_order_flow(cvd_divergence="BULLISH") == pytest.approx(0.0)
        assert score_order_flow(cvd_divergence="BEARISH") == pytest.approx(0.0)

    def test_squeeze_plus_cvd(self):
        # Full squeeze (10) + no signal_direction → no CVD bonus, score = 10
        score = score_order_flow(
            oi_trend="FALLING",
            liq_vol_usd=500_000.0,
            cvd_divergence="BULLISH",
        )
        assert score == pytest.approx(10.0)

    def test_cvd_aligned_long(self):
        # LONG signal + BULLISH CVD divergence → aligned → +5
        assert score_order_flow(cvd_divergence="BULLISH", signal_direction="LONG") == pytest.approx(5.0)

    def test_cvd_contra_long(self):
        # LONG signal + BEARISH CVD divergence → contra → −3 floored at 0
        assert score_order_flow(cvd_divergence="BEARISH", signal_direction="LONG") == pytest.approx(0.0)

    def test_cvd_aligned_short(self):
        # SHORT signal + BEARISH CVD divergence → aligned → +5
        assert score_order_flow(cvd_divergence="BEARISH", signal_direction="SHORT") == pytest.approx(5.0)

    def test_cvd_contra_short(self):
        # SHORT signal + BULLISH CVD divergence → contra → −3 floored at 0
        assert score_order_flow(cvd_divergence="BULLISH", signal_direction="SHORT") == pytest.approx(0.0)

    def test_squeeze_plus_aligned_cvd(self):
        # Full squeeze (10) + aligned CVD = 15, capped at 15
        score = score_order_flow(
            oi_trend="FALLING",
            liq_vol_usd=500_000.0,
            cvd_divergence="BULLISH",
            signal_direction="LONG",
        )
        assert score == pytest.approx(15.0)

    def test_squeeze_with_contra_cvd(self):
        # Partial squeeze (5) + contra CVD penalty (−3) = 2
        score = score_order_flow(
            oi_trend="FALLING",
            liq_vol_usd=0.0,
            cvd_divergence="BEARISH",
            signal_direction="LONG",
        )
        assert score == pytest.approx(2.0)

    def test_direction_provided_no_cvd_divergence(self):
        # signal_direction provided but no CVD divergence → CVD component is 0
        assert score_order_flow(signal_direction="LONG") == pytest.approx(0.0)
        assert score_order_flow(signal_direction="SHORT") == pytest.approx(0.0)

    def test_rising_oi_zero(self):
        # Rising OI → no squeeze bonus (returns 0 for OI component)
        assert score_order_flow(oi_trend="RISING", liq_vol_usd=1_000_000.0) == 0.0

    def test_order_flow_score_in_confidence_breakdown(self):
        """order_flow_score must appear in the confidence breakdown dict."""
        inp = ConfidenceInput(smc_score=20, order_flow_score=10.0)
        result = compute_confidence(inp, session_now=self._EU)
        assert "order_flow" in result.breakdown
        assert result.breakdown["order_flow"] == pytest.approx(10.0)

    def test_order_flow_boosts_total(self):
        """order_flow_score must contribute to the total confidence."""
        base = ConfidenceInput(smc_score=20, trend_score=15)
        with_of = ConfidenceInput(smc_score=20, trend_score=15, order_flow_score=15.0)
        r_base = compute_confidence(base, session_now=self._EU)
        r_of = compute_confidence(with_of, session_now=self._EU)
        assert r_of.total > r_base.total

    def test_squeeze_pushes_confidence_near_max(self):
        """A full squeeze scenario should push algorithmic confidence close to 100."""
        inp = ConfidenceInput(
            smc_score=30,       # max SMC
            trend_score=25,     # max trend
            liquidity_score=20, # max liquidity
            spread_score=10,    # max spread
            data_sufficiency=5,
            multi_exchange=5,
            order_flow_score=15.0,  # full squeeze + CVD divergence
        )
        result = compute_confidence(inp, session_now=self._EU)
        assert result.total >= 95.0


# ---------------------------------------------------------------------------
# score_order_flow with funding_rate parameter
# ---------------------------------------------------------------------------


class TestScoreOrderFlowFundingRate:
    def test_funding_rate_contrarian_long(self):
        """Extreme negative funding + LONG signal → contrarian edge bonus."""
        score = score_order_flow(signal_direction="LONG", funding_rate=-0.03)
        assert score == pytest.approx(5.0)

    def test_funding_rate_contrarian_short(self):
        """Extreme positive funding + SHORT signal → contrarian edge bonus."""
        score = score_order_flow(signal_direction="SHORT", funding_rate=0.03)
        assert score == pytest.approx(5.0)

    def test_funding_rate_not_extreme_no_bonus(self):
        """Funding rate below 1% threshold gives no bonus."""
        score = score_order_flow(signal_direction="LONG", funding_rate=-0.005)
        assert score == pytest.approx(0.0)

    def test_funding_rate_aligned_direction_no_bonus(self):
        """Positive funding + LONG is NOT contrarian → no bonus."""
        score = score_order_flow(signal_direction="LONG", funding_rate=0.03)
        assert score == pytest.approx(0.0)

    def test_funding_rate_no_direction_no_bonus(self):
        """funding_rate without signal_direction → no bonus."""
        score = score_order_flow(funding_rate=-0.03)
        assert score == pytest.approx(0.0)

    def test_partial_funding_bonus(self):
        """funding_rate at -1.5% → 2.5 pts bonus (50% of max 5)."""
        score = score_order_flow(signal_direction="LONG", funding_rate=-0.015)
        assert score == pytest.approx(2.5)

    def test_order_flow_capped_at_20(self):
        """Full squeeze + aligned CVD + contrarian funding → capped at 20."""
        score = score_order_flow(
            oi_trend="FALLING",
            liq_vol_usd=500_000.0,
            cvd_divergence="BULLISH",
            signal_direction="LONG",
            funding_rate=-0.03,
        )
        assert score == pytest.approx(20.0)

    def test_backward_compat_no_funding_rate(self):
        """Omitting funding_rate doesn't break existing behavior."""
        # Full squeeze (10) + aligned CVD (5) = 15, capped at old behavior
        score = score_order_flow(
            oi_trend="FALLING",
            liq_vol_usd=500_000.0,
            cvd_divergence="BULLISH",
            signal_direction="LONG",
        )
        assert score == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# score_liquidity with channel parameter
# ---------------------------------------------------------------------------


class TestScoreLiquidityWithChannel:
    def test_gem_full_score_at_250k(self):
        """360_GEM achieves max score at $250K."""
        assert score_liquidity(250_000.0, channel="360_GEM") == pytest.approx(20.0)

    def test_spot_full_score_at_1m(self):
        """360_SPOT achieves max score at $1M."""
        assert score_liquidity(1_000_000.0, channel="360_SPOT") == pytest.approx(20.0)

    def test_swing_full_score_at_10m(self):
        """360_SWING achieves max score at $10M."""
        assert score_liquidity(10_000_000.0, channel="360_SWING") == pytest.approx(20.0)

    def test_scalp_full_score_at_5m(self):
        """360_SCALP uses $5M threshold (default)."""
        assert score_liquidity(5_000_000.0, channel="360_SCALP") == pytest.approx(20.0)

    def test_gem_scores_higher_than_scalp_at_1m(self):
        """GEM scores 20.0 at $1M; SCALP scores only 4.0."""
        assert score_liquidity(1_000_000.0, channel="360_GEM") > score_liquidity(1_000_000.0, channel="360_SCALP")

    def test_unknown_channel_uses_default(self):
        """Unknown channel falls back to default $5M threshold."""
        assert score_liquidity(5_000_000.0, channel="UNKNOWN") == pytest.approx(20.0)

    def test_no_channel_backward_compat(self):
        """Omitting channel preserves old behavior."""
        assert score_liquidity(5_000_000.0) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# get_session_multiplier with channel parameter
# ---------------------------------------------------------------------------


class TestGetSessionMultiplierWithChannel:
    _ASIAN = datetime(2024, 1, 15, 3, 0, 0, tzinfo=timezone.utc)
    _EU = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    _US = datetime(2024, 1, 15, 20, 0, 0, tzinfo=timezone.utc)

    def test_spot_always_one(self):
        """360_SPOT returns 1.0 regardless of session."""
        for t in (self._ASIAN, self._EU, self._US):
            assert get_session_multiplier(t, channel="360_SPOT") == pytest.approx(1.0)

    def test_gem_always_one(self):
        """360_GEM returns 1.0 regardless of session."""
        for t in (self._ASIAN, self._EU, self._US):
            assert get_session_multiplier(t, channel="360_GEM") == pytest.approx(1.0)

    def test_swing_asian_mild_penalty(self):
        """360_SWING: Asian session → 0.95×."""
        assert get_session_multiplier(self._ASIAN, channel="360_SWING") == pytest.approx(0.95)

    def test_swing_eu_neutral(self):
        """360_SWING: EU session → 1.0×."""
        assert get_session_multiplier(self._EU, channel="360_SWING") == pytest.approx(1.0)

    def test_swing_us_mild_boost(self):
        """360_SWING: US session → 1.02×."""
        assert get_session_multiplier(self._US, channel="360_SWING") == pytest.approx(1.02)

    def test_scalp_asian_full_penalty(self):
        """360_SCALP: Asian session → 0.90×."""
        assert get_session_multiplier(self._ASIAN, channel="360_SCALP") == pytest.approx(0.90)

    def test_scalp_us_full_boost(self):
        """360_SCALP: US session → 1.05×."""
        assert get_session_multiplier(self._US, channel="360_SCALP") == pytest.approx(1.05)

    def test_no_channel_default_behavior(self):
        """Omitting channel behaves like SCALP (full impact)."""
        assert get_session_multiplier(self._ASIAN) == pytest.approx(0.90)

    def test_compute_confidence_spot_ignores_session(self):
        """360_SPOT: Asian and EU sessions give same total (1.0× always)."""
        inp = ConfidenceInput(smc_score=20, trend_score=15)
        r_asian = compute_confidence(inp, session_now=self._ASIAN, channel="360_SPOT")
        r_eu = compute_confidence(inp, session_now=self._EU, channel="360_SPOT")
        assert r_asian.total == r_eu.total
