"""Tests for PR01 – High-Probability Filter Module."""

from __future__ import annotations

import pytest

from src.scanner.filter_module import (
    DEFAULT_PROBABILITY_THRESHOLD,
    _score_spread,
    _score_volume,
    _score_atr,
    _score_regime,
    get_pair_probability,
    is_high_probability,
)


class TestScoreSpread:
    def test_excellent_spread(self):
        assert _score_spread(0.01) == 100.0

    def test_good_spread(self):
        score = _score_spread(0.035)
        assert 70.0 <= score <= 100.0

    def test_fair_spread(self):
        score = _score_spread(0.075)
        assert 40.0 <= score <= 70.0

    def test_poor_spread(self):
        score = _score_spread(0.15)
        assert 0.0 <= score <= 40.0

    def test_zero_spread(self):
        assert _score_spread(0.0) == 100.0

    def test_extreme_spread(self):
        assert _score_spread(0.5) == 0.0


class TestScoreVolume:
    def test_excellent_volume(self):
        assert _score_volume(200_000_000.0) == 100.0

    def test_good_volume(self):
        score = _score_volume(50_000_000.0)
        assert 70.0 <= score <= 100.0

    def test_fair_volume(self):
        score = _score_volume(5_000_000.0)
        assert 40.0 <= score <= 70.0

    def test_zero_volume(self):
        assert _score_volume(0.0) == 0.0

    def test_low_volume(self):
        assert _score_volume(500_000.0) < 40.0


class TestScoreATR:
    def test_ideal_atr(self):
        assert _score_atr(0.5) == 100.0  # Within ideal range

    def test_ideal_atr_boundary(self):
        assert _score_atr(0.2) == 100.0
        assert _score_atr(1.5) == 100.0

    def test_very_low_atr(self):
        score = _score_atr(0.05)
        assert score < 100.0

    def test_zero_atr(self):
        assert _score_atr(0.0) == 20.0

    def test_high_atr(self):
        score = _score_atr(3.0)
        assert 30.0 <= score < 100.0


class TestScoreRegime:
    def test_trending_up(self):
        assert _score_regime("TRENDING_UP") == 85.0

    def test_trending_down(self):
        assert _score_regime("TRENDING_DOWN") == 85.0

    def test_ranging(self):
        assert _score_regime("RANGING") == 75.0

    def test_volatile(self):
        assert _score_regime("VOLATILE") == 55.0

    def test_quiet(self):
        assert _score_regime("QUIET") == 45.0

    def test_unknown_regime(self):
        score = _score_regime("UNKNOWN")
        assert 0.0 <= score <= 100.0

    def test_empty_regime(self):
        score = _score_regime("")
        assert 0.0 <= score <= 100.0

    def test_case_insensitive(self):
        assert _score_regime("trending_up") == _score_regime("TRENDING_UP")


class TestGetPairProbability:
    def _good_data(self, regime: str = "TRENDING_UP") -> dict:
        return {
            "regime": regime,
            "spread_pct": 0.01,
            "volume_24h_usd": 50_000_000.0,
            "atr_pct": 0.5,
            "hit_rate": 0.65,
        }

    def test_returns_float_in_range(self):
        score = get_pair_probability(self._good_data())
        assert 0.0 <= score <= 100.0

    def test_high_quality_pair_scores_high(self):
        score = get_pair_probability(self._good_data())
        assert score >= 70.0

    def test_quiet_regime_scores_lower(self):
        quiet_data = self._good_data("QUIET")
        trending_data = self._good_data("TRENDING_UP")
        assert get_pair_probability(quiet_data) < get_pair_probability(trending_data)

    def test_high_spread_reduces_score(self):
        low_spread = {**self._good_data(), "spread_pct": 0.01}
        high_spread = {**self._good_data(), "spread_pct": 0.25}
        assert get_pair_probability(low_spread) > get_pair_probability(high_spread)

    def test_zero_volume_penalises(self):
        good = self._good_data()
        no_vol = {**good, "volume_24h_usd": 0.0}
        assert get_pair_probability(no_vol) < get_pair_probability(good)

    def test_defaults_used_for_missing_keys(self):
        score = get_pair_probability({})
        assert 0.0 <= score <= 100.0

    def test_hit_rate_clamped_to_1(self):
        data = {**self._good_data(), "hit_rate": 5.0}  # Invalid > 1
        score = get_pair_probability(data)
        assert 0.0 <= score <= 100.0


class TestIsHighProbability:
    def test_high_quality_pair_allowed(self):
        pair_data = {
            "regime": "TRENDING_UP",
            "spread_pct": 0.01,
            "volume_24h_usd": 50_000_000.0,
            "atr_pct": 0.5,
            "hit_rate": 0.7,
        }
        allowed, score = is_high_probability(pair_data, threshold=70.0)
        assert allowed is True
        assert score >= 70.0

    def test_low_quality_pair_blocked(self):
        pair_data = {
            "regime": "QUIET",
            "spread_pct": 0.3,
            "volume_24h_usd": 100_000.0,
            "atr_pct": 0.01,
            "hit_rate": 0.3,
        }
        allowed, score = is_high_probability(pair_data, threshold=70.0)
        assert allowed is False
        assert score < 70.0

    def test_default_threshold(self):
        allowed, score = is_high_probability({})
        assert isinstance(allowed, bool)
        assert isinstance(score, float)

    def test_zero_threshold_always_allows(self):
        pair_data = {
            "regime": "QUIET",
            "spread_pct": 0.5,
            "volume_24h_usd": 0.0,
            "atr_pct": 0.0,
            "hit_rate": 0.0,
        }
        allowed, score = is_high_probability(pair_data, threshold=0.0)
        assert allowed is True

    def test_score_returned_matches_get_pair_probability(self):
        pair_data = {
            "regime": "RANGING",
            "spread_pct": 0.02,
            "volume_24h_usd": 10_000_000.0,
            "atr_pct": 0.4,
            "hit_rate": 0.55,
        }
        direct_score = get_pair_probability(pair_data)
        _, gate_score = is_high_probability(pair_data)
        assert abs(direct_score - gate_score) < 1e-6
