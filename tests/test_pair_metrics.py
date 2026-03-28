"""Tests for src.pair_metrics — pair-level scoring."""

from src.pair_metrics import (
    PairMetrics,
    compute_pair_score,
    score_spread,
    score_volume,
    score_hit_rate,
    score_volatility,
    score_liquidity,
)


def test_score_spread_tight():
    assert score_spread(0.005, 0.02) >= 70.0


def test_score_spread_at_max():
    assert score_spread(0.02, 0.02) == 0.0


def test_score_spread_zero():
    assert score_spread(0.0, 0.02) == 100.0


def test_score_volume_high():
    assert score_volume(50_000_000.0) >= 90.0


def test_score_volume_below_min():
    assert score_volume(2_000_000.0) < 50.0


def test_score_volume_zero():
    assert score_volume(0.0) == 0.0


def test_score_hit_rate_high():
    assert score_hit_rate(0.8) == 80.0


def test_score_hit_rate_zero():
    assert score_hit_rate(0.0) == 0.0


def test_score_volatility_sweet_spot():
    """ATR percentile in 30-70 range should score high."""
    assert score_volatility(50.0) >= 80.0


def test_score_volatility_extreme_low():
    assert score_volatility(5.0) < 50.0


def test_score_volatility_extreme_high():
    assert score_volatility(95.0) < 50.0


def test_score_liquidity_clamp():
    assert score_liquidity(150.0) == 100.0
    assert score_liquidity(-10.0) == 0.0


def test_compute_pair_score_good_pair():
    metrics = PairMetrics(
        spread_pct=0.005,
        volume_24h_usd=20_000_000.0,
        hit_rate=0.7,
        atr_percentile=50.0,
        liquidity_score=80.0,
    )
    score = compute_pair_score(metrics)
    assert 60.0 <= score <= 100.0


def test_compute_pair_score_bad_pair():
    metrics = PairMetrics(
        spread_pct=0.05,
        volume_24h_usd=100_000.0,
        hit_rate=0.2,
        atr_percentile=95.0,
        liquidity_score=10.0,
    )
    score = compute_pair_score(metrics)
    assert score < 40.0


def test_compute_pair_score_range():
    """Score must be between 0 and 100."""
    metrics = PairMetrics()
    score = compute_pair_score(metrics)
    assert 0.0 <= score <= 100.0
