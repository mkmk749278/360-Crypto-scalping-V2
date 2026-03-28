"""Tests for src.scanner.filter_module — high-probability filter."""

from src.scanner.filter_module import (
    check_pair_probability,
    get_pair_probability,
    get_threshold_for_channel,
)


def test_get_threshold_default():
    t = get_threshold_for_channel("360_SCALP")
    assert t == 70.0


def test_get_threshold_swing():
    t = get_threshold_for_channel("360_SWING")
    assert t == 60.0


def test_get_threshold_regime_adjustment_trending():
    t = get_threshold_for_channel("360_SCALP", regime="TRENDING_UP")
    assert t == 65.0  # 70 - 5


def test_get_threshold_regime_adjustment_quiet():
    t = get_threshold_for_channel("360_SCALP", regime="QUIET")
    assert t == 80.0  # 70 + 10


def test_get_pair_probability_good():
    data = {
        "spread_pct": 0.005,
        "volume_24h_usd": 20_000_000.0,
        "hit_rate": 0.7,
        "atr_percentile": 50.0,
        "liquidity_score": 80.0,
    }
    score = get_pair_probability(data, channel="360_SCALP")
    assert 0.0 <= score <= 100.0
    assert score > 50.0


def test_get_pair_probability_quiet_regime_penalised():
    data = {
        "spread_pct": 0.01,
        "volume_24h_usd": 10_000_000.0,
        "hit_rate": 0.5,
        "atr_percentile": 50.0,
        "liquidity_score": 50.0,
    }
    score_normal = get_pair_probability(data, regime="RANGING")
    score_quiet = get_pair_probability(data, regime="QUIET")
    assert score_quiet < score_normal


def test_check_pair_probability_pass():
    data = {
        "spread_pct": 0.003,
        "volume_24h_usd": 50_000_000.0,
        "hit_rate": 0.8,
        "atr_percentile": 50.0,
        "liquidity_score": 90.0,
    }
    passed, score = check_pair_probability(data, channel="360_SCALP", regime="TRENDING_UP")
    assert passed is True
    assert score > 0


def test_check_pair_probability_fail():
    data = {
        "spread_pct": 0.05,
        "volume_24h_usd": 100_000.0,
        "hit_rate": 0.1,
        "atr_percentile": 95.0,
        "liquidity_score": 5.0,
    }
    passed, score = check_pair_probability(data, channel="360_SCALP")
    assert passed is False
