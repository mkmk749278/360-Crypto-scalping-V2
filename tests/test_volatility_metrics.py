"""Tests for src.volatility_metrics — dynamic SL/TP helpers."""

from src.volatility_metrics import (
    VolatilityProfile,
    calculate_dynamic_sl_tp,
    compute_regime_sl_multiplier,
    compute_regime_tp_multiplier,
    compute_volatility_adjusted_sl,
    compute_volatility_adjusted_tp_ratios,
)


def test_regime_sl_multiplier_quiet():
    assert compute_regime_sl_multiplier("QUIET") == 1.25


def test_regime_sl_multiplier_trending():
    assert compute_regime_sl_multiplier("TRENDING_UP") == 0.85


def test_regime_tp_multiplier_trending():
    assert compute_regime_tp_multiplier("TRENDING_UP") == 1.20


def test_regime_tp_multiplier_quiet():
    assert compute_regime_tp_multiplier("QUIET") == 0.80


def test_volatility_adjusted_sl_quiet_widens():
    profile = VolatilityProfile(regime="QUIET", pair_tier="MIDCAP")
    adjusted = compute_volatility_adjusted_sl(100.0, profile)
    assert adjusted > 100.0  # Wider SL in QUIET


def test_volatility_adjusted_sl_trending_tightens():
    profile = VolatilityProfile(regime="TRENDING_UP", pair_tier="MIDCAP")
    adjusted = compute_volatility_adjusted_sl(100.0, profile)
    assert adjusted < 100.0  # Tighter SL in TRENDING


def test_volatility_adjusted_tp_ratios_trending():
    profile = VolatilityProfile(regime="TRENDING_UP")
    adjusted = compute_volatility_adjusted_tp_ratios([1.5, 2.5, 4.0], profile)
    assert all(a > b for a, b in zip(adjusted, [1.5, 2.5, 4.0]))


def test_volatility_adjusted_tp_ratios_quiet():
    profile = VolatilityProfile(regime="QUIET")
    adjusted = compute_volatility_adjusted_tp_ratios([1.5, 2.5, 4.0], profile)
    assert all(a < b for a, b in zip(adjusted, [1.5, 2.5, 4.0]))


def test_calculate_dynamic_sl_tp():
    sl, tp = calculate_dynamic_sl_tp(
        pair="BTCUSDT",
        regime="TRENDING_UP",
        volatility_pct=3.0,
        hit_rate=0.65,
        base_sl_distance=50.0,
        base_tp_ratios=[1.5, 2.5, 4.0],
        pair_tier="MAJOR",
        atr_percentile=50.0,
    )
    assert sl > 0
    assert len(tp) == 3
    assert all(r > 0 for r in tp)


def test_high_atr_percentile_widens_sl():
    profile_normal = VolatilityProfile(regime="RANGING", atr_percentile=50.0)
    profile_high = VolatilityProfile(regime="RANGING", atr_percentile=90.0)
    sl_normal = compute_volatility_adjusted_sl(100.0, profile_normal)
    sl_high = compute_volatility_adjusted_sl(100.0, profile_high)
    assert sl_high > sl_normal


def test_poor_hit_rate_widens_sl():
    profile_good = VolatilityProfile(regime="RANGING", historical_hit_rate=0.8)
    profile_poor = VolatilityProfile(regime="RANGING", historical_hit_rate=0.3)
    sl_good = compute_volatility_adjusted_sl(100.0, profile_good)
    sl_poor = compute_volatility_adjusted_sl(100.0, profile_poor)
    assert sl_poor > sl_good
