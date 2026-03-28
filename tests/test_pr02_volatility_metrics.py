"""Tests for PR02 – Volatility Metrics Helpers."""

from __future__ import annotations

import pytest

from src.volatility_metrics import (
    compute_atr_pct,
    calculate_dynamic_sl_tp,
    classify_volatility,
    _HIGH_VOL_ATR_PCT_THRESHOLD,
    _LOW_VOL_ATR_PCT_THRESHOLD,
)


class TestComputeAtrPct:
    def test_normal_atr(self):
        pct = compute_atr_pct(50.0, 10000.0)
        assert abs(pct - 0.5) < 1e-9

    def test_zero_price_returns_zero(self):
        assert compute_atr_pct(10.0, 0.0) == 0.0

    def test_negative_price_returns_zero(self):
        assert compute_atr_pct(10.0, -100.0) == 0.0

    def test_small_values(self):
        pct = compute_atr_pct(0.001, 1.0)
        assert abs(pct - 0.1) < 1e-9


class TestCalculateDynamicSlTp:
    def test_returns_tuple(self):
        sl_mult, tp_ratios = calculate_dynamic_sl_tp("BTCUSDT", "TRENDING_UP", 0.5)
        assert isinstance(sl_mult, float)
        assert isinstance(tp_ratios, list)
        assert len(tp_ratios) >= 2

    def test_high_vol_widens_sl(self):
        sl_high, _ = calculate_dynamic_sl_tp("BTCUSDT", "RANGING", _HIGH_VOL_ATR_PCT_THRESHOLD + 0.5)
        sl_normal, _ = calculate_dynamic_sl_tp("BTCUSDT", "RANGING", 0.5)
        assert sl_high > sl_normal

    def test_low_vol_tightens_sl(self):
        sl_low, _ = calculate_dynamic_sl_tp("BTCUSDT", "RANGING", _LOW_VOL_ATR_PCT_THRESHOLD - 0.1)
        sl_normal, _ = calculate_dynamic_sl_tp("BTCUSDT", "RANGING", 0.5)
        assert sl_low < sl_normal

    def test_volatile_regime_widens_sl(self):
        sl_vol, _ = calculate_dynamic_sl_tp("BTCUSDT", "VOLATILE", 0.5)
        sl_trending, _ = calculate_dynamic_sl_tp("BTCUSDT", "TRENDING_UP", 0.5)
        assert sl_vol > sl_trending

    def test_altcoin_tier_widens_sl(self):
        sl_alt, _ = calculate_dynamic_sl_tp("DOGEUSDT", "RANGING", 0.5, pair_tier="ALTCOIN")
        sl_major, _ = calculate_dynamic_sl_tp("BTCUSDT", "RANGING", 0.5, pair_tier="MAJOR")
        assert sl_alt > sl_major

    def test_high_hit_rate_boosts_tp(self):
        _, tp_high = calculate_dynamic_sl_tp("BTCUSDT", "TRENDING_UP", 0.5, hit_rate=0.8)
        _, tp_normal = calculate_dynamic_sl_tp("BTCUSDT", "TRENDING_UP", 0.5, hit_rate=0.5)
        # TP ratios should be higher with better hit rate
        assert tp_high[0] >= tp_normal[0]

    def test_custom_base_tp_ratios(self):
        _, tp = calculate_dynamic_sl_tp(
            "BTCUSDT", "RANGING", 0.5,
            base_tp_ratios=[0.6, 1.2, 2.0],
        )
        assert len(tp) == 3

    def test_base_sl_mult_applied(self):
        sl_default, _ = calculate_dynamic_sl_tp("BTCUSDT", "TRENDING_UP", 0.5, base_sl_mult=1.0)
        sl_doubled, _ = calculate_dynamic_sl_tp("BTCUSDT", "TRENDING_UP", 0.5, base_sl_mult=2.0)
        assert abs(sl_doubled / sl_default - 2.0) < 0.01

    def test_sl_is_positive(self):
        sl, _ = calculate_dynamic_sl_tp("BTCUSDT", "QUIET", 0.1)
        assert sl > 0


class TestClassifyVolatility:
    def test_high(self):
        assert classify_volatility(_HIGH_VOL_ATR_PCT_THRESHOLD + 0.1) == "HIGH"

    def test_low(self):
        assert classify_volatility(_LOW_VOL_ATR_PCT_THRESHOLD - 0.1) == "LOW"

    def test_normal(self):
        assert classify_volatility(0.5) == "NORMAL"

    def test_exactly_at_boundary(self):
        # Boundary values are classified normally
        label = classify_volatility(_HIGH_VOL_ATR_PCT_THRESHOLD)
        assert label in ("HIGH", "NORMAL")
