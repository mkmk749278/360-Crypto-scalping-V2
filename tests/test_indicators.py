"""Tests for src.indicators – pure-compute technical indicators."""

import numpy as np
import pytest

from src.indicators import adx, atr, bollinger_bands, ema, momentum, rsi, sma


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class TestEMA:
    def test_ema_basic(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = ema(close, 3)
        # First valid value at index 2 = mean(1,2,3) = 2.0
        assert result[2] == pytest.approx(2.0)
        # Subsequent values use EMA formula
        assert not np.isnan(result[-1])

    def test_ema_short_array(self):
        result = ema(np.array([1.0, 2.0]), 5)
        assert all(np.isnan(result))

    def test_ema_length_preserved(self):
        close = np.arange(1.0, 21.0)
        result = ema(close, 5)
        assert len(result) == len(close)


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


class TestSMA:
    def test_sma_basic(self):
        close = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = sma(close, 3)
        assert result[2] == pytest.approx(4.0)  # mean(2,4,6)
        assert result[3] == pytest.approx(6.0)  # mean(4,6,8)
        assert result[4] == pytest.approx(8.0)  # mean(6,8,10)

    def test_sma_short_array(self):
        result = sma(np.array([1.0]), 3)
        assert all(np.isnan(result))


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


class TestADX:
    def test_adx_returns_correct_length(self):
        n = 100
        h = np.random.uniform(100, 110, n)
        l = h - np.random.uniform(1, 5, n)
        c = (h + l) / 2
        result = adx(h, l, c, 14)
        assert len(result) == n

    def test_adx_has_valid_values(self):
        n = 100
        np.random.seed(42)
        h = np.cumsum(np.random.uniform(0.5, 1.5, n)) + 100
        l = h - np.random.uniform(0.5, 2.0, n)
        c = (h + l) / 2
        result = adx(h, l, c, 14)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert all(0 <= v <= 100 for v in valid)

    def test_adx_short_data(self):
        result = adx(np.array([1.0, 2.0]), np.array([0.5, 1.5]), np.array([0.8, 1.8]))
        assert all(np.isnan(result))


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


class TestATR:
    def test_atr_positive(self):
        n = 50
        h = np.linspace(105, 115, n)
        l = np.linspace(95, 105, n)
        c = (h + l) / 2
        result = atr(h, l, c, 14)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert all(v > 0 for v in valid)

    def test_atr_length(self):
        n = 30
        h = np.ones(n) * 10
        l = np.ones(n) * 9
        c = np.ones(n) * 9.5
        result = atr(h, l, c, 14)
        assert len(result) == n


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_rsi_range(self):
        np.random.seed(7)
        close = np.cumsum(np.random.randn(100)) + 100
        result = rsi(close, 14)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert all(0 <= v <= 100 for v in valid)

    def test_rsi_overbought(self):
        # Monotonically rising → RSI should be high
        close = np.linspace(100, 200, 50)
        result = rsi(close, 14)
        valid = result[~np.isnan(result)]
        assert valid[-1] > 70


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_bb_ordering(self):
        close = np.random.uniform(99, 101, 50)
        upper, mid, lower = bollinger_bands(close, 20, 2.0)
        # Where all three are valid, upper >= mid >= lower
        for i in range(19, 50):
            if not (np.isnan(upper[i]) or np.isnan(lower[i])):
                assert upper[i] >= mid[i] >= lower[i]

    def test_bb_length(self):
        close = np.ones(30) * 50
        u, m, l = bollinger_bands(close, 20)
        assert len(u) == len(m) == len(l) == 30


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


class TestMomentum:
    def test_momentum_positive(self):
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = momentum(close, 3)
        assert result[3] == pytest.approx(3.0)  # (103-100)/100 * 100

    def test_momentum_negative(self):
        close = np.array([100.0, 99.0, 98.0, 97.0])
        result = momentum(close, 3)
        assert result[3] == pytest.approx(-3.0)
