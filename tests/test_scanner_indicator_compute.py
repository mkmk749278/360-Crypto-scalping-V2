"""Tests for src/scanner/indicator_compute.py."""
import numpy as np
import pytest
from src.scanner.indicator_compute import compute_indicators, compute_indicators_for_candle_dict


def test_compute_indicators_returns_dict():
    """compute_indicators returns a populated dict with sufficient data."""
    np.random.seed(42)
    closes = np.random.uniform(100, 200, 300)
    highs = closes * 1.01
    lows = closes * 0.99
    vols = np.random.uniform(1e6, 1e7, 300)
    result = compute_indicators(closes, highs, lows, vols)
    assert isinstance(result, dict)
    assert "ema9_last" in result
    assert "ema21_last" in result
    assert "rsi" in result
    assert result["rsi"] is None or 0 <= result["rsi"] <= 100
    assert "atr_last" in result
    assert "adx_last" in result
    assert "macd_histogram_last" in result
    assert "momentum_last" in result
    assert "bb_upper" in result


def test_compute_indicators_insufficient_data():
    """compute_indicators returns empty dict with too few bars."""
    closes = np.array([100.0, 101.0])
    result = compute_indicators(closes, closes, closes, closes)
    assert result == {}


def test_compute_indicators_exactly_50_bars():
    """50 bars is the minimum for a useful indicator set."""
    np.random.seed(42)
    closes = np.random.uniform(100, 200, 50)
    highs = closes * 1.01
    lows = closes * 0.99
    vols = np.random.uniform(1e6, 1e7, 50)
    result = compute_indicators(closes, highs, lows, vols)
    assert isinstance(result, dict)
    assert len(result) > 0
    assert "ema9_last" in result


def test_compute_indicators_for_candle_dict():
    """compute_indicators_for_candle_dict processes multiple timeframes."""
    np.random.seed(42)
    n = 300
    candle_dict = {
        "5m": {
            "close": list(np.random.uniform(100, 200, n)),
            "high": list(np.random.uniform(100, 200, n) * 1.01),
            "low": list(np.random.uniform(100, 200, n) * 0.99),
            "volume": list(np.random.uniform(1e6, 1e7, n)),
        },
        "1h": {
            "close": list(np.random.uniform(100, 200, n)),
            "high": list(np.random.uniform(100, 200, n) * 1.01),
            "low": list(np.random.uniform(100, 200, n) * 0.99),
            "volume": list(np.random.uniform(1e6, 1e7, n)),
        },
    }
    result = compute_indicators_for_candle_dict(candle_dict)
    assert "5m" in result
    assert "1h" in result
    assert "ema9_last" in result["5m"]
    assert "ema9_last" in result["1h"]


def test_compute_indicators_for_candle_dict_empty_tf():
    """Timeframes with insufficient data return empty dict."""
    candle_dict = {
        "1m": {
            "close": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "volume": [1e6, 1e6],
        },
    }
    result = compute_indicators_for_candle_dict(candle_dict)
    assert result["1m"] == {}


def test_volume_statistics():
    """Volume ratio and SMA are computed correctly."""
    np.random.seed(42)
    n = 300
    closes = np.random.uniform(100, 200, n)
    highs = closes * 1.01
    lows = closes * 0.99
    vols = np.ones(n) * 1e6
    vols[-1] = 2e6  # Double the last volume
    result = compute_indicators(closes, highs, lows, vols)
    assert "volume_sma20" in result
    assert "volume_ratio" in result
    # SMA-20 includes the last bar (2e6), so mean = (19*1e6 + 2e6)/20 = 1.05e6
    # ratio = 2e6 / 1.05e6 ≈ 1.905
    assert result["volume_ratio"] == pytest.approx(2.0 / 1.05, rel=0.01)


def test_compute_indicators_for_candle_dict_compat_keys():
    """compute_indicators_for_candle_dict uses old scanner-compatible key names."""
    np.random.seed(0)
    n = 100
    closes = np.random.uniform(100, 200, n)
    highs = closes * 1.01
    lows = closes * 0.99
    candle_dict = {
        "5m": {
            "close": list(closes),
            "high": list(highs),
            "low": list(lows),
        }
    }
    result = compute_indicators_for_candle_dict(candle_dict)
    tf_result = result["5m"]
    # Verify backward-compatible keys are present
    assert "ema9_last" in tf_result
    assert "ema21_last" in tf_result
    assert "rsi_last" in tf_result
    assert "atr_last" in tf_result
    assert "adx_last" in tf_result
    assert "bb_upper_last" in tf_result
    assert "bb_mid_last" in tf_result
    assert "bb_lower_last" in tf_result
    assert "momentum_last" in tf_result
    assert "macd_histogram_last" in tf_result
    assert "macd_histogram_prev" in tf_result
