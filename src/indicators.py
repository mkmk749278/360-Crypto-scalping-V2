"""Technical indicators used across all channels.

All functions accept numpy arrays (or lists) and return numpy arrays.
They are pure-compute, no I/O.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def ema(close: NDArray, period: int) -> NDArray:
    """Exponential Moving Average."""
    arr = np.asarray(close, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    if len(arr) < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def sma(close: NDArray, period: int) -> NDArray:
    """Simple Moving Average."""
    arr = np.asarray(close, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    if len(arr) < period:
        return out
    cumsum = np.cumsum(arr)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    out[period - 1:] = cumsum[period - 1:] / period
    return out


# ---------------------------------------------------------------------------
# ADX (Average Directional Index)
# ---------------------------------------------------------------------------

def adx(high: NDArray, low: NDArray, close: NDArray, period: int = 14) -> NDArray:
    """Average Directional Index (Wilder smoothing)."""
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    out = np.full(n, np.nan)
    if n < period * 2:
        return out

    tr = np.maximum(h[1:] - l[1:],
                     np.maximum(np.abs(h[1:] - c[:-1]),
                                np.abs(l[1:] - c[:-1])))
    up_move = h[1:] - h[:-1]
    dn_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    atr_val = np.zeros(len(tr))
    atr_val[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr_val[i] = (atr_val[i - 1] * (period - 1) + tr[i]) / period

    sm_plus = np.zeros(len(tr))
    sm_minus = np.zeros(len(tr))
    sm_plus[period - 1] = np.mean(plus_dm[:period])
    sm_minus[period - 1] = np.mean(minus_dm[:period])
    for i in range(period, len(tr)):
        sm_plus[i] = (sm_plus[i - 1] * (period - 1) + plus_dm[i]) / period
        sm_minus[i] = (sm_minus[i - 1] * (period - 1) + minus_dm[i]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus = np.where(atr_val > 0, 100 * sm_plus / atr_val, 0.0)
        di_minus = np.where(atr_val > 0, 100 * sm_minus / atr_val, 0.0)
        di_sum = di_plus + di_minus
        dx = np.where(di_sum > 0, 100 * np.abs(di_plus - di_minus) / di_sum, 0.0)

    adx_val = np.zeros(len(dx))
    start = 2 * period - 1
    if start < len(dx):
        adx_val[start] = np.mean(dx[period:start + 1])
    for i in range(start + 1, len(dx)):
        adx_val[i] = (adx_val[i - 1] * (period - 1) + dx[i]) / period

    out[2 * period:] = adx_val[2 * period - 1: len(dx)]
    return out


# ---------------------------------------------------------------------------
# ATR (Average True Range)
# ---------------------------------------------------------------------------

def atr(high: NDArray, low: NDArray, close: NDArray, period: int = 14) -> NDArray:
    """Average True Range (Wilder smoothing)."""
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    tr = np.maximum(h[1:] - l[1:],
                     np.maximum(np.abs(h[1:] - c[:-1]),
                                np.abs(l[1:] - c[:-1])))
    atr_arr = np.zeros(len(tr))
    atr_arr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i]) / period

    out[period:] = atr_arr[period - 1:]
    return out


# ---------------------------------------------------------------------------
# RSI (Relative Strength Index)
# ---------------------------------------------------------------------------

def rsi(close: NDArray, period: int = 14) -> NDArray:
    """Relative Strength Index (Wilder smoothing)."""
    arr = np.asarray(close, dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    # First valid RSI value
    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + np.mean(gains[:period]) / max(np.mean(losses[:period]), 1e-10))
    return out


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    close: NDArray, period: int = 20, num_std: float = 2.0
) -> tuple[NDArray, NDArray, NDArray]:
    """Return (upper, middle, lower) Bollinger Bands."""
    mid = sma(close, period)
    arr = np.asarray(close, dtype=np.float64)
    std = np.full_like(arr, np.nan)
    for i in range(period - 1, len(arr)):
        std[i] = np.std(arr[i - period + 1: i + 1], ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ---------------------------------------------------------------------------
# Volume Delta (simple tick-level approximation)
# ---------------------------------------------------------------------------

def volume_delta(buy_vol: NDArray, sell_vol: NDArray) -> NDArray:
    """Cumulative Volume Delta."""
    return np.cumsum(np.asarray(buy_vol, dtype=np.float64)
                     - np.asarray(sell_vol, dtype=np.float64))


# ---------------------------------------------------------------------------
# Momentum (% change over N candles)
# ---------------------------------------------------------------------------

def momentum(close: NDArray, n: int = 3) -> NDArray:
    """Percentage change over *n* candles."""
    arr = np.asarray(close, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    if len(arr) <= n:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[n:] = (arr[n:] - arr[:-n]) / arr[:-n] * 100.0
    return out
