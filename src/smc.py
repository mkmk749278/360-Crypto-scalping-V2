"""Smart Money Concepts (SMC) detection algorithms.

* Liquidity Sweep – wick pierces recent high/low, close inside ±0.05 %
* Market Structure Shift (MSS) – close beyond 50 % midpoint of sweep candle
  on a lower timeframe
* Fair Value Gap (FVG) – imbalance gap between candles (used for TP3/exit)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class LiquiditySweep:
    """Detected liquidity sweep."""
    index: int
    direction: Direction
    sweep_level: float
    close_price: float
    wick_high: float
    wick_low: float


@dataclass
class MSSSignal:
    """Market Structure Shift confirmation."""
    index: int
    direction: Direction
    midpoint: float
    confirm_close: float


@dataclass
class FVGZone:
    """Fair Value Gap zone."""
    index: int
    direction: Direction
    gap_high: float
    gap_low: float


# ---------------------------------------------------------------------------
# Liquidity Sweep detection
# ---------------------------------------------------------------------------

def detect_liquidity_sweeps(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    lookback: int = 50,
    tolerance_pct: float = 0.05,
    scan_window: int = 5,
    volume: Optional[NDArray] = None,
    volume_multiplier: float = 1.2,
) -> List[LiquiditySweep]:
    """Detect liquidity sweeps over the last *scan_window* candles.

    Parameters
    ----------
    high, low, close:
        OHLCV price arrays.
    lookback:
        Number of prior candles used to establish the recent high/low range.
    tolerance_pct:
        Wick must close back within this percentage of the swept level.
    scan_window:
        Number of recent candles to scan for sweeps (default 5).  Previously
        only the last candle (scan_window=1) was checked; expanding to 5 catches
        sweeps that occurred 2–4 candles ago.
    volume:
        Optional volume array.  When provided, a sweep candle must have volume
        >= ``volume_multiplier`` × the recent average volume to be counted.
        Low-volume wicks that barely pierce a level are filtered out.
    volume_multiplier:
        Minimum ratio of sweep-candle volume to recent average volume.
        Defaults to 1.2 (sweep candle must be at least 20 % above average).
    """
    h = np.asarray(high, dtype=np.float64).ravel()
    l = np.asarray(low, dtype=np.float64).ravel()
    c = np.asarray(close, dtype=np.float64).ravel()

    vol: Optional[NDArray] = None
    if volume is not None:
        vol = np.asarray(volume, dtype=np.float64).ravel()

    sweeps: List[LiquiditySweep] = []
    n = len(c)
    if n < lookback + 1:
        return sweeps

    seen: set = set()  # deduplicate by (index, direction)

    # Scan the last `scan_window` candles instead of just the last one
    for offset in range(scan_window):
        idx = n - 1 - offset
        if idx < lookback:
            break

        # Recent range is always measured relative to the *current* last candle
        # window so that repeated detections for the same event are consistent.
        recent_high = np.max(h[idx - lookback: idx])
        recent_low = np.min(l[idx - lookback: idx])

        tol_high = recent_high * tolerance_pct / 100.0
        tol_low = recent_low * tolerance_pct / 100.0

        # Volume confirmation: skip low-volume wicks when volume data is available
        volume_ok = True
        if vol is not None and idx >= lookback:
            avg_vol = np.mean(vol[idx - lookback: idx])
            if avg_vol > 0 and vol[idx] < volume_multiplier * avg_vol:
                volume_ok = False

        if not volume_ok:
            continue

        # Bearish sweep (wick above recent high, close back inside)
        key_short = (idx, "SHORT")
        if key_short not in seen and h[idx] > recent_high and c[idx] <= recent_high + tol_high:
            seen.add(key_short)
            sweeps.append(LiquiditySweep(
                index=idx,
                direction=Direction.SHORT,
                sweep_level=recent_high,
                close_price=c[idx],
                wick_high=h[idx],
                wick_low=l[idx],
            ))

        # Bullish sweep (wick below recent low, close back inside)
        key_long = (idx, "LONG")
        if key_long not in seen and l[idx] < recent_low and c[idx] >= recent_low - tol_low:
            seen.add(key_long)
            sweeps.append(LiquiditySweep(
                index=idx,
                direction=Direction.LONG,
                sweep_level=recent_low,
                close_price=c[idx],
                wick_high=h[idx],
                wick_low=l[idx],
            ))

    return sweeps


# ---------------------------------------------------------------------------
# Market Structure Shift (MSS)
# ---------------------------------------------------------------------------

def detect_mss(
    sweep: LiquiditySweep,
    ltf_close: NDArray,
) -> Optional[MSSSignal]:
    """Check if the lower-timeframe close confirms MSS.

    MSS = close beyond 50 % midpoint of the sweep candle.
    """
    c = np.asarray(ltf_close, dtype=np.float64).ravel()
    if len(c) < 2:
        return None

    midpoint = (sweep.wick_high + sweep.wick_low) / 2.0
    last_close = c[-1]

    if sweep.direction == Direction.LONG:
        if last_close > midpoint:
            return MSSSignal(
                index=len(c) - 1,
                direction=Direction.LONG,
                midpoint=midpoint,
                confirm_close=last_close,
            )
    else:
        if last_close < midpoint:
            return MSSSignal(
                index=len(c) - 1,
                direction=Direction.SHORT,
                midpoint=midpoint,
                confirm_close=last_close,
            )
    return None


# ---------------------------------------------------------------------------
# Fair Value Gap (FVG) detection
# ---------------------------------------------------------------------------

def detect_fvg(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    lookback: int = 10,
) -> List[FVGZone]:
    """Detect Fair Value Gaps in recent candles.

    Bullish FVG: low[i+2] > high[i]  (gap up)
    Bearish FVG: high[i+2] < low[i]  (gap down)
    """
    h = np.asarray(high, dtype=np.float64).ravel()
    l = np.asarray(low, dtype=np.float64).ravel()
    n = len(h)

    zones: List[FVGZone] = []
    start = max(0, n - lookback - 2)
    for i in range(start, n - 2):
        # Bullish FVG
        if l[i + 2] > h[i]:
            zones.append(FVGZone(
                index=i + 1,
                direction=Direction.LONG,
                gap_high=l[i + 2],
                gap_low=h[i],
            ))
        # Bearish FVG
        if h[i + 2] < l[i]:
            zones.append(FVGZone(
                index=i + 1,
                direction=Direction.SHORT,
                gap_high=l[i],
                gap_low=h[i + 2],
            ))

    return zones
