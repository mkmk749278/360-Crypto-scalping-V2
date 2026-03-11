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
) -> List[LiquiditySweep]:
    """Detect liquidity sweeps on the last candle relative to recent range."""
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)

    sweeps: List[LiquiditySweep] = []
    n = len(c)
    if n < lookback + 1:
        return sweeps

    idx = n - 1
    recent_high = np.max(h[idx - lookback: idx])
    recent_low = np.min(l[idx - lookback: idx])

    tol_high = recent_high * tolerance_pct / 100.0
    tol_low = recent_low * tolerance_pct / 100.0

    # Bearish sweep (wick above recent high, close back inside)
    if h[idx] > recent_high and c[idx] <= recent_high + tol_high:
        sweeps.append(LiquiditySweep(
            index=idx,
            direction=Direction.SHORT,
            sweep_level=recent_high,
            close_price=c[idx],
            wick_high=h[idx],
            wick_low=l[idx],
        ))

    # Bullish sweep (wick below recent low, close back inside)
    if l[idx] < recent_low and c[idx] >= recent_low - tol_low:
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
    c = np.asarray(ltf_close, dtype=np.float64)
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
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
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
