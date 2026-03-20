"""Smart Money Concepts (SMC) detection algorithms.

* Liquidity Sweep – wick pierces recent high/low, close inside ±0.05 %
* Market Structure Shift (MSS) – lower-timeframe close breaks beyond the
  structural body (Open/Close range) of the sweep candle, not merely the 50 %
  wick midpoint.  This prevents false signals on doji candles.
* Fair Value Gap (FVG) – imbalance gap between candles (used for TP3/exit)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

# Minimum FVG width as a fraction of the reference price.  Gaps narrower than
# this are considered negligible noise and are filtered out.
_FVG_MIN_WIDTH_RATIO: float = 0.0001  # 0.01 % of price


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
    open_price: float = 0.0  # open of the sweep candle; used for MSS body check


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
    open_prices: Optional[NDArray] = None,
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
    open_prices:
        Optional open price array.  When provided, the open of each sweep
        candle is stored in :attr:`LiquiditySweep.open_price` and used by
        :func:`detect_mss` to determine the candle's structural body.
    """
    h = np.asarray(high, dtype=np.float64).ravel()
    l = np.asarray(low, dtype=np.float64).ravel()
    c = np.asarray(close, dtype=np.float64).ravel()

    vol: Optional[NDArray] = None
    if volume is not None:
        vol = np.asarray(volume, dtype=np.float64).ravel()

    op: Optional[NDArray] = None
    if open_prices is not None:
        op = np.asarray(open_prices, dtype=np.float64).ravel()

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
                open_price=float(op[idx]) if op is not None else 0.0,
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
                open_price=float(op[idx]) if op is not None else 0.0,
            ))

    return sweeps


# ---------------------------------------------------------------------------
# Market Structure Shift (MSS)
# ---------------------------------------------------------------------------

def detect_mss(
    sweep: LiquiditySweep,
    ltf_close: NDArray,
) -> Optional[MSSSignal]:
    """Check if the lower-timeframe close confirms a true Market Structure Shift.

    A true MSS requires the lower-timeframe close to break beyond the structural
    body (Open/Close range) of the sweep candle.  Using the full body rather
    than the 50 % wick midpoint prevents false signals on doji candles, where
    the body is negligible and the midpoint sits in dead air.

    When ``sweep.open_price`` is not available (0.0), the sweep candle's
    ``close_price`` is used as the body reference — still stricter than the
    wick midpoint for most candle shapes.
    """
    c = np.asarray(ltf_close, dtype=np.float64).ravel()
    if len(c) < 2:
        return None

    last_close = c[-1]

    # Determine the structural body boundary of the sweep candle.
    # When open_price is available, use the full Open/Close body range.
    # When it is absent (0.0), fall back to close_price as the body reference.
    if sweep.open_price > 0.0:
        body_top = max(sweep.open_price, sweep.close_price)
        body_bottom = min(sweep.open_price, sweep.close_price)
    else:
        body_top = sweep.close_price
        body_bottom = sweep.close_price

    if sweep.direction == Direction.LONG:
        # LONG MSS: LTF close must break above the sweep candle's body top
        if last_close > body_top:
            return MSSSignal(
                index=len(c) - 1,
                direction=Direction.LONG,
                midpoint=body_top,   # stores body_top for downstream anchor use
                confirm_close=last_close,
            )
    else:
        # SHORT MSS: LTF close must break below the sweep candle's body bottom
        if last_close < body_bottom:
            return MSSSignal(
                index=len(c) - 1,
                direction=Direction.SHORT,
                midpoint=body_bottom,  # stores body_bottom for downstream use
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
            gap_high = l[i + 2]
            gap_low = h[i]
            # Filter out negligible gaps (< 0.01% of reference price)
            ref_price = max(abs(gap_high), abs(gap_low), 1e-12)
            if gap_high - gap_low >= ref_price * _FVG_MIN_WIDTH_RATIO:
                zones.append(FVGZone(
                    index=i + 1,
                    direction=Direction.LONG,
                    gap_high=gap_high,
                    gap_low=gap_low,
                ))
        # Bearish FVG
        if h[i + 2] < l[i]:
            gap_high = l[i]
            gap_low = h[i + 2]
            ref_price = max(abs(gap_high), abs(gap_low), 1e-12)
            if gap_high - gap_low >= ref_price * _FVG_MIN_WIDTH_RATIO:
                zones.append(FVGZone(
                    index=i + 1,
                    direction=Direction.SHORT,
                    gap_high=gap_high,
                    gap_low=gap_low,
                ))

    return zones
