"""Volatility Metrics Helpers (PR02) – dynamic SL/TP support functions.

Provides pair-level volatility measurements used by the dynamic SL/TP
system so that stop distances and profit targets adapt to each pair's
current volatility environment rather than using static percentages.

Typical usage::

    from src.volatility_metrics import calculate_dynamic_sl_tp

    sl_dist, tp_ratios = calculate_dynamic_sl_tp(
        pair="BTCUSDT",
        regime="TRENDING_UP",
        atr_pct=0.35,
        hit_rate=0.62,
    )
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ATR-percentile brackets and their SL/TP multipliers
# ---------------------------------------------------------------------------
# High-volatility environment: widen SL so noise doesn't stop us out;
# stretch TP targets to capture the larger expected move.
_HIGH_VOL_ATR_PCT_THRESHOLD: float = 1.5   # ATR% above this → high-vol
_HIGH_VOL_SL_MULT: float = 1.30
_HIGH_VOL_TP_MULT: float = 1.25

# Low-volatility environment: tighter SL preserves capital on failed setups;
# compressed TPs avoid indefinite capital lockup on unreached targets.
_LOW_VOL_ATR_PCT_THRESHOLD: float = 0.25   # ATR% below this → low-vol
_LOW_VOL_SL_MULT: float = 0.85
_LOW_VOL_TP_MULT: float = 0.80

# Regime adjustments
_REGIME_SL_ADJ: dict = {
    "TRENDING_UP":   1.00,
    "TRENDING_DOWN": 1.00,
    "RANGING":       0.90,   # Tighter SL in range — less room for error
    "QUIET":         0.85,
    "VOLATILE":      1.40,   # Widen significantly in volatile conditions
}

# Hit-rate driven TP boost: pairs with a strong historical record get a
# slight TP extension (encourages holding winners longer).
_HIT_RATE_BOOST_THRESHOLD: float = 0.65  # > 65 % hit rate → boost TPs
_HIT_RATE_BOOST_MULT: float = 1.10       # 10 % TP extension

# Default base TP ratios (R-multiples): [TP1, TP2, TP3]
_DEFAULT_TP_RATIOS: List[float] = [0.5, 1.0, 1.5]


def compute_atr_pct(atr_value: float, price: float) -> float:
    """Return ATR expressed as a percentage of the current price.

    Parameters
    ----------
    atr_value:
        Raw ATR value (same units as *price*).
    price:
        Current mid-price.

    Returns
    -------
    float
        ATR percentage (e.g. ``0.35`` for 0.35 %).
    """
    if price <= 0:
        return 0.0
    return (atr_value / price) * 100.0


def calculate_dynamic_sl_tp(
    pair: str,
    regime: str,
    atr_pct: float,
    hit_rate: float = 0.5,
    base_tp_ratios: Optional[List[float]] = None,
    base_sl_mult: float = 1.0,
    pair_tier: str = "MIDCAP",
) -> Tuple[float, List[float]]:
    """Return (sl_multiplier, tp_ratios) adapted for volatility, regime and pair tier.

    The returned *sl_multiplier* should be applied to the raw ATR-based SL
    distance computed by the channel evaluator.  The *tp_ratios* list replaces
    the channel-config static ratios for TP1/TP2/TP3 calculation.

    Parameters
    ----------
    pair:
        Symbol string (informational only; used in log messages).
    regime:
        Market regime string (e.g. ``"TRENDING_UP"``).
    atr_pct:
        ATR as a percentage of the current price.
    hit_rate:
        Historical signal hit rate for this pair (0.0–1.0, default 0.5).
    base_tp_ratios:
        Starting TP ratios before adjustment.  Defaults to ``[0.5, 1.0, 1.5]``.
    base_sl_mult:
        Base SL multiplier before adjustment (e.g. from channel config).
    pair_tier:
        ``"MAJOR"``, ``"MIDCAP"``, or ``"ALTCOIN"``.  ALTCOINs receive wider
        SLs to absorb manipulation wicks.

    Returns
    -------
    (sl_multiplier, tp_ratios)
    """
    if base_tp_ratios is None:
        base_tp_ratios = list(_DEFAULT_TP_RATIOS)

    # --- Volatility-based adjustment ---
    if atr_pct >= _HIGH_VOL_ATR_PCT_THRESHOLD:
        vol_sl = _HIGH_VOL_SL_MULT
        vol_tp = _HIGH_VOL_TP_MULT
    elif atr_pct <= _LOW_VOL_ATR_PCT_THRESHOLD:
        vol_sl = _LOW_VOL_SL_MULT
        vol_tp = _LOW_VOL_TP_MULT
    else:
        vol_sl = 1.0
        vol_tp = 1.0

    # --- Regime-based SL adjustment ---
    regime_sl = _REGIME_SL_ADJ.get(regime.upper() if regime else "", 1.0)

    # --- Pair-tier SL widening ---
    tier_sl = {"MAJOR": 0.95, "MIDCAP": 1.00, "ALTCOIN": 1.20}.get(pair_tier, 1.0)

    # --- Combine SL adjustments ---
    final_sl_mult = base_sl_mult * vol_sl * regime_sl * tier_sl

    # --- TP ratios: apply vol multiplier and optional hit-rate boost ---
    adj_tp = [r * vol_tp for r in base_tp_ratios]
    if hit_rate > _HIT_RATE_BOOST_THRESHOLD:
        adj_tp = [r * _HIT_RATE_BOOST_MULT for r in adj_tp]

    log.debug(
        "Dynamic SL/TP for %s: regime=%s atr_pct=%.3f%% hit_rate=%.2f "
        "→ sl_mult=%.3f tp=%s",
        pair, regime, atr_pct, hit_rate, final_sl_mult, adj_tp,
    )
    return final_sl_mult, adj_tp


def classify_volatility(atr_pct: float) -> str:
    """Return a human-readable volatility label for the given ATR%.

    Parameters
    ----------
    atr_pct:
        ATR as a percentage of the current price.

    Returns
    -------
    str
        One of ``"HIGH"``, ``"NORMAL"``, or ``"LOW"``.
    """
    if atr_pct >= _HIGH_VOL_ATR_PCT_THRESHOLD:
        return "HIGH"
    if atr_pct <= _LOW_VOL_ATR_PCT_THRESHOLD:
        return "LOW"
    return "NORMAL"
