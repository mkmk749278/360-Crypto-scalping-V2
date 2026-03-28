"""Volatility metrics helpers for dynamic SL/TP calculation.

Provides regime-aware volatility analysis used by the dynamic SL/TP
system to adapt stop-loss and take-profit levels based on current
market conditions.

PR 02 Implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.utils import get_logger

log = get_logger("volatility_metrics")


@dataclass
class VolatilityProfile:
    """Volatility analysis result for a single pair."""

    atr_current: float = 0.0
    atr_percentile: float = 50.0       # 0-100 rolling percentile
    bb_width_pct: float = 2.0          # Bollinger Band width as pct
    regime: str = ""                    # Market regime string
    pair_tier: str = "MIDCAP"          # MAJOR / MIDCAP / ALTCOIN
    historical_hit_rate: float = 0.5   # 0-1 historical win rate


# SL multipliers by regime — QUIET gets wider SL (avoid noise stop-outs),
# TRENDING gets tighter SL (clear direction, protect profits faster).
_REGIME_SL_MULTIPLIER = {
    "TRENDING_UP": 0.85,    # Tighter SL in trends
    "TRENDING_DOWN": 0.85,
    "RANGING": 1.0,         # Default
    "VOLATILE": 1.15,       # Slightly wider for volatile
    "QUIET": 1.25,          # Wider SL to survive noise
}

# TP multipliers by regime — TRENDING gets extended TP (ride the trend),
# QUIET gets compressed TP (take profits quickly in low-vol).
_REGIME_TP_MULTIPLIER = {
    "TRENDING_UP": 1.20,    # Extended TP to ride trends
    "TRENDING_DOWN": 1.20,
    "RANGING": 1.0,         # Default
    "VOLATILE": 0.90,       # Compressed TP, take profits quickly
    "QUIET": 0.80,          # Tight TP in quiet markets
}

# Pair tier SL adjustments (majors need tighter, altcoins wider).
_TIER_SL_ADJUSTMENT = {
    "MAJOR": 0.90,
    "MIDCAP": 1.0,
    "ALTCOIN": 1.20,
}


def compute_regime_sl_multiplier(regime: str) -> float:
    """Return the SL width multiplier for the given regime.

    A value >1.0 widens the SL; <1.0 tightens it.
    """
    return _REGIME_SL_MULTIPLIER.get(regime.upper(), 1.0)


def compute_regime_tp_multiplier(regime: str) -> float:
    """Return the TP ratio multiplier for the given regime.

    A value >1.0 extends TP targets; <1.0 compresses them.
    """
    return _REGIME_TP_MULTIPLIER.get(regime.upper(), 1.0)


def compute_volatility_adjusted_sl(
    base_sl_distance: float,
    profile: VolatilityProfile,
) -> float:
    """Compute regime- and volatility-adjusted stop-loss distance.

    Parameters
    ----------
    base_sl_distance:
        Raw SL distance from the channel evaluator.
    profile:
        Volatility analysis for the pair.

    Returns
    -------
    float
        Adjusted SL distance.
    """
    regime_mult = compute_regime_sl_multiplier(profile.regime)
    tier_mult = _TIER_SL_ADJUSTMENT.get(profile.pair_tier, 1.0)

    # ATR percentile adjustment: high ATR → widen SL slightly
    atr_adj = 1.0
    if profile.atr_percentile > 75:
        atr_adj = 1.0 + (profile.atr_percentile - 75) / 100.0  # up to +25%
    elif profile.atr_percentile < 25:
        atr_adj = 1.0 - (25 - profile.atr_percentile) / 200.0  # up to -12.5%

    # Hit rate adjustment: poor hit rate → widen SL for more room
    hr_adj = 1.0
    if profile.historical_hit_rate < 0.4:
        hr_adj = 1.1  # +10% SL width for poor performers
    elif profile.historical_hit_rate > 0.7:
        hr_adj = 0.95  # tighten for consistent winners

    adjusted = base_sl_distance * regime_mult * tier_mult * atr_adj * hr_adj
    log.debug(
        "Dynamic SL: base={:.6f} regime_mult={:.2f} tier={:.2f} "
        "atr_adj={:.2f} hr_adj={:.2f} → {:.6f}",
        base_sl_distance, regime_mult, tier_mult, atr_adj, hr_adj, adjusted,
    )
    return adjusted


def compute_volatility_adjusted_tp_ratios(
    base_ratios: list[float],
    profile: VolatilityProfile,
) -> list[float]:
    """Compute regime- and volatility-adjusted TP ratio multipliers.

    Parameters
    ----------
    base_ratios:
        Base R-multiple ratios (e.g. ``[1.5, 2.5, 4.0]``).
    profile:
        Volatility analysis for the pair.

    Returns
    -------
    list[float]
        Adjusted TP ratios.
    """
    regime_mult = compute_regime_tp_multiplier(profile.regime)

    # High hit rate → stretch TP to capture more; low → compress for safety
    hr_mult = 1.0
    if profile.historical_hit_rate > 0.65:
        hr_mult = 1.05
    elif profile.historical_hit_rate < 0.35:
        hr_mult = 0.90

    adjusted = [round(r * regime_mult * hr_mult, 4) for r in base_ratios]
    log.debug(
        "Dynamic TP ratios: base={} regime_mult={:.2f} hr_mult={:.2f} → {}",
        base_ratios, regime_mult, hr_mult, adjusted,
    )
    return adjusted


def calculate_dynamic_sl_tp(
    pair: str,
    regime: str,
    volatility_pct: float,
    hit_rate: float,
    base_sl_distance: float,
    base_tp_ratios: list[float],
    pair_tier: str = "MIDCAP",
    atr_percentile: float = 50.0,
) -> Tuple[float, list[float]]:
    """Calculate dynamic SL distance and TP ratios.

    Convenience function that wraps the volatility-adjusted computations
    for use by the trade execution pipeline.

    Parameters
    ----------
    pair:
        Symbol name (for logging).
    regime:
        Current market regime string.
    volatility_pct:
        BB width or similar volatility measure (percentage).
    hit_rate:
        Historical hit rate for this pair/channel (0-1).
    base_sl_distance:
        Raw SL distance from channel evaluator.
    base_tp_ratios:
        Base TP R-multiple ratios.
    pair_tier:
        Pair tier classification.
    atr_percentile:
        Current ATR rolling percentile (0-100).

    Returns
    -------
    tuple[float, list[float]]
        ``(adjusted_sl_distance, adjusted_tp_ratios)``
    """
    profile = VolatilityProfile(
        atr_percentile=atr_percentile,
        bb_width_pct=volatility_pct,
        regime=regime,
        pair_tier=pair_tier,
        historical_hit_rate=hit_rate,
    )

    adj_sl = compute_volatility_adjusted_sl(base_sl_distance, profile)
    adj_tp = compute_volatility_adjusted_tp_ratios(base_tp_ratios, profile)

    log.debug(
        "Dynamic SL/TP for {}: regime={} tier={} sl={:.6f} tp={}",
        pair, regime, pair_tier, adj_sl, adj_tp,
    )
    return adj_sl, adj_tp
