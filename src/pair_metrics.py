"""Pair-level metrics and scoring for high-probability signal filtering.

Provides composite scoring for each pair based on spread, volume, volatility,
historical hit rate, and liquidity — used by the filter module to determine
signal probability scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.utils import get_logger

log = get_logger("pair_metrics")


@dataclass
class PairMetrics:
    """Aggregated quality metrics for a single trading pair."""

    spread_pct: float = 0.0
    volume_24h_usd: float = 0.0
    hit_rate: float = 0.5           # Historical signal hit rate (0-1)
    atr_percentile: float = 50.0    # ATR percentile (0-100)
    liquidity_score: float = 50.0   # Order book liquidity (0-100)
    adx_value: float = 25.0         # ADX for trend strength (reserved for future scoring)


def score_spread(spread_pct: float, max_spread: float = 0.02) -> float:
    """Score spread quality (0-100). Tighter spread = higher score."""
    if spread_pct <= 0 or max_spread <= 0:
        return 100.0
    ratio = spread_pct / max_spread
    if ratio >= 1.0:
        return 0.0
    return round(max(0.0, min(100.0, (1.0 - ratio) * 100.0)), 2)


def score_volume(volume_24h_usd: float, min_volume: float = 5_000_000.0) -> float:
    """Score volume quality (0-100). Higher volume = higher score."""
    if volume_24h_usd <= 0:
        return 0.0
    if volume_24h_usd >= min_volume * 5:
        return 100.0
    ratio = volume_24h_usd / min_volume
    if ratio < 1.0:
        return round(max(0.0, ratio * 50.0), 2)
    return round(min(100.0, 50.0 + (ratio - 1.0) / 4.0 * 50.0), 2)


def score_hit_rate(hit_rate: float) -> float:
    """Score historical hit rate (0-100). Higher hit rate = higher score."""
    return round(max(0.0, min(100.0, hit_rate * 100.0)), 2)


def score_volatility(atr_percentile: float) -> float:
    """Score volatility (0-100). Moderate volatility scores highest.

    Extremely low (<15th pct) or extremely high (>85th pct) ATR
    indicates unfavourable scalping conditions.
    """
    if atr_percentile <= 15.0:
        return round(atr_percentile / 15.0 * 50.0, 2)
    if atr_percentile >= 85.0:
        return round(max(0.0, (100.0 - atr_percentile) / 15.0 * 50.0), 2)
    # Sweet spot: 30-70th percentile gets 80-100 score
    if 30.0 <= atr_percentile <= 70.0:
        return round(80.0 + (1.0 - abs(atr_percentile - 50.0) / 20.0) * 20.0, 2)
    # Transition zones
    if atr_percentile < 30.0:
        return round(50.0 + (atr_percentile - 15.0) / 15.0 * 30.0, 2)
    return round(50.0 + (85.0 - atr_percentile) / 15.0 * 30.0, 2)


def score_liquidity(liquidity_score: float) -> float:
    """Normalise liquidity score to 0-100 range."""
    return round(max(0.0, min(100.0, liquidity_score)), 2)


def compute_pair_score(
    metrics: PairMetrics,
    max_spread: float = 0.02,
    min_volume: float = 5_000_000.0,
) -> float:
    """Compute a weighted composite pair quality score (0-100).

    Weights
    -------
    - Spread:     20 %
    - Volume:     25 %
    - Hit Rate:   25 %
    - Volatility: 15 %
    - Liquidity:  15 %
    """
    s_spread = score_spread(metrics.spread_pct, max_spread)
    s_volume = score_volume(metrics.volume_24h_usd, min_volume)
    s_hit = score_hit_rate(metrics.hit_rate)
    s_vol = score_volatility(metrics.atr_percentile)
    s_liq = score_liquidity(metrics.liquidity_score)

    composite = (
        s_spread * 0.20
        + s_volume * 0.25
        + s_hit * 0.25
        + s_vol * 0.15
        + s_liq * 0.15
    )
    return round(max(0.0, min(100.0, composite)), 2)
