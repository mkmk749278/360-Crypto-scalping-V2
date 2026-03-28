"""High-Probability Filter Module (PR01) – adaptive pair probability scoring.

Computes a 0–100 probability score for each pair before signal generation.
Signals are only allowed when the score exceeds a dynamic threshold
(default: 70, adjustable per channel).

Score inputs
------------
* market_regime   – TRENDING/RANGING → higher score; QUIET/VOLATILE → lower.
* spread_pct      – Low spread (< 0.05 %) → high score; high spread → penalised.
* volume_24h_usd  – High liquidity → higher score.
* atr_pct         – Moderate volatility preferred; extreme or near-zero penalised.
* hit_rate        – Historical signal success rate for the pair (0.0–1.0).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Default probability threshold – signals below this score are suppressed.
DEFAULT_PROBABILITY_THRESHOLD: float = 70.0

# Regime weights: multiplier applied to the raw regime sub-score.
_REGIME_SCORE: Dict[str, float] = {
    "TRENDING_UP":   85.0,
    "TRENDING_DOWN": 85.0,
    "RANGING":       75.0,
    "VOLATILE":      55.0,
    "QUIET":         45.0,
}
_REGIME_SCORE_DEFAULT: float = 65.0

# Spread thresholds (percent)
_SPREAD_EXCELLENT: float = 0.02   # ≤ 0.02 % → 100 pts
_SPREAD_GOOD: float = 0.05        # 0.02–0.05 % → 70–100 pts
_SPREAD_FAIR: float = 0.10        # 0.05–0.10 % → 40–70 pts
_SPREAD_POOR: float = 0.20        # > 0.20 % → 0 pts

# Liquidity thresholds (USD)
_VOL_EXCELLENT: float = 100_000_000.0  # ≥ 100 M USD
_VOL_GOOD: float = 10_000_000.0        # ≥ 10 M USD
_VOL_FAIR: float = 1_000_000.0         # ≥ 1 M USD

# ATR% sweet-spot: 0.2–1.5 % is ideal for scalping
_ATR_IDEAL_LOW: float = 0.2
_ATR_IDEAL_HIGH: float = 1.5


def _score_spread(spread_pct: float) -> float:
    """Return a 0–100 score based on bid-ask spread percentage."""
    if spread_pct <= _SPREAD_EXCELLENT:
        return 100.0
    if spread_pct <= _SPREAD_GOOD:
        # Linear interpolation from 100 → 70
        ratio = (spread_pct - _SPREAD_EXCELLENT) / (_SPREAD_GOOD - _SPREAD_EXCELLENT)
        return 100.0 - ratio * 30.0
    if spread_pct <= _SPREAD_FAIR:
        ratio = (spread_pct - _SPREAD_GOOD) / (_SPREAD_FAIR - _SPREAD_GOOD)
        return 70.0 - ratio * 30.0
    if spread_pct <= _SPREAD_POOR:
        ratio = (spread_pct - _SPREAD_FAIR) / (_SPREAD_POOR - _SPREAD_FAIR)
        return 40.0 - ratio * 40.0
    return 0.0


def _score_volume(volume_24h_usd: float) -> float:
    """Return a 0–100 score based on 24-hour USD volume."""
    if volume_24h_usd >= _VOL_EXCELLENT:
        return 100.0
    if volume_24h_usd >= _VOL_GOOD:
        ratio = (volume_24h_usd - _VOL_GOOD) / (_VOL_EXCELLENT - _VOL_GOOD)
        return 70.0 + ratio * 30.0
    if volume_24h_usd >= _VOL_FAIR:
        ratio = (volume_24h_usd - _VOL_FAIR) / (_VOL_GOOD - _VOL_FAIR)
        return 40.0 + ratio * 30.0
    return max(0.0, volume_24h_usd / _VOL_FAIR * 40.0)


def _score_atr(atr_pct: float) -> float:
    """Return a 0–100 score based on ATR as a percentage of price.

    Ideal range for scalping is 0.2–1.5 %.  Extremes are penalised.
    """
    if atr_pct <= 0:
        return 20.0
    if _ATR_IDEAL_LOW <= atr_pct <= _ATR_IDEAL_HIGH:
        return 100.0
    if atr_pct < _ATR_IDEAL_LOW:
        # Below ideal: linear from 20 → 100
        return 20.0 + (atr_pct / _ATR_IDEAL_LOW) * 80.0
    # Above ideal: penalise but keep minimum of 30
    excess = atr_pct - _ATR_IDEAL_HIGH
    penalty = min(excess / 2.0, 1.0) * 70.0
    return max(30.0, 100.0 - penalty)


def _score_regime(regime: str) -> float:
    """Return a 0–100 score for the current market regime."""
    return _REGIME_SCORE.get(regime.upper() if regime else "", _REGIME_SCORE_DEFAULT)


def get_pair_probability(pair_data: Dict[str, Any]) -> float:
    """Compute an adaptive probability score (0–100) for a pair.

    Parameters
    ----------
    pair_data : dict
        Expected keys (all optional with sensible defaults):

        * ``regime``        – Market regime string (e.g. ``"TRENDING_UP"``).
        * ``spread_pct``    – Current bid-ask spread as a percentage.
        * ``volume_24h_usd``– 24-hour notional volume in USD.
        * ``atr_pct``       – ATR as a percentage of the current price.
        * ``hit_rate``      – Historical signal hit rate for this pair (0.0–1.0).

    Returns
    -------
    float
        Probability score in the range [0, 100].
    """
    regime = str(pair_data.get("regime", ""))
    spread_pct = float(pair_data.get("spread_pct", 0.05))
    volume_24h_usd = float(pair_data.get("volume_24h_usd", 0.0))
    atr_pct = float(pair_data.get("atr_pct", 0.5))
    hit_rate = float(pair_data.get("hit_rate", 0.5))

    # Clamp hit_rate to [0, 1]
    hit_rate = max(0.0, min(1.0, hit_rate))

    # Component scores (each 0–100)
    s_regime = _score_regime(regime)
    s_spread = _score_spread(spread_pct)
    s_volume = _score_volume(volume_24h_usd)
    s_atr = _score_atr(atr_pct)
    s_hit_rate = hit_rate * 100.0

    # Weighted average
    # Regime and spread are the most important gates (25% each),
    # followed by volume (20%), ATR (15%), and historical hit rate (15%).
    score = (
        s_regime * 0.25
        + s_spread * 0.25
        + s_volume * 0.20
        + s_atr * 0.15
        + s_hit_rate * 0.15
    )
    score = max(0.0, min(100.0, score))
    log.debug(
        "Pair probability: regime=%.1f spread=%.1f vol=%.1f atr=%.1f "
        "hit_rate=%.1f → score=%.1f",
        s_regime, s_spread, s_volume, s_atr, s_hit_rate, score,
    )
    return score


def is_high_probability(
    pair_data: Dict[str, Any],
    threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
) -> tuple[bool, float]:
    """Return (allowed, score) where *allowed* is True when score ≥ threshold.

    Parameters
    ----------
    pair_data:
        Same dict accepted by :func:`get_pair_probability`.
    threshold:
        Minimum score to allow a signal (default ``70.0``).

    Returns
    -------
    tuple[bool, float]
        ``(True, score)`` when the pair passes, ``(False, score)`` otherwise.
    """
    score = get_pair_probability(pair_data)
    allowed = score >= threshold
    if not allowed:
        log.debug(
            "Signal suppressed by probability filter: score=%.1f < threshold=%.1f "
            "(regime=%s spread=%.4f%% vol=%.0f)",
            score, threshold,
            pair_data.get("regime", ""),
            pair_data.get("spread_pct", 0),
            pair_data.get("volume_24h_usd", 0),
        )
    return allowed, score
