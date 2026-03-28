"""High-Probability Filter Module — adaptive scoring per pair/channel.

Scores each pair based on market regime, spread, liquidity, historical
hit rate, and volatility.  Only signals with a probability score above
a configurable threshold are allowed through.

PR 01 Implementation.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.pair_metrics import PairMetrics, compute_pair_score
from src.utils import get_logger

log = get_logger("filter_module")

# Default probability threshold — signals below this are suppressed.
DEFAULT_PROBABILITY_THRESHOLD: float = float(
    os.getenv("FILTER_PROBABILITY_THRESHOLD", "70.0")
)

# Per-channel threshold overrides (env-configurable).
CHANNEL_THRESHOLDS: Dict[str, float] = {
    "360_SCALP": float(os.getenv("FILTER_THRESHOLD_SCALP", "70.0")),
    "360_SCALP_FVG": float(os.getenv("FILTER_THRESHOLD_SCALP_FVG", "65.0")),
    "360_SCALP_CVD": float(os.getenv("FILTER_THRESHOLD_SCALP_CVD", "65.0")),
    "360_SCALP_VWAP": float(os.getenv("FILTER_THRESHOLD_SCALP_VWAP", "68.0")),
    "360_SCALP_OBI": float(os.getenv("FILTER_THRESHOLD_SCALP_OBI", "68.0")),
    "360_SWING": float(os.getenv("FILTER_THRESHOLD_SWING", "60.0")),
    "360_SPOT": float(os.getenv("FILTER_THRESHOLD_SPOT", "55.0")),
    "360_GEM": float(os.getenv("FILTER_THRESHOLD_GEM", "50.0")),
}

# Regime-based threshold adjustments.
_REGIME_THRESHOLD_ADJUSTMENT: Dict[str, float] = {
    "TRENDING_UP": -5.0,    # Trending markets: lower threshold (easier to trade)
    "TRENDING_DOWN": -5.0,
    "RANGING": 0.0,
    "VOLATILE": 5.0,        # Volatile: raise threshold (harder conditions)
    "QUIET": 10.0,          # Quiet: significantly raise threshold
}


def get_threshold_for_channel(channel: str, regime: str = "") -> float:
    """Return the effective probability threshold for a channel + regime.

    Parameters
    ----------
    channel:
        Channel name (e.g. ``"360_SCALP"``).
    regime:
        Current market regime string.

    Returns
    -------
    float
        Adjusted threshold (0-100).
    """
    base = CHANNEL_THRESHOLDS.get(channel, DEFAULT_PROBABILITY_THRESHOLD)
    adjustment = (
        _REGIME_THRESHOLD_ADJUSTMENT.get(regime.upper(), 0.0) if regime else 0.0
    )
    return max(0.0, min(100.0, base + adjustment))


def get_pair_probability(
    pair_data: Dict[str, Any],
    channel: str = "",
    regime: str = "",
) -> float:
    """Compute the probability score (0-100) for a pair.

    Parameters
    ----------
    pair_data:
        Dictionary with keys: ``spread_pct``, ``volume_24h_usd``,
        ``hit_rate``, ``atr_percentile``, ``liquidity_score``,
        ``max_spread``, ``min_volume``.
    channel:
        Channel name for threshold lookup.
    regime:
        Current market regime for adaptive scoring.

    Returns
    -------
    float
        Probability score 0-100.
    """
    metrics = PairMetrics(
        spread_pct=pair_data.get("spread_pct", 0.0),
        volume_24h_usd=pair_data.get("volume_24h_usd", 0.0),
        hit_rate=pair_data.get("hit_rate", 0.5),
        atr_percentile=pair_data.get("atr_percentile", 50.0),
        liquidity_score=pair_data.get("liquidity_score", 50.0),
    )
    max_spread = pair_data.get("max_spread", 0.02)
    min_volume = pair_data.get("min_volume", 5_000_000.0)
    score = compute_pair_score(metrics, max_spread=max_spread, min_volume=min_volume)

    # Regime-based score adjustment
    regime_upper = regime.upper() if regime else ""
    if regime_upper in ("TRENDING_UP", "TRENDING_DOWN"):
        score = min(100.0, score * 1.05)  # Small boost for trending
    elif regime_upper == "QUIET":
        score *= 0.90  # Penalise quiet markets
    elif regime_upper == "VOLATILE":
        score *= 0.95  # Slight penalty for volatility

    return round(max(0.0, min(100.0, score)), 2)


def check_pair_probability(
    pair_data: Dict[str, Any],
    channel: str = "",
    regime: str = "",
) -> tuple[bool, float]:
    """Check whether a pair passes the probability threshold.

    Returns
    -------
    tuple[bool, float]
        ``(passed, probability_score)`` — ``passed`` is True when the
        score meets or exceeds the channel/regime threshold.
    """
    score = get_pair_probability(pair_data, channel=channel, regime=regime)
    threshold = get_threshold_for_channel(channel, regime=regime)
    passed = score >= threshold

    if not passed:
        log.debug(
            "Pair probability filter suppressed: channel={} score={:.1f} threshold={:.1f} regime={}",
            channel, score, threshold, regime,
        )

    return passed, score
