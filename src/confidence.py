"""Multi-layer confidence scoring engine (0–100).

Factors:
  * SMC signal strength
  * Trend / EMA alignment
  * AI sentiment score
  * Spread & liquidity quality
  * Historical data sufficiency
  * Multi-exchange verification
  * Correlation / position lock
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from config import NEW_PAIR_MIN_CONFIDENCE


@dataclass
class ConfidenceInput:
    """All inputs the scorer needs for one signal evaluation."""
    smc_score: float = 0.0          # 0-25
    trend_score: float = 0.0        # 0-20
    ai_sentiment_score: float = 0.0  # 0-15
    liquidity_score: float = 0.0     # 0-15
    spread_score: float = 0.0        # 0-10
    data_sufficiency: float = 0.0    # 0-10
    multi_exchange: float = 0.0      # 0-5
    has_enough_history: bool = True
    opposing_position_open: bool = False


@dataclass
class ConfidenceResult:
    """Output of the confidence engine."""
    total: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    capped: bool = False
    blocked: bool = False
    reason: str = ""


def score_smc(has_sweep: bool, has_mss: bool, has_fvg: bool) -> float:
    """SMC component (max 25)."""
    s = 0.0
    if has_sweep:
        s += 12.0
    if has_mss:
        s += 9.0
    if has_fvg:
        s += 4.0
    return min(s, 25.0)


def score_trend(ema_aligned: bool, adx_ok: bool, momentum_positive: bool) -> float:
    """Trend component (max 20)."""
    s = 0.0
    if ema_aligned:
        s += 8.0
    if adx_ok:
        s += 7.0
    if momentum_positive:
        s += 5.0
    return min(s, 20.0)


def score_ai_sentiment(sentiment_value: float) -> float:
    """AI sentiment component (max 15).

    *sentiment_value* expected in [-1, 1].
    """
    normalised = (sentiment_value + 1.0) / 2.0  # map to [0, 1]
    return round(min(max(normalised * 15.0, 0.0), 15.0), 2)


def score_liquidity(volume_24h_usd: float, threshold: float = 5_000_000) -> float:
    """Liquidity component (max 15)."""
    if volume_24h_usd <= 0:
        return 0.0
    ratio = min(volume_24h_usd / threshold, 1.0)
    return round(ratio * 15.0, 2)


def score_spread(spread_pct: float, max_spread: float = 0.02) -> float:
    """Spread component (max 10) – lower is better."""
    if spread_pct <= 0:
        return 10.0
    if spread_pct >= max_spread:
        return 0.0
    return round((1.0 - spread_pct / max_spread) * 10.0, 2)


def score_data_sufficiency(candle_count: int, minimum: int = 500) -> float:
    """Data-sufficiency component (max 10)."""
    if candle_count >= minimum:
        return 10.0
    return round((candle_count / minimum) * 10.0, 2)


def score_multi_exchange(verified: bool) -> float:
    """Multi-exchange verification bonus (max 5)."""
    return 5.0 if verified else 0.0


def compute_confidence(inp: ConfidenceInput) -> ConfidenceResult:
    """Combine all sub-scores into the final 0–100 confidence.

    Applies caps for new pairs and blocks opposing-position signals.
    """
    breakdown: Dict[str, float] = {
        "smc": inp.smc_score,
        "trend": inp.trend_score,
        "ai_sentiment": inp.ai_sentiment_score,
        "liquidity": inp.liquidity_score,
        "spread": inp.spread_score,
        "data_sufficiency": inp.data_sufficiency,
        "multi_exchange": inp.multi_exchange,
    }
    total = sum(breakdown.values())
    total = round(min(max(total, 0.0), 100.0), 2)

    capped = False
    if not inp.has_enough_history:
        cap = NEW_PAIR_MIN_CONFIDENCE
        if total > cap:
            total = cap
            capped = True

    blocked = inp.opposing_position_open
    reason = ""
    if blocked:
        reason = "Correlation lock: opposing position already open"

    return ConfidenceResult(
        total=total,
        breakdown=breakdown,
        capped=capped,
        blocked=blocked,
        reason=reason,
    )
