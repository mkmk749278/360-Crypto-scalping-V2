"""Multi-layer confidence scoring engine (0–100).

Factors:
  * SMC signal strength       (0–30)
  * Trend / EMA alignment     (0–25)
  * Liquidity quality         (0–20)
  * Spread quality            (0–10)
  * Historical data sufficiency (0–10)
  * Multi-exchange verification (0–5)
  * Correlation / position lock
  * Trading-session multiplier (Asian / EU / US)

AI sentiment is intentionally excluded from signal scoring so that
high-frequency signals fire with zero external-network latency.
Macro/news AI alerts are handled separately by the MacroWatchdog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from config import NEW_PAIR_MIN_CONFIDENCE

# USD liquidation volume at which the order-flow liq bonus is maximised (5 pts).
_ORDER_FLOW_LIQ_CAP_USD: float = 500_000.0


@dataclass
class ConfidenceInput:
    """All inputs the scorer needs for one signal evaluation."""
    smc_score: float = 0.0          # 0-30
    trend_score: float = 0.0        # 0-25
    liquidity_score: float = 0.0    # 0-20
    spread_score: float = 0.0       # 0-10
    data_sufficiency: float = 0.0   # 0-10
    multi_exchange: float = 0.0     # 0-5
    onchain_score: float = 0.0      # 0-5 (populated by score_onchain(); 0 = no data)
    order_flow_score: float = 0.0   # 0-15 (OI squeeze + CVD divergence bonus)
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


def score_smc(
    has_sweep: bool,
    has_mss: bool,
    has_fvg: bool,
    sweep_depth_pct: float = 0.0,
    fvg_atr_ratio: float = 0.0,
) -> float:
    """SMC component (max 30).

    Parameters
    ----------
    has_sweep:
        Whether a liquidity sweep was detected.
    has_mss:
        Whether a market structure shift was detected.
    has_fvg:
        Whether a fair value gap was detected.
    sweep_depth_pct:
        How deep the sweep went past the level, as a percentage of price.
        Deeper sweeps are more significant.  Clipped to [0, 1] for scoring.
    fvg_atr_ratio:
        Size of the FVG gap relative to ATR.
        Larger gaps are more significant.  Clipped to [0, 2] for scoring.
    """
    s = 0.0
    if has_sweep:
        # Base 10 + up to 5 for depth (deeper sweep = stronger signal)
        depth_bonus = min(sweep_depth_pct / 0.5, 1.0) * 5.0  # max at 0.5%
        s += 10.0 + depth_bonus
    if has_mss:
        s += 11.0
    if has_fvg:
        # Base 2 + up to 2 for size (larger FVG = more significant)
        size_bonus = min(fvg_atr_ratio / 1.5, 1.0) * 2.0  # max at 1.5×ATR
        s += 2.0 + size_bonus
    return min(s, 30.0)


def score_trend(
    ema_aligned: bool,
    adx_ok: bool,
    momentum_positive: bool,
    adx_value: float = 0.0,
    momentum_strength: float = 0.0,
) -> float:
    """Trend component (max 25).

    Parameters
    ----------
    ema_aligned:
        Whether EMA9 > EMA21 (LONG) or EMA9 < EMA21 (SHORT).
    adx_ok:
        Whether ADX >= 20 (trending).
    momentum_positive:
        Whether momentum is in the signal direction.
    adx_value:
        Actual ADX value for gradient scoring.
        ADX 20-25 = minimal trend, ADX 40+ = strong trend.
    momentum_strength:
        Absolute momentum value for gradient scoring.
    """
    s = 0.0
    if ema_aligned:
        s += 10.0
    if adx_ok:
        # Base 4 + up to 5 based on ADX strength (20→4, 40+→9)
        adx_bonus = min(max(adx_value - 20.0, 0.0) / 20.0, 1.0) * 5.0
        s += 4.0 + adx_bonus
    if momentum_positive:
        # Base 2 + up to 4 based on momentum strength
        mom_bonus = min(abs(momentum_strength) / 1.0, 1.0) * 4.0
        s += 2.0 + mom_bonus
    return min(s, 25.0)


def score_liquidity(volume_24h_usd: float, threshold: float = 5_000_000) -> float:
    """Liquidity component (max 20)."""
    if volume_24h_usd <= 0:
        return 0.0
    ratio = min(volume_24h_usd / threshold, 1.0)
    return round(ratio * 20.0, 2)


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


def score_multi_exchange(verified: Optional[bool] = None) -> float:
    """Multi-exchange verification bonus (max 5).

    Parameters
    ----------
    verified:
        ``True``  – second exchange confirms the signal → 5.0.
        ``False`` – second exchange contradicts the signal → 0.0.
        ``None``  – no second exchange configured (neutral) → 2.5.
    """
    if verified is True:
        return 5.0
    if verified is False:
        return 0.0
    return 2.5  # None → neutral


def score_order_flow(
    oi_trend: str = "NEUTRAL",
    liq_vol_usd: float = 0.0,
    cvd_divergence: Optional[str] = None,
) -> float:
    """Order-flow component (max 15).

    Rewards institutional-grade squeeze confirmation (falling OI + liquidations)
    and CVD divergence signals.

    Parameters
    ----------
    oi_trend:
        One of ``"RISING"``, ``"FALLING"``, or ``"NEUTRAL"`` (as returned by
        :func:`src.order_flow.classify_oi_trend`).
    liq_vol_usd:
        Total USD liquidation volume for this symbol in the recent window
        (as returned by :meth:`src.order_flow.OrderFlowStore.get_recent_liq_volume_usd`).
    cvd_divergence:
        ``"BULLISH"``, ``"BEARISH"``, or ``None`` (as returned by
        :func:`src.order_flow.detect_cvd_divergence`).

    Returns
    -------
    float
        0–15 score representing order-flow confirmation quality.
        * Squeeze confirmed (OI falling + liquidations) → up to 10.
        * CVD divergence aligned with signal → +5.
    """
    s = 0.0

    # Squeeze component: falling OI + liquidation activity (0–10)
    if oi_trend == "FALLING":
        # Base squeeze bonus: OI is declining (positions closing / exhaustion)
        s += 5.0
        if liq_vol_usd > 0:
            # Additional bonus for confirmed liquidation activity
            # Scales with USD volume, capped at 5 extra points
            liq_bonus = min(liq_vol_usd / _ORDER_FLOW_LIQ_CAP_USD, 1.0) * 5.0
            s += liq_bonus

    # CVD divergence component (+5 when present)
    if cvd_divergence is not None:
        s += 5.0

    return min(s, 15.0)


def get_session_multiplier(now: Optional[datetime] = None) -> float:
    """Return a confidence multiplier based on the current trading session.

    Crypto markets have different volatility and liquidity profiles across
    the three main sessions (UTC):

    * **Asian session** (00:00–08:00 UTC): lower volume, more false breakouts → 0.9×
    * **European session** (08:00–16:00 UTC): moderate volume, cleaner moves → 1.0×
    * **US session** (16:00–00:00 UTC): highest volume, strongest trends → 1.05×

    Parameters
    ----------
    now:
        Optional UTC datetime for testing.  Defaults to the current UTC time.

    Returns
    -------
    float
        Multiplier to apply to the raw confidence total before capping.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    hour = now.hour  # UTC hour 0–23
    if 0 <= hour < 8:
        return 0.9   # Asian session
    if 8 <= hour < 16:
        return 1.0   # European session
    return 1.05      # US session (16–24)


def compute_confidence(
    inp: ConfidenceInput,
    session_now: Optional[datetime] = None,
) -> ConfidenceResult:
    """Combine all sub-scores into the final 0–100 confidence.

    Applies a trading-session multiplier after summing sub-scores, then caps
    new pairs and blocks opposing-position signals.

    Parameters
    ----------
    inp:
        All sub-score inputs.
    session_now:
        Optional UTC datetime used to determine the active trading session.
        Defaults to the current UTC time.  Pass an explicit value in tests to
        avoid time-dependent results.
    """
    breakdown: Dict[str, float] = {
        "smc": inp.smc_score,
        "trend": inp.trend_score,
        "liquidity": inp.liquidity_score,
        "spread": inp.spread_score,
        "data_sufficiency": inp.data_sufficiency,
        "multi_exchange": inp.multi_exchange,
        "onchain": inp.onchain_score,
        "order_flow": inp.order_flow_score,
    }
    total = sum(breakdown.values())

    # Apply session multiplier before capping
    session_mult = get_session_multiplier(session_now)
    total = total * session_mult

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
