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

# Absolute funding rate (decimal) beyond which positioning is considered extreme.
# Binance funding is typically ±0.01%–0.1%; ≥1% (0.01) is extreme and signals
# contrarian opportunity when aligned with the signal direction.
_EXTREME_FUNDING_RATE: float = 0.01

# Per-channel liquidity thresholds (USD 24h volume).
# SCALP needs $5M+ (tight execution), SWING $10M+ (sustained trend),
# SPOT $1M+ (macro entry), GEM only $250K (micro-cap discovery).
_LIQUIDITY_THRESHOLDS: Dict[str, float] = {
    "360_SCALP":      5_000_000.0,
    "360_SCALP_FVG":  5_000_000.0,
    "360_SCALP_CVD":  5_000_000.0,
    "360_SCALP_VWAP": 5_000_000.0,
    "360_SCALP_OBI":  5_000_000.0,
    "360_SWING":      10_000_000.0,
    "360_SPOT":       1_000_000.0,
    "360_GEM":        250_000.0,
}

# Channel-specific sub-score weight profiles.  Keys match the 8 breakdown
# sub-scores; missing keys default to 1.0 (no scaling).  Scalp channels
# are intentionally flat (1.0 everywhere) so they raw-sum identically to
# the pre-weight behaviour.  SWING, SPOT and GEM profiles tilt weights
# toward the factors that matter most for each investment horizon.
_CHANNEL_WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "360_SCALP": {
        "smc": 1.0, "trend": 1.0, "liquidity": 1.0, "spread": 1.0,
        "data_sufficiency": 1.0, "multi_exchange": 1.0, "onchain": 1.0, "order_flow": 1.0,
    },
    "360_SCALP_FVG": {
        "smc": 1.0, "trend": 1.0, "liquidity": 1.0, "spread": 1.0,
        "data_sufficiency": 1.0, "multi_exchange": 1.0, "onchain": 1.0, "order_flow": 1.0,
    },
    "360_SCALP_CVD": {
        "smc": 1.0, "trend": 1.0, "liquidity": 1.0, "spread": 1.0,
        "data_sufficiency": 1.0, "multi_exchange": 1.0, "onchain": 1.0, "order_flow": 1.0,
    },
    "360_SCALP_VWAP": {
        "smc": 1.0, "trend": 1.0, "liquidity": 1.0, "spread": 1.0,
        "data_sufficiency": 1.0, "multi_exchange": 1.0, "onchain": 1.0, "order_flow": 1.0,
    },
    "360_SCALP_OBI": {
        "smc": 1.0, "trend": 1.0, "liquidity": 1.0, "spread": 1.0,
        "data_sufficiency": 1.0, "multi_exchange": 1.0, "onchain": 1.0, "order_flow": 1.0,
    },
    "360_SWING": {
        "smc": 0.7, "trend": 1.4, "liquidity": 1.0, "spread": 0.8,
        "data_sufficiency": 1.0, "multi_exchange": 1.0, "onchain": 1.2, "order_flow": 0.9,
    },
    "360_SPOT": {
        "smc": 0.5, "trend": 1.4, "liquidity": 0.75, "spread": 0.8,
        "data_sufficiency": 1.0, "multi_exchange": 1.0, "onchain": 1.5, "order_flow": 0.8,
    },
    "360_GEM": {
        "smc": 0.2, "trend": 0.8, "liquidity": 0.5, "spread": 0.5,
        "data_sufficiency": 1.0, "multi_exchange": 0.5, "onchain": 2.0, "order_flow": 0.5,
    },
}


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
    order_flow_score: float = 0.0   # 0-20 (OI squeeze + CVD divergence + funding rate bonus)
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


def score_liquidity(volume_24h_usd: float, threshold: float = 5_000_000, channel: Optional[str] = None) -> float:
    """Liquidity component (max 20).

    Parameters
    ----------
    volume_24h_usd:
        24-hour USD trading volume.
    threshold:
        Default volume threshold.  Overridden per channel when *channel* is provided.
    channel:
        Optional channel name used to select the appropriate liquidity threshold.
        SCALP channels require $5M+, SWING $10M+, SPOT $1M+, GEM only $250K.
    """
    if channel and channel in _LIQUIDITY_THRESHOLDS:
        threshold = _LIQUIDITY_THRESHOLDS[channel]
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
    signal_direction: Optional[str] = None,
    funding_rate: Optional[float] = None,
) -> float:
    """Order-flow component (max 20).

    Rewards institutional-grade squeeze confirmation (falling OI + liquidations),
    CVD divergence signals aligned with the trade direction, and contrarian
    funding-rate alignment (crowd paying against the signal direction).

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
    signal_direction:
        ``"LONG"``, ``"SHORT"``, or ``None``.  When provided, CVD alignment is
        checked: aligned divergence (LONG+BULLISH or SHORT+BEARISH) earns +5
        while a contra divergence (LONG+BEARISH or SHORT+BULLISH) applies a −3
        penalty (total floored at 0).  When ``None`` (backward-compat / no
        direction context), CVD divergence contributes 0 points.
    funding_rate:
        Optional latest funding rate (decimal, e.g. 0.0001 for 0.01%).
        When |funding_rate| ≥ 1% and aligns contrarily with signal direction
        (extreme negative funding + LONG, or extreme positive funding + SHORT),
        a bonus of up to 5 pts is added.

    Returns
    -------
    float
        0–20 score representing order-flow confirmation quality.
        * Squeeze confirmed (OI falling + liquidations) → up to 10.
        * CVD divergence aligned with signal direction → +5.
        * CVD divergence contra to signal direction → −3 (floored at 0).
        * Contrarian funding rate alignment → up to +5.
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

    # CVD divergence component: requires signal_direction to score
    if cvd_divergence is not None and signal_direction is not None:
        aligned = (
            (signal_direction == "LONG" and cvd_divergence == "BULLISH")
            or (signal_direction == "SHORT" and cvd_divergence == "BEARISH")
        )
        if aligned:
            s += 5.0
        else:
            s -= 3.0

    # Funding rate alignment bonus (0–5): contrarian edge when crowd is wrong
    if funding_rate is not None and signal_direction is not None:
        if abs(funding_rate) >= _EXTREME_FUNDING_RATE:
            contrarian = (
                (signal_direction == "LONG" and funding_rate < -_EXTREME_FUNDING_RATE)
                or (signal_direction == "SHORT" and funding_rate > _EXTREME_FUNDING_RATE)
            )
            if contrarian:
                funding_bonus = min(abs(funding_rate) / 0.03, 1.0) * 5.0
                s += funding_bonus

    return min(max(s, 0.0), 20.0)


def get_session_multiplier(now: Optional[datetime] = None, channel: Optional[str] = None) -> float:
    """Return a confidence multiplier based on the current trading session.

    Crypto markets have different volatility and liquidity profiles across
    the three main sessions (UTC):

    * **Asian session** (00:00–08:00 UTC): lower volume, more false breakouts → 0.9×
    * **European session** (08:00–16:00 UTC): moderate volume, cleaner moves → 1.0×
    * **US session** (16:00–00:00 UTC): highest volume, strongest trends → 1.05×

    Higher-timeframe channels (SPOT, GEM) operate on 4h/1d/1w candles where
    intraday session is irrelevant → always 1.0×.  SWING channels see a reduced
    session impact (half penalty/boost).

    Parameters
    ----------
    now:
        Optional UTC datetime for testing.  Defaults to the current UTC time.
    channel:
        Optional channel name.  When ``"360_SPOT"`` or ``"360_GEM"``, the
        session multiplier is always 1.0 (session is irrelevant at 4h/1d/1w).
        When ``"360_SWING"``, a reduced impact is applied.

    Returns
    -------
    float
        Multiplier to apply to the raw confidence total before capping.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Higher-timeframe channels: session is irrelevant → always 1.0
    if channel in ("360_SPOT", "360_GEM"):
        return 1.0

    hour = now.hour  # UTC hour 0–23

    # SWING: reduced session impact (half penalty/boost)
    if channel == "360_SWING":
        if 0 <= hour < 8:
            return 0.95   # Asian: mild penalty
        if 8 <= hour < 16:
            return 1.0    # European session
        return 1.02       # US: mild boost

    # SCALP channels (and unknown channels): full session impact
    if 0 <= hour < 8:
        return 0.90   # Asian session
    if 8 <= hour < 16:
        return 1.0   # European session
    return 1.05      # US session (16–24)


def compute_confidence(
    inp: ConfidenceInput,
    session_now: Optional[datetime] = None,
    channel: Optional[str] = None,
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
    channel:
        Optional channel name passed through to :func:`get_session_multiplier`
        to apply channel-appropriate session weighting.
    """
    weights = _CHANNEL_WEIGHT_PROFILES.get(channel or "", {})
    breakdown: Dict[str, float] = {
        "smc": inp.smc_score * weights.get("smc", 1.0),
        "trend": inp.trend_score * weights.get("trend", 1.0),
        "liquidity": inp.liquidity_score * weights.get("liquidity", 1.0),
        "spread": inp.spread_score * weights.get("spread", 1.0),
        "data_sufficiency": inp.data_sufficiency * weights.get("data_sufficiency", 1.0),
        "multi_exchange": inp.multi_exchange * weights.get("multi_exchange", 1.0),
        "onchain": inp.onchain_score * weights.get("onchain", 1.0),
        "order_flow": inp.order_flow_score * weights.get("order_flow", 1.0),
    }
    total = sum(breakdown.values())

    # Apply session multiplier before capping
    session_mult = get_session_multiplier(session_now, channel=channel)
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
