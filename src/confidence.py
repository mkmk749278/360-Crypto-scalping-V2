"""Multi-layer confidence scoring engine (0–100).

Factors:
  * SMC signal strength       (0–30)
  * Trend / EMA alignment     (0–25)
  * Liquidity quality         (0–20)
  * Spread quality            (0–10)
  * Historical data sufficiency (0–10)
  * Multi-exchange verification (0–5)
  * On-chain / whale flow     (0–10, whale-aware)
  * Order flow                (0–20)
  * AI Sentiment              (0–10, SPOT/GEM only)
  * Correlation / position lock
  * Trading-session multiplier (Asian / EU / US)

AI sentiment is active only for SPOT and GEM channels (4h/1d/1w timeframes
where 10 s of network latency is irrelevant).  SCALP and SWING channels
always receive a neutral 5.0 sentiment score so they fire with zero
external-network latency.
Macro/news AI alerts are handled separately by the MacroWatchdog.

Confidence logging
------------------
When ``CONFIDENCE_LOG_ENABLED`` is True, :func:`compute_confidence` appends a
structured JSON record to ``CONFIDENCE_LOG_PATH`` (default:
``data/confidence_log.jsonl``) for each signal scored.  The log captures all
sub-score breakdowns and can be used offline for logistic-regression analysis
to derive data-driven optimal weight profiles that replace the hand-tuned
``_CHANNEL_WEIGHT_PROFILES``.  The ``outcome`` field in each record is left
empty at scoring time and can be populated later by the performance tracker.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from config import CONFIDENCE_LOG_ENABLED, CONFIDENCE_LOG_PATH, NEW_PAIR_MIN_CONFIDENCE

# Directory where per-channel learned weight files are stored.
# Each file is named ``learned_weights_{channel}.json`` and contains a dict
# of sub-score names → float weights that override ``_CHANNEL_WEIGHT_PROFILES``.
_LEARNED_WEIGHTS_DIR: str = "data"

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
_SCALP_DEFAULT_WEIGHTS: Dict[str, float] = {
    "smc": 1.0, "trend": 1.0, "liquidity": 1.0, "spread": 1.0,
    "data_sufficiency": 1.0, "multi_exchange": 1.0, "onchain": 1.0,
    "order_flow": 1.0, "sentiment": 0.0,
}

_CHANNEL_WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "360_SCALP":      _SCALP_DEFAULT_WEIGHTS,
    "360_SCALP_FVG":  _SCALP_DEFAULT_WEIGHTS,
    "360_SCALP_CVD":  _SCALP_DEFAULT_WEIGHTS,
    "360_SCALP_VWAP": _SCALP_DEFAULT_WEIGHTS,
    "360_SCALP_OBI":  _SCALP_DEFAULT_WEIGHTS,
    "360_SWING": {
        "smc": 0.7, "trend": 1.4, "liquidity": 1.0, "spread": 0.8,
        "data_sufficiency": 1.0, "multi_exchange": 1.0, "onchain": 1.2,
        "order_flow": 0.9, "sentiment": 0.0,
    },
    "360_SPOT": {
        "smc": 0.5, "trend": 1.4, "liquidity": 0.75, "spread": 0.8,
        "data_sufficiency": 1.0, "multi_exchange": 1.0, "onchain": 1.5,
        "order_flow": 0.8, "sentiment": 1.0,
    },
    "360_GEM": {
        "smc": 0.2, "trend": 0.8, "liquidity": 0.5, "spread": 0.5,
        "data_sufficiency": 1.0, "multi_exchange": 0.5, "onchain": 2.0,
        "order_flow": 0.5, "sentiment": 1.2,
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
    onchain_score: float = 0.0      # 0-10 (whale-aware; 0 = no data)
    order_flow_score: float = 0.0   # 0-20 (OI squeeze + CVD divergence + funding rate bonus)
    sentiment_score: float = 0.0    # -1 to +1 from ai_engine.get_ai_insight() (SPOT/GEM only)
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
    suppressed: bool = False
    suppressed_reason: str = ""
    regime: str = ""
    adaptive_threshold: float = 65.0


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
    macd_histogram: Optional[float] = None,
    macd_histogram_prev: Optional[float] = None,
    signal_direction: Optional[str] = None,
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
    macd_histogram:
        Most recent MACD histogram value (MACD line − signal line).
        Positive = bullish momentum, negative = bearish momentum.
    macd_histogram_prev:
        Previous MACD histogram value, used to detect growing/shrinking.
    signal_direction:
        ``"LONG"`` or ``"SHORT"`` – used to align MACD bonus/penalty.
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

    # MACD histogram bonus/penalty (up to +3, or −2 for contradiction)
    if macd_histogram is not None and signal_direction is not None:
        long_signal = signal_direction == "LONG"
        hist_aligned = (long_signal and macd_histogram > 0) or (
            not long_signal and macd_histogram < 0
        )
        hist_growing = (
            macd_histogram_prev is not None
            and (
                (long_signal and macd_histogram > macd_histogram_prev)
                or (not long_signal and macd_histogram < macd_histogram_prev)
            )
        )
        if hist_aligned:
            bonus = 2.0 + (1.0 if hist_growing else 0.0)
            s += bonus
        else:
            s -= 2.0

    return min(max(s, 0.0), 25.0)


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


def score_sentiment(sentiment_score: float, channel: Optional[str] = None) -> float:
    """Sentiment component (max 10). Only active for SPOT and GEM channels.

    Maps a sentiment score of [-1, +1] to a confidence contribution of
    [0, 10].  For SCALP and SWING channels, always returns 5.0 (neutral)
    so that high-frequency signals fire with zero external-network latency.

    Parameters
    ----------
    sentiment_score:
        Aggregate sentiment in the range [-1, +1] from
        :func:`src.ai_engine.get_ai_insight`.  0.0 is neutral.
    channel:
        Optional channel name.  When ``"360_SCALP*"`` or ``"360_SWING"``,
        returns 5.0 regardless of the sentiment value.

    Returns
    -------
    float
        0 (very bearish) → 10 (very bullish); 5.0 is neutral.
    """
    if channel is not None and (
        channel.startswith("360_SCALP") or channel == "360_SWING"
    ):
        return 5.0  # neutral — no latency added for short-term channels
    return round((max(-1.0, min(1.0, sentiment_score)) + 1.0) / 2.0 * 10.0, 2)


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

    # Funding rate alignment: contrarian bonus or crowded-trade penalty
    if funding_rate is not None and signal_direction is not None:
        if abs(funding_rate) >= _EXTREME_FUNDING_RATE:
            funding_positive = funding_rate > 0
            is_long = signal_direction == "LONG"
            contrarian = (funding_positive and not is_long) or (not funding_positive and is_long)
            crowded = (funding_positive and is_long) or (not funding_positive and not is_long)
            if contrarian:
                # Extreme funding in opposite direction = strong contrarian edge
                funding_bonus = min(abs(funding_rate) / 0.03, 1.0) * 5.0
                s += funding_bonus
            elif crowded:
                # Crowd is piling in the same direction = elevated risk
                s -= 3.0

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


def load_learned_weights(channel: str) -> Optional[Dict[str, float]]:
    """Load per-channel learned weights from disk.

    Checks for a JSON file at ``data/learned_weights_{channel}.json``.
    If it exists and is valid, returns the weight dict so it can override
    ``_CHANNEL_WEIGHT_PROFILES`` in :func:`compute_confidence`.  Returns
    ``None`` when the file is absent or cannot be parsed.

    Parameters
    ----------
    channel:
        Channel name (e.g. ``"360_SCALP"``).  Used to construct the file name.

    Returns
    -------
    Optional[Dict[str, float]]
        Loaded weight dict, or ``None`` to fall back to the built-in profiles.
    """
    # Sanitise: strip characters that could allow path traversal or injection.
    safe_channel = "".join(c for c in channel if c.isalnum() or c in ("_", "-"))
    if not safe_channel:
        return None
    path = os.path.join(_LEARNED_WEIGHTS_DIR, f"learned_weights_{safe_channel}.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def save_learned_weights(channel: str, weights: Dict[str, float]) -> None:
    """Persist per-channel learned weights to disk.

    Writes *weights* to ``data/learned_weights_{channel}.json``.  The file is
    created (or overwritten) so that subsequent calls to :func:`load_learned_weights`
    pick up the updated values without restarting the process.

    Parameters
    ----------
    channel:
        Channel name used to construct the output file name.
    weights:
        Dict of sub-score names → float weight values to persist.
    """
    # Sanitise: strip characters that could allow path traversal or injection.
    safe_channel = "".join(c for c in channel if c.isalnum() or c in ("_", "-"))
    if not safe_channel:
        return
    path = os.path.join(_LEARNED_WEIGHTS_DIR, f"learned_weights_{safe_channel}.json")
    try:
        os.makedirs(_LEARNED_WEIGHTS_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(weights, fh, indent=2)
    except OSError:
        pass  # Non-fatal: write failure must not block the scoring pipeline


def log_confidence_breakdown(    signal_id: str,
    channel: str,
    breakdown: Dict[str, float],
    total: float,
    session_multiplier: float,
    outcome: Optional[str] = None,
) -> None:
    """Append a structured confidence breakdown record to the confidence log.

    This function is used for offline analysis to derive data-driven optimal
    weight profiles via logistic regression or similar techniques.  The log
    file (``CONFIDENCE_LOG_PATH``) is in JSON Lines format, with one record
    per line.

    The ``outcome`` field is left empty at scoring time and can be populated
    later by the performance tracker once the trade result is known.

    Parameters
    ----------
    signal_id:
        Unique identifier of the signal being scored.
    channel:
        Channel name (e.g. ``"360_SCALP"``, ``"360_SWING"``).
    breakdown:
        Dict of sub-score names to weighted values (as returned by
        :func:`compute_confidence`).
    total:
        Final confidence total (0–100, post session-multiplier).
    session_multiplier:
        The session multiplier that was applied.
    outcome:
        Optional trade outcome string (e.g. ``"WIN"``, ``"LOSS"``).
        Leave as ``None`` at signal creation time.
    """
    record = {
        "signal_id": signal_id,
        "channel": channel,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **breakdown,
        "total": total,
        "session_multiplier": session_multiplier,
        "outcome": outcome,
    }
    try:
        log_dir = os.path.dirname(CONFIDENCE_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(CONFIDENCE_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass  # Non-fatal: logging failure must not block signal flow


def compute_confidence(
    inp: ConfidenceInput,
    session_now: Optional[datetime] = None,
    channel: Optional[str] = None,
    signal_id: Optional[str] = None,
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
    signal_id:
        Optional signal identifier.  When provided and
        ``CONFIDENCE_LOG_ENABLED`` is True, the full breakdown is written to
        the confidence log file for offline weight-profile optimisation.
    """
    weights = load_learned_weights(channel or "") or _CHANNEL_WEIGHT_PROFILES.get(channel or "", {})
    breakdown: Dict[str, float] = {
        "smc": inp.smc_score * weights.get("smc", 1.0),
        "trend": inp.trend_score * weights.get("trend", 1.0),
        "liquidity": inp.liquidity_score * weights.get("liquidity", 1.0),
        "spread": inp.spread_score * weights.get("spread", 1.0),
        "data_sufficiency": inp.data_sufficiency * weights.get("data_sufficiency", 1.0),
        "multi_exchange": inp.multi_exchange * weights.get("multi_exchange", 1.0),
        "onchain": inp.onchain_score * weights.get("onchain", 1.0),
        "order_flow": inp.order_flow_score * weights.get("order_flow", 1.0),
        "sentiment": score_sentiment(inp.sentiment_score, channel=channel)
            * weights.get("sentiment", 0.0),
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

    # Confidence log: write breakdown for offline weight-profile analysis.
    if CONFIDENCE_LOG_ENABLED and signal_id:
        log_confidence_breakdown(
            signal_id=signal_id,
            channel=channel or "",
            breakdown=breakdown,
            total=total,
            session_multiplier=session_mult,
        )

    return ConfidenceResult(
        total=total,
        breakdown=breakdown,
        capped=capped,
        blocked=blocked,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Adaptive threshold computation (PR: Signal Confidence Scoring)
# ---------------------------------------------------------------------------

# Regime-based threshold offsets — applied to the base minimum confidence.
_REGIME_THRESHOLD_OFFSETS: Dict[str, float] = {
    "TRENDING": -3.0,     # Lower bar in strong trends
    "RANGING": +5.0,      # Higher bar in range-bound markets
    "VOLATILE": +8.0,     # Much higher bar during extreme volatility
    "QUIET": 0.0,         # Neutral
}


def compute_adaptive_threshold(
    base_threshold: float = 65.0,
    regime: str = "",
    volatility_percentile: float = 0.5,
    channel: Optional[str] = None,
) -> float:
    """Compute an adaptive minimum confidence threshold.

    The threshold adjusts based on:

    * **Market regime** — trends lower the bar (more signal opportunities),
      ranging / volatile markets raise it (fewer but higher quality).
    * **Volatility percentile** — extremely high volatility (>90th pctile)
      adds an extra buffer.

    Parameters
    ----------
    base_threshold:
        Starting threshold value.
    regime:
        Market regime string (``"TRENDING"``, ``"RANGING"``, ``"VOLATILE"``,
        ``"QUIET"``, or ``""``).
    volatility_percentile:
        Current volatility relative to historical distribution (0–1).
    channel:
        Optional channel name.  GEM channel gets a lower base threshold.

    Returns
    -------
    float
        Adaptive threshold clamped to ``[50, 90]``.
    """
    threshold = base_threshold

    # Channel-specific base
    if channel == "360_GEM":
        threshold -= 5.0
    elif channel and channel.startswith("360_SCALP"):
        threshold += 2.0

    # Regime adjustment
    threshold += _REGIME_THRESHOLD_OFFSETS.get(regime, 0.0)

    # Extreme volatility buffer
    if volatility_percentile > 0.9:
        threshold += (volatility_percentile - 0.9) * 30.0  # up to +3 at 100th pctile

    return max(50.0, min(90.0, threshold))


def compute_per_signal_confidence(
    inp: ConfidenceInput,
    session_now: Optional[datetime] = None,
    channel: Optional[str] = None,
    signal_id: Optional[str] = None,
    regime: str = "",
    volatility_percentile: float = 0.5,
    cluster_suppressed: bool = False,
    cluster_reason: str = "",
) -> ConfidenceResult:
    """Compute confidence with per-signal metadata and adaptive thresholds.

    This extends :func:`compute_confidence` by adding:

    * Adaptive threshold computation based on regime and volatility.
    * Cluster suppression status.
    * Regime context in the result.

    Parameters
    ----------
    inp:
        All sub-score inputs.
    session_now:
        Optional UTC datetime for session multiplier.
    channel:
        Optional channel name.
    signal_id:
        Optional signal identifier for logging.
    regime:
        Current market regime string.
    volatility_percentile:
        Current volatility percentile (0–1).
    cluster_suppressed:
        Whether the signal was suppressed by cluster detection.
    cluster_reason:
        Reason string from the cluster suppressor.

    Returns
    -------
    ConfidenceResult
        Result with adaptive threshold, suppression status, and regime.
    """
    result = compute_confidence(inp, session_now, channel, signal_id)

    # Compute adaptive threshold
    threshold = compute_adaptive_threshold(
        base_threshold=65.0,
        regime=regime,
        volatility_percentile=volatility_percentile,
        channel=channel,
    )
    result.adaptive_threshold = threshold
    result.regime = regime

    # Apply cluster suppression
    if cluster_suppressed:
        result.suppressed = True
        result.suppressed_reason = cluster_reason

    return result


@dataclass
class ConfidenceMetadata:
    """Unified confidence metadata for a single signal.

    Consolidates regime context, adaptive thresholds, suppression status,
    and AI scoring results into a single structure that can be passed
    alongside signal data through the processing pipeline.

    Attributes
    ----------
    base_confidence:
        Raw confidence before any adjustments.
    final_confidence:
        Adjusted confidence after all processing.
    adaptive_threshold:
        Dynamic minimum threshold for this signal's context.
    regime:
        Market regime at signal creation.
    volatility_percentile:
        Current volatility relative to history (0–1).
    cluster_suppressed:
        Whether the signal was suppressed by cluster detection.
    cluster_reason:
        Reason string from the cluster suppressor.
    ai_adjustment:
        AI-derived confidence adjustment.
    is_high_confidence:
        Whether the signal exceeds its adaptive threshold.
    """

    base_confidence: float = 0.0
    final_confidence: float = 0.0
    adaptive_threshold: float = 65.0
    regime: str = ""
    volatility_percentile: float = 0.5
    cluster_suppressed: bool = False
    cluster_reason: str = ""
    ai_adjustment: float = 0.0
    is_high_confidence: bool = False


def build_confidence_metadata(
    inp: ConfidenceInput,
    session_now: Optional[datetime] = None,
    channel: Optional[str] = None,
    signal_id: Optional[str] = None,
    regime: str = "",
    volatility_percentile: float = 0.5,
    cluster_suppressed: bool = False,
    cluster_reason: str = "",
    ai_adjustment: float = 0.0,
) -> ConfidenceMetadata:
    """Build unified confidence metadata for a signal.

    Combines the results of :func:`compute_per_signal_confidence` with
    AI scoring adjustments into a single :class:`ConfidenceMetadata`
    instance.

    Parameters
    ----------
    inp:
        All sub-score inputs.
    session_now:
        Optional UTC datetime for session multiplier.
    channel:
        Optional channel name.
    signal_id:
        Optional signal identifier for logging.
    regime:
        Current market regime string.
    volatility_percentile:
        Current volatility percentile (0–1).
    cluster_suppressed:
        Whether the signal was suppressed by cluster detection.
    cluster_reason:
        Reason string from the cluster suppressor.
    ai_adjustment:
        AI-derived confidence adjustment to apply.

    Returns
    -------
    ConfidenceMetadata
        Unified metadata with all confidence-related information.
    """
    result = compute_per_signal_confidence(
        inp=inp,
        session_now=session_now,
        channel=channel,
        signal_id=signal_id,
        regime=regime,
        volatility_percentile=volatility_percentile,
        cluster_suppressed=cluster_suppressed,
        cluster_reason=cluster_reason,
    )

    base = result.total
    final = max(0.0, min(100.0, base + ai_adjustment))
    threshold = result.adaptive_threshold

    return ConfidenceMetadata(
        base_confidence=base,
        final_confidence=final,
        adaptive_threshold=threshold,
        regime=regime,
        volatility_percentile=volatility_percentile,
        cluster_suppressed=cluster_suppressed,
        cluster_reason=cluster_reason,
        ai_adjustment=ai_adjustment,
        is_high_confidence=final >= threshold,
    )

