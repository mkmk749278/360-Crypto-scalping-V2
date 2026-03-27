"""Signal-quality helpers for scanner funnel classification and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from src.regime import MarketRegime
from src.smc import Direction
from src.utils import get_logger, price_decimal_fmt

log = get_logger("signal_quality")


class SetupClass(str, Enum):
    TREND_PULLBACK_CONTINUATION = "TREND_PULLBACK_CONTINUATION"
    BREAKOUT_RETEST = "BREAKOUT_RETEST"
    LIQUIDITY_SWEEP_REVERSAL = "LIQUIDITY_SWEEP_REVERSAL"
    RANGE_REJECTION = "RANGE_REJECTION"
    MOMENTUM_EXPANSION = "MOMENTUM_EXPANSION"
    EXHAUSTION_FADE = "EXHAUSTION_FADE"
    RANGE_FADE = "RANGE_FADE"
    WHALE_MOMENTUM = "WHALE_MOMENTUM"


class MarketState(str, Enum):
    STRONG_TREND = "STRONG_TREND"
    WEAK_TREND = "WEAK_TREND"
    CLEAN_RANGE = "CLEAN_RANGE"
    DIRTY_RANGE = "DIRTY_RANGE"
    BREAKOUT_EXPANSION = "BREAKOUT_EXPANSION"
    VOLATILE_UNSUITABLE = "VOLATILE_UNSUITABLE"


class QualityTier(str, Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"


CHANNEL_SETUP_COMPATIBILITY: Dict[str, set[SetupClass]] = {
    "360_SCALP": {
        SetupClass.TREND_PULLBACK_CONTINUATION,
        SetupClass.BREAKOUT_RETEST,
        SetupClass.LIQUIDITY_SWEEP_REVERSAL,
        SetupClass.MOMENTUM_EXPANSION,
        SetupClass.WHALE_MOMENTUM,
        SetupClass.RANGE_FADE,
    },
    "360_SWING": {
        SetupClass.TREND_PULLBACK_CONTINUATION,
        SetupClass.BREAKOUT_RETEST,
        SetupClass.LIQUIDITY_SWEEP_REVERSAL,
    },
    "360_SPOT": {
        SetupClass.TREND_PULLBACK_CONTINUATION,
        SetupClass.BREAKOUT_RETEST,
        SetupClass.LIQUIDITY_SWEEP_REVERSAL,
        SetupClass.MOMENTUM_EXPANSION,
    },
    "360_GEM": {
        SetupClass.TREND_PULLBACK_CONTINUATION,
        SetupClass.BREAKOUT_RETEST,
        SetupClass.LIQUIDITY_SWEEP_REVERSAL,
        SetupClass.RANGE_REJECTION,
        SetupClass.MOMENTUM_EXPANSION,
        SetupClass.EXHAUSTION_FADE,
    },
}


REGIME_SETUP_COMPATIBILITY: Dict[MarketState, set[SetupClass]] = {
    MarketState.STRONG_TREND: {
        SetupClass.TREND_PULLBACK_CONTINUATION,
        SetupClass.BREAKOUT_RETEST,
        SetupClass.MOMENTUM_EXPANSION,
        SetupClass.WHALE_MOMENTUM,
    },
    MarketState.WEAK_TREND: {
        SetupClass.TREND_PULLBACK_CONTINUATION,
        SetupClass.BREAKOUT_RETEST,
        SetupClass.LIQUIDITY_SWEEP_REVERSAL,
        SetupClass.WHALE_MOMENTUM,
    },
    MarketState.CLEAN_RANGE: {
        SetupClass.RANGE_REJECTION,
        SetupClass.LIQUIDITY_SWEEP_REVERSAL,
        SetupClass.EXHAUSTION_FADE,
        SetupClass.RANGE_FADE,
    },
    MarketState.DIRTY_RANGE: {
        SetupClass.LIQUIDITY_SWEEP_REVERSAL,
        SetupClass.RANGE_FADE,
    },
    MarketState.BREAKOUT_EXPANSION: {
        SetupClass.BREAKOUT_RETEST,
        SetupClass.MOMENTUM_EXPANSION,
        SetupClass.TREND_PULLBACK_CONTINUATION,
        SetupClass.LIQUIDITY_SWEEP_REVERSAL,
        SetupClass.WHALE_MOMENTUM,
    },
    MarketState.VOLATILE_UNSUITABLE: set(),
}

# Maximum SL distance (as a percentage of entry) allowed per channel.
# Signals whose structure-based SL would exceed this cap are clamped.
_MAX_SL_PCT_BY_CHANNEL: Dict[str, float] = {
    "360_SCALP": 1.5,
    "360_SCALP_FVG": 1.0,
    "360_SCALP_CVD": 1.0,
    "360_SCALP_VWAP": 1.0,
    "360_SCALP_OBI": 1.0,
    "360_SWING": 3.0,
    "360_SPOT": 2.0,
}


@dataclass
class PairQualityAssessment:
    passed: bool
    score: float
    label: str
    volume_tier: str
    spread_score: float
    volatility_score: float
    noise_score: float
    reason: str = ""


@dataclass
class SetupAssessment:
    setup_class: SetupClass
    thesis: str
    channel_compatible: bool
    regime_compatible: bool
    reason: str = ""


@dataclass
class ExecutionAssessment:
    passed: bool
    trigger_confirmed: bool
    extension_ratio: float
    anchor_price: float
    entry_zone: str
    execution_note: str
    reason: str = ""


@dataclass
class RiskAssessment:
    passed: bool
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    r_multiple: float
    invalidation_summary: str
    reason: str = ""


@dataclass
class ComponentScore:
    components: Dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    quality_tier: QualityTier = QualityTier.C


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _last(values: Any, default: float = 0.0) -> float:
    try:
        if values is None or len(values) == 0:
            return default
        return float(values[-1])
    except (TypeError, ValueError):
        return default


def _wickiness(candles: Optional[dict], lookback: int = 12) -> float:
    if not candles:
        return 1.0
    highs = candles.get("high", [])
    lows = candles.get("low", [])
    closes = candles.get("close", [])
    if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
        return 1.0
    start = max(1, len(closes) - lookback)
    ratios = []
    for idx in range(start, len(closes)):
        high = _safe_float(highs[idx], _safe_float(closes[idx]))
        low = _safe_float(lows[idx], _safe_float(closes[idx]))
        close = _safe_float(closes[idx], 0.0)
        prev_close = _safe_float(closes[idx - 1], close)
        candle_range = max(high - low, max(abs(close), 1.0) * 0.0001)
        body = max(abs(close - prev_close), candle_range * 0.35, max(abs(close), 1.0) * 0.0001)
        wick = max(high - max(close, prev_close), 0.0) + max(min(close, prev_close) - low, 0.0)
        ratios.append(wick / body)
    if not ratios:
        return 1.0
    return round(sum(ratios) / len(ratios), 3)


def _recent_structure(candles: Optional[dict], direction: Direction, lookback: int = 12) -> float:
    if not candles:
        return 0.0
    highs = candles.get("high", [])
    lows = candles.get("low", [])
    if direction == Direction.LONG and len(lows) > 0:
        segment = lows[-lookback:]
        return float(np.min(segment))
    if direction == Direction.SHORT and len(highs) > 0:
        segment = highs[-lookback:]
        return float(np.max(segment))
    return 0.0


def classify_market_state(
    regime_result: Any,
    indicators: Dict[str, Any],
    candles: Optional[dict],
    spread_pct: float,
) -> MarketState:
    adx_val = _safe_float(indicators.get("adx_last"))
    momentum = abs(_safe_float(indicators.get("momentum_last")))
    bb_width = _safe_float(getattr(regime_result, "bb_width_pct", indicators.get("bb_width_pct")))
    atr_val = _safe_float(indicators.get("atr_last"))
    close = _last(candles.get("close", []) if candles else [], 1.0)
    atr_pct = (atr_val / close * 100.0) if close else 0.0
    wickiness = _wickiness(candles)
    regime = getattr(regime_result, "regime", MarketRegime.RANGING)
    if isinstance(regime, str):
        try:
            regime = MarketRegime(regime)
        except ValueError:
            regime = MarketRegime.RANGING

    if spread_pct >= 0.03 or wickiness >= 3.0 or atr_pct >= 4.5:
        return MarketState.VOLATILE_UNSUITABLE
    if regime == MarketRegime.VOLATILE:
        return (
            MarketState.BREAKOUT_EXPANSION
            if adx_val >= 24.0 and momentum >= 0.45 and wickiness <= 2.2
            else MarketState.VOLATILE_UNSUITABLE
        )
    if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
        return MarketState.STRONG_TREND if adx_val >= 30.0 and momentum >= 0.15 else MarketState.WEAK_TREND
    if regime == MarketRegime.QUIET:
        return MarketState.CLEAN_RANGE if wickiness <= 1.5 else MarketState.DIRTY_RANGE
    if regime == MarketRegime.RANGING:
        if adx_val <= 18.0 and wickiness <= 3.1 and (bb_width == 0.0 or bb_width <= 3.2):
            return MarketState.CLEAN_RANGE
        return MarketState.DIRTY_RANGE
    if adx_val >= 24.0 and momentum >= 0.45:
        return MarketState.BREAKOUT_EXPANSION
    return MarketState.DIRTY_RANGE


def assess_pair_quality(
    volume_24h: float,
    spread_pct: float,
    indicators: Dict[str, Any],
    candles: Optional[dict],
) -> PairQualityAssessment:
    atr_val = _safe_float(indicators.get("atr_last"))
    close = _last(candles.get("close", []) if candles else [], 1.0)
    atr_pct = (atr_val / close * 100.0) if close else 0.0
    wickiness = _wickiness(candles)

    spread_score = max(0.0, min(100.0, 100.0 - (spread_pct / 0.02) * 100.0))
    volume_score = max(0.0, min(100.0, (volume_24h / 15_000_000.0) * 100.0))
    if 0.15 <= atr_pct <= 3.5:
        volatility_score = 100.0
    elif atr_pct < 0.15:
        volatility_score = max(20.0, atr_pct / 0.15 * 100.0)
    else:
        volatility_score = max(0.0, 100.0 - ((atr_pct - 3.5) / 3.0) * 100.0)
    noise_score = max(0.0, min(100.0, 100.0 - max(wickiness - 1.0, 0.0) * 35.0))
    total = round(
        spread_score * 0.3 + volume_score * 0.3 + volatility_score * 0.2 + noise_score * 0.2,
        2,
    )

    volume_tier = "ELITE" if volume_24h >= 20_000_000 else "HIGH" if volume_24h >= 10_000_000 else "NORMAL"
    label = "ELITE" if total >= 85 else "GOOD" if total >= 72 else "WEAK"
    passed = total >= 58 and spread_pct <= 0.05 and volume_24h >= 1_000_000
    reason = ""
    if not passed:
        if spread_pct > 0.05:
            reason = "spread too wide"
        elif volume_24h < 1_000_000:
            reason = "liquidity too thin"
        else:
            reason = "pair quality below threshold"

    return PairQualityAssessment(
        passed=passed,
        score=total,
        label=label,
        volume_tier=volume_tier,
        spread_score=round(spread_score, 2),
        volatility_score=round(volatility_score, 2),
        noise_score=round(noise_score, 2),
        reason=reason,
    )


# Per-channel spread limits for `assess_pair_quality_for_channel()`.
# Tighter limits for execution-sensitive scalp channels, wider for longer-hold
# strategies that can absorb higher spread costs over extended holding periods.
_SPREAD_LIMIT_BY_CHANNEL: Dict[str, float] = {
    "360_SCALP":      0.025,  # Tightest — execution-sensitive
    "360_SCALP_FVG":  0.03,
    "360_SCALP_CVD":  0.03,
    "360_SCALP_OBI":  0.03,
    "360_SCALP_VWAP": 0.03,
    "360_SWING":      0.05,   # Wider allowed — longer holding period
    "360_SPOT":       0.06,   # Widest intraday channel — multi-day holds
    "360_GEM":        0.08,   # Gem/altcoin pairs have wider spreads
}

# Minimum 24h volume (USD) for non-SCALP channels.  Scalp channels keep the
# higher $1M floor to ensure adequate liquidity for tight execution.
_MIN_VOLUME_NON_SCALP: float = 500_000.0

# Per-channel minimum composite score thresholds.
_MIN_COMPOSITE_SCORE_BY_CHANNEL: Dict[str, float] = {
    "360_SCALP":      58.0,
    "360_SCALP_FVG":  58.0,
    "360_SCALP_CVD":  58.0,
    "360_SCALP_OBI":  58.0,
    "360_SCALP_VWAP": 58.0,
    "360_SWING":      50.0,
    "360_SPOT":       45.0,
    "360_GEM":        40.0,
}

# Per-channel minimum 24h volume floors (USD).
_MIN_VOLUME_BY_CHANNEL: Dict[str, float] = {
    "360_SCALP":      1_000_000.0,
    "360_SCALP_FVG":  1_000_000.0,
    "360_SCALP_CVD":  1_000_000.0,
    "360_SCALP_OBI":  1_000_000.0,
    "360_SCALP_VWAP": 1_000_000.0,
    "360_SWING":        500_000.0,
    "360_SPOT":         250_000.0,
    "360_GEM":          100_000.0,
}


def assess_pair_quality_for_channel(
    volume_24h: float,
    spread_pct: float,
    indicators: Dict[str, Any],
    candles: Optional[dict],
    channel_name: str,
    current_regime: str = "RANGING",
) -> PairQualityAssessment:
    """Assess pair quality with per-channel spread and volume thresholds.

    Applies channel-specific spread limits from :data:`_SPREAD_LIMIT_BY_CHANNEL`
    instead of the global hard gate.  Non-SCALP channels also use a lower
    minimum volume floor (:data:`_MIN_VOLUME_NON_SCALP`) to avoid excluding
    valid lower-cap futures pairs.

    The composite scoring (spread/volume/volatility/noise) and pass/fail
    reason string are identical to :func:`assess_pair_quality`; only the
    final hard-gate thresholds differ.

    Parameters
    ----------
    volume_24h:
        24-hour traded volume in USD.
    spread_pct:
        Current bid-ask spread as a fraction (e.g. ``0.002`` = 0.2 %).
    indicators:
        Indicator dict for the primary timeframe (must contain ``atr_last``).
    candles:
        Candle OHLCV dict for the primary timeframe.
    channel_name:
        Name of the trading channel (e.g. ``"360_SCALP"``, ``"360_SWING"``).

    Returns
    -------
    :class:`PairQualityAssessment`
    """
    atr_val = _safe_float(indicators.get("atr_last"))
    close = _last(candles.get("close", []) if candles else [], 1.0)
    atr_pct = (atr_val / close * 100.0) if close else 0.0
    wickiness = _wickiness(candles)

    spread_score = max(0.0, min(100.0, 100.0 - (spread_pct / 0.02) * 100.0))
    volume_score = max(0.0, min(100.0, (volume_24h / 15_000_000.0) * 100.0))
    if 0.15 <= atr_pct <= 3.5:
        volatility_score = 100.0
    elif atr_pct < 0.15:
        volatility_score = max(20.0, atr_pct / 0.15 * 100.0)
    else:
        volatility_score = max(0.0, 100.0 - ((atr_pct - 3.5) / 3.0) * 100.0)
    noise_score = max(0.0, min(100.0, 100.0 - max(wickiness - 1.0, 0.0) * 35.0))
    total = round(
        spread_score * 0.3 + volume_score * 0.3 + volatility_score * 0.2 + noise_score * 0.2,
        2,
    )

    volume_tier = "ELITE" if volume_24h >= 20_000_000 else "HIGH" if volume_24h >= 10_000_000 else "NORMAL"
    label = "ELITE" if total >= 85 else "GOOD" if total >= 72 else "WEAK"

    # Per-channel thresholds
    spread_limit = _SPREAD_LIMIT_BY_CHANNEL.get(channel_name, 0.05)
    min_composite = _MIN_COMPOSITE_SCORE_BY_CHANNEL.get(channel_name, 58.0)
    min_volume = _MIN_VOLUME_BY_CHANNEL.get(channel_name, 500_000.0)

    # Regime-aware spread relaxation: widen spread tolerance by 30% in VOLATILE regime
    effective_spread_limit = spread_limit * 1.3 if current_regime == "VOLATILE" else spread_limit

    passed = (
        total >= min_composite
        and spread_pct <= effective_spread_limit
        and volume_24h >= min_volume
    )
    reason = ""
    if not passed:
        if spread_pct > effective_spread_limit:
            reason = "spread too wide"
        elif volume_24h < min_volume:
            reason = "liquidity too thin"
        else:
            reason = "pair quality below threshold"

    return PairQualityAssessment(
        passed=passed,
        score=total,
        label=label,
        volume_tier=volume_tier,
        spread_score=round(spread_score, 2),
        volatility_score=round(volatility_score, 2),
        noise_score=round(noise_score, 2),
        reason=reason,
    )


def classify_setup(
    channel_name: str,
    signal: Any,
    indicators: Dict[str, Dict[str, Any]],
    smc_data: Dict[str, Any],
    market_state: MarketState,
) -> SetupAssessment:
    primary_tf = "4h" if channel_name == "360_SPOT" else "1h" if channel_name == "360_SWING" else "5m"
    primary = indicators.get(primary_tf, indicators.get("5m", indicators.get("1m", {})))
    sweeps = smc_data.get("sweeps", [])
    mss = smc_data.get("mss")
    fvg = smc_data.get("fvg", [])
    whale = smc_data.get("whale_alert")
    delta_spike = bool(smc_data.get("volume_delta_spike"))
    momentum = _safe_float(primary.get("momentum_last"))

    # Check for WHALE_MOMENTUM / RANGE_FADE setups (SCALP sub-paths)
    if channel_name == "360_SCALP" and (whale or delta_spike) and abs(momentum) >= 0.3:
        setup = SetupClass.WHALE_MOMENTUM
    elif channel_name == "360_SCALP" and market_state in (MarketState.CLEAN_RANGE, MarketState.DIRTY_RANGE):
        setup = SetupClass.RANGE_FADE
    elif sweeps and signal.direction == sweeps[0].direction and (mss is not None or abs(momentum) >= 0.2):
        setup = SetupClass.LIQUIDITY_SWEEP_REVERSAL
    elif mss is not None and signal.direction == mss.direction:
        setup = SetupClass.BREAKOUT_RETEST
    elif fvg and market_state in (MarketState.STRONG_TREND, MarketState.WEAK_TREND, MarketState.BREAKOUT_EXPANSION):
        setup = SetupClass.TREND_PULLBACK_CONTINUATION
    elif market_state in (MarketState.CLEAN_RANGE, MarketState.DIRTY_RANGE):
        setup = SetupClass.RANGE_REJECTION
    elif abs(momentum) >= 0.45:
        setup = SetupClass.MOMENTUM_EXPANSION
    else:
        setup = SetupClass.TREND_PULLBACK_CONTINUATION

    channel_ok = setup in CHANNEL_SETUP_COMPATIBILITY.get(channel_name, set())
    regime_ok = setup in REGIME_SETUP_COMPATIBILITY.get(market_state, set())
    thesis = setup.value.replace("_", " ").title()
    reason = ""
    if not channel_ok:
        reason = f"{setup.value} not allowed in {channel_name}"
    elif not regime_ok:
        reason = f"{setup.value} conflicts with {market_state.value}"

    return SetupAssessment(
        setup_class=setup,
        thesis=thesis,
        channel_compatible=channel_ok,
        regime_compatible=regime_ok,
        reason=reason,
    )


def execution_quality_check(
    signal: Any,
    indicators: Dict[str, Dict[str, Any]],
    smc_data: Dict[str, Any],
    setup: SetupClass,
    market_state: MarketState,
) -> ExecutionAssessment:
    primary_tf = "4h" if signal.channel == "360_SPOT" else "1h" if signal.channel == "360_SWING" else "5m"
    primary = indicators.get(primary_tf, indicators.get("5m", indicators.get("1m", {})))
    atr_val = max(_safe_float(primary.get("atr_last")), signal.entry * 0.01)  # 1% floor
    ema_anchor = _safe_float(primary.get("ema21_last"), signal.entry)
    bb_mid = _safe_float(primary.get("bb_mid_last"), signal.entry)
    sweep = smc_data.get("sweeps", [None])[0] if smc_data.get("sweeps") else None
    sweep_level = _safe_float(sweep.sweep_level if sweep else None, signal.entry)
    mss = smc_data.get("mss")
    anchor = ema_anchor
    trigger_confirmed = False
    note = ""

    if setup == SetupClass.RANGE_REJECTION:
        anchor = _safe_float(primary.get("bb_lower_last") if signal.direction == Direction.LONG else primary.get("bb_upper_last"), signal.entry)
        trigger_confirmed = market_state == MarketState.CLEAN_RANGE and abs(signal.entry - anchor) <= atr_val * 0.7
        note = "Fade only at range edge; avoid mid-range entries."
    elif setup == SetupClass.LIQUIDITY_SWEEP_REVERSAL:
        anchor = sweep_level or signal.entry
        trigger_confirmed = bool(smc_data.get("sweeps")) and (
            signal.entry >= anchor if signal.direction == Direction.LONG else signal.entry <= anchor
        )
        note = "Need reclaim after sweep; do not front-run the reversal."
    elif setup == SetupClass.BREAKOUT_RETEST:
        anchor = _safe_float(mss.midpoint if mss is not None else None, ema_anchor)
        trigger_confirmed = mss is not None and (
            signal.entry >= anchor if signal.direction == Direction.LONG else signal.entry <= anchor
        )
        note = "Enter on retest hold, not on the first expansion candle."
    elif setup == SetupClass.MOMENTUM_EXPANSION:
        anchor = _safe_float(primary.get("ema9_last"), ema_anchor)
        trigger_confirmed = abs(_safe_float(primary.get("momentum_last"))) >= 0.45
        note = "Momentum is valid only while flow stays one-sided; do not chase extensions."
    elif setup == SetupClass.EXHAUSTION_FADE:
        anchor = bb_mid or sweep_level or signal.entry
        trigger_confirmed = market_state == MarketState.CLEAN_RANGE and bool(smc_data.get("sweeps"))
        note = "Fade only after exhaustion is obvious and reclaim begins."
    elif setup == SetupClass.RANGE_FADE:
        anchor = _safe_float(primary.get("bb_lower_last") if signal.direction == Direction.LONG else primary.get("bb_upper_last"), signal.entry)
        trigger_confirmed = market_state in (MarketState.CLEAN_RANGE, MarketState.DIRTY_RANGE) and abs(signal.entry - anchor) <= atr_val * 0.8
        note = "Fade at band edge only; avoid mid-range entries."
    elif setup == SetupClass.WHALE_MOMENTUM:
        anchor = _safe_float(primary.get("ema9_last"), ema_anchor)
        trigger_confirmed = abs(_safe_float(primary.get("momentum_last"))) >= 0.3
        note = "Whale flow active; keep trailing stops tight and do not chase extensions."
    else:
        anchor = ema_anchor
        trigger_confirmed = (
            (_safe_float(primary.get("ema9_last")) >= ema_anchor and signal.direction == Direction.LONG)
            or (_safe_float(primary.get("ema9_last")) <= ema_anchor and signal.direction == Direction.SHORT)
        )
        note = "Wait for pullback confirmation around value; avoid late continuation entries."

    extension_ratio = round(abs(signal.entry - anchor) / max(atr_val, signal.entry * 0.0005), 2)
    max_extension = {
        SetupClass.TREND_PULLBACK_CONTINUATION: 1.5,
        SetupClass.BREAKOUT_RETEST: 1.3,
        SetupClass.LIQUIDITY_SWEEP_REVERSAL: 1.1,
        SetupClass.RANGE_REJECTION: 1.2,
        SetupClass.MOMENTUM_EXPANSION: 1.0,
        SetupClass.EXHAUSTION_FADE: 1.0,
        SetupClass.RANGE_FADE: 1.3,
        SetupClass.WHALE_MOMENTUM: 1.2,
    }[setup]
    passed = trigger_confirmed and extension_ratio <= max_extension
    zone_low = min(anchor, signal.entry)
    zone_high = max(anchor, signal.entry)
    # Use dynamic decimal places based on price magnitude for micro-cap tokens
    _zone_fmt = price_decimal_fmt(max(zone_low, zone_high, 1e-12))
    entry_zone = f"{zone_low:{_zone_fmt}} – {zone_high:{_zone_fmt}}"
    reason = ""
    if not trigger_confirmed:
        reason = "entry trigger not confirmed"
    elif extension_ratio > max_extension:
        reason = f"entry overextended ({extension_ratio:.2f} ATR)"

    return ExecutionAssessment(
        passed=passed,
        trigger_confirmed=trigger_confirmed,
        extension_ratio=extension_ratio,
        anchor_price=anchor,
        entry_zone=entry_zone,
        execution_note=note,
        reason=reason,
    )


def build_risk_plan(
    signal: Any,
    indicators: Dict[str, Dict[str, Any]],
    candles: Dict[str, dict],
    smc_data: Dict[str, Any],
    setup: SetupClass,
    spread_pct: float,
    channel: Optional[str] = None,
) -> RiskAssessment:
    primary_tf = "4h" if signal.channel == "360_SPOT" else "1h" if signal.channel == "360_SWING" else "5m"
    primary = indicators.get(primary_tf, indicators.get("5m", indicators.get("1m", {})))
    candle_bucket = candles.get(primary_tf, candles.get("5m", candles.get("1m", {})))
    atr_val = max(_safe_float(primary.get("atr_last")), signal.entry * 0.01)  # 1% of price as minimum ATR
    buffer = max(atr_val * 0.35, signal.entry * (spread_pct / 100.0) * 1.5)
    structure = _recent_structure(candle_bucket, signal.direction)

    if smc_data.get("sweeps"):
        sweep_level = _safe_float(smc_data["sweeps"][0].sweep_level)
        if signal.direction == Direction.LONG and 0 < sweep_level < signal.entry:
            structure = max(structure, sweep_level) if structure else sweep_level
        elif signal.direction == Direction.SHORT and sweep_level > signal.entry:
            structure = min(structure, sweep_level) if structure else sweep_level

    if signal.direction == Direction.LONG:
        structure = structure if 0 < structure < signal.entry else signal.entry - atr_val
        stop_loss = round(structure - buffer, 8)
    else:
        structure = structure if structure > signal.entry else signal.entry + atr_val
        stop_loss = round(structure + buffer, 8)

    # Channel-aware hard cap on SL distance – clamp oversized stops before
    # they inflate risk and produce trades that hit SL within seconds.
    _chan = channel or getattr(signal, "channel", None) or ""
    _max_sl_pct = _MAX_SL_PCT_BY_CHANNEL.get(_chan, 5.0) / 100.0
    if signal.entry > 0:
        _sl_dist_pct = abs(signal.entry - stop_loss) / signal.entry
        if _sl_dist_pct > _max_sl_pct:
            _capped_dist = signal.entry * _max_sl_pct
            if signal.direction == Direction.LONG:
                stop_loss = round(signal.entry - _capped_dist, 8)
            else:
                stop_loss = round(signal.entry + _capped_dist, 8)
            log.warning(
                "SL capped for %s %s: %.2f%% > %.2f%% max (capped to %.8f)",
                _chan,
                signal.direction.value,
                _sl_dist_pct * 100,
                _max_sl_pct * 100,
                stop_loss,
            )

    # Directional sanity check – reject immediately if the computed SL is on
    # the wrong side of the entry price (can happen with unusual price action
    # or very thin markets where ATR/structure estimates are unreliable).
    if signal.direction == Direction.LONG and stop_loss >= signal.entry:
        return RiskAssessment(
            passed=False,
            stop_loss=stop_loss,
            tp1=signal.tp1,
            tp2=signal.tp2,
            tp3=signal.tp3,
            r_multiple=0.0,
            invalidation_summary="SL computed above entry for LONG – risk plan rejected.",
            reason="SL above entry for LONG",
        )
    if signal.direction == Direction.SHORT and stop_loss <= signal.entry:
        return RiskAssessment(
            passed=False,
            stop_loss=stop_loss,
            tp1=signal.tp1,
            tp2=signal.tp2,
            tp3=signal.tp3,
            r_multiple=0.0,
            invalidation_summary="SL computed below entry for SHORT – risk plan rejected.",
            reason="SL below entry for SHORT",
        )

    risk = abs(signal.entry - stop_loss)
    if risk <= max(signal.entry * 0.0003, buffer * 0.5):
        return RiskAssessment(
            passed=False,
            stop_loss=signal.stop_loss,
            tp1=signal.tp1,
            tp2=signal.tp2,
            tp3=signal.tp3,
            r_multiple=0.0,
            invalidation_summary="Risk distance too tight for structural invalidation.",
            reason="risk distance too tight",
        )

    bb_mid = _safe_float(primary.get("bb_mid_last"), signal.entry)
    bb_upper = _safe_float(primary.get("bb_upper_last"), signal.entry + risk)
    bb_lower = _safe_float(primary.get("bb_lower_last"), signal.entry - risk)

    if setup == SetupClass.RANGE_REJECTION:
        if signal.direction == Direction.LONG:
            tp1 = max(signal.entry + risk * 0.9, bb_mid)
            tp2 = max(tp1 + risk * 0.4, bb_upper)
        else:
            tp1 = min(signal.entry - risk * 0.9, bb_mid)
            tp2 = min(tp1 - risk * 0.4, bb_lower)
        tp3 = None
    elif setup in (SetupClass.LIQUIDITY_SWEEP_REVERSAL, SetupClass.EXHAUSTION_FADE):
        tp1 = signal.entry + risk * 1.2 if signal.direction == Direction.LONG else signal.entry - risk * 1.2
        tp2 = signal.entry + risk * 2.1 if signal.direction == Direction.LONG else signal.entry - risk * 2.1
        tp3 = signal.entry + risk * 3.0 if signal.direction == Direction.LONG else signal.entry - risk * 3.0
    elif setup == SetupClass.MOMENTUM_EXPANSION:
        tp1 = signal.entry + risk * 1.4 if signal.direction == Direction.LONG else signal.entry - risk * 1.4
        tp2 = signal.entry + risk * 2.2 if signal.direction == Direction.LONG else signal.entry - risk * 2.2
        tp3 = signal.entry + risk * 3.2 if signal.direction == Direction.LONG else signal.entry - risk * 3.2
    else:
        tp1 = signal.entry + risk * 1.3 if signal.direction == Direction.LONG else signal.entry - risk * 1.3
        tp2 = signal.entry + risk * 2.3 if signal.direction == Direction.LONG else signal.entry - risk * 2.3
        tp3 = signal.entry + risk * 3.4 if signal.direction == Direction.LONG else signal.entry - risk * 3.4

    r_multiple = round(abs(tp1 - signal.entry) / risk, 2) if risk else 0.0
    min_rr = 0.9 if setup == SetupClass.RANGE_REJECTION else 1.2
    passed = r_multiple >= min_rr
    reason = "" if passed else f"rr {r_multiple:.2f} below {min_rr:.2f}"
    # Sanity check: reject if SL or any TP is negative, or SL distance > 5% of entry
    sl_pct = risk / signal.entry if signal.entry > 0 else 0.0
    if stop_loss <= 0 or tp1 <= 0 or tp2 <= 0 or (tp3 is not None and tp3 <= 0):
        passed = False
        reason = "SL or TP computed as non-positive (micro-cap price precision issue)"
    elif sl_pct > 0.05:
        passed = False
        reason = f"SL distance {sl_pct:.1%} exceeds 5% of entry (risk plan rejected)"
    # Dynamic decimal places for invalidation message (micro-cap tokens)
    _struct_fmt = price_decimal_fmt(structure)
    invalidation = f"{'Below' if signal.direction == Direction.LONG else 'Above'} {structure:{_struct_fmt}} structure + volatility buffer"

    return RiskAssessment(
        passed=passed,
        stop_loss=stop_loss,
        tp1=round(tp1, 8),
        tp2=round(tp2, 8),
        tp3=round(tp3, 8) if tp3 is not None else None,
        r_multiple=r_multiple,
        invalidation_summary=invalidation,
        reason=reason,
    )


def score_signal_components(
    *,
    pair_quality: PairQualityAssessment,
    setup: SetupAssessment,
    execution: ExecutionAssessment,
    risk: RiskAssessment,
    legacy_confidence: float,
    cross_verified: Optional[bool],
) -> ComponentScore:
    market_score = round(pair_quality.score * 0.25, 2)
    setup_score = 11.0
    if setup.setup_class in (
        SetupClass.TREND_PULLBACK_CONTINUATION,
        SetupClass.BREAKOUT_RETEST,
        SetupClass.LIQUIDITY_SWEEP_REVERSAL,
    ):
        setup_score += 6.0
    if setup.channel_compatible:
        setup_score += 4.0
    if setup.regime_compatible:
        setup_score += 4.0

    execution_score = round(
        8.0
        + (6.0 if execution.trigger_confirmed else 0.0)
        + max(0.0, 6.0 - max(execution.extension_ratio - 0.4, 0.0) * 4.0),
        2,
    )
    risk_score = round(8.0 + min(risk.r_multiple, 2.5) * 4.8, 2)
    context_score = round(min(max(legacy_confidence, 0.0), 100.0) * 0.1, 2)
    if cross_verified is True:
        context_score = min(10.0, context_score + 1.0)
    elif cross_verified is False:
        context_score = max(0.0, context_score - 2.0)

    components = {
        "market": round(min(market_score, 25.0), 2),
        "setup": round(min(setup_score, 25.0), 2),
        "execution": round(min(execution_score, 20.0), 2),
        "risk": round(min(risk_score, 20.0), 2),
        "context": round(min(context_score, 10.0), 2),
    }
    total = round(sum(components.values()), 2)
    tier = QualityTier.C
    if total >= 90.0:
        tier = QualityTier.A_PLUS
    elif total >= 82.0:
        tier = QualityTier.A
    elif total >= 74.0:
        tier = QualityTier.B
    return ComponentScore(components=components, total=total, quality_tier=tier)


# ---------------------------------------------------------------------------
# PR_09 — Composite Signal Scoring Engine
# ---------------------------------------------------------------------------


@dataclass
class ScoringInput:
    """All data needed to score a signal."""
    # SMC
    sweeps: Optional[List] = None            # List of LiquiditySweep objects
    mss: Optional[Any] = None               # MSSSignal or None
    fvg_zones: Optional[List] = None        # List of FVGZone objects
    # Regime
    regime: str = ""
    setup_class: str = ""
    atr_percentile: float = 50.0
    # Volume
    volume_last_usd: float = 0.0            # Last candle USD volume
    volume_avg_usd: float = 0.0             # 20-period average USD volume
    # Indicators
    macd_histogram_last: Optional[float] = None
    macd_histogram_prev: Optional[float] = None
    rsi_last: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    adx_last: Optional[float] = None
    direction: str = "LONG"
    # Pattern + MTF
    chart_patterns: Optional[List] = None   # List of PatternResult objects
    mtf_score: float = 0.0                  # 0.0–1.0 from MTF gate


class SignalScoringEngine:
    """Scores a candidate signal across six dimensions.

    Produces a deterministic, auditable 0–100 score composed of:
    - SMC confluence (max 25 pts)
    - Regime alignment (max 20 pts)
    - Volume confirmation (max 15 pts)
    - Indicator confluence (max 20 pts)
    - Candlestick patterns (max 10 pts)
    - MTF confirmation (max 10 pts)
    """

    # Setup classes that strongly align with each regime
    _REGIME_SETUP_AFFINITY: Dict[str, List[str]] = {
        "TRENDING_UP": ["LIQUIDITY_SWEEP_REVERSAL", "BREAKOUT_INITIAL", "BREAKOUT_RETEST",
                        "THREE_WHITE_SOLDIERS", "WHALE_MOMENTUM"],
        "TRENDING_DOWN": ["LIQUIDITY_SWEEP_REVERSAL", "BREAKOUT_INITIAL", "BREAKOUT_RETEST",
                          "THREE_BLACK_CROWS", "WHALE_MOMENTUM"],
        "RANGING": ["RANGE_FADE", "SWING_STANDARD"],
        "QUIET": ["RANGE_FADE"],
        "VOLATILE": ["WHALE_MOMENTUM", "LIQUIDITY_SWEEP_REVERSAL"],
    }

    def score(self, inp: ScoringInput) -> Dict[str, float]:
        """Return a dict with per-dimension scores and a 'total' key.

        All scores are in [0, max_for_dimension].
        """
        smc = self._score_smc(inp)
        regime = self._score_regime(inp)
        volume = self._score_volume(inp)
        indicators = self._score_indicators(inp)
        patterns = self._score_patterns(inp)
        mtf = self._score_mtf(inp)
        total = smc + regime + volume + indicators + patterns + mtf
        return {
            "smc": round(smc, 2),
            "regime": round(regime, 2),
            "volume": round(volume, 2),
            "indicators": round(indicators, 2),
            "patterns": round(patterns, 2),
            "mtf": round(mtf, 2),
            "total": round(min(100.0, total), 2),
        }

    # ------------------------------------------------------------------
    def _score_smc(self, inp: ScoringInput) -> float:
        """SMC confluence score, max 25 pts."""
        score = 0.0
        sweeps = inp.sweeps or []
        if sweeps:
            score += 10.0   # Base for any sweep
            # Quality bonus: if sweep is recent (index >= -3, i.e. within the last 3 candles) add 5 pts
            if sweeps[0].index >= -3:
                score += 5.0
        if inp.mss is not None:
            score += 8.0    # MSS confirmation adds weight
        fvg = inp.fvg_zones or []
        if fvg:
            score += 2.0    # FVG presence (minor confluence)
        return min(25.0, score)

    # ------------------------------------------------------------------
    def _score_regime(self, inp: ScoringInput) -> float:
        """Regime alignment score, max 20 pts."""
        if not inp.regime:
            return 10.0   # Neutral when no regime data
        affinity = self._REGIME_SETUP_AFFINITY.get(inp.regime.upper(), [])
        if inp.setup_class in affinity:
            base = 18.0   # Strong alignment
        elif affinity:
            base = 8.0    # Regime known but setup not optimal
        else:
            base = 10.0   # Unknown regime
        # Bonus for high ATR percentile in VOLATILE regime (energy behind the move)
        if inp.regime.upper() == "VOLATILE" and inp.atr_percentile >= 75:
            base = min(20.0, base + 2.0)
        return min(20.0, base)

    # ------------------------------------------------------------------
    def _score_volume(self, inp: ScoringInput) -> float:
        """Volume confirmation score, max 15 pts."""
        if inp.volume_avg_usd <= 0 or inp.volume_last_usd <= 0:
            return 7.5    # Neutral
        ratio = inp.volume_last_usd / inp.volume_avg_usd
        if ratio >= 3.0:
            return 15.0
        if ratio >= 2.0:
            return 12.0
        if ratio >= 1.5:
            return 9.0
        if ratio >= 1.0:
            return 6.0
        return 3.0   # Below-average volume

    # ------------------------------------------------------------------
    def _score_indicators(self, inp: ScoringInput) -> float:
        """Indicator confluence score, max 20 pts."""
        score = 0.0

        # MACD (max 7 pts)
        if inp.macd_histogram_last is not None and inp.macd_histogram_prev is not None:
            rising = inp.macd_histogram_last > inp.macd_histogram_prev
            positive = inp.macd_histogram_last > 0
            if inp.direction == "LONG":
                if rising and positive:
                    score += 7.0
                elif rising or positive:
                    score += 4.0
            else:
                falling = not rising
                negative = inp.macd_histogram_last < 0
                if falling and negative:
                    score += 7.0
                elif falling or negative:
                    score += 4.0

        # RSI (max 7 pts)
        if inp.rsi_last is not None:
            if inp.direction == "LONG":
                if inp.rsi_last <= 45:
                    score += 7.0    # Oversold or neutral — good for LONG
                elif inp.rsi_last <= 60:
                    score += 4.0
                else:
                    score += 1.0    # Overbought — risky
            else:
                if inp.rsi_last >= 55:
                    score += 7.0
                elif inp.rsi_last >= 40:
                    score += 4.0
                else:
                    score += 1.0

        # EMA alignment (max 6 pts)
        if inp.ema_fast is not None and inp.ema_slow is not None:
            aligned = (inp.ema_fast > inp.ema_slow if inp.direction == "LONG"
                       else inp.ema_fast < inp.ema_slow)
            score += 6.0 if aligned else 1.0

        return min(20.0, score)

    # ------------------------------------------------------------------
    def _score_patterns(self, inp: ScoringInput) -> float:
        """Candlestick pattern score, max 10 pts."""
        patterns = inp.chart_patterns or []
        if not patterns:
            return 5.0   # Neutral (no patterns detected either way)
        aligned = [p for p in patterns
                   if getattr(p, "direction", "") == inp.direction or
                      getattr(p, "direction", "") == "NEUTRAL"]
        bonus = sum(getattr(p, "confidence_bonus", 0.0) for p in aligned)
        return max(0.0, min(10.0, 5.0 + bonus * 0.5))

    # ------------------------------------------------------------------
    def _score_mtf(self, inp: ScoringInput) -> float:
        """MTF confirmation score, max 10 pts."""
        return round(inp.mtf_score * 10.0, 2)
