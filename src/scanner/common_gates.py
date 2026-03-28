"""Common Channel Gates (PR05) – shared gating logic for all scalp channels.

Extracts duplicated pre-signal checks from individual scalp channels into a
single authoritative module so bug-fixes and improvements propagate
automatically to every channel that calls these helpers.

Shared gates
------------
* spread / volume basic filters
* ADX gating (trend strength)
* RSI extreme / overbought / oversold guard
* Regime compatibility check
* High-probability pair filter (delegates to filter_module)

Channel-specific logic (FVG zone proximity, CVD divergence, OBI levels,
VWAP band position) continues to live in each channel's own evaluator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from src.scanner.filter_module import is_high_probability, DEFAULT_PROBABILITY_THRESHOLD

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ADX thresholds
# ---------------------------------------------------------------------------
# Channels that want a minimum trend strength gate
ADX_MIN_SCALP: float = 20.0
# Channels that want a maximum ADX (range-fade, mean-reversion)
ADX_MAX_RANGE_FADE: float = 25.0

# ---------------------------------------------------------------------------
# RSI guard levels
# ---------------------------------------------------------------------------
RSI_OVERBOUGHT: float = 75.0
RSI_OVERSOLD: float = 25.0

# ---------------------------------------------------------------------------
# Regimes where scalp channels produce low-quality signals
# ---------------------------------------------------------------------------
_QUIET_REGIME_SCALP_PENALTY: float = 15.0  # confidence penalty in QUIET regime
_VOLATILE_REGIME_SCALP_PENALTY: float = 10.0


def check_basic_filters(
    spread_pct: float,
    volume_24h_usd: float,
    max_spread: float,
    min_volume: float,
) -> bool:
    """Return True when spread and volume pass the channel-configured thresholds.

    This mirrors the ``_pass_basic_filters`` call already present in
    :class:`~src.channels.base.BaseChannel` but adds explicit debug logging
    so suppression reasons are visible in the scan log.
    """
    if spread_pct > max_spread:
        log.debug("Spread gate failed: %.4f%% > %.4f%%", spread_pct, max_spread)
        return False
    if volume_24h_usd < min_volume:
        log.debug("Volume gate failed: %.0f < %.0f", volume_24h_usd, min_volume)
        return False
    return True


def check_adx_gate(
    adx_value: Optional[float],
    adx_min: float = ADX_MIN_SCALP,
) -> bool:
    """Return True when ADX indicates sufficient trend strength.

    Returns True (passes) when *adx_value* is ``None`` (indicator not yet
    available) so that new pairs aren't blocked purely by missing data.
    """
    if adx_value is None:
        return True
    return adx_value >= adx_min


def check_range_fade_adx(
    adx_value: Optional[float],
    adx_max: float = ADX_MAX_RANGE_FADE,
) -> bool:
    """Return True when ADX is below *adx_max* (valid for mean-reversion setups)."""
    if adx_value is None:
        return True
    return adx_value <= adx_max


def check_rsi_extreme_gate(
    rsi_value: Optional[float],
    direction: str,
    overbought: float = RSI_OVERBOUGHT,
    oversold: float = RSI_OVERSOLD,
) -> bool:
    """Return False when RSI is at an extreme that invalidates the trade direction.

    * LONG  signal rejected when RSI ≥ *overbought* (chasing overbought)
    * SHORT signal rejected when RSI ≤ *oversold*  (fading oversold)

    Returns True when *rsi_value* is None (data not available).
    """
    if rsi_value is None:
        return True
    direction_upper = direction.upper()
    if direction_upper == "LONG" and rsi_value >= overbought:
        log.debug("RSI extreme gate: LONG rejected (RSI=%.1f ≥ %.1f)", rsi_value, overbought)
        return False
    if direction_upper == "SHORT" and rsi_value <= oversold:
        log.debug("RSI extreme gate: SHORT rejected (RSI=%.1f ≤ %.1f)", rsi_value, oversold)
        return False
    return True


def check_regime_compatibility(
    regime: str,
    channel_name: str,
    regime_incompatible_map: Optional[Dict[str, list]] = None,
) -> bool:
    """Return False when *channel_name* is incompatible with the current *regime*.

    Parameters
    ----------
    regime:
        Current market regime string.
    channel_name:
        The channel's name (e.g. ``"360_SCALP_VWAP"``).
    regime_incompatible_map:
        Optional override for the default incompatibility map.  When ``None``
        the built-in defaults (VWAP blocked in QUIET, SWING blocked in VOLATILE)
        are used.

    Returns
    -------
    bool
        ``True`` when the channel may run, ``False`` when it is blocked.
    """
    _defaults: Dict[str, list] = {
        "360_SCALP_VWAP": ["QUIET"],
        "360_SWING": ["VOLATILE", "DIRTY_RANGE"],
        "360_SPOT": ["DIRTY_RANGE"],
    }
    compat_map = regime_incompatible_map if regime_incompatible_map is not None else _defaults
    blocked_regimes = compat_map.get(channel_name, [])
    regime_upper = regime.upper() if regime else ""
    if regime_upper in [r.upper() for r in blocked_regimes]:
        log.debug(
            "Regime compatibility gate: %s blocked in %s regime",
            channel_name, regime_upper,
        )
        return False
    return True


def check_probability_gate(
    pair_data: Dict[str, Any],
    threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    channel: str = "",
) -> Tuple[bool, float]:
    """Run the high-probability filter and return (allowed, score).

    Wraps :func:`~src.scanner.filter_module.is_high_probability` with
    additional channel-level logging so every suppression event is visible.

    Parameters
    ----------
    pair_data:
        Dict with keys: ``regime``, ``spread_pct``, ``volume_24h_usd``,
        ``atr_pct``, ``hit_rate``.
    threshold:
        Minimum score to allow a signal (default ``70.0``).
    channel:
        Channel name for log messages.

    Returns
    -------
    (allowed, score)
    """
    allowed, score = is_high_probability(pair_data, threshold)
    if not allowed:
        log.debug(
            "Probability gate suppressed signal: channel=%s score=%.1f < %.1f "
            "(pair=%s regime=%s)",
            channel or "unknown",
            score,
            threshold,
            pair_data.get("symbol", ""),
            pair_data.get("regime", ""),
        )
    return allowed, score


def regime_confidence_adjustment(regime: str, channel_name: str) -> float:
    """Return a confidence adjustment (positive or negative) for the regime.

    Used as a soft penalty rather than a hard gate so that high-quality
    signals can still pass through in adverse regimes.

    Returns
    -------
    float
        Confidence delta (negative = penalty, positive = boost).
    """
    regime_upper = regime.upper() if regime else ""
    # Scalp channels in QUIET regime → significant penalty
    if regime_upper == "QUIET" and "SCALP" in channel_name.upper():
        return -_QUIET_REGIME_SCALP_PENALTY
    # All channels in VOLATILE → moderate penalty (high false-signal rate)
    if regime_upper == "VOLATILE":
        return -_VOLATILE_REGIME_SCALP_PENALTY
    return 0.0
