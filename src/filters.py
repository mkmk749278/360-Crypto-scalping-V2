"""Centralized filter functions shared across all channel strategies.

Each function returns ``True`` when the condition passes (signal may proceed)
and ``False`` when it should be filtered out.
"""

from __future__ import annotations


def check_spread(spread_pct: float, max_spread: float) -> bool:
    """Return True if spread is within acceptable bounds.

    Parameters
    ----------
    spread_pct:
        Current bid-ask spread as a percentage of mid-price.
    max_spread:
        Maximum acceptable spread percentage (from channel config).
    """
    return spread_pct <= max_spread


def check_adx(adx_val: float | None, min_adx: float, max_adx: float = 100.0) -> bool:
    """Return True if ADX is within [min_adx, max_adx].

    A ``None`` value (not yet computed) is treated as a filter failure.
    """
    if adx_val is None:
        return False
    return min_adx <= adx_val <= max_adx


def check_ema_alignment(
    ema_fast: float | None,
    ema_slow: float | None,
    direction: str,
) -> bool:
    """Return True when fast/slow EMAs are aligned with *direction*.

    Parameters
    ----------
    ema_fast:
        Value of the fast EMA (e.g. EMA-9).
    ema_slow:
        Value of the slow EMA (e.g. EMA-21).
    direction:
        ``"LONG"`` or ``"SHORT"``.
    """
    if ema_fast is None or ema_slow is None:
        return False
    if direction == "LONG":
        return ema_fast > ema_slow
    if direction == "SHORT":
        return ema_fast < ema_slow
    return False


def check_volume(volume_24h_usd: float, min_volume: float) -> bool:
    """Return True if 24-hour USD volume meets the minimum threshold.

    Parameters
    ----------
    volume_24h_usd:
        24-hour trading volume in USD.
    min_volume:
        Minimum required volume in USD.
    """
    return volume_24h_usd >= min_volume


def check_rsi(
    rsi_val: float | None,
    overbought: float,
    oversold: float,
    direction: str,
) -> bool:
    """Return True when RSI is not in an extreme zone conflicting with direction.

    For a ``LONG`` signal, RSI must be below the overbought threshold.
    For a ``SHORT`` signal, RSI must be above the oversold threshold.
    A ``None`` RSI value passes (no filter applied).

    Parameters
    ----------
    rsi_val:
        Current RSI value (0-100), or ``None`` if unavailable.
    overbought:
        Overbought threshold (e.g. 70).
    oversold:
        Oversold threshold (e.g. 30).
    direction:
        ``"LONG"`` or ``"SHORT"``.
    """
    if rsi_val is None:
        return True  # no data, don't filter
    if direction == "LONG":
        return rsi_val < overbought
    if direction == "SHORT":
        return rsi_val > oversold
    return True


# ---------------------------------------------------------------------------
# Regime-aware filter thresholds
# ---------------------------------------------------------------------------

# RSI thresholds by regime: (overbought, oversold)
_RSI_THRESHOLDS_BY_REGIME: dict[str, tuple[float, float]] = {
    "TRENDING_UP": (80.0, 20.0),    # Let momentum run further in trends
    "TRENDING_DOWN": (80.0, 20.0),  # Same — wider thresholds for trend continuation
    "RANGING": (70.0, 30.0),        # Tighter — mean-reversion is the edge
    "VOLATILE": (80.0, 20.0),       # Wider — RSI swings are larger
    "QUIET": (70.0, 30.0),          # Tighter — small moves matter more
}

# ADX minimum thresholds by (regime, setup_class)
_ADX_MIN_BY_CONTEXT: dict[tuple[str, str], float] = {
    # Trending setups need strong trend confirmation
    ("TRENDING_UP", "TREND_PULLBACK_CONTINUATION"): 22.0,
    ("TRENDING_UP", "BREAKOUT_RETEST"): 20.0,
    ("TRENDING_UP", "MOMENTUM_EXPANSION"): 25.0,
    ("TRENDING_DOWN", "TREND_PULLBACK_CONTINUATION"): 22.0,
    ("TRENDING_DOWN", "BREAKOUT_RETEST"): 20.0,
    # Range-bound setups need LOW ADX (ranging confirmation)
    ("RANGING", "RANGE_FADE"): 10.0,       # Very low ADX is fine for range-fade
    ("RANGING", "RANGE_REJECTION"): 12.0,
    ("QUIET", "RANGE_FADE"): 8.0,          # Quiet markets: even lower ADX okay
    ("QUIET", "RANGE_REJECTION"): 10.0,
    # Volatile setups
    ("VOLATILE", "WHALE_MOMENTUM"): 15.0,  # Whale momentum doesn't need trend
    ("VOLATILE", "MOMENTUM_EXPANSION"): 20.0,
}

# EMA alignment mode by regime
_EMA_MODE_BY_REGIME: dict[str, str] = {
    "TRENDING_UP": "STRICT",      # Require clear alignment
    "TRENDING_DOWN": "STRICT",
    "RANGING": "RELAXED",         # Don't require alignment for range setups
    "VOLATILE": "MODERATE",       # Require some alignment
    "QUIET": "RELAXED",
}


def get_rsi_thresholds(regime: str = "") -> tuple[float, float]:
    """Return (overbought, oversold) RSI thresholds for the given regime.

    Falls back to (75.0, 25.0) when regime is empty or unknown.
    """
    if not regime:
        return (75.0, 25.0)
    return _RSI_THRESHOLDS_BY_REGIME.get(regime, (75.0, 25.0))


def get_adx_min(regime: str = "", setup_class: str = "") -> float:
    """Return the minimum ADX threshold for the given regime and setup class.

    Falls back to 20.0 when regime/setup is empty or unknown.
    """
    if not regime:
        return 20.0
    key = (regime, setup_class)
    if key in _ADX_MIN_BY_CONTEXT:
        return _ADX_MIN_BY_CONTEXT[key]
    # Fallback by regime only
    regime_defaults = {
        "TRENDING_UP": 20.0,
        "TRENDING_DOWN": 20.0,
        "RANGING": 15.0,
        "VOLATILE": 18.0,
        "QUIET": 12.0,
    }
    return regime_defaults.get(regime, 20.0)


def check_rsi_regime(
    rsi_val: float | None,
    direction: str,
    regime: str = "",
) -> bool:
    """Regime-aware RSI check using adaptive thresholds.

    Uses regime-specific overbought/oversold thresholds instead of
    hard-coded values. Falls back to standard 75/25 when regime is unknown.
    """
    ob, oversold = get_rsi_thresholds(regime)
    return check_rsi(rsi_val, overbought=ob, oversold=oversold, direction=direction)


def check_adx_regime(
    adx_val: float | None,
    regime: str = "",
    setup_class: str = "",
    max_adx: float = 100.0,
) -> bool:
    """Regime-aware ADX check using adaptive minimum threshold.

    Uses regime+setup specific minimum ADX instead of a single config value.
    Falls back to standard 20.0 when regime/setup is unknown.
    """
    min_adx = get_adx_min(regime, setup_class)
    return check_adx(adx_val, min_adx, max_adx)


def check_spread_adaptive(
    spread_pct: float,
    max_spread: float,
    regime: str = "",
    atr_pct: float = 0.0,
) -> bool:
    """Regime-aware spread filter that adjusts tolerance for volatility.

    In VOLATILE regimes or when ATR is high, spreads naturally widen —
    the filter relaxes max_spread by up to 50%.
    In QUIET regimes, spreads should be tighter — the filter tightens
    max_spread by 30%.

    Parameters
    ----------
    spread_pct:
        Current bid-ask spread as a percentage of mid-price.
    max_spread:
        Base maximum acceptable spread percentage (from channel config).
    regime:
        Market regime string. Accepted: "VOLATILE", "QUIET", "TRENDING_UP",
        "TRENDING_DOWN", "RANGING".
    atr_pct:
        ATR as a percentage of price (optional, for fine-grained scaling).
    """
    if not regime:
        return spread_pct <= max_spread

    if regime == "VOLATILE":
        # Allow up to 50% wider spreads in volatile conditions
        adjusted = max_spread * 1.5
    elif regime == "QUIET":
        # Tighten by 30% — small moves mean spread eats more of the edge
        adjusted = max_spread * 0.7
    elif regime in ("TRENDING_UP", "TRENDING_DOWN"):
        # Slight relaxation — trending markets have slightly wider spreads
        adjusted = max_spread * 1.2
    else:
        # RANGING or unknown — use base
        adjusted = max_spread

    # Additional ATR-based scaling: if ATR% is very high, allow proportionally wider spread
    if atr_pct > 1.0:
        atr_bonus = min(atr_pct / 5.0, 0.5)  # Cap at +50% additional relaxation
        adjusted *= (1.0 + atr_bonus)

    return spread_pct <= adjusted


def check_ema_alignment_regime(
    ema_fast: float | None,
    ema_slow: float | None,
    direction: str,
    regime: str = "",
) -> bool:
    """Regime-aware EMA alignment check.

    In RANGING and QUIET regimes, EMA alignment is relaxed (always passes)
    because mean-reversion setups don't require trend alignment.
    In VOLATILE regime, requires moderate alignment (gap > 0.05% of slow EMA).
    In TRENDING regimes, requires strict alignment (standard binary check).
    Falls back to standard check when regime is unknown.
    """
    mode = _EMA_MODE_BY_REGIME.get(regime, "STRICT") if regime else "STRICT"

    if mode == "RELAXED":
        return True  # Range/quiet regimes don't need EMA alignment

    if mode == "MODERATE":
        # Require EMAs to exist but allow smaller gap
        if ema_fast is None or ema_slow is None:
            return False
        if ema_slow == 0:
            return False
        gap_pct = abs(ema_fast - ema_slow) / ema_slow * 100.0
        if gap_pct < 0.05:
            return True  # Very close EMAs are acceptable in volatile regime
        # If there's a meaningful gap, check it's in the right direction
        return check_ema_alignment(ema_fast, ema_slow, direction)

    # STRICT: standard binary check
    return check_ema_alignment(ema_fast, ema_slow, direction)
