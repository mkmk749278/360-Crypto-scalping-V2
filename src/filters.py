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

    # DEPRECATED: Use check_ema_alignment_adaptive() for ATR-normalised, pair-tier-aware
    # EMA alignment checks. This function is retained for backward compatibility.

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


def check_ema_alignment_adaptive(
    ema_fast: float | None,
    ema_slow: float | None,
    direction: str,
    atr_val: float = 0.0,
    close: float = 0.0,
    regime: str = "",
    pair_tier: str = "MIDCAP",
) -> bool:
    """Return True when EMAs are meaningfully aligned with direction.

    Uses an ATR-normalised buffer zone to prevent signals near EMA crossover
    points where fast ≈ slow. The buffer adapts to pair volatility and regime.

    Parameters
    ----------
    ema_fast, ema_slow:
        Current fast and slow EMA values.
    direction:
        ``"LONG"`` or ``"SHORT"``.
    atr_val:
        Current ATR value (absolute, not percentage).
    close:
        Current close price.
    regime:
        Market regime string.
    pair_tier:
        Pair classification tier: ``"MAJOR"``, ``"MIDCAP"``, or ``"ALTCOIN"``.
    """
    if ema_fast is None or ema_slow is None:
        # In RELAXED regimes (RANGING, QUIET), missing EMAs are acceptable
        regime_upper = regime.upper() if regime else ""
        if regime_upper in ("RANGING", "QUIET"):
            return True
        return False

    if ema_slow == 0:
        return False

    # In RANGING/QUIET regimes, EMA alignment is not required (mean-reversion)
    regime_upper = regime.upper() if regime else ""
    if regime_upper in ("RANGING", "QUIET"):
        return True

    atr_pct = (atr_val / close * 100.0) if close > 0 and atr_val > 0 else 0.3

    # Tier-specific minimum buffer floors
    min_buffer = {"MAJOR": 0.10, "MIDCAP": 0.20, "ALTCOIN": 0.30}.get(pair_tier, 0.20)

    # Regime multipliers for the buffer
    regime_mult = {
        "TRENDING_UP": 0.8, "TRENDING_DOWN": 0.8,  # tighter buffer — trend is clear
        "RANGING": 1.2, "QUIET": 1.2,               # wider buffer — avoid whipsaw
        "VOLATILE": 1.5,                             # widest buffer — EMA crossovers are noisy
    }.get(regime_upper, 1.0)

    buffer_pct = max(min_buffer, atr_pct * regime_mult * 0.5)
    buffer_abs = close * buffer_pct / 100.0 if close > 0 else atr_val * 0.5

    ema_diff = ema_fast - ema_slow
    if direction == "LONG":
        return ema_diff >= buffer_abs   # fast must be meaningfully above slow
    if direction == "SHORT":
        return ema_diff <= -buffer_abs  # fast must be meaningfully below slow
    return False


def check_macd_confirmation(
    histogram_last: float | None,
    histogram_prev: float | None,
    direction: str,
    regime: str = "",
    strict: bool = False,
) -> tuple[bool, float]:
    """Check MACD histogram confirms trade direction.

    Returns (passes: bool, confidence_adjustment: float).
    A negative confidence_adjustment is a soft penalty when the check
    fails in a non-strict regime.

    Parameters
    ----------
    histogram_last:
        Most recent MACD histogram value. None → pass (no data).
    histogram_prev:
        Previous MACD histogram value. None → pass (no data).
    direction:
        "LONG" or "SHORT".
    regime:
        Current market regime string.
    strict:
        When True, return (False, 0.0) on failure instead of applying
        a soft penalty. Used for RANGING/QUIET regimes.
    """
    if histogram_last is None or histogram_prev is None:
        return True, 0.0   # Missing data → fail open

    rising = histogram_last > histogram_prev
    positive = histogram_last > 0.0
    falling = histogram_last < histogram_prev
    negative = histogram_last < 0.0

    if direction == "LONG":
        confirmed = rising or positive
    elif direction == "SHORT":
        confirmed = falling or negative
    else:
        return True, 0.0

    if confirmed:
        return True, 0.0   # Clean confirmation — no penalty

    if strict:
        return False, 0.0  # Hard reject in strict (RANGING/QUIET) mode

    # Soft penalty in permissive (TRENDING/VOLATILE) mode
    return True, -5.0


def check_volume_expansion(
    volumes,
    closes,
    lookback: int = 9,
    multiplier: float = 1.8,
) -> bool:
    """Return True when the most recent candle's USD volume exceeds the lookback average.

    Parameters
    ----------
    volumes:
        Raw volume (unit quantity, not USD) for the last N+1 candles.
    closes:
        Close prices for the last N+1 candles. Used to convert volume to USD.
    lookback:
        Number of prior candles to use for the average (excluding the last one).
    multiplier:
        Required ratio of last candle USD volume to average (e.g. 1.8× = 80% above avg).
    """
    import numpy as np
    v = np.asarray(volumes, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = len(v)
    if n < lookback + 1:
        return False
    usd_vol = v * c
    avg_usd = float(np.mean(usd_vol[-(lookback + 1):-1]))
    last_usd = float(usd_vol[-1])
    if avg_usd <= 0:
        return False
    return last_usd >= avg_usd * multiplier
