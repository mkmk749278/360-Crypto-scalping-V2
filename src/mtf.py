"""Multi-Timeframe (MTF) Confluence Matrix.

Evaluates trend alignment across multiple timeframes (e.g. 1m, 15m, 1h).
A lower-timeframe signal is scored based on whether it aligns with the
higher-timeframe EMA/trend direction.

Typical usage
-------------
.. code-block:: python

    from src.mtf import compute_mtf_confluence, MTFResult

    # Each timeframe entry is a dict of {"ema_fast": float, "ema_slow": float,
    # "close": float}.  Timeframes should be ordered from lowest to highest.
    timeframes = {
        "1m":  {"ema_fast": 101.0, "ema_slow": 100.0, "close": 101.5},
        "15m": {"ema_fast": 102.0, "ema_slow": 101.0, "close": 102.0},
        "1h":  {"ema_fast": 103.0, "ema_slow": 101.5, "close": 103.5},
    }
    result = compute_mtf_confluence("LONG", timeframes)
    if result.is_aligned:
        print(f"All TFs aligned  score={result.score:.2f}")
    else:
        print(f"Misaligned: {result.reason}")

The module is **pure-function** – no I/O, no side-effects.  Wire it into
the signal validation pipeline after indicator calculations are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils import get_logger

log = get_logger("mtf")

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Minimum confluence score (0-1) required to pass the MTF gate.
#: 0.5 means at least half the supplied timeframes must agree.
MTF_MIN_SCORE: float = 0.5

#: Score threshold above which the confluence is considered *strong*.
MTF_STRONG_SCORE: float = 0.8

#: Timeframe-proportional weights for MTF confluence scoring.
#: Higher timeframes carry more weight (institutional significance).
#: Timeframes not in this dict default to 1.0.
_TF_WEIGHTS: dict[str, float] = {
    "1m":  0.5,
    "5m":  1.0,
    "15m": 1.5,
    "1h":  2.0,
    "4h":  3.0,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeframeState:
    """Trend state derived for a single timeframe."""

    timeframe: str
    trend: str          # "BULLISH" | "BEARISH" | "NEUTRAL"
    ema_fast: float
    ema_slow: float
    close: float


@dataclass
class MTFResult:
    """Output of :func:`compute_mtf_confluence`."""

    signal_direction: str               # "LONG" | "SHORT"
    score: float                        # 0.0 – 1.0  (aligned TFs / total TFs)
    aligned_count: float                # weighted TF alignment (includes 0.5 for NEUTRAL)
    total_count: int                    # total TFs evaluated
    is_aligned: bool                    # score >= MTF_MIN_SCORE
    is_strong: bool                     # score >= MTF_STRONG_SCORE
    timeframe_states: List[TimeframeState] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_trend(ema_fast: float, ema_slow: float, close: float) -> str:
    """Return "BULLISH", "BEARISH", or "NEUTRAL" for one timeframe."""
    if ema_fast > ema_slow and close > ema_fast:
        return "BULLISH"
    if ema_fast < ema_slow and close < ema_fast:
        return "BEARISH"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------


def compute_mtf_confluence(
    signal_direction: str,
    timeframes: Dict[str, Dict[str, float]],
    min_score: float = MTF_MIN_SCORE,
    tf_weight_overrides: Optional[Dict[str, float]] = None,
) -> MTFResult:
    """Evaluate trend alignment across multiple timeframes.

    Parameters
    ----------
    signal_direction:
        ``"LONG"`` or ``"SHORT"``.
    timeframes:
        Mapping of timeframe label → indicator dict.  Each dict **must**
        contain the keys ``"ema_fast"``, ``"ema_slow"``, and ``"close"``.
        Missing or malformed entries are skipped and logged.
    min_score:
        Minimum fraction of timeframes that must agree with the signal
        direction to be considered aligned.  Defaults to
        :data:`MTF_MIN_SCORE`.
    tf_weight_overrides:
        Optional mapping of timeframe label → weight override.  When
        provided, these weights replace the defaults from :data:`_TF_WEIGHTS`
        for the specified timeframes.  Use this to apply regime-specific
        higher/lower TF weight multipliers without modifying the global table.

    Returns
    -------
    :class:`MTFResult`
    """
    direction = signal_direction.upper()
    states: List[TimeframeState] = []
    aligned: float = 0.0
    weighted_total: float = 0.0

    for tf_label, data in timeframes.items():
        try:
            ema_fast = float(data["ema_fast"])
            ema_slow = float(data["ema_slow"])
            close = float(data["close"])
        except (KeyError, TypeError, ValueError) as exc:
            log.debug("MTF: skipping timeframe {} – bad data: {}", tf_label, exc)
            continue

        trend = _classify_trend(ema_fast, ema_slow, close)
        states.append(TimeframeState(
            timeframe=tf_label,
            trend=trend,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            close=close,
        ))

        weight = (
            tf_weight_overrides[tf_label]
            if tf_weight_overrides and tf_label in tf_weight_overrides
            else _TF_WEIGHTS.get(tf_label, 1.0)
        )
        weighted_total += weight

        wanted = "BULLISH" if direction == "LONG" else "BEARISH"
        if trend == wanted:
            aligned += weight
        elif trend == "NEUTRAL":
            aligned += weight * 0.5  # Partial credit — not opposing the direction

    total = len(states)
    if total == 0:
        return MTFResult(
            signal_direction=direction,
            score=0.0,
            aligned_count=0,
            total_count=0,
            is_aligned=False,
            is_strong=False,
            timeframe_states=states,
            reason="no valid timeframe data provided",
        )

    score = aligned / weighted_total if weighted_total > 0 else 0.0
    is_aligned = score >= min_score
    is_strong = score >= MTF_STRONG_SCORE

    misaligned = [s.timeframe for s in states if s.trend != ("BULLISH" if direction == "LONG" else "BEARISH")]
    reason = ""
    if not is_aligned:
        reason = (
            f"MTF misaligned: {aligned}/{total} TFs agree with {direction}; "
            f"conflicting TFs: {misaligned}"
        )

    return MTFResult(
        signal_direction=direction,
        score=round(score, 4),
        aligned_count=aligned,
        total_count=total,
        is_aligned=is_aligned,
        is_strong=is_strong,
        timeframe_states=states,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Channel-specific gate functions (PR_06)
# ---------------------------------------------------------------------------


def check_mtf_ema_alignment(
    higher_tf_indicators: dict,
    direction: str,
    strict: bool = True,
) -> tuple[bool, str, float]:
    """Check that EMA alignment on a higher timeframe supports the trade direction.

    Parameters
    ----------
    higher_tf_indicators:
        Indicator dict for the confirmation timeframe (e.g. 1h for a 5m scalp).
    direction:
        "LONG" or "SHORT".
    strict:
        When True, return (False, ...) on failure; else apply -10 pts penalty.
    """
    ema_fast = higher_tf_indicators.get("ema9_last")
    ema_slow = higher_tf_indicators.get("ema21_last")
    ema200 = higher_tf_indicators.get("ema200_last")

    if ema_fast is None or ema_slow is None:
        return True, "mtf_ema_no_data", 0.0   # Fail open on missing data

    aligned = (ema_fast > ema_slow) if direction == "LONG" else (ema_fast < ema_slow)

    # If EMA200 available, add extra confirmation weight
    if ema200 is not None:
        price_above_ema200 = ema_fast > ema200
        if direction == "LONG" and not price_above_ema200:
            aligned = False
        if direction == "SHORT" and price_above_ema200:
            aligned = False

    if aligned:
        return True, "mtf_ema_aligned", 0.0
    if strict:
        return False, "mtf_ema_opposed", 0.0
    return True, "mtf_ema_soft_fail", -10.0


def check_mtf_rsi(
    higher_tf_indicators: dict,
    direction: str,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> tuple[bool, str, float]:
    """Check RSI on higher timeframe is not in an extreme zone opposing the signal."""
    rsi_val = higher_tf_indicators.get("rsi_last")
    if rsi_val is None:
        return True, "mtf_rsi_no_data", 0.0
    if direction == "LONG" and rsi_val >= overbought:
        return False, f"mtf_rsi_overbought_{rsi_val:.1f}", 0.0
    if direction == "SHORT" and rsi_val <= oversold:
        return False, f"mtf_rsi_oversold_{rsi_val:.1f}", 0.0
    return True, "mtf_rsi_ok", 0.0


def check_mtf_adx(
    higher_tf_indicators: dict,
    min_adx: float = 20.0,
    max_adx: float = 65.0,
) -> tuple[bool, str, float]:
    """Check ADX on higher timeframe is within [min_adx, max_adx]."""
    adx_val = higher_tf_indicators.get("adx_last")
    if adx_val is None:
        return True, "mtf_adx_no_data", 0.0
    if adx_val < min_adx:
        return False, f"mtf_adx_weak_{adx_val:.1f}", 0.0
    if adx_val > max_adx:
        return False, f"mtf_adx_extreme_{adx_val:.1f}", 0.0
    return True, "mtf_adx_ok", 0.0


def mtf_gate_scalp_standard(
    indicators_1h: dict,
    direction: str,
    regime: str = "",
) -> tuple[bool, str, float]:
    """MTF gate for the SCALP standard path (5m signal → 1h confirmation).

    Strict in TRENDING/VOLATILE; soft penalty in RANGING/QUIET.
    Passes if EMA alignment OR RSI non-extreme on 1h.
    """
    strict = regime.upper() in ("TRENDING_UP", "TRENDING_DOWN", "VOLATILE")
    ema_ok, ema_reason, ema_adj = check_mtf_ema_alignment(indicators_1h, direction, strict=False)
    rsi_ok, rsi_reason, _ = check_mtf_rsi(indicators_1h, direction)

    # ema_adj < 0 indicates genuine misalignment (soft-fail path in check_mtf_ema_alignment);
    # ema_adj == 0 means either aligned or no-data (both treated as "ok" for this gate).
    ema_actually_ok = ema_adj == 0.0

    if ema_actually_ok and rsi_ok:
        return True, f"{ema_reason}+{rsi_reason}", 0.0
    if not ema_actually_ok and not rsi_ok:
        if strict:
            return False, f"{ema_reason}+{rsi_reason}", 0.0
        return True, "mtf_both_soft_fail", -10.0
    # One passes — partial confirmation
    return True, f"mtf_partial_{ema_reason}_{rsi_reason}", -5.0


def mtf_gate_scalp_range_fade(
    indicators_15m: dict,
    direction: str,
) -> tuple[bool, str, float]:
    """MTF gate for RANGE_FADE path (5m signal → 15m RSI confirmation)."""
    rsi_val = indicators_15m.get("rsi_last")
    if rsi_val is None:
        return True, "mtf_15m_rsi_no_data", 0.0
    if direction == "LONG" and rsi_val > 45.0:
        return False, f"mtf_15m_rsi_not_oversold_{rsi_val:.1f}", 0.0
    if direction == "SHORT" and rsi_val < 55.0:
        return False, f"mtf_15m_rsi_not_overbought_{rsi_val:.1f}", 0.0
    return True, "mtf_15m_rsi_ok", 0.0


def mtf_gate_swing(
    indicators_4h: dict,
    direction: str,
) -> tuple[bool, str, float]:
    """MTF gate for SWING (1h signal → 4h EMA + ADX confirmation)."""
    ema_ok, ema_reason, ema_adj = check_mtf_ema_alignment(indicators_4h, direction, strict=True)
    adx_ok, adx_reason, _ = check_mtf_adx(indicators_4h, min_adx=18.0, max_adx=70.0)
    if ema_ok and adx_ok:
        return True, f"{ema_reason}+{adx_reason}", 0.0
    if not ema_ok:
        return False, ema_reason, 0.0
    # EMA ok but ADX fails — soft penalty
    return True, adx_reason, -5.0


def check_mtf_gate(
    signal_direction: str,
    timeframes: Dict[str, Dict[str, float]],
    min_score: float = MTF_MIN_SCORE,
    tf_weight_overrides: Optional[Dict[str, float]] = None,
) -> tuple[bool, str]:
    """Pipeline hook: return ``(allowed, reason)`` for the MTF confluence gate.

    Fails open (returns ``True``) when no valid timeframe data is provided,
    matching the behaviour of the order book and CVD filters.

    Parameters
    ----------
    signal_direction:
        ``"LONG"`` or ``"SHORT"``.
    timeframes:
        Same format as :func:`compute_mtf_confluence`.
    min_score:
        Minimum passing score.
    tf_weight_overrides:
        Optional per-timeframe weight overrides passed through to
        :func:`compute_mtf_confluence`.  Allows callers (e.g. the scanner's
        regime-weighted MTF gate) to adjust how much each timeframe
        contributes without modifying the global weight table.

    Returns
    -------
    ``(allowed, reason)`` – ``allowed`` is ``False`` only when sufficient
    data exists *and* the confluence score falls below *min_score*.
    """
    if not timeframes:
        return True, ""

    result = compute_mtf_confluence(signal_direction, timeframes, min_score, tf_weight_overrides)
    if result.total_count == 0:
        return True, ""

    if not result.is_aligned:
        return False, result.reason

    return True, ""
