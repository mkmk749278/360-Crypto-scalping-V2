"""Cross-Timeframe Volume Divergence Gate.

If the primary timeframe shows a volume spike while a higher timeframe
shows declining volume, the move is likely manipulation or noise rather
than genuine momentum.  This gate blocks such signals.

The detection logic:

1. Compute **primary_ratio** = last candle volume / mean of previous 10 candles
   on the *primary_tf*.
2. Find the next higher timeframe in the standard hierarchy
   ``1m < 5m < 15m < 1h < 4h``.
3. Compute **higher_ratio** = same metric on the higher TF.
4. Block when ``primary_ratio > SPIKE_THRESHOLD`` AND
   ``higher_ratio < DECLINE_THRESHOLD``.

The gate **fails open** (returns ``(True, "")``) when data is insufficient
for any required timeframe.

Typical usage
-------------
.. code-block:: python

    from src.volume_divergence import check_volume_divergence_gate

    # candles: tf → {"close": [...], "volume": [...], ...}
    allowed, reason = check_volume_divergence_gate(
        direction="LONG",
        candles=ctx.candles,
        primary_tf="5m",
    )
    if not allowed:
        print(reason)
"""

from __future__ import annotations

from src.utils import get_logger

log = get_logger("volume_divergence")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Ordered timeframe hierarchy (smallest to largest).
_TF_HIERARCHY: list[str] = ["1m", "5m", "15m", "1h", "4h"]

#: Primary TF volume ratio above which a spike is declared.
SPIKE_THRESHOLD: float = 2.0

#: Higher TF volume ratio below which decline is declared.
DECLINE_THRESHOLD: float = 0.7

#: Minimum number of historical candles required to compute a meaningful average.
MIN_CANDLE_HISTORY: int = 11  # 1 last + 10 for average


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _volume_ratio(volume: list) -> float | None:
    """Return ``last_volume / mean(previous 10)`` or ``None`` if insufficient data."""
    if not volume or len(volume) < MIN_CANDLE_HISTORY:
        return None
    last = float(volume[-1])
    prev = [float(v) for v in volume[-MIN_CANDLE_HISTORY:-1]]
    if not prev:
        return None
    avg = sum(prev) / len(prev)
    if avg <= 0:
        return None
    return last / avg


def _next_higher_tf(primary_tf: str) -> str | None:
    """Return the next timeframe above *primary_tf* in the hierarchy, or ``None``."""
    try:
        idx = _TF_HIERARCHY.index(primary_tf)
    except ValueError:
        return None
    if idx + 1 >= len(_TF_HIERARCHY):
        return None
    return _TF_HIERARCHY[idx + 1]


# ---------------------------------------------------------------------------
# Public gate function
# ---------------------------------------------------------------------------


def check_volume_divergence_gate(
    direction: str,  # noqa: ARG001 — direction reserved for future directional divergence
    candles: dict[str, dict],
    primary_tf: str,
) -> tuple[bool, str]:
    """Pipeline gate: detect primary-TF volume spike with higher-TF volume decline.

    Parameters
    ----------
    direction:
        Signal direction (``"LONG"`` / ``"SHORT"``).  Reserved for future
        directional divergence logic; currently not used in the gate decision.
    candles:
        Dict mapping timeframe label → candle dict
        (``{"close": [...], "volume": [...], ...}``).
    primary_tf:
        The primary timeframe to check (e.g. ``"5m"``).

    Returns
    -------
    ``(allowed, reason)``
        ``(False, reason)`` when a volume divergence pattern is detected;
        ``(True, "")`` otherwise.
    """
    higher_tf = _next_higher_tf(primary_tf)
    if higher_tf is None:
        # No higher timeframe available — cannot detect divergence
        return True, ""

    primary_cd = candles.get(primary_tf) or {}
    higher_cd = candles.get(higher_tf) or {}

    primary_vol = primary_cd.get("volume") or []
    higher_vol = higher_cd.get("volume") or []

    p_ratio = _volume_ratio(list(primary_vol))
    h_ratio = _volume_ratio(list(higher_vol))

    if p_ratio is None or h_ratio is None:
        # Insufficient data — fail open
        log.debug(
            "Volume divergence gate: insufficient data for {}/{} (fail open)",
            primary_tf, higher_tf,
        )
        return True, ""

    log.debug(
        "Volume divergence: {}_ratio={:.2f}  {}_ratio={:.2f}",
        primary_tf, p_ratio, higher_tf, h_ratio,
    )

    if p_ratio > SPIKE_THRESHOLD and h_ratio < DECLINE_THRESHOLD:
        return False, (
            f"volume divergence: {primary_tf} spiking {p_ratio:.1f}× "
            f"but {higher_tf} declining {h_ratio:.1f}×"
        )

    return True, ""
