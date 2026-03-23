"""Adaptive Confidence Decay.

Penalises signals that were generated from data which is becoming stale
relative to the time a human (or downstream system) would act on them.

A signal generated 60 seconds ago on the high-frequency ``360_SCALP``
channel is already borderline stale; the same signal on ``360_SWING``
is still perfectly fresh.  This module applies a channel-appropriate
linear decay factor plus a hard penalty for very old signals.

Typical usage
-------------
.. code-block:: python

    import time
    from src.confidence_decay import apply_confidence_decay

    t0 = time.monotonic()
    # … some work …
    new_conf = apply_confidence_decay(
        confidence=72.5,
        signal_generated_at=t0,
        current_time=time.monotonic(),
        channel="360_SCALP",
    )
"""

from __future__ import annotations

from src.utils import get_logger

log = get_logger("confidence_decay")

# ---------------------------------------------------------------------------
# Per-channel freshness windows (seconds)
# ---------------------------------------------------------------------------

#: Maximum age (seconds) before linear decay begins to reduce confidence.
#: Beyond this window the signal is considered stale.
_MAX_FRESHNESS: dict[str, float] = {
    "360_SCALP":    60.0,
    "360_SWING":   600.0,
    "360_SPOT":   3600.0,
}

#: Default freshness window used when the channel is not found in the table.
_DEFAULT_MAX_FRESHNESS: float = 120.0

#: Fraction of confidence removed per "full freshness window" elapsed.
#: E.g. 0.15 → confidence is reduced by at most 15% once age == max_freshness.
_DECAY_RATE: float = 0.15

#: Hard-penalty multiplier applied when age > 2× max_freshness.
_HARD_PENALTY_MULTIPLIER: float = 0.70


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def apply_confidence_decay(
    confidence: float,
    signal_generated_at: float,
    current_time: float,
    channel: str,
) -> float:
    """Return *confidence* reduced by an age-based decay factor.

    Parameters
    ----------
    confidence:
        Raw confidence value (0–100).
    signal_generated_at:
        ``time.monotonic()`` timestamp captured when the signal was first
        created (at the start of ``_prepare_signal``).
    current_time:
        ``time.monotonic()`` timestamp for "now".
    channel:
        Channel name (e.g. ``"360_SCALP"``).  Used to look up the
        per-channel freshness window.

    Returns
    -------
    float
        Decayed confidence, clamped to ``[0.0, 100.0]``.
    """
    age_seconds = max(0.0, current_time - signal_generated_at)
    max_freshness = _MAX_FRESHNESS.get(channel, _DEFAULT_MAX_FRESHNESS)

    # Very stale: apply hard penalty instead of the smooth decay formula.
    if age_seconds > 2.0 * max_freshness:
        decayed = confidence * _HARD_PENALTY_MULTIPLIER
        log.debug(
            "Confidence decay (hard penalty) for channel {}: age={:.1f}s → {:.1f} → {:.1f}",
            channel, age_seconds, confidence, decayed,
        )
        return max(0.0, min(100.0, decayed))

    # Linear decay: factor shrinks from 1.0 toward (1.0 - _DECAY_RATE) as
    # age approaches max_freshness, then keeps decaying below that for older signals.
    decay_factor = max(0.0, 1.0 - (age_seconds / max_freshness) * _DECAY_RATE)
    decayed = confidence * decay_factor

    log.debug(
        "Confidence decay for channel {}: age={:.1f}s / {:.0f}s max → factor={:.3f} → {:.1f}",
        channel, age_seconds, max_freshness, decay_factor, decayed,
    )
    return max(0.0, min(100.0, decayed))
