"""Suppression Logging Helpers (PR06) – structured suppressed-signal logging.

Provides :func:`log_suppressed_signal` as the single point of entry for
recording any signal that was blocked by a scanner gate, so that the reason,
pair, channel, regime, and probability score are always captured in a
consistent format.

The function writes both to the standard Python logger (for log aggregators)
and to the :class:`~src.suppression_telemetry.SuppressionTracker` singleton
so Telegram ``/suppressed`` digests stay up-to-date.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.suppression_telemetry import SuppressionEvent, SuppressionTracker

log = logging.getLogger(__name__)

# Module-level shared tracker.  The scanner creates its own instance and can
# pass it explicitly; callers that do not have a tracker reference will use this
# fallback so that a single tracker is shared across the process.
_default_tracker: Optional[SuppressionTracker] = None


def get_default_tracker() -> SuppressionTracker:
    """Return (or lazily create) the module-level default tracker."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = SuppressionTracker()
    return _default_tracker


def log_suppressed_signal(
    pair: str,
    channel: str,
    reason: str,
    probability_score: float = 0.0,
    regime: str = "",
    tracker: Optional[SuppressionTracker] = None,
) -> None:
    """Record and log a suppressed signal event.

    Parameters
    ----------
    pair:
        Pair symbol (e.g. ``"BTCUSDT"``).
    channel:
        Channel name (e.g. ``"360_SCALP"``).
    reason:
        Suppression reason constant from
        :mod:`src.suppression_telemetry` (e.g. ``REASON_QUIET_REGIME``).
    probability_score:
        Probability score from :func:`~src.scanner.filter_module.get_pair_probability`
        at the time of suppression (0–100).
    regime:
        Current market regime string (for contextual logging).
    tracker:
        Optional :class:`~src.suppression_telemetry.SuppressionTracker` to
        record into.  When ``None`` the module-level default is used.
    """
    log.debug(
        "Signal suppressed: pair=%s channel=%s reason=%s score=%.1f regime=%s",
        pair, channel, reason, probability_score, regime,
    )

    active_tracker = tracker if tracker is not None else get_default_tracker()
    active_tracker.record(
        SuppressionEvent(
            symbol=pair,
            channel=channel,
            reason=reason,
            regime=regime,
            would_be_confidence=probability_score,
        )
    )
