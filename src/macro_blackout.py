"""Macro event blackout window filter.

Blocks trading signals during a configurable window before and after major
macro events (e.g. FOMC, CPI releases) to avoid extreme whipsaw conditions.

Usage
-----
.. code-block:: python

    from datetime import datetime, timezone
    from src.macro_blackout import is_in_macro_blackout

    events = [
        {
            "event_time": datetime(2024, 3, 20, 18, 0, tzinfo=timezone.utc),
            "name": "FOMC Rate Decision",
            "severity": "CRITICAL",
        },
    ]
    in_blackout, reason = is_in_macro_blackout(events)
    if in_blackout:
        print(f"Signal blocked: {reason}")

The ``upcoming_events`` list is empty by default (fail-open behaviour).
It can be populated by the MacroWatchdog in a future update.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from config import MACRO_BLACKOUT_POST_MINUTES, MACRO_BLACKOUT_PRE_MINUTES
from src.utils import get_logger

log = get_logger("macro_blackout")

# Severity levels that trigger the blackout gate.
_BLACKOUT_SEVERITIES: frozenset = frozenset({"CRITICAL", "HIGH"})


def is_in_macro_blackout(
    upcoming_events: List[dict],
    now: Optional[datetime] = None,
    pre_minutes: int = MACRO_BLACKOUT_PRE_MINUTES,
    post_minutes: int = MACRO_BLACKOUT_POST_MINUTES,
) -> Tuple[bool, str]:
    """Check whether the current time falls within a macro event blackout window.

    Parameters
    ----------
    upcoming_events:
        List of event dicts.  Each dict must have at minimum:

        * ``event_time``: :class:`datetime` (timezone-aware, UTC preferred).
        * ``name``: Human-readable event name (``str``).
        * ``severity``: One of ``"CRITICAL"``, ``"HIGH"``, ``"MEDIUM"``,
          ``"LOW"`` (``str``).  Only CRITICAL and HIGH events trigger blackout.

    now:
        Optional current UTC datetime for testing.  Defaults to
        :func:`datetime.now(timezone.utc)`.
    pre_minutes:
        Minutes before the event to start blocking signals.  Defaults to
        :data:`config.MACRO_BLACKOUT_PRE_MINUTES` (30 minutes).
    post_minutes:
        Minutes after the event to resume accepting signals.  Defaults to
        :data:`config.MACRO_BLACKOUT_POST_MINUTES` (60 minutes).

    Returns
    -------
    (in_blackout, reason)
        ``in_blackout`` is True if the current time is within a blackout
        window.  ``reason`` describes which event is causing the blackout
        (empty string when not in blackout).
    """
    if not upcoming_events:
        return False, ""

    if now is None:
        now = datetime.now(timezone.utc)

    # Ensure *now* is timezone-aware for comparison.
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    for event in upcoming_events:
        severity = str(event.get("severity", "")).upper()
        if severity not in _BLACKOUT_SEVERITIES:
            continue

        event_time: Optional[datetime] = event.get("event_time")
        if event_time is None:
            continue

        # Normalise event_time to UTC-aware if necessary.
        if isinstance(event_time, datetime):
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
        else:
            # Unsupported type — skip this event.
            continue

        from datetime import timedelta
        window_start = event_time - timedelta(minutes=pre_minutes)
        window_end = event_time + timedelta(minutes=post_minutes)

        if window_start <= now <= window_end:
            name = str(event.get("name", "Unknown Event"))
            phase = "pre-event" if now < event_time else "post-event"
            reason = (
                f"Macro blackout ({phase}): {name} at "
                f"{event_time.strftime('%Y-%m-%d %H:%M UTC')} "
                f"[{severity}]"
            )
            log.debug(reason)
            return True, reason

    return False, ""
