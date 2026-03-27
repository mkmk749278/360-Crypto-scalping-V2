"""Suppression Telemetry — tracks and summarises suppressed signal events.

Provides a rolling-window tracker for signals that were suppressed by any
scanner gate (regime, pair quality, OI, cluster, stat filter, lifespan,
confidence).  The telemetry data enables data-driven threshold tuning and
is exposed via a Telegram ``/suppressed`` admin command.

Typical usage
-------------
.. code-block:: python

    from src.suppression_telemetry import SuppressionTracker, SuppressionEvent, REASON_QUIET_REGIME

    tracker = SuppressionTracker()

    tracker.record(SuppressionEvent(
        symbol="ZECUSDT",
        channel="360_SCALP",
        reason=REASON_QUIET_REGIME,
        regime="QUIET",
        would_be_confidence=68.5,
    ))

    print(tracker.format_telegram_digest())
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

# ---------------------------------------------------------------------------
# Reason constants
# ---------------------------------------------------------------------------

REASON_QUIET_REGIME: str = "quiet_regime"
REASON_SPREAD_GATE: str = "spread_gate"
REASON_VOLUME_GATE: str = "volume_gate"
REASON_OI_INVALIDATION: str = "oi_invalidation"
REASON_CLUSTER: str = "cluster"
REASON_STAT_FILTER: str = "stat_filter"
REASON_LIFESPAN: str = "lifespan"
REASON_CONFIDENCE: str = "confidence"

# Default rolling window (4 hours)
_DEFAULT_WINDOW_SECONDS: float = 4 * 3600.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SuppressionEvent:
    """A single signal-suppression event recorded by the scanner."""

    symbol: str
    channel: str
    reason: str
    regime: str = ""
    would_be_confidence: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# SuppressionTracker
# ---------------------------------------------------------------------------


class SuppressionTracker:
    """Rolling-window tracker for suppressed signal events.

    Parameters
    ----------
    window_seconds:
        How far back to look when computing summaries.  Events older than
        this are discarded on the next :meth:`record` call.  Defaults to
        4 hours.
    """

    def __init__(self, window_seconds: float = _DEFAULT_WINDOW_SECONDS) -> None:
        self._window: float = window_seconds
        self._events: Deque[SuppressionEvent] = deque()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, event: SuppressionEvent) -> None:
        """Record a suppression event and prune stale entries."""
        self._events.append(event)
        self._prune()

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._window
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def total_in_window(self) -> int:
        """Return total suppression events in the current rolling window."""
        self._prune()
        return len(self._events)

    def summary(self) -> Dict[str, int]:
        """Return a dict mapping suppression reason → count within the window."""
        self._prune()
        counts: Dict[str, int] = defaultdict(int)
        for evt in self._events:
            counts[evt.reason] += 1
        return dict(counts)

    def by_channel(self) -> Dict[str, int]:
        """Return suppression counts grouped by channel name."""
        self._prune()
        counts: Dict[str, int] = defaultdict(int)
        for evt in self._events:
            counts[evt.channel] += 1
        return dict(counts)

    def by_symbol(self, top_n: int = 10) -> List[tuple[str, int]]:
        """Return the *top_n* most-suppressed symbols within the window."""
        self._prune()
        counts: Dict[str, int] = defaultdict(int)
        for evt in self._events:
            counts[evt.symbol] += 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    def recent_events(self, limit: int = 20) -> List[SuppressionEvent]:
        """Return the *limit* most recent events (newest first)."""
        self._prune()
        items = list(self._events)
        return items[-limit:][::-1]

    # ------------------------------------------------------------------
    # Telegram digest
    # ------------------------------------------------------------------

    def format_telegram_digest(self, window_hours: Optional[float] = None) -> str:
        """Format a human-readable suppression summary for Telegram.

        Parameters
        ----------
        window_hours:
            Label for the time window shown in the header.  Defaults to
            ``self._window / 3600`` (the tracker's configured window).

        Returns
        -------
        str
            Markdown-formatted digest ready to send via ``send_message``.
        """
        self._prune()
        wh = window_hours if window_hours is not None else self._window / 3600.0
        total = self.total_in_window()

        lines = [
            f"🔕 *Suppressed Signals — last {wh:.0f}h*",
            f"Total suppressed: *{total}*",
            "",
        ]

        reason_counts = self.summary()
        if reason_counts:
            lines.append("*By reason:*")
            _label = {
                REASON_QUIET_REGIME:   "Quiet regime",
                REASON_SPREAD_GATE:    "Spread gate",
                REASON_VOLUME_GATE:    "Volume gate",
                REASON_OI_INVALIDATION: "OI invalidation",
                REASON_CLUSTER:        "Cluster suppression",
                REASON_STAT_FILTER:    "Stat filter",
                REASON_LIFESPAN:       "Min lifespan",
                REASON_CONFIDENCE:     "Confidence gate",
            }
            for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
                label = _label.get(reason, reason)
                lines.append(f"  • {label}: {count}")
            lines.append("")

        channel_counts = self.by_channel()
        if channel_counts:
            lines.append("*By channel:*")
            for ch, count in sorted(channel_counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"  • {ch}: {count}")
            lines.append("")

        top_syms = self.by_symbol(top_n=5)
        if top_syms:
            lines.append("*Top suppressed pairs:*")
            for sym, count in top_syms:
                lines.append(f"  • {sym}: {count}")

        return "\n".join(lines)
