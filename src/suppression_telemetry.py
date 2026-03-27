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
REASON_REGIME_PENALTY: str = "regime_penalty"
REASON_PAIR_QUALITY: str = "pair_quality"
REASON_RANGING_ADX: str = "ranging_adx"

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


# ---------------------------------------------------------------------------
# Multi-window analytics
# ---------------------------------------------------------------------------

_ANALYTICS_WINDOWS: Dict[str, float] = {
    "1h":  1 * 3600.0,
    "6h":  6 * 3600.0,
    "24h": 24 * 3600.0,
}


class SuppressionAnalytics:
    """Enhanced suppression analytics with multi-window (1h/6h/24h) tracking.

    Parameters
    ----------
    max_events:
        Maximum total events to retain across all windows.

    Usage::

        analytics = SuppressionAnalytics()
        analytics.record(SuppressionEvent(...))
        summary = analytics.get_suppression_summary("1h")
    """

    def __init__(self, max_events: int = 10000) -> None:
        self._max_events = max_events
        self._trackers: Dict[str, SuppressionTracker] = {
            label: SuppressionTracker(window_seconds=secs)
            for label, secs in _ANALYTICS_WINDOWS.items()
        }
        # Evaluation counters per channel — used to compute suppression rate.
        self._evaluated_by_channel: Dict[str, int] = defaultdict(int)

    def record(self, event: SuppressionEvent, signals_evaluated: int = 0) -> None:
        """Record a suppression event across all time windows."""
        for tracker in self._trackers.values():
            tracker.record(event)
        if signals_evaluated > 0:
            self._evaluated_by_channel[event.channel] += signals_evaluated

    def get_suppression_summary(self, window: str = "24h") -> dict:
        """Return a structured suppression summary for the given window.

        Parameters
        ----------
        window:
            Time window label: ``"1h"``, ``"6h"``, or ``"24h"``.

        Returns
        -------
        dict
            Summary with keys: ``total_suppressed``, ``by_reason``,
            ``by_channel``, ``by_symbol``, ``top_suppressed_pairs``,
            ``suppression_rate_pct``.
        """
        tracker = self._trackers.get(window, self._trackers["24h"])
        total = tracker.total_in_window()
        evaluated = sum(self._evaluated_by_channel.values())
        suppression_rate = (
            total / (total + evaluated) * 100.0 if (total + evaluated) > 0 else 0.0
        )
        return {
            "window": window,
            "total_suppressed": total,
            "by_reason": tracker.summary(),
            "by_channel": tracker.by_channel(),
            "by_symbol": dict(tracker.by_symbol(top_n=20)),
            "top_suppressed_pairs": tracker.by_symbol(top_n=10),
            "suppression_rate_pct": round(suppression_rate, 2),
        }

    def format_report(self, window: str = "24h") -> str:
        """Format a human-readable multi-window suppression report for Telegram."""
        summary = self.get_suppression_summary(window)
        lines = [
            f"📊 *Suppression Analytics — {window} window*",
            f"Total suppressed: *{summary['total_suppressed']}*",
            f"Suppression rate: *{summary['suppression_rate_pct']:.1f}%*",
            "",
        ]
        if summary["by_reason"]:
            lines.append("*By reason:*")
            for reason, count in sorted(summary["by_reason"].items(), key=lambda kv: -kv[1]):
                lines.append(f"  • {reason}: {count}")
            lines.append("")
        if summary["by_channel"]:
            lines.append("*By channel:*")
            for ch, count in sorted(summary["by_channel"].items(), key=lambda kv: -kv[1]):
                lines.append(f"  • {ch}: {count}")
            lines.append("")
        if summary["top_suppressed_pairs"]:
            lines.append("*Top suppressed pairs:*")
            for sym, count in summary["top_suppressed_pairs"][:5]:
                lines.append(f"  • {sym}: {count}")
        return "\n".join(lines)
