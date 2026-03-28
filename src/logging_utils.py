"""Telemetry & logging enhancements — structured suppression logging.

Provides helper functions for logging suppressed signals with probability
scores, tracking suppression reasons, and monitoring latency spikes.

PR 06 Implementation.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils import get_logger

log = get_logger("logging_utils")


@dataclass
class SuppressionLogEntry:
    """A structured suppression log entry with probability scoring."""

    pair: str
    channel: str
    reason: str
    probability_score: float = 0.0
    regime: str = ""
    threshold: float = 70.0
    timestamp: float = field(default_factory=time.time)

    def format_log(self) -> str:
        """Format as a human-readable log string."""
        return (
            f"SUPPRESSED | {self.pair} | {self.channel} | "
            f"reason={self.reason} | prob={self.probability_score:.1f} | "
            f"threshold={self.threshold:.1f} | regime={self.regime}"
        )


class SuppressionLogger:
    """Enhanced suppression logging with per-reason statistics.

    Tracks suppressed signals with their probability scores and reasons,
    providing aggregated statistics for telemetry dashboards.
    """

    def __init__(self, max_entries: int = 5000) -> None:
        self._entries: List[SuppressionLogEntry] = []
        self._max_entries = max_entries
        self._counts_by_reason: Dict[str, int] = defaultdict(int)
        self._counts_by_channel: Dict[str, int] = defaultdict(int)
        self._counts_by_pair: Dict[str, int] = defaultdict(int)
        self._total_suppressed: int = 0

    def log_suppressed_signal(
        self,
        pair: str,
        channel: str,
        reason: str,
        probability_score: float = 0.0,
        regime: str = "",
        threshold: float = 70.0,
    ) -> None:
        """Log a suppressed signal with probability score.

        Parameters
        ----------
        pair:
            Trading pair symbol (e.g. ``"BTCUSDT"``).
        channel:
            Channel that generated the suppressed signal.
        reason:
            Suppression reason (e.g. ``"regime"``, ``"pair_quality"``).
        probability_score:
            Computed probability score at suppression time.
        regime:
            Current market regime.
        threshold:
            Probability threshold that was not met.
        """
        entry = SuppressionLogEntry(
            pair=pair,
            channel=channel,
            reason=reason,
            probability_score=probability_score,
            regime=regime,
            threshold=threshold,
        )
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        self._counts_by_reason[reason] += 1
        self._counts_by_channel[channel] += 1
        self._counts_by_pair[pair] += 1
        self._total_suppressed += 1

        log.debug(entry.format_log())

    @property
    def total_suppressed(self) -> int:
        """Total number of suppressed signals recorded."""
        return self._total_suppressed

    def get_stats_by_reason(self) -> Dict[str, int]:
        """Return suppression counts grouped by reason."""
        return dict(self._counts_by_reason)

    def get_stats_by_channel(self) -> Dict[str, int]:
        """Return suppression counts grouped by channel."""
        return dict(self._counts_by_channel)

    def get_top_suppressed_pairs(self, limit: int = 10) -> List[tuple]:
        """Return the most frequently suppressed pairs.

        Returns
        -------
        list[tuple[str, int]]
            Pairs sorted by suppression count (descending).
        """
        sorted_pairs = sorted(
            self._counts_by_pair.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_pairs[:limit]

    def get_recent_entries(self, count: int = 20) -> List[SuppressionLogEntry]:
        """Return the most recent suppression log entries."""
        return self._entries[-count:]

    def format_telemetry_summary(self) -> str:
        """Format a summary string for telemetry output."""
        by_reason = self.get_stats_by_reason()
        reasons_str = " | ".join(
            f"{r}={c}" for r, c in sorted(by_reason.items(), key=lambda x: -x[1])
        )
        return (
            f"Suppressions: total={self._total_suppressed} | "
            f"by_reason: {reasons_str}"
        )


@dataclass
class LatencyMonitor:
    """Monitors and logs latency spikes in the scan pipeline."""

    _measurements: List[tuple] = field(default_factory=list)  # (ts, component, ms)
    _max_entries: int = 1000
    warn_threshold_ms: float = 5000.0
    alert_threshold_ms: float = 15000.0

    def record(self, component: str, latency_ms: float) -> None:
        """Record a latency measurement for a pipeline component."""
        self._measurements.append((time.monotonic(), component, latency_ms))
        if len(self._measurements) > self._max_entries:
            self._measurements = self._measurements[-self._max_entries:]

        if latency_ms >= self.alert_threshold_ms:
            log.warning(
                "LATENCY ALERT: {} took {:.0f}ms (threshold: {:.0f}ms)",
                component, latency_ms, self.alert_threshold_ms,
            )
        elif latency_ms >= self.warn_threshold_ms:
            log.info(
                "Latency warning: {} took {:.0f}ms", component, latency_ms,
            )

    def get_average(self, component: str, window_s: float = 300.0) -> float:
        """Average latency for a component over the last ``window_s`` seconds."""
        cutoff = time.monotonic() - window_s
        values = [ms for ts, comp, ms in self._measurements if comp == component and ts > cutoff]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def get_p95(self, component: str, window_s: float = 300.0) -> float:
        """P95 latency for a component over the last ``window_s`` seconds."""
        cutoff = time.monotonic() - window_s
        values = sorted(ms for ts, comp, ms in self._measurements if comp == component and ts > cutoff)
        if not values:
            return 0.0
        idx = int(len(values) * 0.95)
        return values[min(idx, len(values) - 1)]
