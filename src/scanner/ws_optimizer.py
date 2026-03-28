"""WebSocket & scan latency optimization helpers.

Provides shard health scoring, adaptive reconnection scheduling,
and latency-aware scan prioritization for the scanner loop.

PR 03 Implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils import get_logger

log = get_logger("ws_optimizer")


@dataclass
class ShardHealth:
    """Health snapshot for a single WebSocket shard (connection)."""

    shard_id: str = ""
    market: str = "spot"               # "spot" or "futures"
    stream_count: int = 0
    last_pong_age_s: float = 0.0       # Seconds since last pong
    ping_latency_ms: float = 0.0       # Last ping/pong RTT (ms)
    message_rate: float = 0.0          # Messages per minute
    reconnect_attempts: int = 0
    degraded: bool = False
    health_score: float = 100.0        # Composite (0-100)


def score_shard_health(
    last_pong_age_s: float,
    ping_latency_ms: float,
    message_rate: float,
    reconnect_attempts: int,
    heartbeat_interval: int = 30,
    min_message_rate: float = 1.0,
) -> float:
    """Compute a composite health score (0-100) for a WS shard.

    Factors
    -------
    - Staleness: how long since last pong relative to heartbeat interval
    - Latency: ping/pong RTT (lower is better)
    - Message rate: messages/min vs minimum threshold
    - Reconnect history: penalise frequent reconnections
    """
    # Staleness score (0-40 points)
    stale_ratio = last_pong_age_s / max(heartbeat_interval * 3, 1)
    staleness_score = max(0.0, 40.0 * (1.0 - min(stale_ratio, 1.0)))

    # Latency score (0-25 points)
    if ping_latency_ms <= 50:
        latency_score = 25.0
    elif ping_latency_ms <= 200:
        latency_score = 25.0 * (1.0 - (ping_latency_ms - 50) / 450)
    elif ping_latency_ms <= 500:
        latency_score = 25.0 * 0.33 * (1.0 - (ping_latency_ms - 200) / 300)
    else:
        latency_score = 0.0

    # Message rate score (0-20 points)
    if message_rate >= min_message_rate * 2:
        msg_score = 20.0
    elif message_rate >= min_message_rate:
        msg_score = 20.0 * (message_rate / (min_message_rate * 2))
    else:
        msg_score = max(0.0, 10.0 * (message_rate / max(min_message_rate, 0.01)))

    # Reconnect penalty (0-15 points)
    reconnect_score = max(0.0, 15.0 - reconnect_attempts * 3.0)

    total = staleness_score + latency_score + msg_score + reconnect_score
    return round(max(0.0, min(100.0, total)), 2)


@dataclass
class LatencyTracker:
    """Tracks scan-cycle latency for adaptive pair prioritization."""

    _history: List[float] = field(default_factory=list)
    _max_history: int = 50
    high_latency_threshold_ms: float = 15_000.0
    critical_latency_threshold_ms: float = 30_000.0

    def record(self, latency_ms: float) -> None:
        """Record a scan cycle latency measurement."""
        self._history.append(latency_ms)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    @property
    def average_ms(self) -> float:
        """Average latency over recent history."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def last_ms(self) -> float:
        """Most recent latency measurement."""
        return self._history[-1] if self._history else 0.0

    @property
    def is_high_latency(self) -> bool:
        """True when recent latency exceeds the warning threshold."""
        return self.last_ms > self.high_latency_threshold_ms

    @property
    def is_critical_latency(self) -> bool:
        """True when recent latency exceeds the critical threshold."""
        return self.last_ms > self.critical_latency_threshold_ms

    def should_skip_low_priority(self) -> bool:
        """Return True if scan should skip low-priority (Tier 2/3) pairs.

        Activated when the last 3 cycles all exceeded the high-latency
        threshold — indicates sustained load rather than a one-off spike.
        """
        if len(self._history) < 3:
            return False
        return all(
            t > self.high_latency_threshold_ms for t in self._history[-3:]
        )

    def get_recommended_pair_limit(self, default_limit: int = 200) -> int:
        """Return a reduced pair limit when latency is elevated."""
        if self.is_critical_latency:
            return max(10, default_limit // 4)
        if self.is_high_latency:
            return max(25, default_limit // 2)
        return default_limit


def compute_reconnect_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_pct: float = 0.25,
) -> float:
    """Compute exponential-backoff reconnection delay with jitter.

    Improves on naive exponential backoff by adding bounded random jitter
    to prevent thundering-herd reconnections across multiple shards.
    """
    import random

    delay = min(base_delay * (2 ** min(attempt, 10)), max_delay)
    jitter = delay * jitter_pct * (2 * random.random() - 1)
    return max(0.1, delay + jitter)


def select_priority_pairs(
    all_pairs: List[str],
    tier1_pairs: List[str],
    latency_tracker: LatencyTracker,
) -> List[str]:
    """Select which pairs to scan based on current latency conditions.

    When latency is high, restricts scanning to Tier 1 (critical) pairs
    only.  When latency is normal, returns all pairs.
    """
    if latency_tracker.is_critical_latency:
        log.warning(
            "Critical latency ({:.0f}ms) — restricting to {} Tier 1 pairs",
            latency_tracker.last_ms, len(tier1_pairs),
        )
        return tier1_pairs
    if latency_tracker.should_skip_low_priority():
        limit = latency_tracker.get_recommended_pair_limit(len(all_pairs))
        selected = all_pairs[:limit]
        log.info(
            "High latency ({:.0f}ms) — reducing scan set from {} to {} pairs",
            latency_tracker.last_ms, len(all_pairs), len(selected),
        )
        return selected
    return all_pairs
