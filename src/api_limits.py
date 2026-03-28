"""Batch scanning & API rate-limit management.

Implements batch scheduling for spot pairs (hourly batches) while
maintaining real-time scanning for top futures.  Tracks Binance
API weight usage to stay within rate limits.

PR 04 Implementation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.utils import get_logger

log = get_logger("api_limits")

# Binance rate limits
BINANCE_WEIGHT_LIMIT_PER_MINUTE: int = 1200
# Safety buffer — stop issuing requests when weight exceeds this
WEIGHT_SAFETY_THRESHOLD: int = int(
    os.getenv("API_WEIGHT_SAFETY_THRESHOLD", "1000")
)
# Number of top futures pairs to scan in real-time every cycle
TOP_FUTURES_REALTIME_COUNT: int = int(
    os.getenv("TOP_FUTURES_REALTIME_COUNT", "100")
)
# Spot batch interval (seconds)
SPOT_BATCH_INTERVAL_SECONDS: int = int(
    os.getenv("SPOT_BATCH_INTERVAL_SECONDS", "3600")
)
# Spot batch size (pairs per batch)
SPOT_BATCH_SIZE: int = int(
    os.getenv("SPOT_BATCH_SIZE", "25")
)


@dataclass
class APIWeightTracker:
    """Tracks Binance API weight consumption with a rolling 1-minute window."""

    _calls: List[tuple] = field(default_factory=list)  # (timestamp, weight)
    _limit: int = BINANCE_WEIGHT_LIMIT_PER_MINUTE
    _safety: int = WEIGHT_SAFETY_THRESHOLD

    def record(self, weight: int = 1) -> None:
        """Record an API call with its weight cost."""
        self._calls.append((time.monotonic(), weight))
        self._prune()

    def _prune(self) -> None:
        """Remove entries older than 60 seconds."""
        cutoff = time.monotonic() - 60.0
        self._calls = [(t, w) for t, w in self._calls if t > cutoff]

    @property
    def current_weight(self) -> int:
        """Total weight consumed in the current 1-minute window."""
        self._prune()
        return sum(w for _, w in self._calls)

    @property
    def remaining_weight(self) -> int:
        """Remaining weight budget before hitting the safety threshold."""
        return max(0, self._safety - self.current_weight)

    @property
    def can_make_request(self) -> bool:
        """True if there is sufficient weight budget for another request."""
        return self.current_weight < self._safety

    @property
    def usage_pct(self) -> float:
        """Weight usage as percentage of limit."""
        return round(self.current_weight / self._limit * 100, 1)

    def calls_last_minute(self) -> int:
        """Number of API calls in the last 60 seconds."""
        self._prune()
        return len(self._calls)


@dataclass
class BatchScheduler:
    """Manages batch scheduling for spot pair scanning.

    Top futures pairs are scanned in real-time every cycle.
    Spot pairs are scanned in hourly batches to conserve API weight.
    """

    _last_spot_batch_time: float = 0.0
    _spot_batch_index: int = 0
    _spot_batch_interval: int = SPOT_BATCH_INTERVAL_SECONDS
    _spot_batch_size: int = SPOT_BATCH_SIZE
    _futures_realtime_count: int = TOP_FUTURES_REALTIME_COUNT

    def get_futures_realtime_pairs(self, all_futures: List[str]) -> List[str]:
        """Return the top N futures pairs for real-time scanning.

        These are scanned every cycle for scalp channel priority.
        """
        return all_futures[: self._futures_realtime_count]

    def should_run_spot_batch(self) -> bool:
        """Check whether it is time to run the next spot batch."""
        now = time.monotonic()
        if now - self._last_spot_batch_time >= self._spot_batch_interval:
            return True
        return False

    def get_spot_batch(self, all_spot: List[str]) -> List[str]:
        """Return the next batch of spot pairs for scanning.

        Rotates through the full spot universe in chunks of
        ``_spot_batch_size`` per invocation.
        """
        if not all_spot:
            return []
        start = self._spot_batch_index * self._spot_batch_size
        end = start + self._spot_batch_size
        batch = all_spot[start:end]
        # Advance index, wrap around when exhausted
        batch_number = self._spot_batch_index
        self._spot_batch_index += 1
        if end >= len(all_spot):
            self._spot_batch_index = 0
        self._last_spot_batch_time = time.monotonic()
        log.info(
            "Spot batch {}: scanning {} pairs (index {}-{})",
            batch_number,
            len(batch),
            start,
            min(end, len(all_spot)) - 1,
        )
        return batch

    def get_scan_pairs(
        self,
        futures_pairs: List[str],
        spot_pairs: List[str],
        force_spot: bool = False,
    ) -> tuple[List[str], List[str]]:
        """Return (futures_to_scan, spot_to_scan) for this cycle.

        Parameters
        ----------
        futures_pairs:
            All futures pairs sorted by priority (volume).
        spot_pairs:
            All spot pairs sorted by priority.
        force_spot:
            When True, includes spot batch regardless of timer.

        Returns
        -------
        tuple[list[str], list[str]]
            Futures pairs (real-time) and spot pairs (batch or empty).
        """
        rt_futures = self.get_futures_realtime_pairs(futures_pairs)
        spot_batch: List[str] = []
        if force_spot or self.should_run_spot_batch():
            spot_batch = self.get_spot_batch(spot_pairs)
        return rt_futures, spot_batch


def check_rate_limit(tracker: APIWeightTracker, required_weight: int = 1) -> bool:
    """Return True if there is enough API budget for the request.

    Parameters
    ----------
    tracker:
        The API weight tracker instance.
    required_weight:
        Weight cost of the planned API call.
    """
    if tracker.current_weight + required_weight > tracker._safety:
        log.warning(
            "API rate limit approaching: {}/{} weight used ({:.1f}%)",
            tracker.current_weight, tracker._limit, tracker.usage_pct,
        )
        return False
    return True
