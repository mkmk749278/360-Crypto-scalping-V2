"""API Rate-Limit Aware Batch Scheduling Helpers (PR04).

Provides utilities for staying within Binance API weight limits while
scanning spot and futures pairs:

* :class:`BatchScheduler` – spreads spot-pair scans across a configurable
  hourly window to avoid exhausting the 1 200 weight/min Binance limit.
* :func:`should_scan_spot_pair` – quick predicate used by the scanner to
  decide whether a spot pair is due for a scan in the current cycle.
* :func:`log_api_usage` – thin helper that writes a structured API-budget
  log entry after each scan cycle.

Design
------
Top 100 futures pairs are always scanned in **real-time** (every cycle).
Spot pairs are divided into buckets and each bucket is rotated through on
a sub-hourly schedule.  The default window is 60 minutes which gives each
spot pair at least one scan per hour.  Reducing ``window_minutes`` increases
scan frequency at the cost of higher API weight consumption.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

# Default spot-scan window: distribute all spot pairs across this many minutes.
DEFAULT_SPOT_WINDOW_MINUTES: float = 60.0

# Number of top futures pairs to always scan in real-time (every cycle).
TOP_FUTURES_REALTIME_COUNT: int = 100

# Minimum rate-limiter budget percentage (0–100) before a non-priority scan
# batch is deferred to the next cycle.
MIN_BUDGET_PCT_FOR_SPOT: float = 30.0  # ≥ 30 % budget remaining required


class BatchScheduler:
    """Schedules spot-pair scans across a rolling time window.

    Divides the spot pair universe into equal-sized buckets and rotates
    through one bucket per cycle so that the full set is covered within
    *window_minutes* minutes.

    Parameters
    ----------
    window_minutes:
        Target interval (minutes) over which all spot pairs are scanned
        at least once.  Lower values mean more-frequent scans but higher
        API weight usage.
    """

    def __init__(self, window_minutes: float = DEFAULT_SPOT_WINDOW_MINUTES) -> None:
        self._window_seconds: float = window_minutes * 60.0
        self._bucket_index: int = 0
        self._n_buckets: int = 1
        self._last_cycle_time: float = 0.0
        # Track total cycles and skipped cycles for observability
        self._total_cycles: int = 0
        self._skipped_cycles: int = 0

    def assign_buckets(self, pairs: Sequence[str], n_buckets: Optional[int] = None) -> None:
        """Partition *pairs* into *n_buckets* rotation buckets.

        Should be called once when the pair universe is initialised or
        refreshed.  The scheduler resets its rotation index to 0.

        Parameters
        ----------
        pairs:
            All spot pair symbols.
        n_buckets:
            Number of rotation buckets.  When ``None`` the scheduler
            auto-computes a value based on the pair count and window so
            that each bucket is scanned approximately once per window.
        """
        n = len(pairs)
        if n == 0:
            self._n_buckets = 1
            return
        if n_buckets is None:
            # Auto: aim for ~10 pairs per cycle to keep per-cycle weight low
            pairs_per_cycle = max(10, min(50, math.ceil(n / 10)))
            self._n_buckets = max(1, math.ceil(n / pairs_per_cycle))
        else:
            self._n_buckets = max(1, n_buckets)
        self._bucket_index = 0
        log.debug(
            "BatchScheduler assigned %d pairs into %d buckets "
            "(window=%.0f min, ~%d pairs/bucket)",
            n, self._n_buckets, self._window_seconds / 60,
            math.ceil(n / self._n_buckets),
        )

    def get_batch(self, pairs: Sequence[str]) -> List[str]:
        """Return the subset of *pairs* scheduled for the current cycle.

        Advances the internal rotation index on each call.

        Parameters
        ----------
        pairs:
            Full list of spot pair symbols (must match the order used in
            :meth:`assign_buckets` for deterministic bucket assignment).

        Returns
        -------
        list[str]
            The batch of symbols to scan this cycle (may be empty if
            *pairs* is empty).
        """
        self._total_cycles += 1
        n = len(pairs)
        if n == 0 or self._n_buckets <= 0:
            return []

        # Slice the pairs list into n_buckets and return the current bucket.
        bucket_size = math.ceil(n / self._n_buckets)
        start = (self._bucket_index % self._n_buckets) * bucket_size
        batch = list(pairs[start: start + bucket_size])
        self._bucket_index = (self._bucket_index + 1) % self._n_buckets

        log.debug(
            "BatchScheduler: cycle %d → bucket %d/%d (%d pairs)",
            self._total_cycles,
            self._bucket_index,  # already incremented
            self._n_buckets,
            len(batch),
        )
        return batch

    def skip_cycle(self) -> None:
        """Record that the current batch was skipped (budget insufficient)."""
        self._skipped_cycles += 1
        log.warning(
            "BatchScheduler: spot scan skipped (budget low). "
            "Total skipped: %d/%d cycles",
            self._skipped_cycles, self._total_cycles,
        )

    @property
    def stats(self) -> Dict[str, int]:
        """Return a snapshot of scheduler counters for telemetry."""
        return {
            "total_cycles": self._total_cycles,
            "skipped_cycles": self._skipped_cycles,
            "n_buckets": self._n_buckets,
            "current_bucket": self._bucket_index,
        }


def should_scan_spot_pair(
    symbol: str,
    spot_batch: Sequence[str],
) -> bool:
    """Return True when *symbol* is in the current spot scan batch.

    Parameters
    ----------
    symbol:
        Pair symbol to test.
    spot_batch:
        The batch returned by :meth:`BatchScheduler.get_batch` for the
        current scan cycle.
    """
    return symbol in spot_batch


def log_api_usage(
    cycle: int,
    spot_budget_used: int,
    futures_budget_used: int,
    spot_budget_total: int,
    futures_budget_total: int,
    pairs_scanned: int,
    elapsed_ms: float,
) -> None:
    """Write a structured API usage log entry for the completed scan cycle.

    Parameters
    ----------
    cycle:
        Scan cycle counter.
    spot_budget_used / futures_budget_used:
        Binance API weight units consumed so far in the current minute window.
    spot_budget_total / futures_budget_total:
        Total weight budget per minute window (typically 1 200).
    pairs_scanned:
        Number of symbols scanned this cycle.
    elapsed_ms:
        Scan cycle wall-clock time in milliseconds.
    """
    spot_pct = (spot_budget_used / spot_budget_total * 100.0) if spot_budget_total > 0 else 0.0
    fut_pct = (futures_budget_used / futures_budget_total * 100.0) if futures_budget_total > 0 else 0.0
    log.info(
        "API usage [cycle=%d]: spot=%d/%d (%.1f%%) futures=%d/%d (%.1f%%) "
        "pairs=%d latency=%.0fms",
        cycle,
        spot_budget_used, spot_budget_total, spot_pct,
        futures_budget_used, futures_budget_total, fut_pct,
        pairs_scanned, elapsed_ms,
    )
    # Warn when either budget exceeds 70 %
    if spot_pct > 70.0 or fut_pct > 70.0:
        log.warning(
            "API budget high [cycle=%d]: spot=%.1f%% futures=%.1f%% "
            "— consider reducing scan scope",
            cycle, spot_pct, fut_pct,
        )
