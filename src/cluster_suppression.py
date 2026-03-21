"""Signal Clustering Suppression.

When many signals fire across many different pairs within a short window
and the vast majority point in the same direction, the move is almost
certainly market-wide (e.g. a sudden BTC pump/dump) rather than
independent high-quality setups.  Publishing all of them simultaneously
would flood subscribers with correlated signals that share the same risk.

This module provides a thread-safe (asyncio-compatible) :class:`ClusterSuppressor`
that records each emitted signal and blocks new signals while a cluster is
active.

Two blocking conditions are detected:

1. **Directional cluster** — too many signals, > 80% in the same direction.
2. **Undirected cluster** — too many signals regardless of direction (noisy
   market with many false triggers).

Typical usage
-------------
.. code-block:: python

    from src.cluster_suppression import ClusterSuppressor

    suppressor = ClusterSuppressor(window_seconds=60.0, max_signals=5)

    # Before publishing a signal:
    allowed, reason = suppressor.check_cluster_gate("SOLUSDT", "LONG")
    if not allowed:
        return  # drop signal

    # After the signal is accepted by the queue:
    suppressor.record_signal("SOLUSDT", "LONG")
"""

from __future__ import annotations

import time
from collections import deque

from src.utils import get_logger

log = get_logger("cluster_suppression")

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

#: Fraction of same-direction signals required to declare a directional cluster.
_DIRECTION_BIAS_THRESHOLD: float = 0.80

#: Factor applied to *max_signals* for the undirected (any direction) block.
_UNDIRECTED_MULTIPLIER: float = 1.5


# ---------------------------------------------------------------------------
# ClusterSuppressor
# ---------------------------------------------------------------------------


class ClusterSuppressor:
    """Suppress bursts of signals that indicate a market-wide event.

    Parameters
    ----------
    window_seconds:
        Sliding window size.  Signals older than this are evicted.
    max_signals:
        Maximum number of unique symbols allowed before cluster blocking
        kicks in (for the directional check).
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        max_signals: int = 5,
    ) -> None:
        # Entries: (monotonic_time, symbol, direction)
        self._recent: deque[tuple[float, str, str]] = deque()
        self._window = window_seconds
        self._max_signals = max_signals

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_signal(self, symbol: str, direction: str) -> None:
        """Record that a signal for *symbol* / *direction* was accepted.

        Must be called **after** the signal has been successfully enqueued so
        that rejected signals do not inflate the cluster counter.
        """
        now = time.monotonic()
        self._recent.append((now, symbol, direction))
        self._prune(now)
        log.debug(
            "Cluster: recorded {} {} (window size={})",
            symbol, direction, len(self._recent),
        )

    def check_cluster_gate(
        self, symbol: str, direction: str
    ) -> tuple[bool, str]:
        """Return ``(False, reason)`` if the current signal would constitute a cluster.

        Parameters
        ----------
        symbol:
            Trading pair of the candidate signal.
        direction:
            ``"LONG"`` or ``"SHORT"``.

        Returns
        -------
        ``(allowed, reason)``
        """
        now = time.monotonic()
        self._prune(now)

        active = list(self._recent)
        unique_symbols = {entry[1] for entry in active}
        total = len(unique_symbols)

        if total == 0:
            return True, ""

        # ── Undirected burst check ────────────────────────────────────────
        undirected_limit = self._max_signals * _UNDIRECTED_MULTIPLIER
        if total > undirected_limit:
            return False, (
                f"Cluster gate: {total} unique symbols in the last "
                f"{self._window:.0f}s window (limit {undirected_limit:.0f}) – "
                "market-wide noise detected"
            )

        # ── Directional bias check ────────────────────────────────────────
        if total > self._max_signals:
            all_directions = [entry[2].upper() for entry in active]
            dir_upper = direction.upper()
            same_dir = sum(1 for d in all_directions if d == dir_upper)
            bias = same_dir / len(all_directions) if all_directions else 0.0
            if bias >= _DIRECTION_BIAS_THRESHOLD:
                return False, (
                    f"Cluster gate: {total} symbols in {self._window:.0f}s, "
                    f"{bias:.0%} {dir_upper} – market-wide move, not individual setup"
                )

        return True, ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        """Remove entries older than the sliding window."""
        cutoff = now - self._window
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()
