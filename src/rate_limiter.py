"""Async Binance API rate limiter with weight tracking.

Provides :class:`RateLimiter` which:

- Tracks consumed API weight against Binance's rolling 60-second window.
- Exposes :meth:`~RateLimiter.acquire` which suspends the caller until
  sufficient budget is available so no request ever exceeds the limit.
- Syncs the authoritative weight counter from the ``X-MBX-USED-WEIGHT-1m``
  response header via :meth:`~RateLimiter.update_from_header`.

A module-level singleton :data:`rate_limiter` is exported and shared by all
:class:`~src.binance.BinanceClient` instances.

Safety targets
--------------
- Normal operation: ~400 weight/min (well under the 1 200/min Binance cap).
- Burst protection: auto-pause when remaining weight < ``_PAUSE_THRESHOLD``.
- Leave ~200 weight/min headroom for WebSocket reconnects and ad-hoc requests.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from src.utils import get_logger

log = get_logger("rate_limiter")

# Binance rolling window duration in seconds
_WEIGHT_WINDOW_S: float = 60.0

# Default budget: 1 000 out of Binance's 1 200/min limit.  The remaining
# ~200 units are reserved for WebSocket reconnects, ad-hoc exchange-info
# calls, and any other requests that bypass the main scan path.
_DEFAULT_BUDGET: int = 1_000

# Warn when usage reaches this fraction of the budget
_WARN_THRESHOLD: float = 0.80


class RateLimiter:
    """Asyncio-safe token-bucket rate limiter for Binance API weight.

    Parameters
    ----------
    budget:
        Maximum weight allowed per 60-second window.  Defaults to 1 000,
        leaving ~200 weight headroom below Binance's hard 1 200 limit.
    window_s:
        Length of the rolling window in seconds (default 60, matching Binance).
    """

    def __init__(
        self,
        budget: int = _DEFAULT_BUDGET,
        window_s: float = _WEIGHT_WINDOW_S,
    ) -> None:
        self._budget = budget
        self._window_s = window_s
        self._used: int = 0
        self._window_start: float = time.monotonic()
        # Single lock serialises weight mutations; asyncio.Lock is not
        # thread-safe, which is fine because the entire bot runs in one
        # event loop.
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def budget(self) -> int:
        """Configured weight budget per rolling window."""
        return self._budget

    @property
    def used(self) -> int:
        """Weight consumed so far in the current window (before any reset)."""
        return self._used

    @property
    def remaining(self) -> int:
        """Estimated remaining weight in the current window."""
        self._maybe_reset()
        return max(0, self._budget - self._used)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def acquire(self, weight: int = 1) -> None:
        """Wait until *weight* units can be consumed without exceeding budget.

        If the remaining budget is insufficient, the coroutine suspends until
        the current window resets, then records the weight and returns.  All
        callers share the same lock so requests are serialised when pausing.

        Parameters
        ----------
        weight:
            Estimated Binance request weight of the upcoming API call.
        """
        async with self._lock:
            self._maybe_reset()
            if self._used + weight > self._budget:
                elapsed = time.monotonic() - self._window_start
                wait_s = max(0.0, self._window_s - elapsed)
                log.warning(
                    "Rate limiter budget exhausted "
                    "(used=%d, budget=%d, weight=%d) – pausing %.1fs",
                    self._used, self._budget, weight, wait_s,
                )
                await asyncio.sleep(wait_s)
                self._reset()
            self._used += weight
            pct = self._used / self._budget * 100
            if pct >= _WARN_THRESHOLD * 100:
                log.warning(
                    "Binance API weight usage at %.0f%% (%d/%d)",
                    pct, self._used, self._budget,
                )

    def update_from_header(self, raw_value: Optional[str]) -> None:
        """Sync the local weight counter from ``X-MBX-USED-WEIGHT-1m`` header.

        The server's value is authoritative.  We take the *maximum* of the
        local estimate and the server-reported value so that parallel in-flight
        requests never cause us to drift below reality.

        Parameters
        ----------
        raw_value:
            Raw string value of the header, e.g. ``"42"``.  ``None`` is a
            no-op so callers can pass ``resp.headers.get(...)`` directly.
        """
        if raw_value is None:
            return
        try:
            server_used = int(raw_value)
        except (ValueError, TypeError):
            return
        self._maybe_reset()
        if server_used > self._used:
            self._used = server_used
        pct = self._used / self._budget * 100
        if pct >= _WARN_THRESHOLD * 100:
            log.warning(
                "Binance reports API weight at %.0f%% of budget (%d/%d)",
                pct, self._used, self._budget,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_reset(self) -> None:
        """Reset weight counter if the rolling window has elapsed."""
        if time.monotonic() - self._window_start >= self._window_s:
            self._reset()

    def _reset(self) -> None:
        self._used = 0
        self._window_start = time.monotonic()
        log.debug("Rate limiter window reset")


# ---------------------------------------------------------------------------
# Module-level singleton shared by all BinanceClient instances
# ---------------------------------------------------------------------------
rate_limiter: RateLimiter = RateLimiter()
