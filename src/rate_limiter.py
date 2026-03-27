"""Async Binance API rate limiter with weight tracking.

Provides :class:`RateLimiter` which:

- Tracks consumed API weight against Binance's rolling 60-second window.
- Exposes :meth:`~RateLimiter.acquire` which suspends the caller until
  sufficient budget is available so no request ever exceeds the limit.
- Syncs the authoritative weight counter from the ``X-MBX-USED-WEIGHT-1m``
  response header via :meth:`~RateLimiter.update_from_header`.
- Supports dynamic budget adjustment via :meth:`~RateLimiter.set_budget`.
- Exposes :attr:`~RateLimiter.is_tier3_paused` and
  :attr:`~RateLimiter.is_tier2_paused` for tier-based preemptive throttling.

Two module-level singletons are exported: :data:`spot_rate_limiter` and
:data:`futures_rate_limiter`.  Binance tracks spot and futures rate limits
independently, so using separate limiters avoids being overly conservative.
The legacy :data:`rate_limiter` alias points at :data:`spot_rate_limiter`
for backward compatibility.

Safety targets
--------------
- Spot: budget 5,000/min out of Binance's 6,000/min Spot cap.
  Leaves ~1,000 weight headroom for WebSocket reconnects and ad-hoc calls.
- Futures: budget 2,000/min out of Binance's 2,400/min Futures cap.
  Leaves ~400 weight headroom for reconnects and ad-hoc requests.
- Burst protection: proactive micro-sleep when remaining weight falls below
  ``_BURST_PROTECTION_THRESHOLD`` (15% of budget) to prevent the engine
  from burning through the last units in a single burst and triggering the
  hard Binance 429 lockout (observed as ~42 s pause at 100% usage).
- Tier throttling: :attr:`~RateLimiter.is_tier3_paused` signals at 70%
  budget usage; :attr:`~RateLimiter.is_tier2_paused` signals at 85%.
  The scanner reads these flags to skip non-essential Tier 3/2 REST calls
  well before hitting the hard cap.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from src.utils import get_logger

log = get_logger("rate_limiter")

# Binance rolling window duration in seconds
_WEIGHT_WINDOW_S: float = 60.0

# Default budget for Spot: 5,000 out of Binance's 6,000/min Spot limit.
# The remaining ~1,000 units are reserved for WebSocket reconnects, ad-hoc
# exchange-info calls, and any other requests that bypass the main scan path.
_DEFAULT_BUDGET: int = 5_000

# Default budget for Futures: 2,000 out of Binance's 2,400/min Futures limit.
# The remaining ~400 units are reserved for reconnects and ad-hoc requests.
_DEFAULT_FUTURES_BUDGET: int = 2_000

# Warn when usage reaches this fraction of the budget
_WARN_THRESHOLD: float = 0.80

# Tier-based preemptive throttling thresholds.
# When the consumed weight fraction exceeds these levels the corresponding
# ``is_tier*_paused`` property returns ``True`` so the scanner can skip
# non-essential REST calls for lower-priority pairs *before* a hard Binance
# 429 lockout is triggered.
#   > 70% → pause Tier 3 (Cold) non-essential requests
#   > 85% → pause Tier 2 (Warm) non-essential requests
# Tier 1 (Hot) pairs and critical trade execution are never artificially
# paused by these thresholds.
_TIER3_PAUSE_THRESHOLD: float = 0.70
_TIER2_PAUSE_THRESHOLD: float = 0.85

# Burst protection: when remaining budget falls below this fraction of the
# total budget, a proportional micro-sleep is injected *before* consuming
# the next weight unit.  This smooths out request bursts and prevents the
# engine from burning through the remaining headroom in a single cycle,
# which was causing the hard Binance 429 lockout (42 s pause at 100% usage).
# The threshold of 15% means throttling kicks in at:
#   Spot:    750 / 5,000  remaining  (Binance hard cap 6,000)
#   Futures: 300 / 2,000  remaining  (Binance hard cap 2,400)
_BURST_PROTECTION_THRESHOLD: float = 0.15
# Maximum micro-sleep (seconds) injected when the budget is almost zero.
# As remaining approaches 0% the sleep grows linearly toward this cap.
_BURST_PROTECTION_MAX_SLEEP_S: float = 0.5
# Minimum micro-sleep to actually issue (avoids an asyncio.sleep(0.001)
# no-op while still suppressing a log line for trivially small delays).
_BURST_PROTECTION_MIN_SLEEP_S: float = 0.01


class RateLimiter:
    """Asyncio-safe token-bucket rate limiter for Binance API weight.

    Parameters
    ----------
    budget:
        Maximum weight allowed per 60-second window.  Defaults to 5,000 for
        Spot (Binance hard cap: 6,000/min) and 2,000 for Futures (hard cap:
        2,400/min), leaving headroom for reconnects and ad-hoc requests.
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

    @property
    def is_tier3_paused(self) -> bool:
        """True when API weight usage exceeds 70% of the budget.

        When this is ``True``, the scanner should skip non-essential REST
        requests for Tier 3 (Cold) pairs to prevent exhausting the budget
        and triggering the hard Binance 429/418 lockout.
        """
        self._maybe_reset()
        if self._budget <= 0:
            return False
        return self._used / self._budget >= _TIER3_PAUSE_THRESHOLD

    @property
    def is_tier2_paused(self) -> bool:
        """True when API weight usage exceeds 85% of the budget.

        When this is ``True``, the scanner should skip non-essential REST
        requests for Tier 2 (Warm) pairs in addition to Tier 3.  Only
        Tier 1 (Hot) pairs and critical trade-execution commands continue
        to be processed.
        """
        self._maybe_reset()
        if self._budget <= 0:
            return False
        return self._used / self._budget >= _TIER2_PAUSE_THRESHOLD

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def acquire(self, weight: int = 1) -> None:
        """Wait until *weight* units can be consumed without exceeding budget.

        If the remaining budget is insufficient, the coroutine suspends until
        the current window resets, then records the weight and returns.

        When the remaining budget is below ``_BURST_PROTECTION_THRESHOLD``
        (15% of the budget) a proportional micro-sleep is injected to smooth
        out request bursts *before* they exhaust the window.  This prevents
        the hard Binance 42 s lockout that occurs when the engine consumes
        the last few hundred weight units in rapid succession.

        The required sleep duration is calculated and state is updated inside
        the lock; ``asyncio.sleep()`` is then called *outside* the lock so
        that other coroutines waiting to acquire the lock are not blocked
        while this coroutine is sleeping.

        Parameters
        ----------
        weight:
            Estimated Binance request weight of the upcoming API call.
        """
        sleep_s = 0.0
        async with self._lock:
            self._maybe_reset()
            if self._used + weight > self._budget:
                elapsed = time.monotonic() - self._window_start
                sleep_s = max(0.0, self._window_s - elapsed)
                log.warning(
                    "Rate limiter budget exhausted "
                    "(used=%d, budget=%d, weight=%d) – pausing %.1fs",
                    self._used, self._budget, weight, sleep_s,
                )
                self._reset()
            else:
                # Proactive burst protection: add a micro-sleep proportional
                # to how close the budget already is to exhaustion *before*
                # this request is consumed.  Checking the pre-consume remaining
                # avoids false triggers on the first call in a fresh window.
                remaining_before = self._budget - self._used
                frac_remaining = remaining_before / self._budget
                if frac_remaining < _BURST_PROTECTION_THRESHOLD:
                    # Linear scale: 0 → 0 s at threshold, full cap at 0 remaining
                    burst_sleep = _BURST_PROTECTION_MAX_SLEEP_S * (
                        1.0 - frac_remaining / _BURST_PROTECTION_THRESHOLD
                    )
                    if burst_sleep > _BURST_PROTECTION_MIN_SLEEP_S:
                        log.debug(
                            "Burst protection: %.0f%% budget remaining — sleeping %.2fs",
                            frac_remaining * 100,
                            burst_sleep,
                        )
                        sleep_s = burst_sleep
            self._used += weight
            pct = self._used / self._budget * 100
            if pct >= _WARN_THRESHOLD * 100:
                log.warning(
                    "Binance API weight usage at %.0f%% (%d/%d)",
                    pct, self._used, self._budget,
                )
        # Sleep outside the lock so other coroutines are not blocked while
        # this task is waiting for the window to reset or burst protection.
        if sleep_s > 0.0:
            await asyncio.sleep(sleep_s)

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

    def set_budget(self, budget: int) -> None:
        """Dynamically adjust the weight budget (e.g. for boot vs steady-state).

        Parameters
        ----------
        budget:
            New maximum weight allowed per 60-second window.
        """
        self._budget = budget
        log.info("Rate limiter budget set to %d", budget)

    def _reset(self) -> None:
        self._used = 0
        self._window_start = time.monotonic()
        log.debug("Rate limiter window reset")


# ---------------------------------------------------------------------------
# Module-level singletons — one per Binance rate-limit domain.
# Binance tracks spot and futures request weight independently, so using
# separate limiters avoids sharing the budget unnecessarily.
# Spot: 5,000/min budget (Binance hard cap: 6,000/min).
# Futures: 2,000/min budget (Binance hard cap: 2,400/min).
# ---------------------------------------------------------------------------
spot_rate_limiter: RateLimiter = RateLimiter(budget=_DEFAULT_BUDGET)
futures_rate_limiter: RateLimiter = RateLimiter(budget=_DEFAULT_FUTURES_BUDGET)

# Backward-compatible alias — existing code importing `rate_limiter` will
# continue to work and will be throttled against the spot budget.
rate_limiter = spot_rate_limiter
