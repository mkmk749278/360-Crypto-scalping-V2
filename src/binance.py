"""Dedicated Binance REST API wrapper.

Provides :class:`BinanceClient` which centralises all Binance REST calls,
tracks request weight, and implements 429/418 retry logic with exponential
back-off so individual modules don't have to duplicate this boilerplate.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from config import (
    BINANCE_FUTURES_REST_BASE,
    BINANCE_REST_BASE,
    DEPTH_CIRCUIT_BREAKER_COOLDOWN,
    DEPTH_CIRCUIT_BREAKER_THRESHOLD,
    DEPTH_MAX_RETRIES,
)
from src.rate_limiter import futures_rate_limiter, spot_rate_limiter
from src.utils import get_logger

log = get_logger("binance_client")

# Binance default rate-limit window (60 s) and request-weight limits.
# Spot hard cap: 6,000 weight/min; Futures hard cap: 2,400 weight/min.
_WEIGHT_WINDOW_S: int = 60
_DEFAULT_WEIGHT_LIMIT: int = 6_000
_DEFAULT_FUTURES_WEIGHT_LIMIT: int = 2_400

# Retry parameters
_MAX_RETRIES: int = 5
_BACKOFF_BASE: float = 1.5  # exponential-backoff base (seconds)

# Default request timeout (seconds).  Depth snapshots use a shorter timeout
# (see _DEPTH_TIMEOUT_S) since they are small payloads and frequent timeouts
# inflate scan latency severely.
_DEFAULT_TIMEOUT_S: float = 8.0
_DEPTH_TIMEOUT_S: float = 5.0

# Depth endpoint paths — used by the per-endpoint circuit breaker.
_DEPTH_PATHS: frozenset = frozenset({"/fapi/v1/depth", "/api/v3/depth"})


class BinanceClient:
    """Async Binance REST client with rate-limit tracking and retry logic.

    Parameters
    ----------
    market:
        ``"spot"`` or ``"futures"``.  Determines which base URL is used.
    """

    # Class-level callback invoked after each successful REST call.
    # Wire this to ``TelemetryCollector.record_api_call`` from ``main.py``.
    on_api_call: Optional[Callable[[], None]] = None

    def __init__(self, market: str = "spot") -> None:
        self.market = market
        self._base_url = (
            BINANCE_FUTURES_REST_BASE if market == "futures" else BINANCE_REST_BASE
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._used_weight: int = 0
        # Use the correct hard cap for this market type.
        self._weight_limit: int = (
            _DEFAULT_FUTURES_WEIGHT_LIMIT if market == "futures" else _DEFAULT_WEIGHT_LIMIT
        )
        self._weight_reset_at: float = time.monotonic() + _WEIGHT_WINDOW_S
        # Select the appropriate rate-limiter for this market.  Binance tracks
        # spot and futures weight limits independently, so each market uses its
        # own budget rather than sharing a single conservative pool.
        self._rate_limiter = (
            futures_rate_limiter if market == "futures" else spot_rate_limiter
        )
        # Per-endpoint depth circuit breaker: tracks consecutive timeouts so
        # that a sustained Binance depth API outage doesn't block scan cycles
        # for 75 s per symbol (5 retries × 15 s timeout each).
        self._depth_consecutive_timeouts: int = 0
        self._depth_circuit_open_until: float = 0.0
        # Semaphore to cap the number of simultaneous in-flight HTTP requests.
        # Without this, hundreds of coroutines can bypass the rate limiter and
        # send requests before the first response header arrives to call
        # update_from_header(), leading to in-flight race conditions.
        self._inflight_sem: asyncio.Semaphore = asyncio.Semaphore(10)

    # ------------------------------------------------------------------
    # Weight tracking
    # ------------------------------------------------------------------

    @property
    def remaining_weight(self) -> int:
        """Estimated remaining request weight in the current window."""
        self._maybe_reset_weight()
        return max(0, self._weight_limit - self._used_weight)

    def _maybe_reset_weight(self) -> None:
        if time.monotonic() >= self._weight_reset_at:
            self._used_weight = 0
            self._weight_reset_at = time.monotonic() + _WEIGHT_WINDOW_S

    def _consume_weight(self, weight: int) -> None:
        self._maybe_reset_weight()
        self._used_weight += weight

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Low-level request helper
    # ------------------------------------------------------------------

    async def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        weight: int = 1,
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute a GET request with retry logic.

        Handles 429 (rate limit) and 418 (IP ban) by waiting and retrying
        with exponential back-off up to ``_MAX_RETRIES`` attempts.

        Parameters
        ----------
        timeout:
            Per-request timeout in seconds.  When ``None``, depth paths use
            ``_DEPTH_TIMEOUT_S`` (5 s) and all other paths use
            ``_DEFAULT_TIMEOUT_S`` (8 s).
        """
        is_depth = path in _DEPTH_PATHS

        # Depth circuit breaker: skip the request entirely if the circuit is
        # open, preventing cumulative timeout delays of 75 s per symbol.
        if is_depth:
            now = time.monotonic()
            if now < self._depth_circuit_open_until:
                remaining = self._depth_circuit_open_until - now
                log.debug(
                    "Depth endpoint circuit breaker open — skipping {} for {:.0f}s",
                    path, remaining,
                )
                return None

        if timeout is None:
            timeout = _DEPTH_TIMEOUT_S if is_depth else _DEFAULT_TIMEOUT_S

        max_retries = DEPTH_MAX_RETRIES if is_depth else _MAX_RETRIES

        session = await self._ensure_session()
        url = self._base_url + path

        # Throttle proactively: wait until the per-market rate-limiter budget
        # has room for this request.  This prevents bursting 200+ requests at
        # once and keeps weight consumption well under Binance's hard cap
        # (6,000/min Spot, 2,400/min Futures).
        await self._rate_limiter.acquire(weight)
        self._consume_weight(weight)

        for attempt in range(max_retries):
            try:
                async with self._inflight_sem:
                    async with session.get(
                        url, params=params, timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # Sync used-weight from the authoritative server header
                            # so our local estimate stays accurate across parallel
                            # requests.  Prefer the 1-minute window header which
                            # Binance always returns on Spot and Futures REST calls.
                            raw_weight = resp.headers.get(
                                "x-mbx-used-weight-1m",
                                resp.headers.get("x-mbx-used-weight"),
                            )
                            self._rate_limiter.update_from_header(raw_weight)
                            if raw_weight is not None:
                                try:
                                    self._used_weight = int(raw_weight)
                                except ValueError:
                                    pass
                            if BinanceClient.on_api_call is not None:
                                BinanceClient.on_api_call()
                            # Reset depth circuit breaker on first successful call.
                            if is_depth and self._depth_consecutive_timeouts > 0:
                                self._depth_consecutive_timeouts = 0
                            return data
                        if resp.status in (429, 418):
                            retry_after = int(resp.headers.get("Retry-After", 5))
                            wait = max(retry_after, _BACKOFF_BASE ** attempt)
                            log.warning(
                                "Binance %s – rate limited (%s). Waiting %.1fs (attempt %d/%d)",
                                path, resp.status, wait, attempt + 1, max_retries,
                            )
                            await asyncio.sleep(wait)
                            continue
                        log.warning("Binance %s returned HTTP %s", path, resp.status)
                        return None
            except asyncio.TimeoutError:
                wait = _BACKOFF_BASE ** attempt
                log.warning("Binance %s timeout – retrying in %.1fs", path, wait)
                if is_depth:
                    self._depth_consecutive_timeouts += 1
                    if self._depth_consecutive_timeouts >= DEPTH_CIRCUIT_BREAKER_THRESHOLD:
                        self._depth_circuit_open_until = (
                            time.monotonic() + DEPTH_CIRCUIT_BREAKER_COOLDOWN
                        )
                        log.warning(
                            "Depth endpoint circuit breaker open — skipping {} for {:.0f}s "
                            "({} consecutive timeouts)",
                            path, DEPTH_CIRCUIT_BREAKER_COOLDOWN,
                            self._depth_consecutive_timeouts,
                        )
                        return None
                await asyncio.sleep(wait)
            except Exception as exc:
                log.error("Binance %s error: %s", path, exc)
                return None

        log.error("Binance %s – max retries (%d) exceeded", path, max_retries)
        return None

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def fetch_ticker_24h(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch 24-hour ticker statistics for *symbol*.

        Weight: 1 (single symbol).
        """
        if self.market == "futures":
            path = "/fapi/v1/ticker/24hr"
        else:
            path = "/api/v3/ticker/24hr"
        return await self._get(path, params={"symbol": symbol}, weight=1)

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> Optional[List[List[Any]]]:
        """Fetch OHLCV klines (candlestick data).

        Weight tiers (Binance exact):
          limit < 100   → weight 1
          100 ≤ limit < 500  → weight 2
          500 ≤ limit ≤ 1000 → weight 5
          limit > 1000       → weight 10
        """
        if self.market == "futures":
            path = "/fapi/v1/klines"
        else:
            path = "/api/v3/klines"
        if limit < 100:
            weight = 1
        elif limit < 500:
            weight = 2
        elif limit <= 1000:
            weight = 5
        else:
            weight = 10
        return await self._get(
            path,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            weight=weight,
        )

    async def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Optional[Dict[str, Any]]:
        """Fetch the order book depth snapshot.

        Weight: 1 (limit ≤ 100).
        """
        if self.market == "futures":
            path = "/fapi/v1/depth"
        else:
            path = "/api/v3/depth"
        return await self._get(
            path,
            params={"symbol": symbol, "limit": limit},
            weight=1,
            timeout=_DEPTH_TIMEOUT_S,
        )

    async def fetch_all_book_tickers(self) -> Optional[Dict[str, Dict[str, str]]]:
        """Fetch best bid/ask prices for **all** symbols in a single request.

        This is the weight-efficient alternative to per-symbol ``/depth``
        polling.  A single call returns the best bid and ask for every active
        symbol instead of issuing one request per symbol.

        Weight
        ------
        * Futures ``/fapi/v1/ticker/bookTicker`` (no ``symbol`` param): **2**
        * Spot    ``/api/v3/ticker/bookTicker``  (no ``symbol`` param): **2**

        Returns
        -------
        dict[str, dict]
            Mapping of symbol → ``{"bidPrice": str, "askPrice": str, ...}``,
            or ``None`` if the request fails.
        """
        if self.market == "futures":
            path = "/fapi/v1/ticker/bookTicker"
        else:
            path = "/api/v3/ticker/bookTicker"
        data = await self._get(path, weight=2)
        if not isinstance(data, list):
            return None
        return {item["symbol"]: item for item in data if "symbol" in item}

    async def fetch_exchange_info(self) -> Optional[Dict[str, Any]]:
        """Fetch exchange trading rules and symbol information.

        Weight: 10.
        """
        if self.market == "futures":
            path = "/fapi/v1/exchangeInfo"
        else:
            path = "/api/v3/exchangeInfo"
        return await self._get(path, weight=10)
