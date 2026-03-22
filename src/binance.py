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

from config import BINANCE_FUTURES_REST_BASE, BINANCE_REST_BASE
from src.rate_limiter import rate_limiter
from src.utils import get_logger

log = get_logger("binance_client")

# Binance default rate-limit window (60 s) and request-weight limit
_WEIGHT_WINDOW_S: int = 60
_DEFAULT_WEIGHT_LIMIT: int = 1_200

# Retry parameters
_MAX_RETRIES: int = 5
_BACKOFF_BASE: float = 1.5  # exponential-backoff base (seconds)


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
        self._weight_reset_at: float = time.monotonic() + _WEIGHT_WINDOW_S

    # ------------------------------------------------------------------
    # Weight tracking
    # ------------------------------------------------------------------

    @property
    def remaining_weight(self) -> int:
        """Estimated remaining request weight in the current window."""
        self._maybe_reset_weight()
        return max(0, _DEFAULT_WEIGHT_LIMIT - self._used_weight)

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
    ) -> Any:
        """Execute a GET request with retry logic.

        Handles 429 (rate limit) and 418 (IP ban) by waiting and retrying
        with exponential back-off up to ``_MAX_RETRIES`` attempts.
        """
        session = await self._ensure_session()
        url = self._base_url + path

        # Throttle proactively: wait until the shared rate-limiter budget has
        # room for this request.  This prevents bursting 200+ requests at once
        # and keeps weight consumption well under Binance's 1 200/min cap.
        await rate_limiter.acquire(weight)
        self._consume_weight(weight)

        for attempt in range(_MAX_RETRIES):
            try:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=15)
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
                        rate_limiter.update_from_header(raw_weight)
                        if raw_weight is not None:
                            try:
                                self._used_weight = int(raw_weight)
                            except ValueError:
                                pass
                        if BinanceClient.on_api_call is not None:
                            BinanceClient.on_api_call()
                        return data
                    if resp.status in (429, 418):
                        retry_after = int(resp.headers.get("Retry-After", 5))
                        wait = max(retry_after, _BACKOFF_BASE ** attempt)
                        log.warning(
                            "Binance %s – rate limited (%s). Waiting %.1fs (attempt %d/%d)",
                            path, resp.status, wait, attempt + 1, _MAX_RETRIES,
                        )
                        await asyncio.sleep(wait)
                        continue
                    log.warning("Binance %s returned HTTP %s", path, resp.status)
                    return None
            except asyncio.TimeoutError:
                wait = _BACKOFF_BASE ** attempt
                log.warning("Binance %s timeout – retrying in %.1fs", path, wait)
                await asyncio.sleep(wait)
            except Exception as exc:
                log.error("Binance %s error: %s", path, exc)
                return None

        log.error("Binance %s – max retries (%d) exceeded", path, _MAX_RETRIES)
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

        Weight: 1–10 depending on *limit*.
        """
        if self.market == "futures":
            path = "/fapi/v1/klines"
        else:
            path = "/api/v3/klines"
        weight = max(1, limit // 100)
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
            path, params={"symbol": symbol, "limit": limit}, weight=1
        )

    async def fetch_exchange_info(self) -> Optional[Dict[str, Any]]:
        """Fetch exchange trading rules and symbol information.

        Weight: 10.
        """
        if self.market == "futures":
            path = "/fapi/v1/exchangeInfo"
        else:
            path = "/api/v3/exchangeInfo"
        return await self._get(path, weight=10)
