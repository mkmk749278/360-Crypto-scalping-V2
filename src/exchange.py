"""Multi-Exchange Abstraction.

Provides :class:`ExchangeManager` which wraps multiple exchange clients and
enables cross-exchange signal verification.  Initially ships with Binance
support and a stub for a second exchange (Bybit / OKX).
"""

from __future__ import annotations

from typing import Optional

import aiohttp

from src.utils import get_logger

log = get_logger("exchange_mgr")

# Price tolerance for cross-exchange validation (percentage)
_PRICE_TOLERANCE_PCT: float = 0.5


class ExchangeManager:
    """Wraps multiple exchange REST clients for cross-exchange verification.

    Only Binance is fully implemented. A second exchange can be added by
    providing a ``second_exchange_url`` and implementing ``_fetch_price_second``.

    Parameters
    ----------
    second_exchange_url:
        Optional base URL for a second exchange ticker endpoint.
        If not provided, :meth:`verify_signal_cross_exchange` returns ``False``
        (unable to verify), which the confidence scorer maps to a neutral score.
    """

    def __init__(self, second_exchange_url: Optional[str] = None) -> None:
        self._second_url = second_exchange_url
        self._session: Optional[aiohttp.ClientSession] = None

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
    # Cross-exchange verification
    # ------------------------------------------------------------------

    async def verify_signal_cross_exchange(
        self,
        symbol: str,
        direction: str,
        price: float,
    ) -> bool:
        """Check a second exchange to confirm the signal direction.

        Returns ``True`` only when:
          1. The second-exchange price is within the price tolerance, **and**
          2. The second exchange's recent price movement agrees with *direction*.

        Returns ``False`` when no second exchange is configured, prices diverge
        significantly, or the second exchange's directional bias contradicts the
        signal direction.

        Parameters
        ----------
        symbol:
            Trading symbol, e.g. ``"BTCUSDT"``.
        direction:
            ``"LONG"`` or ``"SHORT"``.
        price:
            Entry price from the primary exchange.
        """
        if not self._second_url:
            log.debug("No second exchange configured – cross-exchange check skipped")
            return False

        second_price = await self._fetch_price_second(symbol)
        if second_price is None:
            return False

        tolerance = price * _PRICE_TOLERANCE_PCT / 100.0
        spread = abs(second_price - price)

        if spread > tolerance:
            log.debug(
                "%s cross-exchange price divergence %.4f (primary) vs %.4f (second) – "
                "spread %.4f > tol %.4f",
                symbol, price, second_price, spread, tolerance,
            )
            return False

        # Directional agreement check: the second-exchange price must be on the
        # same side of the primary price as the signal direction implies.
        # A LONG signal expects second_price >= primary (or neutral within tol).
        # A SHORT signal expects second_price <= primary (or neutral within tol).
        direction_upper = direction.upper()
        if direction_upper == "LONG" and second_price < price * (1.0 - _PRICE_TOLERANCE_PCT / 100.0 / 2):
            log.debug(
                "%s cross-exchange directional mismatch: LONG signal but second=%.4f < primary=%.4f",
                symbol, second_price, price,
            )
            return False
        if direction_upper == "SHORT" and second_price > price * (1.0 + _PRICE_TOLERANCE_PCT / 100.0 / 2):
            log.debug(
                "%s cross-exchange directional mismatch: SHORT signal but second=%.4f > primary=%.4f",
                symbol, second_price, price,
            )
            return False

        log.debug(
            "%s cross-exchange verified: primary=%.4f second=%.4f direction=%s",
            symbol, price, second_price, direction,
        )
        return True

    # ------------------------------------------------------------------
    # Second-exchange price fetch (override to support specific exchanges)
    # ------------------------------------------------------------------

    async def _fetch_price_second(self, symbol: str) -> Optional[float]:
        """Fetch the latest price from the second exchange.

        Supports URLs that already contain query parameters (e.g. Bybit V5
        ``?category=linear``) by appending ``&symbol=`` instead of
        ``?symbol=``.  Handles both Bybit V5 nested response format
        (``{"result": {"list": [{"lastPrice": "..."}]}}``) and Binance-style
        flat format (``{"price": "..."}`` or ``{"lastPrice": "..."}`).
        """
        if not self._second_url:
            return None
        try:
            session = await self._ensure_session()
            sep = "&" if "?" in self._second_url else "?"
            url = f"{self._second_url}{sep}symbol={symbol}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                # Bybit V5 nested format: {"result": {"list": [{"lastPrice": "..."}]}}
                if "result" in data and "list" in data.get("result", {}):
                    items = data["result"]["list"]
                    if items:
                        raw = items[0].get("lastPrice") or items[0].get("last")
                        return float(raw) if raw else None
                # Binance-style flat format: {"price": "..."} or {"lastPrice": "..."}
                raw = data.get("price") or data.get("lastPrice") or data.get("last")
                return float(raw) if raw is not None else None
        except Exception as exc:
            log.debug("Second-exchange price fetch for %s failed: %s", symbol, exc)
            return None
