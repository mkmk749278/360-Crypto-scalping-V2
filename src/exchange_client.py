"""CCXT-based exchange client wrapper for direct order execution (V3).

Wraps ``ccxt.pro`` (async) to provide a minimal, testable interface for
:class:`~src.order_manager.OrderManager`.

All methods degrade gracefully when ``ccxt`` is not installed — in that
case every call raises :class:`NotImplementedError` so the engine can
still run in signal-only mode without the optional dependency.

Usage
-----
Instantiate :class:`CCXTClient` and pass it as ``exchange_client`` to
:class:`~src.order_manager.OrderManager`:

.. code-block:: python

    client = CCXTClient(
        exchange_id="binance",
        api_key="...",
        secret="...",
        sandbox=True,
    )
    order_mgr = OrderManager(auto_execution_enabled=True, exchange_client=client)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.utils import get_logger

log = get_logger("exchange_client")

# Attempt to import ccxt.pro (async); fall back to synchronous ccxt if not
# available, and finally set a sentinel so callers know CCXT is absent.
try:
    import ccxt.pro as _ccxtpro  # type: ignore[import]
    _CCXT_AVAILABLE = True
    _ccxt_module = _ccxtpro
except ImportError:
    try:
        import ccxt as _ccxt  # type: ignore[import]
        _CCXT_AVAILABLE = True
        _ccxt_module = _ccxt
    except ImportError:
        _CCXT_AVAILABLE = False
        _ccxt_module = None  # type: ignore[assignment]


class CCXTClient:
    """Minimal async wrapper around a CCXT exchange instance.

    Parameters
    ----------
    exchange_id:
        CCXT exchange identifier (e.g. ``"binance"``, ``"bybit"``).
    api_key:
        Exchange API key.
    secret:
        Exchange API secret.
    sandbox:
        When ``True`` the exchange is configured in sandbox/testnet mode.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        secret: str = "",
        sandbox: bool = False,
    ) -> None:
        self._exchange_id = exchange_id
        self._sandbox = sandbox
        self._exchange: Optional[Any] = None

        if not _CCXT_AVAILABLE:
            log.warning(
                "ccxt is not installed — CCXTClient will raise NotImplementedError "
                "for all order operations.  Install ccxt to enable live execution."
            )
            return

        try:
            exchange_class = getattr(_ccxt_module, exchange_id)
            self._exchange = exchange_class(
                {
                    "apiKey": api_key,
                    "secret": secret,
                    "sandbox": sandbox,
                    "options": {"defaultType": "spot"},
                }
            )
            if sandbox:
                self._exchange.set_sandbox_mode(True)
            log.info(
                "CCXTClient initialised: exchange=%s sandbox=%s",
                exchange_id,
                sandbox,
            )
        except AttributeError:
            log.error(
                "Exchange '%s' not found in ccxt — CCXTClient disabled.",
                exchange_id,
            )
            self._exchange = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> Dict[str, Any]:
        """Place a limit order.

        Parameters
        ----------
        symbol:
            CCXT market symbol, e.g. ``"BTC/USDT"``.
        side:
            ``"buy"`` or ``"sell"``.
        amount:
            Order size in base currency.
        price:
            Limit price.

        Returns
        -------
        dict
            CCXT order response dict (always contains ``"id"``).
        """
        self._require_exchange()
        log.info(
            "create_limit_order: %s %s %s @ %s (sandbox=%s)",
            side, amount, symbol, price, self._sandbox,
        )
        return await self._exchange.create_limit_order(symbol, side, amount, price)  # type: ignore[union-attr]

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
    ) -> Dict[str, Any]:
        """Place a market order.

        Parameters
        ----------
        symbol:
            CCXT market symbol, e.g. ``"BTC/USDT"``.
        side:
            ``"buy"`` or ``"sell"``.
        amount:
            Order size in base currency.

        Returns
        -------
        dict
            CCXT order response dict (always contains ``"id"``).
        """
        self._require_exchange()
        log.info(
            "create_market_order: %s %s %s (sandbox=%s)",
            side, amount, symbol, self._sandbox,
        )
        return await self._exchange.create_market_order(symbol, side, amount)  # type: ignore[union-attr]

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an open order.

        Parameters
        ----------
        order_id:
            Exchange-assigned order identifier.
        symbol:
            CCXT market symbol.

        Returns
        -------
        dict
            CCXT cancellation response.
        """
        self._require_exchange()
        log.info("cancel_order: id=%s symbol=%s (sandbox=%s)", order_id, symbol, self._sandbox)
        return await self._exchange.cancel_order(order_id, symbol)  # type: ignore[union-attr]

    async def fetch_balance(self) -> Dict[str, Any]:
        """Fetch account balance.

        Returns
        -------
        dict
            CCXT balance response (``{"USDT": {"free": ..., "total": ...}, ...}``).
        """
        self._require_exchange()
        return await self._exchange.fetch_balance()  # type: ignore[union-attr]

    async def close(self) -> None:
        """Close the underlying exchange connection / session."""
        if self._exchange is not None and hasattr(self._exchange, "close"):
            try:
                await self._exchange.close()
            except Exception as exc:
                log.debug("CCXTClient.close() error: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_exchange(self) -> None:
        """Raise if CCXT is unavailable or the exchange failed to initialise."""
        if not _CCXT_AVAILABLE:
            raise NotImplementedError(
                "ccxt is not installed.  "
                "Install it with: pip install ccxt"
            )
        if self._exchange is None:
            raise NotImplementedError(
                f"Exchange '{self._exchange_id}' could not be initialised via ccxt."
            )
