"""Order manager – structural foundation for direct exchange execution (V3).

This module provides the :class:`OrderManager` interface so that the rest of
the engine can already call ``await order_manager.place_limit_order(sig)``
without any live exchange logic being wired in yet.

The stubs log the intent and return ``None``.  In V3, swap each stub body
with real CCXT / Binance Trade-API calls.  The calling code in
:class:`src.trade_monitor.TradeMonitor` does not need to change.

Design notes
------------
* Limit orders are used for DCA / swing strategies (``360_SWING``, ``360_SPOT``)
  to capture maker-fee rebates and reduce slippage on fills.
* Market orders are used for high-frequency scalp strategies
  (``360_SCALP``) where immediate fill is more important
  than the maker/taker fee delta.
* Auto-execution is **off by default** (``AUTO_EXECUTION_ENABLED=false``).
  The engine still publishes to Telegram as normal; the order stubs simply
  no-op until the feature flag is enabled.
"""

from __future__ import annotations

from typing import Any, Optional

from src.utils import get_logger

log = get_logger("order_manager")

# Channels for which limit orders should be preferred (maker-fee strategy).
_LIMIT_ORDER_CHANNELS = {"360_SWING", "360_SPOT"}


class OrderManager:
    """Manages direct exchange order placement (stub implementation).

    Parameters
    ----------
    auto_execution_enabled:
        Master toggle.  When ``False`` all methods are no-ops; signals are
        still routed to Telegram as usual.
    exchange_client:
        Future: a CCXT ``AsyncExchange`` instance or a Binance Trade-API
        wrapper.  Pass ``None`` until the real client is available.
    """

    def __init__(
        self,
        auto_execution_enabled: bool = False,
        exchange_client: Optional[Any] = None,
    ) -> None:
        self._enabled = auto_execution_enabled
        self._client = exchange_client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """Return ``True`` when auto-execution is active."""
        return self._enabled

    async def place_limit_order(
        self,
        signal: Any,
        *,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
    ) -> Optional[str]:
        """Place a limit (maker) order on the exchange.

        Used by mean-reversion and DCA strategies (``360_RANGE``) to post
        resting bids/offers and capture maker-fee rebates.

        Parameters
        ----------
        signal:
            The :class:`src.channels.base.Signal` driving the order.
        price:
            Explicit limit price.  When ``None`` the signal's ``entry``
            price is used.
        quantity:
            Order size in base currency.  When ``None`` the exchange
            client's position-sizing logic should determine the quantity.

        Returns
        -------
        str or None
            Exchange order-ID on success; ``None`` when execution is
            disabled or the stub has not been implemented yet.
        """
        if not self._enabled:
            return None

        limit_price = price if price is not None else signal.entry
        log.info(
            "[OrderManager] STUB place_limit_order: {} {} {} @ {} (qty={})",
            signal.symbol,
            signal.channel,
            signal.direction.value,
            limit_price,
            quantity,
        )
        # TODO (V3): replace with real exchange API call, e.g.:
        #   order = await self._client.create_limit_order(
        #       symbol=signal.symbol,
        #       side="buy" if signal.direction.value == "LONG" else "sell",
        #       amount=quantity,
        #       price=limit_price,
        #   )
        #   return order["id"]
        return None

    async def place_market_order(
        self,
        signal: Any,
        *,
        quantity: Optional[float] = None,
    ) -> Optional[str]:
        """Place a market (taker) order on the exchange.

        Used by high-frequency strategies (``360_SCALP``, ``360_THE_TAPE``)
        where immediate fill certainty outweighs the taker-fee cost.

        Parameters
        ----------
        signal:
            The :class:`src.channels.base.Signal` driving the order.
        quantity:
            Order size in base currency.

        Returns
        -------
        str or None
            Exchange order-ID on success; ``None`` when disabled / stub.
        """
        if not self._enabled:
            return None

        log.info(
            "[OrderManager] STUB place_market_order: {} {} {} (qty={})",
            signal.symbol,
            signal.channel,
            signal.direction.value,
            quantity,
        )
        # TODO (V3): replace with real exchange API call, e.g.:
        #   order = await self._client.create_market_order(
        #       symbol=signal.symbol,
        #       side="buy" if signal.direction.value == "LONG" else "sell",
        #       amount=quantity,
        #   )
        #   return order["id"]
        return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open exchange order.

        Parameters
        ----------
        order_id:
            The exchange-assigned order identifier returned by
            :meth:`place_limit_order` or :meth:`place_market_order`.
        symbol:
            Trading-pair symbol (e.g. ``"BTCUSDT"``).

        Returns
        -------
        bool
            ``True`` when the cancellation was confirmed; ``False`` when
            execution is disabled or the stub has not been implemented.
        """
        if not self._enabled:
            return False

        log.info(
            "[OrderManager] STUB cancel_order: order_id={} symbol={}",
            order_id,
            symbol,
        )
        # TODO (V3): replace with real exchange API call, e.g.:
        #   result = await self._client.cancel_order(order_id, symbol)
        #   return result.get("status") == "canceled"
        return False

    async def execute_signal(self, signal: Any) -> Optional[str]:
        """Dispatch an order for *signal* using the appropriate order type.

        Convenience wrapper that selects limit vs. market order based on the
        signal's channel:

        * ``360_RANGE`` / ``360_SWING`` → :meth:`place_limit_order` (maker)
        * All other channels → :meth:`place_market_order` (taker)

        Parameters
        ----------
        signal:
            The :class:`src.channels.base.Signal` to execute.

        Returns
        -------
        str or None
            Exchange order-ID, or ``None`` when disabled / stub.
        """
        if not self._enabled:
            return None

        if signal.channel in _LIMIT_ORDER_CHANNELS:
            return await self.place_limit_order(signal)
        return await self.place_market_order(signal)
