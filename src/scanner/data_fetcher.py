"""Kline and order-book data retrieval extracted from scanner.py."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DataFetcher:
    """Encapsulates all market data retrieval for the scanner.

    Extracts kline fetching, order-book spread computation, and
    multi-timeframe data assembly from the monolithic scanner.

    Parameters
    ----------
    data_store : Any
        The candle/tick data store (typically ``HistoricalDataManager``).
    exchange_mgr : Any
        Exchange client for order-book queries.
    spot_client : Optional[Any]
        Spot exchange client (for spot-specific data).
    """

    def __init__(
        self,
        data_store: Any,
        exchange_mgr: Any,
        spot_client: Optional[Any] = None,
    ) -> None:
        self._data_store = data_store
        self._exchange_mgr = exchange_mgr
        self._spot_client = spot_client
        # Spread cache: symbol → (spread_pct, timestamp)
        self._spread_cache: Dict[str, Tuple[float, float]] = {}
        self._spread_cache_ttl: float = 30.0

    def load_candles(self, symbol: str) -> Dict[str, dict]:
        """Load candles for all available timeframes from data store.

        Returns a dict keyed by timeframe string (e.g. "5m", "1h")
        where each value is a dict with keys: open, high, low, close, volume.
        Returns empty dict if no data available.
        """
        candles: Dict[str, dict] = {}
        store = self._data_store
        sym_candles = getattr(store, 'candles', {})
        if symbol not in sym_candles:
            return candles

        sym_data = sym_candles[symbol]
        for tf_key, ohlcv in sym_data.items():
            if isinstance(ohlcv, dict) and 'close' in ohlcv:
                candles[tf_key] = ohlcv
        return candles

    async def fetch_spread(self, symbol: str) -> float:
        """Fetch the current bid-ask spread percentage for a symbol.

        Uses a short-lived cache to avoid excessive API calls.
        Returns 0.0 on error or if unavailable.
        """
        now = time.monotonic()
        cached = self._spread_cache.get(symbol)
        if cached is not None:
            spread_val, cached_at = cached
            if now - cached_at < self._spread_cache_ttl:
                return spread_val

        try:
            if self._exchange_mgr is not None and hasattr(self._exchange_mgr, 'get_orderbook_spread'):
                spread = await self._exchange_mgr.get_orderbook_spread(symbol)
                spread_pct = float(spread) if spread is not None else 0.0
            else:
                spread_pct = 0.0
            self._spread_cache[symbol] = (spread_pct, now)
            return spread_pct
        except Exception as exc:
            logger.debug("Spread fetch failed for %s: %s", symbol, exc)
            self._spread_cache[symbol] = (0.0, now)
            return 0.0

    async def fetch_all_timeframes(
        self, symbol: str, timeframes: Optional[List[str]] = None
    ) -> Dict[str, list]:
        """Concurrently fetch klines for multiple timeframes.

        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g. "BTCUSDT").
        timeframes : list of str, optional
            Timeframes to fetch. Defaults to ["1m", "5m", "1h", "4h", "1d"].

        Returns
        -------
        dict
            Mapping of timeframe → list of candle dicts.
        """
        if timeframes is None:
            timeframes = ["1m", "5m", "1h", "4h", "1d"]

        tasks = {}
        for tf in timeframes:
            tasks[tf] = self._fetch_single_timeframe(symbol, tf)

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            tf: (r if not isinstance(r, BaseException) else [])
            for tf, r in zip(tasks.keys(), results)
        }

    async def _fetch_single_timeframe(self, symbol: str, timeframe: str) -> list:
        """Fetch klines for a single timeframe."""
        try:
            if self._exchange_mgr and hasattr(self._exchange_mgr, 'get_klines'):
                return await self._exchange_mgr.get_klines(symbol, timeframe)
            return []
        except Exception as exc:
            logger.debug("Kline fetch failed for %s %s: %s", symbol, timeframe, exc)
            return []
