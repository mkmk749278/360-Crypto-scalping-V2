"""Historical data seeding – fetch OHLCV and recent trades on boot.

Uses public Binance REST endpoints with rate-limit-compliant delays.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

from config import (
    BATCH_REQUEST_DELAY,
    BINANCE_FUTURES_REST_BASE,
    BINANCE_REST_BASE,
    SEED_TICK_LIMIT,
    SEED_TIMEFRAMES,
)
from src.pair_manager import PairManager
from src.utils import get_logger

log = get_logger("historical")


class HistoricalDataStore:
    """In-memory store for OHLCV and tick data, keyed by symbol + timeframe."""

    def __init__(self) -> None:
        # candles[symbol][timeframe] = {"open": [], "high": [], "low": [], "close": [], "volume": []}
        self.candles: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        # ticks[symbol] = [{"price": float, "qty": float, "isBuyerMaker": bool, "time": int}, …]
        self.ticks: Dict[str, List[Dict[str, Any]]] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    # ------------------------------------------------------------------
    # OHLCV fetch
    # ------------------------------------------------------------------

    async def fetch_candles(
        self, symbol: str, interval: str, limit: int, market: str = "spot",
    ) -> Dict[str, np.ndarray]:
        """Fetch OHLCV candles for one symbol/interval."""
        session = await self._ensure_session()
        if market == "futures":
            url = f"{BINANCE_FUTURES_REST_BASE}/fapi/v1/klines"
        else:
            url = f"{BINANCE_REST_BASE}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    log.warning("Candle fetch %s %s returned %s", symbol, interval, resp.status)
                    return {}
                raw = await resp.json()
        except Exception as exc:
            log.error("Candle fetch error %s %s: %s", symbol, interval, exc)
            return {}

        if not raw:
            return {}

        opens = np.array([float(c[1]) for c in raw])
        highs = np.array([float(c[2]) for c in raw])
        lows = np.array([float(c[3]) for c in raw])
        closes = np.array([float(c[4]) for c in raw])
        volumes = np.array([float(c[5]) for c in raw])

        return {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}

    # ------------------------------------------------------------------
    # Recent trades fetch
    # ------------------------------------------------------------------

    async def fetch_recent_trades(
        self, symbol: str, limit: int = SEED_TICK_LIMIT, market: str = "spot",
    ) -> List[Dict[str, Any]]:
        session = await self._ensure_session()
        if market == "futures":
            url = f"{BINANCE_FUTURES_REST_BASE}/fapi/v1/trades"
        else:
            url = f"{BINANCE_REST_BASE}/api/v3/trades"
        params = {"symbol": symbol, "limit": min(limit, 1000)}

        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    log.warning("Trade fetch %s returned %s", symbol, resp.status)
                    return []
                raw = await resp.json()
        except Exception as exc:
            log.error("Trade fetch error %s: %s", symbol, exc)
            return []

        return [
            {
                "price": float(t["price"]),
                "qty": float(t["qty"]),
                "isBuyerMaker": t.get("isBuyerMaker", False),
                "time": t.get("time", 0),
            }
            for t in raw
        ]

    # ------------------------------------------------------------------
    # Full seed for one symbol
    # ------------------------------------------------------------------

    async def seed_symbol(self, symbol: str, market: str = "spot") -> None:
        """Seed all timeframes + ticks for a single symbol."""
        self.candles.setdefault(symbol, {})

        for tf in SEED_TIMEFRAMES:
            data = await self.fetch_candles(symbol, tf.interval, tf.limit, market)
            if data:
                self.candles[symbol][tf.interval] = data
                log.debug("Seeded %s %s: %d candles", symbol, tf.interval, len(data["close"]))
            await asyncio.sleep(BATCH_REQUEST_DELAY)

        ticks = await self.fetch_recent_trades(symbol, SEED_TICK_LIMIT, market)
        if ticks:
            self.ticks[symbol] = ticks
            log.debug("Seeded %s ticks: %d", symbol, len(ticks))
        await asyncio.sleep(BATCH_REQUEST_DELAY)

    # ------------------------------------------------------------------
    # Full boot seed
    # ------------------------------------------------------------------

    async def seed_all(self, pair_mgr: PairManager) -> None:
        """Seed historical data for every active pair."""
        log.info("Starting historical data seed for %d pairs …", len(pair_mgr.pairs))
        for sym, info in pair_mgr.pairs.items():
            await self.seed_symbol(sym, info.market)
            pair_mgr.record_candles(
                sym, "all",
                sum(
                    len(d.get("close", []))
                    for d in self.candles.get(sym, {}).values()
                ),
            )
        log.info("Historical data seed complete.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_candles(self, symbol: str, interval: str) -> Optional[Dict[str, np.ndarray]]:
        return self.candles.get(symbol, {}).get(interval)

    def update_candle(self, symbol: str, interval: str, candle: Dict[str, float]) -> None:
        """Append a single candle (from WebSocket) to the store."""
        bucket = self.candles.setdefault(symbol, {}).setdefault(
            interval,
            {"open": np.array([]), "high": np.array([]), "low": np.array([]), "close": np.array([]), "volume": np.array([])},
        )
        for key in ("open", "high", "low", "close", "volume"):
            bucket[key] = np.append(bucket[key], candle.get(key, 0.0))

    def append_tick(self, symbol: str, tick: Dict[str, Any]) -> None:
        self.ticks.setdefault(symbol, []).append(tick)
        # Keep only the last SEED_TICK_LIMIT ticks
        if len(self.ticks[symbol]) > SEED_TICK_LIMIT:
            self.ticks[symbol] = self.ticks[symbol][-SEED_TICK_LIMIT:]

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
