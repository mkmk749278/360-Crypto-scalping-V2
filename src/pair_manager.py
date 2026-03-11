"""Dynamic pair management – auto-fetch top Spot & Futures pairs from Binance.

Pairs are refreshed every ``PAIR_FETCH_INTERVAL_HOURS`` using public REST
endpoints.  New pairs start with a reduced confidence cap until enough
historical data has been accumulated.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import aiohttp

from config import (
    BATCH_REQUEST_DELAY,
    BINANCE_FUTURES_REST_BASE,
    BINANCE_REST_BASE,
    PAIR_FETCH_INTERVAL_HOURS,
    TOP_PAIRS_COUNT,
)
from src.utils import get_logger

log = get_logger("pair_manager")


@dataclass
class PairInfo:
    symbol: str
    market: str  # "spot" or "futures"
    base_asset: str = ""
    quote_asset: str = ""
    volume_24h_usd: float = 0.0
    is_new: bool = True
    candle_counts: Dict[str, int] = field(default_factory=dict)


class PairManager:
    """Fetches and maintains the active pair universe."""

    def __init__(self) -> None:
        self.pairs: Dict[str, PairInfo] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> List[str]:
        return list(self.pairs.keys())

    @property
    def spot_symbols(self) -> List[str]:
        return [s for s, p in self.pairs.items() if p.market == "spot"]

    @property
    def futures_symbols(self) -> List[str]:
        return [s for s, p in self.pairs.items() if p.market == "futures"]

    def has_enough_history(self, symbol: str, min_candles: int = 500) -> bool:
        info = self.pairs.get(symbol)
        if info is None:
            return False
        return all(v >= min_candles for v in info.candle_counts.values()) if info.candle_counts else False

    def record_candles(self, symbol: str, timeframe: str, count: int) -> None:
        if symbol in self.pairs:
            self.pairs[symbol].candle_counts[timeframe] = count
            total = sum(self.pairs[symbol].candle_counts.values())
            if total >= 500:
                self.pairs[symbol].is_new = False

    # ------------------------------------------------------------------
    # Fetch from Binance
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def fetch_top_spot_pairs(self, limit: int = TOP_PAIRS_COUNT) -> List[PairInfo]:
        """Fetch top *limit* USDT spot pairs by 24h volume."""
        session = await self._ensure_session()
        pairs: List[PairInfo] = []
        try:
            url = f"{BINANCE_REST_BASE}/api/v3/ticker/24hr"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    log.warning("Spot ticker fetch returned %s", resp.status)
                    return pairs
                data = await resp.json()

            usdt_pairs = [
                t for t in data
                if t.get("symbol", "").endswith("USDT")
                and float(t.get("quoteVolume", 0)) > 0
            ]
            usdt_pairs.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)

            for t in usdt_pairs[:limit]:
                sym = t["symbol"]
                pairs.append(PairInfo(
                    symbol=sym,
                    market="spot",
                    base_asset=sym.replace("USDT", ""),
                    quote_asset="USDT",
                    volume_24h_usd=float(t.get("quoteVolume", 0)),
                ))
                await asyncio.sleep(BATCH_REQUEST_DELAY * 0.1)
        except Exception as exc:
            log.error("fetch_top_spot_pairs error: %s", exc)
        return pairs

    async def fetch_top_futures_pairs(self, limit: int = TOP_PAIRS_COUNT) -> List[PairInfo]:
        """Fetch top *limit* USDT-M futures pairs by 24h volume."""
        session = await self._ensure_session()
        pairs: List[PairInfo] = []
        try:
            url = f"{BINANCE_FUTURES_REST_BASE}/fapi/v1/ticker/24hr"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    log.warning("Futures ticker fetch returned %s", resp.status)
                    return pairs
                data = await resp.json()

            usdt_pairs = [
                t for t in data
                if t.get("symbol", "").endswith("USDT")
                and float(t.get("quoteVolume", 0)) > 0
            ]
            usdt_pairs.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)

            for t in usdt_pairs[:limit]:
                sym = t["symbol"]
                pairs.append(PairInfo(
                    symbol=sym,
                    market="futures",
                    base_asset=sym.replace("USDT", ""),
                    quote_asset="USDT",
                    volume_24h_usd=float(t.get("quoteVolume", 0)),
                ))
                await asyncio.sleep(BATCH_REQUEST_DELAY * 0.1)
        except Exception as exc:
            log.error("fetch_top_futures_pairs error: %s", exc)
        return pairs

    async def refresh_pairs(self) -> None:
        """Refresh the active pair universe (spot + futures)."""
        log.info("Refreshing pair universe …")
        spot, futures = await asyncio.gather(
            self.fetch_top_spot_pairs(),
            self.fetch_top_futures_pairs(),
        )
        new_count = 0
        for p in spot + futures:
            if p.symbol not in self.pairs:
                new_count += 1
                self.pairs[p.symbol] = p
            else:
                self.pairs[p.symbol].volume_24h_usd = p.volume_24h_usd
        log.info(
            "Pair refresh done – total %d pairs (%d new)",
            len(self.pairs), new_count,
        )

    async def run_periodic_refresh(self) -> None:
        """Infinite loop that refreshes pairs every N hours."""
        while True:
            await self.refresh_pairs()
            await asyncio.sleep(PAIR_FETCH_INTERVAL_HOURS * 3600)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
