"""Dynamic pair management – auto-fetch top Spot & Futures pairs from Binance.

Pairs are refreshed every ``PAIR_FETCH_INTERVAL_HOURS`` using public REST
endpoints.  New pairs start with a reduced confidence cap until enough
historical data has been accumulated.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import (
    BATCH_REQUEST_DELAY,
    GEM_MIN_VOLUME_USD,
    GEM_PAIRS_COUNT,
    PAIR_FETCH_INTERVAL_HOURS,
    TOP_PAIRS_COUNT,
)
from src.binance import BinanceClient
from src.utils import get_logger

log = get_logger("pair_manager")

# Stablecoin-vs-stablecoin pairs produce no tradeable signal: the spread
# alone exceeds the entire TP range.  These pairs appear near the top of
# volume rankings so they must be explicitly excluded.
_STABLECOIN_BLACKLIST: frozenset = frozenset({
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT", "FDUSDUSDT",
    "USD1USDT", "DAIUSDT", "EURUSDT", "USDCBUSD", "USDTDAI",
    # USD-pegged stablecoins that produce untradeable signals against USDT
    "RLUSDUSDT", "PYUSDUSDT", "USDDUSDT", "GUSDUSDT",
    "FRAXUSDT", "LUSDUSDT", "SUSDUSDT", "CUSDUSDT",
})


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
        self._spot_client = BinanceClient("spot")
        self._futures_client = BinanceClient("futures")

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

    async def fetch_top_spot_pairs(self, limit: int = TOP_PAIRS_COUNT) -> List[PairInfo]:
        """Fetch top *limit* USDT spot pairs by 24h volume."""
        pairs: List[PairInfo] = []
        try:
            data = await self._spot_client._get("/api/v3/ticker/24hr", weight=40)
            if data is None:
                log.warning("Spot ticker fetch returned no data")
                return pairs

            usdt_pairs = [
                t for t in data
                if t.get("symbol", "").endswith("USDT")
                and t.get("symbol", "") not in _STABLECOIN_BLACKLIST
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
        pairs: List[PairInfo] = []
        try:
            data = await self._futures_client._get("/fapi/v1/ticker/24hr", weight=40)
            if data is None:
                log.warning("Futures ticker fetch returned no data")
                return pairs

            usdt_pairs = [
                t for t in data
                if t.get("symbol", "").endswith("USDT")
                and t.get("symbol", "") not in _STABLECOIN_BLACKLIST
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

    async def fetch_gem_universe(self, limit: int = GEM_PAIRS_COUNT) -> List[PairInfo]:
        """Fetch a wider set of USDT spot pairs for gem scanning.

        This is a **separate** fetch from the main pair universe — it uses a
        lower minimum volume threshold (``GEM_MIN_VOLUME_USD``) and returns up
        to *limit* pairs sorted by 24h USD volume descending.  The result is
        only used by the gem scanner; it does not modify ``self.pairs``.
        """
        pairs: List[PairInfo] = []
        try:
            data = await self._spot_client._get("/api/v3/ticker/24hr", weight=40)
            if data is None:
                log.warning("Gem universe fetch returned no data")
                return pairs

            usdt_pairs = [
                t for t in data
                if t.get("symbol", "").endswith("USDT")
                and t.get("symbol", "") not in _STABLECOIN_BLACKLIST
                and float(t.get("quoteVolume", 0)) >= GEM_MIN_VOLUME_USD
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
            log.error("fetch_gem_universe error: %s", exc)
        log.info("Gem universe fetched – %d pairs (limit=%d, min_vol=$%,.0f)",
                 len(pairs), limit, GEM_MIN_VOLUME_USD)
        return pairs

    async def refresh_pairs(
        self,
        market: Optional[str] = None,
        count: Optional[int] = None,
    ) -> List[str]:
        """Refresh the active pair universe.

        Parameters
        ----------
        market:
            ``"spot"``, ``"futures"``, or ``None`` (both).
        count:
            Override for the number of top pairs to fetch.  Falls back to
            ``TOP_PAIRS_COUNT`` when ``None``.

        Returns
        -------
        List[str]
            Symbols that were newly added to the universe during this refresh.
        """
        log.info("Refreshing pair universe (market=%s, count=%s) …", market, count)
        limit = count if count is not None else TOP_PAIRS_COUNT

        if market == "spot":
            spot = await self.fetch_top_spot_pairs(limit)
            futures: List[PairInfo] = []
        elif market == "futures":
            spot = []
            futures = await self.fetch_top_futures_pairs(limit)
        else:
            spot, futures = await asyncio.gather(
                self.fetch_top_spot_pairs(limit),
                self.fetch_top_futures_pairs(limit),
            )
        new_count = 0
        new_symbols: List[str] = []
        for p in spot + futures:
            if p.symbol not in self.pairs:
                new_count += 1
                new_symbols.append(p.symbol)
                self.pairs[p.symbol] = p
            else:
                self.pairs[p.symbol].volume_24h_usd = p.volume_24h_usd
        log.info(
            "Pair refresh done – total %d pairs (%d new)",
            len(self.pairs), new_count,
        )
        return new_symbols

    async def run_periodic_refresh(self) -> None:
        """Infinite loop that refreshes pairs every N hours."""
        while True:
            await self.refresh_pairs()
            await asyncio.sleep(PAIR_FETCH_INTERVAL_HOURS * 3600)

    async def close(self) -> None:
        await self._spot_client.close()
        await self._futures_client.close()
