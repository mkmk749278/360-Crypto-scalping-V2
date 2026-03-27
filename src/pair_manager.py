"""Dynamic pair management – auto-fetch top Spot & Futures pairs from Binance.

Pairs are refreshed every ``PAIR_FETCH_INTERVAL_HOURS`` using public REST
endpoints.  New pairs start with a reduced confidence cap until enough
historical data has been accumulated.  The pair universe is partitioned into
three tiers:

* **Tier 1** — Core (top ``TIER1_PAIR_COUNT`` by volume): full scan every
  cycle, all channels, WebSocket + order book.
* **Tier 2** — Discovery (rank ``TIER1_PAIR_COUNT``–``TIER2_PAIR_COUNT`` by
  volume): scan every N cycles, SWING + SPOT channels only, REST klines.
* **Tier 3** — Full Universe (all remaining USDT pairs): lightweight volume /
  momentum scan on a time-gated interval; auto-promoted to Tier 2 on volume
  surges.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from config import (
    GEM_MIN_VOLUME_USD,
    GEM_PAIRS_COUNT,
    PAIR_FETCH_INTERVAL_HOURS,
    PAIR_PROFILES,
    PAIR_PRUNE_ENABLED,
    PAIR_TIER_MAP,
    PairProfile,
    TIER1_PAIR_COUNT,
    TIER2_PAIR_COUNT,
    TIER3_VOLUME_SURGE_MULTIPLIER,
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


def classify_pair_tier(symbol: str, volume_24h_usd: float = 0.0) -> PairProfile:
    """Return the PairProfile for a given symbol.

    Falls back to volume-based heuristic for unlisted pairs:
    - volume >= $500M/day → MAJOR
    - volume >= $50M/day  → MIDCAP
    - otherwise           → ALTCOIN
    """
    tier = PAIR_TIER_MAP.get(symbol.upper())
    if tier is None:
        if volume_24h_usd >= 500_000_000:
            tier = "MAJOR"
        elif volume_24h_usd >= 50_000_000:
            tier = "MIDCAP"
        else:
            tier = "ALTCOIN"
    return PAIR_PROFILES[tier]


class PairTier(str, Enum):
    """Volume-ranked tier classification for the active pair universe."""
    TIER1 = "TIER1"  # Core — full scan every cycle, all channels, WS + OB
    TIER2 = "TIER2"  # Discovery — periodic scan, SWING+SPOT only, REST
    TIER3 = "TIER3"  # Universe — lightweight scan, auto-promote on volume surge


@dataclass
class PairInfo:
    symbol: str
    market: str  # "spot" or "futures"
    base_asset: str = ""
    quote_asset: str = ""
    volume_24h_usd: float = 0.0
    is_new: bool = True
    candle_counts: Dict[str, int] = field(default_factory=dict)
    tier: PairTier = PairTier.TIER1


class PairManager:
    """Fetches and maintains the active pair universe."""

    def __init__(self) -> None:
        self.pairs: Dict[str, PairInfo] = {}
        self._spot_client = BinanceClient("spot")
        self._futures_client = BinanceClient("futures")
        # Previous 24h volume per symbol — used for Tier 3 volume surge detection.
        self._prev_volumes: Dict[str, float] = {}

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

    @property
    def tier1_symbols(self) -> List[str]:
        return [s for s, p in self.pairs.items() if p.tier == PairTier.TIER1]

    @property
    def tier2_symbols(self) -> List[str]:
        return [s for s, p in self.pairs.items() if p.tier == PairTier.TIER2]

    @property
    def tier3_symbols(self) -> List[str]:
        return [s for s, p in self.pairs.items() if p.tier == PairTier.TIER3]

    def get_tiered_pairs(self) -> Dict[str, List[str]]:
        """Return pairs categorized into scanning tiers.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary with keys ``"tier1"``, ``"tier2"``, ``"tier3"`` mapping
            to lists of symbols in each scanning tier.
        """
        return {
            "tier1": self.tier1_symbols,
            "tier2": self.tier2_symbols,
            "tier3": self.tier3_symbols,
        }

    @property
    def tier1_spot_symbols(self) -> List[str]:
        return [s for s, p in self.pairs.items() if p.tier == PairTier.TIER1 and p.market == "spot"]

    @property
    def tier1_futures_symbols(self) -> List[str]:
        return [s for s, p in self.pairs.items() if p.tier == PairTier.TIER1 and p.market == "futures"]

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
        except Exception as exc:
            log.error("fetch_top_futures_pairs error: %s", exc)
        return pairs

    async def fetch_all_spot_pairs(self) -> List[PairInfo]:
        """Fetch **all** USDT spot pairs by 24h volume (no limit slice).

        Unlike :meth:`fetch_top_spot_pairs`, this method returns the complete
        sorted list so that the tier classification can assign every pair a
        volume rank.  It is used exclusively by :meth:`refresh_pairs`.
        """
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

            for t in usdt_pairs:
                sym = t["symbol"]
                pairs.append(PairInfo(
                    symbol=sym,
                    market="spot",
                    base_asset=sym.replace("USDT", ""),
                    quote_asset="USDT",
                    volume_24h_usd=float(t.get("quoteVolume", 0)),
                ))
        except Exception as exc:
            log.error("fetch_all_spot_pairs error: %s", exc)
        return pairs

    async def fetch_all_futures_pairs(self) -> List[PairInfo]:
        """Fetch **all** USDT-M futures pairs by 24h volume (no limit slice).

        Like :meth:`fetch_all_spot_pairs`, this returns every pair so that
        :meth:`refresh_pairs` can classify the full universe into tiers.
        """
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

            for t in usdt_pairs:
                sym = t["symbol"]
                pairs.append(PairInfo(
                    symbol=sym,
                    market="futures",
                    base_asset=sym.replace("USDT", ""),
                    quote_asset="USDT",
                    volume_24h_usd=float(t.get("quoteVolume", 0)),
                ))
        except Exception as exc:
            log.error("fetch_all_futures_pairs error: %s", exc)
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
        except Exception as exc:
            log.error("fetch_gem_universe error: %s", exc)
        log.info("Gem universe fetched – %d pairs (limit=%d, min_vol=$%,.0f)",
                 len(pairs), limit, GEM_MIN_VOLUME_USD)
        return pairs

    async def refresh_pairs(
        self,
        market: Optional[str] = None,
        count: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """Refresh the active pair universe with tiered classification.

        Fetches **all** available USDT pairs from Binance, classifies them
        into three tiers based on 24h volume ranking, and optionally prunes
        pairs that are no longer present on the exchange.

        Parameters
        ----------
        market:
            ``"spot"``, ``"futures"``, or ``None`` (both).
        count:
            Unused — kept for backward API compatibility.  Tier boundaries are
            now controlled by ``TIER1_PAIR_COUNT`` and ``TIER2_PAIR_COUNT``.

        Returns
        -------
        Tuple[List[str], List[str]]
            ``(new_symbols, removed_symbols)`` where *new_symbols* are symbols
            added to the universe this refresh cycle and *removed_symbols* are
            symbols that were pruned because they no longer appear in the
            exchange response (requires ``PAIR_PRUNE_ENABLED=true``).
        """
        log.info("Refreshing pair universe (market=%s) …", market)

        if market == "spot":
            spot_raw = await self.fetch_all_spot_pairs()
            futures_raw: List[PairInfo] = []
        elif market == "futures":
            spot_raw = []
            futures_raw = await self.fetch_all_futures_pairs()
        else:
            spot_raw, futures_raw = await asyncio.gather(
                self.fetch_all_spot_pairs(),
                self.fetch_all_futures_pairs(),
            )

        # Build a rank-ordered list of ALL fetched symbols for tier assignment.
        # Futures are given priority over spot at the same volume rank so that
        # perpetual contracts (higher OI data availability) reach Tier 1.
        all_fetched: List[PairInfo] = []
        seen_in_fetch: set = set()

        # Merge: futures first (higher data quality), then spot
        # Merge: futures first (higher data quality), then spot
        for p in itertools.chain(futures_raw, spot_raw):
            if p.symbol not in seen_in_fetch:
                seen_in_fetch.add(p.symbol)
                all_fetched.append(p)

        # Classify tiers by volume rank.
        for rank, p in enumerate(all_fetched):
            if rank < TIER1_PAIR_COUNT:
                p.tier = PairTier.TIER1
            elif rank < TIER2_PAIR_COUNT:
                p.tier = PairTier.TIER2
            else:
                p.tier = PairTier.TIER3

        # --- Update the active pair universe ---
        new_symbols: List[str] = []
        for p in all_fetched:
            if p.symbol not in self.pairs:
                new_symbols.append(p.symbol)
                self.pairs[p.symbol] = p
            else:
                # Update mutable fields; preserve historical tracking fields.
                self._prev_volumes[p.symbol] = self.pairs[p.symbol].volume_24h_usd
                self.pairs[p.symbol].volume_24h_usd = p.volume_24h_usd
                self.pairs[p.symbol].tier = p.tier

        # --- Prune delisted / dropped pairs ---
        removed_symbols: List[str] = []
        if PAIR_PRUNE_ENABLED and seen_in_fetch:
            stale = [sym for sym in self.pairs if sym not in seen_in_fetch]
            for sym in stale:
                removed_symbols.append(sym)
                del self.pairs[sym]
                self._prev_volumes.pop(sym, None)
            if removed_symbols:
                log.info(
                    "Pruned %d stale pairs: %s%s",
                    len(removed_symbols),
                    removed_symbols[:10],
                    " …" if len(removed_symbols) > 10 else "",
                )

        tier_counts = {
            PairTier.TIER1: sum(1 for p in self.pairs.values() if p.tier == PairTier.TIER1),
            PairTier.TIER2: sum(1 for p in self.pairs.values() if p.tier == PairTier.TIER2),
            PairTier.TIER3: sum(1 for p in self.pairs.values() if p.tier == PairTier.TIER3),
        }
        log.info(
            "Pair refresh done – total %d pairs (%d new, %d removed) "
            "[T1=%d T2=%d T3=%d]",
            len(self.pairs), len(new_symbols), len(removed_symbols),
            tier_counts[PairTier.TIER1], tier_counts[PairTier.TIER2], tier_counts[PairTier.TIER3],
        )
        return new_symbols, removed_symbols

    def check_promotions(self) -> List[str]:
        """Check Tier 3 pairs for volume surges and promote them to Tier 2.

        A pair is promoted when its current 24h volume exceeds
        ``TIER3_VOLUME_SURGE_MULTIPLIER`` × its recorded previous volume.
        Promoted pairs are immediately assigned :attr:`PairTier.TIER2` so that
        the scanner picks them up on the next Tier 2 scan cycle.

        Returns
        -------
        List[str]
            Symbols that were promoted from Tier 3 → Tier 2.
        """
        promoted: List[str] = []
        for sym, info in self.pairs.items():
            if info.tier != PairTier.TIER3:
                continue
            prev_vol = self._prev_volumes.get(sym, 0.0)
            if prev_vol > 0 and info.volume_24h_usd >= prev_vol * TIER3_VOLUME_SURGE_MULTIPLIER:
                info.tier = PairTier.TIER2
                promoted.append(sym)
                log.info(
                    "Promoted %s Tier 3 → Tier 2 (vol surge: $%,.0f → $%,.0f, ×%.1f)",
                    sym, prev_vol, info.volume_24h_usd,
                    info.volume_24h_usd / prev_vol,
                )
        return promoted

    async def run_periodic_refresh(self) -> None:
        """Infinite loop that refreshes pairs every N hours."""
        while True:
            await self.refresh_pairs()
            await asyncio.sleep(PAIR_FETCH_INTERVAL_HOURS * 3600)

    async def close(self) -> None:
        await self._spot_client.close()
        await self._futures_client.close()
