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

import time

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
    TOP50_FUTURES_COUNT,
    TOP50_UPDATE_INTERVAL_SECONDS,
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
    volatility_24h: float = 0.0       # 24h price change % (absolute)
    spread_avg: float = 0.0           # Average bid-ask spread %
    rank_score: float = 0.0           # Composite ranking score


class PairManager:
    """Fetches and maintains the active pair universe."""

    def __init__(self) -> None:
        self.pairs: Dict[str, PairInfo] = {}
        self._spot_client = BinanceClient("spot")
        self._futures_client = BinanceClient("futures")
        # Previous 24h volume per symbol — used for Tier 3 volume surge detection.
        self._prev_volumes: Dict[str, float] = {}
        # Historical pair metrics for analytics.
        self._pair_metrics_history: Dict[str, List[Dict]] = {}
        # Top-50 futures cache (PR1): cached list + last refresh timestamp.
        self._top50_futures_cache: List[str] = []
        self._top50_last_refresh: float = 0.0

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

    # ------------------------------------------------------------------
    # Dynamic pair ranking and scoring
    # ------------------------------------------------------------------

    def rank_pairs(
        self,
        volume_weight: float = 0.5,
        volatility_weight: float = 0.3,
        liquidity_weight: float = 0.2,
    ) -> List[str]:
        """Rank all active pairs by a composite score of volume, volatility, and liquidity.

        Higher scores indicate pairs that are more suitable for scalping:
        high volume (tight spreads), moderate volatility (tradeable moves),
        and good liquidity (execution quality).

        Parameters
        ----------
        volume_weight:
            Weight for the volume component (0–1).
        volatility_weight:
            Weight for the volatility component (0–1).
        liquidity_weight:
            Weight for the liquidity / spread component (0–1).

        Returns
        -------
        List[str]
            Symbols sorted by rank score (best first).
        """
        if not self.pairs:
            return []

        max_vol = max((p.volume_24h_usd for p in self.pairs.values()), default=1.0)
        max_vol = max(max_vol, 1.0)
        max_volatility = max(
            (p.volatility_24h for p in self.pairs.values()), default=1.0
        )
        max_volatility = max(max_volatility, 0.01)

        for info in self.pairs.values():
            vol_score = info.volume_24h_usd / max_vol
            # Moderate volatility is best: penalise both extremes
            if max_volatility > 0:
                vol_norm = info.volatility_24h / max_volatility
                volatility_score = 1.0 - abs(vol_norm - 0.5) * 2.0
            else:
                volatility_score = 0.5
            # Lower spread = better liquidity
            liq_score = max(0.0, 1.0 - info.spread_avg * 100.0) if info.spread_avg > 0 else 0.5

            info.rank_score = (
                vol_score * volume_weight
                + volatility_score * volatility_weight
                + liq_score * liquidity_weight
            )

        ranked = sorted(self.pairs.keys(), key=lambda s: self.pairs[s].rank_score, reverse=True)
        return ranked

    def get_top_ranked_pairs(self, n: int = 20) -> List[str]:
        """Return the top *n* pairs by composite rank score.

        Calls :meth:`rank_pairs` internally to recompute scores.
        """
        ranked = self.rank_pairs()
        return ranked[:n]

    def update_pair_volatility(self, symbol: str, volatility_24h: float) -> None:
        """Update the 24h volatility for a specific pair.

        Parameters
        ----------
        symbol:
            Trading pair.
        volatility_24h:
            Absolute 24h price change percentage.
        """
        if symbol in self.pairs:
            self.pairs[symbol].volatility_24h = volatility_24h

    def update_pair_spread(self, symbol: str, spread_avg: float) -> None:
        """Update the average spread for a specific pair.

        Parameters
        ----------
        symbol:
            Trading pair.
        spread_avg:
            Average bid-ask spread as a decimal (e.g. 0.001 = 0.1%).
        """
        if symbol in self.pairs:
            self.pairs[symbol].spread_avg = spread_avg

    def record_pair_metrics(self, symbol: str) -> None:
        """Snapshot current metrics for a pair into the history store.

        Used for analytics and historical analysis of pair behaviour.
        """
        info = self.pairs.get(symbol)
        if info is None:
            return
        snapshot = {
            "timestamp": time.time(),
            "volume_24h_usd": info.volume_24h_usd,
            "volatility_24h": info.volatility_24h,
            "spread_avg": info.spread_avg,
            "tier": info.tier.value,
            "rank_score": info.rank_score,
        }
        self._pair_metrics_history.setdefault(symbol, []).append(snapshot)
        # Cap history at 500 entries per pair
        if len(self._pair_metrics_history[symbol]) > 500:
            self._pair_metrics_history[symbol] = self._pair_metrics_history[symbol][-500:]

    def get_pair_metrics_history(self, symbol: str) -> List[Dict]:
        """Return the stored metric snapshots for a pair."""
        return list(self._pair_metrics_history.get(symbol, []))

    def detect_volume_spikes(self, multiplier: float = 3.0) -> List[str]:
        """Detect pairs with sudden volume spikes.

        A pair is flagged when its current 24h volume exceeds
        *multiplier* × its previous recorded volume.

        Parameters
        ----------
        multiplier:
            Volume increase factor to trigger a spike detection.

        Returns
        -------
        List[str]
            Symbols with detected volume spikes.
        """
        spiked: List[str] = []
        for sym, info in self.pairs.items():
            prev = self._prev_volumes.get(sym, 0.0)
            if prev > 0 and info.volume_24h_usd >= prev * multiplier:
                spiked.append(sym)
                log.info(
                    "Volume spike detected: %s ($%,.0f → $%,.0f, ×%.1f)",
                    sym, prev, info.volume_24h_usd,
                    info.volume_24h_usd / prev,
                )
        return spiked

    # ------------------------------------------------------------------
    # Integrated watchlist management (PR: Dynamic Pair Selection)
    # ------------------------------------------------------------------

    def update_watchlist(
        self,
        top_n: int = 50,
        volume_spike_multiplier: float = 3.0,
    ) -> Dict[str, List[str]]:
        """Update the active watchlist by combining ranking and volume spikes.

        This method integrates multiple pair selection signals:
        1. Top N pairs by composite rank score (volume + volatility + liquidity)
        2. Pairs with sudden volume spikes (regardless of rank)
        3. De-duplicated combined list

        Parameters
        ----------
        top_n:
            Number of top-ranked pairs to include.
        volume_spike_multiplier:
            Volume surge factor for spike detection.

        Returns
        -------
        Dict with keys:
            ``"top_ranked"``: Top N pairs by score.
            ``"volume_spikes"``: Pairs with volume spikes.
            ``"watchlist"``: Combined de-duplicated watchlist.
        """
        top_ranked = self.get_top_ranked_pairs(n=top_n)
        spiked = self.detect_volume_spikes(multiplier=volume_spike_multiplier)

        # Combine and de-duplicate: ranked pairs first, then spike additions
        seen = set(top_ranked)
        combined = list(top_ranked)
        for sym in spiked:
            if sym not in seen:
                combined.append(sym)
                seen.add(sym)

        log.info(
            "Watchlist updated: %d top-ranked + %d volume spikes = %d total",
            len(top_ranked), len(spiked), len(combined),
        )

        return {
            "top_ranked": top_ranked,
            "volume_spikes": spiked,
            "watchlist": combined,
        }

    def suppress_low_quality_signals(
        self,
        symbol: str,
        confidence: float = 0.0,
        min_volume_usd: float = 1_000_000.0,
        min_confidence: float = 55.0,
        min_rank_score: float = 0.1,
    ) -> Tuple[bool, str]:
        """Check whether a signal on a pair should be suppressed.

        A signal is suppressed when the pair fails quality checks:
        - Volume below minimum threshold
        - Confidence below minimum threshold
        - Rank score below minimum (pair not in active watchlist)

        Parameters
        ----------
        symbol:
            Trading pair to check.
        confidence:
            Signal confidence score (0–100).
        min_volume_usd:
            Minimum 24h volume threshold in USD.
        min_confidence:
            Minimum confidence threshold.
        min_rank_score:
            Minimum composite rank score.

        Returns
        -------
        Tuple[bool, str]
            ``(suppressed, reason)`` where suppressed is True when the
            signal should be dropped, and reason explains why.
        """
        info = self.pairs.get(symbol)

        if info is None:
            return True, f"Pair {symbol} not in active universe"

        if info.volume_24h_usd < min_volume_usd:
            return True, (
                f"Low volume: ${info.volume_24h_usd:,.0f} < "
                f"${min_volume_usd:,.0f}"
            )

        if confidence < min_confidence:
            return True, (
                f"Low confidence: {confidence:.1f} < {min_confidence:.1f}"
            )

        if 0 < info.rank_score < min_rank_score:
            return True, (
                f"Low rank score: {info.rank_score:.3f} < {min_rank_score:.3f}"
            )

        return False, ""

    def get_watchlist_summary(self) -> Dict[str, object]:
        """Return a summary of the current pair universe for analytics.

        Returns
        -------
        Dict
            Contains tier counts, top pairs, volume stats.
        """
        if not self.pairs:
            return {
                "total_pairs": 0,
                "tier_counts": {},
                "top_5_by_volume": [],
                "avg_volume_usd": 0.0,
            }

        tier_counts: Dict[str, int] = {}
        for p in self.pairs.values():
            tier_counts[p.tier.value] = tier_counts.get(p.tier.value, 0) + 1

        sorted_by_vol = sorted(
            self.pairs.values(),
            key=lambda p: p.volume_24h_usd,
            reverse=True,
        )
        top_5 = [
            {"symbol": p.symbol, "volume_24h_usd": p.volume_24h_usd, "tier": p.tier.value}
            for p in sorted_by_vol[:5]
        ]
        avg_vol = (
            sum(p.volume_24h_usd for p in self.pairs.values()) / len(self.pairs)
            if self.pairs
            else 0.0
        )

        return {
            "total_pairs": len(self.pairs),
            "tier_counts": tier_counts,
            "top_5_by_volume": top_5,
            "avg_volume_usd": avg_vol,
        }

    # ------------------------------------------------------------------
    # Top-50 futures-only API (PR1)
    # ------------------------------------------------------------------

    def is_top50_futures(self, symbol: str) -> bool:
        """Return True when *symbol* is in the current top-50 futures list.

        Uses the most recent result of :meth:`get_top50_futures_pairs` (or the
        cached snapshot if the refresh interval has not elapsed).  Returns
        ``False`` when the cache is empty (not yet populated).
        """
        return symbol in self._top50_futures_cache

    def get_top50_futures_pairs(self) -> List[str]:
        """Return the cached list of top-50 futures pairs (does not refresh).

        Call :meth:`refresh_top50_futures` to populate / update the cache.
        """
        return list(self._top50_futures_cache)

    async def refresh_top50_futures(
        self,
        count: Optional[int] = None,
        force: bool = False,
    ) -> List[str]:
        """Refresh and return the top-*count* USDT-M futures pairs by volume.

        The result is cached and the method is rate-limited to at most one
        real fetch per ``TOP50_UPDATE_INTERVAL_SECONDS`` seconds (default 90s).
        Pass ``force=True`` to bypass the interval guard.

        Parameters
        ----------
        count:
            Number of top futures pairs to keep.  Defaults to
            ``TOP50_FUTURES_COUNT`` (env ``TOP50_FUTURES_COUNT``, default 50).
        force:
            Skip the minimum-interval guard and always fetch.

        Returns
        -------
        List[str]
            Symbols of the top-*count* futures pairs, ordered by 24h volume
            descending.
        """
        count = count or TOP50_FUTURES_COUNT
        now = time.monotonic()
        if not force and (now - self._top50_last_refresh) < TOP50_UPDATE_INTERVAL_SECONDS:
            log.debug(
                "refresh_top50_futures: interval not elapsed (%.0fs < %ds), "
                "returning cached list of %d pairs",
                now - self._top50_last_refresh,
                TOP50_UPDATE_INTERVAL_SECONDS,
                len(self._top50_futures_cache),
            )
            return list(self._top50_futures_cache)

        futures_pairs = await self.fetch_top_futures_pairs(limit=count)
        self._top50_futures_cache = [p.symbol for p in futures_pairs[:count]]
        self._top50_last_refresh = time.monotonic()
        log.info(
            "Top-50 futures refreshed: %d pairs (requested=%d)",
            len(self._top50_futures_cache), count,
        )
        # Ensure all top-50 futures are registered in self.pairs so the
        # scanner and AI engine can look them up by symbol.
        for p in futures_pairs[:count]:
            if p.symbol not in self.pairs:
                p.tier = PairTier.TIER1
                self.pairs[p.symbol] = p
            else:
                self._prev_volumes[p.symbol] = self.pairs[p.symbol].volume_24h_usd
                self.pairs[p.symbol].volume_24h_usd = p.volume_24h_usd
                self.pairs[p.symbol].tier = PairTier.TIER1
        return list(self._top50_futures_cache)

    async def run_periodic_top50_refresh(self) -> None:
        """Infinite loop that refreshes the top-50 futures list at the
        configured minimum interval (``TOP50_UPDATE_INTERVAL_SECONDS``).
        """
        while True:
            try:
                await self.refresh_top50_futures(force=True)
            except Exception as exc:
                log.warning("run_periodic_top50_refresh error: %s", exc)
            await asyncio.sleep(TOP50_UPDATE_INTERVAL_SECONDS)

    async def close(self) -> None:
        await self._spot_client.close()
        await self._futures_client.close()
