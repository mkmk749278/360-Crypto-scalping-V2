"""Dynamic Tiering Manager — PR 2/4: Market Watchdog & Dynamic Tiering.

Background service that periodically polls the Binance global 24-hour ticker
endpoints for both spot and futures markets and re-ranks the entire pair
universe into three priority tiers based on a composite score of liquidity
(``quoteVolume``) and volatility (``priceChangePercent``).

Tier definitions
----------------
* **Tier 1 (Hot)** – Top ``DYNAMIC_TIER1_HOT_COUNT`` pairs (default: 50).
  Highest volume + volatility. Receive maximum scan resources: full WebSocket
  coverage, per-cycle REST depth polling, and all signal channels.
* **Tier 2 (Warm)** – Next ``DYNAMIC_TIER2_WARM_COUNT`` pairs (default: 150).
  Scanned on a reduced cadence via REST klines; spot/swing channels only.
* **Tier 3 (Cold)** – All remaining pairs (typically 600+).
  Lightweight monitoring using the aggregate ``bookTicker`` endpoint;
  auto-promoted to Tier 2 on detected volume/volatility surges.

State management
----------------
Tier membership is stored in Redis as plain sets under configurable key names
(``tier_1_active``, ``tier_2_active``, ``tier_3_active`` by default).  When
Redis is unavailable the manager falls back to in-memory sets so the rest of
the engine continues to function.

The scanner and WebSocket manager can call :meth:`TierManager.get_tier` to
quickly look up a symbol's current tier, or inspect
:attr:`TierManager.tier1_symbols`, :attr:`TierManager.tier2_symbols`, and
:attr:`TierManager.tier3_symbols` directly.

Configuration (via environment variables)
------------------------------------------
  DYNAMIC_TIER_ENABLED          – "true" to enable (default: "true")
  DYNAMIC_TIER_POLL_INTERVAL    – seconds between polls (default: "300")
  DYNAMIC_TIER1_HOT_COUNT       – Tier 1 pair count (default: "50")
  DYNAMIC_TIER12_WARM_CUTOFF    – combined Tier 1+2 boundary (default: "200")
  DYNAMIC_TIER_VOLUME_WEIGHT    – volume score weight (default: "0.7")
  DYNAMIC_TIER_VOLATILITY_WEIGHT – volatility score weight (default: "0.3")
  DYNAMIC_TIER1_REDIS_KEY       – Redis set key for Tier 1 (default: "tier_1_active")
  DYNAMIC_TIER2_REDIS_KEY       – Redis set key for Tier 2 (default: "tier_2_active")
  DYNAMIC_TIER3_REDIS_KEY       – Redis set key for Tier 3 (default: "tier_3_active")
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Set

import aiohttp

from config import (
    BINANCE_FUTURES_REST_BASE,
    BINANCE_REST_BASE,
    DYNAMIC_TIER1_HOT_COUNT,
    DYNAMIC_TIER12_WARM_CUTOFF,
    DYNAMIC_TIER1_REDIS_KEY,
    DYNAMIC_TIER2_REDIS_KEY,
    DYNAMIC_TIER3_REDIS_KEY,
    DYNAMIC_TIER_ENABLED,
    DYNAMIC_TIER_POLL_INTERVAL,
    DYNAMIC_TIER_VOLATILITY_WEIGHT,
    DYNAMIC_TIER_VOLUME_WEIGHT,
)
from src.pair_manager import PairTier
from src.utils import get_logger

log = get_logger("tier_manager")

# Stablecoins that must be excluded from tiering (same list as pair_manager).
_STABLECOIN_BLACKLIST: frozenset = frozenset({
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT", "FDUSDUSDT",
    "USD1USDT", "DAIUSDT", "EURUSDT", "USDCBUSD", "USDTDAI",
    "RLUSDUSDT", "PYUSDUSDT", "USDDUSDT", "GUSDUSDT",
    "FRAXUSDT", "LUSDUSDT", "SUSDUSDT", "CUSDUSDT",
})

# Binance 24hr aggregate ticker endpoints (single call returns all symbols).
_FUTURES_TICKER_PATH = "/fapi/v1/ticker/24hr"
_SPOT_TICKER_PATH = "/api/v3/ticker/24hr"

# Request timeout for aggregate ticker fetches (they are fast, but be generous).
_TICKER_TIMEOUT_S: float = 10.0


class TierManager:
    """Background service that maintains dynamic pair-tier assignments.

    Parameters
    ----------
    redis_client:
        Optional :class:`~src.redis_client.RedisClient` instance.  When
        provided and available, tier sets are persisted in Redis so other
        processes (e.g. the WS manager) can read them without importing this
        module.  When ``None`` or unavailable, the manager operates in
        in-memory mode.
    poll_interval:
        Override for :data:`DYNAMIC_TIER_POLL_INTERVAL`.
    tier1_hot_count:
        Override for :data:`DYNAMIC_TIER1_HOT_COUNT`.
    tier12_warm_cutoff:
        Override for :data:`DYNAMIC_TIER12_WARM_CUTOFF`.
    volume_weight:
        Override for :data:`DYNAMIC_TIER_VOLUME_WEIGHT`.
    volatility_weight:
        Override for :data:`DYNAMIC_TIER_VOLATILITY_WEIGHT`.
    """

    def __init__(
        self,
        redis_client: Optional[object] = None,
        poll_interval: Optional[float] = None,
        tier1_hot_count: Optional[int] = None,
        tier12_warm_cutoff: Optional[int] = None,
        volume_weight: Optional[float] = None,
        volatility_weight: Optional[float] = None,
    ) -> None:
        self._redis = redis_client
        self._poll_interval: float = (
            poll_interval if poll_interval is not None else DYNAMIC_TIER_POLL_INTERVAL
        )
        self._tier1_count: int = (
            tier1_hot_count if tier1_hot_count is not None else DYNAMIC_TIER1_HOT_COUNT
        )
        self._tier12_cutoff: int = (
            tier12_warm_cutoff if tier12_warm_cutoff is not None else DYNAMIC_TIER12_WARM_CUTOFF
        )
        self._volume_weight: float = (
            volume_weight if volume_weight is not None else DYNAMIC_TIER_VOLUME_WEIGHT
        )
        self._volatility_weight: float = (
            volatility_weight if volatility_weight is not None else DYNAMIC_TIER_VOLATILITY_WEIGHT
        )

        # In-memory tier sets (always maintained; Redis mirrors these when available).
        self._tier1: Set[str] = set()
        self._tier2: Set[str] = set()
        self._tier3: Set[str] = set()

        # Reverse lookup: symbol → PairTier for O(1) get_tier() calls.
        self._symbol_tier: Dict[str, PairTier] = {}

        # Timestamps for monitoring.
        self._last_poll_time: float = 0.0
        self._last_poll_symbol_count: int = 0

        # Background task handle.
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background polling loop."""
        if not DYNAMIC_TIER_ENABLED:
            log.info("TierManager disabled by configuration – skipping start")
            return
        if self._task is not None and not self._task.done():
            log.debug("TierManager already running")
            return
        # Run an initial poll immediately so tiers are populated before the
        # first scan cycle.
        await self._poll_tickers()
        self._task = asyncio.create_task(self._poll_loop(), name="tier_manager")
        log.info(
            "TierManager started (poll_interval={}s, tier1={}, tier2_cutoff={})",
            self._poll_interval,
            self._tier1_count,
            self._tier12_cutoff,
        )

    async def stop(self) -> None:
        """Cancel the polling loop and release resources."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        log.info("TierManager stopped")

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def get_tier(self, symbol: str) -> PairTier:
        """Return the current :class:`~src.pair_manager.PairTier` for *symbol*.

        Falls back to :attr:`PairTier.TIER3` for unknown symbols so the caller
        never needs to handle a missing-key case.

        Parameters
        ----------
        symbol:
            Normalised trading pair string, e.g. ``"BTCUSDT"``.
        """
        return self._symbol_tier.get(symbol, PairTier.TIER3)

    @property
    def tier1_symbols(self) -> List[str]:
        """Snapshot of current Tier 1 (Hot) symbols."""
        return list(self._tier1)

    @property
    def tier2_symbols(self) -> List[str]:
        """Snapshot of current Tier 2 (Warm) symbols."""
        return list(self._tier2)

    @property
    def tier3_symbols(self) -> List[str]:
        """Snapshot of current Tier 3 (Cold) symbols."""
        return list(self._tier3)

    @property
    def last_poll_time(self) -> float:
        """Monotonic timestamp of the most recent successful poll."""
        return self._last_poll_time

    @property
    def last_poll_symbol_count(self) -> int:
        """Number of symbols ranked in the most recent poll."""
        return self._last_poll_symbol_count

    def status_text(self) -> str:
        """Return a one-line status string for telemetry / Telegram."""
        elapsed = time.monotonic() - self._last_poll_time if self._last_poll_time else -1
        return (
            f"TierManager | T1={len(self._tier1)} T2={len(self._tier2)} "
            f"T3={len(self._tier3)} | last_poll={elapsed:.0f}s ago"
        )

    # ------------------------------------------------------------------
    # Background polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Infinite loop: poll tickers, update tiers, sleep."""
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._poll_tickers()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("TierManager poll error: {}", exc)

    # ------------------------------------------------------------------
    # Core ranking logic
    # ------------------------------------------------------------------

    async def _poll_tickers(self) -> None:
        """Fetch aggregate 24hr tickers, rank pairs, and update tier sets."""
        session = await self._ensure_session()

        # Fetch both markets concurrently for efficiency (single aggregate call each).
        futures_data, spot_data = await asyncio.gather(
            self._fetch_ticker(session, BINANCE_FUTURES_REST_BASE, _FUTURES_TICKER_PATH),
            self._fetch_ticker(session, BINANCE_REST_BASE, _SPOT_TICKER_PATH),
            return_exceptions=True,
        )

        all_tickers: List[dict] = []

        if isinstance(futures_data, list):
            all_tickers.extend(futures_data)
        else:
            log.warning("TierManager: futures ticker fetch failed – {}", futures_data)

        if isinstance(spot_data, list):
            # Deduplicate: futures take priority (higher OI data quality).
            futures_symbols = {t["symbol"] for t in all_tickers}
            for t in spot_data:
                if t["symbol"] not in futures_symbols:
                    all_tickers.append(t)
        else:
            log.warning("TierManager: spot ticker fetch failed – {}", spot_data)

        if not all_tickers:
            log.warning("TierManager: no ticker data received – tiers unchanged")
            return

        # Filter: USDT pairs only, exclude stablecoins, require non-zero volume.
        usdt_tickers = [
            t for t in all_tickers
            if t.get("symbol", "").endswith("USDT")
            and t.get("symbol", "") not in _STABLECOIN_BLACKLIST
            and float(t.get("quoteVolume", 0)) > 0
        ]

        ranked = self._rank_tickers(usdt_tickers)
        await self._apply_tiers(ranked)

        self._last_poll_time = time.monotonic()
        self._last_poll_symbol_count = len(ranked)
        log.info(
            "TierManager ranked {} pairs → T1={} T2={} T3={}",
            len(ranked),
            len(self._tier1),
            len(self._tier2),
            len(self._tier3),
        )

    def _rank_tickers(self, tickers: List[dict]) -> List[str]:
        """Compute composite score and return symbols sorted highest → lowest.

        Score = ``volume_weight × normalised_quote_volume``
              + ``volatility_weight × normalised_abs_price_change``

        Both components are normalised to [0, 1] across the current universe
        so neither dominates purely by magnitude.
        """
        if not tickers:
            return []

        volumes = [float(t.get("quoteVolume", 0)) for t in tickers]
        changes = [abs(float(t.get("priceChangePercent", 0))) for t in tickers]

        max_vol = max(volumes) or 1.0
        max_chg = max(changes) or 1.0

        scored: List[tuple] = []
        for t, vol, chg in zip(tickers, volumes, changes):
            score = (
                self._volume_weight * (vol / max_vol)
                + self._volatility_weight * (chg / max_chg)
            )
            scored.append((score, t["symbol"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [sym for _, sym in scored]

    async def _apply_tiers(self, ranked_symbols: List[str]) -> None:
        """Update in-memory tier sets and sync to Redis."""
        tier1: Set[str] = set()
        tier2: Set[str] = set()
        tier3: Set[str] = set()
        symbol_tier: Dict[str, PairTier] = {}

        for rank, sym in enumerate(ranked_symbols):
            if rank < self._tier1_count:
                tier1.add(sym)
                symbol_tier[sym] = PairTier.TIER1
            elif rank < self._tier12_cutoff:
                tier2.add(sym)
                symbol_tier[sym] = PairTier.TIER2
            else:
                tier3.add(sym)
                symbol_tier[sym] = PairTier.TIER3

        # Detect and log promotions / demotions for observability.
        self._log_tier_changes(tier1, tier2, tier3)

        self._tier1 = tier1
        self._tier2 = tier2
        self._tier3 = tier3
        self._symbol_tier = symbol_tier

        await self._sync_to_redis(tier1, tier2, tier3)

    def _log_tier_changes(
        self, new_tier1: Set[str], new_tier2: Set[str], new_tier3: Set[str]
    ) -> None:
        """Log promotions and demotions between tier cycles at DEBUG level."""
        if not self._symbol_tier:
            return  # First run; no previous state to compare.

        new_map: Dict[str, PairTier] = {}
        for s in new_tier1:
            new_map[s] = PairTier.TIER1
        for s in new_tier2:
            new_map[s] = PairTier.TIER2
        for s in new_tier3:
            new_map[s] = PairTier.TIER3

        for sym, old_tier in self._symbol_tier.items():
            new_tier = new_map.get(sym)
            if new_tier is None or new_tier == old_tier:
                continue
            log.debug(
                "TierManager {} {}→{}", sym, old_tier.value, new_tier.value
            )

    # ------------------------------------------------------------------
    # Redis synchronisation
    # ------------------------------------------------------------------

    async def _sync_to_redis(
        self, tier1: Set[str], tier2: Set[str], tier3: Set[str]
    ) -> None:
        """Atomically replace Redis tier sets with the current membership."""
        if self._redis is None or not self._redis.available:
            return

        redis = self._redis.client
        if redis is None:
            return

        try:
            pipe = redis.pipeline()
            for key, members in (
                (DYNAMIC_TIER1_REDIS_KEY, tier1),
                (DYNAMIC_TIER2_REDIS_KEY, tier2),
                (DYNAMIC_TIER3_REDIS_KEY, tier3),
            ):
                pipe.delete(key)
                if members:
                    pipe.sadd(key, *members)
            await pipe.execute()
        except Exception as exc:
            log.warning("TierManager: Redis sync failed – {}", exc)
            self._redis.mark_unavailable("tier_sync", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    @staticmethod
    async def _fetch_ticker(
        session: aiohttp.ClientSession, base_url: str, path: str
    ) -> List[dict]:
        """Fetch a Binance aggregate 24hr ticker endpoint and return the JSON list."""
        url = base_url.rstrip("/") + path
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=_TICKER_TIMEOUT_S)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            if not isinstance(data, list):
                raise ValueError(f"Unexpected response type from {url}: {type(data)}")
            return data
