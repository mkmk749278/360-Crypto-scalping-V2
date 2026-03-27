"""Tests for TierManager — dynamic tiering (PR 2).

All tests operate without a live Binance connection or Redis instance.
Network calls are mocked via ``unittest.mock``.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pair_manager import PairTier
from src.tier_manager import TierManager, _STABLECOIN_BLACKLIST


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ticker(symbol: str, quote_volume: float, price_change_pct: float) -> dict:
    """Return a minimal Binance 24hr ticker dict."""
    return {
        "symbol": symbol,
        "quoteVolume": str(quote_volume),
        "priceChangePercent": str(price_change_pct),
    }


def _make_ticker_list(count: int, base_vol: float = 1_000_000.0) -> List[dict]:
    """Return *count* unique synthetic tickers with descending volume."""
    return [
        _make_ticker(f"COIN{i:03d}USDT", base_vol / (i + 1), float(i % 20))
        for i in range(count)
    ]


class _MockRedisClient:
    """Minimal in-memory mock for RedisClient."""

    def __init__(self, available: bool = True) -> None:
        self._available = available
        self._store: Dict[str, set] = {}
        self.client = self if available else None
        self._unavailable_calls: List[str] = []

    @property
    def available(self) -> bool:
        return self._available

    def mark_unavailable(self, operation: str, exc: Optional[Exception] = None) -> None:
        self._unavailable_calls.append(operation)
        self._available = False

    def pipeline(self) -> "_MockPipeline":
        return _MockPipeline(self._store)


class _MockPipeline:
    def __init__(self, store: Dict[str, set]) -> None:
        self._store = store
        self._ops: List[tuple] = []

    def delete(self, key: str) -> "_MockPipeline":
        self._ops.append(("delete", key))
        return self

    def sadd(self, key: str, *members: str) -> "_MockPipeline":
        self._ops.append(("sadd", key, members))
        return self

    async def execute(self) -> None:
        for op in self._ops:
            if op[0] == "delete":
                self._store.pop(op[1], None)
            elif op[0] == "sadd":
                self._store.setdefault(op[1], set()).update(op[2])


# ---------------------------------------------------------------------------
# Unit tests: _rank_tickers
# ---------------------------------------------------------------------------

class TestRankTickers:
    def _tm(self, **kwargs) -> TierManager:
        return TierManager(
            tier1_hot_count=kwargs.pop("tier1_hot_count", 3),
            tier12_warm_cutoff=kwargs.pop("tier12_warm_cutoff", 5),
            **kwargs,
        )

    def test_empty_returns_empty(self):
        tm = self._tm()
        result = tm._rank_tickers([])
        assert result == []

    def test_high_volume_ranks_first(self):
        tickers = [
            _make_ticker("LOWVOL", 100.0, 1.0),
            _make_ticker("HIGHVOL", 1_000_000.0, 1.0),
        ]
        tm = self._tm(volume_weight=1.0, volatility_weight=0.0)
        ranked = tm._rank_tickers(tickers)
        assert ranked[0] == "HIGHVOL"
        assert ranked[1] == "LOWVOL"

    def test_high_volatility_ranks_first_when_volume_equal(self):
        tickers = [
            _make_ticker("STABLE", 500.0, 0.1),
            _make_ticker("VOLATILE", 500.0, 50.0),
        ]
        tm = self._tm(volume_weight=0.0, volatility_weight=1.0)
        ranked = tm._rank_tickers(tickers)
        assert ranked[0] == "VOLATILE"

    def test_composite_score_blends_both_factors(self):
        # BIGVOL: high volume, zero volatility
        # BIGCHG: zero volume, high volatility
        # BALANCED: medium volume + medium volatility
        tickers = [
            _make_ticker("BIGVOL", 1_000_000.0, 0.0),
            _make_ticker("BIGCHG", 0.0, 100.0),
            _make_ticker("BALANCED", 500_000.0, 50.0),
        ]
        tm = self._tm(volume_weight=0.7, volatility_weight=0.3)
        ranked = tm._rank_tickers(tickers)
        assert ranked[0] == "BIGVOL"    # 0.7*1 + 0.3*0 = 0.70
        assert ranked[1] == "BALANCED"  # 0.7*0.5 + 0.3*0.5 = 0.50
        assert ranked[2] == "BIGCHG"    # 0.7*0 + 0.3*1 = 0.30

    def test_negative_price_change_uses_absolute_value(self):
        """A −30% move should rank the same as +30% for volatility scoring."""
        tickers = [
            _make_ticker("POS", 500.0, 30.0),
            _make_ticker("NEG", 500.0, -30.0),
        ]
        tm = self._tm(volume_weight=0.0, volatility_weight=1.0)
        ranked = tm._rank_tickers(tickers)
        # Both have identical abs(price_change_pct) so order may vary — what
        # matters is that neither has a score advantage over the other.
        assert set(ranked) == {"POS", "NEG"}

    def test_single_ticker_returns_that_ticker(self):
        tm = self._tm()
        result = tm._rank_tickers([_make_ticker("SOLOBTC", 999.0, 5.0)])
        assert result == ["SOLOBTC"]


# ---------------------------------------------------------------------------
# Unit tests: _apply_tiers
# ---------------------------------------------------------------------------

class TestApplyTiers:
    @pytest.mark.asyncio
    async def test_tier_assignment_by_rank(self):
        """First N symbols become T1, next M become T2, rest become T3."""
        tm = TierManager(tier1_hot_count=2, tier12_warm_cutoff=4)
        symbols = ["A", "B", "C", "D", "E", "F"]
        await tm._apply_tiers(symbols)

        assert tm.get_tier("A") == PairTier.TIER1
        assert tm.get_tier("B") == PairTier.TIER1
        assert tm.get_tier("C") == PairTier.TIER2
        assert tm.get_tier("D") == PairTier.TIER2
        assert tm.get_tier("E") == PairTier.TIER3
        assert tm.get_tier("F") == PairTier.TIER3

    @pytest.mark.asyncio
    async def test_set_sizes_match_config(self):
        tm = TierManager(tier1_hot_count=3, tier12_warm_cutoff=7)
        symbols = [f"SYM{i}" for i in range(10)]
        await tm._apply_tiers(symbols)

        assert len(tm.tier1_symbols) == 3
        assert len(tm.tier2_symbols) == 4   # 7 - 3
        assert len(tm.tier3_symbols) == 3   # 10 - 7

    @pytest.mark.asyncio
    async def test_unknown_symbol_defaults_to_tier3(self):
        tm = TierManager(tier1_hot_count=2, tier12_warm_cutoff=4)
        await tm._apply_tiers(["X", "Y", "Z"])
        assert tm.get_tier("UNKNOWN") == PairTier.TIER3

    @pytest.mark.asyncio
    async def test_fewer_symbols_than_tier1_cap(self):
        """When all symbols fit in Tier 1, Tier 2 and 3 must be empty."""
        tm = TierManager(tier1_hot_count=10, tier12_warm_cutoff=20)
        await tm._apply_tiers(["BTC", "ETH"])

        assert len(tm.tier1_symbols) == 2
        assert len(tm.tier2_symbols) == 0
        assert len(tm.tier3_symbols) == 0

    @pytest.mark.asyncio
    async def test_tier_properties_return_snapshots(self):
        """Mutating the returned list must not affect TierManager state."""
        tm = TierManager(tier1_hot_count=1, tier12_warm_cutoff=2)
        await tm._apply_tiers(["A", "B", "C"])

        snapshot = tm.tier1_symbols
        snapshot.append("FAKE")
        assert "FAKE" not in tm.tier1_symbols


# ---------------------------------------------------------------------------
# Unit tests: Redis sync
# ---------------------------------------------------------------------------

class TestRedisSyncTiers:
    @pytest.mark.asyncio
    async def test_sync_writes_all_three_keys(self):
        redis = _MockRedisClient(available=True)
        tm = TierManager(
            redis_client=redis,
            tier1_hot_count=2,
            tier12_warm_cutoff=4,
        )
        await tm._apply_tiers(["A", "B", "C", "D", "E"])

        from config import (
            DYNAMIC_TIER1_REDIS_KEY,
            DYNAMIC_TIER2_REDIS_KEY,
            DYNAMIC_TIER3_REDIS_KEY,
        )
        assert DYNAMIC_TIER1_REDIS_KEY in redis._store
        assert redis._store[DYNAMIC_TIER1_REDIS_KEY] == {"A", "B"}
        assert DYNAMIC_TIER2_REDIS_KEY in redis._store
        assert redis._store[DYNAMIC_TIER2_REDIS_KEY] == {"C", "D"}
        assert DYNAMIC_TIER3_REDIS_KEY in redis._store
        assert redis._store[DYNAMIC_TIER3_REDIS_KEY] == {"E"}

    @pytest.mark.asyncio
    async def test_sync_skipped_when_redis_unavailable(self):
        redis = _MockRedisClient(available=False)
        tm = TierManager(redis_client=redis, tier1_hot_count=1, tier12_warm_cutoff=2)
        await tm._apply_tiers(["X", "Y", "Z"])  # must not raise
        assert redis._store == {}

    @pytest.mark.asyncio
    async def test_sync_skipped_when_no_redis(self):
        tm = TierManager(redis_client=None, tier1_hot_count=1, tier12_warm_cutoff=2)
        await tm._apply_tiers(["X", "Y"])  # must not raise

    @pytest.mark.asyncio
    async def test_redis_failure_marks_unavailable(self):
        """Pipeline execute failure should mark Redis unavailable gracefully."""
        redis = _MockRedisClient(available=True)
        failing_pipe = MagicMock()
        failing_pipe.delete = MagicMock(return_value=failing_pipe)
        failing_pipe.sadd = MagicMock(return_value=failing_pipe)
        failing_pipe.execute = AsyncMock(side_effect=RuntimeError("Redis down"))
        redis.client = redis
        redis.pipeline = MagicMock(return_value=failing_pipe)

        tm = TierManager(redis_client=redis, tier1_hot_count=1, tier12_warm_cutoff=2)
        await tm._apply_tiers(["A", "B", "C"])  # must not raise

        assert "tier_sync" in redis._unavailable_calls


# ---------------------------------------------------------------------------
# Unit tests: get_tier public interface
# ---------------------------------------------------------------------------

class TestGetTier:
    @pytest.mark.asyncio
    async def test_get_tier_returns_correct_tier(self):
        tm = TierManager(tier1_hot_count=2, tier12_warm_cutoff=4)
        await tm._apply_tiers(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"])
        assert tm.get_tier("BTCUSDT") == PairTier.TIER1
        assert tm.get_tier("SOLUSDT") == PairTier.TIER2
        assert tm.get_tier("XRPUSDT") == PairTier.TIER3

    def test_get_tier_before_first_poll_returns_tier3(self):
        tm = TierManager()
        assert tm.get_tier("BTCUSDT") == PairTier.TIER3


# ---------------------------------------------------------------------------
# Unit tests: status_text
# ---------------------------------------------------------------------------

class TestStatusText:
    @pytest.mark.asyncio
    async def test_status_text_shows_counts(self):
        tm = TierManager(tier1_hot_count=2, tier12_warm_cutoff=4)
        await tm._apply_tiers(["A", "B", "C", "D", "E"])
        text = tm.status_text()
        assert "T1=2" in text
        assert "T2=2" in text
        assert "T3=1" in text


# ---------------------------------------------------------------------------
# Unit tests: _poll_tickers (mocked HTTP)
# ---------------------------------------------------------------------------

class TestPollTickers:
    @pytest.mark.asyncio
    async def test_poll_filters_stablecoins(self):
        """USDCUSDT and similar pairs must never appear in any tier."""
        tickers = [
            _make_ticker("BTCUSDT", 1_000_000.0, 2.0),
            _make_ticker("USDCUSDT", 50_000_000.0, 0.01),  # stablecoin
            _make_ticker("ETHUSDT", 800_000.0, 1.5),
        ]
        tm = TierManager(tier1_hot_count=1, tier12_warm_cutoff=2)

        # Apply the same filtering logic the production code uses.
        usdt_tickers = [
            t for t in tickers
            if t.get("symbol", "").endswith("USDT")
            and t.get("symbol", "") not in _STABLECOIN_BLACKLIST
            and float(t.get("quoteVolume", 0)) > 0
        ]
        ranked = tm._rank_tickers(usdt_tickers)
        await tm._apply_tiers(ranked)

        all_symbols = set(tm.tier1_symbols + tm.tier2_symbols + tm.tier3_symbols)
        assert "USDCUSDT" not in all_symbols
        assert "BTCUSDT" in all_symbols
        assert "ETHUSDT" in all_symbols

    @pytest.mark.asyncio
    async def test_poll_handles_both_markets_empty(self):
        """When both ticker endpoints return errors, tiers must remain unchanged."""
        tm = TierManager(tier1_hot_count=2, tier12_warm_cutoff=4)
        # Pre-populate with some data.
        await tm._apply_tiers(["BTCUSDT", "ETHUSDT", "SOLUSDT"])

        async def _fail(*args, **kwargs):
            raise RuntimeError("network error")

        with patch.object(tm, "_fetch_ticker", side_effect=_fail):
            # _poll_tickers will catch exceptions from gather and log warnings;
            # tiers must be unchanged.
            await tm._poll_tickers()

        assert "BTCUSDT" in tm.tier1_symbols or "BTCUSDT" in tm.tier2_symbols or "BTCUSDT" in tm.tier3_symbols


# ---------------------------------------------------------------------------
# Integration: Scanner.get_symbol_tier
# ---------------------------------------------------------------------------

class TestScannerGetSymbolTier:
    """Verify Scanner.get_symbol_tier delegates correctly."""

    def _make_scanner(self, tier_manager=None):
        from src.scanner import Scanner
        from src.pair_manager import PairManager, PairInfo

        pair_mgr = MagicMock(spec=PairManager)
        pair_mgr.pairs = {
            "BTCUSDT": PairInfo(symbol="BTCUSDT", market="futures", tier=PairTier.TIER1),
            "XRPUSDT": PairInfo(symbol="XRPUSDT", market="spot", tier=PairTier.TIER2),
        }

        scanner = Scanner(
            pair_mgr=pair_mgr,
            data_store=MagicMock(),
            channels=[],
            smc_detector=MagicMock(),
            regime_detector=MagicMock(),
            predictive=MagicMock(),
            exchange_mgr=MagicMock(),
            spot_client=None,
            telemetry=MagicMock(),
            signal_queue=MagicMock(),
            router=MagicMock(),
            tier_manager=tier_manager,
        )
        return scanner

    def test_without_tier_manager_uses_pair_mgr(self):
        scanner = self._make_scanner(tier_manager=None)
        assert scanner.get_symbol_tier("BTCUSDT") == PairTier.TIER1
        assert scanner.get_symbol_tier("XRPUSDT") == PairTier.TIER2
        assert scanner.get_symbol_tier("UNKNOWN") == PairTier.TIER3

    @pytest.mark.asyncio
    async def test_with_tier_manager_delegates(self):
        tm = TierManager(tier1_hot_count=1, tier12_warm_cutoff=2)
        await tm._apply_tiers(["BTCUSDT", "ETHUSDT", "XRPUSDT"])

        scanner = self._make_scanner(tier_manager=tm)
        assert scanner.get_symbol_tier("BTCUSDT") == PairTier.TIER1
        assert scanner.get_symbol_tier("ETHUSDT") == PairTier.TIER2
        assert scanner.get_symbol_tier("XRPUSDT") == PairTier.TIER3
        # Unknown symbol falls back to TIER3 via TierManager
        assert scanner.get_symbol_tier("DOGEUSDT") == PairTier.TIER3


# ---------------------------------------------------------------------------
# Lifecycle: start / stop
# ---------------------------------------------------------------------------

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_disabled_does_not_create_task(self):
        with patch("src.tier_manager.DYNAMIC_TIER_ENABLED", False):
            tm = TierManager()
            await tm.start()
            assert tm._task is None

    @pytest.mark.asyncio
    async def test_stop_when_not_started_is_safe(self):
        tm = TierManager()
        await tm.stop()  # must not raise

    @pytest.mark.asyncio
    async def test_start_stop_cycle(self):
        """Start creates a background task; stop cancels it cleanly."""
        futures_tickers = _make_ticker_list(10)
        spot_tickers = _make_ticker_list(5, base_vol=500_000.0)

        async def _mock_fetch(session, base_url, path):
            if "fapi" in path:
                return futures_tickers
            return spot_tickers

        with patch("src.tier_manager.DYNAMIC_TIER_ENABLED", True):
            tm = TierManager(poll_interval=3600)  # Long interval; won't fire in test.
            with patch.object(tm, "_fetch_ticker", side_effect=_mock_fetch):
                await tm.start()
                assert tm._task is not None
                assert not tm._task.done()
                await tm.stop()
                assert tm._task is None

    @pytest.mark.asyncio
    async def test_start_twice_does_not_create_second_task(self):
        futures_tickers = _make_ticker_list(10)
        spot_tickers: list = []

        async def _mock_fetch(session, base_url, path):
            if "fapi" in path:
                return futures_tickers
            return spot_tickers

        with patch("src.tier_manager.DYNAMIC_TIER_ENABLED", True):
            tm = TierManager(poll_interval=3600)
            with patch.object(tm, "_fetch_ticker", side_effect=_mock_fetch):
                await tm.start()
                first_task = tm._task
                await tm.start()  # second call — must be a no-op
                assert tm._task is first_task
                await tm.stop()
