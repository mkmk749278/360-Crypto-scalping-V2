"""Tests for PR 3 — Tier-Aware REST Scanner Fallback Overhaul.

Covers:
* BinanceClient.fetch_all_book_tickers() — global bookTicker endpoint
* Scanner._fetch_global_book_tickers() — cache pre-population for Tier 2/3
* Scanner._get_spread_pct() tier-aware gating when WS is degraded
* scan_loop integration: global bookTicker call issued only when WS degrades
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.binance import BinanceClient
from src.pair_manager import PairTier
from src.scanner import (
    Scanner,
    _BOOK_TICKER_CACHE_TTL,
    _SPREAD_CACHE_TTL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scanner(**overrides) -> Scanner:
    """Create a minimal Scanner with mocked dependencies."""
    signal_queue = MagicMock()
    signal_queue.put = AsyncMock(return_value=True)
    router_mock = MagicMock(active_signals={})
    router_mock.cleanup_expired.return_value = 0

    defaults = dict(
        pair_mgr=MagicMock(),
        data_store=MagicMock(),
        channels=[],
        smc_detector=MagicMock(),
        regime_detector=MagicMock(),
        predictive=MagicMock(),
        exchange_mgr=MagicMock(),
        spot_client=None,
        telemetry=MagicMock(),
        signal_queue=signal_queue,
        router=router_mock,
    )
    defaults.update(overrides)
    return Scanner(**defaults)


def _make_book_ticker_response(*symbols_bids_asks) -> Dict[str, dict]:
    """Return a synthetic bookTicker mapping."""
    return {
        sym: {"symbol": sym, "bidPrice": str(bid), "askPrice": str(ask)}
        for sym, bid, ask in symbols_bids_asks
    }


# ---------------------------------------------------------------------------
# BinanceClient.fetch_all_book_tickers
# ---------------------------------------------------------------------------

class TestFetchAllBookTickers:
    @pytest.mark.asyncio
    async def test_returns_dict_keyed_by_symbol(self):
        raw = [
            {"symbol": "BTCUSDT", "bidPrice": "50000.0", "askPrice": "50001.0"},
            {"symbol": "ETHUSDT", "bidPrice": "3000.0",  "askPrice": "3000.5"},
        ]
        client = BinanceClient("futures")
        with patch.object(client, "_get", AsyncMock(return_value=raw)):
            result = await client.fetch_all_book_tickers()

        assert isinstance(result, dict)
        assert "BTCUSDT" in result
        assert result["BTCUSDT"]["bidPrice"] == "50000.0"
        assert "ETHUSDT" in result

    @pytest.mark.asyncio
    async def test_uses_futures_path(self):
        client = BinanceClient("futures")
        calls = []

        async def _mock_get(path, **kwargs):
            calls.append(path)
            return []

        with patch.object(client, "_get", side_effect=_mock_get):
            await client.fetch_all_book_tickers()

        assert calls[0] == "/fapi/v1/ticker/bookTicker"

    @pytest.mark.asyncio
    async def test_uses_spot_path(self):
        client = BinanceClient("spot")
        calls = []

        async def _mock_get(path, **kwargs):
            calls.append(path)
            return []

        with patch.object(client, "_get", side_effect=_mock_get):
            await client.fetch_all_book_tickers()

        assert calls[0] == "/api/v3/ticker/bookTicker"

    @pytest.mark.asyncio
    async def test_uses_weight_2(self):
        client = BinanceClient("futures")
        weights = []

        async def _mock_get(path, weight=1, **kwargs):
            weights.append(weight)
            return []

        with patch.object(client, "_get", side_effect=_mock_get):
            await client.fetch_all_book_tickers()

        assert weights[0] == 2

    @pytest.mark.asyncio
    async def test_returns_none_when_api_returns_non_list(self):
        client = BinanceClient("futures")
        with patch.object(client, "_get", AsyncMock(return_value={"error": "bad"})):
            result = await client.fetch_all_book_tickers()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_api_returns_none(self):
        client = BinanceClient("futures")
        with patch.object(client, "_get", AsyncMock(return_value=None)):
            result = await client.fetch_all_book_tickers()
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_entries_without_symbol(self):
        raw = [
            {"bidPrice": "100.0", "askPrice": "101.0"},  # missing symbol
            {"symbol": "SOLUSDT", "bidPrice": "20.0", "askPrice": "20.1"},
        ]
        client = BinanceClient("futures")
        with patch.object(client, "_get", AsyncMock(return_value=raw)):
            result = await client.fetch_all_book_tickers()

        assert result is not None
        assert "SOLUSDT" in result
        assert len(result) == 1  # bad entry skipped


# ---------------------------------------------------------------------------
# Scanner._fetch_global_book_tickers
# ---------------------------------------------------------------------------

class TestFetchGlobalBookTickers:
    def _scanner_with_tiers(self, tier_map: Dict[str, PairTier]) -> Scanner:
        scanner = _make_scanner()
        tm = MagicMock()
        tm.get_tier = lambda sym: tier_map.get(sym, PairTier.TIER3)
        scanner.tier_manager = tm
        return scanner

    @pytest.mark.asyncio
    async def test_populates_cache_for_tier2_and_tier3(self):
        tier_map = {
            "BTCUSDT": PairTier.TIER1,
            "ETHUSDT": PairTier.TIER2,
            "SOLUSDT": PairTier.TIER3,
        }
        scanner = self._scanner_with_tiers(tier_map)

        tickers = _make_book_ticker_response(
            ("BTCUSDT", 50000, 50001),
            ("ETHUSDT", 3000, 3000.5),
            ("SOLUSDT", 20, 20.1),
        )
        mock_client = MagicMock()
        mock_client.fetch_all_book_tickers = AsyncMock(return_value=tickers)
        scanner.futures_client = mock_client

        await scanner._fetch_global_book_tickers(market="futures")

        # Tier 2 and 3 should be cached
        assert "ETHUSDT" in scanner._order_book_cache
        assert "SOLUSDT" in scanner._order_book_cache
        # Tier 1 must NOT be seeded by bookTicker (reserved for /depth)
        assert "BTCUSDT" not in scanner._order_book_cache

    @pytest.mark.asyncio
    async def test_computed_spread_stored_in_cache(self):
        tier_map = {"XRPUSDT": PairTier.TIER2}
        scanner = self._scanner_with_tiers(tier_map)

        # bid=100.0, ask=100.4 → spread = 0.4/100.2 * 100 ≈ 0.399%
        tickers = _make_book_ticker_response(("XRPUSDT", 100.0, 100.4))
        mock_client = MagicMock()
        mock_client.fetch_all_book_tickers = AsyncMock(return_value=tickers)
        scanner.futures_client = mock_client

        await scanner._fetch_global_book_tickers(market="futures")

        spread_pct, _ = scanner._order_book_cache["XRPUSDT"]
        expected = (100.4 - 100.0) / ((100.4 + 100.0) / 2.0) * 100.0
        assert abs(spread_pct - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_cache_ttl_uses_book_ticker_ttl(self):
        tier_map = {"BNBUSDT": PairTier.TIER3}
        scanner = self._scanner_with_tiers(tier_map)

        tickers = _make_book_ticker_response(("BNBUSDT", 300.0, 300.5))
        mock_client = MagicMock()
        mock_client.fetch_all_book_tickers = AsyncMock(return_value=tickers)
        scanner.futures_client = mock_client

        t_before = time.monotonic()
        await scanner._fetch_global_book_tickers(market="futures")
        t_after = time.monotonic()

        _, expiry = scanner._order_book_cache["BNBUSDT"]
        assert t_before + _BOOK_TICKER_CACHE_TTL <= expiry <= t_after + _BOOK_TICKER_CACHE_TTL

    @pytest.mark.asyncio
    async def test_does_not_overwrite_fresh_cache_entry(self):
        tier_map = {"ADAUSDT": PairTier.TIER2}
        scanner = self._scanner_with_tiers(tier_map)

        # Pre-populate with a fresh cache entry
        fresh_spread = 0.005
        scanner._order_book_cache["ADAUSDT"] = (fresh_spread, time.monotonic() + 100)

        tickers = _make_book_ticker_response(("ADAUSDT", 0.5, 0.6))
        mock_client = MagicMock()
        mock_client.fetch_all_book_tickers = AsyncMock(return_value=tickers)
        scanner.futures_client = mock_client

        await scanner._fetch_global_book_tickers(market="futures")

        # Cache entry must be unchanged
        assert scanner._order_book_cache["ADAUSDT"][0] == fresh_spread

    @pytest.mark.asyncio
    async def test_handles_api_returning_none_gracefully(self):
        scanner = _make_scanner()
        mock_client = MagicMock()
        mock_client.fetch_all_book_tickers = AsyncMock(return_value=None)
        scanner.futures_client = mock_client

        # Must not raise
        await scanner._fetch_global_book_tickers(market="futures")
        assert scanner._order_book_cache == {}

    @pytest.mark.asyncio
    async def test_handles_timeout_gracefully(self):
        scanner = _make_scanner()
        mock_client = MagicMock()
        mock_client.fetch_all_book_tickers = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        scanner.futures_client = mock_client

        # Must not raise; cache stays empty
        await scanner._fetch_global_book_tickers(market="futures")
        assert scanner._order_book_cache == {}

    @pytest.mark.asyncio
    async def test_handles_arbitrary_exception_gracefully(self):
        scanner = _make_scanner()
        mock_client = MagicMock()
        mock_client.fetch_all_book_tickers = AsyncMock(
            side_effect=RuntimeError("network error")
        )
        scanner.futures_client = mock_client

        await scanner._fetch_global_book_tickers(market="futures")
        assert scanner._order_book_cache == {}

    @pytest.mark.asyncio
    async def test_skips_entries_with_zero_prices(self):
        tier_map = {"ZEROUSDT": PairTier.TIER3}
        scanner = self._scanner_with_tiers(tier_map)

        tickers = {"ZEROUSDT": {"symbol": "ZEROUSDT", "bidPrice": "0", "askPrice": "0"}}
        mock_client = MagicMock()
        mock_client.fetch_all_book_tickers = AsyncMock(return_value=tickers)
        scanner.futures_client = mock_client

        await scanner._fetch_global_book_tickers(market="futures")
        assert "ZEROUSDT" not in scanner._order_book_cache

    @pytest.mark.asyncio
    async def test_lazily_creates_futures_client(self):
        tier_map = {"DOTUSDT": PairTier.TIER2}
        scanner = self._scanner_with_tiers(tier_map)
        assert scanner.futures_client is None

        tickers = _make_book_ticker_response(("DOTUSDT", 10.0, 10.1))
        with patch(
            "src.scanner.BinanceClient",
            return_value=MagicMock(
                fetch_all_book_tickers=AsyncMock(return_value=tickers)
            ),
        ):
            await scanner._fetch_global_book_tickers(market="futures")

        assert scanner.futures_client is not None

    @pytest.mark.asyncio
    async def test_lazily_creates_spot_client(self):
        tier_map = {"LINKUSDT": PairTier.TIER3}
        scanner = self._scanner_with_tiers(tier_map)
        assert scanner.spot_client is None

        tickers = _make_book_ticker_response(("LINKUSDT", 15.0, 15.2))
        with patch(
            "src.scanner.BinanceClient",
            return_value=MagicMock(
                fetch_all_book_tickers=AsyncMock(return_value=tickers)
            ),
        ):
            await scanner._fetch_global_book_tickers(market="spot")

        assert scanner.spot_client is not None


# ---------------------------------------------------------------------------
# Scanner._get_spread_pct — tier-aware gating when WS is degraded
# ---------------------------------------------------------------------------

class TestGetSpreadPctTierGating:
    def _scanner_with_tier(self, symbol: str, tier: PairTier) -> Scanner:
        scanner = _make_scanner()
        tm = MagicMock()
        tm.get_tier = lambda sym: tier if sym == symbol else PairTier.TIER3
        scanner.tier_manager = tm
        scanner._ws_any_degraded_this_cycle = True
        return scanner

    @pytest.mark.asyncio
    async def test_tier2_skips_depth_when_ws_degraded_returns_fallback(self):
        """When WS is degraded a Tier 2 symbol should return the fallback spread
        without issuing a /depth call, since the global bookTicker pre-fetch
        is responsible for its cache entry."""
        scanner = self._scanner_with_tier("ETHUSDT", PairTier.TIER2)

        mock_client = MagicMock()
        mock_client.fetch_order_book = AsyncMock()
        scanner.futures_client = mock_client

        result = await scanner._get_spread_pct("ETHUSDT", market="futures")

        assert result == 0.01  # default fallback
        mock_client.fetch_order_book.assert_not_called()

    @pytest.mark.asyncio
    async def test_tier3_skips_depth_when_ws_degraded(self):
        scanner = self._scanner_with_tier("DOGEUSDT", PairTier.TIER3)

        mock_client = MagicMock()
        mock_client.fetch_order_book = AsyncMock()
        scanner.futures_client = mock_client

        result = await scanner._get_spread_pct("DOGEUSDT", market="futures")

        assert result == 0.01
        mock_client.fetch_order_book.assert_not_called()

    @pytest.mark.asyncio
    async def test_tier1_still_fetches_depth_when_ws_degraded(self):
        """Tier 1 (Hot) pairs must still get the full /depth fetch even when
        WS is degraded, because they need accurate order-book data."""
        scanner = self._scanner_with_tier("BTCUSDT", PairTier.TIER1)

        book_response = {
            "bids": [["50000.0", "1.0"]],
            "asks": [["50001.0", "1.0"]],
        }
        mock_client = MagicMock()
        mock_client.fetch_order_book = AsyncMock(return_value=book_response)
        scanner.futures_client = mock_client

        # Ensure rate-limiter guard passes
        with patch("src.scanner.futures_rate_limiter") as mock_limiter:
            mock_limiter.remaining = 9999
            result = await scanner._get_spread_pct("BTCUSDT", market="futures")

        assert result > 0.0  # actual spread, not fallback
        mock_client.fetch_order_book.assert_called_once()

    @pytest.mark.asyncio
    async def test_tier2_returns_cached_spread_from_prefetch(self):
        """If the bookTicker pre-fetch has already seeded the cache for a
        Tier 2 symbol, _get_spread_pct should return the cached value."""
        scanner = self._scanner_with_tier("ADAUSDT", PairTier.TIER2)

        # Simulate what _fetch_global_book_tickers writes
        expected_spread = 0.12345
        scanner._order_book_cache["ADAUSDT"] = (
            expected_spread, time.monotonic() + _BOOK_TICKER_CACHE_TTL
        )

        result = await scanner._get_spread_pct("ADAUSDT", market="futures")
        assert result == expected_spread

    @pytest.mark.asyncio
    async def test_tier1_returns_cached_spread_without_fetch(self):
        """Even for Tier 1, a valid cache entry must be returned without a new fetch."""
        scanner = self._scanner_with_tier("BTCUSDT", PairTier.TIER1)

        cached_spread = 0.02
        scanner._order_book_cache["BTCUSDT"] = (
            cached_spread, time.monotonic() + _SPREAD_CACHE_TTL
        )
        mock_client = MagicMock()
        mock_client.fetch_order_book = AsyncMock()
        scanner.futures_client = mock_client

        result = await scanner._get_spread_pct("BTCUSDT", market="futures")

        assert result == cached_spread
        mock_client.fetch_order_book.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_depth_guard_when_ws_healthy(self):
        """When WS is healthy, depth fetches are NOT gated by tier — even Tier 3
        symbols should reach the /depth call path (subject to other guards)."""
        scanner = _make_scanner()
        scanner._ws_any_degraded_this_cycle = False  # healthy WS

        book_response = {
            "bids": [["1.0", "100.0"]],
            "asks": [["1.001", "100.0"]],
        }
        mock_client = MagicMock()
        mock_client.fetch_order_book = AsyncMock(return_value=book_response)
        scanner.spot_client = mock_client

        with patch("src.scanner.rate_limiter") as mock_limiter:
            mock_limiter.remaining = 9999
            result = await scanner._get_spread_pct("XRPUSDT", market="spot")

        assert result > 0.0
        mock_client.fetch_order_book.assert_called_once()


# ---------------------------------------------------------------------------
# scan_loop integration: global bookTicker called iff WS degraded
# ---------------------------------------------------------------------------

class TestScanLoopBookTickerIntegration:
    """Verify that _fetch_global_book_tickers is called exactly once per
    degraded scan cycle and not called in healthy cycles."""

    def _minimal_scanner(self) -> Scanner:
        scanner = _make_scanner()
        scanner.telemetry = MagicMock()
        scanner.telemetry.scan_latency_ms = 0.0
        scanner.telemetry.set_scan_latency = MagicMock()
        scanner.telemetry.set_pairs_monitored = MagicMock()
        scanner.telemetry.set_active_signals = MagicMock()
        scanner.telemetry.get_admin_alert_callback = MagicMock(return_value=None)
        scanner.router = MagicMock(active_signals={})
        scanner.router.cleanup_expired.return_value = 0

        # Pair manager with one pair per tier
        from src.pair_manager import PairInfo
        pair_mgr = MagicMock()
        pair_mgr.pairs = {
            "BTCUSDT": PairInfo(symbol="BTCUSDT", market="futures", tier=PairTier.TIER1, volume_24h_usd=1e9),
            "ETHUSDT": PairInfo(symbol="ETHUSDT", market="futures", tier=PairTier.TIER2, volume_24h_usd=5e8),
        }
        pair_mgr.check_promotions.return_value = []
        scanner.pair_mgr = pair_mgr
        return scanner

    @pytest.mark.asyncio
    async def test_global_book_ticker_called_when_ws_degraded(self):
        scanner = self._minimal_scanner()

        ws_mock = MagicMock()
        ws_mock.is_healthy = True  # spot WS healthy
        ws_mock.health_ratio = 0.3  # below WS_PARTIAL_HEALTH_THRESHOLD → partial degradation
        scanner.ws_spot = ws_mock
        scanner.ws_futures = MagicMock(is_healthy=True, health_ratio=1.0)

        prefetch_calls = []

        async def _fake_prefetch(market="futures"):
            prefetch_calls.append(market)

        scanner._fetch_global_book_tickers = _fake_prefetch

        # Also patch _scan_symbol_bounded to avoid full scan pipeline
        scanner._scan_symbol_bounded = AsyncMock(return_value=None)
        scanner._lightweight_tier3_scan = AsyncMock()

        # Run just one iteration by cancelling after the first sleep
        async def _run_one_cycle():
            task = asyncio.create_task(scanner.scan_loop())
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await _run_one_cycle()
        assert len(prefetch_calls) >= 1
        assert prefetch_calls[0] == "futures"

    @pytest.mark.asyncio
    async def test_global_book_ticker_not_called_when_ws_healthy(self):
        scanner = self._minimal_scanner()

        ws_mock = MagicMock()
        ws_mock.is_healthy = True
        ws_mock.health_ratio = 1.0  # fully healthy
        scanner.ws_spot = ws_mock
        scanner.ws_futures = MagicMock(is_healthy=True, health_ratio=1.0)

        prefetch_calls = []

        async def _fake_prefetch(market="futures"):
            prefetch_calls.append(market)

        scanner._fetch_global_book_tickers = _fake_prefetch
        scanner._scan_symbol_bounded = AsyncMock(return_value=None)
        scanner._lightweight_tier3_scan = AsyncMock()

        async def _run_one_cycle():
            task = asyncio.create_task(scanner.scan_loop())
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await _run_one_cycle()
        assert prefetch_calls == []
