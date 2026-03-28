"""Tests for the Top-50 futures reconstruction PRs (PR1–PR5)."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pair_manager import PairInfo, PairManager, PairTier
from src.ai_engine.predictor import PredictionFeatures, SignalPredictor
from src.telemetry import TelemetryCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_futures_pair(symbol: str, volume: float = 10_000_000.0) -> PairInfo:
    return PairInfo(
        symbol=symbol,
        market="futures",
        base_asset=symbol.replace("USDT", ""),
        quote_asset="USDT",
        volume_24h_usd=volume,
        tier=PairTier.TIER1,
    )


def _make_spot_pair(symbol: str, volume: float = 5_000_000.0) -> PairInfo:
    return PairInfo(
        symbol=symbol,
        market="spot",
        base_asset=symbol.replace("USDT", ""),
        quote_asset="USDT",
        volume_24h_usd=volume,
        tier=PairTier.TIER2,
    )


# ---------------------------------------------------------------------------
# PR1: pair_manager top-50 futures API
# ---------------------------------------------------------------------------

class TestTop50FuturesPairManager:
    """PR1: Top-50 futures-only pair manager."""

    def test_get_top50_futures_pairs_empty_initially(self):
        pm = PairManager()
        assert pm.get_top50_futures_pairs() == []

    def test_is_top50_futures_false_when_empty(self):
        pm = PairManager()
        assert pm.is_top50_futures("BTCUSDT") is False

    def test_is_top50_futures_true_after_cache_set(self):
        pm = PairManager()
        pm._top50_futures_cache = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        assert pm.is_top50_futures("BTCUSDT") is True
        assert pm.is_top50_futures("ETHUSDT") is True
        assert pm.is_top50_futures("XYZUSDT") is False

    def test_get_top50_futures_pairs_returns_copy(self):
        pm = PairManager()
        pm._top50_futures_cache = ["BTCUSDT", "ETHUSDT"]
        result = pm.get_top50_futures_pairs()
        result.append("INJECTED")
        assert "INJECTED" not in pm._top50_futures_cache

    async def test_refresh_top50_futures_uses_cache_within_interval(self):
        pm = PairManager()
        pm._top50_futures_cache = ["BTCUSDT", "ETHUSDT"]
        pm._top50_last_refresh = time.monotonic()  # just refreshed

        # Should return cached list without calling fetch
        fetch_mock = AsyncMock(return_value=[])
        pm.fetch_top_futures_pairs = fetch_mock

        result = await pm.refresh_top50_futures()
        assert result == ["BTCUSDT", "ETHUSDT"]
        fetch_mock.assert_not_called()

    async def test_refresh_top50_futures_fetches_when_stale(self):
        pm = PairManager()
        pm._top50_last_refresh = 0.0  # never refreshed

        mock_pairs = [
            _make_futures_pair(f"SYM{i}USDT", volume=(100 - i) * 1_000_000)
            for i in range(60)
        ]
        pm.fetch_top_futures_pairs = AsyncMock(return_value=mock_pairs)

        result = await pm.refresh_top50_futures(count=50)
        assert len(result) == 50
        assert result[0] == "SYM0USDT"

    async def test_refresh_top50_futures_force_bypasses_interval(self):
        pm = PairManager()
        pm._top50_futures_cache = ["OLDUSDT"]
        pm._top50_last_refresh = time.monotonic()  # just refreshed

        new_pair = _make_futures_pair("NEWUSDT")
        pm.fetch_top_futures_pairs = AsyncMock(return_value=[new_pair])

        result = await pm.refresh_top50_futures(force=True)
        assert "NEWUSDT" in result

    async def test_refresh_top50_futures_registers_pairs(self):
        pm = PairManager()
        pm._top50_last_refresh = 0.0

        mock_pairs = [_make_futures_pair("BTCUSDT", volume=1_000_000_000)]
        pm.fetch_top_futures_pairs = AsyncMock(return_value=mock_pairs)

        await pm.refresh_top50_futures(count=1)
        assert "BTCUSDT" in pm.pairs
        assert pm.pairs["BTCUSDT"].tier == PairTier.TIER1

    async def test_refresh_top50_futures_updates_existing_pair_volume(self):
        pm = PairManager()
        pm.pairs["BTCUSDT"] = _make_futures_pair("BTCUSDT", volume=500_000_000)
        pm._top50_last_refresh = 0.0

        updated = _make_futures_pair("BTCUSDT", volume=1_000_000_000)
        pm.fetch_top_futures_pairs = AsyncMock(return_value=[updated])

        await pm.refresh_top50_futures(count=1)
        assert pm.pairs["BTCUSDT"].volume_24h_usd == 1_000_000_000


# ---------------------------------------------------------------------------
# PR3: AI engine allowed-pairs filter
# ---------------------------------------------------------------------------

class TestSignalPredictorAllowedPairs:
    """PR3: AI pipeline top-50 filter."""

    async def test_predict_all_symbols_without_filter(self):
        predictor = SignalPredictor()
        result = await predictor.predict("BTCUSDT", PredictionFeatures())
        assert result.symbol == "BTCUSDT"
        assert result.direction in ("LONG", "SHORT", "NEUTRAL")

    async def test_predict_returns_neutral_for_non_allowed_symbol(self):
        predictor = SignalPredictor()
        predictor.set_allowed_pairs(["BTCUSDT", "ETHUSDT"])

        result = await predictor.predict("XYZUSDT", PredictionFeatures())
        assert result.symbol == "XYZUSDT"
        assert result.direction == "NEUTRAL"
        assert result.probability == 0.5

    async def test_predict_processes_allowed_symbol(self):
        predictor = SignalPredictor()
        predictor.set_allowed_pairs(["BTCUSDT"])

        result = await predictor.predict("BTCUSDT", PredictionFeatures())
        assert result.symbol == "BTCUSDT"
        # Allowed symbol should go through the pipeline (prediction_count incremented)
        assert predictor.prediction_count == 1

    async def test_predict_non_allowed_does_not_increment_count(self):
        predictor = SignalPredictor()
        predictor.set_allowed_pairs(["BTCUSDT"])

        await predictor.predict("XYZUSDT", PredictionFeatures())
        assert predictor.prediction_count == 0

    def test_set_allowed_pairs_clears_filter(self):
        predictor = SignalPredictor()
        predictor.set_allowed_pairs(["BTCUSDT"])
        assert predictor._allowed_pairs is not None

        predictor.set_allowed_pairs([])
        assert predictor._allowed_pairs is None

    def test_set_allowed_pairs_stores_upper_case(self):
        predictor = SignalPredictor()
        predictor.set_allowed_pairs(["btcusdt", "EthUSDT"])
        assert "BTCUSDT" in predictor._allowed_pairs
        assert "ETHUSDT" in predictor._allowed_pairs
        assert "btcusdt" not in predictor._allowed_pairs

    def test_set_allowed_pairs_multiple_sets_replaces(self):
        predictor = SignalPredictor()
        predictor.set_allowed_pairs(["BTCUSDT"])
        predictor.set_allowed_pairs(["ETHUSDT", "SOLUSDT"])
        assert predictor._allowed_pairs == {"ETHUSDT", "SOLUSDT"}
        assert "BTCUSDT" not in predictor._allowed_pairs

    async def test_predict_case_insensitive(self):
        predictor = SignalPredictor()
        predictor.set_allowed_pairs(["btcusdt"])  # lowercase

        # Should still be allowed (normalised to upper)
        result = await predictor.predict("BTCUSDT", PredictionFeatures())
        assert predictor.prediction_count == 1

    async def test_predict_batch_filters_non_allowed(self):
        predictor = SignalPredictor()
        predictor.set_allowed_pairs(["BTCUSDT", "ETHUSDT"])

        symbols_features = {
            "BTCUSDT": PredictionFeatures(),
            "ETHUSDT": PredictionFeatures(),
            "XYZUSDT": PredictionFeatures(),
        }
        results = await predictor.predict_batch(symbols_features)
        # Only BTCUSDT and ETHUSDT go through the pipeline
        assert predictor.prediction_count == 2
        # XYZUSDT is not in results (filtered out before spawning tasks)
        assert "XYZUSDT" not in results

    async def test_predict_batch_without_filter(self):
        predictor = SignalPredictor()
        symbols_features = {
            "BTCUSDT": PredictionFeatures(),
            "ETHUSDT": PredictionFeatures(),
        }
        results = await predictor.predict_batch(symbols_features)
        assert set(results.keys()) == {"BTCUSDT", "ETHUSDT"}


# ---------------------------------------------------------------------------
# PR5: Telemetry top-50 filter
# ---------------------------------------------------------------------------

class TestTelemetryTop50:
    """PR5: Telemetry reduced-logging mode."""

    def test_is_top50_pair_returns_true_when_no_filter(self):
        t = TelemetryCollector()
        assert t.is_top50_pair("BTCUSDT") is True
        assert t.is_top50_pair("RANDOMUSDT") is True

    def test_set_top50_pairs_enables_filter(self):
        t = TelemetryCollector()
        t.set_top50_pairs(["BTCUSDT", "ETHUSDT"])
        assert t.is_top50_pair("BTCUSDT") is True
        assert t.is_top50_pair("ETHUSDT") is True
        assert t.is_top50_pair("RANDOMUSDT") is False

    def test_set_top50_pairs_clears_filter(self):
        t = TelemetryCollector()
        t.set_top50_pairs(["BTCUSDT"])
        t.set_top50_pairs([])
        assert t.is_top50_pair("RANDOMUSDT") is True

    def test_set_top50_pairs_case_insensitive(self):
        t = TelemetryCollector()
        t.set_top50_pairs(["btcusdt"])
        assert t.is_top50_pair("BTCUSDT") is True

    def test_set_active_trades_only(self):
        t = TelemetryCollector()
        assert t._active_trades_only is False
        t.set_active_trades_only(True)
        assert t._active_trades_only is True
        t.set_active_trades_only(False)
        assert t._active_trades_only is False

    def test_top50_pairs_set_count(self):
        t = TelemetryCollector()
        pairs = [f"SYM{i}USDT" for i in range(50)]
        t.set_top50_pairs(pairs)
        assert len(t._top50_pairs) == 50
