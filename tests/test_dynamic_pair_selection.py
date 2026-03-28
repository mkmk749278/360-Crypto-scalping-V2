"""Tests for the enhanced pair_manager (PR: Dynamic Pair Selection)."""

from __future__ import annotations

import pytest

from src.pair_manager import PairInfo, PairManager, PairTier


def _make_pair(
    symbol: str,
    volume: float = 1_000_000.0,
    volatility: float = 5.0,
    spread: float = 0.001,
    market: str = "spot",
) -> PairInfo:
    return PairInfo(
        symbol=symbol,
        market=market,
        base_asset=symbol.replace("USDT", ""),
        quote_asset="USDT",
        volume_24h_usd=volume,
        volatility_24h=volatility,
        spread_avg=spread,
    )


class TestPairRanking:
    def test_rank_pairs_returns_sorted_list(self):
        pm = PairManager()
        pm.pairs["BTCUSDT"] = _make_pair("BTCUSDT", volume=100_000_000, volatility=3.0, spread=0.0002)
        pm.pairs["ETHUSDT"] = _make_pair("ETHUSDT", volume=50_000_000, volatility=5.0, spread=0.0005)
        pm.pairs["SOLUSDT"] = _make_pair("SOLUSDT", volume=10_000_000, volatility=8.0, spread=0.001)

        ranked = pm.rank_pairs()
        assert isinstance(ranked, list)
        assert len(ranked) == 3
        # BTC should rank highest (highest volume)
        assert ranked[0] == "BTCUSDT"

    def test_rank_pairs_empty(self):
        pm = PairManager()
        assert pm.rank_pairs() == []

    def test_get_top_ranked_pairs(self):
        pm = PairManager()
        for i in range(10):
            pm.pairs[f"SYM{i}USDT"] = _make_pair(f"SYM{i}USDT", volume=(10 - i) * 1_000_000)
        top = pm.get_top_ranked_pairs(n=3)
        assert len(top) == 3

    def test_rank_score_stored_on_pair_info(self):
        pm = PairManager()
        pm.pairs["BTCUSDT"] = _make_pair("BTCUSDT", volume=100_000_000)
        pm.rank_pairs()
        assert pm.pairs["BTCUSDT"].rank_score > 0


class TestPairVolatilityAndSpread:
    def test_update_pair_volatility(self):
        pm = PairManager()
        pm.pairs["BTCUSDT"] = _make_pair("BTCUSDT")
        pm.update_pair_volatility("BTCUSDT", 4.5)
        assert pm.pairs["BTCUSDT"].volatility_24h == 4.5

    def test_update_pair_volatility_nonexistent(self):
        pm = PairManager()
        pm.update_pair_volatility("NOPE", 3.0)  # should not raise

    def test_update_pair_spread(self):
        pm = PairManager()
        pm.pairs["ETHUSDT"] = _make_pair("ETHUSDT")
        pm.update_pair_spread("ETHUSDT", 0.0003)
        assert pm.pairs["ETHUSDT"].spread_avg == pytest.approx(0.0003)

    def test_update_pair_spread_nonexistent(self):
        pm = PairManager()
        pm.update_pair_spread("NOPE", 0.001)  # should not raise


class TestPairMetricsHistory:
    def test_record_pair_metrics(self):
        pm = PairManager()
        pm.pairs["BTCUSDT"] = _make_pair("BTCUSDT")
        pm.record_pair_metrics("BTCUSDT")
        history = pm.get_pair_metrics_history("BTCUSDT")
        assert len(history) == 1
        assert "volume_24h_usd" in history[0]
        assert "volatility_24h" in history[0]

    def test_metrics_history_capped(self):
        pm = PairManager()
        pm.pairs["BTCUSDT"] = _make_pair("BTCUSDT")
        for _ in range(600):
            pm.record_pair_metrics("BTCUSDT")
        history = pm.get_pair_metrics_history("BTCUSDT")
        assert len(history) <= 500

    def test_get_pair_metrics_nonexistent(self):
        pm = PairManager()
        assert pm.get_pair_metrics_history("NOPE") == []


class TestVolumeSpikes:
    def test_detect_volume_spikes(self):
        pm = PairManager()
        pm.pairs["BTCUSDT"] = _make_pair("BTCUSDT", volume=10_000_000)
        pm._prev_volumes["BTCUSDT"] = 2_000_000  # 5x spike
        spiked = pm.detect_volume_spikes(multiplier=3.0)
        assert "BTCUSDT" in spiked

    def test_no_spike_below_threshold(self):
        pm = PairManager()
        pm.pairs["BTCUSDT"] = _make_pair("BTCUSDT", volume=3_000_000)
        pm._prev_volumes["BTCUSDT"] = 2_000_000  # 1.5x — not a spike
        spiked = pm.detect_volume_spikes(multiplier=3.0)
        assert "BTCUSDT" not in spiked

    def test_no_spike_without_previous_volume(self):
        pm = PairManager()
        pm.pairs["BTCUSDT"] = _make_pair("BTCUSDT", volume=10_000_000)
        spiked = pm.detect_volume_spikes()
        assert spiked == []


class TestNewPairInfoFields:
    def test_default_new_fields(self):
        p = PairInfo(symbol="TEST", market="spot")
        assert p.volatility_24h == 0.0
        assert p.spread_avg == 0.0
        assert p.rank_score == 0.0
