"""Tests for src/scanner/data_fetcher.py."""
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
from src.scanner.data_fetcher import DataFetcher


@pytest.mark.asyncio
async def test_fetch_klines_returns_empty_on_error():
    """DataFetcher handles exceptions gracefully."""
    client = MagicMock()
    client.get_klines = AsyncMock(side_effect=Exception("timeout"))
    fetcher = DataFetcher(data_store=Mock(), exchange_mgr=client)
    result = await fetcher._fetch_single_timeframe("BTCUSDT", "5m")
    assert result == []


@pytest.mark.asyncio
async def test_fetch_all_timeframes_concurrent():
    """fetch_all_timeframes returns data for all requested timeframes."""
    client = MagicMock()
    client.get_klines = AsyncMock(return_value=[{"o": 1, "h": 2, "l": 0.5, "c": 1.5}])
    fetcher = DataFetcher(data_store=Mock(), exchange_mgr=client)
    result = await fetcher.fetch_all_timeframes("BTCUSDT", ["1m", "5m", "1h"])
    assert set(result.keys()) == {"1m", "5m", "1h"}


def test_load_candles_returns_empty_on_missing_symbol():
    """load_candles returns empty dict for unknown symbols."""
    store = Mock()
    store.candles = {}
    fetcher = DataFetcher(data_store=store, exchange_mgr=Mock())
    assert fetcher.load_candles("UNKNOWN") == {}


@pytest.mark.asyncio
async def test_spread_cache():
    """Spread values are cached for TTL duration."""
    client = MagicMock()
    client.get_orderbook_spread = AsyncMock(return_value=0.05)
    fetcher = DataFetcher(data_store=Mock(), exchange_mgr=client)

    spread1 = await fetcher.fetch_spread("BTCUSDT")
    spread2 = await fetcher.fetch_spread("BTCUSDT")

    assert spread1 == 0.05
    assert spread2 == 0.05
    # Should only call once due to cache
    assert client.get_orderbook_spread.call_count == 1


@pytest.mark.asyncio
async def test_fetch_spread_returns_zero_on_no_method():
    """fetch_spread returns 0.0 when exchange_mgr lacks get_orderbook_spread."""
    client = MagicMock(spec=[])  # No methods
    fetcher = DataFetcher(data_store=Mock(), exchange_mgr=client)
    result = await fetcher.fetch_spread("BTCUSDT")
    assert result == 0.0


@pytest.mark.asyncio
async def test_fetch_spread_returns_zero_on_exception():
    """fetch_spread returns 0.0 when the exchange call raises."""
    client = MagicMock()
    client.get_orderbook_spread = AsyncMock(side_effect=RuntimeError("network error"))
    fetcher = DataFetcher(data_store=Mock(), exchange_mgr=client)
    result = await fetcher.fetch_spread("BTCUSDT")
    assert result == 0.0


def test_load_candles_filters_non_dict_entries():
    """load_candles only includes timeframes with dict OHLCV data."""
    store = Mock()
    store.candles = {
        "BTCUSDT": {
            "5m": {"close": [100, 101], "high": [102, 103], "low": [99, 100]},
            "1h": "not_a_dict",
        }
    }
    fetcher = DataFetcher(data_store=store, exchange_mgr=Mock())
    result = fetcher.load_candles("BTCUSDT")
    assert "5m" in result
    assert "1h" not in result
