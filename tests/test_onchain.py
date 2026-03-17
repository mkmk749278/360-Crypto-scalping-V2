"""Tests for src.onchain — OnChainData, OnChainClient, score_onchain, helpers."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.onchain import (
    OnChainClient,
    OnChainData,
    _NEUTRAL_SCORE,
    _net_flow_to_score,
    _parse_glassnode_latest,
    _strip_quote_currency,
    score_onchain,
)


# ---------------------------------------------------------------------------
# _strip_quote_currency
# ---------------------------------------------------------------------------

class TestStripQuoteCurrency:
    def test_usdt(self):
        assert _strip_quote_currency("BTCUSDT") == "BTC"

    def test_busd(self):
        assert _strip_quote_currency("ETHBUSD") == "ETH"

    def test_usdc(self):
        assert _strip_quote_currency("SOLUSDC") == "SOL"

    def test_already_clean(self):
        assert _strip_quote_currency("BTC") == "BTC"

    def test_lowercase(self):
        assert _strip_quote_currency("btcusdt") == "BTC"


# ---------------------------------------------------------------------------
# _parse_glassnode_latest
# ---------------------------------------------------------------------------

class TestParseGlassnodeLatest:
    def test_returns_last_value(self):
        data = [{"t": 1000, "v": 12345.6}, {"t": 2000, "v": 99999.9}]
        assert _parse_glassnode_latest(data) == pytest.approx(99999.9)

    def test_empty_list(self):
        assert _parse_glassnode_latest([]) is None

    def test_not_a_list(self):
        assert _parse_glassnode_latest({"v": 100}) is None

    def test_missing_v_key(self):
        assert _parse_glassnode_latest([{"t": 1000}]) is None

    def test_v_none(self):
        assert _parse_glassnode_latest([{"t": 1000, "v": None}]) is None


# ---------------------------------------------------------------------------
# _net_flow_to_score
# ---------------------------------------------------------------------------

class TestNetFlowToScore:
    def test_neutral_when_zero_total(self):
        assert _net_flow_to_score(0, 0, 0) == _NEUTRAL_SCORE

    def test_pure_outflow_gives_max_score(self):
        # All coins leaving exchange → bullish → score near 5
        score = _net_flow_to_score(-1000, 0, 1000)  # net = inflow(0) - outflow(1000) = -1000
        assert score == pytest.approx(5.0)

    def test_pure_inflow_gives_min_score(self):
        # All coins entering exchange → bearish → score near 0
        score = _net_flow_to_score(1000, 1000, 0)  # net = inflow(1000) - outflow(0) = 1000
        assert score == pytest.approx(0.0)

    def test_balanced_flow_gives_neutral(self):
        score = _net_flow_to_score(0, 500, 500)
        assert score == pytest.approx(_NEUTRAL_SCORE)

    def test_score_in_range(self):
        for inflow, outflow in [(100, 900), (500, 500), (900, 100)]:
            net = inflow - outflow
            score = _net_flow_to_score(net, inflow, outflow)
            assert 0.0 <= score <= 5.0


# ---------------------------------------------------------------------------
# score_onchain
# ---------------------------------------------------------------------------

class TestScoreOnchain:
    def test_none_input_returns_neutral(self):
        assert score_onchain(None) == _NEUTRAL_SCORE

    def test_uses_onchain_data_score(self):
        data = OnChainData(symbol="BTC", score=4.0)
        assert score_onchain(data) == 4.0

    def test_clamps_above_max(self):
        data = OnChainData(symbol="BTC", score=100.0)
        assert score_onchain(data) == 5.0

    def test_clamps_below_zero(self):
        data = OnChainData(symbol="BTC", score=-5.0)
        assert score_onchain(data) == 0.0


# ---------------------------------------------------------------------------
# OnChainData
# ---------------------------------------------------------------------------

class TestOnChainData:
    def test_defaults(self):
        d = OnChainData()
        assert d.symbol == ""
        assert d.net_flow_usd == 0.0
        assert d.source == ""
        assert d.score == _NEUTRAL_SCORE

    def test_custom(self):
        d = OnChainData(symbol="ETH", net_flow_usd=-500.0, source="glassnode", score=4.2)
        assert d.symbol == "ETH"
        assert d.net_flow_usd == -500.0
        assert d.score == 4.2


# ---------------------------------------------------------------------------
# OnChainClient — disabled (no API key)
# ---------------------------------------------------------------------------

class TestOnChainClientDisabled:
    def test_enabled_false_without_key(self):
        client = OnChainClient(api_key="")
        assert client.enabled is False

    @pytest.mark.asyncio
    async def test_returns_neutral_when_disabled(self):
        client = OnChainClient(api_key="")
        result = await client.get_exchange_flow("BTCUSDT")
        assert result.score == _NEUTRAL_SCORE
        assert result.symbol == "BTC"

    @pytest.mark.asyncio
    async def test_close_no_session(self):
        client = OnChainClient(api_key="")
        await client.close()  # should not raise


# ---------------------------------------------------------------------------
# OnChainClient — enabled (mocked HTTP)
# ---------------------------------------------------------------------------

class TestOnChainClientEnabled:
    def _make_client(self) -> OnChainClient:
        return OnChainClient(api_key="test-glassnode-key")

    def _mock_session_with_inflow_outflow(
        self, inflow_value: float, outflow_value: float
    ) -> MagicMock:
        """Create a mock session returning glassnode-format data for inflow/outflow calls."""
        inflow_data = [{"t": 1000, "v": inflow_value}]
        outflow_data = [{"t": 1000, "v": outflow_value}]

        call_count = 0

        async def mock_json(content_type=None):
            nonlocal call_count
            call_count += 1
            # First call = inflow, second call = outflow
            return inflow_data if call_count == 1 else outflow_data

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = mock_json
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)
        return mock_session

    @pytest.mark.asyncio
    async def test_returns_bullish_score_on_net_outflow(self):
        client = self._make_client()
        # More outflow than inflow = bullish
        client._session = self._mock_session_with_inflow_outflow(100.0, 900.0)

        result = await client.get_exchange_flow("BTCUSDT")
        assert result.score > _NEUTRAL_SCORE  # bullish → above neutral
        assert result.source == "glassnode"
        assert result.symbol == "BTC"

    @pytest.mark.asyncio
    async def test_returns_bearish_score_on_net_inflow(self):
        client = self._make_client()
        # More inflow than outflow = bearish
        client._session = self._mock_session_with_inflow_outflow(900.0, 100.0)

        result = await client.get_exchange_flow("BTCUSDT")
        assert result.score < _NEUTRAL_SCORE  # bearish → below neutral

    @pytest.mark.asyncio
    async def test_returns_neutral_on_api_error(self):
        client = self._make_client()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=Exception("timeout"))
        client._session = mock_session

        result = await client.get_exchange_flow("BTCUSDT")
        assert result.score == _NEUTRAL_SCORE

    @pytest.mark.asyncio
    async def test_caches_result(self):
        client = self._make_client()
        call_count = 0
        base_resp_data = [[{"t": 1, "v": 100.0}], [{"t": 1, "v": 200.0}]]

        async def mock_json(content_type=None):
            nonlocal call_count
            val = base_resp_data[call_count % 2]
            call_count += 1
            return val

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = mock_json
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)
        client._session = mock_session

        result1 = await client.get_exchange_flow("BTCUSDT")
        result2 = await client.get_exchange_flow("BTCUSDT")

        # Both should have same score (second came from cache)
        assert result1.score == result2.score
        # Only 2 API calls (inflow + outflow for the first request)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_expires_triggers_new_call(self):
        client = self._make_client()
        # Pre-seed cache with expired entry
        client._cache["BTC"] = (time.monotonic() - 400, OnChainData(symbol="BTC", score=1.0))

        call_count = 0

        async def mock_json(content_type=None):
            nonlocal call_count
            call_count += 1
            return [{"t": 1, "v": 500.0}]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = mock_json
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)
        client._session = mock_session

        await client.get_exchange_flow("BTCUSDT")
        assert call_count > 0  # expired cache → new HTTP calls

    @pytest.mark.asyncio
    async def test_returns_neutral_on_non_200(self):
        client = self._make_client()
        mock_resp = AsyncMock()
        mock_resp.status = 403
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)
        client._session = mock_session

        result = await client.get_exchange_flow("BTCUSDT")
        assert result.score == _NEUTRAL_SCORE

    @pytest.mark.asyncio
    async def test_close_session(self):
        client = self._make_client()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        client._session = mock_session

        await client.close()
        mock_session.close.assert_called_once()


# ---------------------------------------------------------------------------
# Fix 6: On-chain score skips unsupported assets
# ---------------------------------------------------------------------------


class TestUnsupportedAssetSkip:
    """Non-BTC/ETH assets must immediately return 0.0 without any API call."""

    @pytest.mark.asyncio
    async def test_sol_returns_zero_score(self):
        client = OnChainClient(api_key="test_key")
        # Patch _fetch_glassnode so we can verify it's NOT called
        called = []

        async def _fake_fetch(coin):
            called.append(coin)
            return OnChainData(symbol=coin, score=3.0)

        client._fetch_glassnode = _fake_fetch
        result = await client.get_exchange_flow("SOLUSDT")
        assert result.score == 0.0
        assert called == [], "Glassnode must NOT be called for unsupported assets"

    @pytest.mark.asyncio
    async def test_btc_proceeds_to_fetch(self):
        """BTC is supported – the fetch path must be reached."""
        client = OnChainClient(api_key="test_key")
        called = []

        async def _fake_fetch(coin):
            called.append(coin)
            return OnChainData(symbol=coin, score=4.0, source="glassnode")

        client._fetch_glassnode = _fake_fetch
        result = await client.get_exchange_flow("BTCUSDT")
        assert result.score == 4.0
        assert "BTC" in called

    @pytest.mark.asyncio
    async def test_eth_proceeds_to_fetch(self):
        """ETH is supported – the fetch path must be reached."""
        client = OnChainClient(api_key="test_key")
        called = []

        async def _fake_fetch(coin):
            called.append(coin)
            return OnChainData(symbol=coin, score=2.0, source="glassnode")

        client._fetch_glassnode = _fake_fetch
        result = await client.get_exchange_flow("ETHUSDT")
        assert result.score == 2.0
        assert "ETH" in called

    @pytest.mark.asyncio
    async def test_unsupported_returns_source_unsupported(self):
        """Unsupported assets should have source='unsupported'."""
        client = OnChainClient(api_key="test_key")
        result = await client.get_exchange_flow("DOGEUSDT")
        assert result.source == "unsupported"
        assert result.score == 0.0
