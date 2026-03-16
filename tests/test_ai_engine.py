"""Tests for src.ai_engine — real API integrations, caching, and graceful degradation."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.ai_engine as ai_engine
from src.ai_engine import (
    SentimentResult,
    _cache,
    _get_cached,
    _get_shared_session,
    _prune_cache,
    _set_cached,
    _strip_quote_currency,
    close_shared_session,
    detect_volume_delta_spike,
    detect_whale_trade,
    fetch_fear_greed_index,
    fetch_news_sentiment,
    fetch_social_sentiment,
    get_ai_insight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_cache(*keys: str) -> None:
    """Remove specific keys from the module-level cache."""
    for k in keys:
        _cache.pop(k, None)


# ---------------------------------------------------------------------------
# _strip_quote_currency
# ---------------------------------------------------------------------------

class TestStripQuoteCurrency:
    def test_usdt_suffix(self):
        assert _strip_quote_currency("BTCUSDT") == "BTC"

    def test_busd_suffix(self):
        assert _strip_quote_currency("ETHBUSD") == "ETH"

    def test_usdc_suffix(self):
        assert _strip_quote_currency("SOLUSDC") == "SOL"

    def test_already_clean(self):
        assert _strip_quote_currency("BTC") == "BTC"

    def test_lowercase_input(self):
        assert _strip_quote_currency("btcusdt") == "BTC"

    def test_mixed_case(self):
        assert _strip_quote_currency("BtcUsdt") == "BTC"


# ---------------------------------------------------------------------------
# TTL cache helpers
# ---------------------------------------------------------------------------

class TestTTLCache:
    def setup_method(self):
        _clear_cache("test_key")

    def test_miss_returns_none(self):
        assert _get_cached("test_key", ttl=60.0) is None

    def test_hit_within_ttl(self):
        _set_cached("test_key", "value")
        assert _get_cached("test_key", ttl=60.0) == "value"

    def test_expired_returns_none(self):
        _cache["test_key"] = (time.monotonic() - 120.0, "stale")
        assert _get_cached("test_key", ttl=60.0) is None

    def test_second_set_overwrites(self):
        _set_cached("test_key", "first")
        _set_cached("test_key", "second")
        assert _get_cached("test_key", ttl=60.0) == "second"

    def test_prune_removes_stale_entries(self):
        _cache["test_key"] = (time.monotonic() - 7200.0, "stale")
        _prune_cache(max_age=60.0)
        assert "test_key" not in _cache


# ---------------------------------------------------------------------------
# fetch_news_sentiment
# ---------------------------------------------------------------------------

class TestFetchNewsSentiment:
    def setup_method(self):
        _clear_cache("news:BTC", "news:ETH", "news:SOL")

    @pytest.mark.asyncio
    async def test_no_api_key_returns_neutral(self):
        with patch("src.ai_engine.NEWS_API_KEY", ""):
            result = await fetch_news_sentiment("BTCUSDT")
        assert result.score == 0.0
        assert result.label == "Neutral"
        assert "No API key" in result.summary

    @pytest.mark.asyncio
    async def test_successful_bullish_response(self):
        mock_data = {
            "results": [
                {"kind": "bullish", "title": "BTC soars", "source": {"domain": "coindesk.com"}},
                {"kind": "bullish", "title": "BTC all time high", "source": {"domain": "cointelegraph.com"}},
                {"kind": "neutral", "title": "Market update", "source": {"domain": "decrypt.co"}},
            ]
        }
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()

        with patch("src.ai_engine.NEWS_API_KEY", "test-key"):
            result = await fetch_news_sentiment("BTCUSDT", session=mock_session)

        assert result.score == pytest.approx(2 / 3, rel=1e-3)
        assert result.label == "Positive"
        assert result.summary == "BTC soars"
        assert "coindesk.com" in result.sources

    @pytest.mark.asyncio
    async def test_successful_bearish_response(self):
        mock_data = {
            "results": [
                {"kind": "bearish", "title": "BTC dumps", "source": {"domain": "coindesk.com"}},
                {"kind": "bearish", "title": "Crash incoming", "source": {"domain": "cointelegraph.com"}},
                {"kind": "bearish", "title": "Bear market", "source": {"domain": "decrypt.co"}},
                {"kind": "neutral", "title": "Market update", "source": {}},
            ]
        }
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch("src.ai_engine.NEWS_API_KEY", "test-key"):
            result = await fetch_news_sentiment("BTCUSDT", session=mock_session)

        assert result.score == pytest.approx(-3 / 4, rel=1e-3)
        assert result.label == "Negative"

    @pytest.mark.asyncio
    async def test_http_error_returns_neutral(self):
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch("src.ai_engine.NEWS_API_KEY", "test-key"):
            result = await fetch_news_sentiment("BTCUSDT", session=mock_session)

        assert result.score == 0.0
        assert result.label == "Neutral"

    @pytest.mark.asyncio
    async def test_exception_returns_neutral(self):
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Connection error"))

        with patch("src.ai_engine.NEWS_API_KEY", "test-key"):
            result = await fetch_news_sentiment("BTCUSDT", session=mock_session)

        assert result.score == 0.0
        assert result.label == "Neutral"

    @pytest.mark.asyncio
    async def test_cache_hit_skips_request(self):
        cached = SentimentResult(score=0.5, label="Positive", summary="cached")
        _set_cached("news:BTC", cached)

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=AssertionError("Should not be called"))

        with patch("src.ai_engine.NEWS_API_KEY", "test-key"):
            result = await fetch_news_sentiment("BTCUSDT", session=mock_session)

        assert result is cached

    @pytest.mark.asyncio
    async def test_score_normalised_to_minus_one_plus_one(self):
        mock_data = {
            "results": [
                {"kind": "bullish", "title": "t", "source": {"domain": "a.com"}}
                for _ in range(100)
            ]
        }
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch("src.ai_engine.NEWS_API_KEY", "test-key"):
            result = await fetch_news_sentiment("BTCUSDT", session=mock_session)

        assert -1.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_symbol_stripped_for_api_call(self):
        """Verify BTCUSDT is stripped to BTC for the CryptoPanic URL."""
        captured_url: list[str] = []

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"results": []})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        def fake_get(url: str, **kwargs: Any) -> Any:
            captured_url.append(url)
            return mock_resp

        mock_session = MagicMock()
        mock_session.get = fake_get

        with patch("src.ai_engine.NEWS_API_KEY", "test-key"):
            await fetch_news_sentiment("BTCUSDT", session=mock_session)

        assert len(captured_url) == 1
        assert "currencies=BTC" in captured_url[0]
        assert "USDT" not in captured_url[0]

    @pytest.mark.asyncio
    async def test_shared_session_reused_when_session_not_provided(self):
        await close_shared_session()
        _clear_cache("news:BTC", "news:ETH")

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"results": []})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()

        with patch("src.ai_engine.NEWS_API_KEY", "test-key"), patch(
            "src.ai_engine.aiohttp.ClientSession", return_value=mock_session
        ) as session_ctor:
            await fetch_news_sentiment("BTCUSDT")
            await fetch_news_sentiment("ETHUSDT")

        assert session_ctor.call_count == 1
        await close_shared_session()

    @pytest.mark.asyncio
    async def test_close_shared_session_clears_cached_session_reference(self):
        await close_shared_session()

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        with patch("src.ai_engine.aiohttp.ClientSession", return_value=mock_session):
            session = await _get_shared_session()

        assert session is mock_session

        await close_shared_session()

        mock_session.close.assert_awaited_once()
        assert ai_engine._shared_session is None


# ---------------------------------------------------------------------------
# fetch_social_sentiment
# ---------------------------------------------------------------------------

class TestFetchSocialSentiment:
    def setup_method(self):
        _clear_cache("social:BTC", "social:ETH")

    @pytest.mark.asyncio
    async def test_no_api_key_returns_neutral(self):
        with patch("src.ai_engine.SOCIAL_SENTIMENT_API_KEY", ""):
            result = await fetch_social_sentiment("BTCUSDT")
        assert result.score == 0.0
        assert result.label == "Neutral"

    @pytest.mark.asyncio
    async def test_sentiment_score_normalised(self):
        mock_data = {"data": {"sentiment_score": 75, "social_volume": 12345, "galaxy_score": 60}}
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch("src.ai_engine.SOCIAL_SENTIMENT_API_KEY", "test-key"):
            result = await fetch_social_sentiment("BTCUSDT", session=mock_session)

        assert result.score == pytest.approx((75 - 50) / 50, rel=1e-3)
        assert result.label == "Positive"
        assert "social_volume=12345" in result.summary

    @pytest.mark.asyncio
    async def test_galaxy_score_fallback(self):
        mock_data = {"data": {"galaxy_score": 30}}
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch("src.ai_engine.SOCIAL_SENTIMENT_API_KEY", "test-key"):
            result = await fetch_social_sentiment("BTCUSDT", session=mock_session)

        # galaxy_score=30 → (30-50)/50 = -0.4 → Negative
        assert result.score == pytest.approx(-0.4, rel=1e-3)
        assert result.label == "Negative"

    @pytest.mark.asyncio
    async def test_missing_score_returns_zero(self):
        mock_data = {"data": {}}
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch("src.ai_engine.SOCIAL_SENTIMENT_API_KEY", "test-key"):
            result = await fetch_social_sentiment("BTCUSDT", session=mock_session)

        assert result.score == 0.0
        assert result.label == "Neutral"

    @pytest.mark.asyncio
    async def test_http_error_returns_neutral(self):
        mock_resp = AsyncMock()
        mock_resp.status = 401
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch("src.ai_engine.SOCIAL_SENTIMENT_API_KEY", "test-key"):
            result = await fetch_social_sentiment("BTCUSDT", session=mock_session)

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_exception_returns_neutral(self):
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("timeout"))

        with patch("src.ai_engine.SOCIAL_SENTIMENT_API_KEY", "test-key"):
            result = await fetch_social_sentiment("BTCUSDT", session=mock_session)

        assert result.score == 0.0
        assert result.label == "Neutral"

    @pytest.mark.asyncio
    async def test_score_clamped_to_range(self):
        # score > 100 should be clamped to 1.0
        mock_data = {"data": {"sentiment_score": 150}}
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch("src.ai_engine.SOCIAL_SENTIMENT_API_KEY", "test-key"):
            result = await fetch_social_sentiment("BTCUSDT", session=mock_session)

        assert -1.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        cached = SentimentResult(score=0.3, label="Positive", summary="cached")
        _set_cached("social:ETH", cached)

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=AssertionError("Should not be called"))

        with patch("src.ai_engine.SOCIAL_SENTIMENT_API_KEY", "test-key"):
            result = await fetch_social_sentiment("ETHUSDT", session=mock_session)

        assert result is cached


# ---------------------------------------------------------------------------
# fetch_fear_greed_index
# ---------------------------------------------------------------------------

class TestFetchFearGreedIndex:
    def setup_method(self):
        _clear_cache("fear_greed")

    @pytest.mark.asyncio
    async def test_successful_response(self):
        mock_data = {
            "data": [{"value": "72", "value_classification": "Greed", "timestamp": "1234567890"}]
        }
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch("src.ai_engine.FEAR_GREED_API_URL", "https://api.alternative.me/fng/?limit=1"):
            result = await fetch_fear_greed_index(session=mock_session)

        assert result["value"] == 72
        assert result["classification"] == "Greed"
        assert result["timestamp"] == "1234567890"

    @pytest.mark.asyncio
    async def test_failure_returns_neutral_defaults(self):
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Network error"))

        result = await fetch_fear_greed_index(session=mock_session)

        assert result["value"] == 50
        assert result["classification"] == "Neutral"
        assert result["timestamp"] == ""

    @pytest.mark.asyncio
    async def test_http_error_returns_neutral_defaults(self):
        mock_resp = AsyncMock()
        mock_resp.status = 503
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        result = await fetch_fear_greed_index(session=mock_session)

        assert result["value"] == 50
        assert result["classification"] == "Neutral"

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        cached = {"value": 35, "classification": "Fear", "timestamp": "9999"}
        _set_cached("fear_greed", cached)

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=AssertionError("Should not be called"))

        result = await fetch_fear_greed_index(session=mock_session)

        assert result is cached

    @pytest.mark.asyncio
    async def test_result_dict_has_required_keys(self):
        mock_data = {
            "data": [{"value": "50", "value_classification": "Neutral", "timestamp": "0"}]
        }
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        result = await fetch_fear_greed_index(session=mock_session)

        assert "value" in result
        assert "classification" in result
        assert "timestamp" in result
        assert isinstance(result["value"], int)


# ---------------------------------------------------------------------------
# get_ai_insight
# ---------------------------------------------------------------------------

class TestGetAiInsight:
    def setup_method(self):
        _clear_cache("news:BTC", "social:BTC", "fear_greed")

    @pytest.mark.asyncio
    async def test_combines_all_three_sources(self):
        news_result = SentimentResult(score=0.6, label="Positive", summary="News bullish")
        social_result = SentimentResult(score=0.4, label="Positive", summary="Social bullish")
        fg_result = {"value": 72, "classification": "Greed", "timestamp": "0"}

        with (
            patch("src.ai_engine.fetch_news_sentiment", return_value=news_result),
            patch("src.ai_engine.fetch_social_sentiment", return_value=social_result),
            patch("src.ai_engine.fetch_fear_greed_index", return_value=fg_result),
        ):
            result = await get_ai_insight("BTCUSDT")

        assert result.score == pytest.approx(0.5, rel=1e-3)
        assert result.label == "Positive"
        assert "Fear&Greed: 72 (Greed)" in result.summary
        assert "News: positive" in result.summary
        assert "Social: positive" in result.summary

    @pytest.mark.asyncio
    async def test_no_api_keys_returns_neutral_with_fear_greed(self):
        """Without API keys, news/social are neutral; Fear&Greed is always included."""
        _clear_cache("fear_greed")
        fg_result = {"value": 30, "classification": "Fear", "timestamp": "0"}

        with (
            patch("src.ai_engine.NEWS_API_KEY", ""),
            patch("src.ai_engine.SOCIAL_SENTIMENT_API_KEY", ""),
            patch("src.ai_engine.fetch_fear_greed_index", return_value=fg_result),
        ):
            result = await get_ai_insight("BTCUSDT")

        assert result.score == 0.0
        assert result.label == "Neutral"
        assert "Fear&Greed: 30 (Fear)" in result.summary

    @pytest.mark.asyncio
    async def test_combined_score_average_of_news_and_social(self):
        news_result = SentimentResult(score=0.8, label="Positive")
        social_result = SentimentResult(score=-0.2, label="Neutral")
        fg_result = {"value": 50, "classification": "Neutral", "timestamp": "0"}

        with (
            patch("src.ai_engine.fetch_news_sentiment", return_value=news_result),
            patch("src.ai_engine.fetch_social_sentiment", return_value=social_result),
            patch("src.ai_engine.fetch_fear_greed_index", return_value=fg_result),
        ):
            result = await get_ai_insight("BTCUSDT")

        assert result.score == pytest.approx(0.3, rel=1e-3)

    @pytest.mark.asyncio
    async def test_negative_sentiment_label(self):
        news_result = SentimentResult(score=-0.5, label="Negative")
        social_result = SentimentResult(score=-0.4, label="Negative")
        fg_result = {"value": 20, "classification": "Extreme Fear", "timestamp": "0"}

        with (
            patch("src.ai_engine.fetch_news_sentiment", return_value=news_result),
            patch("src.ai_engine.fetch_social_sentiment", return_value=social_result),
            patch("src.ai_engine.fetch_fear_greed_index", return_value=fg_result),
        ):
            result = await get_ai_insight("BTCUSDT")

        assert result.label == "Negative"
        assert "Fear&Greed: 20 (Extreme Fear)" in result.summary


# ---------------------------------------------------------------------------
# Existing whale / volume delta tests (regression)
# ---------------------------------------------------------------------------

class TestDetectWhaleTrade:
    def test_above_threshold(self):
        alert = detect_whale_trade(50000.0, 25.0)  # $1.25M
        assert alert is not None
        assert alert.amount_usd == pytest.approx(1_250_000.0)

    def test_below_threshold(self):
        alert = detect_whale_trade(50000.0, 1.0)  # $50k
        assert alert is None

    def test_exact_threshold(self):
        alert = detect_whale_trade(100.0, 10000.0)  # $1M exactly
        assert alert is not None


class TestDetectVolumeDeltaSpike:
    def test_spike_detected(self):
        assert detect_volume_delta_spike(500.0, 100.0) is True

    def test_no_spike(self):
        assert detect_volume_delta_spike(150.0, 100.0) is False

    def test_zero_avg_delta(self):
        assert detect_volume_delta_spike(1000.0, 0.0) is False

    def test_negative_delta(self):
        assert detect_volume_delta_spike(-500.0, 100.0) is True
