"""Tests for src.openai_evaluator — EvalResult, OpenAIEvaluator, caching, degradation."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.openai_evaluator import EvalResult, OpenAIEvaluator, _MAX_ADJUSTMENT


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

class TestEvalResult:
    def test_defaults(self):
        r = EvalResult()
        assert r.adjustment == 0.0
        assert r.recommended is True
        assert r.reasoning == ""
        assert r.model == ""

    def test_custom_values(self):
        r = EvalResult(adjustment=5.0, recommended=False, reasoning="bearish", model="gpt-4o-mini")
        assert r.adjustment == 5.0
        assert r.recommended is False
        assert r.reasoning == "bearish"
        assert r.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# OpenAIEvaluator — disabled (no API key)
# ---------------------------------------------------------------------------

class TestOpenAIEvaluatorDisabled:
    def test_enabled_false_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ev = OpenAIEvaluator()
        assert ev.enabled is False

    @pytest.mark.asyncio
    async def test_evaluate_returns_neutral_when_disabled(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ev = OpenAIEvaluator()
        result = await ev.evaluate(
            symbol="BTCUSDT",
            direction="LONG",
            channel="360_SCALP",
            entry_price=50000.0,
            stop_loss=49000.0,
            tp1=51000.0,
            tp2=52000.0,
            indicators={},
            smc_summary="None",
            ai_sentiment_summary="Neutral",
            market_phase="TRENDING_UP",
            confidence_before=75.0,
        )
        assert result.recommended is True
        assert result.adjustment == 0.0
        assert "configured" in result.reasoning.lower() or "not" in result.reasoning.lower()


# ---------------------------------------------------------------------------
# OpenAIEvaluator — enabled (mocked HTTP)
# ---------------------------------------------------------------------------

class TestOpenAIEvaluatorEnabled:
    def _make_evaluator(self, monkeypatch, api_key: str = "test-key") -> OpenAIEvaluator:
        monkeypatch.setenv("OPENAI_API_KEY", api_key)
        return OpenAIEvaluator()

    def _mock_response(self, content: str, status: int = 200):
        """Create a mock aiohttp response."""
        mock_resp = AsyncMock()
        mock_resp.status = status
        mock_resp.json = AsyncMock(return_value={
            "choices": [{"message": {"content": content}}]
        })
        mock_resp.text = AsyncMock(return_value="error text")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        return mock_resp

    @pytest.mark.asyncio
    async def test_enabled_true_with_key(self, monkeypatch):
        ev = self._make_evaluator(monkeypatch)
        assert ev.enabled is True

    @pytest.mark.asyncio
    async def test_evaluate_parses_valid_response(self, monkeypatch):
        ev = self._make_evaluator(monkeypatch)
        import json
        content = json.dumps({
            "confidence_adjustment": 8.0,
            "recommended": True,
            "reasoning": "Strong bullish setup",
        })
        mock_resp = self._mock_response(content)
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)

        ev._session = mock_session

        result = await ev.evaluate(
            symbol="BTCUSDT",
            direction="LONG",
            channel="360_SCALP",
            entry_price=50000.0,
            stop_loss=49000.0,
            tp1=51000.0,
            tp2=52000.0,
            indicators={"ema9_last": 50100.0, "ema21_last": 49900.0},
            smc_summary="Sweep LONG at 49500",
            ai_sentiment_summary="News: bullish",
            market_phase="TRENDING_UP",
            confidence_before=75.0,
        )
        assert result.adjustment == 8.0
        assert result.recommended is True
        assert result.reasoning == "Strong bullish setup"

    @pytest.mark.asyncio
    async def test_evaluate_clamps_adjustment_to_max(self, monkeypatch):
        ev = self._make_evaluator(monkeypatch)
        import json
        content = json.dumps({
            "confidence_adjustment": 999.0,
            "recommended": True,
            "reasoning": "extreme",
        })
        mock_resp = self._mock_response(content)
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)
        ev._session = mock_session

        result = await ev.evaluate(
            "BTCUSDT", "LONG", "360_SCALP",
            50000.0, 49000.0, 51000.0, 52000.0,
            {}, "", "", "TRENDING_UP", 70.0,
        )
        assert result.adjustment == _MAX_ADJUSTMENT

    @pytest.mark.asyncio
    async def test_evaluate_clamps_negative_adjustment(self, monkeypatch):
        ev = self._make_evaluator(monkeypatch)
        import json
        content = json.dumps({
            "confidence_adjustment": -999.0,
            "recommended": False,
            "reasoning": "skip",
        })
        mock_resp = self._mock_response(content)
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)
        ev._session = mock_session

        result = await ev.evaluate(
            "BTCUSDT", "LONG", "360_SCALP",
            50000.0, 49000.0, 51000.0, 52000.0,
            {}, "", "", "TRENDING_UP", 70.0,
        )
        assert result.adjustment == -_MAX_ADJUSTMENT
        assert result.recommended is False

    @pytest.mark.asyncio
    async def test_evaluate_returns_neutral_on_api_error(self, monkeypatch):
        ev = self._make_evaluator(monkeypatch)
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=Exception("network error"))
        ev._session = mock_session

        result = await ev.evaluate(
            "BTCUSDT", "LONG", "360_SCALP",
            50000.0, 49000.0, 51000.0, 52000.0,
            {}, "", "", "TRENDING_UP", 70.0,
        )
        assert result.recommended is True
        assert result.adjustment == 0.0
        assert "error" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_evaluate_returns_neutral_on_json_parse_error(self, monkeypatch):
        ev = self._make_evaluator(monkeypatch)
        mock_resp = self._mock_response("not valid json at all!!!")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)
        ev._session = mock_session

        result = await ev.evaluate(
            "BTCUSDT", "LONG", "360_SCALP",
            50000.0, 49000.0, 51000.0, 52000.0,
            {}, "", "", "TRENDING_UP", 70.0,
        )
        assert result.recommended is True
        assert result.adjustment == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_parses_fenced_json_response(self, monkeypatch):
        ev = self._make_evaluator(monkeypatch)
        content = '```json\n{"confidence_adjustment": 4, "recommended": "true", "reasoning": "ok"}\n```'
        mock_resp = self._mock_response(content)
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)
        ev._session = mock_session

        result = await ev.evaluate(
            "BTCUSDT", "LONG", "360_SCALP",
            50000.0, 49000.0, 51000.0, 52000.0,
            {}, "", "", "TRENDING_UP", 70.0,
        )

        assert result.adjustment == 4.0
        assert result.recommended is True

    @pytest.mark.asyncio
    async def test_evaluate_returns_neutral_on_non_200(self, monkeypatch):
        ev = self._make_evaluator(monkeypatch)
        mock_resp = self._mock_response("{}", status=429)
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_resp)
        ev._session = mock_session

        result = await ev.evaluate(
            "BTCUSDT", "LONG", "360_SCALP",
            50000.0, 49000.0, 51000.0, 52000.0,
            {}, "", "", "TRENDING_UP", 70.0,
        )
        assert result.recommended is True
        assert result.adjustment == 0.0

    @pytest.mark.asyncio
    async def test_cache_returns_same_result(self, monkeypatch):
        """Second call with same symbol+channel should use cache without HTTP."""
        ev = self._make_evaluator(monkeypatch)
        import json
        content = json.dumps({"confidence_adjustment": 3.0, "recommended": True, "reasoning": "ok"})
        call_count = 0

        def make_resp():
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={
                "choices": [{"message": {"content": content}}]
            })
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            return mock_resp

        def counting_post(*a, **kw):
            nonlocal call_count
            call_count += 1
            return make_resp()

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = counting_post
        ev._session = mock_session

        await ev.evaluate("BTCUSDT", "LONG", "360_SCALP", 50000, 49000, 51000, 52000, {}, "", "", "T", 70)
        await ev.evaluate("BTCUSDT", "LONG", "360_SCALP", 50000, 49000, 51000, 52000, {}, "", "", "T", 70)
        assert call_count == 1  # second call served from cache

    @pytest.mark.asyncio
    async def test_cache_expired_triggers_new_call(self, monkeypatch):
        """Cache with expired TTL should trigger a new HTTP call."""
        ev = self._make_evaluator(monkeypatch)
        import json
        content = json.dumps({"confidence_adjustment": 3.0, "recommended": True, "reasoning": "ok"})
        call_count = 0

        def make_resp():
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={
                "choices": [{"message": {"content": content}}]
            })
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            return mock_resp

        def counting_post(*a, **kw):
            nonlocal call_count
            call_count += 1
            return make_resp()

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = counting_post
        ev._session = mock_session

        # Pre-seed cache with an expired entry
        expired_key = ev._build_cache_key(
            "BTCUSDT", "LONG", "360_SCALP",
            50000.0, 49000.0, 51000.0, 52000.0,
            {}, "", "", "T", 70.0,
        )
        ev._cache[expired_key] = (time.monotonic() - 200, EvalResult())

        await ev.evaluate("BTCUSDT", "LONG", "360_SCALP", 50000, 49000, 51000, 52000, {}, "", "", "T", 70)
        assert call_count == 1  # cache was expired, new call made

    @pytest.mark.asyncio
    async def test_distinct_signal_contexts_do_not_share_cache(self, monkeypatch):
        ev = self._make_evaluator(monkeypatch)
        import json
        content = json.dumps({"confidence_adjustment": 3.0, "recommended": True, "reasoning": "ok"})
        call_count = 0

        def make_resp():
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={
                "choices": [{"message": {"content": content}}]
            })
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            return mock_resp

        def counting_post(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            return make_resp()

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = counting_post
        ev._session = mock_session

        await ev.evaluate("BTCUSDT", "LONG", "360_SCALP", 50000, 49000, 51000, 52000, {"ema9_last": 1}, "", "", "T", 70)
        await ev.evaluate("BTCUSDT", "LONG", "360_SCALP", 50010, 49000, 51000, 52000, {"ema9_last": 1}, "", "", "T", 70)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_close_session(self, monkeypatch):
        ev = self._make_evaluator(monkeypatch)
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        ev._session = mock_session

        await ev.close()
        mock_session.close.assert_called_once()
        assert ev._session is None


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_prompt_contains_signal_details(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        ev = OpenAIEvaluator()
        prompt = ev._build_prompt(
            symbol="ETHUSDT",
            direction="SHORT",
            channel="360_SWING",
            entry_price=3000.0,
            stop_loss=3100.0,
            tp1=2900.0,
            tp2=2800.0,
            indicators={"ema9_last": 3010.0, "ema21_last": 2990.0, "rsi_last": 65.0},
            smc_summary="Sweep SHORT at 3050",
            ai_sentiment_summary="News: bearish",
            market_phase="TRENDING_DOWN",
            confidence_before=80.0,
        )
        assert "ETHUSDT SHORT" in prompt
        assert "360_SWING" in prompt
        assert "TRENDING_DOWN" in prompt
        assert "News: bearish" in prompt
        assert "confidence_adjustment" in prompt
        assert "recommended" in prompt

    def test_prompt_handles_missing_indicators(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        ev = OpenAIEvaluator()
        prompt = ev._build_prompt(
            "BTCUSDT", "LONG", "360_SCALP",
            50000.0, 49000.0, 51000.0, 52000.0,
            {},  # empty indicators
            "", "", "RANGING", 70.0,
        )
        assert "N/A" in prompt  # missing indicators shown as N/A
