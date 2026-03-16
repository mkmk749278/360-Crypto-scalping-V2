"""OpenAI GPT-4 trade evaluator — AI analyst replacement.

Sends signal context (SMC events, indicators, sentiment, Fear & Greed) to
GPT-4o-mini and receives a confidence rating + reasoning.

Degrades gracefully: if ``OPENAI_API_KEY`` is not set the evaluator is
disabled and every call returns a neutral :class:`EvalResult` immediately.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import aiohttp

from src.utils import get_logger

log = get_logger("openai_evaluator")

_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_CACHE_TTL: float = 120.0  # seconds
_TIMEOUT: float = 5.0       # HTTP timeout
_MAX_ADJUSTMENT: float = 15.0
_CACHE_MAX_ITEMS: int = 256


@dataclass
class EvalResult:
    """Result of an OpenAI trade evaluation."""
    adjustment: float = 0.0   # -15 to +15 confidence adjustment
    recommended: bool = True  # False = AI says skip this trade
    reasoning: str = ""
    model: str = ""


class OpenAIEvaluator:
    """Async wrapper around the OpenAI Chat Completions API.

    Sends a structured prompt describing the current signal and returns a
    :class:`EvalResult` with a confidence adjustment and trade recommendation.

    All calls are cached per evaluation fingerprint for :data:`_CACHE_TTL`
    seconds to avoid spamming the API on every scan cycle.
    """

    def __init__(self) -> None:
        self._api_key: str = os.getenv("OPENAI_API_KEY", "")
        self._model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._enabled: bool = bool(self._api_key)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[float, EvalResult]] = {}

    @property
    def enabled(self) -> bool:
        """Return ``True`` when the API key is configured."""
        return self._enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        symbol: str,
        direction: str,
        channel: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        indicators: Dict[str, Any],
        smc_summary: str,
        ai_sentiment_summary: str,
        market_phase: str,
        confidence_before: float,
    ) -> EvalResult:
        """Evaluate a trade signal via GPT-4o-mini.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. ``"BTCUSDT"``.
        direction:
            ``"LONG"`` or ``"SHORT"``.
        channel:
            Channel name, e.g. ``"360_SCALP"``.
        entry_price, stop_loss, tp1, tp2:
            Signal price levels.
        indicators:
            Dict of computed indicators (``ema9_last``, ``ema21_last``,
            ``adx_last``, ``rsi_last``, ``atr_last``, …).
        smc_summary:
            Human-readable SMC event description.
        ai_sentiment_summary:
            Combined news/social/fear-greed summary string.
        market_phase:
            Market regime label, e.g. ``"TRENDING_UP"``.
        confidence_before:
            Confidence score (0–100) before this evaluation.

        Returns
        -------
        EvalResult
        """
        if not self._enabled:
            return EvalResult(
                adjustment=0.0,
                reasoning="OpenAI not configured",
                recommended=True,
            )

        self._prune_cache()
        cache_key = self._build_cache_key(
            symbol=symbol,
            direction=direction,
            channel=channel,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            indicators=indicators,
            smc_summary=smc_summary,
            ai_sentiment_summary=ai_sentiment_summary,
            market_phase=market_phase,
            confidence_before=confidence_before,
        )
        cached = self._cache.get(cache_key)
        if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL:
            return cached[1]

        prompt = self._build_prompt(
            symbol=symbol,
            direction=direction,
            channel=channel,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            indicators=indicators,
            smc_summary=smc_summary,
            ai_sentiment_summary=ai_sentiment_summary,
            market_phase=market_phase,
            confidence_before=confidence_before,
        )

        try:
            result = await self._call_api(prompt)
        except Exception as exc:
            log.debug("OpenAI evaluation failed for {}: {}", symbol, exc)
            return EvalResult(adjustment=0.0, reasoning="OpenAI error", recommended=True)

        self._cache[cache_key] = (time.monotonic(), result)
        return result

    async def close(self) -> None:
        """Close the underlying :class:`aiohttp.ClientSession` if open."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prune_cache(self) -> None:
        now = time.monotonic()
        stale_keys = [
            key for key, (ts, _) in self._cache.items()
            if (now - ts) >= _CACHE_TTL
        ]
        for key in stale_keys:
            self._cache.pop(key, None)
        if len(self._cache) <= _CACHE_MAX_ITEMS:
            return
        for key, _ in sorted(self._cache.items(), key=lambda item: item[1][0])[: len(self._cache) - _CACHE_MAX_ITEMS]:
            self._cache.pop(key, None)

    def _build_cache_key(
        self,
        symbol: str,
        direction: str,
        channel: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        indicators: Dict[str, Any],
        smc_summary: str,
        ai_sentiment_summary: str,
        market_phase: str,
        confidence_before: float,
    ) -> str:
        payload = {
            "symbol": symbol,
            "direction": direction,
            "channel": channel,
            "entry_price": round(entry_price, 8),
            "stop_loss": round(stop_loss, 8),
            "tp1": round(tp1, 8),
            "tp2": round(tp2, 8),
            "market_phase": market_phase,
            "confidence_before": round(confidence_before, 4),
            "smc_summary": smc_summary.strip(),
            "ai_sentiment_summary": ai_sentiment_summary.strip(),
            "indicators": {
                key: indicators.get(key)
                for key in sorted(indicators)
            },
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
        return f"{symbol}:{channel}:{digest}"

    def _parse_response_content(self, content: str) -> Dict[str, Any]:
        raw = content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            parsed = json.loads(raw[start: end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI response must be a JSON object")
        return parsed

    @staticmethod
    def _coerce_recommended(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "skip"}
        return bool(value)

    def _build_prompt(
        self,
        symbol: str,
        direction: str,
        channel: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        indicators: Dict[str, Any],
        smc_summary: str,
        ai_sentiment_summary: str,
        market_phase: str,
        confidence_before: float,
    ) -> str:
        ema9 = indicators.get("ema9_last", "N/A")
        ema21 = indicators.get("ema21_last", "N/A")
        adx = indicators.get("adx_last", "N/A")
        rsi = indicators.get("rsi_last", "N/A")
        atr = indicators.get("atr_last", "N/A")

        def _fmt(v: Any) -> str:
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        return (
            "You are an expert crypto trading analyst. Evaluate this signal and provide your assessment.\n\n"
            "Signal Details:\n"
            f"- Pair: {symbol} {direction}\n"
            f"- Channel: {channel}\n"
            f"- Entry: {_fmt(entry_price)}\n"
            f"- Stop Loss: {_fmt(stop_loss)}\n"
            f"- TP1: {_fmt(tp1)}, TP2: {_fmt(tp2)}\n"
            f"- Current Confidence: {confidence_before:.1f}%\n\n"
            "Technical Indicators:\n"
            f"- EMA9: {_fmt(ema9)}, EMA21: {_fmt(ema21)}\n"
            f"- ADX: {_fmt(adx)}, RSI: {_fmt(rsi)}\n"
            f"- ATR: {_fmt(atr)}\n\n"
            f"Smart Money Concepts: {smc_summary}\n"
            f"AI Sentiment: {ai_sentiment_summary}\n"
            f"Market Phase: {market_phase}\n\n"
            'Respond ONLY with valid JSON (no markdown, no code fences):\n'
            '{"confidence_adjustment": <number -15 to 15>, "recommended": <true or false>, "reasoning": "<short explanation>"}'
        )

    async def _call_api(self, prompt: str) -> EvalResult:
        """POST to the OpenAI chat completions endpoint and parse the result."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a crypto trading analyst. Respond only with JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 150,
        }

        timeout = aiohttp.ClientTimeout(total=_TIMEOUT)
        async with self._session.post(
            _OPENAI_CHAT_URL, headers=headers, json=body, timeout=timeout
        ) as resp:
            if resp.status != 200:
                log.warning("OpenAI API returned status {}", resp.status)
                return EvalResult(
                    adjustment=0.0, reasoning="OpenAI API error", recommended=True
                )
            data = await resp.json(content_type=None)

        try:
            content = data["choices"][0]["message"]["content"]
            parsed = self._parse_response_content(str(content))
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            log.debug("Failed to parse OpenAI response: {}", exc)
            return EvalResult(
                adjustment=0.0,
                reasoning="Invalid OpenAI response",
                recommended=True,
                model=self._model,
            )

        try:
            raw_adj = float(parsed.get("confidence_adjustment", 0.0))
        except (TypeError, ValueError):
            raw_adj = 0.0
        adjustment = max(-_MAX_ADJUSTMENT, min(_MAX_ADJUSTMENT, raw_adj))
        recommended = self._coerce_recommended(parsed.get("recommended", True))
        reasoning = str(parsed.get("reasoning", ""))

        return EvalResult(
            adjustment=adjustment,
            recommended=recommended,
            reasoning=reasoning,
            model=self._model,
        )
