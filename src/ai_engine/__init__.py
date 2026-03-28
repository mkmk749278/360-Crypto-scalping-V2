"""AI & Predictive modules – sentiment, whale detection, optional LSTM/Transformer.

This package provides *async* helpers that can be wired into the confidence
scorer. All external API calls are optional and degrade gracefully.

Real API integrations:
- CryptoPanic (https://cryptopanic.com/api/v1/) — news sentiment (free tier)
- LunarCrush (https://lunarcrush.com/api4/) — social sentiment (free tier)
- Alternative.me Fear & Greed Index (https://api.alternative.me/fng/) — free, no key

Sub-modules:
- ``predictor`` — async multi-factor signal prediction pipeline
- ``scorer`` — per-pair dynamic AI confidence scoring
- ``feedback`` — AI prediction tracking and feedback integration
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from config import FEAR_GREED_API_URL, NEWS_API_KEY, SOCIAL_SENTIMENT_API_KEY
from src.utils import get_logger

log = get_logger("ai_engine")


# ---------------------------------------------------------------------------
# Simple TTL cache (module-level, avoids spamming free-tier APIs)
# ---------------------------------------------------------------------------

_cache: Dict[str, Tuple[float, Any]] = {}
_CACHE_MAX_AGE: float = 3600.0
_CACHE_MAX_ITEMS: int = 256
_shared_session: Optional[aiohttp.ClientSession] = None


def _prune_cache(max_age: float = _CACHE_MAX_AGE) -> None:
    """Drop stale cache entries and trim oversized caches."""
    now = time.monotonic()
    stale_keys = [key for key, (ts, _) in _cache.items() if (now - ts) >= max_age]
    for key in stale_keys:
        _cache.pop(key, None)
    if len(_cache) <= _CACHE_MAX_ITEMS:
        return
    for key, _ in sorted(_cache.items(), key=lambda item: item[1][0])[: len(_cache) - _CACHE_MAX_ITEMS]:
        _cache.pop(key, None)


def _get_cached(key: str, ttl: float) -> Optional[Any]:
    """Return cached value if still valid, else ``None``."""
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry[0]) < ttl:
        return entry[1]
    if entry is not None:
        _cache.pop(key, None)
    return None


def _set_cached(key: str, value: Any) -> None:
    """Store *value* in the module-level cache under *key*."""
    _prune_cache()
    _cache[key] = (time.monotonic(), value)


async def _get_shared_session() -> aiohttp.ClientSession:
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        _shared_session = aiohttp.ClientSession()
    return _shared_session


async def close_shared_session() -> None:
    global _shared_session
    if _shared_session is not None and not _shared_session.closed:
        await _shared_session.close()
    _shared_session = None


def _strip_quote_currency(symbol: str) -> str:
    """Strip common quote-currency suffixes to get the base coin name.

    Examples::

        "BTCUSDT"  → "BTC"
        "ETHBUSD"  → "ETH"
        "SOLUSDC"  → "SOL"
        "BTC"      → "BTC"  (already clean)
    """
    for suffix in ("USDT", "BUSD", "USDC"):
        if symbol.upper().endswith(suffix):
            return symbol[: -len(suffix)].upper()
    return symbol.upper()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    """Aggregated sentiment for a symbol."""
    score: float = 0.0        # -1 (bearish) to +1 (bullish)
    label: str = "Neutral"    # Positive / Negative / Neutral / Bullish / Bearish
    summary: str = ""
    sources: List[str] = field(default_factory=list)
    fear_greed_value: int = 50  # 0–100; 50 = neutral (default when unavailable)


@dataclass
class WhaleAlert:
    """Whale trade or wallet movement."""
    symbol: str = ""
    side: str = ""            # BUY / SELL
    amount_usd: float = 0.0
    exchange: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_to_label(score: float) -> str:
    """Convert a normalised [-1, +1] score to a human-readable label."""
    if score > 0.2:
        return "Positive"
    if score < -0.2:
        return "Negative"
    return "Neutral"


# ---------------------------------------------------------------------------
# News sentiment — CryptoPanic API
# ---------------------------------------------------------------------------

async def fetch_news_sentiment(
    symbol: str,
    session: Optional[aiohttp.ClientSession] = None,
) -> SentimentResult:
    """Fetch news sentiment for *symbol* via the CryptoPanic API.

    Uses the ``NEWS_API_KEY`` environment variable as the CryptoPanic auth
    token.  If the key is not set, or the request fails for any reason, a
    neutral stub is returned so downstream code never crashes.

    Results are cached per symbol for **60 seconds** to respect the free-tier
    rate limit of ~5 requests/min.

    Args:
        symbol: Trading pair such as ``"BTCUSDT"`` or ``"ETHBUSD"``.
        session: Optional shared :class:`aiohttp.ClientSession`.  If ``None``
            a temporary session is created and closed automatically.

    Returns:
        A :class:`SentimentResult` with score in ``[-1, +1]``.
    """
    if not NEWS_API_KEY:
        return SentimentResult(score=0.0, label="Neutral", summary="No API key")

    coin = _strip_quote_currency(symbol)
    cache_key = f"news:{coin}"
    cached = _get_cached(cache_key, ttl=60.0)
    if cached is not None:
        return cached

    try:
        session = session or await _get_shared_session()
        url = (
            f"https://cryptopanic.com/api/v1/posts/"
            f"?auth_token={NEWS_API_KEY}&currencies={coin}&filter=hot&kind=news"
        )
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                results = data.get("results", [])
                bullish = sum(1 for r in results if r.get("kind") == "bullish")
                bearish = sum(1 for r in results if r.get("kind") == "bearish")
                total = max(len(results), 1)
                score = float(bullish - bearish) / total
                # Clamp to [-1, +1] just in case
                score = max(-1.0, min(1.0, score))
                label = _score_to_label(score)
                top_title = results[0].get("title", "") if results else ""
                sources = [
                    r.get("source", {}).get("domain", "")
                    for r in results
                    if r.get("source", {}).get("domain")
                ]
                result = SentimentResult(
                    score=score,
                    label=label,
                    summary=top_title,
                    sources=sources,
                )
                _set_cached(cache_key, result)
                return result
    except Exception as exc:
        log.debug("News sentiment fetch failed for {}: {}", symbol, exc)
    return SentimentResult(score=0.0, label="Neutral", summary="Fetch failed")


# ---------------------------------------------------------------------------
# Social-media sentiment — LunarCrush API
# ---------------------------------------------------------------------------

async def fetch_social_sentiment(
    symbol: str,
    session: Optional[aiohttp.ClientSession] = None,
) -> SentimentResult:
    """Fetch social-media sentiment for *symbol* via the LunarCrush API v4.

    Uses the ``SOCIAL_SENTIMENT_API_KEY`` environment variable as a Bearer
    token.  Falls back to neutral when the key is absent or the request fails.

    LunarCrush scores are typically 0–100; they are normalised to ``[-1, +1]``
    via ``(score - 50) / 50``.  ``sentiment_score`` is preferred; if absent
    ``galaxy_score`` is used instead.

    Results are cached per symbol for **300 seconds** (social data updates
    less frequently than news).

    Args:
        symbol: Trading pair such as ``"BTCUSDT"``.
        session: Optional shared :class:`aiohttp.ClientSession`.

    Returns:
        A :class:`SentimentResult` with score in ``[-1, +1]``.
    """
    if not SOCIAL_SENTIMENT_API_KEY:
        return SentimentResult(score=0.0, label="Neutral", summary="No API key")

    coin = _strip_quote_currency(symbol)
    cache_key = f"social:{coin}"
    cached = _get_cached(cache_key, ttl=300.0)
    if cached is not None:
        return cached

    try:
        session = session or await _get_shared_session()
        url = f"https://lunarcrush.com/api4/public/coins/{coin}/v1"
        headers = {"Authorization": f"Bearer {SOCIAL_SENTIMENT_API_KEY}"}
        async with session.get(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                coin_data = data.get("data", {})
                raw_score: Optional[float] = coin_data.get("sentiment_score")
                if raw_score is None:
                    raw_score = coin_data.get("galaxy_score")
                if raw_score is not None:
                    # Normalise 0-100 → -1 to +1
                    score = (float(raw_score) - 50.0) / 50.0
                    score = max(-1.0, min(1.0, score))
                else:
                    score = 0.0
                label = _score_to_label(score)
                social_volume = coin_data.get("social_volume")
                galaxy_score = coin_data.get("galaxy_score")
                parts = []
                if social_volume is not None:
                    parts.append(f"social_volume={social_volume}")
                if galaxy_score is not None:
                    parts.append(f"galaxy_score={galaxy_score}")
                summary = ", ".join(parts) if parts else ""
                result = SentimentResult(
                    score=score,
                    label=label,
                    summary=summary,
                    sources=["lunarcrush.com"],
                )
                _set_cached(cache_key, result)
                return result
    except Exception as exc:
        log.debug("Social sentiment fetch failed for {}: {}", symbol, exc)
    return SentimentResult(score=0.0, label="Neutral", summary="Fetch failed")


# ---------------------------------------------------------------------------
# Fear & Greed Index — Alternative.me (free, no key required)
# ---------------------------------------------------------------------------

async def fetch_fear_greed_index(
    session: Optional[aiohttp.ClientSession] = None,
) -> Dict[str, Any]:
    """Fetch the Bitcoin Fear & Greed Index from Alternative.me.

    No API key is required — the endpoint is completely free and public.

    Results are cached for **3 600 seconds** (1 hour) because the index only
    updates once per day.

    Args:
        session: Optional shared :class:`aiohttp.ClientSession`.

    Returns:
        A dict with keys:

        * ``value`` (int, 0–100)
        * ``classification`` (str, e.g. ``"Greed"``, ``"Fear"``)
        * ``timestamp`` (str, Unix timestamp string from the API)
    """
    cache_key = "fear_greed"
    cached = _get_cached(cache_key, ttl=3600.0)
    if cached is not None:
        return cached  # type: ignore[return-value]

    _neutral: Dict[str, Any] = {"value": 50, "classification": "Neutral", "timestamp": ""}

    try:
        session = session or await _get_shared_session()
        async with session.get(
            FEAR_GREED_API_URL, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status == 200:
                data = await resp.json(content_type=None)
                entry = data.get("data", [{}])[0]
                result: Dict[str, Any] = {
                    "value": int(entry.get("value", 50)),
                    "classification": entry.get("value_classification", "Neutral"),
                    "timestamp": entry.get("timestamp", ""),
                }
                _set_cached(cache_key, result)
                return result
    except Exception as exc:
        log.debug("Fear & Greed fetch failed: {}", exc)
    return _neutral


# ---------------------------------------------------------------------------
# Whale detection (tick-level)
# ---------------------------------------------------------------------------

def detect_whale_trade(
    price: float,
    quantity: float,
    threshold_usd: float = 1_000_000,
) -> Optional[WhaleAlert]:
    """Return a :class:`WhaleAlert` if *price × quantity* ≥ threshold."""
    notional = price * quantity
    if notional >= threshold_usd:
        return WhaleAlert(amount_usd=notional)
    return None


def detect_volume_delta_spike(
    cum_delta: float,
    avg_delta: float,
    multiplier: float = 2.0,
) -> bool:
    """Return True if current cumulative delta is ≥ *multiplier* × average."""
    if avg_delta == 0:
        return False
    return abs(cum_delta) >= multiplier * abs(avg_delta)


# ---------------------------------------------------------------------------
# Aggregate AI insight for a symbol
# ---------------------------------------------------------------------------

async def get_ai_insight(
    symbol: str,
    session: Optional[aiohttp.ClientSession] = None,
) -> SentimentResult:
    """Combine news + social sentiment + Fear & Greed into a single insight.

    All three sources are fetched concurrently.  If any source fails or has no
    API key configured, it contributes a neutral score of 0.0, so the overall
    result degrades gracefully.

    The combined score is a simple average of news and social scores (the Fear
    & Greed index is included in the summary string but not averaged in, so
    that the existing 0–15 confidence-scorer component is unaffected).

    Args:
        symbol: Trading pair such as ``"BTCUSDT"``.
        session: Optional shared :class:`aiohttp.ClientSession`.

    Returns:
        A :class:`SentimentResult` whose ``summary`` includes a human-readable
        description of all three sources, e.g.
        ``"News: bullish — Social: neutral — Fear&Greed: 72 (Greed)"``.
    """
    news, social, fear_greed = await asyncio.gather(
        fetch_news_sentiment(symbol, session),
        fetch_social_sentiment(symbol, session),
        fetch_fear_greed_index(session),
    )
    combined_score = (news.score + social.score) / 2.0
    label = _score_to_label(combined_score)

    fg_value = fear_greed.get("value", 50)
    fg_class = fear_greed.get("classification", "Neutral")

    news_label = news.label.lower() if news.label else "neutral"
    social_label = social.label.lower() if social.label else "neutral"
    summary = (
        f"News: {news_label} — Social: {social_label} — "
        f"Fear&Greed: {fg_value} ({fg_class})"
    )

    return SentimentResult(
        score=combined_score,
        label=label,
        summary=summary,
        sources=news.sources + social.sources,
        fear_greed_value=int(fg_value),
    )


# ---------------------------------------------------------------------------
# Sub-module re-exports for convenience
# ---------------------------------------------------------------------------

from src.ai_engine.predictor import PredictionFeatures, SignalPrediction, SignalPredictor
from src.ai_engine.scorer import AIConfidenceScorer, AIScoreResult
from src.ai_engine.feedback import AIFeedbackAdapter, PredictionRecord

__all__ = [
    # Original exports
    "SentimentResult",
    "WhaleAlert",
    "close_shared_session",
    "detect_volume_delta_spike",
    "detect_whale_trade",
    "fetch_fear_greed_index",
    "fetch_news_sentiment",
    "fetch_social_sentiment",
    "get_ai_insight",
    # New sub-module exports
    "AIConfidenceScorer",
    "AIFeedbackAdapter",
    "AIScoreResult",
    "PredictionFeatures",
    "PredictionRecord",
    "SignalPrediction",
    "SignalPredictor",
]
