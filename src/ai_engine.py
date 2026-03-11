"""AI & Predictive modules – sentiment, whale detection, optional LSTM/Transformer.

This module provides *async* helpers that can be wired into the confidence
scorer. All external API calls are optional and degrade gracefully.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp

from config import NEWS_API_KEY, SOCIAL_SENTIMENT_API_KEY
from src.utils import get_logger

log = get_logger("ai_engine")


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


@dataclass
class WhaleAlert:
    """Whale trade or wallet movement."""
    symbol: str = ""
    side: str = ""            # BUY / SELL
    amount_usd: float = 0.0
    exchange: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# News sentiment (stub – wired to optional external API)
# ---------------------------------------------------------------------------

async def fetch_news_sentiment(
    symbol: str,
    session: Optional[aiohttp.ClientSession] = None,
) -> SentimentResult:
    """Fetch news sentiment for *symbol*.

    If ``NEWS_API_KEY`` is not configured, returns a neutral stub.
    """
    if not NEWS_API_KEY:
        return SentimentResult(score=0.0, label="Neutral", summary="No API key")
    try:
        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()
        try:
            url = f"https://newsapi.example.com/v1/sentiment?q={symbol}&apiKey={NEWS_API_KEY}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    score = float(data.get("score", 0.0))
                    label = "Positive" if score > 0.2 else ("Negative" if score < -0.2 else "Neutral")
                    return SentimentResult(
                        score=score,
                        label=label,
                        summary=data.get("summary", ""),
                        sources=data.get("sources", []),
                    )
        finally:
            if own_session and session is not None:
                await session.close()
    except Exception as exc:
        log.debug("News sentiment fetch failed for %s: %s", symbol, exc)
    return SentimentResult(score=0.0, label="Neutral", summary="Fetch failed")


# ---------------------------------------------------------------------------
# Social-media sentiment (stub)
# ---------------------------------------------------------------------------

async def fetch_social_sentiment(
    symbol: str,
    session: Optional[aiohttp.ClientSession] = None,
) -> SentimentResult:
    """Fetch social-media sentiment for *symbol*."""
    if not SOCIAL_SENTIMENT_API_KEY:
        return SentimentResult(score=0.0, label="Neutral", summary="No API key")
    try:
        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()
        try:
            url = (
                f"https://social-api.example.com/v1/sentiment"
                f"?symbol={symbol}&key={SOCIAL_SENTIMENT_API_KEY}"
            )
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    score = float(data.get("score", 0.0))
                    label = "Positive" if score > 0.2 else ("Negative" if score < -0.2 else "Neutral")
                    return SentimentResult(
                        score=score,
                        label=label,
                        summary=data.get("summary", ""),
                        sources=data.get("sources", []),
                    )
        finally:
            if own_session and session is not None:
                await session.close()
    except Exception as exc:
        log.debug("Social sentiment fetch failed for %s: %s", symbol, exc)
    return SentimentResult(score=0.0, label="Neutral", summary="Fetch failed")


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
    """Combine news + social sentiment into a single insight."""
    news, social = await asyncio.gather(
        fetch_news_sentiment(symbol, session),
        fetch_social_sentiment(symbol, session),
    )
    combined_score = (news.score + social.score) / 2.0
    if combined_score > 0.2:
        label = "Positive"
    elif combined_score < -0.2:
        label = "Negative"
    else:
        label = "Neutral"

    parts = []
    if news.summary:
        parts.append(news.summary)
    if social.summary:
        parts.append(social.summary)
    summary = " — ".join(parts) if parts else "No data"

    return SentimentResult(
        score=combined_score,
        label=label,
        summary=summary,
        sources=news.sources + social.sources,
    )
