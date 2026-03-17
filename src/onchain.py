"""On-Chain Intelligence — exchange flow data as a confidence sub-score.

Fetches net exchange flow data (coins entering / leaving exchanges) from
Glassnode or CryptoQuant and converts the signal into a 0–5 confidence
sub-score:

* Large net **outflows** (coins leaving exchanges) → bullish → score near 5
* Large net **inflows** (coins entering exchanges) → bearish → score near 0
* Neutral / unavailable → 2.5

All API calls degrade gracefully.  If ``ONCHAIN_API_KEY`` is not configured
the client returns a neutral score of ``2.5`` for every symbol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import aiohttp

from src.utils import get_logger

log = get_logger("onchain")

_CACHE_TTL: float = 300.0   # 5 minutes – on-chain data updates slowly
_NEUTRAL_SCORE: float = 2.5
_MAX_SCORE: float = 5.0

# Assets supported by Glassnode's free tier.  All other coins will get a 0.0
# score immediately without making any API calls.
_SUPPORTED_ONCHAIN_ASSETS: frozenset = frozenset({"BTC", "ETH"})

# Glassnode endpoint for BTC exchange net-flow (free tier)
_GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics/transactions"


@dataclass
class OnChainData:
    """On-chain exchange flow snapshot for a single asset."""
    symbol: str = ""
    net_flow_usd: float = 0.0    # positive = inflow (bearish), negative = outflow (bullish)
    source: str = ""
    score: float = _NEUTRAL_SCORE  # 0–5 confidence contribution


class OnChainClient:
    """Async client for on-chain exchange flow data.

    Supports Glassnode (``ONCHAIN_API_KEY`` env-var).  When the key is absent
    every call immediately returns a neutral :class:`OnChainData` so the rest
    of the pipeline is unaffected.

    Results are cached per asset for :data:`_CACHE_TTL` seconds.
    """

    def __init__(self, api_key: str = "") -> None:
        self._api_key: str = api_key
        self._enabled: bool = bool(api_key)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[float, OnChainData]] = {}

    @property
    def enabled(self) -> bool:
        """Return ``True`` when an API key is configured."""
        return self._enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_exchange_flow(self, symbol: str) -> OnChainData:
        """Return on-chain exchange flow data for *symbol*.

        Parameters
        ----------
        symbol:
            Trading pair such as ``"BTCUSDT"`` or just ``"BTC"``.

        Returns
        -------
        OnChainData
            Always returns a valid object; score is :data:`_NEUTRAL_SCORE`
            when data is unavailable.
        """
        coin = _strip_quote_currency(symbol)
        neutral = OnChainData(symbol=coin, source="none", score=_NEUTRAL_SCORE)

        if not self._enabled:
            return neutral

        # Glassnode free-tier only supports BTC and ETH.
        # Skip the API call entirely for other assets to avoid errors and latency.
        if coin.upper() not in _SUPPORTED_ONCHAIN_ASSETS:
            log.debug("On-chain data not supported for %s – returning neutral score", coin)
            return OnChainData(symbol=coin, source="unsupported", score=0.0)

        cached = self._cache.get(coin)
        if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL:
            return cached[1]

        try:
            result = await self._fetch_glassnode(coin)
        except Exception as exc:
            log.debug("On-chain fetch failed for %s: %s", symbol, exc)
            return neutral

        self._cache[coin] = (time.monotonic(), result)
        return result

    async def close(self) -> None:
        """Close the underlying :class:`aiohttp.ClientSession` if open."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _fetch_glassnode(self, coin: str) -> OnChainData:
        """Fetch net exchange flow from Glassnode for *coin*."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        asset = coin.lower()
        url = f"{_GLASSNODE_BASE}/transfers_to_exchanges_sum"
        params: Dict[str, Any] = {
            "a": asset,
            "api_key": self._api_key,
            "i": "24h",
            "limit": 1,
        }

        timeout = aiohttp.ClientTimeout(total=10)
        async with self._session.get(url, params=params, timeout=timeout) as resp:
            if resp.status != 200:
                log.debug(
                    "Glassnode returned %d for %s exchange inflow",
                    resp.status, coin,
                )
                return OnChainData(symbol=coin, source="glassnode", score=_NEUTRAL_SCORE)
            inflow_data: Any = await resp.json(content_type=None)

        # Fetch outflow
        url_out = f"{_GLASSNODE_BASE}/transfers_from_exchanges_sum"
        async with self._session.get(
            url_out, params=params, timeout=timeout
        ) as resp2:
            if resp2.status != 200:
                return OnChainData(symbol=coin, source="glassnode", score=_NEUTRAL_SCORE)
            outflow_data: Any = await resp2.json(content_type=None)

        inflow = _parse_glassnode_latest(inflow_data)
        outflow = _parse_glassnode_latest(outflow_data)

        if inflow is None or outflow is None:
            return OnChainData(symbol=coin, source="glassnode", score=_NEUTRAL_SCORE)

        net_flow = inflow - outflow  # positive = net inflow (bearish)
        score = _net_flow_to_score(net_flow, inflow, outflow)

        log.debug(
            "On-chain %s: inflow=%.2f outflow=%.2f net=%.2f score=%.2f",
            coin, inflow, outflow, net_flow, score,
        )
        return OnChainData(
            symbol=coin,
            net_flow_usd=net_flow,
            source="glassnode",
            score=score,
        )


# ---------------------------------------------------------------------------
# Module-level scoring helper (used by confidence.py import)
# ---------------------------------------------------------------------------

def score_onchain(onchain_data: Optional["OnChainData"]) -> float:
    """Convert an :class:`OnChainData` snapshot to a 0–5 confidence score.

    Parameters
    ----------
    onchain_data:
        Result from :meth:`OnChainClient.get_exchange_flow`, or ``None`` when
        on-chain intelligence is unavailable.

    Returns
    -------
    float
        0 (bearish on-chain) → 5 (bullish on-chain); 2.5 is neutral.
    """
    if onchain_data is None:
        return _NEUTRAL_SCORE
    return float(min(max(onchain_data.score, 0.0), _MAX_SCORE))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_quote_currency(symbol: str) -> str:
    """Strip common quote-currency suffixes to get the base coin name."""
    for suffix in ("USDT", "BUSD", "USDC"):
        if symbol.upper().endswith(suffix):
            return symbol[: -len(suffix)].upper()
    return symbol.upper()


def _parse_glassnode_latest(data: Any) -> Optional[float]:
    """Extract the most recent value from a Glassnode API response list."""
    if not isinstance(data, list) or len(data) == 0:
        return None
    entry = data[-1]
    if isinstance(entry, dict):
        v = entry.get("v")
        return float(v) if v is not None else None
    return None


def _net_flow_to_score(net_flow: float, inflow: float, outflow: float) -> float:
    """Map net exchange flow to a 0–5 confidence score.

    A large net outflow (coins leaving exchanges) is bullish → score near 5.
    A large net inflow (coins entering exchanges) is bearish → score near 0.
    Near-zero net flow is neutral → score near 2.5.
    """
    total = inflow + outflow
    if total <= 0:
        return _NEUTRAL_SCORE

    # Normalise net flow relative to total volume: range [-1, +1]
    normalised = max(-1.0, min(1.0, -net_flow / total))  # invert: outflow → positive
    # Map [-1, +1] → [0, 5]
    return round((normalised + 1.0) / 2.0 * _MAX_SCORE, 2)
