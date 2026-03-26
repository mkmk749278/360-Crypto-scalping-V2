# PR_18 — Funding Rate Divergence Module

**PR Number:** PR_18  
**Branch:** `feature/pr18-funding-rate-divergence-module`  
**Category:** Signal Intelligence (Phase 2B)  
**Priority:** P1  
**Dependency:** PR_01, PR_02 (Phase 1 — Regime Detector and Per-Pair Config)  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Monitor Binance perpetual futures funding rates in real-time and use divergence between funding rate direction and price direction as a scoring modifier (±5 points). A positive funding rate with falling price suggests over-leveraged longs are being squeezed (bearish confirmation for SHORT signals). A negative funding rate with rising price suggests short-squeeze potential (bullish confirmation for LONG signals).

---

## Current State

No funding rate integration exists. The existing `src/oi_filter.py` polls open interest but not funding rate. `src/order_flow.py` tracks OI snapshots and CVD but not funding rate divergence. Binance provides funding rate data via the `/fapi/v1/fundingRate` REST endpoint, which is not currently consumed by any module.

---

## Proposed Changes

### New file: `src/funding_rate.py`

```python
"""Funding rate divergence monitor for Binance perpetual futures."""
from __future__ import annotations
import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

FUNDING_POLL_INTERVAL_SECONDS = 60 * 8   # Binance settles every 8h; poll every minute
FUNDING_DIVERGENCE_THRESHOLD   = 0.0001  # 0.01% — meaningful funding vs price divergence
FUNDING_SCORE_MODIFIER          = 5.0    # points added or subtracted

@dataclass
class FundingSnapshot:
    symbol: str
    rate: float          # raw funding rate (can be negative)
    timestamp: datetime

class FundingRateMonitor:
    """
    Polls Binance funding rates and detects divergence from price direction.
    Thread-safe via asyncio; designed to run as a background task.
    """

    def __init__(self, binance_client, symbols: list[str]):
        self._client = binance_client
        self._symbols = symbols
        self._latest: Dict[str, FundingSnapshot] = {}
        self._history: Dict[str, deque] = {s: deque(maxlen=10) for s in symbols}
        self._running = False

    async def start(self) -> None:
        self._running = True
        while self._running:
            await self._poll_all()
            await asyncio.sleep(FUNDING_POLL_INTERVAL_SECONDS)

    def stop(self) -> None:
        self._running = False

    async def _poll_all(self) -> None:
        for symbol in self._symbols:
            try:
                data = await self._client.get_funding_rate(symbol)
                snap = FundingSnapshot(
                    symbol=symbol,
                    rate=float(data.get("lastFundingRate", 0)),
                    timestamp=datetime.now(timezone.utc),
                )
                self._latest[symbol] = snap
                self._history[symbol].append(snap)
            except Exception as exc:
                logger.warning("Failed to fetch funding rate for %s: %s", symbol, exc)

    def get_score_modifier(self, symbol: str, price_direction: str) -> float:
        """
        Return a score modifier based on funding rate divergence.

        - price_direction: "LONG" or "SHORT"
        - Returns +5 if funding supports the signal direction.
        - Returns -5 if funding contradicts the signal direction.
        - Returns 0 if funding rate is within noise threshold or data unavailable.
        """
        snap = self._latest.get(symbol)
        if snap is None:
            return 0.0

        rate = snap.rate
        if abs(rate) < FUNDING_DIVERGENCE_THRESHOLD:
            return 0.0   # noise level — no meaningful signal

        if price_direction == "LONG" and rate < 0:
            # Price rising + negative funding = shorts getting squeezed → bullish confirmation
            return +FUNDING_SCORE_MODIFIER
        if price_direction == "SHORT" and rate > 0:
            # Price falling + positive funding = longs getting liquidated → bearish confirmation
            return +FUNDING_SCORE_MODIFIER
        if price_direction == "LONG" and rate > 0:
            # Positive funding = over-leveraged longs → bearish pressure → penalise LONG
            return -FUNDING_SCORE_MODIFIER
        if price_direction == "SHORT" and rate < 0:
            # Negative funding = shorts being squeezed → bullish pressure → penalise SHORT
            return -FUNDING_SCORE_MODIFIER

        return 0.0

    def latest_rate(self, symbol: str) -> Optional[float]:
        snap = self._latest.get(symbol)
        return snap.rate if snap else None
```

### Wire into `src/signal_quality.py`

```python
from src.funding_rate import FundingRateMonitor

# Module-level monitor (started in main.py alongside other background tasks):
_funding_monitor: Optional[FundingRateMonitor] = None

def apply_funding_rate_modifier(signal) -> None:
    """In-place: adjust signal.post_ai_confidence by funding rate divergence score."""
    if _funding_monitor is None:
        return
    modifier = _funding_monitor.get_score_modifier(signal.symbol, signal.direction)
    if modifier != 0.0:
        old = signal.post_ai_confidence
        signal.post_ai_confidence = max(0.0, min(100.0, old + modifier))
        logger.debug(
            "Funding rate modifier %+.1f applied to %s %s: %.1f→%.1f",
            modifier, signal.symbol, signal.direction, old, signal.post_ai_confidence,
        )
```

### Config additions in `src/config/__init__.py`

```python
FUNDING_RATE_ENABLED:          bool  = os.getenv("FUNDING_RATE_ENABLED", "true").lower() == "true"
FUNDING_DIVERGENCE_THRESHOLD:  float = float(os.getenv("FUNDING_DIVERGENCE_THRESHOLD", "0.0001"))
FUNDING_SCORE_MODIFIER:        float = float(os.getenv("FUNDING_SCORE_MODIFIER", "5.0"))
```

---

## Implementation Steps

1. Create `src/funding_rate.py` with `FundingRateMonitor` class.
2. Add `get_funding_rate(symbol)` method to `BinanceClient` in `src/binance.py` (calls `/fapi/v1/fundingRate`).
3. Instantiate `FundingRateMonitor` in `main.py` and start it as an async background task.
4. In `signal_quality.py`, add `apply_funding_rate_modifier()` and call it in the signal scoring pipeline.
5. Add config constants to `config/__init__.py`.
6. Write unit tests in `tests/test_funding_rate.py`.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/funding_rate.py` | New — `FundingRateMonitor` class |
| `src/binance.py` | Add `get_funding_rate()` method |
| `src/signal_quality.py` | Add `apply_funding_rate_modifier()` |
| `src/main.py` | Start `FundingRateMonitor` background task |
| `src/config/__init__.py` | Add funding rate config constants |
| `tests/test_funding_rate.py` | New test file |

---

## Testing Requirements

```python
# tests/test_funding_rate.py
def test_positive_funding_penalises_long():
    monitor = FundingRateMonitor(client=None, symbols=["BTCUSDT"])
    monitor._latest["BTCUSDT"] = FundingSnapshot("BTCUSDT", rate=0.0005,
                                                   timestamp=datetime.now(timezone.utc))
    modifier = monitor.get_score_modifier("BTCUSDT", "LONG")
    assert modifier == -5.0

def test_negative_funding_supports_long():
    monitor = FundingRateMonitor(client=None, symbols=["BTCUSDT"])
    monitor._latest["BTCUSDT"] = FundingSnapshot("BTCUSDT", rate=-0.0003,
                                                   timestamp=datetime.now(timezone.utc))
    modifier = monitor.get_score_modifier("BTCUSDT", "LONG")
    assert modifier == +5.0

def test_noise_level_returns_zero():
    monitor = FundingRateMonitor(client=None, symbols=["BTCUSDT"])
    monitor._latest["BTCUSDT"] = FundingSnapshot("BTCUSDT", rate=0.00005,
                                                   timestamp=datetime.now(timezone.utc))
    modifier = monitor.get_score_modifier("BTCUSDT", "LONG")
    assert modifier == 0.0

def test_missing_data_returns_zero():
    monitor = FundingRateMonitor(client=None, symbols=["BTCUSDT"])
    modifier = monitor.get_score_modifier("ETHUSDT", "SHORT")
    assert modifier == 0.0
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Funding rate awareness | None | Live polling every 60s |
| Score accuracy on over-leveraged market | No adjustment | ±5 pts based on funding divergence |
| Short-squeeze detection | None | Negative funding + LONG = +5 boost |
| Signals against funding pressure | No penalty | −5 pts warning |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Binance REST rate limits | Funding rate changes only every 8h; poll at most every 60s well within limits |
| Spot pairs have no funding rate | Check if symbol is a perp (`USDT` perp) before calling; return 0 for spot |
| Stale data if poller crashes | Use timestamp check; if data >15min old, return 0 modifier |
| API key not set for futures | Funding rate endpoint is public; no API key required |
