# PR_27 — Scanner Decomposition (Part 2: Signal Dispatch & Orchestration)

**PR Number:** PR_27  
**Branch:** `feature/pr27-scanner-decomposition-part2`  
**Category:** Codebase Health (Phase 2E)  
**Priority:** P0 (unblocks PR_28 and PR_29)  
**Dependency:** PR_14 (Scanner Decomposition Part 1)  
**Effort estimate:** Large (3–4 days)

---

## Objective

Complete the scanner decomposition started in PR_14 by extracting signal dispatch and scan orchestration into dedicated modules. After this PR, the original `scanner.py` becomes a thin wrapper of less than 100 lines. No single file in the `src/scanner/` subpackage exceeds 25KB.

---

## Current State

After PR_14, `scanner.py` has been reduced from ≈90KB to ≈60KB by extracting data fetching and indicator computation. It still contains:
1. **Signal dispatch** — creating `Signal` objects and routing them to Telegram channels.
2. **Scan orchestration** — the main loop that iterates over all pairs, manages scheduling, and coordinates async tasks.

These two remaining responsibilities are large enough to warrant their own focused modules.

---

## Proposed Changes

### New file: `src/scanner/signal_dispatch.py`

```python
"""Signal creation and routing handoff — extracted from scanner.py."""
from __future__ import annotations
import logging
from typing import Optional

from src.channels.base import Signal

logger = logging.getLogger(__name__)

class SignalDispatcher:
    """
    Constructs Signal objects from channel evaluation results
    and hands them off to the signal router.
    """

    def __init__(self, signal_router, performance_tracker, anomaly_monitor=None):
        self._router = signal_router
        self._tracker = performance_tracker
        self._anomaly_monitor = anomaly_monitor

    async def dispatch(self, signal: Signal) -> bool:
        """
        Validate, score, and route a signal.

        Returns True if the signal was successfully dispatched.
        """
        if signal is None:
            return False

        # Gate: minimum score check
        if signal.post_ai_confidence < signal.channel_config.min_confidence:
            logger.debug(
                "Signal suppressed — score %.1f < threshold %.1f",
                signal.post_ai_confidence,
                signal.channel_config.min_confidence,
            )
            return False

        # Notify anomaly monitor
        if self._anomaly_monitor:
            self._anomaly_monitor.notify_signal(signal.channel)

        # Route via signal router (handles circuit breaker, correlation filter, etc.)
        dispatched = await self._router.route(signal)
        if dispatched:
            self._tracker.record_signal_sent(signal)

        return dispatched

    async def dispatch_batch(self, signals: list) -> int:
        """Dispatch a batch of signals; return count of successfully dispatched."""
        count = 0
        for sig in signals:
            if await self.dispatch(sig):
                count += 1
        return count
```

### New file: `src/scanner/orchestrator.py`

```python
"""Main scan loop — pair iteration and scheduling."""
from __future__ import annotations
import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class ScanOrchestrator:
    """
    Coordinates the main scanning loop:
    1. Iterates over all configured pairs.
    2. Fetches data via DataFetcher.
    3. Computes indicators via compute_indicators().
    4. Passes results to each channel for evaluation.
    5. Dispatches signals via SignalDispatcher.
    """

    def __init__(
        self,
        pairs: List[str],
        channels: list,
        data_fetcher,
        signal_dispatcher,
        scan_interval_seconds: int = 60,
    ):
        self._pairs = pairs
        self._channels = channels
        self._fetcher = data_fetcher
        self._dispatcher = signal_dispatcher
        self._interval = scan_interval_seconds
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("ScanOrchestrator started — %d pairs, %d channels",
                    len(self._pairs), len(self._channels))
        while self._running:
            await self._scan_all()
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        self._running = False

    async def _scan_all(self) -> None:
        tasks = [self._scan_pair(pair) for pair in self._pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for pair, result in zip(self._pairs, results):
            if isinstance(result, Exception):
                logger.warning("Scan failed for %s: %s", pair, result)

    async def _scan_pair(self, symbol: str) -> None:
        from src.scanner.data_fetcher import DataFetcher
        from src.scanner.indicator_compute import compute_indicators
        import numpy as np

        MIN_CANDLES_FOR_EVALUATION = 50  # WHY: most indicators (EMA200, ADX14, RSI14) need ≥50 bars

        # Fetch multi-timeframe data
        candles = await self._fetcher.fetch_all_timeframes(
            symbol, timeframes=["1m", "5m", "1h", "4h", "1d"]
        )

        # Evaluate each channel
        signals = []
        for channel in self._channels:
            tf_data = candles.get(channel.primary_timeframe, [])
            if len(tf_data) < MIN_CANDLES_FOR_EVALUATION:
                continue
            closes  = np.array([c["close"] for c in tf_data])
            highs   = np.array([c["high"]  for c in tf_data])
            lows    = np.array([c["low"]   for c in tf_data])
            volumes = np.array([c["volume"] for c in tf_data])
            indicators = compute_indicators(closes, highs, lows, volumes)
            signal = await channel.evaluate(symbol, candles, indicators)
            if signal:
                signals.append(signal)

        await self._dispatcher.dispatch_batch(signals)
```

### Reduce `src/scanner.py` to thin wrapper

After extraction, `scanner.py` becomes:

```python
"""
Scanner entry point — thin wrapper over ScanOrchestrator.
Import Scanner from this module for backward compatibility.
"""
from src.scanner.orchestrator import ScanOrchestrator as Scanner

__all__ = ["Scanner"]
```

---

## Implementation Steps

1. Create `src/scanner/signal_dispatch.py` with `SignalDispatcher`.
2. Create `src/scanner/orchestrator.py` with `ScanOrchestrator`.
3. Move remaining signal dispatch logic from `scanner.py` into `SignalDispatcher.dispatch()`.
4. Move the main scan loop from `scanner.py` into `ScanOrchestrator._scan_all()` and `_scan_pair()`.
5. Reduce `scanner.py` to a thin re-export wrapper.
6. Run full test suite to confirm no regressions.
7. Verify no single file in `src/scanner/` exceeds 25KB.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/scanner/signal_dispatch.py` | New — `SignalDispatcher` class |
| `src/scanner/orchestrator.py` | New — `ScanOrchestrator` class |
| `src/scanner.py` | Reduced to thin re-export wrapper (<100 lines) |

---

## Testing Requirements

```python
# tests/test_scanner_orchestrator.py
async def test_orchestrator_skips_short_data():
    fetcher = AsyncMock(return_value={"5m": [{"close": 100, "high": 101, "low": 99,
                                              "volume": 1000}] * 10})  # only 10 candles < MIN_CANDLES_FOR_EVALUATION
    channel = AsyncMock()
    channel.primary_timeframe = "5m"
    channel.evaluate = AsyncMock(return_value=None)
    dispatcher = AsyncMock()
    dispatcher.dispatch_batch = AsyncMock(return_value=0)
    orch = ScanOrchestrator(["BTCUSDT"], [channel], fetcher, dispatcher)
    await orch._scan_pair("BTCUSDT")
    channel.evaluate.assert_not_called()  # skipped due to insufficient data

async def test_dispatcher_routes_valid_signal():
    router = AsyncMock()
    router.route = AsyncMock(return_value=True)
    tracker = Mock()
    tracker.record_signal_sent = Mock()
    dispatcher = SignalDispatcher(router, tracker)
    signal = Mock()
    signal.post_ai_confidence = 75.0
    signal.channel_config.min_confidence = 60.0
    signal.channel = "SCALP"
    result = await dispatcher.dispatch(signal)
    assert result is True
    router.route.assert_called_once()

async def test_dispatcher_suppresses_low_score():
    router = AsyncMock()
    dispatcher = SignalDispatcher(router, Mock())
    signal = Mock()
    signal.post_ai_confidence = 45.0
    signal.channel_config.min_confidence = 60.0
    result = await dispatcher.dispatch(signal)
    assert result is False
    router.route.assert_not_called()
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| `scanner.py` size | ~60KB (post PR_14) | <5KB (thin wrapper) |
| Scan orchestration testability | Integrated into monolith | Unit-testable `ScanOrchestrator` |
| Signal dispatch testability | Integrated into monolith | Unit-testable `SignalDispatcher` |
| Largest file in `src/scanner/` | N/A | <25KB |
| Code navigation time | Search 60KB file | Open targeted module |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Breaking changes to `Scanner` import in external code | `scanner.py` re-exports `ScanOrchestrator as Scanner` |
| Async context issues during extraction | Test with `pytest-asyncio`; run full integration suite |
| Missing edge cases in orchestration logic | Diff old and new scan paths; add regression tests |
| Circular imports between submodules | Use absolute imports throughout; no cross-module imports within `src/scanner/` |
