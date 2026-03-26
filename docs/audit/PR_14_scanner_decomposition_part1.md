# PR_14 — Scanner Decomposition (Part 1: Data Fetching & Indicators)

**PR Number:** PR_14  
**Branch:** `feature/pr14-scanner-decomposition-part1`  
**Category:** Risk & Reliability / Codebase Health (Phase 2A)  
**Priority:** P0 (unblocks PR_17 and PR_27)  
**Dependency:** None  
**Effort estimate:** Large (3–4 days)

---

## Objective

Extract data-fetching and indicator-computation responsibilities from the monolithic `scanner.py` (≈90KB) into a dedicated `src/scanner/` subpackage. After this PR, `scanner.py` retains only orchestration and signal dispatch; the new subpackage contains independently testable, focused modules. Part 2 (PR_27) will complete the decomposition by extracting the scan loop and dispatch logic.

---

## Current State

`scanner.py` is a single ≈90KB file that handles:
1. Kline and order-book data retrieval from Binance.
2. Indicator computation (EMA, ATR, MACD, RSI, etc.) per pair.
3. Signal dispatch (routing results to channels and Telegram).
4. Orchestration (pair iteration, scheduling).

This violates the single-responsibility principle and makes unit testing individual concerns difficult.

---

## Proposed Changes

### New directory structure

```
src/
  scanner/
    __init__.py            # Re-exports Scanner for backward compatibility
    data_fetcher.py        # Kline and order-book retrieval
    indicator_compute.py   # Indicator calculation per pair
  scanner.py               # Reduced; imports from src/scanner/
```

### `src/scanner/data_fetcher.py`

```python
"""Binance data retrieval helpers used by the scanner pipeline."""
from __future__ import annotations
import asyncio
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    """Fetches OHLCV klines and order-book snapshots for a list of symbols."""

    def __init__(self, binance_client):
        self._client = binance_client

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
    ) -> List[dict]:
        """Return a list of kline dicts for *symbol* at *interval*."""
        try:
            return await self._client.get_klines(symbol, interval, limit=limit)
        except Exception as exc:
            logger.warning("fetch_klines failed for %s/%s: %s", symbol, interval, exc)
            return []

    async def fetch_orderbook(
        self,
        symbol: str,
        depth: int = 20,
    ) -> Dict:
        """Return the current order book snapshot."""
        try:
            return await self._client.get_orderbook(symbol, limit=depth)
        except Exception as exc:
            logger.warning("fetch_orderbook failed for %s: %s", symbol, exc)
            return {"bids": [], "asks": []}

    async def fetch_all_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        limit: int = 200,
    ) -> Dict[str, List[dict]]:
        """Concurrently fetch klines for multiple timeframes."""
        tasks = {tf: self.fetch_klines(symbol, tf, limit) for tf in timeframes}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            tf: (r if not isinstance(r, Exception) else [])
            for tf, r in zip(tasks.keys(), results)
        }
```

### `src/scanner/indicator_compute.py`

```python
"""Per-pair indicator computation extracted from scanner.py."""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import logging

from src.indicators import ema, atr, rsi, macd, bollinger_bands, adx

logger = logging.getLogger(__name__)

def compute_indicators(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
) -> Dict:
    """Compute all indicators for a single pair/timeframe and return a flat dict."""
    if len(closes) < 50:
        return {}

    ema9  = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200) if len(closes) >= 200 else np.full_like(closes, np.nan)
    atr14 = atr(highs, lows, closes, 14)
    rsi14 = rsi(closes, 14)
    macd_line, signal_line, histogram = macd(closes)
    bb_upper, bb_mid, bb_lower = bollinger_bands(closes)
    adx14 = adx(highs, lows, closes, 14)

    return {
        "ema9": float(ema9[-1]),
        "ema21": float(ema21[-1]),
        "ema50": float(ema50[-1]),
        "ema200": float(ema200[-1]) if not np.isnan(ema200[-1]) else None,
        "atr": float(atr14[-1]),
        "rsi": float(rsi14[-1]),
        "macd_histogram_last": float(histogram[-1]),
        "macd_histogram_prev": float(histogram[-2]) if len(histogram) > 1 else 0.0,
        "bb_upper": float(bb_upper[-1]),
        "bb_mid": float(bb_mid[-1]),
        "bb_lower": float(bb_lower[-1]),
        "adx": float(adx14[-1]),
    }
```

### `src/scanner/__init__.py`

```python
"""Scanner subpackage — import the top-level Scanner for backward compatibility."""
from src.scanner_core import Scanner  # noqa: F401  (scanner_core = reduced scanner.py)
```

---

## Implementation Steps

1. Create `src/scanner/` directory with `__init__.py`, `data_fetcher.py`, and `indicator_compute.py`.
2. Extract all `get_klines` / `get_orderbook` call sites from `scanner.py` into `DataFetcher` methods.
3. Extract indicator computation blocks from `scanner.py` into `compute_indicators()` in `indicator_compute.py`.
4. In `scanner.py`, replace extracted code with calls to `DataFetcher` and `compute_indicators`.
5. Run full test suite to verify no regressions.
6. Measure file size reduction in `scanner.py` (target: ≤60KB after Part 1).

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/scanner/` | New directory |
| `src/scanner/__init__.py` | New — backward-compat re-export |
| `src/scanner/data_fetcher.py` | New — kline and order-book retrieval |
| `src/scanner/indicator_compute.py` | New — indicator computation |
| `src/scanner.py` | Reduced — remove extracted blocks; import from subpackage |

---

## Testing Requirements

```python
# tests/test_scanner_data_fetcher.py
async def test_fetch_klines_returns_empty_on_error():
    client = Mock(side_effect=Exception("timeout"))
    fetcher = DataFetcher(client)
    result = await fetcher.fetch_klines("BTCUSDT", "5m")
    assert result == []

async def test_fetch_all_timeframes_concurrent():
    client = AsyncMock(return_value=[{"o": 1}])
    fetcher = DataFetcher(client)
    result = await fetcher.fetch_all_timeframes("BTCUSDT", ["1m", "5m", "1h"])
    assert set(result.keys()) == {"1m", "5m", "1h"}

# tests/test_scanner_indicator_compute.py
def test_compute_indicators_returns_dict():
    closes = np.random.uniform(100, 200, 300)
    highs  = closes * 1.01
    lows   = closes * 0.99
    vols   = np.random.uniform(1e6, 1e7, 300)
    result = compute_indicators(closes, highs, lows, vols)
    assert "ema9" in result
    assert "rsi" in result
    assert 0 <= result["rsi"] <= 100

def test_compute_indicators_insufficient_data():
    closes = np.array([100.0, 101.0])
    result = compute_indicators(closes, closes, closes, closes)
    assert result == {}
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| `scanner.py` size | ~90KB | ~60KB (Part 1) |
| Data fetching testability | Only via integration test | Unit-testable in isolation |
| Indicator testability | Entangled with scan loop | Fully unit-testable |
| Time to locate data-fetching code | Search 90KB file | Open `data_fetcher.py` directly |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Import cycle between `scanner.py` and `src/scanner/` | Name the reduced scanner `scanner_core.py` during transition; `__init__.py` re-exports `Scanner` |
| Missing edge cases in extracted indicator code | Run existing test suite before and after extraction; diff output |
| Async context issues in `DataFetcher` | Test with `pytest-asyncio`; ensure event loop is properly shared |
