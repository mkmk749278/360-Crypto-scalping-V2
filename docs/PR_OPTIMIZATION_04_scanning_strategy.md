# PR-OPT-04 — Optimized Binance API Scanning Strategy

**Priority:** P2  
**Estimated Impact:** Tier 1 scalp signals within 15–30s (from 60–90s); no rate limit exhaustion  
**Dependencies:** None  
**Status:** 📋 PLANNED

---

## Objective

Redesign the pair scanning architecture to prioritise time-critical Tier 1 scalp signals and
prevent Tier 2 pair scanning from stalling due to rate limit exhaustion. The current sequential
scan model treats all tiers uniformly, causing 30–60 second delays on scalp signals when Tier 2
processing consumes API budget.

---

## Problem

### Current Architecture

The scanner in `src/scanner/__init__.py` uses a single `scan_loop()` coroutine (line 476) that:

1. Processes all Tier 1 pairs every cycle
2. Processes Tier 2 pairs every `TIER2_SCAN_EVERY_N_CYCLES` cycles (default: 3)
3. Drops Tier 2 scans entirely when rate limit budget exceeds 85% (`skip_tier2_for_latency`)
4. Runs Tier 3 lightweight scan every `TIER3_SCAN_INTERVAL_MINUTES` minutes (default: 30)

```python
# src/scanner/__init__.py — line 550
scan_tier2 = (self._scan_cycle_count % TIER2_SCAN_EVERY_N_CYCLES == 0)
skip_tier2_for_latency = (
    self._rate_limiter.usage_pct() > 0.85
)
if skip_tier2_for_latency:
    # ... Tier 2 completely dropped
```

**Problems with this approach:**

1. All Tier 1 pairs (futures + spot) are processed sequentially within a single `asyncio.gather`
   call, creating contention between scalp-critical futures pairs and lower-priority spot pairs.
2. When rate limit hits 85%, Tier 2 is completely dropped rather than throttled — discovery
   of emerging opportunities is lost for an entire scan cycle.
3. Scalp signals (time-sensitive) share API budget with non-time-critical SWING/SPOT scans.
4. No dedicated WebSocket stream for Tier 1 real-time order book updates — REST OB fetches
   add unnecessary latency for scalp channel evaluation.
5. The current per-symbol REST pattern consumes weight proportionally — scanning 800 pairs
   individually instead of using aggregate endpoints wastes significant API budget.

---

## Solution — Async Multi-Tier Scanning with Staggered Scheduling

### New Architecture: `TieredScanScheduler`

```python
# New scanning architecture — src/scanner/__init__.py
class TieredScanScheduler:
    """Manages separate scan loops for each tier with independent timing."""

    async def run_tier1_loop(self):
        """Top 50 futures: scan every 15–30 seconds via WebSocket + cached OB."""
        while True:
            symbols = self.pair_mgr.tier1_futures_symbols[:50]
            await asyncio.gather(*[self._scan_symbol(s) for s in symbols])
            await asyncio.sleep(15)

    async def run_tier2_loop(self):
        """Next 100 pairs: scan every 2–5 minutes via REST."""
        while True:
            symbols = self.pair_mgr.tier2_symbols
            # Batch in groups of 20 to respect rate limits
            for batch in chunked(symbols, 20):
                await asyncio.gather(*[self._scan_symbol(s) for s in batch])
                await asyncio.sleep(3)  # 3s between batches
            await asyncio.sleep(120)  # 2 min between full cycles

    async def run_tier3_hourly_loop(self):
        """Universe pairs: lightweight scan every hour."""
        while True:
            symbols = self.pair_mgr.tier3_symbols
            for batch in chunked(symbols, 10):
                await asyncio.gather(*[self._scan_symbol_lightweight(s) for s in batch])
                await asyncio.sleep(5)
            await asyncio.sleep(3600)
```

### Rate Limit Budget Allocation

Split the 1200 weight/min Binance API budget across tiers:

```python
# config/__init__.py
TIER1_RATE_BUDGET_PCT: float = float(os.getenv("TIER1_RATE_BUDGET_PCT", "0.60"))
# 720 weight/min dedicated to top pairs

TIER2_RATE_BUDGET_PCT: float = float(os.getenv("TIER2_RATE_BUDGET_PCT", "0.30"))
# 360 weight/min for discovery

TIER3_RATE_BUDGET_PCT: float = float(os.getenv("TIER3_RATE_BUDGET_PCT", "0.10"))
# 120 weight/min for universe
```

---

## Solution — Aggregate Endpoint Optimization

Replace per-symbol REST calls with Binance aggregate endpoints to dramatically reduce
weight consumption:

| Current Pattern | Weight | Optimized Pattern | Weight | Saving |
|----------------|--------|------------------|--------|--------|
| `/fapi/v1/ticker/24hr?symbol=X` per symbol | 1/symbol | `/fapi/v1/ticker/24hr` (all) | 40 total | ~96% |
| `/fapi/v1/depth?symbol=X` per symbol | 2/symbol | Cached WS book for Tier 1 | 0 | 100% |
| `/fapi/v1/klines?symbol=X` | 1/symbol | `/fapi/v1/klines` batched | 1/symbol | 0% |

For 800 symbols, the aggregate ticker saves: `800 × 1 = 800 weight` → `40 weight`.

---

## Solution — WebSocket Optimization for Tier 1

- Use combined stream `!miniTicker@arr` for all futures pairs in a single WebSocket connection
- Dedicated kline streams only for Tier 1 futures (`<symbol>@kline_1m`, `<symbol>@kline_5m`)
- Maintain a real-time order book cache for Tier 1 via `<symbol>@depth@100ms` streams
- REST-only for Tier 2/3 (no WebSocket overhead)

```python
# src/websocket_manager.py — proposed addition
TIER1_WS_STREAMS = [
    "!miniTicker@arr",                    # All pairs price/volume ticker
    "{symbol}@kline_1m",                  # 1m klines for each T1 symbol
    "{symbol}@kline_5m",                  # 5m klines for each T1 symbol
    "{symbol}@depth@100ms",               # Real-time OB for T1 scalp
]
```

---

## Changes Needed

### `config/__init__.py`

```python
# Tiered scan scheduling
TIER1_SCAN_INTERVAL_SECONDS: int = int(os.getenv("TIER1_SCAN_INTERVAL_SECONDS", "15"))
TIER2_SCAN_INTERVAL_SECONDS: int = int(os.getenv("TIER2_SCAN_INTERVAL_SECONDS", "120"))
TIER3_SCAN_INTERVAL_SECONDS: int = int(os.getenv("TIER3_SCAN_INTERVAL_SECONDS", "3600"))
TIER1_RATE_BUDGET_PCT: float = float(os.getenv("TIER1_RATE_BUDGET_PCT", "0.60"))
TIER2_RATE_BUDGET_PCT: float = float(os.getenv("TIER2_RATE_BUDGET_PCT", "0.30"))
TIER3_RATE_BUDGET_PCT: float = float(os.getenv("TIER3_RATE_BUDGET_PCT", "0.10"))
```

### `src/scanner/__init__.py`

- Refactor `scan_loop()` (line 476) into three independent async coroutines
- Each coroutine checks its own rate budget slice before processing
- Replace `skip_tier2_for_latency` hard drop with Tier 2 throttling (extend sleep interval)
- Add `_last_tier2_cycle_time` and `_last_tier3_cycle_time` tracking

### `src/scanner.py`

- Same changes to keep both scanner paths in sync with `src/scanner/__init__.py`

### `src/pair_manager.py`

- `tier1_futures_symbols` property (line 136) — already returns top futures symbols
- Add `tier1_futures_symbols_hot` property for top 50 (most active) futures

### `src/rate_limiter.py`

- Add `budget_for_tier(tier: int) -> float` method that returns remaining budget for a tier
  based on the configured tier budget percentages

### `src/websocket_manager.py`

- Add Tier 1 combined stream subscription using `!miniTicker@arr`
- Maintain a per-symbol real-time price cache for O(1) Tier 1 price lookups

---

## Scanning Tiers — Summary

| Tier | Pairs | Scan Interval | Channels | Data Source |
|------|-------|--------------|----------|-------------|
| TIER1-HOT (top 50 futures) | 50 | 15s | All channels | WS klines + WS OB |
| TIER1-WARM (futures 51–100) | 50 | 30s | All channels | WS klines + REST OB |
| TIER2 (spot top 100) | 100 | 2 min | SWING + SPOT + GEM | REST batch |
| TIER3 (remaining universe) | 400+ | 60 min | Lightweight volume | Aggregate ticker |

---

## Expected Impact

- Tier 1 scalp signals: generated within 15–30s of setup forming (currently 60–90s)
- Tier 2 signals: within 2–5 min (no longer completely dropped at 85% rate limit)
- API weight savings: ~70% reduction from aggregate endpoint usage
- No rate limit exhaustion for standard scanning workloads

---

## Rollback Plan

All scanning parameters are env-var configurable. To revert to current behaviour:

```bash
TIER1_SCAN_INTERVAL_SECONDS=60
TIER2_SCAN_EVERY_N_CYCLES=3
TIER3_SCAN_INTERVAL_MINUTES=30
```

Setting `TIER1_SCAN_INTERVAL_SECONDS=60` effectively restores the current scan cadence.

---

## Modules Affected

- `src/scanner/__init__.py` — scan loop refactoring
- `src/scanner.py` — scan loop refactoring (keep in sync)
- `src/pair_manager.py` — `tier1_futures_symbols_hot` property
- `src/rate_limiter.py` — `budget_for_tier()` method
- `src/websocket_manager.py` — combined stream subscription for Tier 1
- `config/__init__.py` — new tier scan interval and budget config vars
