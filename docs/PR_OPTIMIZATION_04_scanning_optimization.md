# PR-OPT-04 — Scanning Optimization Strategy (Aggregate Endpoints)

**Priority:** P2  
**Estimated Impact:** ~70% reduction in Binance API weight consumption per scan cycle  
**Dependencies:** None  
**Status:** 📋 PLANNED (infrastructure change, not yet implemented)

---

## Objective

Replace per-symbol order book fetches with weight-efficient aggregate Binance endpoints.  The
current architecture fetches `/fapi/v1/depth?symbol=X` individually for each pair, costing 2–5
weight units per call.  Switching to aggregate endpoints (`/fapi/v1/ticker/bookTicker`) reduces
weight from ~4,840/cycle (800 pairs) to ~350/cycle for the same coverage.

---

## Problems Addressed

- REST rate limit pressure: scanning 800+ pairs with individual `/depth` calls exhausts the
  1,200 weight/minute Binance limit.
- When WS is degraded, the system already caps the scan set but this is reactive, not proactive.
- Tier 2 and Tier 3 pairs do not need high-precision order book depth for SWING/SPOT/GEM signals.

---

## Module / Strategy Affected

- `src/scanner/__init__.py` — `_fetch_global_book_tickers()` and spread cache logic
- `src/scanner.py` — same

---

## Recommended Changes (Not Yet Implemented)

### Aggregate endpoint usage

```python
# Instead of: GET /fapi/v1/depth?symbol=BTCUSDT  (weight: 2–10)
# Use:        GET /fapi/v1/ticker/bookTicker       (weight: 2 for ALL symbols)
await self._fetch_global_book_tickers(market="futures")
```

The `_fetch_global_book_tickers()` method already exists and is used when WS is degraded.
This PR would make it the default for Tier 2/3 pairs, reserving `/depth` for Tier 1 only.

### Tiered approach

| Tier | Endpoint | Weight | Frequency |
|------|----------|--------|-----------|
| TIER1-HOT (top 50) | `/fapi/v1/depth?limit=5` | 2/call | Every cycle |
| TIER1-WARM (51-100) | `/fapi/v1/depth?limit=5` | 2/call | Every 2 cycles |
| TIER2 (100-300) | `/fapi/v1/ticker/bookTicker` (global) | 2 total | Every 3 cycles |
| TIER3 (300+) | `/fapi/v1/ticker/bookTicker` (global) | 2 total | 30-min interval |

---

## Rollback Procedure

This change is additive — the existing per-symbol fetch path remains as fallback.
To rollback: stop calling the global bookTicker pre-fetch for Tier 2/3 pairs.
