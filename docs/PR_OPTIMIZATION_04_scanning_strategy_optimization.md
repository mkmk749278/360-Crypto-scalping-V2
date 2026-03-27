# PR-OPT-04 — Scanning Strategy Optimization (API Efficiency + Throughput)

**Priority:** P2  
**Estimated Throughput Improvement:** 2–3× scanning throughput; sustainable 200+ pair coverage  
**Dependencies:** None (infrastructure change, does not depend on signal logic PRs)

---

## Objective

Optimise the Binance API scanning strategy to maximise signal coverage within rate limits. The current per-symbol REST fetch pattern consumes ~5 weight per symbol (1× depth, 3× kline TFs, 1× ticker). For 800 pairs this approaches 4,000 weight/cycle — dangerously close to the 5,000-weight spot limit. By leveraging aggregate endpoints and a WebSocket-first strategy, the same coverage can be achieved with ~350 weight/cycle.

---

## Analysis of Current Architecture

### Rate Budget

| Exchange | Limit | Budget (safe) |
|----------|-------|---------------|
| Binance Spot REST | 6,000 req/min | 5,000 |
| Binance Futures REST | 2,400 req/min | 2,000 |

### Per-Symbol Cost (current pattern)

| Endpoint | Weight | Usage |
|----------|--------|-------|
| `/fapi/v1/depth?symbol=X` | 2 | Per-symbol order book |
| `/fapi/v1/klines?symbol=X` (per TF) | 1 | 3 TFs per symbol = 3 |
| `/fapi/v1/ticker/bookTicker?symbol=X` | 1 | Per-symbol spread |
| **Total per symbol** | **~5–6** | For TIER1 full scan |

800 pairs × 6 = **4,800 weight/cycle** — virtually exhausting the budget and leaving no headroom for unexpected spikes.

### Scanner Configuration

```python
_MAX_CONCURRENT_SCANS = 10
_MAX_ORDER_BOOK_FETCHES_PER_CYCLE = 50
```

---

## Recommended Changes

### Change 1 — Aggregate Endpoint Migration

**File:** `src/binance.py` (and/or `src/scanner.py`)

Replace per-symbol calls with no-parameter aggregate endpoints:

```python
# BEFORE — per-symbol ticker (1 weight per symbol)
async def fetch_book_ticker(symbol: str) -> dict:
    return await self._get(f"/fapi/v1/ticker/bookTicker?symbol={symbol}")

# AFTER — all symbols in one call (2 weight TOTAL for all symbols)
async def fetch_all_book_tickers(self) -> List[dict]:
    """Fetch spread data for ALL futures symbols in one request (weight: 2)."""
    return await self._get("/fapi/v1/ticker/bookTicker")  # no symbol param

# BEFORE — per-symbol 24h stats (1 weight per symbol)
async def fetch_ticker_24hr(symbol: str) -> dict:
    return await self._get(f"/fapi/v1/ticker/24hr?symbol={symbol}")

# AFTER — all symbols in one call (weight: 40 for ALL vs. 1 per symbol)
async def fetch_all_tickers_24hr(self) -> List[dict]:
    """Fetch 24h volume and price stats for ALL symbols (weight: 40 total)."""
    return await self._get("/fapi/v1/ticker/24hr")  # no symbol param
```

**Weight savings from aggregate endpoints alone:**
- Book tickers: 800 × 1 = 800 weight → 2 weight (save 798/cycle)
- 24h tickers: 800 × 1 = 800 weight → 40 weight (save 760/cycle)
- Total saved: ~1,558 weight/cycle (~32% of the budget freed)

### Change 2 — Priority Queue Scanning Tiers

**File:** `src/scanner.py` / `src/tier_manager.py`

Replace the current simple TIER1/TIER2/TIER3 model with a 4-tier priority queue:

```python
# New scanning tier definitions
_SCAN_TIER_HOT_LIMIT:   int = 50    # Top 50 futures: full scan every 60s
_SCAN_TIER_WARM_LIMIT:  int = 100   # Futures 51–100: full scan every 120s
_SCAN_TIER_SPOT_LIMIT:  int = 100   # Top 100 spot: batch scan every 300s
# Remaining pairs: volume-surge detection only (no deep scan)

_SCAN_INTERVALS = {
    "HOT":  60,    # seconds
    "WARM": 120,
    "SPOT": 300,
    "COLD": None,  # event-driven only (volume spike detected via aggregate ticker)
}

_COLD_VOLUME_SURGE_MULTIPLIER: float = 3.0  # Promote to WARM if vol > 3× 24h avg
```

Tier assignment logic:

```python
def assign_scan_tier(symbol: str, market_type: str, rank_by_volume: int) -> str:
    if market_type == "futures":
        if rank_by_volume <= 50:
            return "HOT"
        if rank_by_volume <= 100:
            return "WARM"
    elif market_type == "spot":
        if rank_by_volume <= 100:
            return "SPOT"
    return "COLD"
```

### Change 3 — Dynamic Concurrency Based on Rate Limiter State

**File:** `src/scanner.py`

```python
# Before (static)
_MAX_CONCURRENT_SCANS = 10

# After (dynamic, respects remaining rate budget)
WEIGHT_PER_FULL_SCAN = 5  # depth(2) + 3 kline TFs(3) after aggregate pre-filter

async def _compute_max_concurrent(self) -> int:
    remaining = await self._rate_limiter.get_remaining_weight()
    # Reserve 20% of remaining budget for non-scan requests
    safe_remaining = int(remaining * 0.8)
    dynamic_limit = safe_remaining // WEIGHT_PER_FULL_SCAN
    return max(1, min(dynamic_limit, config.SCANNER_MAX_CONCURRENCY))
```

Add to `config/__init__.py`:

```python
SCANNER_MAX_CONCURRENCY: int = int(os.getenv("SCANNER_MAX_CONCURRENCY", "25"))
```

### Change 4 — WebSocket-First Strategy for HOT Tier

**File:** `src/websocket_manager.py` / `src/scanner.py`

```python
# HOT tier pairs use WS kline streams (zero REST weight)
# Fall back to REST only when WS connection is degraded

WS_DEGRADED_MAX_PAIRS: int = int(os.getenv("WS_DEGRADED_MAX_PAIRS", "100"))
# ↑ Increase from previous value; when WS is healthy, REST is not needed for top 100

async def _get_klines_for_hot_pair(self, symbol: str, tf: str) -> list:
    """Prefer cached WS klines, fall back to REST if WS buffer is stale."""
    ws_data = self._ws_manager.get_klines(symbol, tf)
    if ws_data and not ws_data.is_stale(max_age_seconds=120):
        return ws_data.candles
    # REST fallback — counted against rate budget
    return await self._binance.fetch_klines(symbol, tf)
```

### Change 5 — Pre-Filter Using Aggregate Data

**File:** `src/scanner.py`

Use aggregate ticker data to pre-filter pairs before expensive per-symbol depth/kline calls:

```python
async def _pre_filter_pairs(self, all_tickers: List[dict]) -> List[str]:
    """
    Use aggregate 24h ticker data to identify pairs worth deep scanning.
    Skips deep scan for pairs with:
    - Volume < 500K USD (insufficient liquidity for any channel)
    - Price change < 0.05% (dead market, no scalp opportunity)
    """
    eligible = []
    for ticker in all_tickers:
        volume = float(ticker.get("quoteVolume", 0))
        price_change_pct = abs(float(ticker.get("priceChangePercent", 0)))
        if volume >= 500_000 and price_change_pct >= 0.05:
            eligible.append(ticker["symbol"])
    return eligible
```

This pre-filter alone can cut the deep-scan list by 30–40% for stable/dead markets, saving additional weight.

---

## Scanning Architecture Summary (Before vs. After)

| Metric | Before | After |
|--------|--------|-------|
| Weight per 800-pair cycle | ~4,800 | ~780 |
| Max pairs within budget | ~800 (marginal) | 2,000+ |
| HOT tier scan latency | ~60s | ~60s (unchanged via WS) |
| COLD tier detection | None | Volume-surge auto-promote |
| Concurrency limit | Static 10 | Dynamic up to 25 |
| WS fallback threshold | Fixed | Configurable (env var) |

---

## Modules Affected

| Module | Change |
|--------|--------|
| `src/binance.py` | Add `fetch_all_book_tickers()`, `fetch_all_tickers_24hr()` aggregate methods |
| `src/scanner.py` | New 4-tier priority queue; dynamic concurrency; pre-filter logic |
| `src/scanner/__init__.py` | Update tier config constants |
| `src/rate_limiter.py` | Add `get_remaining_weight()` method if not present |
| `src/tier_manager.py` | Update tier assignment logic |
| `src/websocket_manager.py` | Increase `WS_DEGRADED_MAX_PAIRS`; add staleness check |
| `config/__init__.py` | Add `SCANNER_MAX_CONCURRENCY`, `WS_DEGRADED_MAX_PAIRS` env vars |

---

## Test Cases

1. **`test_aggregate_endpoint_weight`** — `fetch_all_book_tickers()` makes one request with weight 2, not N requests.
2. **`test_pre_filter_removes_dead_pairs`** — Pairs with <$500K volume are excluded from deep scan list.
3. **`test_dynamic_concurrency_respects_budget`** — When remaining weight = 100, max concurrent = min(16, 25) = 16.
4. **`test_hot_tier_uses_ws_klines`** — HOT tier pair with fresh WS buffer does not make klines REST call.
5. **`test_ws_stale_falls_back_to_rest`** — HOT tier pair with stale WS buffer (>120s) makes REST fallback.
6. **`test_cold_tier_promoted_on_volume_surge`** — COLD pair with 3.5× volume spike is promoted to WARM tier.
7. **`test_tier_assignment_futures_top50`** — Futures rank 1–50 → HOT tier.
8. **`test_tier_assignment_spot_top100`** — Spot rank 1–100 → SPOT tier.

---

## Rollback Procedure

1. Restore `_MAX_CONCURRENT_SCANS = 10` static config.
2. Remove aggregate endpoint methods from `binance.py` (or keep as additive).
3. Restore original TIER1/TIER2/TIER3 scan interval logic.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Aggregate endpoint returns stale data vs. per-symbol call | Low | Binance aggregate endpoints have same update frequency as per-symbol |
| Dynamic concurrency causes burst rate limit hit | Low | 20% headroom reservation + rate limiter guardrails |
| Pre-filter removes pairs that momentarily have low volume but fire a signal | Low | COLD tier volume-surge detection catches these within 60s |
| WS-first fallback logic adds code complexity | Medium | Comprehensive unit tests for WS/REST decision logic |

---

## Expected Impact

- **~85% reduction in REST weight consumption** for 800+ pair coverage
- **Sustainable 200+ pair deep scan** within futures rate budget (2,000/min)
- **HOT tier latency unchanged** — top 50 futures still scanned every 60s via WS
- **COLD tier pairs auto-promoted** on volume surges — no signals missed during quiet periods
