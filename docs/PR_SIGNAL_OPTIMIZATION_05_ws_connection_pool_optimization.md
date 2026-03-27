# PR-SIG-OPT-05 — WebSocket Connection Pool Optimization

**Priority:** P2 — Reduces REST fallback frequency; improves data continuity for futures pairs  
**Estimated Impact:** REST fallback duration -40%; critical pair data continuity 90% → 98%; scan latency -25% during WS degradation  
**Dependencies:** PR-SIG-OPT-04 (Tiered Scheduler) recommended for parallel deployment  
**Relates To:** Addresses root cause of frequent REST fallback triggering described in the scanner/trade_monitor logs  
**Status:** 📋 Planned

---

## Objective

Optimize the `WebSocketManager` in `src/websocket_manager.py` to reduce the frequency
and duration of REST fallback activation. The current implementation uses a flat
`WS_FALLBACK_BULK_LIMIT=200` for all timeframes, polls all critical pairs every 5
seconds uniformly, and reconnects all degraded connections simultaneously. This PR
introduces intelligent bulk-seed limits per timeframe, adaptive poll intervals based
on API response latency, staggered reconnection, and an increased critical pair count.

---

## Problem Analysis

### Issue 1: Flat `WS_FALLBACK_BULK_LIMIT=200` For All Timeframes

**File:** `config/__init__.py` — line 550 and `src/websocket_manager.py` — line 188

```python
# config/__init__.py — line 550
WS_FALLBACK_BULK_LIMIT: int = int(os.getenv("WS_FALLBACK_BULK_LIMIT", "200"))

# websocket_manager.py — line 188 (bulk seed call)
await self._data_store.fetch_and_store_fallback(
    symbol, interval=interval, limit=WS_FALLBACK_BULK_LIMIT, market=self._market
)
```

`WS_FALLBACK_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]` (config line 552).

Fetching 200 candles for the 4h timeframe seeds **800 hours** (33 days) of data — far
more than any indicator requires. The 1m timeframe needs at most 50 candles for ADX(14)
and Bollinger(20). This wastes significant rate-limit weight during fallback activation:

- 200 candles × 5 timeframes × N critical pairs × Binance weight-per-kline-request
- For 10 critical pairs: 10 symbols × 5 TFs × 1 weight = 50 weight per cycle, but
  each REST kline request costs up to 10 weight for limit>100.
- At WS_FALLBACK_BULK_LIMIT=200: `cost = ceil(200/100) * 5 weight = 10 weight` per
  symbol per timeframe. For 10 pairs × 5 TFs: **500 weight** consumed at fallback start.
  (Binance limit: 1200/min)

### Issue 2: Fixed 5-Second Poll Interval Regardless of API Health

```python
# websocket_manager.py — line ~230
await asyncio.sleep(5)
```

A fixed 5s poll applies regardless of whether Binance is responding quickly (should
poll faster) or slowly under load (should back off to avoid 429 errors). During high-
volatility events when WS drops are most likely, the fixed 5s interval either wastes
rate limit (when latency is <1s) or fails to keep up with fast price moves.

### Issue 3: Simultaneous Reconnection of All Degraded Connections

Looking at `_health_watchdog()` in `websocket_manager.py`, all degraded connections
attempt reconnection simultaneously. When a Binance WS outage affects all connections
(common during maintenance), this creates a reconnection storm — all connections race
to reconnect, amplifying the load on Binance's WS servers and triggering throttling.

### Issue 4: Critical Pair Count Limited to ~10 in `src/bootstrap.py`

The `auto_populate_critical_pairs()` call typically receives only the top 10 futures
pairs. With 150+ pairs being scanned, this leaves 90%+ of pairs without REST fallback
coverage.

---

## Required Changes

### Change 1 — Per-Timeframe Bulk Seed Limits

**File:** `config/__init__.py` — replace `WS_FALLBACK_BULK_LIMIT` with a dict

```python
# Before (line 550)
WS_FALLBACK_BULK_LIMIT: int = int(os.getenv("WS_FALLBACK_BULK_LIMIT", "200"))

# After — timeframe-aware bulk limits
WS_FALLBACK_BULK_LIMIT_BY_TF: Dict[str, int] = {
    "1m":  int(os.getenv("WS_FALLBACK_BULK_1M",  "50")),    # ADX(14) needs 28 candles
    "5m":  int(os.getenv("WS_FALLBACK_BULK_5M",  "50")),    # BB(20) needs 20 candles
    "15m": int(os.getenv("WS_FALLBACK_BULK_15M", "50")),    # EMA(21) needs 21 candles
    "1h":  int(os.getenv("WS_FALLBACK_BULK_1H",  "100")),   # Higher-TF context
    "4h":  int(os.getenv("WS_FALLBACK_BULK_4H",  "100")),   # SPOT/GEM channel primary TF
}
# Backward-compat shim: keep WS_FALLBACK_BULK_LIMIT as default for unknown TFs
WS_FALLBACK_BULK_LIMIT: int = int(os.getenv("WS_FALLBACK_BULK_LIMIT", "100"))
```

**File:** `src/websocket_manager.py` — update bulk seed call (line ~188)

```python
# Before
await self._data_store.fetch_and_store_fallback(
    symbol, interval=interval, limit=WS_FALLBACK_BULK_LIMIT, market=self._market
)

# After
from config import WS_FALLBACK_BULK_LIMIT_BY_TF, WS_FALLBACK_BULK_LIMIT
bulk_limit = WS_FALLBACK_BULK_LIMIT_BY_TF.get(interval, WS_FALLBACK_BULK_LIMIT)
await self._data_store.fetch_and_store_fallback(
    symbol, interval=interval, limit=bulk_limit, market=self._market
)
log.info(
    "REST fallback: bulk-seeded {} {} candles for {}",
    bulk_limit, interval, symbol,
)
```

**Rate limit savings:** With new per-TF limits:
- 10 critical pairs × (50+50+50+100+100) candles per pair = 10 × 350 = **3,500 candles**
- Old flat limit: 10 pairs × (200 × 5 TFs) = **10,000 candles**
- Reduction: (10,000 - 3,500) / 10,000 = **65% fewer candles fetched during bulk seed**

### Change 2 — Adaptive Poll Interval via `_adaptive_poll_interval()`

**File:** `src/websocket_manager.py` — add method to `WebSocketManager`

```python
async def _adaptive_poll_interval(
    self,
    base_interval: float = 2.0,
    max_interval: float = 10.0,
    latency_ema: float = 0.0,
) -> float:
    """Compute adaptive poll interval based on rolling API response latency.

    Starts at 2s, backs off exponentially if latency > 1s, recovers when
    latency returns to baseline. Capped at 10s to prevent stale data.

    Args:
        base_interval: Starting interval in seconds.
        max_interval: Maximum backoff interval in seconds.
        latency_ema: Exponential moving average of recent response times (seconds).

    Returns:
        Recommended sleep duration in seconds.
    """
    if latency_ema < 0.5:
        return base_interval           # Fast: poll at 2s
    elif latency_ema < 1.0:
        return base_interval * 1.5    # Slightly slow: 3s
    elif latency_ema < 2.0:
        return base_interval * 2.5    # Slow: 5s
    else:
        return min(max_interval, base_interval * 4.0)   # Very slow: back off to 8-10s
```

Update `_rest_fallback_loop()` to track latency and use adaptive interval:

```python
# Replace the fixed sleep at end of fallback loop:
# Before
await asyncio.sleep(5)

# After
_latency_ema: float = 0.0
_alpha: float = 0.2   # EMA smoothing factor

# At the start of each poll cycle:
_poll_start = time.monotonic()
# ... (existing poll logic) ...
_poll_elapsed = time.monotonic() - _poll_start
_latency_ema = _alpha * _poll_elapsed + (1 - _alpha) * _latency_ema

_sleep = await self._adaptive_poll_interval(
    base_interval=2.0, max_interval=10.0, latency_ema=_latency_ema
)
await asyncio.sleep(_sleep)
```

Also add the initial `await asyncio.sleep(0.2)` stagger between pairs from 0.2s to
`0.1 + 0.05 * pair_index` to spread requests over time.

### Change 3 — Staggered Reconnection for Degraded Connections

**File:** `src/websocket_manager.py` — in `_health_watchdog()` or the connection failure handler

```python
# Add WS_RECONNECT_STAGGER_MS to config/__init__.py
WS_RECONNECT_STAGGER_MS: int = int(os.getenv("WS_RECONNECT_STAGGER_MS", "500"))  # 500ms between reconnects

# In the reconnection logic:
degraded_connections = [c for c in self._connections if c.degraded]
for i, conn in enumerate(degraded_connections):
    # Stagger reconnections: conn 0 reconnects immediately, conn 1 after 500ms, etc.
    if i > 0:
        await asyncio.sleep(WS_RECONNECT_STAGGER_MS / 1000.0)
    # Existing reconnect logic...
    conn.task = asyncio.create_task(self._run_connection(conn))
```

### Change 4 — Increase Critical Pair Count

**File:** `src/bootstrap.py` — where `auto_populate_critical_pairs()` is called

```python
# Before (typical pattern)
ws_manager.auto_populate_critical_pairs(top_pairs[:10])

# After
from config import WS_CRITICAL_PAIR_COUNT
ws_manager.auto_populate_critical_pairs(top_pairs[:WS_CRITICAL_PAIR_COUNT])
```

**File:** `config/__init__.py` — add new constant

```python
#: Number of top-volume pairs to include in REST fallback critical pair set.
#: These pairs receive dedicated REST polling when WS feed degrades.
#: Higher values improve data continuity but consume more rate-limit budget.
WS_CRITICAL_PAIR_COUNT: int = int(os.getenv("WS_CRITICAL_PAIR_COUNT", "20"))
```

### Change 5 — Add WS Health Score to Telemetry

**File:** `src/websocket_manager.py` — add to `WebSocketManager`

```python
def get_health_score(self) -> Dict[str, Any]:
    """Return a health score dict for telemetry reporting.

    Returns
    -------
    dict with keys:
        score      – 0–100; 100 = all connections healthy
        degraded   – number of degraded connections
        total      – total connections
        fallback   – whether REST fallback is currently active
        uptime_pct – percentage of connections currently healthy
    """
    total = len(self._connections)
    if total == 0:
        return {"score": 0, "degraded": 0, "total": 0, "fallback": False, "uptime_pct": 0.0}
    degraded = sum(1 for c in self._connections if c.degraded)
    healthy = total - degraded
    uptime_pct = (healthy / total) * 100.0
    score = int(uptime_pct) - (20 if self._rest_fallback_active else 0)
    return {
        "score": max(0, score),
        "degraded": degraded,
        "total": total,
        "fallback": self._rest_fallback_active,
        "uptime_pct": uptime_pct,
    }
```

Call this in `src/bootstrap.py` periodic health reporting.

---

## Rate Limit Budget Analysis

### Before This PR (worst case: 10 critical pairs, 5 TFs, bulk_limit=200)
| Operation | Weight | Frequency | Budget/min |
|-----------|--------|-----------|------------|
| Bulk seed at startup | 200 candles × weight=10 per pair-TF | Once on fallback | 500 weight |
| REST polling | 1m+5m per pair, limit=1 | Every 5s × 10 pairs | 120 weight/min |
| Scan kline fetches | 5 TFs per pair | Every scan cycle | Variable |

### After This PR (20 critical pairs, TF-aware limits, adaptive 2-10s poll)
| Operation | Weight | Frequency | Budget/min |
|-----------|--------|-----------|------------|
| Bulk seed at startup | TF-specific (50-100) × weight=5-10 | Once on fallback | ~200 weight |
| REST polling | 1m+5m, adaptive 2-10s | 6-30 polls/min × 20 pairs | 40-240 weight/min |
| **Total worst case** | | | **440/1200 weight** = 37% utilized |

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| REST fallback bulk-seed weight cost | 500 weight | ~200 weight (-60%) |
| REST poll frequency | Fixed 5s | Adaptive 2–10s |
| Reconnection storm on outage | All simultaneous | Staggered 500ms apart |
| Critical pairs with REST coverage | ~10 | 20 |
| WS data continuity for critical pairs | ~90% | ~98% |
| Rate limit headroom | ~400/1200 (33%) free | ~760/1200 (63%) free |

---

## Testing Criteria

```bash
# Run targeted tests
python -m pytest tests/test_websocket_and_formatting.py -v

# Verify per-TF bulk limits
python -c "
from config import WS_FALLBACK_BULK_LIMIT_BY_TF
assert WS_FALLBACK_BULK_LIMIT_BY_TF['1m'] == 50, 'Expected 50'
assert WS_FALLBACK_BULK_LIMIT_BY_TF['4h'] == 100, 'Expected 100'
print('Per-TF bulk limits: PASS ✅')
"

# Verify WS_CRITICAL_PAIR_COUNT
python -c "
from config import WS_CRITICAL_PAIR_COUNT
assert WS_CRITICAL_PAIR_COUNT == 20, f'Expected 20, got {WS_CRITICAL_PAIR_COUNT}'
print(f'WS_CRITICAL_PAIR_COUNT: {WS_CRITICAL_PAIR_COUNT} ✅')
"

# Verify health score method exists
python -c "
from src.websocket_manager import WebSocketManager
ws = WebSocketManager(on_message=lambda m: None, market='spot')
score = ws.get_health_score()
assert 'score' in score and 'fallback' in score
print(f'Health score structure: {score} ✅')
"

# Env var override
WS_FALLBACK_BULK_4H=50 WS_CRITICAL_PAIR_COUNT=30 python -m pytest tests/test_websocket_and_formatting.py -v
```
