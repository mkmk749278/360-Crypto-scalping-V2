# Audit Report: Scaling to ~800 Pairs — Bottlenecks & Implementation Plan

**Repository:** `360-Crypto-scalping-V2`  
**Date:** 2026-03-27  
**Status:** Step 1 of N — Architectural Review (no code changes in this PR)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Observed Symptoms](#2-observed-symptoms)
3. [Feasibility Assessment — Why 800 Concurrent Pairs Fail](#3-feasibility-assessment--why-800-concurrent-pairs-fail)
   - 3.1 [Binance REST Weight Exhaustion](#31-binance-rest-weight-exhaustion)
   - 3.2 [WebSocket Buffer Overflow & Stream-Count Limits](#32-websocket-buffer-overflow--stream-count-limits)
   - 3.3 [Scan Latency Cascade](#33-scan-latency-cascade)
4. [Recommended Optimizations](#4-recommended-optimizations)
   - 4.1 [Dynamic Tiering (Hot / Warm / Cold)](#41-dynamic-tiering-hot--warm--cold)
   - 4.2 [Global Aggregate Endpoints](#42-global-aggregate-endpoints)
   - 4.3 [Concurrency & Async Optimizations](#43-concurrency--async-optimizations)
5. [Risk & Signal Integrity](#5-risk--signal-integrity)
6. [Actionable Implementation Plan](#6-actionable-implementation-plan)
   - [Step 1 — Fix WS Architecture](#step-1--fix-ws-architecture)
   - [Step 2 — Market Watchdog for Dynamic Tiering](#step-2--market-watchdog-for-dynamic-tiering)
   - [Step 3 — Rewrite Scanner REST Fallback](#step-3--rewrite-scanner-rest-fallback)
   - [Step 4 — Intelligent Rate Limiting](#step-4--intelligent-rate-limiting)
7. [Existing Partial Mitigations](#7-existing-partial-mitigations)
8. [Success Metrics](#8-success-metrics)

---

## 1. Executive Summary

The scalping engine currently attempts to scan approximately 800 USDT perpetual / spot pairs per cycle.
At this scale, the Binance REST API weight budget is exhausted within a single scan cycle, WebSocket
connections degrade under stream-count pressure, and the resulting REST fallback latency reaches 30–50 s
per cycle — rendering real-time scalping signals unreliable.

This document provides:

- A root-cause analysis of the resource exhaustion.
- A tiered-scanning architecture that keeps the engine within safe API budget at all times.
- A concrete, sequenced 4-step implementation plan.

> **Scope:** This PR is documentation-only. Code changes are tracked in subsequent PRs.

---

## 2. Observed Symptoms

| Symptom | Log Evidence | Root Cause |
|---------|-------------|------------|
| REST 429 / 418 hard lock-out | `rate limit exhausted — sleeping 42 s` | Weight window fully consumed |
| WS degradation | `WS=300(ok=False)` | Stream count exceeds per-connection limit; stale pong |
| Slow scan cycles | REST fallback latencies 30–50 s | Looping individual `/depth` calls for hundreds of pairs |
| Missed signals | Tier 1 pairs queued behind Tier 3 REST calls | No priority differentiation across 800 pairs |
| Cascading timeouts | `DEPTH_CIRCUIT_BREAKER` trips repeatedly | Single-point depth endpoint overload |

---

## 3. Feasibility Assessment — Why 800 Concurrent Pairs Fail

### 3.1 Binance REST Weight Exhaustion

Binance enforces a rolling **60-second weight window**. The limits relevant to this engine are:

| Market | Hard Limit | Engine Budget (current) | Headroom |
|--------|-----------|------------------------|----------|
| Spot   | 6,000 / min | 5,000 / min (`_DEFAULT_BUDGET`) | ~1,000 |
| Futures | 2,400 / min | 2,000 / min (`_DEFAULT_FUTURES_BUDGET`) | ~400 |

#### Weight cost per scan cycle with 800 pairs

| Call type | Binance weight | Calls per cycle | Total weight |
|-----------|---------------|-----------------|--------------|
| `/fapi/v1/depth` (limit=5) | 2 | 800 | **1,600** |
| `/fapi/v1/klines` (limit=100) | 1 | 800 × 3 TFs | **2,400** |
| `/fapi/v1/ticker/24hr` (single symbol) | 1 | 800 | **800** |
| `/fapi/v1/exchangeInfo` | 40 | 1 (periodic) | 40 |
| **Total per cycle** | | | **≈ 4,840** |

The Futures budget of **2,000/min** is exhausted more than twice over in a single scan cycle.
Even ignoring kline fetches, depth calls alone (1,600 weight) exceed the Futures soft cap.

The engine's existing burst-protection mechanism (`_BURST_PROTECTION_THRESHOLD = 0.15`) injects
micro-sleeps once remaining budget drops below 15%, but with 800 pairs the budget is gone before
the protection threshold can meaningfully slow the flood.

### 3.2 WebSocket Buffer Overflow & Stream-Count Limits

Binance imposes the following WebSocket constraints:

| Constraint | Binance Limit | Engine Config (`WS_MAX_STREAMS_PER_CONN`) |
|-----------|--------------|------------------------------------------|
| Streams per combined-stream URL | **200** | **50** (currently conservative) |
| Messages per second per connection | **~50 msg/s** | Not enforced in code |
| Connections per IP | No published hard limit, but >10 triggers review | Scales with pair count |

With 800 pairs and typically 2–3 streams per pair (kline, depth, force-order), the engine
subscribes **1,600–2,400 streams**, requiring **32–48 connections** (at 50 streams/conn) or
**8–12 connections** (at the Binance limit of 200 streams/conn).

**Buffer overflow path:**

1. Each connection receives a combined message stream at peak ~30–80 msg/s during volatile markets.
2. The `aiohttp` WebSocket receive buffer backs up when the event loop is blocked processing earlier messages.
3. Missed pong responses cause the engine to mark the connection `degraded`.
4. All degraded connections trigger REST fallback for all 800 pairs — the worst-case scenario.

**Current result:** `WS=300(ok=False)` indicates 300 streams active but health check failing,
consistent with pong-staleness triggered by message-processing backlog.

### 3.3 Scan Latency Cascade

When WS degrades, the REST fallback loop iterates over all registered critical pairs **sequentially**:

```
for symbol in critical_pairs:          # up to 800 symbols
    for interval in timeframes:        # 3 timeframes
        GET /fapi/v1/klines?symbol=X&interval=Y&limit=200
```

Each request consumes ~0.2–0.4 s under normal load. At 800 pairs × 3 timeframes = 2,400 requests,
the fallback loop takes **480–960 seconds** — an entire order of magnitude longer than a 30 s
scan cycle. Even with the `WS_DEGRADED_MAX_PAIRS = 50` guard, the engine still issues 150
kline requests sequentially, contributing 30–60 s of latency.

---

## 4. Recommended Optimizations

### 4.1 Dynamic Tiering (Hot / Warm / Cold)

The `PairTier` enum and `PairManager` in `src/pair_manager.py` already implement a three-tier
structure (TIER1 / TIER2 / TIER3). The recommended enhancement is to make tier assignment
**dynamic** — driven by real-time volume, volatility, and funding-rate signals rather than
static rank thresholds.

#### Proposed tier definitions

| Tier | Label | Pair count target | Scan frequency | Data source |
|------|-------|------------------|----------------|-------------|
| Tier 1 | **Hot** | 30–50 | Every cycle (≤ 30 s) | Full WS + REST |
| Tier 2 | **Warm** | 100–150 | Every N cycles (2–5 min) | WS kline only; REST on signal |
| Tier 3 | **Cold** | Remainder (~600+) | Lazy — aggregate poll only | `/fapi/v1/ticker/bookTicker` batch |

#### Promotion / demotion triggers

```
Tier 3 → Tier 2:  24h volume spike ≥ 2× 7-day median  OR  |funding rate| ≥ 0.05%
Tier 2 → Tier 1:  RSI divergence + kline momentum threshold met  OR  breakout detected
Tier 1 → Tier 2:  Volume normalises for 3 consecutive cycles
Tier 2 → Tier 3:  No signal activity for N cycles (configurable TTL)
```

This bounds the active full-scan set to ≤ 200 pairs, keeping weight consumption well within
budget:

| Tier 1 (50 pairs) | weight/cycle |
|-------------------|-------------|
| `/depth` × 50     | 100         |
| `/klines` × 50×3  | 150         |
| Aggregate ticker  | 2 (entire market, single call) |
| **Total**         | **≈ 252**   |

This is **~5%** of the current 4,840 weight/cycle estimate for 800 pairs.

### 4.2 Global Aggregate Endpoints

Instead of fetching ticker data per-symbol for Cold pairs, a single batch call returns data for
the entire market:

#### `/fapi/v1/ticker/bookTicker` (no symbol parameter)

- Returns best bid/ask for **all** Futures symbols in a single response.
- **Weight: 2** (regardless of symbol count).
- Suitable for real-time spread monitoring of Tier 3 pairs.
- Response payload: ~120 KB for 300 symbols — negligible bandwidth.

#### `/fapi/v1/ticker/24hr` (no symbol parameter)

- Returns 24h price change, volume, high/low for **all** symbols.
- **Weight: 40** (full market) vs. **1 per symbol** (individual calls).
- Break-even at 40 symbols — beyond that the batch call is strictly cheaper.
- Use to rank all pairs by `quoteVolume` for tier promotion decisions.

#### Recommended integration

```python
# Once per watchdog cycle (e.g., every 60 s):
book_tickers = await client.get("/fapi/v1/ticker/bookTicker")   # weight: 2
tickers_24h  = await client.get("/fapi/v1/ticker/24hr")         # weight: 40
# Total: 42 weight per watchdog cycle vs. 800 weight for individual calls
```

This single pair of calls replaces ~800 individual ticker fetches and provides sufficient
information to run tier promotion/demotion logic for all Cold pairs.

### 4.3 Concurrency & Async Optimizations

#### Semaphore tuning

The scanner uses `self._scan_semaphore` to bound concurrent pair scans. The semaphore limit
should be aligned with the available REST budget:

```
max_concurrent = min(
    config.SCANNER_CONCURRENCY,                    # current env var
    rate_limiter.remaining // WEIGHT_PER_SCAN,     # budget-aware cap
)
```

This prevents a burst of concurrent scans from simultaneously exhausting the budget.

#### `asyncio.gather` with weight pre-allocation

Before issuing a batch of kline or depth requests, pre-check `rate_limiter.remaining` and
subdivide the batch into chunks that fit within the budget window. This avoids the current
pattern where all tasks are started simultaneously and the rate limiter serialises them
one-by-one after the fact.

#### Deduplication of kline subscriptions

With 800 pairs × 3 timeframes = 2,400 streams, many stream handlers process the same kline
interval redundantly. Merging to a single handler per (symbol, interval) pair and dispatching
internally reduces WS message fan-out and event-loop overhead.

---

## 5. Risk & Signal Integrity

The main risk of tiering is **missing breakouts on Cold pairs** — a Tier 3 symbol surges but
the engine is not watching it closely enough to generate a timely scalp signal.

### Mitigations

| Risk | Mitigation |
|------|------------|
| **Cold pair breakout missed** | Watchdog polls aggregate `bookTicker` every 10–15 s; sudden spread tightening + volume spike triggers immediate Tier 2 promotion |
| **Promotion lag** | Tier 2 pre-loads 200 candles via bulk kline fetch on promotion (existing `_rest_fallback_loop` backfill pattern) so indicators are warm immediately |
| **False promotion thrash** | Demotion requires 3 consecutive calm cycles (hysteresis) to prevent pairs from bouncing between tiers |
| **WS connection loss hides Cold pair activity** | Aggregate REST endpoints are independent of WS; the watchdog always runs via REST, providing a floor of visibility |
| **High-volume altcoin not in Tier 1** | Tier 1 capacity is not fixed at 30 pairs — it is volume-ranked; any pair achieving sufficient volume automatically displaces a lower-volume Tier 1 pair |
| **Funding rate traps (perpetuals)** | Tier 3 → Tier 2 promotion on `|funding| ≥ 0.05%` ensures abnormally-funded pairs get signal coverage before the funding event resolves |

### Signal integrity contract

- **Tier 1** pairs: full signal fidelity — all channels, all timeframes, order-book depth, WS + REST.
- **Tier 2** pairs: SWING + SPOT channels, 1h/4h timeframes only; SCALP channel suppressed (existing
  scanner logic already enforces this for Tier 2).
- **Tier 3** pairs: no active channel scanning; only promotion eligibility checks via aggregate data.

---

## 6. Actionable Implementation Plan

The following steps are ordered by impact and independence. Each step is a standalone PR.

---

### Step 1 — Fix WS Architecture

**Goal:** Cap streams per connection at a safe value and prevent pong-stale degradation from
propagating to REST fallback for the entire pair universe.

**Files:** `src/websocket_manager.py`, `config/__init__.py`

**Changes:**

1. **Raise `WS_MAX_STREAMS_PER_CONN`** from 50 to **150** (Binance allows 200; leaving 50 as
   headroom for dynamic subscriptions).

   ```python
   # config/__init__.py
   WS_MAX_STREAMS_PER_CONN: int = int(os.getenv("WS_MAX_STREAMS_PER_CONN", "150"))
   ```

2. **Add per-connection message-rate limiter** inside `_run_connection` to drop stale messages
   rather than blocking the receive loop, preventing pong-timeout from message backlog.

3. **Scope REST fallback to Tier 1 only.** When WS degrades, `_critical_pairs` should be
   automatically restricted to `tier1_symbols` (≤ 50 pairs) rather than the full pair universe.

   ```python
   # websocket_manager.py — _sync_rest_fallback_state
   if all_degraded and not self._critical_pairs:
       # Only fall back on Tier 1 pairs, not all 800
       self._critical_pairs = set(tier1_symbols[:50])
   ```

4. **Log connection health per-bucket** (`streams_per_conn`, `msg_rate`, `pong_latency`) to
   make future performance regressions immediately visible.

**Expected outcome:** WS stability improves; REST fallback load drops from 800 pairs to ≤ 50
pairs during outages.

---

### Step 2 — Market Watchdog for Dynamic Tiering

**Goal:** Implement a background coroutine that polls aggregate endpoints every 60 s and
promotes/demotes pairs between tiers based on live volume and funding rate signals.

**Files:** `src/pair_manager.py`, `src/main.py` (or equivalent engine entry point)

**Changes:**

1. **Add `MarketWatchdog` class** to `src/pair_manager.py`:

   ```python
   class MarketWatchdog:
       """Polls /fapi/v1/ticker/24hr and /fapi/v1/ticker/bookTicker every
       ``poll_interval_s`` seconds and adjusts PairTier assignments."""

       POLL_INTERVAL_S: int = 60
       VOLUME_SURGE_MULTIPLIER: float = 2.0      # vs. 7-day median
       FUNDING_RATE_THRESHOLD: float = 0.0005    # 0.05%
       DEMOTION_CALM_CYCLES: int = 3

       async def run(self) -> None: ...
       async def _evaluate_promotions(self, tickers: list[dict]) -> None: ...
       async def _evaluate_demotions(self) -> None: ...
   ```

2. **Persist tier state** across watchdog cycles using an in-memory `Dict[str, int]`
   (`calm_cycle_count`) to enforce hysteresis on demotion.

3. **Integrate with `PairManager.refresh_pairs()`** so that watchdog promotions are respected
   on the next scan cycle without waiting for the full pair refresh interval.

4. **Emit structured log lines** on every promotion/demotion for observability:

   ```
   INFO  MarketWatchdog  XRPUSDT TIER3→TIER2 reason=volume_surge vol_ratio=2.41
   INFO  MarketWatchdog  SOLUSDT TIER1→TIER2 reason=calm_cycles=3
   ```

**Expected outcome:** Active full-scan set dynamically adjusts to market conditions; Tier 1
never exceeds 50 pairs; Tier 3 pairs are watched cheaply via aggregate endpoints.

---

### Step 3 — Rewrite Scanner REST Fallback

**Goal:** Replace the per-symbol depth/ticker loop in the scanner with aggregate endpoint calls.

**Files:** `src/scanner.py` (and/or `src/scanner/__init__.py`), `src/binance.py`

**Changes:**

1. **Add `BinanceClient.get_all_book_tickers()`** that calls `/fapi/v1/ticker/bookTicker`
   (no `symbol` param) and returns a `Dict[str, dict]` keyed by symbol. Weight: 2.

2. **Add `BinanceClient.get_all_24h_tickers()`** that calls `/fapi/v1/ticker/24hr`
   (no `symbol` param) and returns a `Dict[str, dict]` keyed by symbol. Weight: 40.

3. **Replace the per-symbol loop** in `scan_loop` for Tier 2/3 pre-filter with a single
   aggregate call cached for the TTL of the scan cycle:

   ```python
   # Before (800 individual calls — weight: 800)
   for sym in pairs:
       ticker = await client.get_ticker_24hr(sym)

   # After (1 batch call — weight: 40)
   all_tickers = await client.get_all_24h_tickers()  # cached per cycle
   ticker = all_tickers.get(sym, {})
   ```

4. **Deprecate `_get_spread_pct` per-symbol order-book fetch** for Tier 3 pairs; use
   `bookTicker` bid/ask spread from the aggregate cache instead.

5. **Update `_MAX_ORDER_BOOK_FETCHES_PER_CYCLE`** guard: depth fetches (weight=2 each) should
   only be issued for Tier 1 pairs where order-book depth is critical for signal quality.

**Expected outcome:** REST weight per cycle drops from ~4,840 to ~250–300 for a typical 800-pair
universe; scan cycle completes in < 5 s instead of 30–50 s.

---

### Step 4 — Intelligent Rate Limiting

**Goal:** Make the rate limiter respond to the **server-authoritative** weight counter in every
API response rather than relying solely on local estimates.

**Files:** `src/rate_limiter.py`, `src/binance.py`

**Changes:**

1. **Parse `x-mbx-used-weight-1m` on every response** (not just on failure paths). The existing
   `update_from_header` method is already implemented; the gap is that `BinanceClient` only calls
   it in some code paths. Audit all `_request` / `_get` / `_post` methods to ensure every response
   triggers `rate_limiter.update_from_header(resp.headers.get("x-mbx-used-weight-1m"))`.

2. **Add a futures-specific header path.** Binance Futures uses `x-mbx-used-weight-1m` as well,
   but the engine uses a separate `futures_rate_limiter` singleton. Ensure the Futures client
   updates `futures_rate_limiter`, not `rate_limiter`.

3. **Implement adaptive budget scaling.** When the server-reported weight exceeds 80% of the
   hard cap, automatically reduce `_scan_semaphore` concurrency to half:

   ```python
   # In scan_loop, after receiving server weight:
   if futures_rate_limiter.used > 0.80 * FUTURES_HARD_CAP:
       sem = asyncio.Semaphore(max(1, self._scan_concurrency // 2))
   else:
       sem = self._scan_semaphore
   ```

4. **Expose a `/metrics` or log-line summary** of rate-limiter state every cycle:

   ```
   INFO  rate_limiter  spot=342/5000 (6.8%)  futures=187/2000 (9.4%)
   ```

   This makes budget headroom observable without inspecting log files manually.

5. **Add a `retry-after` header parser** for 429/418 responses: extract the `Retry-After`
   value and pass it to `rate_limiter.wait_until(timestamp)` rather than sleeping a fixed 42 s.

**Expected outcome:** Rate limiter tracks reality within ±5% at all times; adaptive concurrency
prevents hard 429 lockouts; retry delays are minimised by using the server-provided backoff value.

---

## 7. Existing Partial Mitigations

The following mitigations are already present in the codebase and should be **preserved**
(not replaced) by the above changes:

| Mitigation | Location | Notes |
|-----------|----------|-------|
| `WS_DEGRADED_MAX_PAIRS = 50` | `config/__init__.py` | Caps REST scan during WS degradation — Step 3 makes this redundant for Tier 3 but keeps it as a safety net |
| `WS_PARTIAL_HEALTH_THRESHOLD = 0.5` | `config/__init__.py` | Triggers degraded mode at 50% WS health — keep as-is |
| `_BURST_PROTECTION_THRESHOLD = 0.15` | `src/rate_limiter.py` | Micro-sleep when budget < 15% — keep, Step 4 complements it |
| `DEPTH_CIRCUIT_BREAKER_THRESHOLD = 5` | `config/__init__.py` | Trips after 5 consecutive depth timeouts — keep |
| `_MAX_ORDER_BOOK_FETCHES_PER_CYCLE = 50` | `src/scanner/__init__.py` | Hard cap on order-book fetches — Step 3 reduces demand to ≤ 50 Tier 1 pairs naturally |
| `update_from_header` | `src/rate_limiter.py` | Reads `x-mbx-used-weight-1m` — Step 4 ensures universal coverage |
| `auto_populate_critical_pairs` | `src/websocket_manager.py` | Extracts symbols from kline streams for REST fallback — Step 1 restricts to Tier 1 |

---

## 8. Success Metrics

After all four steps are implemented, the engine should achieve:

| Metric | Current (estimated) | Target |
|--------|--------------------|----|
| Futures weight / cycle | ~4,840 | ≤ 300 |
| Spot weight / cycle | ~3,200 | ≤ 250 |
| Scan cycle duration (normal) | 30–50 s | ≤ 5 s |
| Scan cycle duration (WS degraded) | 30–60 s | ≤ 10 s |
| Active WS streams | ~1,600–2,400 | ≤ 300 (Tier 1+2 only) |
| WS connections | 32–48 | ≤ 3 |
| Hard 429 lockouts / day | multiple | 0 |
| Tier 3 breakout detection lag | >5 min (missed entirely) | ≤ 90 s (watchdog cycle) |

---

*This document was authored as part of a systematic architectural review. Implementation PRs
will reference this document and track completion of each numbered step.*
