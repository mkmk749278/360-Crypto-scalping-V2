# PR-SIG-OPT-04 — Dynamic Tiered Scan Scheduler with Futures Priority

**Priority:** P2 — Reduces scan latency; critical for futures pairs that need frequent re-evaluation  
**Estimated Impact:** Top-50 futures scan latency -65%; REST fallback frequency -60%; 150+ pairs handled within API limits  
**Dependencies:** PR-SIG-OPT-05 (WS pool optimization) should be deployed concurrently  
**Relates To:** Extends PR-OPT-04 (Scanning Strategy Optimization) — adds dynamic scheduling and `TieredScanScheduler` class  
**Status:** 📋 Planned

---

## Objective

Implement a `TieredScanScheduler` class in `src/scanner/__init__.py` that separates
the 150+ pairs into three priority queues with different scan cadences. Futures pairs
get dedicated hot-queue treatment (30s scan cycle), spot pairs are batched into warm
and cold queues, and concurrent scanning via `asyncio.Semaphore` prevents rate-limit
exhaustion.

---

## Problem Analysis

### Current State: `src/scanner/__init__.py` — Lines 494–640

The existing `scan_loop()` uses a simple counter-based tiering system:

```python
# Lines 571–572
scan_tier2 = (self._scan_cycle_count % TIER2_SCAN_EVERY_N_CYCLES == 0)
scan_tier3 = (self._scan_cycle_count % TIER3_SCAN_EVERY_N_CYCLES == 0)
```

And concurrent scanning via semaphore:

```python
# Line 432
self._scan_semaphore: asyncio.Semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SCANS)
# Line 161
_MAX_CONCURRENT_SCANS: int = 15
```

**Problems:**

1. **Counter-based tiers are time-blind**: If a scan cycle takes 45s (due to API
   latency), Tier 1 futures pairs are only rescanned every 45s instead of the
   intended ~30s. Counter-based logic doesn't account for elapsed wall-clock time.

2. **Futures pairs not prioritized within Tier 1**: `scan_loop()` iterates pairs in
   dictionary order. Futures and spot pairs are interleaved without priority.

3. **Config values**: `TIER2_SCAN_EVERY_N_CYCLES=3` and `TIER3_SCAN_EVERY_N_CYCLES=6`
   (from `config/__init__.py` lines 174–180) mean Tier 2 scans every 3rd cycle and
   Tier 3 every 6th. With a cycle time of 30–45s, this puts Tier 2 at ~90–135s and
   Tier 3 at ~180–270s — which is reasonable, but variable.

4. **`_MAX_CONCURRENT_SCANS = 15`** (line 161): This limits concurrent pair evaluations
   to 15, which is sensible. However, when 150 pairs share the same semaphore, hot-path
   futures pairs queue behind slow cold-path altcoin evaluations.

5. **REST fallback consumption**: When the WS feed degrades, `_rest_fallback_loop()` in
   `src/websocket_manager.py` bulk-fetches klines for critical pairs. This consumes rate
   limit budget, slowing down scan cycles for non-critical pairs.

### `config/__init__.py` — Relevant Constants

```python
TOP_PAIRS_COUNT: int = 150        # Total pairs to scan (line 157)
TIER2_SCAN_EVERY_N_CYCLES: int = 3  # Tier 2 scanned every 3 cycles (line 174)
TIER3_SCAN_EVERY_N_CYCLES: int = 6  # Tier 3 every 6 cycles (line 180)
```

### `src/pair_manager.py` — Existing Tier Classification

```python
# PairTier enum — lines 74–90
class PairTier(str, Enum):
    TIER1 = "TIER1"   # Top ~75 pairs by volume
    TIER2 = "TIER2"   # Next ~50 pairs
    TIER3 = "TIER3"   # Remaining pairs
```

The `PairManager` already classifies pairs into tiers (lines 380–384). The missing
piece is a **time-based scheduler** that uses these tiers to control scan frequency,
with futures pairs elevated within each tier.

---

## Required Changes

### Change 1 — Add `TieredScanScheduler` class to `src/scanner/__init__.py`

Add after the existing constants block (after line ~161):

```python
import heapq
from src.pair_manager import PairTier

@dataclass
class _ScanItem:
    """Priority queue item for the tiered scan scheduler."""
    next_scan_time: float          # Absolute epoch seconds for next scan
    priority: int                  # Lower = higher priority (0=futures-hot, 1=spot-warm, 2=cold)
    symbol: str
    market: str                    # "futures" or "spot"

    def __lt__(self, other: "_ScanItem") -> bool:
        # Primary sort: next_scan_time; secondary: priority (futures over spot)
        if self.next_scan_time != other.next_scan_time:
            return self.next_scan_time < other.next_scan_time
        return self.priority < other.priority


class TieredScanScheduler:
    """Time-based tiered scan scheduler.

    Manages a priority queue of pairs grouped by scan tier:
      - Tier 0 (Hot):  Top-50 futures pairs → target 30s scan interval
      - Tier 1 (Warm): Top-50 spot + next-50 futures → target 60s interval
      - Tier 2 (Cold): Remaining pairs → target 180s interval

    Uses wall-clock time rather than cycle counters to ensure consistent
    scan cadences regardless of individual cycle duration.
    """

    TIER_INTERVALS: Dict[int, float] = {
        0: float(os.getenv("SCAN_TIER1_INTERVAL", "30")),    # Hot: 30s
        1: float(os.getenv("SCAN_TIER2_INTERVAL", "60")),    # Warm: 60s
        2: float(os.getenv("SCAN_TIER3_INTERVAL", "180")),   # Cold: 180s
    }

    def __init__(self) -> None:
        self._heap: List[_ScanItem] = []
        self._symbol_map: Dict[str, _ScanItem] = {}

    def populate(self, pair_mgr: "PairManager") -> None:
        """Populate the scheduler from PairManager tier data."""
        now = time.monotonic()
        for symbol, pair_info in pair_mgr.pairs.items():
            tier_priority = self._compute_priority(pair_info)
            item = _ScanItem(
                next_scan_time=now,          # All pairs eligible immediately at start
                priority=tier_priority,
                symbol=symbol,
                market=pair_info.market,
            )
            heapq.heappush(self._heap, item)
            self._symbol_map[symbol] = item

    def _compute_priority(self, pair_info: Any) -> int:
        """Assign queue priority: futures Tier1=0, spot Tier1/futures Tier2=1, rest=2."""
        is_futures = getattr(pair_info, "market", "spot") == "futures"
        tier = getattr(pair_info, "tier", PairTier.TIER3)
        if is_futures and tier == PairTier.TIER1:
            return 0   # Hot: top-50 futures
        if tier in (PairTier.TIER1, PairTier.TIER2):
            return 1   # Warm: top-100 spot + Tier2 futures
        return 2       # Cold: remaining

    def get_due_symbols(self, max_count: int = 50) -> List[str]:
        """Return up to max_count symbols whose scan time has elapsed."""
        now = time.monotonic()
        due = []
        deferred = []
        while self._heap and len(due) < max_count:
            item = heapq.heappop(self._heap)
            if item.next_scan_time <= now:
                due.append(item.symbol)
                # Do NOT reschedule here — caller must call mark_scanned() after scan
            else:
                # Not yet due — defer and stop (heap is ordered, all remaining are later)
                deferred.append(item)
                break
        # Drain any further not-yet-due items that remain after the first deferred one
        while self._heap:
            item = heapq.heappop(self._heap)
            deferred.append(item)
        # Restore all deferred items back into the heap
        for item in deferred:
            heapq.heappush(self._heap, item)
        return due

    def mark_scanned(self, symbol: str) -> None:
        """Update next scan time for a symbol after it has been scanned."""
        item = self._symbol_map.get(symbol)
        if item is None:
            return
        interval = self.TIER_INTERVALS.get(item.priority, 60.0)
        new_item = _ScanItem(
            next_scan_time=time.monotonic() + interval,
            priority=item.priority,
            symbol=symbol,
            market=item.market,
        )
        self._symbol_map[symbol] = new_item
        heapq.heappush(self._heap, new_item)
```

### Change 2 — Integrate `TieredScanScheduler` into `scan_loop()`

**File:** `src/scanner/__init__.py` — `Scanner.__init__()` and `scan_loop()`

In `__init__()` (line ~432), add:

```python
self._tiered_scheduler: TieredScanScheduler = TieredScanScheduler()
```

In `scan_loop()` (line 494), replace the counter-based tier logic with scheduler-driven batches:

```python
# Before (simplified)
scan_tier2 = (self._scan_cycle_count % TIER2_SCAN_EVERY_N_CYCLES == 0)
scan_tier3 = (self._scan_cycle_count % TIER3_SCAN_EVERY_N_CYCLES == 0)
# ... iterate all pairs with tier filtering ...

# After
if not self._tiered_scheduler._symbol_map:
    self._tiered_scheduler.populate(self.pair_mgr)

# Get up to 50 due symbols per scan pass
due_symbols = self._tiered_scheduler.get_due_symbols(max_count=50)

# Scan due symbols with concurrency limit
tasks = [
    self._scan_symbol_with_semaphore(symbol)
    for symbol in due_symbols
]
await asyncio.gather(*tasks, return_exceptions=True)

# Mark all scanned
for sym in due_symbols:
    self._tiered_scheduler.mark_scanned(sym)
```

### Change 3 — Add separate semaphores for hot vs cold queues

**File:** `src/scanner/__init__.py` — in `Scanner.__init__()`

```python
# Before
self._scan_semaphore: asyncio.Semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SCANS)

# After — separate limits for hot and cold to prevent cold pairs starving hot ones
_HOT_CONCURRENCY: int = int(os.getenv("SCAN_HOT_CONCURRENCY", "10"))
_COLD_CONCURRENCY: int = int(os.getenv("SCAN_COLD_CONCURRENCY", "5"))

self._hot_semaphore: asyncio.Semaphore = asyncio.Semaphore(_HOT_CONCURRENCY)
self._cold_semaphore: asyncio.Semaphore = asyncio.Semaphore(_COLD_CONCURRENCY)
```

### Change 4 — Add `get_tiered_scan_groups()` to `src/pair_manager.py`

**File:** `src/pair_manager.py` — add after `get_tier3()` method (line ~129)

```python
def get_tiered_scan_groups(self) -> Dict[str, List[str]]:
    """Return pairs grouped by scan priority.

    Returns
    -------
    dict with keys:
        "hot"  — Top-50 futures pairs (scan every 30s)
        "warm" — Top-50 spot + remaining futures Tier1/Tier2 (scan every 60s)
        "cold" — Remaining pairs (scan every 180s)
    """
    hot, warm, cold = [], [], []
    for symbol, pair_info in self.pairs.items():
        is_futures = getattr(pair_info, "market", "spot") == "futures"
        tier = pair_info.tier
        if is_futures and tier == PairTier.TIER1:
            hot.append(symbol)
        elif tier in (PairTier.TIER1, PairTier.TIER2):
            warm.append(symbol)
        else:
            cold.append(symbol)
    return {"hot": hot[:50], "warm": warm[:100], "cold": cold}
```

### Change 5 — Add new env vars to `config/__init__.py`

```python
# Tiered scan scheduler intervals (seconds)
SCAN_TIER1_INTERVAL: int = int(os.getenv("SCAN_TIER1_INTERVAL", "30"))     # Hot: futures top-50
SCAN_TIER2_INTERVAL: int = int(os.getenv("SCAN_TIER2_INTERVAL", "60"))     # Warm: spot+futures Tier2
SCAN_TIER3_INTERVAL: int = int(os.getenv("SCAN_TIER3_INTERVAL", "180"))    # Cold: remaining pairs
SCAN_HOT_CONCURRENCY: int = int(os.getenv("SCAN_HOT_CONCURRENCY", "10"))   # Concurrent hot scans
SCAN_COLD_CONCURRENCY: int = int(os.getenv("SCAN_COLD_CONCURRENCY", "5"))  # Concurrent cold scans
```

### Change 6 — Add scan cycle metrics

Add a `_scan_metrics` dict to the `Scanner` class for observability:

```python
self._scan_metrics: Dict[str, Any] = {
    "hot_scanned": 0,
    "warm_scanned": 0,
    "cold_scanned": 0,
    "hot_scan_duration_ms": 0.0,
    "signals_generated_hot": 0,
    "signals_generated_warm": 0,
    "signals_generated_cold": 0,
    "last_cycle_ts": 0.0,
}
```

Log metrics at INFO level at the end of each scan cycle:

```python
log.info(
    "Scan cycle complete: hot={} warm={} cold={} | signals hot={} warm={} cold={} | duration={:.0f}ms",
    metrics["hot_scanned"], metrics["warm_scanned"], metrics["cold_scanned"],
    metrics["signals_generated_hot"], metrics["signals_generated_warm"],
    metrics["signals_generated_cold"],
    metrics["hot_scan_duration_ms"],
)
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Top-50 futures scan interval | 30–45s (variable) | 30s (guaranteed) |
| Scan latency for top-50 pairs | ~45s | ~15s (10 concurrent) |
| REST fallback triggers | Frequent | -60% (faster scan = less WS gap) |
| Rate limit headroom | Tight at >150 pairs | Comfortable (batched by cadence) |
| Cold pair scan interval | ~90–270s (counter-based) | 180s (consistent) |

---

## Testing Criteria

```bash
# Run targeted tests
python -m pytest tests/test_tiered_pairs.py -v
python -m pytest tests/test_tier_manager.py -v

# Verify scheduler assigns correct priorities
python -c "
from src.scanner import TieredScanScheduler
from src.pair_manager import PairManager
# Create mock PairManager with futures Tier1 and spot Tier1 pairs
# Assert futures Tier1 gets priority=0, spot Tier1 gets priority=1
print('Priority assignment: PASS')
"

# Verify env var overrides
SCAN_TIER1_INTERVAL=15 python -m pytest tests/ -k "test_tiered" -v
```
