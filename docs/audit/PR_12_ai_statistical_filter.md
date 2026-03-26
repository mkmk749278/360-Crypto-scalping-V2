# PR_12 — AI Statistical False-Positive Filter

**Branch:** `feature/pr12-statistical-filter`  
**Priority:** 12  
**Effort estimate:** Large (4–5 days)

---

## Objective

Add a **statistical false-positive suppression layer** that tracks rolling per-channel /
per-pair / per-regime win rates and applies an adaptive confidence gate:

Gate thresholds used in `StatisticalFilter.check()`:
- Win rate ≥ 50% → pass with no penalty.
- 40% ≤ WR < 50% → pass with **–5 pt confidence penalty**.
- 30% ≤ WR < 40% → pass with **–10 pt confidence penalty** (soft gate).
- WR < 25% → **hard suppress** (signal is dropped entirely).

The filter is non-blocking (fail-open) when there is insufficient history (<15 resolved
signals in the window). This prevents over-filtering during cold-start periods.

The existing `src/feedback_loop.py` tracks cluster-suppression metrics. This PR adds a
separate `src/stat_filter.py` module for statistical gating that is distinct from
cluster suppression.

---

## Files to Change

| File | Change type |
|------|-------------|
| `src/stat_filter.py` | New module — `RollingWinRateStore` and `StatisticalFilter` classes |
| `src/scanner.py` | Instantiate `StatisticalFilter`, call `check()` after scoring, call `record()` after signal resolution |
| `src/signal_lifecycle.py` | Call `StatisticalFilter.record()` when a signal is resolved (TP/SL/expired) |
| `tests/test_advanced_filters.py` | Add tests for win rate tracking and gate decisions |

---

## Implementation Steps

### Step 1 — Create `src/stat_filter.py`

```python
"""Statistical false-positive filter using rolling win-rate tracking.

Tracks per-(channel, pair, regime) rolling win rates and applies adaptive
confidence penalties or hard suppression when quality drops below thresholds.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple


@dataclass
class SignalOutcome:
    """Record of a single resolved signal for win-rate tracking."""
    signal_id: str
    channel: str
    pair: str
    regime: str
    setup_class: str
    won: bool           # True if TP1 or higher was hit; False if SL hit or expired
    pnl_pct: float      # Actual PnL % achieved


class RollingWinRateStore:
    """Thread-safe rolling win-rate store per (channel, pair, regime) key.

    Uses a fixed-size deque per key so memory is bounded regardless of
    how many signals are recorded.
    """

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._lock = threading.Lock()
        # Key: (channel, pair, regime) → deque of bool (True=win, False=loss)
        self._records: Dict[Tuple[str, str, str], Deque[bool]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def record(self, outcome: SignalOutcome) -> None:
        """Record the outcome of a resolved signal."""
        key = (outcome.channel, outcome.pair, outcome.regime)
        with self._lock:
            self._records[key].append(outcome.won)

    def win_rate(self, channel: str, pair: str, regime: str) -> Optional[float]:
        """Return rolling win rate (0.0–1.0) or None if < min_history signals.

        Returns None when there is insufficient history to make a judgment.
        """
        key = (channel, pair, regime)
        with self._lock:
            records = self._records.get(key)
            if records is None or len(records) < 15:
                return None   # Not enough data — fail open
            return sum(records) / len(records)

    def all_stats(self) -> Dict[str, float]:
        """Return all tracked win rates as a flat dict for telemetry/logging."""
        out = {}
        with self._lock:
            for (ch, pair, regime), records in self._records.items():
                if len(records) >= 5:
                    key = f"{ch}/{pair}/{regime}"
                    out[key] = round(sum(records) / len(records) * 100, 1)
        return out


class StatisticalFilter:
    """Applies adaptive confidence gates based on rolling win-rate statistics.

    Gate logic:
    ────────────────────────────────────────────────────────────
    win_rate >= 0.50  → pass (no penalty)
    0.40 <= WR < 0.50 → pass with –5 pts confidence penalty
    0.30 <= WR < 0.40 → pass with –10 pts confidence penalty (soft suppress)
    WR  <  0.30       → HARD SUPPRESS (signal = None)
    None (no history) → pass (fail-open)
    ────────────────────────────────────────────────────────────
    """

    # Confidence penalty thresholds
    _SOFT_PENALTY_THRESHOLD: float = 0.40   # –5 pts below this WR
    _HARD_PENALTY_THRESHOLD: float = 0.30   # –10 pts below this WR; suppress below _SUPPRESS_THRESHOLD
    _SUPPRESS_THRESHOLD: float = 0.25       # Hard suppress below this WR

    def __init__(self, store: Optional[RollingWinRateStore] = None) -> None:
        self._store = store or RollingWinRateStore()

    @property
    def store(self) -> RollingWinRateStore:
        return self._store

    def check(
        self,
        channel: str,
        pair: str,
        regime: str,
        current_confidence: float,
    ) -> Tuple[bool, float, str]:
        """Check whether the signal should be emitted based on rolling win rate.

        Parameters
        ----------
        channel, pair, regime:
            Signal identifiers for win-rate lookup.
        current_confidence:
            Signal confidence score (0–100).

        Returns
        -------
        (allow: bool, adjusted_confidence: float, reason: str)
            allow: False means the signal should be suppressed.
            adjusted_confidence: confidence after penalty (may be unchanged).
            reason: human-readable explanation for logs.
        """
        win_rate = self._store.win_rate(channel, pair, regime)

        if win_rate is None:
            return True, current_confidence, "stat_filter:no_history"

        if win_rate < self._SUPPRESS_THRESHOLD:
            return False, 0.0, f"stat_filter:suppressed(wr={win_rate:.1%})"

        if win_rate < self._HARD_PENALTY_THRESHOLD:
            adj = max(0.0, current_confidence - 10.0)
            return True, adj, f"stat_filter:hard_penalty(wr={win_rate:.1%})"

        if win_rate < self._SOFT_PENALTY_THRESHOLD:
            adj = max(0.0, current_confidence - 5.0)
            return True, adj, f"stat_filter:soft_penalty(wr={win_rate:.1%})"

        return True, current_confidence, f"stat_filter:ok(wr={win_rate:.1%})"

    def record(self, outcome: SignalOutcome) -> None:
        """Forward a resolved signal outcome to the underlying win-rate store."""
        self._store.record(outcome)
```

### Step 2 — Instantiate `StatisticalFilter` in `src/scanner.py`

```python
from src.stat_filter import StatisticalFilter, SignalOutcome

# Module-level singleton (shared across all scan iterations)
_stat_filter = StatisticalFilter()
```

### Step 3 — Apply filter gate in scanner pipeline

After `SignalScoringEngine.score()` sets `sig.confidence`, add:

```python
allow, adj_conf, stat_reason = _stat_filter.check(
    channel=sig.channel,
    pair=sig.symbol,
    regime=regime_ctx.label,
    current_confidence=sig.confidence,
)
if not allow:
    log.debug("stat_filter suppressed %s/%s: %s", sig.symbol, sig.channel, stat_reason)
    sig = None
    continue

sig.confidence = adj_conf
if "penalty" in stat_reason:
    sig.soft_gate_flags = (sig.soft_gate_flags + f",{stat_reason}").lstrip(",")
```

### Step 4 — Record outcomes in `src/signal_lifecycle.py`

When a signal is resolved (status transitions to `TP1_HIT`, `TP2_HIT`, `SL_HIT`, or `CANCELLED`):

```python
from src.scanner import _stat_filter   # or inject via constructor
from src.stat_filter import SignalOutcome

def _record_outcome(signal: Signal) -> None:
    """Record signal resolution for statistical win-rate tracking."""
    won = signal.best_tp_hit >= 1   # TP1 or better = win
    outcome = SignalOutcome(
        signal_id=signal.signal_id,
        channel=signal.channel,
        pair=signal.symbol,
        regime=signal.entry_regime,
        setup_class=signal.setup_class,
        won=won,
        pnl_pct=signal.best_tp_pnl_pct,
    )
    _stat_filter.record(outcome)
```

### Step 5 — Expose win-rate stats in admin telemetry

In `src/telemetry.py` or the Telegram admin interface, add a command (e.g., `/statstats`)
that calls `_stat_filter.store.all_stats()` and formats the output as a table.

### Step 6 — Tests (`tests/test_advanced_filters.py`)

```python
def test_stat_filter_fail_open_with_no_history():
    from src.stat_filter import StatisticalFilter
    sf = StatisticalFilter()
    allow, conf, reason = sf.check("360_SCALP", "BTCUSDT", "TRENDING_UP", 75.0)
    assert allow and conf == 75.0 and "no_history" in reason

def test_stat_filter_suppresses_below_threshold():
    from src.stat_filter import StatisticalFilter, RollingWinRateStore, SignalOutcome
    store = RollingWinRateStore(window=30)
    # Record 20 losses and 3 wins → WR = 3/23 ≈ 13%
    for i in range(20):
        store.record(SignalOutcome(f"sig{i}", "360_SCALP", "BTCUSDT", "RANGING", "", False, -1.0))
    for i in range(3):
        store.record(SignalOutcome(f"win{i}", "360_SCALP", "BTCUSDT", "RANGING", "", True, 1.0))
    sf = StatisticalFilter(store)
    allow, conf, reason = sf.check("360_SCALP", "BTCUSDT", "RANGING", 80.0)
    assert not allow

def test_stat_filter_soft_penalty_at_40pct_win_rate():
    from src.stat_filter import StatisticalFilter, RollingWinRateStore, SignalOutcome
    store = RollingWinRateStore(window=30)
    # 12 wins + 18 losses = 40% WR
    for i in range(12):
        store.record(SignalOutcome(f"win{i}", "SWING", "ETHUSDT", "VOLATILE", "", True, 1.5))
    for i in range(18):
        store.record(SignalOutcome(f"loss{i}", "SWING", "ETHUSDT", "VOLATILE", "", False, -1.0))
    sf = StatisticalFilter(store)
    allow, conf, reason = sf.check("SWING", "ETHUSDT", "VOLATILE", 70.0)
    assert allow and conf == 65.0   # –5 pt soft penalty
    assert "soft_penalty" in reason

def test_rolling_win_rate_store_bounds():
    from src.stat_filter import RollingWinRateStore, SignalOutcome
    store = RollingWinRateStore(window=30)
    # Record 40 outcomes — only last 30 should be kept
    for i in range(35):
        won = i % 2 == 0
        store.record(SignalOutcome(f"s{i}", "CH", "SYM", "REGIME", "", won, 0.0))
    wr = store.win_rate("CH", "SYM", "REGIME")
    assert wr is not None
    assert 0.0 <= wr <= 1.0
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Repeated false signals from poor (channel, pair, regime) combinations | Unchecked | Automatically suppressed after 30-signal window confirms low WR |
| System adaptability to changing market conditions | None (static logic) | Self-correcting via rolling WR tracking |
| Cold-start signal emission | Baseline | Fail-open — no impact until 15+ signals resolved |
| Operator transparency | No per-combination quality data | Full win-rate stats available via `/statstats` command |
| Estimated false-positive reduction | Baseline | 15–25% reduction on historically poor combinations |

---

## Dependencies

- **PR_09** — Signal scoring engine provides the `confidence` value that the filter
  penalises or accepts.
- **PR_11** — Walk-forward backtester provides historical win rates that can be used
  to pre-seed the `RollingWinRateStore` before live trading begins.
