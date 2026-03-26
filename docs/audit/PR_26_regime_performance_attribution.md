# PR_26 — Regime Performance Attribution Reports

**PR Number:** PR_26  
**Branch:** `feature/pr26-regime-performance-attribution`  
**Category:** Monitoring & Observability (Phase 2D)  
**Priority:** P2  
**Dependency:** PR_11 (Phase 1 — Backtester Per-Pair + Regime, merged as #137)  
**Effort estimate:** Small–Medium (1–2 days)

---

## Objective

Tag every closed trade record with the market regime that was active at entry time, then generate weekly automated reports showing P&L breakdown by regime per channel. This enables the operator to identify which regimes are profitable, which are lossy, and dynamically adjust regime-specific parameters.

---

## Current State

`src/performance_tracker.py` records trade outcomes but does not capture the market regime at entry time. The `RegimeContext` object is computed during each scan cycle but is not persisted alongside trade records. Without regime tagging, it is impossible to attribute performance differences to regime shifts.

---

## Proposed Changes

### Extend trade recording in `src/performance_tracker.py`

```python
@dataclass
class TradeRecord:
    signal_id:      str
    channel:        str
    symbol:         str
    direction:      str
    entry_price:    float
    exit_price:     float
    pnl_pct:        float
    score:          float
    duration_min:   float
    regime:         str = "UNKNOWN"   # NEW: regime label at entry time
    closed_at:      Optional[datetime] = None
```

Update the `record_trade()` call in `trade_observer.py` to include the regime:

```python
# In trade_observer.py, when recording a closed trade:
performance_tracker.record_trade(TradeRecord(
    ...existing fields...,
    regime=signal.regime_context_label,   # propagated from RegimeContext.label
))
```

### Add `generate_regime_report()` to `src/performance_tracker.py`

```python
from collections import defaultdict
from typing import Dict, List

REGIME_LABELS = [
    "TRENDING_UP",
    "TRENDING_DOWN",
    "RANGING",
    "VOLATILE",
    "QUIET",
    "UNKNOWN",
]

def generate_regime_report(
    self,
    channel: Optional[str] = None,
    window_days: int = 7,
) -> str:
    """
    Generate a formatted weekly P&L breakdown by regime.

    If *channel* is None, report covers all channels combined.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    trades = [
        t for t in self._trades
        if t.closed_at and t.closed_at >= cutoff
        and (channel is None or t.channel == channel)
    ]

    by_regime: Dict[str, List[float]] = defaultdict(list)
    for t in trades:
        by_regime[t.regime].append(t.pnl_pct)

    lines = [
        f"📈 *Regime Attribution Report* "
        f"({'All Channels' if channel is None else channel}) — last {window_days}d",
        "",
    ]
    for regime in REGIME_LABELS:
        pnls = by_regime.get(regime, [])
        if not pnls:
            continue
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n
        avg_pnl = sum(pnls) / n
        lines.append(f"*{regime}*  ({n} trades)")
        lines.append(f"  Win Rate:  {wr:.1%}")
        lines.append(f"  Avg P&L:   {avg_pnl:+.2%}")
        lines.append("")

    if not any(by_regime.values()):
        lines.append("No trades recorded for this period.")

    return "\n".join(lines)
```

### Add weekly auto-report scheduler in `src/main.py`

```python
import asyncio
from datetime import datetime, timezone

async def weekly_regime_report_task(performance_tracker, telegram_bot):
    """Send weekly regime attribution report every Monday at 09:00 UTC."""
    while True:
        now = datetime.now(timezone.utc)
        # Calculate seconds until next Monday 09:00 UTC
        days_until_monday = (7 - now.weekday()) % 7 or 7
        next_monday = now.replace(hour=9, minute=0, second=0, microsecond=0)
        next_monday = next_monday + timedelta(days=days_until_monday)
        await asyncio.sleep((next_monday - now).total_seconds())
        report = performance_tracker.generate_regime_report()
        await telegram_bot.send_admin_message(report, parse_mode="Markdown")
```

### Propagate regime label through signal pipeline

Ensure that `signal.regime_context_label` (a plain string, e.g., `"TRENDING_UP"`) is set in `scanner.py` when a signal is prepared:

```python
# In scanner.py _prepare_signal():
sig.regime_context_label = regime_ctx.label   # already computed as part of PR_01
```

---

## Implementation Steps

1. Add `regime` field to `TradeRecord` dataclass in `performance_tracker.py`.
2. Update `trade_observer.py` to populate the regime field when recording closed trades.
3. In `scanner.py`, ensure `sig.regime_context_label` is set (may already be done via PR_01).
4. Add `generate_regime_report()` method to `PerformanceTracker`.
5. Add weekly scheduler task in `main.py`.
6. Write unit tests in `tests/test_regime_attribution.py`.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/performance_tracker.py` | Add `regime` field to `TradeRecord`; add `generate_regime_report()` |
| `src/trade_observer.py` | Populate `regime` field when recording trade |
| `src/scanner.py` | Set `sig.regime_context_label` if not already set |
| `src/main.py` | Add `weekly_regime_report_task` background task |
| `tests/test_regime_attribution.py` | New test file |

---

## Testing Requirements

```python
# tests/test_regime_attribution.py
def make_trades(regimes=None):
    regimes = regimes or ["TRENDING_UP"] * 5 + ["RANGING"] * 5
    return [
        TradeRecord(
            signal_id=f"sig-{i}", channel="SCALP", symbol="BTCUSDT",
            direction="LONG", entry_price=50_000, exit_price=51_000,
            pnl_pct=0.01 if i % 2 == 0 else -0.005,
            score=70, duration_min=30,
            regime=r, closed_at=datetime.now(timezone.utc),
        )
        for i, r in enumerate(regimes)
    ]

def test_report_contains_regime_labels():
    tracker = PerformanceTracker()
    for t in make_trades():
        tracker._trades.append(t)
    report = tracker.generate_regime_report()
    assert "TRENDING_UP" in report
    assert "RANGING" in report

def test_report_shows_win_rate_per_regime():
    tracker = PerformanceTracker()
    for t in make_trades(regimes=["TRENDING_UP"] * 10):
        tracker._trades.append(t)
    report = tracker.generate_regime_report()
    assert "Win Rate" in report

def test_empty_period_handled():
    tracker = PerformanceTracker()
    report = tracker.generate_regime_report()
    assert "No trades recorded" in report
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Regime-specific performance visibility | None | Weekly automated breakdown |
| Time to identify losing regime | Manual analysis (hours) | Reading weekly report (minutes) |
| Parameter tuning data quality | Intuitive | Data-driven per regime |
| Trade record richness | No regime tag | Full regime context per trade |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Regime label not propagated to trade record | Add assertion in `record_trade()`; fallback to "UNKNOWN" |
| Weekly report fires at wrong time after restart | Use wall-clock recalculation; do not persist sleep duration |
| Old trade records missing regime field | Handle `regime` default "UNKNOWN" gracefully in report |
| Report message too long for Telegram | Split into per-channel messages if total length > 3000 chars |
