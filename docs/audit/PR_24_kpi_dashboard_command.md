# PR_24 — KPI Dashboard Command

**PR Number:** PR_24  
**Branch:** `feature/pr24-kpi-dashboard-command`  
**Category:** Monitoring & Observability (Phase 2D)  
**Priority:** P1  
**Dependency:** PR_13 (Portfolio-Level Drawdown Circuit Breaker)  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Create a `/dashboard` Telegram command that renders real-time KPIs per channel and in aggregate. The dashboard provides a concise, actionable snapshot of system health without requiring manual log inspection.

---

## Current State

`src/performance_tracker.py` and `src/performance_report.py` exist and compute metrics such as win rate, P&L, and trade count. However:
- There is no `/dashboard` Telegram command.
- Metrics are only accessible by reading logs or triggering manual report generation.
- No Sharpe ratio, profit factor, false positive rate, or score distribution is surfaced to operators.
- There is no real-time aggregate view across all channels.

---

## Proposed Changes

### New file: `src/commands/dashboard.py`

```python
"""Telegram /dashboard command handler."""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DASHBOARD_EMOJI = {
    "win_rate_good":  "🟢",
    "win_rate_warn":  "🟡",
    "win_rate_bad":   "🔴",
    "drawdown_ok":    "✅",
    "drawdown_warn":  "⚠️",
    "drawdown_crit":  "🚨",
}

def _win_rate_emoji(win_rate: float) -> str:
    if win_rate >= 0.55:
        return DASHBOARD_EMOJI["win_rate_good"]
    if win_rate >= 0.45:
        return DASHBOARD_EMOJI["win_rate_warn"]
    return DASHBOARD_EMOJI["win_rate_bad"]

def _drawdown_emoji(drawdown_pct: float) -> str:
    if drawdown_pct > -0.03:
        return DASHBOARD_EMOJI["drawdown_ok"]
    if drawdown_pct > -0.05:
        return DASHBOARD_EMOJI["drawdown_warn"]
    return DASHBOARD_EMOJI["drawdown_crit"]

def render_dashboard(
    performance_tracker,
    portfolio_circuit_breaker=None,
) -> str:
    """
    Render the full dashboard text for all channels + aggregate.

    Returns a formatted string ready to send as a Telegram message.
    """
    lines = ["📊 *360-Crypto Dashboard*", ""]

    channels = performance_tracker.get_channel_names()
    for channel in channels:
        stats = performance_tracker.get_channel_stats(channel, window_days=7)
        wr_emoji = _win_rate_emoji(stats.get("win_rate", 0))
        lines.append(f"*{channel}* {wr_emoji}")
        lines.append(f"  Win Rate (7d):      {stats.get('win_rate', 0):.1%}")
        lines.append(f"  Profit Factor:      {stats.get('profit_factor', 0):.2f}")
        lines.append(f"  Sharpe Ratio:       {stats.get('sharpe', 0):.2f}")
        lines.append(f"  Max Drawdown:       {stats.get('max_drawdown', 0):.1%}")
        lines.append(f"  Signal Freq:        {stats.get('signals_per_hour', 0):.1f}/h")
        lines.append(f"  False Pos Rate:     {stats.get('false_positive_rate', 0):.1%}")
        lines.append(f"  Avg Score (W/L):    {stats.get('avg_score_winners', 0):.0f} / "
                     f"{stats.get('avg_score_losers', 0):.0f}")
        lines.append(f"  Avg Duration:       {stats.get('avg_duration_min', 0):.0f}m")
        lines.append("")

    # Aggregate
    agg = performance_tracker.get_aggregate_stats(window_days=7)
    dd_emoji = _drawdown_emoji(agg.get("max_drawdown", 0))
    lines.append(f"*Aggregate* {dd_emoji}")
    lines.append(f"  Win Rate (7d):      {agg.get('win_rate', 0):.1%}")
    lines.append(f"  Total P&L:          {agg.get('total_pnl_pct', 0):+.2%}")
    lines.append(f"  Max Drawdown:       {agg.get('max_drawdown', 0):.1%}")
    lines.append(f"  Active Positions:   {agg.get('active_positions', 0)}")

    # Circuit breaker status
    if portfolio_circuit_breaker:
        state = portfolio_circuit_breaker._state
        cb_emoji = {"GREEN": "✅", "YELLOW": "⚠️", "RED": "🔴", "BLACK": "⛔"}.get(
            state.level, "❓"
        )
        lines.append(f"  Circuit Breaker:    {cb_emoji} {state.level}")

    return "\n".join(lines)
```

### Register `/dashboard` command in bot handler

```python
# In src/telegram_bot.py or equivalent command router:
from src.commands.dashboard import render_dashboard

async def cmd_dashboard(update, context):
    """Handle /dashboard command."""
    text = render_dashboard(
        performance_tracker=performance_tracker,
        portfolio_circuit_breaker=portfolio_cb,
    )
    await update.message.reply_text(text, parse_mode="Markdown")
```

### Extend `src/performance_tracker.py`

Add the following methods if not already present:

```python
def get_channel_stats(self, channel: str, window_days: int = 7) -> dict:
    """Return KPI dict for a single channel over the past N days."""
    # ... compute from stored trade records ...
    return {
        "win_rate": ...,
        "profit_factor": ...,
        "sharpe": ...,
        "max_drawdown": ...,
        "signals_per_hour": ...,
        "false_positive_rate": ...,
        "avg_score_winners": ...,
        "avg_score_losers": ...,
        "avg_duration_min": ...,
    }

def get_aggregate_stats(self, window_days: int = 7) -> dict:
    """Return aggregate KPI dict across all channels."""
    # ... aggregate per-channel stats ...

def get_channel_names(self) -> list:
    """Return list of tracked channel names."""
```

---

## Implementation Steps

1. Create `src/commands/` directory with `__init__.py`.
2. Create `src/commands/dashboard.py` with `render_dashboard()` and helpers.
3. Extend `src/performance_tracker.py` with `get_channel_stats()`, `get_aggregate_stats()`, `get_channel_names()`.
4. Register `/dashboard` command in the Telegram bot command handler.
5. Write unit tests in `tests/test_dashboard.py`.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/commands/__init__.py` | New empty package init |
| `src/commands/dashboard.py` | New — dashboard renderer |
| `src/performance_tracker.py` | Add `get_channel_stats()`, `get_aggregate_stats()`, `get_channel_names()` |
| `src/telegram_bot.py` (or equivalent) | Register `/dashboard` command |
| `tests/test_dashboard.py` | New test file |

---

## Testing Requirements

```python
# tests/test_dashboard.py
def make_mock_tracker(win_rate=0.55, drawdown=-0.02):
    tracker = Mock()
    tracker.get_channel_names.return_value = ["360_SCALP", "360_SWING"]
    tracker.get_channel_stats.return_value = {
        "win_rate": win_rate, "profit_factor": 1.4, "sharpe": 0.9,
        "max_drawdown": drawdown, "signals_per_hour": 2.1,
        "false_positive_rate": 0.22, "avg_score_winners": 78,
        "avg_score_losers": 62, "avg_duration_min": 45,
    }
    tracker.get_aggregate_stats.return_value = {
        "win_rate": win_rate, "total_pnl_pct": 0.05,
        "max_drawdown": drawdown, "active_positions": 3,
    }
    return tracker

def test_dashboard_renders_without_error():
    text = render_dashboard(make_mock_tracker())
    assert "360_SCALP" in text
    assert "Win Rate" in text

def test_green_emoji_good_win_rate():
    text = render_dashboard(make_mock_tracker(win_rate=0.60))
    assert "🟢" in text

def test_red_emoji_bad_win_rate():
    text = render_dashboard(make_mock_tracker(win_rate=0.35))
    assert "🔴" in text

def test_circuit_breaker_status_shown():
    from src.circuit_breaker import PortfolioCircuitBreaker
    cb = PortfolioCircuitBreaker(starting_balance=10_000)
    text = render_dashboard(make_mock_tracker(), portfolio_circuit_breaker=cb)
    assert "Circuit Breaker" in text
    assert "GREEN" in text
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Time to get system overview | Minutes (log search) | Seconds (/dashboard command) |
| Metrics surfaced to operator | Ad-hoc | 9 KPIs per channel + aggregate |
| Circuit breaker visibility | Zero | Shown in dashboard |
| Score quality tracking | None | Avg score winners vs losers |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Performance tracker missing required methods | Add with default implementations; existing data structures |
| Dashboard text too long for Telegram (4096 char limit) | Split across multiple messages if >4000 chars |
| Slow stats computation blocking bot event loop | Compute stats in background; cache for 60s |
| No trade data yet (cold start) | Return "No data yet" gracefully for missing metrics |
