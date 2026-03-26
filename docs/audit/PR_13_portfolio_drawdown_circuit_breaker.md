# PR_13 — Portfolio-Level Drawdown Circuit Breaker

**PR Number:** PR_13  
**Branch:** `feature/pr13-portfolio-drawdown-circuit-breaker`  
**Category:** Risk & Reliability (Phase 2A)  
**Priority:** P0 (safety-critical — implement first in Phase 2)  
**Dependency:** PR_12 (Phase 1 — AI Statistical Filter merged as #138)  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Implement aggregate portfolio-level drawdown monitoring with tiered throttling. The existing `circuit_breaker.py` handles per-channel circuit breaking but has no awareness of cumulative portfolio-level losses. This PR adds a `PortfolioCircuitBreaker` that tracks aggregate unrealised P&L across all active positions and enforces three escalating protection levels:

- **Yellow** (−3% daily): reduce all new position sizes by 50%.
- **Red** (−5% daily): halt new signal emission for 4 hours.
- **Black** (−8% daily): halt all trading for 24 hours + notify admin via Telegram.

---

## Current State

`src/circuit_breaker.py` (≈15KB) exists for per-channel circuit breaking:
- Tracks consecutive losses per channel.
- Applies a cool-down period after N consecutive losses.
- Does **not** aggregate unrealised P&L across channels.
- Does **not** apply portfolio-level halt logic.

No portfolio-level drawdown protection exists anywhere in the codebase.

---

## Proposed Changes

### New class: `PortfolioCircuitBreaker` in `src/circuit_breaker.py`

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

DRAWDOWN_YELLOW = -0.03   # −3%: reduce position sizes
DRAWDOWN_RED    = -0.05   # −5%: halt new signals 4h
DRAWDOWN_BLACK  = -0.08   # −8%: halt all 24h + admin alert

@dataclass
class DrawdownState:
    level: str = "GREEN"          # "GREEN", "YELLOW", "RED", "BLACK"
    halt_until: Optional[datetime] = None
    position_size_multiplier: float = 1.0

class PortfolioCircuitBreaker:
    """Monitors aggregate portfolio drawdown and enforces tiered trading halts."""

    def __init__(self, starting_balance: float, admin_notifier=None):
        self._start_balance = starting_balance
        self._daily_start_balance = starting_balance
        self._admin_notifier = admin_notifier
        self._state = DrawdownState()
        self._daily_reset_date: Optional[str] = None

    def update(self, current_portfolio_value: float) -> DrawdownState:
        """Call once per scan cycle with the latest portfolio mark-to-market value."""
        self._maybe_reset_daily(current_portfolio_value)
        drawdown = (current_portfolio_value - self._daily_start_balance) / self._daily_start_balance

        now = datetime.now(timezone.utc)
        if self._state.halt_until and now < self._state.halt_until:
            return self._state  # still in existing halt window

        if drawdown <= DRAWDOWN_BLACK:
            self._transition("BLACK", now, hours=24)
        elif drawdown <= DRAWDOWN_RED:
            self._transition("RED", now, hours=4)
        elif drawdown <= DRAWDOWN_YELLOW:
            self._transition("YELLOW", now, hours=0)
        else:
            self._state = DrawdownState(level="GREEN")

        return self._state

    def get_position_multiplier(self) -> float:
        return self._state.position_size_multiplier

    def is_halted(self) -> bool:
        if self._state.level in ("RED", "BLACK"):
            now = datetime.now(timezone.utc)
            return self._state.halt_until is None or now < self._state.halt_until
        return False

    def _transition(self, level: str, now: datetime, hours: int) -> None:
        if self._state.level == level:
            return
        halt_until = None
        mult = 1.0
        if level == "YELLOW":
            mult = 0.5
        elif level == "RED":
            halt_until = now + timedelta(hours=hours)
            mult = 0.0
        elif level == "BLACK":
            halt_until = now + timedelta(hours=hours)
            mult = 0.0
            if self._admin_notifier:
                self._admin_notifier(
                    f"🚨 BLACK circuit breaker triggered — portfolio drawdown critical. "
                    f"All trading halted for 24h."
                )
        logger.warning("Portfolio circuit breaker: %s → %s", self._state.level, level)
        self._state = DrawdownState(level=level, halt_until=halt_until,
                                    position_size_multiplier=mult)

    def _maybe_reset_daily(self, current_value: float) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_reset_date != today:
            self._daily_start_balance = current_value
            self._daily_reset_date = today
            if self._state.level != "BLACK":
                self._state = DrawdownState()
```

### Wire into `src/signal_router.py`

Add a pre-dispatch check before every signal is routed:

```python
# In SignalRouter.route() or equivalent dispatch method:
state = portfolio_cb.update(portfolio_tracker.current_value())
if state.level in ("RED", "BLACK"):
    logger.info("Signal suppressed by portfolio circuit breaker (level=%s)", state.level)
    return None
if state.level == "YELLOW":
    signal.position_size_override = signal.position_size * state.position_size_multiplier
```

### Config additions in `src/config/__init__.py`

```python
PORTFOLIO_CB_YELLOW_PCT: float = float(os.getenv("PORTFOLIO_CB_YELLOW_PCT", "-0.03"))
PORTFOLIO_CB_RED_PCT:    float = float(os.getenv("PORTFOLIO_CB_RED_PCT",    "-0.05"))
PORTFOLIO_CB_BLACK_PCT:  float = float(os.getenv("PORTFOLIO_CB_BLACK_PCT",  "-0.08"))
```

---

## Implementation Steps

1. Add `PortfolioCircuitBreaker` class and `DrawdownState` dataclass to `src/circuit_breaker.py`.
2. Add the three threshold constants (`DRAWDOWN_YELLOW`, `DRAWDOWN_RED`, `DRAWDOWN_BLACK`) to `src/circuit_breaker.py` and expose them from `config/__init__.py` as env-configurable overrides.
3. Instantiate `PortfolioCircuitBreaker` in `main.py` (or wherever `SignalRouter` is created), passing the starting balance and admin notifier callable.
4. In `signal_router.py`, call `portfolio_cb.update()` before each signal is dispatched and apply position multiplier or halt as described.
5. Write unit tests in `tests/test_portfolio_circuit_breaker.py`.
6. Update `docs/audit/MASTER_AUDIT_REPORT.md` Phase 2 table to mark PR_13 as in-progress.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/circuit_breaker.py` | Add `PortfolioCircuitBreaker` class and `DrawdownState` dataclass |
| `src/signal_router.py` | Wire in pre-dispatch portfolio CB check |
| `src/config/__init__.py` | Add three drawdown threshold config constants |
| `tests/test_portfolio_circuit_breaker.py` | New test file |

---

## Testing Requirements

```python
# tests/test_portfolio_circuit_breaker.py
def test_green_state_at_start():
    cb = PortfolioCircuitBreaker(starting_balance=10_000)
    state = cb.update(10_000)
    assert state.level == "GREEN"

def test_yellow_triggers_at_minus_3_pct():
    cb = PortfolioCircuitBreaker(starting_balance=10_000)
    state = cb.update(9_690)   # −3.1%
    assert state.level == "YELLOW"
    assert state.position_size_multiplier == 0.5

def test_red_triggers_at_minus_5_pct():
    cb = PortfolioCircuitBreaker(starting_balance=10_000)
    state = cb.update(9_450)   # −5.5%
    assert state.level == "RED"
    assert cb.is_halted()

def test_black_triggers_at_minus_8_pct():
    notified = []
    cb = PortfolioCircuitBreaker(starting_balance=10_000,
                                  admin_notifier=notified.append)
    state = cb.update(9_100)   # −9%
    assert state.level == "BLACK"
    assert len(notified) == 1

def test_daily_reset():
    cb = PortfolioCircuitBreaker(starting_balance=10_000)
    cb._daily_start_balance = 9_000   # simulate prior day low
    cb._daily_reset_date = "2099-01-01"  # force reset on next call
    state = cb.update(9_800)   # new day, fresh balance
    assert state.level == "GREEN"
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Portfolio max daily drawdown | Unbounded | Capped at −8% |
| Response to losing streak | No automatic action | Tiered auto-throttle within one scan cycle |
| Admin awareness | Manual monitoring | Instant Telegram alert on BLACK |
| Position size in drawdown | Full | 50% reduction at −3% |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Starting balance not set correctly | Validate on startup; raise `ConfigurationError` if balance ≤ 0 |
| Daily reset at wrong UTC time | Tie reset to explicit UTC midnight check, not wall-clock drift |
| False positive BLACK on data spike | Add 2-minute confirmation window before BLACK transition (require two consecutive readings) |
| Admin notifier throws exception | Wrap in try/except; log error but do not prevent halt from taking effect |
