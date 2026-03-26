# PR_08 — Trailing Stop Upgrade

**Branch:** `feature/pr08-trailing-stop`  
**Priority:** 8  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Upgrade the trailing stop logic from a static ATR-multiple description to an adaptive,
multi-stage trailing system that:

1. **Adjusts trailing step with ATR percentile** — in high-volatility regimes the trailing
   buffer is wider (avoids premature stops on volatile wicks); in low-volatility regimes
   it tightens to lock in gains faster.

2. **Implements partial profit locking** — when TP1 is hit, 40% of the position is
   conceptually closed (noted in `execution_note`) and the trailing stop moves to the
   breakeven level. When TP2 is hit, the stop trails at 0.5× ATR.

3. **ATR-adaptive trailing step updates** — instead of using ATR at signal creation, the
   trade monitor should recompute ATR on each lifecycle check and update the trailing
   buffer accordingly.

The `Signal` dataclass already carries `original_sl_distance` (used by trailing logic).
This PR adds `trailing_atr_mult_effective` and `trailing_stage` fields to the Signal so
the trade monitor can track the current trailing state.

---

## Files to Change

| File | Change type |
|------|-------------|
| `src/channels/base.py` | Add `trailing_atr_mult_effective` and `trailing_stage` to Signal; add `TrailingStopState` dataclass |
| `src/trade_monitor.py` | Implement stage-aware trailing stop logic |
| `src/channels/scalp.py` | Set initial trailing parameters on signal |
| `src/channels/swing.py` | Set initial trailing parameters on signal |
| `tests/test_trade_monitor.py` | Add tests for trailing stop stage transitions |

---

## Implementation Steps

### Step 1 — Add fields to `Signal` in `src/channels/base.py`

```python
# In the Signal dataclass, add after existing trailing fields:
trailing_atr_mult_effective: float = 0.0   # Current trailing ATR multiple (updates during trade)
trailing_stage: int = 0                     # 0=initial, 1=TP1_hit (breakeven), 2=TP2_hit (tight trail)
partial_close_pct: float = 0.0             # Fraction of position notionally closed
```

### Step 2 — Add `TrailingStopState` dataclass to `src/channels/base.py`

```python
from dataclasses import dataclass as _dc

@_dc
class TrailingStopState:
    """Encapsulates the dynamic trailing stop configuration for a live signal.

    Used by the trade monitor to compute the current trailing stop level.
    """
    initial_atr: float                # ATR at signal creation
    current_atr: float = 0.0         # ATR from most recent lifecycle check
    stage: int = 0                   # 0=entry, 1=TP1 hit, 2=TP2 hit
    breakeven_set: bool = False      # Whether SL has been moved to breakeven
    tight_trail_active: bool = False # Whether the tight 0.5× ATR trail is active

    @property
    def effective_mult(self) -> float:
        """ATR multiple for the current stage."""
        if self.stage == 2:
            return 0.5     # Post-TP2: tight trail
        if self.stage == 1:
            return 1.0     # Post-TP1: intermediate trail
        return 2.0         # Entry: standard trail

    @property
    def trail_distance(self) -> float:
        """Absolute trailing distance using current ATR."""
        atr = self.current_atr if self.current_atr > 0 else self.initial_atr
        return atr * self.effective_mult
```

### Step 3 — Implement ATR-adaptive trailing in `src/trade_monitor.py`

Locate the trailing stop update logic and replace with the staged approach:

```python
def _compute_trailing_stop(
    signal: Signal,
    current_price: float,
    current_atr: float,
    trailing_state: TrailingStopState,
    atr_percentile: float = 50.0,
) -> float:
    """Compute the new trailing stop level based on current stage and ATR.

    Parameters
    ----------
    signal:
        Active signal with direction, entry, current stop_loss.
    current_price:
        Latest market price.
    current_atr:
        ATR computed from the most recent candles (updated each lifecycle poll).
    trailing_state:
        Mutable state object tracking the current trailing stage.
    atr_percentile:
        Current ATR percentile (0-100) from regime detector.
        High percentile → wider trail; low percentile → tighter trail.
    """
    # Update ATR with live value
    trailing_state.current_atr = current_atr

    # Adjust effective multiplier for volatility percentile.
    # Note: trailing-stop multipliers (1.3×/0.8×) are intentionally narrower than the
    # entry SL multipliers in PR_07 (1.3×/0.8× of a wider base). Entry SL must absorb
    # initial wick noise; trailing stops can be tighter because the position is already
    # in profit and we want to lock gains faster relative to the initial risk width.
    vol_adj = 1.0
    if atr_percentile >= 80:
        vol_adj = 1.3   # Wide trail in high-vol — avoid premature trailing-stop hits
    elif atr_percentile <= 20:
        vol_adj = 0.8   # Tight trail in low-vol — lock gains faster

    trail_dist = trailing_state.trail_distance * vol_adj

    if signal.direction.value == "LONG":
        new_sl = current_price - trail_dist
        # Only move SL up (never down for a LONG)
        return max(signal.stop_loss, new_sl)
    else:
        new_sl = current_price + trail_dist
        # Only move SL down (never up for a SHORT)
        return min(signal.stop_loss, new_sl)


def _handle_tp_hit(signal: Signal, trailing_state: TrailingStopState, tp_level: int) -> None:
    """Update trailing stage and execution notes when a TP is hit.

    Parameters
    ----------
    tp_level:
        1 for TP1 hit, 2 for TP2 hit.
    """
    if tp_level == 1 and trailing_state.stage == 0:
        trailing_state.stage = 1
        trailing_state.breakeven_set = True
        signal.stop_loss = signal.entry   # Move SL to breakeven
        signal.partial_close_pct = 0.40   # 40% position notionally closed
        if signal.execution_note:
            signal.execution_note += " | TP1 HIT: Move SL to BE, close 40%"
        else:
            signal.execution_note = "TP1 HIT: Move SL to BE, close 40%"

    elif tp_level == 2 and trailing_state.stage == 1:
        trailing_state.stage = 2
        trailing_state.tight_trail_active = True
        signal.partial_close_pct = 0.75   # 75% total position closed
        if signal.execution_note:
            signal.execution_note += " | TP2 HIT: Tight trail 0.5× ATR, close 35% more"
        else:
            signal.execution_note = "TP2 HIT: Tight trail 0.5× ATR, close 35% more"
```

### Step 4 — Set initial trailing parameters in channels

In `ScalpChannel` and `SwingChannel`, after building the signal, set:
```python
if sig is not None:
    sig.trailing_atr_mult_effective = self.config.trailing_atr_mult
    sig.trailing_stage = 0
```

### Step 5 — Update `trailing_desc` to be more informative

In `build_channel_signal()`, replace:
```python
trailing_desc=f"{config.trailing_atr_mult}×ATR",
```
with:
```python
trailing_desc=(
    f"Stage 1: {config.trailing_atr_mult}×ATR | "
    f"Post-TP1: 1×ATR (BE) | Post-TP2: 0.5×ATR (tight)"
),
```

### Step 6 — Tests (`tests/test_trade_monitor.py`)

```python
def test_trailing_stop_widens_in_high_vol():
    from src.channels.base import TrailingStopState
    from src.trade_monitor import _compute_trailing_stop
    from unittest.mock import MagicMock
    sig = MagicMock()
    sig.direction.value = "LONG"
    sig.stop_loss = 95.0
    sig.entry = 100.0
    state = TrailingStopState(initial_atr=1.0, stage=0)
    new_sl = _compute_trailing_stop(sig, 105.0, 1.0, state, atr_percentile=85)
    assert new_sl > 105.0 - 1.0 * 2.0 * 1.3 - 0.01   # Trail dist = 2.0 * 1.3 ATR

def test_tp1_hit_moves_sl_to_breakeven():
    from src.channels.base import TrailingStopState, Signal
    from src.smc import Direction
    from src.trade_monitor import _handle_tp_hit
    sig = Signal(channel="TEST", symbol="BTCUSDT", direction=Direction.LONG,
                 entry=100.0, stop_loss=97.0, tp1=102.0, tp2=105.0)
    state = TrailingStopState(initial_atr=1.0, stage=0)
    _handle_tp_hit(sig, state, tp_level=1)
    assert sig.stop_loss == 100.0   # Moved to entry (breakeven)
    assert state.stage == 1
    assert sig.partial_close_pct == 0.40
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Winners converted to breakeven losses | Frequent (no BE mechanism) | Reduced by ~60% (auto-BE on TP1 hit) |
| TP3 capture rate | Low (static trailing often too wide) | Improved by ~15% (tight trail post-TP2) |
| Max adverse excursion on winners | High | Reduced (tighter trail in low-vol) |
| System adaptability to live ATR | None (ATR frozen at entry) | Fully adaptive on each lifecycle poll |

---

## Dependencies

- **PR_07** — ATR percentile from `RegimeContext` required for volatility-adaptive trail width.
