# PR_10 — Duplicate Code Refactor

**Branch:** `feature/pr10-duplicate-refactor`  
**Priority:** 10  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Eliminate duplicated logic identified in the Master Audit Report across channel
implementations, centralising shared computations into reusable filter and utility
functions. This reduces maintenance burden (a fix in one place propagates everywhere),
removes subtle inconsistencies (e.g., SL being computed twice with potentially different
results), and makes the codebase easier to test.

Targeted duplications:

| Issue | Current location | Refactored to |
|-------|-----------------|---------------|
| SL/TP double computation | `scalp.py:_calc_levels()` + `base.py:build_channel_signal()` | `build_channel_signal()` only |
| Volume expansion check | `spot.py:_try_long/short()` inline | `filters.py:check_volume_expansion()` |
| Trailing stop description | hardcoded in every channel | `base.py:_default_trailing_desc()` |
| Adaptive spread check inconsistency | `scalp.py` uses non-adaptive, others use adaptive | All channels use `check_spread_adaptive()` |
| RSI threshold definition | implicit inside `check_rsi_regime()` | configurable defaults exposed via `PairProfile` |

---

## Files to Change

| File | Change type |
|------|-------------|
| `src/filters.py` | Add `check_volume_expansion()` function |
| `src/channels/base.py` | Add `_default_trailing_desc()` helper; deprecate inline SL/TP re-computation in channels |
| `src/channels/scalp.py` | Remove `_calc_levels()` method; use `check_volume_expansion()` and `check_spread_adaptive()` |
| `src/channels/swing.py` | Replace inline SL/TP with `build_channel_signal()` unified path |
| `src/channels/spot.py` | Replace inline volume expansion with `check_volume_expansion()` |
| `tests/test_channels.py` | Verify refactored paths produce identical results to old paths |

---

## Implementation Steps

### Step 1 — Add `check_volume_expansion()` to `src/filters.py`

```python
def check_volume_expansion(
    volumes: list | "np.ndarray",
    closes: list | "np.ndarray",
    lookback: int = 9,
    multiplier: float = 1.8,
) -> bool:
    """Return True when the most recent candle's USD volume exceeds the lookback average.

    Parameters
    ----------
    volumes:
        Raw volume (unit quantity, not USD) for the last N+1 candles.
    closes:
        Close prices for the last N+1 candles. Used to convert volume to USD.
    lookback:
        Number of prior candles to use for the average (excluding the last one).
    multiplier:
        Required ratio of last candle USD volume to average (e.g. 1.8× = 80% above avg).
    """
    import numpy as np
    v = np.asarray(volumes, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = len(v)
    if n < lookback + 1:
        return False
    usd_vol = v * c
    avg_usd = float(np.mean(usd_vol[-(lookback + 1):-1]))
    last_usd = float(usd_vol[-1])
    if avg_usd <= 0:
        return False
    return last_usd >= avg_usd * multiplier
```

### Step 2 — Remove `_calc_levels()` from `ScalpChannel`

The `_calc_levels()` method in `src/channels/scalp.py` computes `sl, tp1, tp2, tp3` and
then passes them directly to `build_channel_signal()`, which recalculates them internally
when `bb_width_pct` or param overrides are present.

**Before:**
```python
sl, tp1, tp2, tp3 = self._calc_levels(close, sl_dist, direction)
sig = build_channel_signal(..., sl=sl, tp1=tp1, tp2=tp2, tp3=tp3, ...)
```

**After:**
Let `build_channel_signal()` own all TP computation by passing sentinel values that
the function replaces:
```python
# Pass sl from the pre-computed sl_dist; tp values are computed by build_channel_signal
sl = close - sl_dist if direction == Direction.LONG else close + sl_dist
sig = build_channel_signal(
    ..., sl=sl, tp1=0.0, tp2=0.0, tp3=0.0, sl_dist=sl_dist, ...
)
```

Update `build_channel_signal()` to always recompute TP from `sl_dist` and `adj_ratios`,
ignoring the `tp1/tp2/tp3` arguments that are now deprecated:

```python
# In build_channel_signal(), after computing adj_ratios:
# Always compute TP from sl_dist (tp1/tp2/tp3 args are deprecated but kept for API compat)
if direction == Direction.LONG:
    tp1 = close + sl_dist * adj_ratios[0]
    tp2 = close + sl_dist * adj_ratios[1]
    tp3 = close + sl_dist * adj_ratios[2] if len(adj_ratios) > 2 else close + sl_dist * 2.0
else:
    tp1 = close - sl_dist * adj_ratios[0]
    tp2 = close - sl_dist * adj_ratios[1]
    tp3 = close - sl_dist * adj_ratios[2] if len(adj_ratios) > 2 else close - sl_dist * 2.0
```

### Step 3 — Replace inline volume expansion in `spot.py`

**Before (in `_try_long` and `_try_short`):**
```python
usd_volumes = [float(v) * float(c) for v, c in zip(volumes[-10:], closes_list[-10:])]
avg_usd_vol = sum(usd_volumes[:-1]) / 9
if usd_volumes[-1] < avg_usd_vol * self._volume_expansion_mult():
    return None
```

**After:**
```python
from src.filters import check_volume_expansion

if not check_volume_expansion(
    volumes, closes_list, lookback=9, multiplier=self._volume_expansion_mult()
):
    return None
```

### Step 4 — Unify spread check in `ScalpChannel`

In `src/channels/scalp.py`, replace:
```python
def _pass_basic_filters(self, spread_pct, volume_24h_usd):
    return (
        check_spread(spread_pct, self.config.spread_max)
        and check_volume(volume_24h_usd, self.config.min_volume)
    )
```
with:
```python
def _pass_basic_filters(self, spread_pct, volume_24h_usd, regime=""):
    from src.filters import check_spread_adaptive
    return (
        check_spread_adaptive(spread_pct, self.config.spread_max, regime=regime)
        and check_volume(volume_24h_usd, self.config.min_volume)
    )
```

Update all three `_evaluate_*` calls to pass `regime` to `_pass_basic_filters()`.

### Step 5 — Add `_default_trailing_desc()` to `src/channels/base.py`

```python
def _default_trailing_desc(trailing_atr_mult: float) -> str:
    """Return a standardised trailing stop description string."""
    return (
        f"Stage 1: {trailing_atr_mult}×ATR | "
        f"Post-TP1: 1×ATR (BE) | Post-TP2: 0.5×ATR (tight)"
    )
```

Replace all hardcoded `trailing_desc=f"{config.trailing_atr_mult}×ATR"` occurrences
with `trailing_desc=_default_trailing_desc(config.trailing_atr_mult)`.

### Step 6 — Deprecation markers

Add `# DEPRECATED: use build_channel_signal() for TP computation` comments to the old
`_calc_levels()` method body before removing it in a follow-up cleanup PR.

### Step 7 — Verification tests

```python
def test_volume_expansion_returns_false_when_below_threshold():
    from src.filters import check_volume_expansion
    volumes = [1000.0] * 10 + [800.0]   # Last candle is below average
    closes  = [100.0] * 11
    assert not check_volume_expansion(volumes, closes, lookback=9, multiplier=1.8)

def test_volume_expansion_returns_true_when_above():
    from src.filters import check_volume_expansion
    volumes = [1000.0] * 10 + [2500.0]   # Last candle is 2.5× average
    closes  = [100.0] * 11
    assert check_volume_expansion(volumes, closes, lookback=9, multiplier=1.8)

def test_scalp_channel_no_calc_levels_method():
    """After refactor, ScalpChannel should not have _calc_levels."""
    from src.channels.scalp import ScalpChannel
    ch = ScalpChannel()
    assert not hasattr(ch, "_calc_levels"), \
        "_calc_levels should be removed; TP is computed by build_channel_signal()"
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Lines of duplicate SL/TP code | ~30 lines across 3 paths | 0 (single source of truth) |
| Volume expansion implementations | 2 independent | 1 shared function with tests |
| Trailing desc synchronisation | Manual across 3+ files | Single helper function |
| Spread check consistency | Adaptive in Swing/Spot, non-adaptive in Scalp | Uniform adaptive across all |

---

## Dependencies

- **PR_01 through PR_06** should be merged first to ensure the refactored paths are
  compatible with the new filter and regime logic. Specifically, `check_spread_adaptive()`
  requires the `regime` argument which comes from PR_01.
