# PR-OPT-03 — Graduated OI Validation with Noise Threshold

**Priority:** P1  
**Estimated Signal Recovery:** +3–8% by eliminating spurious OI-based rejections  
**Dependencies:** None  
**Status:** ✅ IMPLEMENTED

---

## Objective

Prevent spurious hard-rejections from the OI validation layer when the Open Interest change is
below a meaningful noise threshold.  On Binance perpetuals, routine inter-candle OI fluctuations
of < 1% are common and do not indicate aggressive new positioning.  The previous implementation
invalidated ALL signals with any rising OI trend, regardless of magnitude.

---

## Problems Addressed

- **STGUSDT and CUSDT** signals invalidated during short sweeps when OI rises by < 1%.  These
  micro-fluctuations are market noise, not genuine short-selling pressure.
- `is_oi_invalidated()` in `src/order_flow.py` returned `True` for ANY `OITrend.RISING` regardless
  of magnitude — a 0.1% OI tick was treated identically to a 5% OI surge.
- `check_oi_gate()` in `src/oi_filter.py` had no notion of OI change magnitude for SQUEEZE and
  DISTRIBUTION patterns.

---

## Module / Strategy Affected

- `src/oi_filter.py` — `check_oi_gate()` function and new `OI_NOISE_THRESHOLD` constant
- `src/order_flow.py` — `is_oi_invalidated()` function and new `get_oi_change_pct()` method
- `src/detector.py` — updated call to `is_oi_invalidated()` to pass the actual OI change pct
- `tests/test_order_flow.py` — updated `TestIsOIInvalidated` test class

---

## Changes Made

### `src/oi_filter.py`

1. Added `OI_NOISE_THRESHOLD: float = 0.01` constant (1%).

2. Added `min_oi_change_pct` parameter to `check_oi_gate()` (default = `OI_NOISE_THRESHOLD`):
   - For SQUEEZE patterns (LONG direction): if `abs(oi_change_pct) < min_oi_change_pct`, the
     signal is allowed through with a debug log instead of a hard rejection.
   - Same logic for DISTRIBUTION patterns (SHORT direction).

### `src/order_flow.py`

1. Added `oi_change_pct: float = 0.0` parameter to `is_oi_invalidated()`.
   - When `oi_trend == OITrend.RISING`, the signal is only invalidated if
     `abs(oi_change_pct) >= 0.01` (1% threshold).
   - A 0.0 default means callers that don't have OI pct data default to "no invalidation".

2. Added `get_oi_change_pct(symbol, lookback=5)` method to `OrderFlowStore`:
   - Returns the fractional OI change over the last `lookback` snapshots.
   - Used by `detector.py` to pass the magnitude to `is_oi_invalidated()`.

### `src/detector.py`

Updated the `is_oi_invalidated()` call to pass the actual OI change pct:

```python
oi_change_pct = order_flow_store.get_oi_change_pct(symbol)
if is_oi_invalidated(oi_trend, primary_sweep.direction.value, oi_change_pct):
    result.oi_invalidated = True
```

---

## Expected Impact

| OI Change | Old Behavior | New Behavior |
|-----------|-------------|--------------|
| 0 – 0.5% | Hard reject | No action (below OI_CHANGE_THRESHOLD) |
| 0.5% – 1% | Hard reject | Allowed (below noise threshold) |
| 1% – 5% | Hard reject | Hard reject (significant OI rise) |
| > 5% | Hard reject | Hard reject |

---

## Rollback Procedure

1. Remove `OI_NOISE_THRESHOLD` from `src/oi_filter.py`.
2. Remove `min_oi_change_pct` parameter from `check_oi_gate()`.
3. Restore original `is_oi_invalidated()` signature (remove `oi_change_pct` parameter).
4. Remove `get_oi_change_pct()` from `OrderFlowStore`.
5. Revert `detector.py` to call `is_oi_invalidated(oi_trend, direction)`.
6. Run `python -m pytest tests/test_order_flow.py` to confirm.
