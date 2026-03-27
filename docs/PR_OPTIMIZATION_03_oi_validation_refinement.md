# PR-OPT-03 — OI Validation Refinement (Graduated Thresholds)

**Priority:** P1  
**Estimated Signal Recovery:** ~30% of previously hard-rejected signals recovered as soft-penalised  
**Dependencies:** None

---

## Objective

Replace the binary `is_oi_invalidated()` function with a graduated response system. Currently, **any** OI increase above 0.5% causes a hard rejection regardless of magnitude or funding rate context. This over-filters legitimate signals — a 0.6% OI rise during a short sweep is treated identically to a 6% aggressive short accumulation.

---

## Analysis of Current Code

### `src/order_flow.py` — Lines 55–180

```python
_OI_CHANGE_THRESHOLD_PCT: float = 0.5  # 0.5% OI change triggers classification

def classify_oi_trend(current_oi: float, prev_oi: float) -> OITrend:
    change_pct = ((current_oi - prev_oi) / prev_oi) * 100.0 if prev_oi else 0.0
    if change_pct <= -_OI_CHANGE_THRESHOLD_PCT:
        return OITrend.FALLING
    if change_pct >= _OI_CHANGE_THRESHOLD_PCT:
        return OITrend.RISING
    return OITrend.FLAT

def is_oi_invalidated(oi_trend: OITrend, signal_direction: str) -> bool:
    """Return True when rising OI contradicts the proposed signal direction."""
    return oi_trend == OITrend.RISING
```

**Problems:**

1. **Binary outcome** — any `OITrend.RISING` result causes full signal invalidation. A 0.5001% OI rise produces the same result as a 10% spike.
2. **No magnitude information** — `classify_oi_trend` returns only `RISING`/`FALLING`/`FLAT` but discards the actual percentage. `is_oi_invalidated` therefore cannot make a proportional decision.
3. **No funding rate cross-check** — if the funding rate confirms the signal direction, rising OI may actually support the trade (new positions entering in the signal direction).
4. **`signal_direction` parameter is unused** — the current implementation ignores direction entirely.

### `src/oi_filter.py` (if present)

```python
OI_CHANGE_THRESHOLD: float = 0.005  # 0.5% — mirrors order_flow.py
```

---

## Recommended Changes

### Change 1 — Extend `classify_oi_trend` to return magnitude

**File:** `src/order_flow.py`

```python
from dataclasses import dataclass
from enum import Enum

class OITrend(str, Enum):
    RISING  = "RISING"
    FALLING = "FALLING"
    FLAT    = "FLAT"

@dataclass(frozen=True)
class OITrendResult:
    """Rich OI trend result including magnitude for graduated response."""
    trend: OITrend
    change_pct: float          # Raw signed percentage change
    abs_change_pct: float      # Absolute value for threshold comparisons

def classify_oi_trend(current_oi: float, prev_oi: float) -> OITrendResult:
    """Classify OI trend and capture magnitude for downstream graduated logic."""
    if prev_oi <= 0:
        return OITrendResult(trend=OITrend.FLAT, change_pct=0.0, abs_change_pct=0.0)
    change_pct = ((current_oi - prev_oi) / prev_oi) * 100.0
    abs_change = abs(change_pct)
    if change_pct <= -_OI_CHANGE_THRESHOLD_PCT:
        trend = OITrend.FALLING
    elif change_pct >= _OI_CHANGE_THRESHOLD_PCT:
        trend = OITrend.RISING
    else:
        trend = OITrend.FLAT
    return OITrendResult(trend=trend, change_pct=change_pct, abs_change_pct=abs_change)
```

> **Backward compatibility:** Keep the original `classify_oi_trend` signature by making it return `OITrendResult` — callers that only check `.trend` continue to work.

### Change 2 — Replace `is_oi_invalidated` with graduated `evaluate_oi_impact`

**File:** `src/order_flow.py`

```python
# Graduated OI invalidation thresholds
_OI_SOFT_PENALTY_LOW_PCT:  float = 2.0   # 0.5–2.0%: soft penalty only
_OI_SOFT_PENALTY_HIGH_PCT: float = 5.0   # 2.0–5.0%: strong penalty + warning
_OI_HARD_REJECT_PCT:       float = 5.0   # >5.0%: hard reject (current behaviour for extremes)

# Direction-aware hard-reject threshold (new shorts/longs aggressively entering)
_OI_DIRECTIONAL_HARD_REJECT_PCT: float = 3.0

@dataclass(frozen=True)
class OIEvaluation:
    """Result of graduated OI evaluation."""
    invalidated: bool            # True → hard reject signal
    confidence_penalty: float    # Negative float subtracted from signal confidence
    reason: str                  # Human-readable reason for telemetry / logging
    oi_change_pct: float         # Raw change for metadata

def evaluate_oi_impact(
    oi_result: OITrendResult,
    signal_direction: str,
    funding_rate: Optional[float] = None,
) -> OIEvaluation:
    """
    Graduated OI evaluation replacing binary is_oi_invalidated().

    Rules:
      OI change 0.5–2.0%  → soft penalty (-5 confidence)
      OI change 2.0–5.0%  → strong penalty (-15 confidence) + warning
      OI change > 5.0%    → hard reject (extreme new-position accumulation)
      Any direction with OI > 3.0%  → hard reject (direction-aware gate)
      Funding rate confirms signal  → reduce penalty by 50%
    """
    if oi_result.trend != OITrend.RISING:
        return OIEvaluation(
            invalidated=False,
            confidence_penalty=0.0,
            reason="oi_neutral",
            oi_change_pct=oi_result.change_pct,
        )

    change = oi_result.abs_change_pct

    # Hard reject on extreme OI accumulation
    if change >= _OI_HARD_REJECT_PCT:
        return OIEvaluation(
            invalidated=True,
            confidence_penalty=0.0,
            reason=f"oi_hard_reject change={change:.2f}%",
            oi_change_pct=oi_result.change_pct,
        )

    # Direction-aware hard reject (aggressive new position build > 3%)
    if change >= _OI_DIRECTIONAL_HARD_REJECT_PCT:
        return OIEvaluation(
            invalidated=True,
            confidence_penalty=0.0,
            reason=f"oi_directional_hard_reject dir={signal_direction} change={change:.2f}%",
            oi_change_pct=oi_result.change_pct,
        )

    # Determine base penalty
    if change >= 2.0:
        base_penalty = -15.0
        reason = f"oi_strong_penalty change={change:.2f}%"
    else:
        base_penalty = -5.0
        reason = f"oi_soft_penalty change={change:.2f}%"

    # Funding rate cross-check: if funding confirms signal direction, halve penalty
    if funding_rate is not None:
        funding_confirms = (
            (signal_direction == "LONG"  and funding_rate < 0) or
            (signal_direction == "SHORT" and funding_rate > 0)
        )
        if funding_confirms:
            base_penalty *= 0.5
            reason += " (funding_confirmed, penalty_halved)"

    return OIEvaluation(
        invalidated=False,
        confidence_penalty=base_penalty,
        reason=reason,
        oi_change_pct=oi_result.change_pct,
    )


# Preserve backward-compatible shim for callers that still use is_oi_invalidated
def is_oi_invalidated(oi_trend: OITrend, signal_direction: str) -> bool:
    """Deprecated shim — prefer evaluate_oi_impact() for graduated response."""
    if isinstance(oi_trend, OITrendResult):
        result = evaluate_oi_impact(oi_trend, signal_direction)
        return result.invalidated
    return oi_trend == OITrend.RISING
```

### Change 3 — Attach `oi_invalidation_reason` to signal metadata

**File:** `src/scanner/__init__.py` and `src/scanner.py`

Where OI invalidation is currently checked:

```python
# Before
if is_oi_invalidated(oi_trend, direction):
    _log.debug("oi_invalidated sym=%s dir=%s", symbol, direction)
    continue

# After
oi_eval = evaluate_oi_impact(oi_result, direction, funding_rate=indicators.get("funding_rate"))
if oi_eval.invalidated:
    _log.info(
        "oi_hard_rejected sym=%s dir=%s reason=%s change_pct=%.2f",
        symbol, direction, oi_eval.reason, oi_eval.oi_change_pct,
    )
    continue

# Apply soft penalty to confidence
if oi_eval.confidence_penalty < 0:
    final_confidence += oi_eval.confidence_penalty
    _log.debug(
        "oi_penalty_applied sym=%s penalty=%.1f new_conf=%.1f reason=%s",
        symbol, oi_eval.confidence_penalty, final_confidence, oi_eval.reason,
    )

# Attach to signal metadata for post-analysis
signal_metadata["oi_invalidation_reason"] = oi_eval.reason
signal_metadata["oi_change_pct"] = round(oi_eval.oi_change_pct, 3)
```

---

## Modules Affected

| Module | Change |
|--------|--------|
| `src/order_flow.py` | Add `OITrendResult`, `OIEvaluation` dataclasses; new `evaluate_oi_impact()`; shim for `is_oi_invalidated` |
| `src/oi_filter.py` | Update to use `OITrendResult` if it calls `classify_oi_trend` |
| `src/scanner/__init__.py` | Replace `is_oi_invalidated` call with `evaluate_oi_impact`; apply confidence penalty |
| `src/scanner.py` | Same as above |

---

## Test Cases

### Unit Tests

1. **`test_oi_flat_no_penalty`** — OI change 0.3% → `OITrend.FLAT`, no penalty, not invalidated.
2. **`test_oi_soft_penalty`** — OI change 1.0% rising → not invalidated, penalty = -5.
3. **`test_oi_strong_penalty`** — OI change 3.0% rising, below directional hard-reject → not invalidated, penalty = -15.
4. **`test_oi_directional_hard_reject`** — OI change 3.5% rising → hard reject.
5. **`test_oi_extreme_hard_reject`** — OI change 7.0% rising → hard reject regardless of direction.
6. **`test_oi_funding_halves_penalty`** — OI 1.5% rising, SHORT signal, funding_rate = +0.02 (confirms SHORT) → penalty = -2.5 (halved).
7. **`test_oi_funding_no_confirm`** — OI 1.5% rising, LONG signal, funding_rate = +0.02 (does NOT confirm LONG) → penalty = -5.
8. **`test_backward_compat_shim`** — `is_oi_invalidated(OITrend.RISING, "LONG")` still returns `True`.
9. **`test_oi_change_pct_in_metadata`** — After evaluation, `signal_metadata["oi_change_pct"]` is populated.

### Integration Tests

10. **`test_stgusdt_oi_recovery`** — STGUSDT signal with 0.8% OI rise is recovered with -5 confidence penalty instead of being hard-rejected.
11. **`test_cusdt_oi_recovery`** — CUSDT signal with 1.2% OI rise is recovered with -5 penalty.

---

## Rollback Procedure

1. Remove `OITrendResult`, `OIEvaluation`, `evaluate_oi_impact` from `src/order_flow.py`.
2. Restore the original single-line `is_oi_invalidated` function.
3. Restore `classify_oi_trend` to return `OITrend` enum directly.
4. Remove confidence penalty application from scanner loop.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Recovering OI-penalised signals increases losses in adverse OI conditions | Medium | Hard-reject still triggers at >3% directional and >5% absolute |
| Funding rate data unavailable for some pairs | Medium | `funding_rate=None` is handled — penalty not halved when data absent |
| Backward-compat break for callers using `OITrend` enum directly | Low | Shim function `is_oi_invalidated` preserved with identical signature |
| Signal metadata fields cause downstream serialisation errors | Low | Optional fields in metadata dict; serialiser must handle extra keys |

---

## Expected Impact

- **~30% of previously hard-rejected signals** (STGUSDT, CUSDT type patterns) recovered
- **Confidence reduction** on OI-penalised signals instead of silent discard
- **OI reason field** in signal metadata enables post-analysis of which pairs trigger OI penalties
