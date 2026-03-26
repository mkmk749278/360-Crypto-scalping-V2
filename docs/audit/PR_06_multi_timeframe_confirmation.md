# PR_06 — Multi-Timeframe Confirmation

**Branch:** `feature/pr06-mtf-confirmation`  
**Priority:** 6  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Implement a mandatory multi-timeframe (MTF) confirmation gate that requires alignment
across at least two timeframes before a signal is emitted. The existing `src/mtf.py`
module provides a `pipeline gate` function returning `(bool, str)`. This PR upgrades it
from a soft-advisory tool to a configurable hard/soft gate wired into the signal pipeline.

**Rules per channel:**

| Channel | Primary TF | Confirmation TF | Condition |
|---------|-----------|----------------|-----------|
| SCALP (standard) | 5m | 1h | EMA alignment OR RSI non-extreme on 1h |
| SCALP (range fade) | 5m | 15m | RSI ≤ 40 (LONG) / ≥ 60 (SHORT) on 15m |
| SWING | 1h | 4h | EMA200 bias on 4h; ADX ≥ 20 on 4h |
| SPOT | 4h | 1d | EMA50 daily alignment (already partially implemented) |

The gate is **hard** for SCALP and SWING in TRENDING/VOLATILE regimes and **soft** (–10 pts
confidence penalty) in RANGING/QUIET regimes where single-TF signals are more self-contained.

---

## Files to Change

| File | Change type |
|------|-------------|
| `src/mtf.py` | Upgrade from advisory to configurable gate |
| `src/channels/scalp.py` | Call MTF gate in each evaluate path |
| `src/channels/swing.py` | Call MTF gate before signal emission |
| `src/scanner.py` | Pass higher-TF indicators to MTF gate |
| `tests/test_mtf.py` | Add tests for each channel-TF combination |

---

## Implementation Steps

### Step 1 — Upgrade `src/mtf.py`

Replace or extend the existing gate function with:

```python
"""Multi-timeframe confluence checks.

Each function returns (passes: bool, reason: str, confidence_adj: float).
"""
from __future__ import annotations
from typing import Optional


def check_mtf_ema_alignment(
    higher_tf_indicators: dict,
    direction: str,
    strict: bool = True,
) -> tuple[bool, str, float]:
    """Check that EMA alignment on a higher timeframe supports the trade direction.

    Parameters
    ----------
    higher_tf_indicators:
        Indicator dict for the confirmation timeframe (e.g. 1h for a 5m scalp).
    direction:
        "LONG" or "SHORT".
    strict:
        When True, return (False, ...) on failure; else apply –10 pts penalty.
    """
    ema_fast = higher_tf_indicators.get("ema9_last")
    ema_slow = higher_tf_indicators.get("ema21_last")
    ema200 = higher_tf_indicators.get("ema200_last")

    if ema_fast is None or ema_slow is None:
        return True, "mtf_ema_no_data", 0.0   # Fail open on missing data

    aligned = (ema_fast > ema_slow) if direction == "LONG" else (ema_fast < ema_slow)

    # If EMA200 available, add extra confirmation weight
    if ema200 is not None and ema_fast is not None:
        price_above_ema200 = ema_fast > ema200
        if direction == "LONG" and not price_above_ema200:
            aligned = False
        if direction == "SHORT" and price_above_ema200:
            aligned = False

    if aligned:
        return True, "mtf_ema_aligned", 0.0
    if strict:
        return False, "mtf_ema_opposed", 0.0
    return True, "mtf_ema_soft_fail", -10.0


def check_mtf_rsi(
    higher_tf_indicators: dict,
    direction: str,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> tuple[bool, str, float]:
    """Check RSI on higher timeframe is not in an extreme zone opposing the signal.

    Returns (True, ...) when RSI is in a non-extreme zone or data is missing.
    """
    rsi_val = higher_tf_indicators.get("rsi_last")
    if rsi_val is None:
        return True, "mtf_rsi_no_data", 0.0
    if direction == "LONG" and rsi_val >= overbought:
        return False, f"mtf_rsi_overbought_{rsi_val:.1f}", 0.0
    if direction == "SHORT" and rsi_val <= oversold:
        return False, f"mtf_rsi_oversold_{rsi_val:.1f}", 0.0
    return True, "mtf_rsi_ok", 0.0


def check_mtf_adx(
    higher_tf_indicators: dict,
    min_adx: float = 20.0,
    max_adx: float = 65.0,
) -> tuple[bool, str, float]:
    """Check ADX on higher timeframe is within [min_adx, max_adx]."""
    adx_val = higher_tf_indicators.get("adx_last")
    if adx_val is None:
        return True, "mtf_adx_no_data", 0.0
    if adx_val < min_adx:
        return False, f"mtf_adx_weak_{adx_val:.1f}", 0.0
    if adx_val > max_adx:
        return False, f"mtf_adx_extreme_{adx_val:.1f}", 0.0
    return True, "mtf_adx_ok", 0.0


def mtf_gate_scalp_standard(
    indicators_1h: dict,
    direction: str,
    regime: str = "",
) -> tuple[bool, str, float]:
    """MTF gate for the SCALP standard path (5m signal → 1h confirmation).

    Strict in TRENDING/VOLATILE; soft penalty in RANGING/QUIET.
    Passes if EMA alignment OR RSI non-extreme on 1h.
    """
    strict = regime.upper() in ("TRENDING_UP", "TRENDING_DOWN", "VOLATILE")
    ema_ok, ema_reason, ema_adj = check_mtf_ema_alignment(indicators_1h, direction, strict=False)
    rsi_ok, rsi_reason, _ = check_mtf_rsi(indicators_1h, direction)

    if ema_ok and rsi_ok:
        return True, f"{ema_reason}+{rsi_reason}", 0.0
    if not ema_ok and not rsi_ok:
        if strict:
            return False, f"{ema_reason}+{rsi_reason}", 0.0
        return True, "mtf_both_soft_fail", -10.0
    # One passes — partial confirmation
    return True, f"mtf_partial_{ema_reason}_{rsi_reason}", -5.0


def mtf_gate_scalp_range_fade(
    indicators_15m: dict,
    direction: str,
) -> tuple[bool, str, float]:
    """MTF gate for RANGE_FADE path (5m signal → 15m RSI confirmation)."""
    rsi_val = indicators_15m.get("rsi_last")
    if rsi_val is None:
        return True, "mtf_15m_rsi_no_data", 0.0
    if direction == "LONG" and rsi_val > 45.0:
        return False, f"mtf_15m_rsi_not_oversold_{rsi_val:.1f}", 0.0
    if direction == "SHORT" and rsi_val < 55.0:
        return False, f"mtf_15m_rsi_not_overbought_{rsi_val:.1f}", 0.0
    return True, "mtf_15m_rsi_ok", 0.0


def mtf_gate_swing(
    indicators_4h: dict,
    direction: str,
) -> tuple[bool, str, float]:
    """MTF gate for SWING (1h signal → 4h EMA + ADX confirmation)."""
    ema_ok, ema_reason, ema_adj = check_mtf_ema_alignment(indicators_4h, direction, strict=True)
    adx_ok, adx_reason, _ = check_mtf_adx(indicators_4h, min_adx=18.0, max_adx=70.0)
    if ema_ok and adx_ok:
        return True, f"{ema_reason}+{adx_reason}", 0.0
    if not ema_ok:
        return False, ema_reason, 0.0
    return False, adx_reason, 0.0
```

### Step 2 — Apply MTF gate in `ScalpChannel._evaluate_standard()`

```python
from src.mtf import mtf_gate_scalp_standard

indicators_1h = indicators.get("1h", {})
mtf_ok, mtf_reason, mtf_adj = mtf_gate_scalp_standard(indicators_1h, direction.value, regime)
if not mtf_ok:
    return None
```

After building signal, apply soft adjustment:
```python
if sig is not None and mtf_adj != 0.0:
    sig.confidence += mtf_adj
    sig.soft_gate_flags = (sig.soft_gate_flags + f",MTF:{mtf_reason}").lstrip(",")
```

### Step 3 — Apply MTF gate in `ScalpChannel._evaluate_range_fade()`

```python
from src.mtf import mtf_gate_scalp_range_fade

indicators_15m = indicators.get("15m", {})
mtf_ok, mtf_reason, _ = mtf_gate_scalp_range_fade(indicators_15m, direction.value)
if not mtf_ok:
    return None
```

### Step 4 — Apply MTF gate in `SwingChannel.evaluate()`

```python
from src.mtf import mtf_gate_swing

mtf_ok, mtf_reason, mtf_adj = mtf_gate_swing(ind_h4, direction.value)
if not mtf_ok:
    return None
if mtf_adj != 0.0 and sig is not None:
    sig.confidence += mtf_adj
```

### Step 5 — Update `Signal.mtf_score` field

In `src/scanner.py`, after signal construction, set:
```python
sig.mtf_score = 1.0 if mtf_ok else 0.5  # Binary pass or partial
```

### Step 6 — Tests (`tests/test_mtf.py`)

```python
def test_mtf_gate_scalp_fails_when_ema_opposed_in_trending():
    from src.mtf import mtf_gate_scalp_standard
    ind_1h = {"ema9_last": 98.0, "ema21_last": 100.0, "rsi_last": 72.0}
    ok, reason, adj = mtf_gate_scalp_standard(ind_1h, "LONG", "TRENDING_UP")
    assert not ok

def test_mtf_gate_swing_passes_with_aligned_4h():
    from src.mtf import mtf_gate_swing
    ind_4h = {"ema9_last": 102.0, "ema21_last": 100.0, "adx_last": 28.0}
    ok, reason, adj = mtf_gate_swing(ind_4h, "LONG")
    assert ok
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Counter-trend SCALP signals in TRENDING regime | Frequent | Eliminated by 1h EMA gate |
| False positives on RANGE_FADE in trending markets | Moderate | Reduced by ~30% via 15m RSI gate |
| SWING signals against higher-TF trend | Occasional | Eliminated by 4h EMA gate |
| Signal frequency reduction | Baseline | Estimated –10 to –15% (net quality improvement) |

---

## Dependencies

- **PR_01** — regime string used to determine gate strictness.
- **PR_02** — pair profile used for RSI threshold overrides.
