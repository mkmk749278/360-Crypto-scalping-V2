# PR_03 — Adaptive EMA Thresholds

**Branch:** `feature/pr03-adaptive-ema-thresholds`  
**Priority:** 3  
**Effort estimate:** Small (1–2 days)

---

## Objective

Replace static EMA buffer zones (used in `swing.py` and `scalp.py`) with ATR-normalised
adaptive thresholds that scale per pair and per market regime. The current implementation
uses a fixed `_EMA200_BUFFER_PCT = 0.5%` in `swing.py` and fixed EMA fast/slow comparison
in `scalp.py`. On BTC (price ~$80 000) a 0.5% buffer means a $400 dead-zone. On a $0.05
altcoin the same 0.5% becomes $0.00025 — effectively no buffer at all.

This PR computes the EMA buffer dynamically as:

```
ema_buffer = max(min_buffer_pct, atr_pct × regime_multiplier)
```

Where:
- `min_buffer_pct` is a per-pair-tier floor (e.g., 0.1% for MAJOR, 0.3% for ALTCOIN)
- `atr_pct` is the ATR as a percentage of current price
- `regime_multiplier` scales the buffer by regime sensitivity (tighter in TRENDING, wider in VOLATILE)

---

## Files to Change

| File | Change type |
|------|-------------|
| `src/channels/swing.py` | Replace `_EMA200_BUFFER_PCT` constant with adaptive computation |
| `src/channels/scalp.py` | Adapt EMA9/EMA21 comparison to use ATR-normalised zone |
| `src/filters.py` | Add `check_ema_alignment_adaptive()` function |
| `tests/test_channels.py` | Add tests for adaptive buffer at different ATR/regime inputs |

---

## Implementation Steps

### Step 1 — Add `check_ema_alignment_adaptive()` to `src/filters.py`

```python
def check_ema_alignment_adaptive(
    ema_fast: float | None,
    ema_slow: float | None,
    direction: str,
    atr_val: float = 0.0,
    close: float = 0.0,
    regime: str = "",
    pair_tier: str = "MIDCAP",
) -> bool:
    """Return True when EMAs are meaningfully aligned with direction.

    Uses an ATR-normalised buffer zone to prevent signals near EMA crossover
    points where fast ≈ slow. The buffer adapts to pair volatility and regime.

    Parameters
    ----------
    ema_fast, ema_slow:
        Current EMA values. None → fail (no data).
    direction:
        "LONG" or "SHORT".
    atr_val:
        Current ATR in price units.
    close:
        Current price (for converting ATR to percentage).
    regime:
        Current market regime string.
    pair_tier:
        "MAJOR", "MIDCAP", or "ALTCOIN" (from PairProfile).
    """
    if ema_fast is None or ema_slow is None:
        return False

    # Compute ATR-normalised separation
    atr_pct = (atr_val / close * 100.0) if close > 0 and atr_val > 0 else 0.3

    # Tier-specific minimum buffer floors
    min_buffer = {"MAJOR": 0.10, "MIDCAP": 0.20, "ALTCOIN": 0.30}.get(pair_tier, 0.20)

    # Regime multipliers for the buffer
    regime_mult = {
        "TRENDING_UP": 0.8, "TRENDING_DOWN": 0.8,  # tighter buffer — trend is clear
        "RANGING": 1.2, "QUIET": 1.2,               # wider buffer — avoid whipsaw
        "VOLATILE": 1.5,                             # widest buffer — EMA crossovers are noisy
    }.get(regime.upper() if regime else "", 1.0)

    buffer_pct = max(min_buffer, atr_pct * regime_mult * 0.5)
    buffer_abs = close * buffer_pct / 100.0 if close > 0 else atr_val * 0.5

    ema_diff = ema_fast - ema_slow
    if direction == "LONG":
        return ema_diff >= buffer_abs   # fast must be meaningfully above slow
    if direction == "SHORT":
        return ema_diff <= -buffer_abs  # fast must be meaningfully below slow
    return False
```

### Step 2 — Update `swing.py` EMA200 buffer

Replace the static buffer constant:
```python
# Before:
_EMA200_BUFFER_PCT: float = 0.5

ema200_distance_pct = abs(close_h1 - ema200) / ema200 * 100.0
if ema200_distance_pct < _EMA200_BUFFER_PCT:
    return None
```

With dynamic computation:
```python
# After:
atr_pct = (atr_val / close * 100.0) if close > 0 and atr_val > 0 else 0.3
profile = smc_data.get("pair_profile")
pair_tier = profile.tier if profile else "MIDCAP"
tier_floors = {"MAJOR": 0.3, "MIDCAP": 0.5, "ALTCOIN": 0.7}
regime_mults = {
    "TRENDING_UP": 0.8, "TRENDING_DOWN": 0.8,
    "RANGING": 1.2, "QUIET": 1.2,
    "VOLATILE": 1.5,
}
regime_mult = regime_mults.get(regime.upper() if regime else "", 1.0)
ema200_buffer_pct = max(tier_floors.get(pair_tier, 0.5), atr_pct * regime_mult * 0.4)

ema200_distance_pct = abs(close_h1 - ema200) / ema200 * 100.0
if ema200_distance_pct < ema200_buffer_pct:
    return None
```

### Step 3 — Update `scalp.py` EMA alignment check

In `_evaluate_standard()`, replace:
```python
if not check_ema_alignment_regime(ema_fast, ema_slow, direction.value, regime=regime):
    return None
```

With:
```python
from src.filters import check_ema_alignment_adaptive

profile = smc_data.get("pair_profile")
pair_tier = profile.tier if profile else "MIDCAP"
if not check_ema_alignment_adaptive(
    ema_fast, ema_slow, direction.value,
    atr_val=atr_val, close=close,
    regime=regime, pair_tier=pair_tier,
):
    return None
```

### Step 4 — Deprecate `check_ema_alignment_regime()` (soft deprecation)

Add a deprecation comment to the existing function in `filters.py` noting that new code
should use `check_ema_alignment_adaptive()`. Do not remove it yet (backward compat).

### Step 5 — Tests

Add to `tests/test_channels.py`:

```python
def test_adaptive_ema_buffer_scales_with_atr():
    from src.filters import check_ema_alignment_adaptive
    # BTC-like: tight ATR, MAJOR tier, TRENDING_UP — buffer should be smaller
    assert check_ema_alignment_adaptive(101.0, 100.0, "LONG",
        atr_val=0.3, close=100.0, regime="TRENDING_UP", pair_tier="MAJOR")
    # Altcoin: high ATR, ALTCOIN tier — same absolute gap should fail the wider buffer
    assert not check_ema_alignment_adaptive(100.15, 100.0, "LONG",
        atr_val=1.0, close=100.0, regime="VOLATILE", pair_tier="ALTCOIN")
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| False signals near EMA crossover (BTC) | Moderate (0.5% buffer sometimes too wide) | Reduced — buffer scales down for low-ATR assets |
| False signals near EMA crossover (DOGE) | High (0.5% sometimes too narrow for 1.5% ATR asset) | Reduced — buffer scales up for high-ATR assets |
| Swing channel EMA200 rejection accuracy | Uniform, regime-blind | Regime-adaptive and pair-tier-aware |
| Signal frequency loss | Minimal | ~5% fewer spurious signals with ~2% more valid signals captured |

---

## Dependencies

- **PR_02** — Requires `PairProfile.tier` to be available in `smc_data["pair_profile"]`.
