# PR_07 — Dynamic SL/TP Based on ATR Percentile and Regime

**Branch:** `feature/pr07-dynamic-sl-tp`  
**Priority:** 7  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Replace the static SL/TP ratio system (fixed percentages in `config/__init__.py` and
`signal_params.py`) with a dynamic calculation that adjusts ratios based on:

1. **ATR percentile** (from PR_01 `RegimeContext.atr_percentile`): when volatility is in
   the top 80th percentile, widen both SL and TP proportionally to avoid premature stops
   and capture full moves; when in the bottom 20th percentile, tighten targets.

2. **Market regime**: TRENDING regimes use asymmetric TP scaling (TP3 boosted for
   trend-following), RANGING uses symmetric compressed targets, VOLATILE widens SL
   more aggressively, QUIET compresses all targets.

3. **Pair tier** (from PR_02 `PairProfile`): ALTCOIN pairs receive a wider SL multiplier
   to account for manipulation wicks; MAJOR pairs use tighter defaults.

The existing `build_channel_signal()` in `base.py` already has a `bb_width_pct` path for
volatility-adaptive ratios. This PR upgrades it to a full ATR-percentile + regime + pair
tier three-axis decision.

---

## Files to Change

| File | Change type |
|------|-------------|
| `src/channels/base.py` | Add `compute_dynamic_sl_tp_ratios()` helper; update `build_channel_signal()` signature |
| `src/channels/signal_params.py` | Add `DynamicRatioProfile` dataclass used by the new helper |
| `config/__init__.py` | Add `DYNAMIC_SL_TP_ENABLED` flag (default True) |
| `tests/test_channels.py` | Add tests for dynamic ratio computation at key ATR percentile breakpoints |

---

## Implementation Steps

### Step 1 — Add `compute_dynamic_sl_tp_ratios()` to `src/channels/base.py`

```python
def compute_dynamic_sl_tp_ratios(
    base_tp_ratios: list[float],
    base_sl_mult: float,
    atr_percentile: float,
    regime: str,
    pair_tier: str = "MIDCAP",
) -> tuple[float, list[float]]:
    """Return (sl_multiplier, tp_ratios) adjusted for volatility, regime, and pair tier.

    Parameters
    ----------
    base_tp_ratios:
        Default TP ratios from channel config (e.g. [0.5, 1.0, 1.5]).
    base_sl_mult:
        Default SL multiplier (1.0 = no scaling).
    atr_percentile:
        Rolling ATR percentile 0–100 (from RegimeContext).
    regime:
        Current market regime string.
    pair_tier:
        "MAJOR", "MIDCAP", or "ALTCOIN".

    Returns
    -------
    (sl_multiplier, tp_ratios)
        sl_multiplier: float to multiply the ATR-based SL distance.
        tp_ratios: list of adjusted TP ratios.
    """
    # --- Volatility-percentile SL adjustment ---
    if atr_percentile >= 80:
        vol_sl_adj = 1.3    # Widen SL in high-vol environment
        vol_tp_adj = 1.25   # Wider TP targets too
    elif atr_percentile <= 20:
        vol_sl_adj = 0.8    # Tighter SL in low-vol
        vol_tp_adj = 0.75
    else:
        vol_sl_adj = 1.0
        vol_tp_adj = 1.0

    # --- Regime SL/TP adjustments ---
    regime_upper = regime.upper() if regime else ""
    regime_sl = {
        "TRENDING_UP": 1.0, "TRENDING_DOWN": 1.0,
        "RANGING": 0.9,      "QUIET": 0.85,
        "VOLATILE": 1.4,
    }.get(regime_upper, 1.0)
    # TP scaling: in trending regimes, boost TP3 (the runner target) by 20%
    regime_tp = [1.0] * len(base_tp_ratios)
    if regime_upper in ("TRENDING_UP", "TRENDING_DOWN"):
        if len(regime_tp) >= 3:
            regime_tp[-1] = 1.2   # Boost only the runner TP
    elif regime_upper in ("RANGING", "QUIET"):
        regime_tp = [0.9] * len(base_tp_ratios)  # Compress all TPs
    elif regime_upper == "VOLATILE":
        regime_tp = [1.1] * len(base_tp_ratios)

    # --- Pair-tier SL widening ---
    tier_sl = {"MAJOR": 0.95, "MIDCAP": 1.0, "ALTCOIN": 1.20}.get(pair_tier, 1.0)

    # Combine all adjustments
    final_sl_mult = base_sl_mult * vol_sl_adj * regime_sl * tier_sl
    final_tp = [
        r * vol_tp_adj * regime_tp[i]
        for i, r in enumerate(base_tp_ratios)
    ]
    return final_sl_mult, final_tp
```

### Step 2 — Update `build_channel_signal()` signature

Add two new optional parameters:

```python
def build_channel_signal(
    ...
    atr_percentile: float = 50.0,          # From RegimeContext
    pair_tier: str = "MIDCAP",             # From PairProfile
) -> Optional[Signal]:
```

Inside the function body, replace the `bb_width_pct` conditional block:

```python
from config import DYNAMIC_SL_TP_ENABLED

if DYNAMIC_SL_TP_ENABLED:
    final_sl_mult, adj_ratios = compute_dynamic_sl_tp_ratios(
        base_ratios, params.sl_multiplier, atr_percentile, regime, pair_tier
    )
    sl_dist = sl_dist * final_sl_mult
else:
    adj_ratios = base_ratios

# Compute TP levels from adj_ratios (replaces the old bb_width_pct block)
if direction == Direction.LONG:
    tp1 = close + sl_dist * adj_ratios[0]
    tp2 = close + sl_dist * adj_ratios[1]
    tp3 = close + sl_dist * adj_ratios[2] if len(adj_ratios) > 2 else tp3
else:
    tp1 = close - sl_dist * adj_ratios[0]
    tp2 = close - sl_dist * adj_ratios[1]
    tp3 = close - sl_dist * adj_ratios[2] if len(adj_ratios) > 2 else tp3
```

### Step 3 — Propagate `atr_percentile` and `pair_tier` from scanner

In `src/scanner.py`, after computing the regime context and pair profile:

```python
regime_ctx = detect_regime(candles, indicators)
profile = classify_pair_tier(symbol, volume_24h_usd)

# Pass through to channel evaluate via smc_data
smc_data["atr_percentile"] = regime_ctx.atr_percentile
smc_data["pair_tier"] = profile.tier
```

In each channel's `build_channel_signal()` call, add:
```python
build_channel_signal(
    ...
    atr_percentile=smc_data.get("atr_percentile", 50.0),
    pair_tier=smc_data.get("pair_tier", "MIDCAP"),
)
```

### Step 4 — Add `DYNAMIC_SL_TP_ENABLED` flag to `config/__init__.py`

```python
DYNAMIC_SL_TP_ENABLED: bool = os.getenv("DYNAMIC_SL_TP_ENABLED", "true").lower() in (
    "true", "1", "yes"
)
```

Setting to `false` reverts to the static `signal_params.py` behaviour for safety.

### Step 5 — Tests

```python
def test_dynamic_sl_high_volatility_widens():
    from src.channels.base import compute_dynamic_sl_tp_ratios
    sl_mult, tp = compute_dynamic_sl_tp_ratios([0.5, 1.0, 1.5], 1.0, 85.0, "VOLATILE", "ALTCOIN")
    assert sl_mult > 1.3   # High vol + VOLATILE regime + ALTCOIN tier all widen SL
    assert tp[0] > 0.5     # TP also widened

def test_dynamic_sl_low_volatility_tightens():
    from src.channels.base import compute_dynamic_sl_tp_ratios
    sl_mult, tp = compute_dynamic_sl_tp_ratios([0.5, 1.0, 1.5], 1.0, 15.0, "QUIET", "MAJOR")
    assert sl_mult < 0.85

def test_trending_regime_boosts_tp3():
    from src.channels.base import compute_dynamic_sl_tp_ratios
    sl_mult, tp = compute_dynamic_sl_tp_ratios([0.5, 1.0, 2.0], 1.0, 50.0, "TRENDING_UP", "MIDCAP")
    assert tp[2] > 2.0   # Runner TP boosted
    assert tp[0] == 0.5  # TP1 unchanged
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Premature SL hits in VOLATILE regime | High (fixed SL) | Reduced by ~20% (1.4× SL widening) |
| TP3 hit rate in TRENDING regime | Low (static ratio) | Increased by ~10% (1.2× runner TP) |
| Overextended TP targets in QUIET regime | Occasional | Eliminated (0.75–0.9× compression) |
| Altcoin SL hit rate from manipulation wicks | High | Reduced (1.20× tier multiplier) |

---

## Dependencies

- **PR_01** — `RegimeContext.atr_percentile` required for volatility-percentile calculation.
- **PR_02** — `PairProfile.tier` required for pair-tier SL adjustment.
