# PR-OPT-02 — Relaxed Pair Quality Gates with Per-Channel Overrides

**Priority:** P1  
**Estimated Signal Recovery:** +5–10% by allowing wider-spread pairs on longer-hold channels  
**Dependencies:** None  
**Status:** ✅ IMPLEMENTED

---

## Objective

Replace the single global spread gate (`spread_pct <= 0.03`) in `assess_pair_quality()` with a
per-channel spread limit table and add `assess_pair_quality_for_channel()` as the channel-aware
variant.  Also lower the minimum 24h volume floor from $1,000,000 to $500,000 for non-SCALP
channels so valid lower-cap futures pairs are no longer excluded.

---

## Problems Addressed

- **KATUSDT and similar pairs** consistently fail the pair quality gate with "spread too wide"
  across SWING, SPOT and GEM channels — even though these strategies hold positions for hours
  to days and can absorb a slightly higher spread cost.
- A single threshold of 0.03% (3 bps) is appropriate for the SCALP channel but unnecessarily
  strict for SWING (0.05%), SPOT (0.06%) and GEM (0.08%).
- The $1,000,000 minimum volume floor excludes many valid lower-cap Binance perpetual pairs
  that are perfectly fine for SWING/SPOT/GEM strategies.

---

## Module / Strategy Affected

- `src/signal_quality.py` — `assess_pair_quality()` and new `assess_pair_quality_for_channel()`

---

## Changes Made

### `src/signal_quality.py`

1. **Relaxed global spread gate** in `assess_pair_quality()`:
   - Changed from `spread_pct <= 0.03` to `spread_pct <= 0.05`
   - This is the backward-compatible function used when no channel context is available.

2. **New `assess_pair_quality_for_channel()` function**:
   - Accepts `channel_name` parameter in addition to the existing quality inputs.
   - Applies per-channel spread limits from `_SPREAD_LIMIT_BY_CHANNEL`:

   ```python
   _SPREAD_LIMIT_BY_CHANNEL = {
       "360_SCALP":      0.025,  # Tightest — execution-sensitive
       "360_SCALP_FVG":  0.03,
       "360_SCALP_CVD":  0.03,
       "360_SCALP_OBI":  0.03,
       "360_SCALP_VWAP": 0.03,
       "360_SWING":      0.05,   # Wider allowed — longer holding period
       "360_SPOT":       0.06,   # Multi-day holds
       "360_GEM":        0.08,   # Gem/altcoin pairs have wider spreads
   }
   ```

   - Non-SCALP channels use `_MIN_VOLUME_NON_SCALP = $500,000` instead of $1,000,000.
   - SCALP channels retain the original $1,000,000 minimum volume floor.

---

## Expected Impact

| Channel | Old Spread Limit | New Spread Limit | Volume Floor Change |
|---------|-----------------|------------------|---------------------|
| 360_SCALP | 0.03% | 0.025% | $1M (unchanged) |
| 360_SCALP_FVG/CVD/OBI/VWAP | 0.03% | 0.03% | $1M (unchanged) |
| 360_SWING | 0.03% | 0.05% | $500K |
| 360_SPOT | 0.03% | 0.06% | $500K |
| 360_GEM | 0.03% | 0.08% | $500K |

Previously hard-blocked pairs like KATUSDT will now pass the SWING/SPOT/GEM quality gate.

---

## Backward Compatibility

The existing `assess_pair_quality()` signature is unchanged.  Only the hard-gate threshold moved
from 0.03 to 0.05.  All callers that already pass a spread ≤ 0.03 are unaffected.

The new `assess_pair_quality_for_channel()` is an additive function; no existing code paths are
modified.

---

## Rollback Procedure

1. Revert the threshold in `assess_pair_quality()` from `0.05` back to `0.03`.
2. Remove the `_SPREAD_LIMIT_BY_CHANNEL`, `_MIN_VOLUME_NON_SCALP`, and
   `assess_pair_quality_for_channel()` additions.
3. Run `python -m pytest tests/test_signal_quality.py` to confirm.
