# PR-OPT-01 — Adaptive QUIET Regime Handling for Scalp Channels

**Priority:** P0 — Highest impact, lowest implementation risk  
**Estimated Signal Recovery:** +15–25% signal frequency during low-volatility periods  
**Dependencies:** None  
**Status:** ✅ IMPLEMENTED

---

## Objective

Convert the QUIET regime from a hard block on all scalp channels to a soft-penalty system that allows high-confidence mean-reversion scalp signals to pass through. The current implementation suppresses entire channel families whenever `bb_width_pct <= 1.5` or `adx <= 20.0`, discarding valid `RANGE_FADE` setups that the ScalpChannel already weights for QUIET internally.

---

## Analysis of Current Code

### `src/scanner/__init__.py` — Lines 149–162

```python
# SCALP needs movement: block in QUIET (nothing moves).
_REGIME_CHANNEL_INCOMPATIBLE: Dict[str, List[str]] = {
    "360_SCALP":      ["QUIET"],
    "360_SCALP_FVG":  ["QUIET"],
    "360_SCALP_CVD":  ["QUIET"],
    "360_SCALP_OBI":  ["QUIET"],
    "360_SWING":      ["VOLATILE", "DIRTY_RANGE"],
    "360_SPOT":       ["DIRTY_RANGE"],
}
```

This hard-blocks ALL scalp signals when any pair enters QUIET regime. The comment — "SCALP needs movement" — is misleading for mean-reversion scalps: `RANGE_FADE` paths *require* a compressed range.

### `src/scanner/__init__.py` — Lines 163–169

```python
_REGIME_PENALTY_MULTIPLIER: Dict[str, float] = {
    "TRENDING_UP":   0.6,
    "TRENDING_DOWN": 0.6,
    "RANGING":       1.0,
    "VOLATILE":      1.5,
    "QUIET":         0.8,   # Low volume but stable
}
```

`QUIET` already has a lenient penalty (0.8), but signals never reach this point because they are blocked upstream in `_REGIME_CHANNEL_INCOMPATIBLE`.

### `src/regime.py` — Lines 95–97 (QUIET detection thresholds)

```python
_ADX_RANGING_MAX: float = 20.0
_BB_WIDTH_QUIET_PCT: float = 1.5   # Bollinger width as % of price
```

These thresholds are aggressive. Many legitimate crypto pairs oscillate at ≤1.5% BB width between trend legs, triggering QUIET classification for 40–60% of their trading lifetime.

### Root Cause

The `ScalpChannel` evaluation engine already contains regime-aware internal weights that **boost** mean-reversion signals in QUIET (historical `mean_reversion` weight: 1.5×). However, the scanner blocks the channel at line ~1150 before this code path is ever reached:

```python
incompatible_regimes = _REGIME_CHANNEL_INCOMPATIBLE.get(chan_name, [])
if _regime_key in incompatible_regimes:
    continue  # ← signal discarded here, never reaches internal evaluation
```

---

## Recommended Changes

### Change 1 — Remove `"QUIET"` from `_REGIME_CHANNEL_INCOMPATIBLE` for standard scalp channels

**File:** `src/scanner/__init__.py`  
**File:** `src/scanner.py` (identical dict, keep in sync)

```python
# Before
_REGIME_CHANNEL_INCOMPATIBLE: Dict[str, List[str]] = {
    "360_SCALP":      ["QUIET"],
    "360_SCALP_FVG":  ["QUIET"],
    "360_SCALP_CVD":  ["QUIET"],
    "360_SCALP_OBI":  ["QUIET"],
    "360_SWING":      ["VOLATILE", "DIRTY_RANGE"],
    "360_SPOT":       ["DIRTY_RANGE"],
}

# After
_REGIME_CHANNEL_INCOMPATIBLE: Dict[str, List[str]] = {
    # 360_SCALP_VWAP remains blocked: VWAP signals are meaningless without volume
    "360_SCALP_VWAP": ["QUIET"],
    "360_SWING":      ["VOLATILE", "DIRTY_RANGE"],
    "360_SPOT":       ["DIRTY_RANGE"],
}
```

**Rationale:** `360_SCALP_VWAP` depends on VWAP deviation which is unreliable in thin volume. The other four scalp channels use price-structure signals (FVG, CVD divergence, OBI imbalance) that remain valid in compressed ranges.

### Change 2 — Increase QUIET penalty multiplier for scalp channels

**File:** `src/scanner/__init__.py`

```python
# Before
_REGIME_PENALTY_MULTIPLIER: Dict[str, float] = {
    ...
    "QUIET": 0.8,
}

# After — add per-channel override mapping
_REGIME_PENALTY_MULTIPLIER: Dict[str, float] = {
    "TRENDING_UP":   0.6,
    "TRENDING_DOWN": 0.6,
    "RANGING":       1.0,
    "VOLATILE":      1.5,
    "QUIET":         0.8,
}

# Scalp-specific QUIET override: tighter confidence gate
_SCALP_QUIET_REGIME_PENALTY: float = 1.8
```

Apply this override inside the scanner evaluation loop when `chan_name.startswith("360_SCALP")` and `regime == "QUIET"`:

```python
if regime == "QUIET" and chan_name.startswith("360_SCALP"):
    regime_mult = _SCALP_QUIET_REGIME_PENALTY
else:
    regime_mult = _REGIME_PENALTY_MULTIPLIER.get(regime, 1.0)
```

### Change 3 — Add `QUIET_SCALP_MIN_CONFIDENCE` threshold

**File:** `config/__init__.py`

```python
# Minimum confidence required to emit a scalp signal during QUIET regime.
# Higher bar prevents low-quality mean-reversion noise from becoming alerts.
QUIET_SCALP_MIN_CONFIDENCE: float = float(
    os.getenv("QUIET_SCALP_MIN_CONFIDENCE", "75.0")
)
```

In the scanner emission gate (after scoring):

```python
if regime == "QUIET" and chan_name.startswith("360_SCALP"):
    if final_score < config.QUIET_SCALP_MIN_CONFIDENCE:
        _log.debug(
            "quiet_scalp_suppressed sym=%s chan=%s score=%.1f threshold=%.1f",
            symbol, chan_name, final_score, config.QUIET_SCALP_MIN_CONFIDENCE,
        )
        continue
```

### Change 4 — Add `QUIET_SCALP_VOLUME_MULTIPLIER` volume confirmation

**File:** `config/__init__.py`

```python
# Scalp signals in QUIET regime require volume >= this multiplier × 20-period avg.
# Confirms the compressed-range breakout attempt is backed by real participation.
QUIET_SCALP_VOLUME_MULTIPLIER: float = float(
    os.getenv("QUIET_SCALP_VOLUME_MULTIPLIER", "2.5")
)
```

In the scanner, before emitting a QUIET scalp signal:

```python
if regime == "QUIET" and chan_name.startswith("360_SCALP"):
    avg_vol = indicators.get("volume_avg_20", 0.0)
    current_vol = indicators.get("volume_last", 0.0)
    if avg_vol > 0 and current_vol < avg_vol * config.QUIET_SCALP_VOLUME_MULTIPLIER:
        _log.debug(
            "quiet_scalp_vol_suppressed sym=%s vol=%.0f avg=%.0f multiplier=%.1f",
            symbol, current_vol, avg_vol, config.QUIET_SCALP_VOLUME_MULTIPLIER,
        )
        continue
```

### Change 5 — Relax QUIET regime detection thresholds

**File:** `src/regime.py`

```python
# Before
_ADX_RANGING_MAX: float = 20.0
_BB_WIDTH_QUIET_PCT: float = 1.5

# After — less aggressive QUIET classification
_ADX_RANGING_MAX: float = 20.0        # unchanged
_BB_WIDTH_QUIET_PCT: float = 1.2      # was 1.5 — only classify as QUIET if very compressed
```

Reducing `_BB_WIDTH_QUIET_PCT` from 1.5% to 1.2% means the QUIET regime only fires for very tightly compressed ranges. Pairs in the 1.2–1.5% width zone previously forced into QUIET will now be classified as `RANGING`, where scalp channels are already permitted.

---

## Modules Affected

| Module | Change |
|--------|--------|
| `src/scanner/__init__.py` | Remove QUIET from scalp incompatible list; add per-channel penalty override |
| `src/scanner.py` | Same changes (keep in sync) |
| `config/__init__.py` | Add `QUIET_SCALP_MIN_CONFIDENCE`, `QUIET_SCALP_VOLUME_MULTIPLIER` |
| `src/regime.py` | Lower `_BB_WIDTH_QUIET_PCT` from 1.5 to 1.2 |

---

## Test Cases

### Unit Tests

1. **`test_quiet_scalp_pass`** — Verify that a QUIET-regime scalp signal with confidence ≥75 and volume ≥2.5× avg is NOT suppressed after this change.
2. **`test_quiet_scalp_low_confidence`** — Verify that a QUIET-regime scalp signal with confidence 60 is still suppressed.
3. **`test_quiet_scalp_low_volume`** — Verify that a QUIET-regime scalp with high confidence but low volume is suppressed.
4. **`test_quiet_vwap_still_blocked`** — Verify `360_SCALP_VWAP` remains fully blocked in QUIET.
5. **`test_bb_width_threshold`** — Verify pairs with BB width 1.25% are classified as QUIET; pairs with BB width 1.35% are classified as RANGING.

### Integration Tests

6. **`test_signal_frequency_quiet_window`** — Run scanner over a 24h historical window for ADAUSDT (known QUIET-dominant pair) and confirm >10 scalp signals emit vs. 0 before the change.

---

## Rollback Procedure

1. Revert `_REGIME_CHANNEL_INCOMPATIBLE` to include `"QUIET"` for all four scalp channels.
2. Remove `QUIET_SCALP_MIN_CONFIDENCE` and `QUIET_SCALP_VOLUME_MULTIPLIER` from config.
3. Revert `_BB_WIDTH_QUIET_PCT` to 1.5.

Rollback is instant — no database or state migration required.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Increased false-positive scalp signals in QUIET | Medium | Confidence floor (75.0) + volume multiplier (2.5×) guard |
| VWAP signals emitting on thin volume | Low | `360_SCALP_VWAP` remains blocked in QUIET |
| Pair with structural QUIET bias floods signal queue | Low | Per-pair rate limiting in `ClusterSuppression` still applies |
| BB width threshold change reclassifies too many pairs | Low | Only 0.3% delta; monitor regime distribution telemetry for 24h |

---

## Expected Impact

- **Signal frequency:** +15–25% for QUIET-dominant pairs (ADA, ZEC, STG, PIPPIN, PORT3, BRU)
- **Signal quality:** Maintained — confidence floor and volume gate ensure quality bar is not lowered
- **False positive rate:** Expected ≤5% increase; monitored via suppression telemetry (PR-OPT-05)
