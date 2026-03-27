# PR-SIG-OPT-01 — Regime Soft-Gate Overhaul

**Priority:** P0 — Highest impact, directly unblocks suppressed SCALP signals  
**Estimated Signal Recovery:** +30–40% SCALP signals during low-volatility periods  
**Dependencies:** None  
**Relates To:** Extends PR-OPT-01 (Adaptive QUIET Regime) with graduated penalties and env-configurable thresholds  
**Status:** 📋 Planned

---

## Objective

Replace the remaining hard regime-blocking logic in the SCALP channel pipeline with a
**soft-gate confidence-penalty system**. Hard blocks discard signals entirely; soft gates
allow signals through with a confidence penalty, letting downstream consumers decide
whether to act. This restores signal visibility during QUIET/RANGING markets without
sacrificing signal quality.

---

## Problem Analysis

### Current State: `src/scanner/__init__.py` — Lines 143–192

#### 1. `_RANGING_ADX_SUPPRESS_THRESHOLD` hard-blocks SCALP in RANGING

```python
# Line 145 — current value
_RANGING_ADX_SUPPRESS_THRESHOLD: float = 15.0
```

In `_should_skip_channel()` (line ~1233):

```python
if (
    chan_name == "360_SCALP"
    and ctx.is_ranging
    and ctx.adx_val < _RANGING_ADX_SUPPRESS_THRESHOLD
):
    log.debug("Suppressing SCALP signal for {} (RANGING, ADX={:.1f})", symbol, ctx.adx_val)
    self._suppression_counters[f"ranging_low_adx:{chan_name}"] += 1
    return True   # ← hard block
```

Pairs like BANUSDT, SKRUSDT, SUSHIUSDT, WAVESUSDT (all observed with ADX 12–16) are
completely suppressed. A RANGE_FADE setup is perfectly valid at ADX=12 — the problem is
the threshold is too high and there is no graduated penalty path.

#### 2. `QUIET_SCALP_MIN_CONFIDENCE` acts as a secondary hard gate

```python
# src/scanner/__init__.py — line ~2122
if sig.confidence < QUIET_SCALP_MIN_CONFIDENCE:
    log.debug("QUIET_SCALP_BLOCK {} {} conf={:.1f} < min={:.1f}", ...)
    continue   # ← hard block even after soft penalty was applied
```

`QUIET_SCALP_MIN_CONFIDENCE` defaults to 72.0 (from `config/__init__.py` line 691).
This is fine as a floor, but pairs with confidence 68–71 that are legitimate
mean-reversion setups are silently discarded with no downstream visibility.

#### 3. `_REGIME_CHANNEL_INCOMPATIBLE` still blocks SCALP_VWAP in QUIET (acceptable)

```python
# Lines 171–175 — current state after PR-OPT-01
_REGIME_CHANNEL_INCOMPATIBLE: Dict[str, List[str]] = {
    "360_SCALP_VWAP": ["QUIET"],
    "360_SWING":      ["VOLATILE", "DIRTY_RANGE"],
    "360_SPOT":       ["DIRTY_RANGE"],
}
```

VWAP block in QUIET is correct — VWAP is volume-anchored and meaningless in thin
markets. Keep this. The focus here is on the RANGING hard-block and the confidence
floor.

#### 4. `_BB_WIDTH_QUIET_PCT` over-classifies pairs as QUIET

```python
# src/regime.py — line ~100
_BB_WIDTH_QUIET_PCT: float = 1.2
```

A BB width of 1.2% is very tight. Legitimate ranging/consolidating pairs frequently
trade at 1.0–1.4% width between trend legs. This causes ~30–45% of observations to
be classified as QUIET, then blocked.

---

## Required Changes

### Change 1 — Lower `_RANGING_ADX_SUPPRESS_THRESHOLD` and replace hard block with penalty

**File:** `src/scanner/__init__.py`

```python
# Before (line 145)
_RANGING_ADX_SUPPRESS_THRESHOLD: float = 15.0

# After
_RANGING_ADX_SUPPRESS_THRESHOLD: float = float(
    os.getenv("RANGING_ADX_SUPPRESS_THRESHOLD", "12.0")
)

# Add new constant near line 147 (aligns with config var REGIME_RANGING_PENALTY)
_RANGING_LOW_ADX_CONF_PENALTY: float = float(
    os.getenv("REGIME_RANGING_PENALTY", "5.0")
)
```

Replace the hard-block in `_should_skip_channel()` (line ~1233) with a penalty flag:

```python
# Before
if (
    chan_name == "360_SCALP"
    and ctx.is_ranging
    and ctx.adx_val < _RANGING_ADX_SUPPRESS_THRESHOLD
):
    ...
    return True

# After
if (
    chan_name == "360_SCALP"
    and ctx.is_ranging
    and ctx.adx_val < _RANGING_ADX_SUPPRESS_THRESHOLD
):
    # Soft-gate: mark as RANGING_ADJUSTED instead of hard-blocking.
    # Confidence penalty applied post-evaluation in _evaluate_channel().
    ctx.confidence_adjustments["RANGING_LOW_ADX"] = -_RANGING_LOW_ADX_CONF_PENALTY
    log.debug(
        "RANGING low-ADX soft penalty for {} {} (ADX={:.1f}, penalty={:.1f}pts)",
        symbol, chan_name, ctx.adx_val, _RANGING_LOW_ADX_CONF_PENALTY,
    )
```

Apply the penalty in the post-evaluation step where `sig.confidence` is assembled:

```python
# In _evaluate_channel() or the confidence aggregation block
for reason, delta in ctx.confidence_adjustments.items():
    sig.confidence += delta
    sig.tags.add(f"REGIME_ADJUSTED:{reason}")
```

### Change 2 — Lower `QUIET_SCALP_MIN_CONFIDENCE` from 72.0 to 68.0

**File:** `config/__init__.py` — line 692

```python
# Before
QUIET_SCALP_MIN_CONFIDENCE: float = float(
    os.getenv("QUIET_SCALP_MIN_CONFIDENCE", "72.0")
)

# After
QUIET_SCALP_MIN_CONFIDENCE: float = float(
    os.getenv("QUIET_SCALP_MIN_CONFIDENCE", "68.0")
)
```

**Rationale:** Signals at confidence 68–71 in QUIET are top-tier mean-reversion setups.
The existing `_SCALP_QUIET_REGIME_PENALTY = 1.8` multiplier already penalises these
significantly; reducing the floor by 4 points recovers ~15% of suppressed signals
without meaningfully increasing noise.

### Change 3 — Add `REGIME_QUIET_PENALTY` env var and graduated penalty constant

**File:** `config/__init__.py` — add after `QUIET_SCALP_MIN_CONFIDENCE`

```python
#: Confidence penalty applied to SCALP signals in QUIET regime.
#: Replaces the implicit penalty currently baked into _SCALP_QUIET_REGIME_PENALTY.
REGIME_QUIET_PENALTY: float = float(os.getenv("REGIME_QUIET_PENALTY", "8.0"))

#: Confidence penalty applied to SCALP signals in RANGING regime with ADX
#: below RANGING_ADX_SUPPRESS_THRESHOLD.
REGIME_RANGING_PENALTY: float = float(os.getenv("REGIME_RANGING_PENALTY", "5.0"))

#: ADX threshold below which SCALP signals receive a soft penalty in RANGING.
RANGING_ADX_SUPPRESS_THRESHOLD: float = float(
    os.getenv("RANGING_ADX_SUPPRESS_THRESHOLD", "12.0")
)
```

Import these in `src/scanner/__init__.py`:

```python
from config import (
    ...
    REGIME_QUIET_PENALTY,
    REGIME_RANGING_PENALTY,
    RANGING_ADX_SUPPRESS_THRESHOLD,
    ...
)
```

### Change 4 — Widen `_BB_WIDTH_QUIET_PCT` threshold in `src/regime.py`

**File:** `src/regime.py` — line ~100

```python
# Before
_BB_WIDTH_QUIET_PCT: float = 1.2

# After — fewer pairs classified as QUIET; 1.5 was too aggressive, 1.0 is too tight
_BB_WIDTH_QUIET_PCT: float = float(os.getenv("BB_WIDTH_QUIET_PCT", "1.0"))
```

**Rationale:** The `_BB_WIDTH_QUIET_PCT` was reduced from 1.5 → 1.2 in PR-OPT-01.
Reducing further to 1.0 ensures only genuinely compressed markets are classified
as QUIET, recovering pairs like SUSHIUSDT and STPTUSDT that oscillate at 1.1–1.3%
width while still having valid directional setups.

### Change 5 — Narrow `_ADX_RANGING_MAX` to reduce RANGING misclassification

**File:** `src/regime.py` — line ~99

```python
# Before
_ADX_RANGING_MAX: float = 20.0

# After
_ADX_RANGING_MAX: float = float(os.getenv("ADX_RANGING_MAX", "18.0"))
```

**Rationale:** Pairs with ADX 18–20 often have emerging directional momentum. Keeping
them in RANGING suppresses SCALP breakout setups. Narrowing to 18.0 reduces false
RANGING classifications for pairs like OMNIUSDT and WAVESUSDT.

---

## Signal Tagging

Signals that pass through via soft gates must be tagged so downstream consumers
(Telegram formatter, performance tracker, signal lifecycle monitor) can apply
appropriate weighting:

```python
# Tags added to sig.tags (Set[str]) when soft-gate penalties are applied
"QUIET_ADJUSTED"         # Signal passed QUIET soft-gate with -8pt confidence penalty
"RANGING_ADJUSTED"       # Signal passed RANGING low-ADX soft-gate with -5pt penalty
"REGIME_ADJUSTED:REASON" # Generic form used by _evaluate_channel() penalty applier
```

The Telegram formatter in `src/telegram_bot.py` should display a ⚠️ indicator when
these tags are present.

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| SCALP signals in QUIET | 0 (hard blocked below conf 72) | All ≥68 pass with QUIET_ADJUSTED tag |
| SCALP signals in RANGING (ADX < 15) | 0 (hard blocked) | All pass with RANGING_ADJUSTED tag |
| SCALP signals in RANGING (ADX 12–15) | 0 (still hard blocked at 15) | Pass with soft penalty after threshold→12 |
| False-QUIET classification rate | ~35–45% of observations | ~20–30% (BB threshold tightened to 1.0%) |
| Estimated signal frequency increase | baseline | +30–40% for SCALP channels |

---

## Implementation Notes

1. `ScanContext` (defined in `src/scanner/__init__.py`) needs a `confidence_adjustments`
   field: `confidence_adjustments: Dict[str, float] = field(default_factory=dict)`
2. The penalty application loop must run **after** channel evaluation but **before**
   the `QUIET_SCALP_MIN_CONFIDENCE` gate check to avoid double-applying penalties.
3. Keep `_SCALP_QUIET_REGIME_PENALTY = 1.8` multiplier — this is a separate mechanism
   that scales the **base penalty weight** during evaluation, not the final confidence.
4. Add `RANGING_ADX_SUPPRESS_THRESHOLD` to `.env.example` with documented defaults.

---

## Testing Criteria

```bash
# Run targeted tests
python -m pytest tests/test_regime_soft_penalty.py -v
python -m pytest tests/test_regime_filters.py -v
python -m pytest tests/test_scanner_indicator_compute.py -v

# Verify RANGING low-ADX signals now pass (previously all suppressed)
# Set ADX=13.5 (below old threshold=15, above new threshold=12) → expect RANGING_ADJUSTED tag
# Set ADX=11.0 (below new threshold=12) → expect hard block still applies

# Verify QUIET signals at conf=69 now pass (previously blocked at min=72)
# Set confidence=69, regime=QUIET → expect QUIET_ADJUSTED tag in signal

# Verify env vars work
RANGING_ADX_SUPPRESS_THRESHOLD=10.0 python -m pytest tests/ -k "test_regime" -v
QUIET_SCALP_MIN_CONFIDENCE=75.0 python -m pytest tests/ -k "test_regime" -v
```

## Verification of Impact

After deployment, monitor suppression logs for 24h:
```
grep "RANGING_ADJUSTED\|QUIET_ADJUSTED" trade_monitor.log | wc -l
# Expected: >50 occurrences per 24h for active market periods
```

Compare pre/post signal rates using `/stats` Telegram command. Expect SCALP channel
frequency to increase by 30–40% during sessions where BB width is 1.0–1.5%.
