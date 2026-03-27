# PR-SIG-OPT-02 — Pair Quality Gates: Fully Channel-Adaptive with Tiered Thresholds

**Priority:** P1 — High impact on SWING/SPOT/GEM channel signal recovery  
**Estimated Signal Recovery:** +15–25% for SWING/SPOT/GEM; altcoin pairs like ALICEUSDT, IOTXUSDT re-enabled  
**Dependencies:** None (standalone change to `src/signal_quality.py`)  
**Relates To:** Extends PR-OPT-02 (Dynamic Pair Quality Gates) — adds per-channel composite score thresholds and regime-aware spread relaxation  
**Status:** 📋 Planned

---

## Objective

Make the pair quality gate fully channel-adaptive by introducing **per-channel composite
score thresholds** and **per-channel volume floors**, replacing the current universal
`total >= 58` gate that applies identically to SCALP and GEM channels. Additionally,
add regime-aware spread relaxation so that pairs in VOLATILE regime (where spreads
temporarily widen) are not permanently blocked.

---

## Problem Analysis

### Current State: `src/signal_quality.py` — Lines 297–407

#### 1. Universal composite score threshold of 58 blocks low-cap altcoins

Both `assess_pair_quality()` and `assess_pair_quality_for_channel()` use:

```python
# Lines 297 and 400
passed = total >= 58 and spread_pct <= spread_limit and volume_24h >= min_volume
```

The composite score `total` is computed as:

```python
total = spread_score * 0.3 + volume_score * 0.3 + volatility_score * 0.2 + noise_score * 0.2
```

Where `volume_score = (volume_24h / 15_000_000.0) * 100.0`. A pair with $500K daily
volume scores `volume_score = 3.3`, dragging the total well below 58 even with perfect
spread (100), volatility (100), and noise (100) scores:

```
total = 100*0.3 + 3.3*0.3 + 100*0.2 + 100*0.2 = 30 + 1 + 20 + 20 = 71 → PASSES at $500K
```

Wait — actually with $500K the volume score = 3.3, so:
```
total = 100*0.3 + 3.3*0.3 + 100*0.2 + 100*0.2 = 30 + 1 + 20 + 20 = 71
```

But the volume gate `volume_24h >= 1_000_000` (for SCALP) blocks it outright regardless
of score. For GEM channels, `_MIN_VOLUME_NON_SCALP = 500_000` (line 335), but the
`total >= 58` gate with poor spread (e.g. 0.06) gives:

```
spread_score = 100 - (0.06/0.02)*100 = 100 - 300 = 0 (clamped to 0)
total = 0*0.3 + 3.3*0.3 + 100*0.2 + 100*0.2 = 0 + 1 + 20 + 20 = 41 → FAILS
```

Pairs like ALICEUSDT ($300K volume, 0.07% spread) fail both the score threshold and
volume floor for GEM. But GEM is a portfolio channel for altcoins — a $300K volume pair
with reasonable volatility is perfectly valid.

#### 2. Fixed `_MIN_VOLUME_NON_SCALP = 500_000` too high for GEM altcoins

```python
# Line 335
_MIN_VOLUME_NON_SCALP: float = 500_000.0
```

GEM candidates (KATUSDT, BSBUSDT, TSTUSDT, DMCUSDT) frequently trade $50K–$250K/day.
The $500K floor, combined with the `total >= 58` gate, creates a near-impenetrable wall.

#### 3. No regime-aware spread relaxation

During VOLATILE regime, bid-ask spreads temporarily widen by 1.5–3×. A pair with a
normal spread of 0.03% may temporarily show 0.07% during high-volatility events,
triggering a quality gate failure that persists until the next scan cycle.

#### 4. Quality gate failures logged at DEBUG level

```python
# src/scanner/__init__.py — line ~1173
log.debug(
    "Skipping {} {} – pair quality gate failed: {}",
    symbol, chan_name, chan_quality.reason,
)
```

Failures are invisible without setting log level to DEBUG. Operators cannot diagnose
which pairs are being blocked and why without enabling verbose logging.

---

## Required Changes

### Change 1 — Add per-channel composite score thresholds

**File:** `src/signal_quality.py` — add after line 335

```python
# Per-channel minimum composite score thresholds.
# SCALP requires the highest quality (execution-sensitive, tight risk/reward).
# GEM allows the lowest quality since these are speculative portfolio plays
# evaluated on longer timeframes where short-term noise matters less.
_MIN_COMPOSITE_SCORE_BY_CHANNEL: Dict[str, float] = {
    "360_SCALP":      58.0,   # Current default — unchanged for SCALP
    "360_SCALP_FVG":  58.0,
    "360_SCALP_CVD":  58.0,
    "360_SCALP_OBI":  58.0,
    "360_SCALP_VWAP": 58.0,
    "360_SWING":      50.0,   # Wider tolerance — longer hold, spread cost lower
    "360_SPOT":       45.0,   # Multi-day hold — short-term noise irrelevant
    "360_GEM":        40.0,   # Speculative altcoins — low threshold by design
}

# Per-channel minimum 24h volume floors (USD).
_MIN_VOLUME_BY_CHANNEL: Dict[str, float] = {
    "360_SCALP":      1_000_000.0,   # Must have deep liquidity for tight scalps
    "360_SCALP_FVG":  1_000_000.0,
    "360_SCALP_CVD":  1_000_000.0,
    "360_SCALP_OBI":  1_000_000.0,
    "360_SCALP_VWAP": 1_000_000.0,
    "360_SWING":        500_000.0,   # Half the scalp floor — swing can absorb slippage
    "360_SPOT":         250_000.0,   # Spot portfolio — lower liquidity acceptable
    "360_GEM":          100_000.0,   # Altcoin gems — micro-cap discovery
}
```

**File:** `config/__init__.py` — add configurable overrides

```python
# Per-channel pair quality thresholds (overridable via env vars)
PAIR_QUALITY_THRESHOLD_SCALP: float = float(os.getenv("PAIR_QUALITY_THRESHOLD_SCALP", "58.0"))
PAIR_QUALITY_THRESHOLD_SWING: float = float(os.getenv("PAIR_QUALITY_THRESHOLD_SWING", "50.0"))
PAIR_QUALITY_THRESHOLD_SPOT:  float = float(os.getenv("PAIR_QUALITY_THRESHOLD_SPOT",  "45.0"))
PAIR_QUALITY_THRESHOLD_GEM:   float = float(os.getenv("PAIR_QUALITY_THRESHOLD_GEM",   "40.0"))

PAIR_QUALITY_VOLUME_FLOOR_SWING: float = float(os.getenv("PAIR_QUALITY_VOLUME_FLOOR_SWING", "500000.0"))
PAIR_QUALITY_VOLUME_FLOOR_SPOT:  float = float(os.getenv("PAIR_QUALITY_VOLUME_FLOOR_SPOT",  "250000.0"))
PAIR_QUALITY_VOLUME_FLOOR_GEM:   float = float(os.getenv("PAIR_QUALITY_VOLUME_FLOOR_GEM",   "100000.0"))
```

### Change 2 — Update `assess_pair_quality_for_channel()` to use channel thresholds

**File:** `src/signal_quality.py` — modify `assess_pair_quality_for_channel()` (lines 396–410)

```python
# Before
spread_limit = _SPREAD_LIMIT_BY_CHANNEL.get(channel_name, 0.05)
is_scalp = channel_name.startswith("360_SCALP")
min_volume = 1_000_000.0 if is_scalp else _MIN_VOLUME_NON_SCALP

passed = total >= 58 and spread_pct <= spread_limit and volume_24h >= min_volume

# After
spread_limit = _SPREAD_LIMIT_BY_CHANNEL.get(channel_name, 0.05)
min_composite = _MIN_COMPOSITE_SCORE_BY_CHANNEL.get(channel_name, 58.0)
min_volume = _MIN_VOLUME_BY_CHANNEL.get(channel_name, 500_000.0)

# Regime-aware spread relaxation: widen spread tolerance by 30% in VOLATILE regime
# to avoid blocking valid pairs during temporary volatility spikes.
effective_spread_limit = spread_limit
if current_regime == "VOLATILE":
    effective_spread_limit = spread_limit * 1.3

passed = (
    total >= min_composite
    and spread_pct <= effective_spread_limit
    and volume_24h >= min_volume
)
```

The `current_regime` parameter must be added to the function signature:

```python
def assess_pair_quality_for_channel(
    volume_24h: float,
    spread_pct: float,
    indicators: Dict[str, Any],
    candles: Optional[dict],
    channel_name: str,
    current_regime: str = "RANGING",   # ← new parameter
) -> PairQualityAssessment:
```

Update the call site in `src/scanner/__init__.py` (line ~1157):

```python
chan_quality = assess_pair_quality_for_channel(
    volume_24h=_vol,
    spread_pct=ctx.spread_pct,
    indicators=_regime_ind,
    candles=_regime_candles,
    channel_name=chan_name,
    current_regime=ctx.regime_result.regime.value,   # ← pass current regime
)
```

### Change 3 — Elevate quality gate failures to INFO level in scanner

**File:** `src/scanner/__init__.py` — line ~1173

```python
# Before
log.debug(
    "Skipping {} {} – pair quality gate failed: {}",
    symbol, chan_name, chan_quality.reason,
)

# After
log.info(
    "Quality gate FAIL: {} {} | score={:.1f} spread={:.4f} vol={:.0f} reason={}",
    symbol,
    chan_name,
    chan_quality.score,
    ctx.spread_pct,
    _vol,
    chan_quality.reason,
)
```

Include the full component breakdown from the `PairQualityAssessment` fields:
`spread_score`, `volatility_score`, `noise_score`.

---

## Example: How ALICEUSDT Would Pass After This Change

Assume ALICEUSDT: `volume_24h=$350K`, `spread_pct=0.07`, `atr_pct=1.8`, `wickiness=1.2`

**Score computation:**
- `spread_score = max(0, 100 - (0.07/0.02)*100) = max(0, -250) = 0`  (wide spread)
- `volume_score = (350_000/15_000_000)*100 = 2.3`
- `volatility_score = 100.0` (ATR in valid range 0.15–3.5%)
- `noise_score = max(0, 100 - (1.2-1.0)*35) = 93.0`
- `total = 0*0.3 + 2.3*0.3 + 100*0.2 + 93*0.2 = 0 + 0.7 + 20 + 18.6 = 39.3`

**Current behaviour (all channels blocked):**
- SCALP: `total(39.3) < 58` → FAIL | SWING: `total(39.3) < 58` → FAIL | GEM: `total(39.3) < 58` → FAIL

**After this change (GEM channel):**
- GEM threshold: `total >= 40.0` — borderline at 39.3 with wickiness=1.2
- With cleaner price action (`wickiness=1.0`): `noise_score = 100`, `total = 0.7 + 20 + 20 = 40.7 ≥ 40` → **PASSES** ✅
- GEM volume floor: $100K → $350K passes ✅
- GEM spread limit: 0.08 → 0.07 passes ✅

---

## Expected Impact

| Channel | Pairs Recovered | Key Examples |
|---------|-----------------|--------------|
| SCALP | ~0 (thresholds unchanged) | — |
| SWING | ~12–18% more pairs pass | ALICEUSDT, IOTXUSDT (if spread improves) |
| SPOT | ~20–25% more pairs pass | DMCUSDT, SOPHUSDT |
| GEM | ~40–50% more pairs pass | KATUSDT, BSBUSDT, TSTUSDT, ALICEUSDT |

---

## Testing Criteria

```bash
# Run targeted tests
python -m pytest tests/test_signal_quality.py -v
python -m pytest tests/test_signal_quality_improvements.py -v

# Validate per-channel thresholds
python -c "
from src.signal_quality import assess_pair_quality_for_channel
# GEM with low volume should pass
result = assess_pair_quality_for_channel(
    volume_24h=150_000, spread_pct=0.07,
    indicators={'atr_last': 0.003}, candles=None,
    channel_name='360_GEM', current_regime='RANGING'
)
assert result.passed, f'Expected PASS but got: {result.reason}'
print('GEM low-volume test: PASS')

# SCALP with low volume should still fail
result = assess_pair_quality_for_channel(
    volume_24h=150_000, spread_pct=0.02,
    indicators={'atr_last': 0.003}, candles=None,
    channel_name='360_SCALP', current_regime='RANGING'
)
assert not result.passed, 'Expected FAIL but SCALP low-volume passed'
print('SCALP low-volume test: correctly FAILS')
"

# Validate regime-aware spread relaxation
python -c "
from src.signal_quality import assess_pair_quality_for_channel
# SWING with spread=0.065 in VOLATILE should pass (0.05*1.3=0.065)
result = assess_pair_quality_for_channel(
    volume_24h=600_000, spread_pct=0.065,
    indicators={'atr_last': 0.003}, candles=None,
    channel_name='360_SWING', current_regime='VOLATILE'
)
print(f'SWING VOLATILE spread relaxation: passed={result.passed}')
"

# Env var overrides
PAIR_QUALITY_THRESHOLD_GEM=35.0 python -m pytest tests/test_signal_quality.py -v
```
