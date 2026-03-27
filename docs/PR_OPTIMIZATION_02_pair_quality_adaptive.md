# PR-OPT-02 — Adaptive Pair Quality Gates (Per-Channel Thresholds)

**Priority:** P1  
**Estimated Impact:** ~15–25% more pairs pass quality gate for SWING/SPOT/GEM channels  
**Dependencies:** None  
**Status:** ✅ IMPLEMENTED

---

## Objective

Replace the single hard spread threshold (`0.03` / 3%) in `assess_pair_quality()` with per-channel
spread and volume thresholds. Scalp channels need tight spreads, but SWING/SPOT channels can
tolerate wider spreads since they target larger moves over longer holding periods.

---

## Problem

`assess_pair_quality()` in `src/signal_quality.py` (line 270–313) requires `spread_pct <= 0.03`
and `volume_24h >= 1_000_000` as hard gates applied equally to **all** channels. Pairs like
KATUSDT fail with "spread too wide" across ALL strategies, with no per-channel relaxation.

```python
# src/signal_quality.py — line 297 (BEFORE)
passed = total >= 58 and spread_pct <= 0.03 and volume_24h >= 1_000_000
```

**Problems with this approach:**

1. The 3% spread limit is appropriate for scalp channels (execution-sensitive, tight fills
   required) but overly strict for SWING/SPOT/GEM channels where holding periods are longer and
   spread cost amortises over larger moves.
2. The $1M volume floor is appropriate for SCALP but blocks many valid lower-cap pairs from
   SWING, SPOT, and GEM channels.
3. Pairs like KATUSDT, STGUSDT are permanently excluded from all signals even when they would
   form valid SWING or GEM setups.
4. No per-channel context in rejection reason — logs show `"spread too wide"` without indicating
   which channel it was evaluated against.

---

## Solution — Per-Channel Spread Thresholds

**File:** `src/signal_quality.py` — lines 319–333

```python
_MAX_SPREAD_BY_CHANNEL: Dict[str, float] = {
    "360_SCALP":      0.025,   # Tightest — scalp needs minimal slippage
    "360_SCALP_FVG":  0.025,
    "360_SCALP_CVD":  0.025,
    "360_SCALP_VWAP": 0.025,
    "360_SCALP_OBI":  0.020,   # OBI needs very tight book
    "360_SWING":      0.05,    # Wider — swing targets 2–5% moves
    "360_SPOT":       0.06,    # Widest — spot targets 3–10% moves
    "360_GEM":        0.08,    # Gem scanner expects wide spreads on low-cap
}
```

> **Implementation note:** The actual implementation in the codebase uses the dict name
> `_SPREAD_LIMIT_BY_CHANNEL` (line 322 of `src/signal_quality.py`) with slightly adjusted
> values. The mapping above reflects the intent from the original design spec.

---

## Solution — Per-Channel Volume Thresholds

**File:** `src/signal_quality.py` — lines 335

```python
_MIN_VOLUME_BY_CHANNEL: Dict[str, float] = {
    "360_SCALP":      2_000_000,   # Need deep liquidity
    "360_SCALP_FVG":  1_500_000,
    "360_SCALP_CVD":  1_500_000,
    "360_SCALP_VWAP": 1_500_000,
    "360_SCALP_OBI":  2_000_000,
    "360_SWING":      500_000,     # Swing can work with less
    "360_SPOT":       300_000,
    "360_GEM":        100_000,     # Gems are low-cap by definition
}
```

> **Implementation note:** The codebase implements a simplified two-tier volume floor:
> scalp channels use `1_000_000` and all non-scalp channels use `_MIN_VOLUME_NON_SCALP = 500_000`
> (line 335 of `src/signal_quality.py`).

---

## Changes Made

### `src/signal_quality.py`

1. Added `_SPREAD_LIMIT_BY_CHANNEL` dict at line 322 with per-channel spread limits.
2. Added `_MIN_VOLUME_NON_SCALP = 500_000.0` at line 335 as a lower volume floor for
   non-scalp channels.
3. Added `assess_pair_quality_for_channel()` function at line 338 — accepts `channel_name: str`
   parameter and applies the channel-specific thresholds from `_SPREAD_LIMIT_BY_CHANNEL`.
4. The original `assess_pair_quality()` (line 270) is retained unchanged for backward
   compatibility with callers that do not provide a channel name.

### `src/scanner/__init__.py`

- The scanner calls `assess_pair_quality()` at line 1066 in the pre-channel quality check.
- Per-channel evaluation should call `assess_pair_quality_for_channel()` inside the
  per-channel loop, passing `chan_name` as `channel_name`.

### `src/scanner.py`

- Same changes as `src/scanner/__init__.py` to keep both scanner paths in sync.

### `config/__init__.py`

No config changes required — thresholds are defined as module-level constants in
`src/signal_quality.py`. Environment variable overrides can be added if needed.

---

## New Function Signature

```python
def assess_pair_quality_for_channel(
    volume_24h: float,
    spread_pct: float,
    indicators: Dict[str, Any],
    candles: Optional[dict],
    channel_name: str,
) -> PairQualityAssessment:
    """Assess pair quality with per-channel spread and volume thresholds.

    Applies channel-specific spread limits from _SPREAD_LIMIT_BY_CHANNEL
    instead of the global hard gate. Non-SCALP channels also use a lower
    minimum volume floor (_MIN_VOLUME_NON_SCALP) to avoid excluding
    valid lower-cap futures pairs.
    """
```

---

## Expected Impact

| Channel | Old Spread Limit | New Spread Limit | Volume Floor | Impact |
|---------|-----------------|-----------------|--------------|--------|
| 360_SCALP | 3% | 2.5% | $1M | Tighter — fewer false entries |
| 360_SCALP_OBI | 3% | 2.5% | $1M | Tighter — needs tight book |
| 360_SWING | 3% | 5% | $500K | +15–20% more pairs pass |
| 360_SPOT | 3% | 6% | $500K | +20–25% more pairs pass |
| 360_GEM | 3% | 8% | $500K | +30–40% more pairs pass |

Pairs like KATUSDT (spread ~3.5%) will now pass for SWING/SPOT/GEM channels while remaining
blocked from scalp channels. This is the correct behaviour — KATUSDT is a valid SWING setup
but would incur too much slippage for scalp entries.

---

## Tests to Update

**File:** `tests/test_signal_quality.py`

- Add test: `assess_pair_quality_for_channel("360_SWING", spread_pct=0.04)` → passes
- Add test: `assess_pair_quality_for_channel("360_SCALP", spread_pct=0.04)` → fails
- Add test: `assess_pair_quality_for_channel("360_GEM", spread_pct=0.07)` → passes
- Add test: `assess_pair_quality_for_channel("360_SCALP_OBI", spread_pct=0.03)` → passes
- Add test: unknown channel falls back to default spread limit of 0.05

---

## Modules Affected

- `src/signal_quality.py` — primary change
- `src/scanner/__init__.py` — call site update
- `src/scanner.py` — call site update (keep in sync)
- `config/__init__.py` — optional env var additions
