# PR-SIG-OPT-03 — Per-Channel Minimum Signal Lifespan

**Priority:** P1 — Directly unblocks SCALP signal skipping (e.g. SUPERUSDT < 21600s)  
**Estimated Signal Recovery:** +20% SCALP signal throughput; eliminates false SL/TP suppression on short-lived scalps  
**Dependencies:** None (standalone change to `config/__init__.py` and `src/trade_monitor.py`)  
**Relates To:** Extends PR-OPT-06 (Lifespan Channel-Aware) — specifically addresses SCALP timeframe mismatch  
**Status:** 📋 Planned

---

## Objective

Replace the implicit global 6-hour minimum signal lifespan with **per-channel,
env-configurable values** that match each channel's operational timeframe. The current
configuration applies an excessively long lifespan guard to SCALP channels that operate
on M1/M5 timeframes, causing legitimate short-lived scalp signals (like SUPERUSDT) to
be skipped during SL/TP evaluation.

---

## Problem Analysis

### Current State: `config/__init__.py` — Lines 674–683

```python
# Anti-noise: minimum signal lifespan before SL/TP checks are applied (secs)
MIN_SIGNAL_LIFESPAN_SECONDS: Dict[str, int] = {
    "360_SCALP": 180,
    "360_SWING": 300,
    "360_SPOT": 600,
    "360_GEM": 21600,  # 6 hours
}
```

These values appear reasonable at first glance: SCALP at 180s (3 min), SWING at 300s
(5 min). However, there is a **second lifespan mechanism** in `src/signal_lifecycle.py`
(line 522) that governs when a signal is considered "active" for lifecycle monitoring:

```python
# src/signal_lifecycle.py — line 522
interval_seconds = LIFECYCLE_CHECK_INTERVAL.get(signal.channel, 21600)
```

The `LIFECYCLE_CHECK_INTERVAL` dict (`config/__init__.py` lines 657–660) contains:

```python
LIFECYCLE_CHECK_INTERVAL: Dict[str, int] = {
    "360_SWING": int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SWING", "14400")),   # 4 hours
    "360_SPOT":  int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SPOT",  "21600")),   # 6 hours
    "360_GEM":   int(os.getenv("LIFECYCLE_CHECK_INTERVAL_GEM",  "43200")),    # 12 hours
}
```

**Critical Finding:** `LIFECYCLE_CHECK_INTERVAL` has **no entry for `360_SCALP`**, so
`signal_lifecycle.py` falls back to the default value of `21600` (6 hours) for any
SCALP signal. This means SCALP signals are considered "too new" for lifecycle checks
for 6 hours, preventing SL/TP evaluation for the entire lifespan of the signal.

### Additional Issue: `src/trade_monitor.py` — Lines 545–552

```python
# trade_monitor.py — minimum lifespan guard
min_lifespan = MIN_SIGNAL_LIFESPAN_SECONDS.get(sig.channel, 10)

if age_secs < min_lifespan:
    log.debug(
        "Signal %s %s too new (%.1fs < %ds min lifespan) – skipping SL/TP eval",
        sig.symbol, sig.channel, age_secs, min_lifespan,
    )
```

The `trade_monitor.py` correctly uses `MIN_SIGNAL_LIFESPAN_SECONDS` (180s for SCALP).
The problem is the `signal_lifecycle.py` fallback of 21600s — this is the mechanism
causing "SUPERUSDT < 21600s" log entries.

### Root Cause Summary

1. `LIFECYCLE_CHECK_INTERVAL` missing `360_SCALP` → defaults to 21600s (6h fallback)
2. `LIFECYCLE_CHECK_INTERVAL` missing all SCALP variants (`360_SCALP_FVG`, etc.)
3. SCALP signals on M1/M5 timeframes complete (hit TP or SL) within 15–60 minutes,
   but lifecycle monitoring never evaluates them within the 6h window
4. Result: SCALP signals appear "stuck" as open positions, depressing channel
   capacity metrics and preventing new signal generation (MAX_CONCURRENT_SIGNALS cap)

---

## Required Changes

### Change 1 — Add SCALP entries to `LIFECYCLE_CHECK_INTERVAL`

**File:** `config/__init__.py` — lines 657–660

```python
# Before
LIFECYCLE_CHECK_INTERVAL: Dict[str, int] = {
    "360_SWING": int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SWING", "14400")),   # 4 hours
    "360_SPOT":  int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SPOT",  "21600")),   # 6 hours
    "360_GEM":   int(os.getenv("LIFECYCLE_CHECK_INTERVAL_GEM",  "43200")),    # 12 hours
}

# After — per-channel values aligned with operational timeframe
LIFECYCLE_CHECK_INTERVAL: Dict[str, int] = {
    "360_SCALP":      int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SCALP",      "900")),    # 15 min
    "360_SCALP_FVG":  int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SCALP_FVG",  "900")),    # 15 min
    "360_SCALP_CVD":  int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SCALP_CVD",  "900")),    # 15 min
    "360_SCALP_VWAP": int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SCALP_VWAP", "900")),    # 15 min
    "360_SCALP_OBI":  int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SCALP_OBI",  "900")),    # 15 min
    "360_SWING":      int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SWING",      "7200")),   # 2 hours
    # Note: SWING reduced from 14400 (4h) to 7200 (2h) because swing signals
    # operate primarily on the 1h timeframe and significant adverse moves can
    # develop in 2h. Faster re-evaluation catches trend reversals sooner.
    "360_SPOT":       int(os.getenv("LIFECYCLE_CHECK_INTERVAL_SPOT",       "21600")),  # 6 hours
    "360_GEM":        int(os.getenv("LIFECYCLE_CHECK_INTERVAL_GEM",        "43200")),  # 12 hours
}
```

**Note:** SWING reduced from 14400 (4h) to 7200 (2h) since swing signals operate on
the 1h timeframe and significant moves can occur within 2h.

### Change 2 — Add `ChannelConfig.min_signal_lifespan` field

**File:** `config/__init__.py` — `ChannelConfig` dataclass (line ~236)

```python
# Before
@dataclass(frozen=True)
class ChannelConfig:
    name: str
    emoji: str
    timeframes: List[str]
    sl_pct_range: tuple
    tp_ratios: List[float]
    trailing_atr_mult: float
    adx_min: float
    adx_max: float
    spread_max: float
    min_confidence: float
    min_volume: float = 1_000_000.0
    dca_enabled: bool = False
    ...

# After — add min_signal_lifespan field
@dataclass(frozen=True)
class ChannelConfig:
    name: str
    emoji: str
    timeframes: List[str]
    sl_pct_range: tuple
    tp_ratios: List[float]
    trailing_atr_mult: float
    adx_min: float
    adx_max: float
    spread_max: float
    min_confidence: float
    min_volume: float = 1_000_000.0
    min_signal_lifespan: int = 900   # seconds; default 15min for scalp
    dca_enabled: bool = False
    ...
```

Update each `ChannelConfig` instantiation:

```python
# CHANNEL_SCALP (line ~336)
CHANNEL_SCALP = ChannelConfig(
    ...
    min_signal_lifespan=int(os.getenv("SCALP_MIN_LIFESPAN", "900")),  # 15 min
    ...
)

# CHANNEL_SWING (line ~351)
CHANNEL_SWING = ChannelConfig(
    ...
    min_signal_lifespan=int(os.getenv("SWING_MIN_LIFESPAN", "7200")),  # 2 hours
    ...
)

# CHANNEL_SPOT (line ~381)
CHANNEL_SPOT = ChannelConfig(
    ...
    min_signal_lifespan=int(os.getenv("SPOT_MIN_LIFESPAN", "21600")),  # 6 hours
    ...
)

# CHANNEL_GEM (line ~366)
CHANNEL_GEM = ChannelConfig(
    ...
    min_signal_lifespan=int(os.getenv("GEM_MIN_LIFESPAN", "43200")),  # 12 hours
    ...
)
```

### Change 3 — Update `src/signal_lifecycle.py` to use channel config

**File:** `src/signal_lifecycle.py` — line 522

```python
# Before
interval_seconds = LIFECYCLE_CHECK_INTERVAL.get(signal.channel, 21600)

# After — use LIFECYCLE_CHECK_INTERVAL with sensible per-channel default
from config import LIFECYCLE_CHECK_INTERVAL, ALL_CHANNELS

# Build a fallback map from ChannelConfig if available
def _get_lifecycle_interval(channel_name: str) -> int:
    if channel_name in LIFECYCLE_CHECK_INTERVAL:
        return LIFECYCLE_CHECK_INTERVAL[channel_name]
    # Fallback: look up from ChannelConfig.min_signal_lifespan
    for ch in ALL_CHANNELS:
        if ch.name == channel_name:
            return ch.min_signal_lifespan
    # Last resort: SCALP-like default for unknown channels
    if "SCALP" in channel_name:
        return 900
    return 3600  # 1 hour generic fallback (was 21600)

interval_seconds = _get_lifecycle_interval(signal.channel)
```

### Change 4 — Update `MIN_SIGNAL_LIFESPAN_SECONDS` to be env-configurable

**File:** `config/__init__.py` — lines 676–683

```python
# Before
MIN_SIGNAL_LIFESPAN_SECONDS: Dict[str, int] = {
    "360_SCALP": 180,
    "360_SWING": 300,
    "360_SPOT": 600,
    "360_GEM": 21600,
}

# After — env-configurable, aligned with channel timeframes
MIN_SIGNAL_LIFESPAN_SECONDS: Dict[str, int] = {
    "360_SCALP":      int(os.getenv("MIN_LIFESPAN_SCALP",      "180")),    # 3 min — unchanged
    "360_SCALP_FVG":  int(os.getenv("MIN_LIFESPAN_SCALP_FVG",  "180")),
    "360_SCALP_CVD":  int(os.getenv("MIN_LIFESPAN_SCALP_CVD",  "180")),
    "360_SCALP_VWAP": int(os.getenv("MIN_LIFESPAN_SCALP_VWAP", "180")),
    "360_SCALP_OBI":  int(os.getenv("MIN_LIFESPAN_SCALP_OBI",  "180")),
    "360_SWING":      int(os.getenv("MIN_LIFESPAN_SWING",       "300")),    # 5 min — unchanged
    "360_SPOT":       int(os.getenv("MIN_LIFESPAN_SPOT",        "600")),    # 10 min — unchanged
    "360_GEM":        int(os.getenv("MIN_LIFESPAN_GEM",         "21600")),  # 6 hours — unchanged
}
```

---

## Rationale: Per-Channel Lifecycle Values

| Channel | Timeframe | Old Lifecycle Default | New Lifecycle Value | Reason |
|---------|-----------|----------------------|---------------------|--------|
| 360_SCALP | M1/M5 | 21600s (6h!) | 900s (15min) | Scalps complete in 15–60 min |
| 360_SCALP_FVG | M1/M5 | 21600s (6h!) | 900s (15min) | Same as above |
| 360_SWING | H1/H4 | 14400s (4h) | 7200s (2h) | Swing moves develop in 2h |
| 360_SPOT | H4/D1 | 21600s (6h) | 21600s (6h) | Unchanged — correct |
| 360_GEM | D1/W1 | 43200s (12h) | 43200s (12h) | Unchanged — correct |

---

## Expected Impact

| Scenario | Before | After |
|----------|--------|-------|
| SUPERUSDT SCALP signal, age=300s | Skipped (lifecycle: 21600s default) | Evaluated (lifecycle: 900s) ✅ |
| SCALP signal occupying "slot" for 6h | Blocks new signals for 6h | Slot freed after 15min ✅ |
| MAX_CONCURRENT_SIGNALS cap hit | 5 stale SCALP signals blocking new ones | Cap correctly reflects live positions ✅ |
| SWING lifecycle re-check | Only after 4h | After 2h — catches trend reversals sooner ✅ |

Estimated: **+20% SCALP signal frequency** by eliminating stale signal slot occupancy,
and **+15% SWING signal re-evaluation quality** by detecting trend reversals sooner.

---

## Implementation Notes

1. The `LIFECYCLE_CHECK_INTERVAL` dict is used in `src/signal_lifecycle.py` and should
   be the single source of truth. The `ChannelConfig.min_signal_lifespan` field is
   additive — it provides structured access but should not duplicate the dict.
2. The `_get_lifecycle_interval()` helper must be extracted to a utility function
   or placed at module level to avoid repeated imports.
3. Add all new env vars to `.env.example` with documented defaults.
4. The `LIFECYCLE_SCAN_INTERVAL_MINUTES` constant (used for background loop cadence)
   should remain unchanged — this change only affects **per-signal** minimum age
   checks, not the loop frequency.

---

## Testing Criteria

```bash
# Run targeted tests
python -m pytest tests/test_engine_lifecycle.py -v
python -m pytest tests/test_signal_execution_timing.py -v
python -m pytest tests/test_trade_monitor.py -v

# Verify SCALP signals no longer blocked by 21600s default
python -c "
from config import LIFECYCLE_CHECK_INTERVAL
scalp_interval = LIFECYCLE_CHECK_INTERVAL.get('360_SCALP', 21600)
assert scalp_interval == 900, f'Expected 900s but got {scalp_interval}s'
print(f'SCALP lifecycle interval: {scalp_interval}s (expected 900s) ✅')

# Verify all SCALP variants are covered
scalp_variants = ['360_SCALP', '360_SCALP_FVG', '360_SCALP_CVD', '360_SCALP_VWAP', '360_SCALP_OBI']
for ch in scalp_variants:
    interval = LIFECYCLE_CHECK_INTERVAL.get(ch, 21600)
    assert interval < 3600, f'{ch} interval {interval}s is too long!'
    print(f'{ch}: {interval}s ✅')
"

# Verify env var override works
LIFECYCLE_CHECK_INTERVAL_SCALP=600 python -c "
from config import LIFECYCLE_CHECK_INTERVAL
assert LIFECYCLE_CHECK_INTERVAL.get('360_SCALP') == 600
print('Env override test: PASS ✅')
"
```
