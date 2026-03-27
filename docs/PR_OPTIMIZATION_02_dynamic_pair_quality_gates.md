# PR-OPT-02 — Dynamic Pair Quality Gates (Tier-Adaptive Thresholds)

**Priority:** P1  
**Estimated Additional Pairs Passing:** +10–20 pairs, primarily SWING and GEM channels  
**Dependencies:** None

---

## Objective

Replace the single fixed pair quality gate (`spread_pct <= 0.03`, `volume_24h >= 1_000_000`) with tier-dependent and channel-dependent thresholds. The current gate uniformly rejects low-cap and mid-cap pairs like KATUSDT across **all** strategies, even SWING and SPOT channels that tolerate wider spreads due to their longer holding periods.

---

## Analysis of Current Code

### `src/signal_quality.py` — Lines 270–313

```python
def assess_pair_quality(
    volume_24h: float,
    spread_pct: float,
    indicators: Dict[str, Any],
    candles: Optional[dict],
) -> PairQualityAssessment:
    ...
    passed = total >= 58 and spread_pct <= 0.03 and volume_24h >= 1_000_000
    reason = ""
    if not passed:
        if spread_pct > 0.03:
            reason = "spread too wide"
        elif volume_24h < 1_000_000:
            reason = "liquidity too thin"
        else:
            reason = "pair quality below threshold"
```

**Problems:**
1. The 3 bps spread hard gate is applied identically regardless of channel, pair tier, or intended holding period.
2. The $1M volume floor is applied uniformly — appropriate for SCALP, but overly strict for GEM (macro, long-hold) and SPOT (institutional) channels.
3. Rejections are logged as `"spread too wide"` but the channel context is absent, making it impossible to post-analyse which channels are losing signals.
4. There is no mechanism to adjust confidence proportional to spread width — a pair at 3.1% spread is treated identically to one at 8% spread.

---

## Recommended Changes

### Change 1 — Tier-dependent spread thresholds

**File:** `src/signal_quality.py`

Add a `channel` parameter to `assess_pair_quality` (default `None` for backward compatibility):

```python
# Tier-adaptive spread limits per channel group.
# SCALP → tight execution, spread is a direct cost centre.
# SWING / SPOT → medium-term, spread amortised over holding period.
# GEM → macro position, spread negligible vs. multi-day move target.
_CHANNEL_MAX_SPREAD_PCT: Dict[str, float] = {
    "360_SCALP":      0.03,   # 3 bps — keep current
    "360_SCALP_FVG":  0.03,
    "360_SCALP_CVD":  0.03,
    "360_SCALP_VWAP": 0.03,
    "360_SCALP_OBI":  0.03,
    "360_SWING":      0.05,   # 5 bps — relax for medium-term hold
    "360_SPOT":       0.05,   # 5 bps — relax for spot holds
    "360_GEM":        0.08,   # 8 bps — macro, spread is noise
}
_DEFAULT_MAX_SPREAD_PCT: float = 0.03   # fallback
```

Updated gate logic:

```python
def assess_pair_quality(
    volume_24h: float,
    spread_pct: float,
    indicators: Dict[str, Any],
    candles: Optional[dict],
    channel: Optional[str] = None,          # NEW parameter
) -> PairQualityAssessment:
    ...
    max_spread = _CHANNEL_MAX_SPREAD_PCT.get(channel, _DEFAULT_MAX_SPREAD_PCT)
    min_volume = _CHANNEL_MIN_VOLUME.get(channel, 1_000_000)

    passed = total >= 58 and spread_pct <= max_spread and volume_24h >= min_volume
```

### Change 2 — Channel-dependent volume minimums

**File:** `src/signal_quality.py`

```python
_CHANNEL_MIN_VOLUME: Dict[str, float] = {
    "360_SCALP":      1_000_000,   # $1M — tight execution requires deep book
    "360_SCALP_FVG":  1_000_000,
    "360_SCALP_CVD":  1_000_000,
    "360_SCALP_VWAP": 1_000_000,
    "360_SCALP_OBI":  1_000_000,
    "360_SWING":        500_000,   # $500K — longer hold tolerates thinner book
    "360_SPOT":         500_000,
    "360_GEM":          200_000,   # $200K — macro, entered in tranches
}
```

### Change 3 — `spread_adjusted_confidence` field

**File:** `src/signal_quality.py`

Add a new field to `PairQualityAssessment`:

```python
@dataclass
class PairQualityAssessment:
    passed: bool
    score: float
    label: str
    volume_tier: str
    spread_score: float
    volatility_score: float
    noise_score: float
    reason: str
    spread_adjusted_confidence_delta: float = 0.0   # NEW — negative adjustment
```

Compute the adjustment when spread exceeds TIER1 threshold but is within channel-specific limit:

```python
# If spread is above the TIER1 floor (0.03) but below the channel limit,
# apply a graduated confidence reduction so downstream scoring reflects cost.
_TIER1_SPREAD_FLOOR: float = 0.03
if spread_pct > _TIER1_SPREAD_FLOOR and passed:
    excess_bps = (spread_pct - _TIER1_SPREAD_FLOOR) * 10_000
    spread_adjusted_confidence_delta = -round(min(excess_bps * 1.5, 20.0), 2)
else:
    spread_adjusted_confidence_delta = 0.0
```

Downstream signal evaluation should add `spread_adjusted_confidence_delta` to the base confidence:

```python
adjusted_conf = base_confidence + quality.spread_adjusted_confidence_delta
```

### Change 4 — Dedicated suppressed pair logging

**File:** `src/signal_quality.py`

```python
import logging as _std_logging

_suppression_log = _std_logging.getLogger("suppressed_signals")

# In assess_pair_quality, replace the existing not-passed debug log with:
if not passed:
    _suppression_log.info(
        "pair_quality_gate sym=%s channel=%s spread_pct=%.4f spread_limit=%.4f "
        "volume=%.0f volume_limit=%.0f score=%.1f reason=%s",
        indicators.get("symbol", "UNKNOWN"),
        channel or "UNSPECIFIED",
        spread_pct, max_spread,
        volume_24h, min_volume,
        total, reason,
    )
```

Configure a dedicated `suppressed_signals.log` file handler in the logging bootstrap:

```python
# In src/logger.py or src/bootstrap.py
suppression_handler = logging.FileHandler("logs/suppressed_signals.log")
suppression_handler.setLevel(logging.INFO)
logging.getLogger("suppressed_signals").addHandler(suppression_handler)
```

---

## Modules Affected

| Module | Change |
|--------|--------|
| `src/signal_quality.py` | Add channel param, tier-adaptive thresholds, confidence delta field |
| `src/pair_manager.py` | Pass channel name when calling `assess_pair_quality` |
| `src/scanner/__init__.py` | Pass channel name to quality gate |
| `src/scanner.py` | Pass channel name to quality gate |
| `config/__init__.py` | Optionally expose threshold env vars for runtime tuning |
| `src/logger.py` / `src/bootstrap.py` | Add suppressed_signals.log handler |

---

## Test Cases

### Unit Tests

1. **`test_scalp_strict_spread`** — Pair with 4 bps spread must FAIL for `360_SCALP` channel.
2. **`test_swing_relaxed_spread`** — Pair with 4 bps spread must PASS for `360_SWING` channel.
3. **`test_gem_relaxed_spread`** — Pair with 7 bps spread must PASS for `360_GEM` channel.
4. **`test_gem_strict_volume`** — Pair with $150K volume must FAIL for `360_GEM` channel ($200K min).
5. **`test_swing_volume_floor`** — Pair with $400K volume must FAIL for `360_SWING`, must PASS for `360_GEM`.
6. **`test_spread_adjusted_confidence_delta`** — Pair at 4 bps spread on `360_SWING` channel must have negative `spread_adjusted_confidence_delta`.
7. **`test_no_delta_at_tier1`** — Pair at exactly 3 bps spread must have zero delta.
8. **`test_backward_compat_no_channel`** — Calling `assess_pair_quality` without `channel` param must preserve existing behavior (3 bps, $1M thresholds).

### Integration Tests

9. **`test_katusdt_passes_swing`** — KATUSDT (known spread >3 bps) passes quality gate for `360_SWING` and `360_GEM`.
10. **`test_suppression_log_written`** — After a gate failure, verify an entry appears in `suppressed_signals.log`.

---

## Rollback Procedure

1. Remove `channel` parameter from `assess_pair_quality` (or restore default-only behavior).
2. Remove `_CHANNEL_MAX_SPREAD_PCT` and `_CHANNEL_MIN_VOLUME` dicts.
3. Restore `passed = total >= 58 and spread_pct <= 0.03 and volume_24h >= 1_000_000`.
4. Remove `spread_adjusted_confidence_delta` field from `PairQualityAssessment`.

No database migration required. The `suppressed_signals.log` file can be left in place.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| SWING / GEM signals on very wide spreads cause trade losses | Medium | Confidence delta reduces position sizing for wide-spread signals |
| Pairs with $200K volume have excessive slippage for large positions | Low | GEM signals use scaled entry (DCA module) — first tranche is small |
| Backward-compat break if callers don't pass `channel` | Low | Default `None` preserves old 3 bps / $1M behavior |
| Suppression log disk usage | Low | Log rotation via standard Python `RotatingFileHandler` |

---

## Expected Impact

- **KATUSDT** and similar mid-cap pairs pass SWING / GEM quality gates
- **+10–20 additional pairs** per scan cycle entering active signal evaluation
- Suppression log provides data-driven evidence for future threshold calibration
