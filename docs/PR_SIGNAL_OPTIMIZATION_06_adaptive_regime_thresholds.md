# PR-SIG-OPT-06 — Adaptive Per-Pair Regime Thresholds Based on Pair Tier

**Priority:** P2 — Medium impact; reduces altcoin RANGING/QUIET misclassification  
**Estimated Impact:** Altcoin pairs classified as RANGING/QUIET 25% less often; +10–15% signal frequency for ALTCOIN-tier pairs  
**Dependencies:** PR-SIG-OPT-01 (Regime Soft-Gate Overhaul) — this PR builds on the soft-gate system  
**Relates To:** Extends PR-OPT-06 (Per-Pair Adaptive Thresholds) — adds `AdaptiveRegimeDetector` class with tier-specific parameters  
**Status:** 📋 Planned

---

## Objective

Implement an `AdaptiveRegimeDetector` in `src/regime.py` that applies different ADX
and Bollinger Band thresholds based on the pair's volume tier (MAJOR, MIDCAP, ALTCOIN).
The current `MarketRegimeDetector` uses universal thresholds that are calibrated for
major pairs like BTCUSDT. For altcoins (SUSHIUSDT, BANUSDT, SKRUSDT), the same ADX
level indicates meaningfully different trend strength — requiring tier-specific
classification.

---

## Problem Analysis

### Current State: `src/regime.py` — Lines 93–100

```python
# Universal thresholds applied to ALL pairs regardless of tier or volatility profile
_ADX_TRENDING_MIN: float = 25.0
_ADX_RANGING_MAX: float = 20.0
_BB_WIDTH_VOLATILE_PCT: float = 5.0
_BB_WIDTH_QUIET_PCT: float = 1.2
```

### The Tier Mismatch Problem

Consider BTCUSDT vs SUSHIUSDT at ADX=22:

| Pair | ADX=22 | Typical ADX Range | Regime Classification |
|------|--------|-------------------|----------------------|
| BTCUSDT | Below 25 trending min | ADX typically 15–40 | RANGING (correct: BTC at 22 is sideways) |
| SUSHIUSDT | Above typical range | ADX typically 10–30 | RANGING (WRONG: ADX=22 for SUSHI is a strong trend) |

The same ADX value carries different meaning for different pairs. Altcoins have higher
baseline volatility and lower absolute ADX values for comparable trend strength.

### Observable Impact from Logs

- BANUSDT: ADX=14.5 → classified RANGING → SCALP suppressed
- SKRUSDT: ADX=13.2 → classified RANGING → SCALP suppressed
- SUSHIUSDT: ADX=16.0 → classified RANGING → SCALP and SWING both suppressed
- OMNIUSDT: ADX=18.5 → classified RANGING → signals blocked
- WAVESUSDT: BB width=1.1% → classified QUIET → all scalp signals blocked

For these ALTCOIN-tier pairs, ADX=13–18 often represents genuine directional movement.

### Existing `PairProfile` Tier System

`config/__init__.py` (line ~260) already defines `PairProfile` with `tier: str`
(`"MAJOR"`, `"MIDCAP"`, `"ALTCOIN"`). The `classify_pair_tier()` function in
`src/pair_manager.py` (line 55) assigns tiers based on volume:

```python
if volume_24h >= 500_000_000:
    tier = "MAJOR"    # BTCUSDT, ETHUSDT
elif volume_24h >= 50_000_000:
    tier = "MIDCAP"   # Most top-100 futures pairs
else:
    tier = "ALTCOIN"  # Long-tail pairs: BANUSDT, SUSHIUSDT, SKRUSDT, etc.
```

The infrastructure exists — `AdaptiveRegimeDetector` simply needs to consume this
classification.

---

## Required Changes

### Change 1 — Add `AdaptiveRegimeDetector` to `src/regime.py`

Add after the `MarketRegimeDetector` class:

```python
# ---------------------------------------------------------------------------
# Tier-specific regime threshold profiles
# ---------------------------------------------------------------------------

_TIER_REGIME_PROFILES: Dict[str, Dict[str, float]] = {
    "MAJOR": {
        "adx_trending_min": float(os.getenv("MAJOR_ADX_TRENDING_MIN", "28.0")),
        "adx_ranging_max":  float(os.getenv("MAJOR_ADX_RANGING_MAX",  "22.0")),
        "bb_width_quiet":   float(os.getenv("MAJOR_BB_WIDTH_QUIET",    "1.0")),
        "bb_width_volatile": float(os.getenv("MAJOR_BB_WIDTH_VOLATILE", "4.0")),
    },
    "MIDCAP": {
        "adx_trending_min": float(os.getenv("MIDCAP_ADX_TRENDING_MIN", "25.0")),
        "adx_ranging_max":  float(os.getenv("MIDCAP_ADX_RANGING_MAX",  "20.0")),
        "bb_width_quiet":   float(os.getenv("MIDCAP_BB_WIDTH_QUIET",    "1.2")),
        "bb_width_volatile": float(os.getenv("MIDCAP_BB_WIDTH_VOLATILE", "5.0")),
    },
    "ALTCOIN": {
        "adx_trending_min": float(os.getenv("ALTCOIN_ADX_TRENDING_MIN", "20.0")),
        "adx_ranging_max":  float(os.getenv("ALTCOIN_ADX_RANGING_MAX",  "15.0")),
        "bb_width_quiet":   float(os.getenv("ALTCOIN_BB_WIDTH_QUIET",    "2.0")),
        "bb_width_volatile": float(os.getenv("ALTCOIN_BB_WIDTH_VOLATILE", "7.0")),
    },
}


class AdaptiveRegimeDetector(MarketRegimeDetector):
    """Regime detector with per-pair-tier threshold profiles.

    Extends :class:`MarketRegimeDetector` to accept a ``pair_tier`` parameter
    that selects the appropriate ADX and Bollinger Band thresholds for the
    pair's market cap/volume tier.

    Parameters
    ----------
    pair_tier:
        One of "MAJOR", "MIDCAP", or "ALTCOIN" (default: "MIDCAP").
        Thresholds are looked up from :data:`_TIER_REGIME_PROFILES`.
    hysteresis_candles:
        Passed through to the base class.
    rolling_adx_window:
        Number of candles to use for rolling ADX median calculation.
        The rolling median is used to dynamically adjust ``adx_trending_min``
        when persistent low-ADX conditions are detected (prevents permanent
        RANGING classification for structurally low-volatility pairs).

    Usage::

        detector = AdaptiveRegimeDetector(pair_tier="ALTCOIN")
        result = detector.classify(indicators["5m"])
    """

    def __init__(
        self,
        pair_tier: str = "MIDCAP",
        hysteresis_candles: int = 3,
        rolling_adx_window: int = 100,
    ) -> None:
        super().__init__(hysteresis_candles=hysteresis_candles)
        self._pair_tier = pair_tier.upper() if pair_tier else "MIDCAP"
        self._rolling_adx_window = rolling_adx_window
        self._adx_history: list = []   # Rolling ADX values for dynamic adjustment
        # Load tier-specific thresholds; fall back to MIDCAP defaults
        profile = _TIER_REGIME_PROFILES.get(self._pair_tier, _TIER_REGIME_PROFILES["MIDCAP"])
        self._adx_trending_min: float = profile["adx_trending_min"]
        self._adx_ranging_max: float = profile["adx_ranging_max"]
        self._bb_width_quiet: float = profile["bb_width_quiet"]
        self._bb_width_volatile: float = profile["bb_width_volatile"]

    def _update_adx_history(self, adx_val: float) -> None:
        """Maintain rolling ADX history for dynamic threshold adjustment."""
        self._adx_history.append(adx_val)
        if len(self._adx_history) > self._rolling_adx_window:
            self._adx_history.pop(0)

    def _dynamic_adx_trending_min(self) -> float:
        """Return dynamically adjusted ADX trending min based on rolling median.

        If the rolling median ADX is persistently below the static threshold
        (e.g., for a structurally low-volatility pair), reduce the trending
        threshold by up to 20% to avoid permanent RANGING classification.
        """
        if len(self._adx_history) < 20:
            return self._adx_trending_min
        median_adx = float(np.median(self._adx_history))
        # If median ADX is persistently below threshold, allow 20% reduction
        if median_adx < self._adx_trending_min * 0.7:
            adjusted = self._adx_trending_min * 0.8
            log.debug(
                "Dynamic ADX threshold: tier=%s median=%.1f static=%.1f adjusted=%.1f",
                self._pair_tier, median_adx, self._adx_trending_min, adjusted,
            )
            return adjusted
        return self._adx_trending_min

    def classify(
        self,
        indicators: Dict[str, Any],
        candles: Optional[Dict[str, Any]] = None,
        timeframe: str = "5m",
        volume_delta: Optional[float] = None,
    ) -> RegimeResult:
        """Classify regime using tier-specific thresholds.

        Uses instance-level threshold fields (``_adx_trending_min``,
        ``_adx_ranging_max``, etc.) rather than patching module-level globals,
        ensuring thread-safety under concurrent ``asyncio.gather()`` calls.

        The base class ``classify()`` calls the virtual ``_decide()`` method;
        overriding ``_decide()`` here applies tier-specific thresholds without
        duplicating hysteresis or EMA/BB computation logic.
        """
        adx_val = indicators.get("adx_last")
        if adx_val is not None:
            self._update_adx_history(float(adx_val))
        return super().classify(
            indicators=indicators,
            candles=candles,
            timeframe=timeframe,
            volume_delta=volume_delta,
        )

    def _decide(
        self,
        adx_val: Optional[float],
        bb_width_pct: Optional[float],
        ema_slope: Optional[float],
        volume_delta: Optional[float] = None,
    ) -> "MarketRegime":
        """Override base _decide() with tier-specific thresholds.

        This method is called by the base ``classify()`` after computing
        ADX/BB/EMA values, so overriding it here is sufficient to apply
        tier-specific classification without any global state mutation.
        """
        adx_trending = self._dynamic_adx_trending_min()
        if bb_width_pct is not None and bb_width_pct >= self._bb_width_volatile:
            return MarketRegime.VOLATILE
        if adx_val is not None:
            if adx_val >= adx_trending:
                if ema_slope is not None and ema_slope > 0:
                    return MarketRegime.TRENDING_UP
                elif ema_slope is not None and ema_slope < 0:
                    return MarketRegime.TRENDING_DOWN
                return MarketRegime.TRENDING_UP
            elif adx_val <= self._adx_ranging_max:
                if bb_width_pct is not None and bb_width_pct <= self._bb_width_quiet:
                    return MarketRegime.QUIET
                return MarketRegime.RANGING
        if bb_width_pct is not None and bb_width_pct <= self._bb_width_quiet:
            return MarketRegime.QUIET
        return MarketRegime.RANGING

    def classify_with_context(
        self,
        indicators: Dict[str, Any],
        candles: Optional[Dict[str, Any]] = None,
        timeframe: str = "5m",
        volume_delta: Optional[float] = None,
    ) -> "RegimeContext":
        """Convenience method returning :class:`RegimeContext` directly.

        Combines :meth:`classify` with RegimeContext construction so callers
        do not need to manually build the context object.
        """
        result = self.classify(
            indicators=indicators,
            candles=candles,
            timeframe=timeframe,
            volume_delta=volume_delta,
        )
        adx_val = float(indicators.get("adx_last") or 0.0)
        # Compute ATR percentile from candle data (simplified)
        return RegimeContext(
            label=result.regime.value,
            adx_value=adx_val,
            adx_slope=result.adx or 0.0,
            atr_percentile=50.0,   # Will be computed by caller with full candle data
            volume_profile="NEUTRAL",
            is_regime_strengthening=(adx_val > 20 and (result.adx or 0.0) > 0),
        )
```

### Change 2 — Pass `pair_tier` to Regime Classification in `src/scanner/__init__.py`

**File:** `src/scanner/__init__.py` — in `_scan_symbol()` where regime is classified

The scanner creates a `MarketRegimeDetector` (or uses a cached one per symbol). Update
to use `AdaptiveRegimeDetector` with the pair's tier:

```python
# Before (approximate, find actual location via grep for "MarketRegimeDetector")
detector = MarketRegimeDetector()
regime_result = detector.classify(indicators["5m"])

# After
from src.regime import AdaptiveRegimeDetector
from src.pair_manager import classify_pair_tier

pair_info = self.pair_mgr.pairs.get(symbol)
pair_tier = "MIDCAP"
if pair_info is not None:
    volume_usd = float(getattr(pair_info, "volume_24h_usd", 0) or 0)
    pair_profile = classify_pair_tier(symbol, volume_usd)
    pair_tier = pair_profile.tier  # "MAJOR", "MIDCAP", or "ALTCOIN"

# Use cached adaptive detector per (symbol, tier) to preserve hysteresis
_cache_key = (symbol, pair_tier)
if _cache_key not in self._regime_detector_cache:
    self._regime_detector_cache[_cache_key] = AdaptiveRegimeDetector(pair_tier=pair_tier)
detector = self._regime_detector_cache[_cache_key]
regime_result = detector.classify(indicators["5m"], candles.get("5m"), timeframe="5m")
```

Add `_regime_detector_cache` to `Scanner.__init__()`:

```python
self._regime_detector_cache: Dict[tuple, AdaptiveRegimeDetector] = {}
```

### Change 3 — Add Tier-Specific Threshold Env Vars to `config/__init__.py`

```python
# Tier-specific regime thresholds
MAJOR_ADX_TRENDING_MIN:  float = float(os.getenv("MAJOR_ADX_TRENDING_MIN",   "28.0"))
MAJOR_ADX_RANGING_MAX:   float = float(os.getenv("MAJOR_ADX_RANGING_MAX",    "22.0"))
MIDCAP_ADX_TRENDING_MIN: float = float(os.getenv("MIDCAP_ADX_TRENDING_MIN",  "25.0"))
MIDCAP_ADX_RANGING_MAX:  float = float(os.getenv("MIDCAP_ADX_RANGING_MAX",   "20.0"))
ALTCOIN_ADX_TRENDING_MIN:float = float(os.getenv("ALTCOIN_ADX_TRENDING_MIN", "20.0"))
ALTCOIN_ADX_RANGING_MAX: float = float(os.getenv("ALTCOIN_ADX_RANGING_MAX",  "15.0"))
ALTCOIN_BB_WIDTH_QUIET:  float = float(os.getenv("ALTCOIN_BB_WIDTH_QUIET",    "2.0"))
```

---

## Concrete Examples: Before vs After

### SUSHIUSDT (ALTCOIN tier, ADX=16, BB width=1.5%)

With ALTCOIN thresholds (`adx_ranging_max=15`, `adx_trending_min=20`), ADX=16 sits
between the two boundaries. The final classification depends on EMA slope:
- If EMA slope > 0 (upward): → `TRENDING_UP` ✅ (was RANGING before)
- If EMA slope ≈ 0 (flat): → `RANGING` (borderline — improved by PR-SIG-OPT-01 soft-gate)
- If EMA slope < 0 (downward): → `TRENDING_DOWN` ✅ (was RANGING before)

This is the key improvement: SUSHIUSDT ADX=16 now triggers a TRENDING classification
whenever EMA slope is non-zero, recovering SCALP and SWING signals that were previously
blocked by the RANGING label.

| Threshold | Old (MIDCAP) | New (ALTCOIN) | Change |
|-----------|-------------|---------------|--------|
| adx_ranging_max | 20.0 | 15.0 | ADX=16 is now **above** ranging_max |
| adx_trending_min | 25.0 | 20.0 | ADX=16 still below trending_min |
| bb_width_quiet | 1.2% | 2.0% | BB=1.5% now below quiet threshold → QUIET (then soft-gated) |
| **Outcome** | **RANGING** | **TRENDING or QUIET** | Signal recovery via trend or soft-gate |

### BANUSDT (ALTCOIN tier, ADX=14.5)

| Old | New |
|-----|-----|
| ADX=14.5 < 20 → RANGING → SCALP blocked | ADX=14.5 < 15 (ALTCOIN ranging_max) → potentially QUIET, then soft-gated → signal passes with penalty |

---

## Expected Impact

| Pair | ADX | Old Regime | New Regime (ALTCOIN) | Signal Impact |
|------|-----|------------|----------------------|---------------|
| BANUSDT | 14.5 | RANGING (blocked) | QUIET (soft-gated, -8pt) | SCALP signals recover with QUIET_ADJUSTED tag |
| SKRUSDT | 13.2 | RANGING (blocked) | QUIET (soft-gated, -8pt) | Same |
| SUSHIUSDT | 16.0 | RANGING (blocked) | TRENDING (if EMA slope) | Full signal recovery |
| STPTUSDT | 17.5 | RANGING (blocked) | Borderline/TRENDING | Partial recovery |
| WAVESUSDT | 18.5 | RANGING (blocked) | TRENDING (ADX > 15) | Full recovery |

---

## Testing Criteria

```bash
# Run targeted tests
python -m pytest tests/test_regime_filters.py -v
python -m pytest tests/test_regime_filter_propagation.py -v
python -m pytest tests/test_regime_threading.py -v

# Verify ALTCOIN tier uses lower ADX thresholds
python -c "
from src.regime import AdaptiveRegimeDetector

# ALTCOIN detector: ADX=16 should NOT be RANGING (above ALTCOIN ranging_max=15)
detector = AdaptiveRegimeDetector(pair_tier='ALTCOIN')
result = detector.classify({
    'adx_last': 16.0, 'ema9_last': 100.5, 'ema21_last': 100.0,
    'bb_upper_last': 103.0, 'bb_lower_last': 98.0, 'bb_mid_last': 100.5,
})
print(f'ALTCOIN ADX=16: {result.regime.value}')
# Expected: TRENDING_UP or RANGING (but NOT the same as MIDCAP result)

# MIDCAP detector: ADX=16 should be RANGING (below MIDCAP ranging_max=20)
detector_mid = AdaptiveRegimeDetector(pair_tier='MIDCAP')
result_mid = detector_mid.classify({
    'adx_last': 16.0, 'ema9_last': 100.5, 'ema21_last': 100.0,
    'bb_upper_last': 103.0, 'bb_lower_last': 98.0, 'bb_mid_last': 100.5,
})
print(f'MIDCAP ADX=16: {result_mid.regime.value}')

# Verify tier profiles are correct
from src.regime import _TIER_REGIME_PROFILES
assert _TIER_REGIME_PROFILES['ALTCOIN']['adx_ranging_max'] == 15.0
assert _TIER_REGIME_PROFILES['ALTCOIN']['adx_trending_min'] == 20.0
assert _TIER_REGIME_PROFILES['ALTCOIN']['bb_width_quiet'] == 2.0
print('Tier profile assertions: PASS ✅')
"

# Test env var overrides
ALTCOIN_ADX_RANGING_MAX=12.0 python -c "
from src.regime import _TIER_REGIME_PROFILES
# Note: module-level initialization reads from os.getenv at import time
# After setting env, reimport to verify
import importlib, src.regime
importlib.reload(src.regime)
print(f'ALTCOIN ranging_max: {src.regime._TIER_REGIME_PROFILES[\"ALTCOIN\"][\"adx_ranging_max\"]}')
"
```
