# PR_01 — Market Regime Detector Enhancement

**Branch:** `feature/pr01-regime-detector`  
**Priority:** 1 (Foundation — all other PRs depend on improved regime data)  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Extract and enhance the existing market regime detection logic from `src/regime.py` into a
standalone, more accurate module. The current regime classifier assigns one of five labels
(`TRENDING_UP`, `TRENDING_DOWN`, `RANGING`, `VOLATILE`, `QUIET`) based on raw ADX and price
action. This PR upgrades it with three additional signals:

1. **Volatility percentile** — 14-period ATR expressed as a rolling percentile over the last
   200 bars, giving a normalised 0–100 score that is comparable across pairs.
2. **ADX slope** — 1-bar change in ADX value to detect whether a regime is strengthening or
   weakening (rising ADX = regime reinforcing; falling ADX = regime fading).
3. **Volume profile classification** — ratio of above-VWAP volume to below-VWAP volume over
   the session, identifying accumulation (buyer-dominant) vs distribution (seller-dominant).

The enhanced detector outputs a `RegimeContext` dataclass that downstream modules can query
for richer information than a plain string label.

---

## Files to Change

| File | Change type |
|------|-------------|
| `src/regime.py` | Extend `MarketRegime` detection with percentile ATR, ADX slope, volume profile |
| `src/channels/signal_params.py` | Update lookup keys to accept `RegimeContext` where applicable |
| `src/scanner.py` | Pass full `RegimeContext` to `channel.evaluate()` calls |
| `tests/test_regime_threading.py` | Add tests for new regime signals |

---

## Implementation Steps

### Step 1 — Add `RegimeContext` dataclass to `src/regime.py`

```python
from dataclasses import dataclass

@dataclass
class RegimeContext:
    label: str                    # TRENDING_UP / TRENDING_DOWN / RANGING / VOLATILE / QUIET
    adx_value: float              # Raw ADX
    adx_slope: float              # adx[t] - adx[t-1]; positive = strengthening
    atr_percentile: float         # 0-100 rolling percentile of current ATR vs last 200 bars
    volume_profile: str           # "ACCUMULATION", "DISTRIBUTION", "NEUTRAL"
    is_regime_strengthening: bool # adx_slope > 0 and adx_value > 20
```

### Step 2 — Implement `atr_percentile()` helper

```python
def atr_percentile(atr_series: np.ndarray, lookback: int = 200) -> float:
    """Return rolling percentile (0-100) of the last ATR value vs prior `lookback` bars."""
    if len(atr_series) < 2:
        return 50.0
    window = atr_series[-lookback:] if len(atr_series) >= lookback else atr_series
    current = float(atr_series[-1])
    return float(np.sum(window <= current) / len(window) * 100)
```

### Step 3 — Implement `volume_profile_classify()` helper

```python
def volume_profile_classify(
    volumes: np.ndarray,
    closes: np.ndarray,
    vwap: float,
    lookback: int = 20,
) -> str:
    """Classify volume profile as ACCUMULATION, DISTRIBUTION, or NEUTRAL.

    Above-VWAP candles with high volume indicate accumulation (buyers in control).
    Below-VWAP candles with high volume indicate distribution (sellers in control).
    """
    if vwap <= 0 or len(closes) < lookback or len(volumes) < lookback:
        return "NEUTRAL"
    c = np.asarray(closes[-lookback:], dtype=float)
    v = np.asarray(volumes[-lookback:], dtype=float)
    above_vol = float(np.sum(v[c >= vwap]))
    below_vol = float(np.sum(v[c < vwap]))
    total = above_vol + below_vol
    if total == 0:
        return "NEUTRAL"
    ratio = above_vol / total
    if ratio > 0.60:
        return "ACCUMULATION"
    if ratio < 0.40:
        return "DISTRIBUTION"
    return "NEUTRAL"
```

### Step 4 — Refactor `detect_regime()` to return `RegimeContext`

Update the primary regime detection function signature from:
```python
def detect_regime(candles: dict, indicators: dict) -> str:
```
to:
```python
def detect_regime(candles: dict, indicators: dict) -> RegimeContext:
```

Internally compute:
- ADX slope: `adx_slope = adx_series[-1] - adx_series[-2]`
- ATR percentile: call `atr_percentile(atr_series)`
- Volume profile: call `volume_profile_classify(volumes, closes, vwap)`
- Label classification (existing logic unchanged)
- `is_regime_strengthening = adx_slope > 0 and adx_value > 20`

### Step 5 — Update `src/scanner.py` call sites

Replace:
```python
regime_str = detect_regime(candles, indicators)
channel.evaluate(..., regime=regime_str)
```
with:
```python
regime_ctx = detect_regime(candles, indicators)
channel.evaluate(..., regime=regime_ctx.label)
# Also attach context to signal for downstream use
sig.market_phase = (
    f"{regime_ctx.label} | ATR%ile={regime_ctx.atr_percentile:.0f} | "
    f"Vol={regime_ctx.volume_profile}"
)
```

Backward compatibility is maintained because `channel.evaluate()` still receives a string for
the `regime` parameter. The richer context is only used at the scanner level.

### Step 6 — Expose `RegimeContext` in `Signal` dataclass (optional, low risk)

Add an optional field to `Signal` in `src/channels/base.py`:
```python
regime_context: Optional[str] = ""   # Serialised regime context for logging
```

### Step 7 — Tests

In `tests/test_regime_threading.py`:
- Add test that `detect_regime()` returns a `RegimeContext` object.
- Assert `atr_percentile` is in [0, 100].
- Assert `volume_profile` is one of `{"ACCUMULATION", "DISTRIBUTION", "NEUTRAL"}`.
- Assert `adx_slope` is computed correctly from mock indicator data.

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Regime accuracy | Binary ADX/EMA | ATR percentile + slope + volume profile |
| False positives in VOLATILE regime | Moderate | Reduced by ~15–20% (slope confirms regime is active) |
| Regime transition detection | Lagging (ADX lags by ~5 bars) | Faster via ADX slope |
| Per-signal context richness | Plain string label | Full `RegimeContext` dataclass |

---

## Dependencies

None. This PR is a pure enhancement to existing functionality and has no upstream code dependencies.

---

## Rollback Plan

The `RegimeContext.label` field is a string identical to the current return value of `detect_regime()`. If any downstream consumer breaks, reverting `scanner.py` to use `regime_ctx.label` directly (as it did previously) restores full backward compatibility within minutes.
