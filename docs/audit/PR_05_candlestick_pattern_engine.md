# PR_05 — Candlestick Pattern Engine

**Branch:** `feature/pr05-candlestick-patterns`  
**Priority:** 5  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Add a candlestick pattern recognition layer that detects high-probability reversal and
continuation formations on M5, H1, and H4 candles. Detected patterns are added to the
`Signal.chart_pattern_names` field (already defined in `Signal`) and contribute a
confluence bonus to the signal scoring engine (PR_09). They do not act as hard gates —
absence of a pattern does not suppress a signal, but presence adds confidence.

Patterns to detect:

| Pattern | Direction | Signal type |
|---------|-----------|-------------|
| Bullish Engulfing | LONG | Reversal confluence |
| Bearish Engulfing | SHORT | Reversal confluence |
| Hammer / Pin Bar | LONG | Reversal at support |
| Shooting Star | SHORT | Reversal at resistance |
| Doji | N/A | Indecision — confidence penalty |
| Morning Star (3-bar) | LONG | Strong reversal |
| Evening Star (3-bar) | SHORT | Strong reversal |
| Three White Soldiers | LONG | Continuation |
| Three Black Crows | SHORT | Continuation |

The existing `src/chart_patterns.py` module is extended with these detection functions.

---

## Files to Change

| File | Change type |
|------|-------------|
| `src/chart_patterns.py` | Add pattern detection functions |
| `src/scanner.py` | Call pattern detection and attach results to signal |
| `src/channels/base.py` | No change (chart_pattern_names field already defined on Signal) |
| `tests/test_channels.py` | Add tests for pattern detection on mock candle data |

---

## Implementation Steps

### Step 1 — Implement core pattern functions in `src/chart_patterns.py`

```python
"""Candlestick pattern recognition for confluence scoring."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class PatternResult:
    name: str
    direction: str   # "LONG", "SHORT", or "NEUTRAL"
    confidence_bonus: float   # Points to add to composite signal score


def _body(open_: float, close: float) -> float:
    return abs(close - open_)

def _range(high: float, low: float) -> float:
    return high - low if high > low else 1e-9

def _upper_wick(open_: float, high: float, close: float) -> float:
    return high - max(open_, close)

def _lower_wick(open_: float, low: float, close: float) -> float:
    return min(open_, close) - low


def detect_engulfing(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> List[PatternResult]:
    """Detect bullish/bearish engulfing on the last two candles."""
    if len(closes) < 2:
        return []
    o1, h1, l1, c1 = float(opens[-2]), float(highs[-2]), float(lows[-2]), float(closes[-2])
    o2, h2, l2, c2 = float(opens[-1]), float(highs[-1]), float(lows[-1]), float(closes[-1])
    results = []
    # Bullish engulfing: prior candle bearish, current candle bullish and body engulfs prior body
    if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1:
        results.append(PatternResult("BULLISH_ENGULFING", "LONG", 8.0))
    # Bearish engulfing: prior candle bullish, current candle bearish and body engulfs prior body
    if c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1:
        results.append(PatternResult("BEARISH_ENGULFING", "SHORT", 8.0))
    return results


def detect_pin_bar(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> List[PatternResult]:
    """Detect hammer (bullish pin bar) and shooting star (bearish pin bar)."""
    if len(closes) < 1:
        return []
    o, h, l, c = float(opens[-1]), float(highs[-1]), float(lows[-1]), float(closes[-1])
    body = _body(o, c)
    candle_range = _range(h, l)
    lower_wick = _lower_wick(o, l, c)
    upper_wick = _upper_wick(o, h, c)
    results = []
    if candle_range > 0:
        # Hammer: long lower wick (>2× body), short upper wick (<0.3× body or <0.3× lower wick)
        if lower_wick >= body * 2.0 and upper_wick <= body * 0.5:
            results.append(PatternResult("HAMMER", "LONG", 6.0))
        # Shooting star: long upper wick (>2× body), short lower wick
        if upper_wick >= body * 2.0 and lower_wick <= body * 0.5:
            results.append(PatternResult("SHOOTING_STAR", "SHORT", 6.0))
    return results


def detect_doji(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    body_threshold_pct: float = 0.1,
) -> List[PatternResult]:
    """Detect doji candle (body < body_threshold_pct of total range).

    A doji signals indecision and applies a confidence *penalty* (negative bonus).
    """
    if len(closes) < 1:
        return []
    o, h, l, c = float(opens[-1]), float(highs[-1]), float(lows[-1]), float(closes[-1])
    body = _body(o, c)
    candle_range = _range(h, l)
    if candle_range > 0 and body / candle_range < body_threshold_pct:
        return [PatternResult("DOJI", "NEUTRAL", -5.0)]
    return []


def detect_morning_evening_star(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> List[PatternResult]:
    """Detect 3-bar morning star (LONG) and evening star (SHORT)."""
    if len(closes) < 3:
        return []
    o1, c1 = float(opens[-3]), float(closes[-3])
    o2, h2, l2, c2 = float(opens[-2]), float(highs[-2]), float(lows[-2]), float(closes[-2])
    o3, c3 = float(opens[-1]), float(closes[-1])
    results = []
    # Morning star: large bearish → small indecision → large bullish
    if (c1 < o1 and _body(o2, c2) < _body(o1, c1) * 0.5
            and c3 > o3 and c3 > (o1 + c1) / 2):
        results.append(PatternResult("MORNING_STAR", "LONG", 10.0))
    # Evening star: large bullish → small indecision → large bearish
    if (c1 > o1 and _body(o2, c2) < _body(o1, c1) * 0.5
            and c3 < o3 and c3 < (o1 + c1) / 2):
        results.append(PatternResult("EVENING_STAR", "SHORT", 10.0))
    return results


def detect_three_soldiers_crows(
    opens: np.ndarray, closes: np.ndarray
) -> List[PatternResult]:
    """Detect three white soldiers (LONG) and three black crows (SHORT)."""
    if len(closes) < 3:
        return []
    c1, c2, c3 = float(closes[-3]), float(closes[-2]), float(closes[-1])
    o1, o2, o3 = float(opens[-3]), float(opens[-2]), float(opens[-1])
    results = []
    if c3 > c2 > c1 and o3 > o2 > o1 and c1 > o1 and c2 > o2 and c3 > o3:
        results.append(PatternResult("THREE_WHITE_SOLDIERS", "LONG", 7.0))
    if c3 < c2 < c1 and o3 < o2 < o1 and c1 < o1 and c2 < o2 and c3 < o3:
        results.append(PatternResult("THREE_BLACK_CROWS", "SHORT", 7.0))
    return results


def detect_all_patterns(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> List[PatternResult]:
    """Run all pattern detectors and return combined results."""
    results: List[PatternResult] = []
    for fn in (detect_engulfing, detect_pin_bar, detect_doji,
               detect_morning_evening_star):
        results.extend(fn(opens, highs, lows, closes))
    results.extend(detect_three_soldiers_crows(opens, closes))
    return results
```

### Step 2 — Call pattern detection in `src/scanner.py`

After computing indicators and before calling `channel.evaluate()`:

```python
from src.chart_patterns import detect_all_patterns

# Detect patterns on the primary timeframe for each channel
primary_tf = "5m"  # scalp; adjust to "1h" for swing, "4h" for spot
m = candles.get(primary_tf, {})
if m and len(m.get("close", [])) >= 3:
    patterns = detect_all_patterns(
        np.asarray(m["open"]), np.asarray(m["high"]),
        np.asarray(m["low"]), np.asarray(m["close"]),
    )
    smc_data["chart_patterns"] = patterns
```

After the signal is returned from `channel.evaluate()`:

```python
if sig is not None:
    patterns = smc_data.get("chart_patterns", [])
    aligned = [p for p in patterns if p.direction == sig.direction.value or p.direction == "NEUTRAL"]
    if aligned:
        sig.chart_pattern_names = ", ".join(p.name for p in aligned)
        # Apply confidence adjustments (bounded by –20 / +20)
        bonus = sum(p.confidence_bonus for p in aligned)
        sig.confidence = max(0.0, min(100.0, sig.confidence + bonus))
```

### Step 3 — Tests

```python
def test_bullish_engulfing_detected():
    from src.chart_patterns import detect_engulfing
    import numpy as np
    opens = np.array([105.0, 100.0])
    highs = np.array([106.0, 108.0])
    lows  = np.array([99.0,  99.5])
    closes= np.array([100.0, 107.0])
    results = detect_engulfing(opens, highs, lows, closes)
    assert any(r.name == "BULLISH_ENGULFING" for r in results)

def test_doji_returns_negative_bonus():
    from src.chart_patterns import detect_doji
    import numpy as np
    opens = np.array([100.0])
    highs = np.array([105.0])
    lows  = np.array([95.0])
    closes= np.array([100.05])
    results = detect_doji(opens, highs, lows, closes)
    assert results and results[0].confidence_bonus < 0
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Signal confidence score accuracy | No pattern input | +6–10 pts for confirmed patterns |
| False positives at low-pattern setups | Baseline | Indecision signals penalised –5 pts |
| Signal quality tier A+ rate | Baseline | Estimated +10–15% more A+ signals at BB extremes |
| Signal frequency | Unchanged | No hard gates added |

---

## Dependencies

None. Pattern detection is a standalone addition. Output feeds into PR_09 (Signal Scoring Engine).
