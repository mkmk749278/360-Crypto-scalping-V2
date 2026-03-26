# PR_09 — Signal Scoring Engine

**Branch:** `feature/pr09-signal-scoring`  
**Priority:** 9  
**Effort estimate:** Medium-Large (3–4 days)

---

## Objective

Replace the ad-hoc confidence score currently accumulated through additive soft penalties
and partial bonuses across `scanner.py` with a **structured composite signal scoring
engine** that produces a deterministic, auditable 0–100 score. The score is computed
from six dimension sub-scores, each with a defined maximum contribution.

| Dimension | Max pts | Source |
|-----------|---------|--------|
| SMC confluence | 25 | Sweep quality, FVG proximity, MSS body size |
| Regime alignment | 20 | Does the regime favour this setup type? |
| Volume confirmation | 15 | Relative volume vs 20-period average |
| Indicator confluence | 20 | MACD + RSI + EMA alignment all agree |
| Candlestick patterns | 10 | From PR_05 PatternResult list |
| MTF confirmation | 10 | From PR_06 mtf_score |

Signals score ≥ 80 → tier `A+`; 65–79 → `B`; 50–64 → `WATCHLIST`; <50 → `FILTERED` (not emitted).

The existing `src/confidence.py` scoring system remains as-is for the live confidence
display. The new engine sets `Signal.confidence` (used for tier gating) and writes a
detailed breakdown to `Signal.component_scores`.

---

## Files to Change

| File | Change type |
|------|-------------|
| `src/signal_quality.py` | Add `SignalScoringEngine` class with `score()` method |
| `src/scanner.py` | Call `SignalScoringEngine.score()` after signal creation; use result for tier gating |
| `src/channels/base.py` | No change (component_scores and signal_tier fields already exist) |
| `tests/test_signal_quality.py` | Add tests for each scoring dimension |

---

## Implementation Steps

### Step 1 — Create `SignalScoringEngine` in `src/signal_quality.py`

```python
"""Composite signal scoring engine.

Computes a structured 0–100 score for each signal across six dimensions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class ScoringInput:
    """All data needed to score a signal."""
    # SMC
    sweeps: list = None                  # List of LiquiditySweep objects
    mss: object = None                   # MSSSignal or None
    fvg_zones: list = None               # List of FVGZone objects
    # Regime
    regime: str = ""
    setup_class: str = ""
    atr_percentile: float = 50.0
    # Volume
    volume_last_usd: float = 0.0         # Last candle USD volume
    volume_avg_usd: float = 0.0          # 20-period average USD volume
    # Indicators
    macd_histogram_last: Optional[float] = None
    macd_histogram_prev: Optional[float] = None
    rsi_last: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    adx_last: Optional[float] = None
    direction: str = "LONG"
    # Pattern + MTF
    chart_patterns: list = None          # List of PatternResult objects
    mtf_score: float = 0.0              # 0.0–1.0 from MTF gate


class SignalScoringEngine:
    """Scores a candidate signal across six dimensions."""

    # Setup classes that strongly align with each regime
    _REGIME_SETUP_AFFINITY: Dict[str, List[str]] = {
        "TRENDING_UP": ["LIQUIDITY_SWEEP_REVERSAL", "BREAKOUT_INITIAL", "BREAKOUT_RETEST",
                        "THREE_WHITE_SOLDIERS", "WHALE_MOMENTUM"],
        "TRENDING_DOWN": ["LIQUIDITY_SWEEP_REVERSAL", "BREAKOUT_INITIAL", "BREAKOUT_RETEST",
                          "THREE_BLACK_CROWS", "WHALE_MOMENTUM"],
        "RANGING": ["RANGE_FADE", "SWING_STANDARD"],
        "QUIET": ["RANGE_FADE"],
        "VOLATILE": ["WHALE_MOMENTUM", "LIQUIDITY_SWEEP_REVERSAL"],
    }

    def score(self, inp: ScoringInput) -> Dict[str, float]:
        """Return a dict with per-dimension scores and a 'total' key.

        All scores are in [0, max_for_dimension].
        """
        smc = self._score_smc(inp)
        regime = self._score_regime(inp)
        volume = self._score_volume(inp)
        indicators = self._score_indicators(inp)
        patterns = self._score_patterns(inp)
        mtf = self._score_mtf(inp)
        total = smc + regime + volume + indicators + patterns + mtf
        return {
            "smc": round(smc, 2),
            "regime": round(regime, 2),
            "volume": round(volume, 2),
            "indicators": round(indicators, 2),
            "patterns": round(patterns, 2),
            "mtf": round(mtf, 2),
            "total": round(min(100.0, total), 2),
        }

    # ------------------------------------------------------------------
    def _score_smc(self, inp: ScoringInput) -> float:
        """SMC confluence score, max 25 pts."""
        score = 0.0
        sweeps = inp.sweeps or []
        if sweeps:
            score += 10.0   # Base for any sweep
            # Quality bonus: if sweep is recent (index = -1 or -2) add 5 pts
            if sweeps[0].index >= -3:
                score += 5.0
        if inp.mss is not None:
            score += 8.0    # MSS confirmation adds weight
        fvg = inp.fvg_zones or []
        if fvg:
            score += 2.0    # FVG presence (minor confluence)
        return min(25.0, score)

    # ------------------------------------------------------------------
    def _score_regime(self, inp: ScoringInput) -> float:
        """Regime alignment score, max 20 pts."""
        if not inp.regime:
            return 10.0   # Neutral when no regime data
        affinity = self._REGIME_SETUP_AFFINITY.get(inp.regime.upper(), [])
        if inp.setup_class in affinity:
            base = 18.0   # Strong alignment
        elif affinity:
            base = 8.0    # Regime known but setup not optimal
        else:
            base = 10.0   # Unknown regime
        # Bonus for high ATR percentile in VOLATILE regime (energy behind the move)
        if inp.regime.upper() == "VOLATILE" and inp.atr_percentile >= 75:
            base = min(20.0, base + 2.0)
        return min(20.0, base)

    # ------------------------------------------------------------------
    def _score_volume(self, inp: ScoringInput) -> float:
        """Volume confirmation score, max 15 pts."""
        if inp.volume_avg_usd <= 0 or inp.volume_last_usd <= 0:
            return 7.5    # Neutral
        ratio = inp.volume_last_usd / inp.volume_avg_usd
        if ratio >= 3.0:
            return 15.0
        if ratio >= 2.0:
            return 12.0
        if ratio >= 1.5:
            return 9.0
        if ratio >= 1.0:
            return 6.0
        return 3.0   # Below-average volume

    # ------------------------------------------------------------------
    def _score_indicators(self, inp: ScoringInput) -> float:
        """Indicator confluence score, max 20 pts."""
        score = 0.0
        checks = 0

        # MACD (max 7 pts)
        if inp.macd_histogram_last is not None and inp.macd_histogram_prev is not None:
            checks += 1
            rising = inp.macd_histogram_last > inp.macd_histogram_prev
            positive = inp.macd_histogram_last > 0
            if inp.direction == "LONG":
                if rising and positive:
                    score += 7.0
                elif rising or positive:
                    score += 4.0
            else:
                falling = not rising
                negative = inp.macd_histogram_last < 0
                if falling and negative:
                    score += 7.0
                elif falling or negative:
                    score += 4.0

        # RSI (max 7 pts)
        if inp.rsi_last is not None:
            checks += 1
            if inp.direction == "LONG":
                if inp.rsi_last <= 45:
                    score += 7.0    # Oversold or neutral — good for LONG
                elif inp.rsi_last <= 60:
                    score += 4.0
                else:
                    score += 1.0    # Overbought — risky
            else:
                if inp.rsi_last >= 55:
                    score += 7.0
                elif inp.rsi_last >= 40:
                    score += 4.0
                else:
                    score += 1.0

        # EMA alignment (max 6 pts)
        if inp.ema_fast is not None and inp.ema_slow is not None:
            checks += 1
            aligned = (inp.ema_fast > inp.ema_slow if inp.direction == "LONG"
                       else inp.ema_fast < inp.ema_slow)
            score += 6.0 if aligned else 1.0

        return min(20.0, score)

    # ------------------------------------------------------------------
    def _score_patterns(self, inp: ScoringInput) -> float:
        """Candlestick pattern score, max 10 pts."""
        patterns = inp.chart_patterns or []
        if not patterns:
            return 5.0   # Neutral (no patterns detected either way)
        aligned = [p for p in patterns
                   if getattr(p, "direction", "") == inp.direction or
                      getattr(p, "direction", "") == "NEUTRAL"]
        bonus = sum(getattr(p, "confidence_bonus", 0.0) for p in aligned)
        # Doji penalty: if any NEUTRAL/DOJI with negative bonus, reduce score
        return max(0.0, min(10.0, 5.0 + bonus * 0.5))

    # ------------------------------------------------------------------
    def _score_mtf(self, inp: ScoringInput) -> float:
        """MTF confirmation score, max 10 pts."""
        return round(inp.mtf_score * 10.0, 2)
```

### Step 2 — Wire into `src/scanner.py`

After `channel.evaluate()` returns a non-None signal:

```python
from src.signal_quality import SignalScoringEngine, ScoringInput

_scoring_engine = SignalScoringEngine()   # instantiated once at module level

scoring_input = ScoringInput(
    sweeps=smc_result.sweeps,
    mss=smc_result.mss,
    fvg_zones=smc_result.fvg,
    regime=regime_ctx.label,
    setup_class=sig.setup_class,
    atr_percentile=regime_ctx.atr_percentile,
    volume_last_usd=volume_last_usd,       # compute from last candle
    volume_avg_usd=volume_avg_usd,         # 20-period average
    macd_histogram_last=ind.get("macd_histogram_last"),
    macd_histogram_prev=ind.get("macd_histogram_prev"),
    rsi_last=ind.get("rsi_last"),
    ema_fast=ind.get("ema9_last"),
    ema_slow=ind.get("ema21_last"),
    adx_last=ind.get("adx_last"),
    direction=sig.direction.value,
    chart_patterns=smc_data.get("chart_patterns", []),
    mtf_score=sig.mtf_score,
)
score_result = _scoring_engine.score(scoring_input)
sig.component_scores = score_result
sig.confidence = score_result["total"]

# Apply tier gating
if score_result["total"] >= 80:
    sig.signal_tier = "A+"
elif score_result["total"] >= 65:
    sig.signal_tier = "B"
elif score_result["total"] >= 50:
    sig.signal_tier = "WATCHLIST"
else:
    sig = None   # Below threshold — filtered
```

### Step 3 — Tests

```python
def test_scoring_high_smc_and_volume():
    from src.signal_quality import SignalScoringEngine, ScoringInput
    from unittest.mock import MagicMock
    sweep = MagicMock(); sweep.index = -1
    engine = SignalScoringEngine()
    inp = ScoringInput(sweeps=[sweep], mss=MagicMock(), regime="TRENDING_UP",
                       setup_class="LIQUIDITY_SWEEP_REVERSAL",
                       volume_last_usd=3_000_000, volume_avg_usd=1_000_000,
                       macd_histogram_last=0.5, macd_histogram_prev=0.3,
                       rsi_last=42.0, ema_fast=101.0, ema_slow=100.0,
                       direction="LONG", mtf_score=1.0)
    result = engine.score(inp)
    # A well-confirmed signal should score >= 80; the scanner will set signal_tier = "A+"
    assert result["total"] >= 80

def test_scoring_filters_low_quality():
    from src.signal_quality import SignalScoringEngine, ScoringInput
    engine = SignalScoringEngine()
    inp = ScoringInput(regime="RANGING", setup_class="LIQUIDITY_SWEEP_REVERSAL",
                       volume_last_usd=500_000, volume_avg_usd=2_000_000,
                       rsi_last=68.0, ema_fast=100.0, ema_slow=101.0,
                       direction="LONG", mtf_score=0.0)
    result = engine.score(inp)
    assert result["total"] < 50
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Signal quality tier assignment | Based on ad-hoc confidence accumulation | Structured, auditable 0–100 score |
| False positive filter effectiveness | Variable | Deterministic (all signals < 50 pts filtered) |
| Component score transparency | None | Full breakdown in `Signal.component_scores` |
| A+ tier signal rate | Uncontrolled | ~20–25% of filtered signals |
| Win rate improvement | Baseline | Estimated +8–12% by filtering low-score signals |

---

## Dependencies

- **PR_04** — MACD histogram values needed for `indicators` dimension.
- **PR_05** — `chart_patterns` list from candlestick engine.
- **PR_06** — `mtf_score` from MTF gate.
