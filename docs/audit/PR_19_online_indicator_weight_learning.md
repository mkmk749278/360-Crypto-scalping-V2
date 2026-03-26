# PR_19 — Online Indicator Weight Learning

**PR Number:** PR_19  
**Branch:** `feature/pr19-online-indicator-weight-learning`  
**Category:** Signal Intelligence (Phase 2B)  
**Priority:** P2  
**Dependency:** PR_12 (Phase 1 — AI Statistical Filter, merged as #138)  
**Effort estimate:** Large (3–5 days)

---

## Objective

Replace static weight tables in `signal_params.py` with an EWMA (Exponentially Weighted Moving Average) logistic regression that automatically adjusts indicator weights per (channel, pair, regime) based on closed trade outcomes. After 200+ trades, indicators with consistently negative contribution are auto-demoted. Static tables serve as a cold-start fallback.

---

## Current State

`src/signal_params.py` contains static lookup tables mapping `(channel, setup_class, regime)` → weight overrides for each indicator (MACD, RSI, SMC, candlestick, MTF, volume, etc.). These weights were set manually during the initial audit and have not been updated based on actual trade outcomes. Over time, market conditions shift and static weights become misaligned with reality.

---

## Proposed Changes

### New file: `src/weight_learner.py`

```python
"""Online EWMA-based indicator weight learner for signal quality adaptation."""
from __future__ import annotations
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EWMA_ALPHA        = 0.05    # exponential decay — recent trades weighted more
MIN_SAMPLES       = 200     # minimum trades before learning overrides statics
DEMOTION_ALPHA    = -0.02   # auto-demote if weight contribution is this negative
WEIGHT_FLOOR      = 0.10    # minimum weight multiplier (never zero)
WEIGHT_CEILING    = 2.00    # maximum weight multiplier

_WeightKey = Tuple[str, str, str]  # (channel, pair, regime)

@dataclass
class IndicatorStats:
    total_trades: int = 0
    ewma_contribution: float = 0.0   # running EWMA of (indicator_value × outcome)
    weight_multiplier: float = 1.0   # current learned multiplier

class WeightLearner:
    """
    Lightweight online learning layer for indicator weights.
    Persists state to a JSON file for survival across restarts.
    """

    def __init__(self, state_path: str = "data/weight_learner_state.json"):
        self._state_path = Path(state_path)
        self._stats: Dict[_WeightKey, Dict[str, IndicatorStats]] = defaultdict(dict)
        self._load()

    def record_outcome(
        self,
        channel: str,
        symbol: str,
        regime: str,
        indicator_values: Dict[str, float],   # normalised 0–1 values per indicator
        outcome: float,                        # +1 = win, -1 = loss
    ) -> None:
        """Update EWMA stats for each indicator after a trade closes."""
        key: _WeightKey = (channel, symbol, regime)
        for name, value in indicator_values.items():
            if name not in self._stats[key]:
                self._stats[key][name] = IndicatorStats()
            stats = self._stats[key][name]
            contribution = value * outcome
            stats.ewma_contribution = (
                EWMA_ALPHA * contribution
                + (1 - EWMA_ALPHA) * stats.ewma_contribution
            )
            stats.total_trades += 1
            if stats.total_trades >= MIN_SAMPLES:
                stats.weight_multiplier = self._compute_multiplier(stats.ewma_contribution)
        self._save()

    def get_multiplier(
        self,
        channel: str,
        symbol: str,
        regime: str,
        indicator: str,
    ) -> float:
        """
        Return the learned weight multiplier for an indicator.
        Falls back to 1.0 (static weights) until MIN_SAMPLES is reached.
        """
        key: _WeightKey = (channel, symbol, regime)
        stats = self._stats.get(key, {}).get(indicator)
        if stats is None or stats.total_trades < MIN_SAMPLES:
            return 1.0   # cold start: use static weights unchanged
        return stats.weight_multiplier

    def _compute_multiplier(self, ewma_contribution: float) -> float:
        """Map EWMA contribution to a weight multiplier in [WEIGHT_FLOOR, WEIGHT_CEILING]."""
        if ewma_contribution <= DEMOTION_ALPHA:
            return WEIGHT_FLOOR
        # Linear mapping: 0 → 1.0, +0.1 → 2.0
        raw = 1.0 + ewma_contribution * 10.0
        return float(np.clip(raw, WEIGHT_FLOOR, WEIGHT_CEILING))

    def _save(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, "w") as f:
                json.dump(self._serialise(), f, indent=2)
        except Exception as exc:
            logger.warning("WeightLearner: failed to save state: %s", exc)

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                self._deserialise(json.load(f))
        except Exception as exc:
            logger.warning("WeightLearner: failed to load state: %s", exc)

    def _serialise(self) -> dict:
        out = {}
        for key, indicators in self._stats.items():
            k = "|".join(key)
            out[k] = {
                name: {
                    "total_trades": s.total_trades,
                    "ewma_contribution": s.ewma_contribution,
                    "weight_multiplier": s.weight_multiplier,
                }
                for name, s in indicators.items()
            }
        return out

    def _deserialise(self, data: dict) -> None:
        for k, indicators in data.items():
            key = tuple(k.split("|", 2))
            self._stats[key] = {
                name: IndicatorStats(**vals)
                for name, vals in indicators.items()
            }
```

### Integrate with `src/signal_params.py`

```python
# In signal_params.py, after looking up static weights:
from src.weight_learner import WeightLearner
_learner: Optional[WeightLearner] = None

def get_learned_weights(
    channel: str,
    symbol: str,
    regime: str,
    static_weights: dict,
) -> dict:
    """Apply learned multipliers on top of static base weights."""
    if _learner is None:
        return static_weights
    return {
        indicator: weight * _learner.get_multiplier(channel, symbol, regime, indicator)
        for indicator, weight in static_weights.items()
    }
```

### Wire into `src/stat_filter.py` (trade close callback)

```python
# When a trade closes with a known outcome:
from src.weight_learner import _learner

def on_trade_closed(channel, symbol, regime, indicator_values, pnl_pct):
    outcome = 1.0 if pnl_pct > 0 else -1.0
    if _learner:
        _learner.record_outcome(channel, symbol, regime, indicator_values, outcome)
```

---

## Implementation Steps

1. Create `src/weight_learner.py` with `WeightLearner` class and `IndicatorStats` dataclass.
2. Add `_learner` singleton initialisation in `main.py`.
3. Modify `signal_params.py` to call `get_learned_weights()` after static lookup.
4. In `stat_filter.py` (or `trade_observer.py`), hook `on_trade_closed` to call `_learner.record_outcome()`.
5. Ensure `indicator_values` dict is populated before signal scoring (normalise each indicator to 0–1 range).
6. Write unit tests in `tests/test_weight_learner.py`.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/weight_learner.py` | New — `WeightLearner` class |
| `src/signal_params.py` | Add `get_learned_weights()` wrapper |
| `src/stat_filter.py` | Add `on_trade_closed` callback |
| `src/main.py` | Instantiate `WeightLearner` singleton |
| `tests/test_weight_learner.py` | New test file |

---

## Testing Requirements

```python
# tests/test_weight_learner.py
def test_cold_start_returns_1():
    wl = WeightLearner(state_path="/tmp/wl_test.json")
    assert wl.get_multiplier("SCALP", "BTCUSDT", "TRENDING_UP", "rsi") == 1.0

def test_negative_contribution_demotes_weight():
    wl = WeightLearner(state_path="/tmp/wl_test.json")
    for _ in range(MIN_SAMPLES + 10):
        wl.record_outcome("SCALP", "BTCUSDT", "TRENDING_UP",
                          {"rsi": 0.8}, outcome=-1.0)
    mult = wl.get_multiplier("SCALP", "BTCUSDT", "TRENDING_UP", "rsi")
    assert mult == WEIGHT_FLOOR

def test_positive_contribution_boosts_weight():
    wl = WeightLearner(state_path="/tmp/wl_test.json")
    for _ in range(MIN_SAMPLES + 10):
        wl.record_outcome("SCALP", "BTCUSDT", "TRENDING_UP",
                          {"macd": 0.9}, outcome=+1.0)
    mult = wl.get_multiplier("SCALP", "BTCUSDT", "TRENDING_UP", "macd")
    assert mult > 1.0

def test_state_survives_reload(tmp_path):
    path = str(tmp_path / "state.json")
    wl = WeightLearner(state_path=path)
    for _ in range(MIN_SAMPLES + 10):
        wl.record_outcome("SCALP", "BTCUSDT", "TRENDING_UP", {"rsi": 0.5}, 1.0)
    wl2 = WeightLearner(state_path=path)
    assert wl2._stats  # state loaded
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Indicator weight adaptability | Static (manual update required) | Auto-updated after each trade |
| Weight accuracy over time | Degrades as market shifts | Continuously improves |
| Cold-start behaviour | Immediate (static tables) | Static tables used until 200 trades |
| Per-pair regime learning | None | Separate model per (channel, pair, regime) |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| EWMA overfits to recent losing streak | Alpha = 0.05 means very slow adaptation; 200 trade minimum |
| State file corruption | Try/except on load; reinitialise from scratch if corrupt |
| Indicator values not normalised | Normalisation must happen before `record_outcome()`; add assertion |
| Learning diverges if outcome labels are wrong | Validate outcome = ±1 before recording |
