# PR_16 — Signal-Score-Weighted Position Sizing

**PR Number:** PR_16  
**Branch:** `feature/pr16-signal-score-weighted-position-sizing`  
**Category:** Risk & Reliability (Phase 2A)  
**Priority:** P1  
**Dependency:** PR_09 (Phase 1 — Signal Scoring Engine, merged as #135)  
**Effort estimate:** Small–Medium (1–2 days)

---

## Objective

Wire the composite signal score (0–100) produced by the PR_09 scoring engine into `risk.py::_position_size()` so that higher-conviction signals receive proportionally larger capital allocations. Low-conviction signals are still emitted but with reduced size rather than being suppressed, giving the system more granular control over capital deployment.

---

## Current State

- `risk.py::_position_size()` uses a fixed-risk formula (risk % × portfolio value ÷ SL distance) that does not incorporate the composite signal score.
- `calculate_position_size()` has a `confidence` parameter and multiplier, but it is not fully plumbed through the live signal pipeline — the score from the scoring engine is not passed to the risk module.
- High-score signals (90 pts) and low-score signals (55 pts) receive identical position sizes despite very different predicted success probabilities.

---

## Proposed Changes

### Modify `src/risk.py`

```python
# Scoring tiers for position sizing multiplier
_SCORE_TIERS = [
    (80, 1.00),   # Score 80–100 → 100% of base position
    (65, 0.75),   # Score 65–79  → 75%
    (50, 0.50),   # Score 50–64  → 50%
    (  0, 0.25),  # Score <50    → 25% (minimum; signal still emitted)
]

def _score_to_multiplier(score: float) -> float:
    """Convert a composite signal score (0–100) to a position size multiplier."""
    for threshold, multiplier in _SCORE_TIERS:
        if score >= threshold:
            return multiplier
    return 0.25

class RiskManager:
    # existing code ...

    def calculate_risk(self, signal, portfolio_value: float) -> dict:
        base_size = self._position_size(
            portfolio_value=portfolio_value,
            risk_pct=signal.config.risk_pct,
            sl_distance=signal.sl_distance,
        )
        score = getattr(signal, "post_ai_confidence", None) or getattr(signal, "confidence", 60.0)
        multiplier = _score_to_multiplier(float(score))
        adjusted_size = base_size * multiplier
        return {
            "position_size": adjusted_size,
            "score_multiplier": multiplier,
            "base_position_size": base_size,
        }
```

### Update `src/scanner.py` call sites

Ensure the signal's `post_ai_confidence` (set by the scoring engine in PR_09) is passed through to `RiskManager.calculate_risk()`. The scanner currently computes the score and attaches it to the signal; this PR ensures it is consumed by risk management before the signal is dispatched.

```python
# In scanner.py, after scoring:
risk_result = risk_manager.calculate_risk(signal, portfolio_value)
signal.position_size = risk_result["position_size"]
signal.score_multiplier = risk_result["score_multiplier"]  # for logging
```

### Config additions

```python
# config/__init__.py — allow override of tier thresholds
SCORE_TIER_HIGH:   int = int(os.getenv("SCORE_TIER_HIGH",   "80"))
SCORE_TIER_MEDIUM: int = int(os.getenv("SCORE_TIER_MEDIUM", "65"))
SCORE_TIER_LOW:    int = int(os.getenv("SCORE_TIER_LOW",    "50"))
```

---

## Implementation Steps

1. Add `_SCORE_TIERS` constant and `_score_to_multiplier()` helper to `src/risk.py`.
2. Modify `RiskManager.calculate_risk()` to read `signal.post_ai_confidence` and apply the multiplier.
3. In `scanner.py`, confirm `signal.post_ai_confidence` is set before the risk calculation call and pass result back to `signal.position_size`.
4. Add `score_multiplier` to `Signal` dataclass (or attach as a temporary attribute) for downstream logging.
5. Write unit tests in `tests/test_risk.py` extending the existing risk test coverage.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/risk.py` | Add `_SCORE_TIERS`, `_score_to_multiplier()`, update `calculate_risk()` |
| `src/scanner.py` | Pass score into `calculate_risk()`; write `position_size` back to signal |
| `src/config/__init__.py` | Add score tier threshold env-var overrides |
| `tests/test_risk.py` | Extend with score-weighted sizing tests |

---

## Testing Requirements

```python
# tests/test_risk.py (additions)
def test_high_score_full_position():
    rm = RiskManager(...)
    signal = make_signal(confidence=85.0, sl_distance=0.01)
    result = rm.calculate_risk(signal, portfolio_value=10_000)
    assert result["score_multiplier"] == 1.0

def test_medium_score_reduced_position():
    rm = RiskManager(...)
    signal = make_signal(confidence=70.0, sl_distance=0.01)
    result = rm.calculate_risk(signal, portfolio_value=10_000)
    assert result["score_multiplier"] == 0.75

def test_low_score_half_position():
    rm = RiskManager(...)
    signal = make_signal(confidence=55.0, sl_distance=0.01)
    result = rm.calculate_risk(signal, portfolio_value=10_000)
    assert result["score_multiplier"] == 0.50

def test_below_minimum_score_quarter_position():
    rm = RiskManager(...)
    signal = make_signal(confidence=40.0, sl_distance=0.01)
    result = rm.calculate_risk(signal, portfolio_value=10_000)
    assert result["score_multiplier"] == 0.25

def test_missing_score_defaults_to_medium():
    rm = RiskManager(...)
    signal = make_signal(confidence=None, sl_distance=0.01)
    result = rm.calculate_risk(signal, portfolio_value=10_000)
    # Default 60 → tier 50–64 → 0.50
    assert result["score_multiplier"] == 0.50
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Capital deployed per high-score signal | Same as low-score | 2× more than minimum-score signal |
| Win-rate-weighted return | Flat across score bands | Increases with score-outcome correlation |
| Max capital at risk on low-conviction signals | Full base risk % | 25% of base risk % |
| Implementation complexity | Low | Very low — single function addition |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `post_ai_confidence` not set on all signal paths | Use `getattr(signal, "post_ai_confidence", 60.0)` as safe default |
| Score inflation causing over-sizing | Tier thresholds configurable via env vars; can tighten during testing |
| Reduced position size on borderline signals affects profitability | Monitor via KPI dashboard (PR_24); adjust tiers if needed |
