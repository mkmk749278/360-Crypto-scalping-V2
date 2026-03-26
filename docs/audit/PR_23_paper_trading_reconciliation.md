# PR_23 — Paper Trading Reconciliation

**PR Number:** PR_23  
**Branch:** `feature/pr23-paper-trading-reconciliation`  
**Category:** Backtesting Maturity (Phase 2C)  
**Priority:** P2  
**Dependency:** PR_21 (Realistic Slippage & Fee Model)  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Continuously validate that backtest-predicted performance matches actual paper-trading outcomes. When the divergence between predicted and observed metrics exceeds 2 standard deviations over a 30-trade rolling window, send an automated alert. This closes the feedback loop between the backtester and the live paper-trading environment.

---

## Current State

`src/paper_portfolio.py` tracks paper-trading positions and computes running P&L, win rate, and Sharpe ratio. However:
- There is no comparison between paper-trading outcomes and what the backtester predicted for those same signals.
- Divergence between backtest and paper-trading performance is not monitored.
- No automated alert exists when the two diverge significantly.

---

## Proposed Changes

### New file: `src/reconciliation.py`

```python
"""Paper trading reconciliation — compare backtest predictions with live outcomes."""
from __future__ import annotations
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

RECONCILIATION_WINDOW       = 30     # rolling trades to compare
DIVERGENCE_SIGMA_THRESHOLD  = 2.0   # alert if divergence > 2σ

@dataclass
class SignalPrediction:
    """Backtest-derived prediction for a signal before it fires live."""
    signal_id: str
    predicted_pnl_pct: float    # expected P&L% from backtest simulation
    predicted_win_prob: float   # win probability from scoring engine
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class SignalOutcome:
    """Actual outcome after the trade closes in paper portfolio."""
    signal_id: str
    actual_pnl_pct: float
    closed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class PaperTradingValidator:
    """
    Records backtest predictions and actual paper-trading outcomes.
    Computes rolling divergence and raises alerts when it exceeds 2σ.
    """

    def __init__(
        self,
        window: int = RECONCILIATION_WINDOW,
        sigma_threshold: float = DIVERGENCE_SIGMA_THRESHOLD,
        alert_callback=None,
    ):
        self._window = window
        self._sigma = sigma_threshold
        self._alert_callback = alert_callback
        self._predictions: Dict[str, SignalPrediction] = {}
        self._errors: deque = deque(maxlen=window)  # prediction_error per trade

    def record_prediction(self, prediction: SignalPrediction) -> None:
        """Store a backtest prediction before the signal fires."""
        self._predictions[prediction.signal_id] = prediction

    def record_outcome(self, outcome: SignalOutcome) -> None:
        """Record the actual trade outcome and compute prediction error."""
        pred = self._predictions.pop(outcome.signal_id, None)
        if pred is None:
            logger.debug("No prediction found for signal_id=%s", outcome.signal_id)
            return

        error = outcome.actual_pnl_pct - pred.predicted_pnl_pct
        self._errors.append(error)
        logger.info(
            "Reconciliation: signal=%s predicted=%.2f%% actual=%.2f%% error=%.2f%%",
            outcome.signal_id,
            pred.predicted_pnl_pct * 100,
            outcome.actual_pnl_pct * 100,
            error * 100,
        )

        if len(self._errors) >= self._window:
            self._check_divergence()

    def _check_divergence(self) -> None:
        """Alert if rolling prediction error exceeds sigma threshold."""
        errors = np.array(list(self._errors))
        mean_error = float(errors.mean())
        std_error = float(errors.std())
        if std_error == 0:
            return

        z_score = abs(mean_error / std_error)
        if z_score > self._sigma:
            msg = (
                f"⚠️ Backtest-to-paper divergence alert!\n"
                f"Rolling {self._window}-trade mean error: {mean_error*100:+.2f}%\n"
                f"Z-score: {z_score:.2f}σ (threshold: {self._sigma}σ)\n"
                f"Action: review slippage model and parameter calibration."
            )
            logger.warning(msg)
            if self._alert_callback:
                self._alert_callback(msg)

    def summary(self) -> dict:
        """Return current reconciliation statistics."""
        errors = list(self._errors)
        if not errors:
            return {"n": 0, "mean_error": None, "std_error": None}
        arr = np.array(errors)
        return {
            "n": len(errors),
            "mean_error_pct": float(arr.mean() * 100),
            "std_error_pct": float(arr.std() * 100),
            "window": self._window,
        }
```

### Integrate into `src/paper_portfolio.py`

```python
from src.reconciliation import PaperTradingValidator, SignalOutcome

class PaperPortfolio:
    def __init__(self, ..., validator: Optional[PaperTradingValidator] = None):
        self._validator = validator
        # existing init ...

    def close_trade(self, signal_id: str, actual_pnl_pct: float) -> None:
        # existing close logic ...
        if self._validator:
            self._validator.record_outcome(
                SignalOutcome(signal_id=signal_id, actual_pnl_pct=actual_pnl_pct)
            )
```

### Wire prediction recording in scanner/signal dispatch

```python
# When a signal is dispatched (scanner.py or signal_router.py):
from src.reconciliation import SignalPrediction

if validator and backtest_prediction:
    validator.record_prediction(SignalPrediction(
        signal_id=signal.id,
        predicted_pnl_pct=backtest_prediction.expected_pnl_pct,
        predicted_win_prob=signal.post_ai_confidence / 100.0,
    ))
```

---

## Implementation Steps

1. Create `src/reconciliation.py` with `PaperTradingValidator`, `SignalPrediction`, `SignalOutcome`.
2. Add optional `validator` parameter to `PaperPortfolio.__init__()`.
3. Call `validator.record_outcome()` on every trade close in `paper_portfolio.py`.
4. In the signal dispatch path, call `validator.record_prediction()` when a signal fires.
5. Instantiate `PaperTradingValidator` in `main.py` with the Telegram alert callback.
6. Write unit tests in `tests/test_reconciliation.py`.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/reconciliation.py` | New — `PaperTradingValidator`, `SignalPrediction`, `SignalOutcome` |
| `src/paper_portfolio.py` | Accept optional `validator`; call `record_outcome()` on close |
| `src/main.py` | Instantiate `PaperTradingValidator`; pass to `PaperPortfolio` |
| `tests/test_reconciliation.py` | New test file |

---

## Testing Requirements

```python
# tests/test_reconciliation.py
def test_no_divergence_no_alert():
    alerts = []
    v = PaperTradingValidator(window=5, alert_callback=alerts.append)
    for i in range(5):
        v.record_prediction(SignalPrediction(f"sig-{i}", predicted_pnl_pct=0.01,
                                              predicted_win_prob=0.6))
        v.record_outcome(SignalOutcome(f"sig-{i}", actual_pnl_pct=0.01))
    assert not alerts

def test_divergence_triggers_alert():
    alerts = []
    v = PaperTradingValidator(window=5, sigma_threshold=1.0, alert_callback=alerts.append)
    for i in range(5):
        v.record_prediction(SignalPrediction(f"sig-{i}", predicted_pnl_pct=0.02,
                                              predicted_win_prob=0.7))
        v.record_outcome(SignalOutcome(f"sig-{i}", actual_pnl_pct=-0.01))   # consistently worse
    assert alerts

def test_missing_prediction_ignored():
    v = PaperTradingValidator(window=10)
    v.record_outcome(SignalOutcome("nonexistent", actual_pnl_pct=0.01))
    assert v.summary()["n"] == 0

def test_summary_after_trades():
    v = PaperTradingValidator(window=5)
    for i in range(3):
        v.record_prediction(SignalPrediction(f"sig-{i}", 0.01, 0.6))
        v.record_outcome(SignalOutcome(f"sig-{i}", 0.015))
    s = v.summary()
    assert s["n"] == 3
    assert s["mean_error_pct"] > 0
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Backtest-to-live gap visibility | Manual and delayed | Automated rolling 30-trade check |
| Response time to model drift | Hours/days | Minutes (15-min monitoring cycle via PR_25) |
| Slippage model calibration | Set once | Validated continuously against live data |
| Admin awareness of divergence | None | Telegram alert at 2σ |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Signal ID mismatch between prediction and outcome | Use UUID for signal ID; validate format on record |
| Old predictions accumulate if trade never closes | Add TTL (48h) to `_predictions` dict; evict stale entries |
| 30-trade window too small → noisy statistics | Window configurable; document that 100+ trades gives stable results |
| Backtest prediction not always available for every live signal | Make prediction optional; skip recording if not available |
