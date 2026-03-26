# PR_22 — Monte Carlo Equity Simulation

**PR Number:** PR_22  
**Branch:** `feature/pr22-monte-carlo-equity-simulation`  
**Category:** Backtesting Maturity (Phase 2C)  
**Priority:** P1  
**Dependency:** PR_21 (Realistic Slippage & Fee Model)  
**Effort estimate:** Medium (2–3 days)

---

## Objective

After every standard backtest run, execute 1 000 randomised Monte Carlo simulations to produce P5/P50/P95 confidence intervals for final equity, maximum drawdown, and Sharpe ratio. This transforms a single-path backtest result into a probabilistic range, exposing sequence-of-returns risk and parameter sensitivity that a deterministic replay cannot reveal.

---

## Current State

`src/backtester.py` runs a single sequential replay of historical trade outcomes in chronological order. The result is a single data point per metric (final equity, Sharpe, max drawdown), which tells you nothing about:
- What happens if the first 10 trades are losses (sequence risk).
- How much variation exists if 10% of signals fail to fill.
- The confidence interval around the Sharpe ratio.

---

## Proposed Changes

### New file: `src/monte_carlo.py`

```python
"""Monte Carlo equity simulation for probabilistic backtest evaluation."""
from __future__ import annotations
import logging
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

N_SIMULATIONS   = 1_000
DROP_RATE       = 0.10   # randomly drop 10% of signals (execution failure)
ENTRY_EXIT_JITTER_ATR = 1.0  # vary entry/exit by ±1 ATR fraction

@dataclass
class TradeRecord:
    pnl_pct: float          # actual P&L as % of position
    atr_pct: float          # ATR at signal time as % of price (e.g. 0.005 = 0.5%)

@dataclass
class SimulationResult:
    final_equity_p5:   float
    final_equity_p50:  float
    final_equity_p95:  float
    max_drawdown_p5:   float
    max_drawdown_p50:  float
    max_drawdown_p95:  float
    sharpe_p5:         float
    sharpe_p50:        float
    sharpe_p95:        float
    n_simulations:     int

class MonteCarloSimulator:
    """
    Runs randomised simulations over a set of historical trade records.

    Each simulation:
    1. Randomly reorders trade outcomes (sequence-of-returns risk).
    2. Varies each entry/exit by ±1 ATR (execution uncertainty).
    3. Randomly drops DROP_RATE fraction of signals (execution failure).
    """

    def __init__(
        self,
        n_simulations: int = N_SIMULATIONS,
        drop_rate: float = DROP_RATE,
        jitter_atr: float = ENTRY_EXIT_JITTER_ATR,
        seed: Optional[int] = None,
    ):
        self._n = n_simulations
        self._drop_rate = drop_rate
        self._jitter_atr = jitter_atr
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def run(
        self,
        trades: List[TradeRecord],
        starting_equity: float = 10_000.0,
        risk_per_trade_pct: float = 0.01,
    ) -> SimulationResult:
        """Run N Monte Carlo simulations and return confidence intervals."""
        if not trades:
            raise ValueError("Cannot run Monte Carlo with empty trade list")

        final_equities: List[float] = []
        max_drawdowns:  List[float] = []
        sharpes:        List[float] = []

        for _ in range(self._n):
            sim_trades = self._perturb(trades)
            equity_curve = self._simulate_equity(sim_trades, starting_equity, risk_per_trade_pct)
            final_equities.append(equity_curve[-1])
            max_drawdowns.append(self._max_drawdown(equity_curve))
            sharpes.append(self._sharpe(equity_curve))

        fe = np.array(final_equities)
        md = np.array(max_drawdowns)
        sh = np.array(sharpes)

        return SimulationResult(
            final_equity_p5=float(np.percentile(fe, 5)),
            final_equity_p50=float(np.percentile(fe, 50)),
            final_equity_p95=float(np.percentile(fe, 95)),
            max_drawdown_p5=float(np.percentile(md, 5)),
            max_drawdown_p50=float(np.percentile(md, 50)),
            max_drawdown_p95=float(np.percentile(md, 95)),
            sharpe_p5=float(np.percentile(sh, 5)),
            sharpe_p50=float(np.percentile(sh, 50)),
            sharpe_p95=float(np.percentile(sh, 95)),
            n_simulations=self._n,
        )

    def _perturb(self, trades: List[TradeRecord]) -> List[TradeRecord]:
        """Apply randomisation: shuffle, jitter, drop."""
        shuffled = list(trades)
        random.shuffle(shuffled)
        result = []
        for t in shuffled:
            if random.random() < self._drop_rate:
                continue   # simulate execution failure
            jitter = random.uniform(-self._jitter_atr, self._jitter_atr) * t.atr_pct
            result.append(TradeRecord(pnl_pct=t.pnl_pct + jitter, atr_pct=t.atr_pct))
        return result

    def _simulate_equity(
        self,
        trades: List[TradeRecord],
        start: float,
        risk_pct: float,
    ) -> np.ndarray:
        equity = start
        curve = [equity]
        for t in trades:
            equity *= (1 + risk_pct * t.pnl_pct)
            curve.append(equity)
        return np.array(curve)

    @staticmethod
    def _max_drawdown(equity_curve: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return float(np.min(drawdown))

    @staticmethod
    def _sharpe(equity_curve: np.ndarray, periods_per_year: int = 252) -> float:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))
```

### Integrate into `src/backtester.py`

```python
from src.monte_carlo import MonteCarloSimulator, TradeRecord

class Backtester:
    # existing code ...

    def run_with_monte_carlo(
        self,
        signals,
        n_simulations: int = 1_000,
        seed: Optional[int] = 42,
    ) -> dict:
        """Run standard backtest then Monte Carlo simulations."""
        base_result = self.run(signals)
        trades = [
            TradeRecord(pnl_pct=t["pnl_pct"], atr_pct=t.get("atr_pct", 0.005))
            for t in base_result["trades"]
        ]
        mc = MonteCarloSimulator(n_simulations=n_simulations, seed=seed)
        mc_result = mc.run(trades, starting_equity=self._starting_equity)
        return {**base_result, "monte_carlo": mc_result}
```

---

## Implementation Steps

1. Create `src/monte_carlo.py` with `MonteCarloSimulator`, `TradeRecord`, and `SimulationResult`.
2. In `backtester.py`, add `run_with_monte_carlo()` method.
3. Ensure `BacktestResult.trades` contains `pnl_pct` and `atr_pct` per trade (extend trade record if needed).
4. Add MC configuration constants to `config/__init__.py`.
5. Write unit tests in `tests/test_monte_carlo.py`.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/monte_carlo.py` | New — `MonteCarloSimulator`, `TradeRecord`, `SimulationResult` |
| `src/backtester.py` | Add `run_with_monte_carlo()` method |
| `src/config/__init__.py` | Add MC simulation count and seed config constants |
| `tests/test_monte_carlo.py` | New test file |

---

## Testing Requirements

```python
# tests/test_monte_carlo.py
def make_trades(n=100, win_rate=0.55):
    trades = []
    for i in range(n):
        pnl = 1.5 if i / n < win_rate else -1.0
        trades.append(TradeRecord(pnl_pct=pnl, atr_pct=0.005))
    return trades

def test_simulation_returns_result():
    mc = MonteCarloSimulator(n_simulations=100, seed=42)
    result = mc.run(make_trades())
    assert isinstance(result, SimulationResult)
    assert result.n_simulations == 100

def test_p95_greater_than_p5():
    mc = MonteCarloSimulator(n_simulations=200, seed=42)
    result = mc.run(make_trades(win_rate=0.6))
    assert result.final_equity_p95 > result.final_equity_p5

def test_empty_trades_raises():
    mc = MonteCarloSimulator(n_simulations=10, seed=42)
    with pytest.raises(ValueError):
        mc.run([])

def test_drop_rate_reduces_trade_count():
    # With 100% drop rate, no trades → equity stays flat
    mc = MonteCarloSimulator(n_simulations=10, drop_rate=1.0, seed=42)
    result = mc.run(make_trades(n=50))
    assert result.final_equity_p50 == pytest.approx(10_000.0, rel=1e-3)
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Backtest output richness | Single P&L and Sharpe | P5/P50/P95 confidence intervals |
| Sequence-of-returns risk visibility | None | Exposed via equity curve variation |
| Parameter optimisation safety | Overfitting risk | MC confidence intervals reveal fragility |
| Execution failure simulation | None | 10% drop rate injection |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| 1 000 simulations slow on large trade sets | Vectorise with NumPy; target <5s for 1 000 trades × 1 000 sims |
| Non-reproducible results | Seed parameter (default 42) for reproducibility |
| Trade records missing `atr_pct` | Default to pair-tier typical ATR (0.3% MAJOR, 0.8% ALTCOIN) |
