# PR_11 — Backtester Per-Pair Regime Enhancement

**Branch:** `feature/pr11-backtester-per-pair-regime`  
**Priority:** 11  
**Effort estimate:** Large (4–5 days)

---

## Objective

Enhance `src/backtester.py` with three capabilities that transform it from a simple
single-pass signal replay tool into a rigorous parameter evaluation framework:

1. **Per-pair parameter sweeps** — run the backtest with different config parameter sets
   across different symbols and collect per-pair metrics.

2. **Regime-tagged results** — compute and record the market regime active during each
   historical signal, enabling performance slicing by regime (e.g., "win rate in TRENDING
   vs RANGING").

3. **Walk-forward validation** — split the test window into rolling in-sample / out-of-sample
   segments and report average out-of-sample performance to detect overfitting.

---

## Files to Change

| File | Change type |
|------|-------------|
| `src/backtester.py` | Add `BacktestConfig`, `RegimeTaggedResult`, `WalkForwardReport`; extend `Backtester.run()` |
| `src/regime.py` | Expose `detect_regime_from_arrays()` for use on historical candle arrays |
| `tests/test_backtester.py` | Add tests for per-pair sweep, regime tagging, walk-forward |

---

## Implementation Steps

### Step 1 — Add `BacktestConfig` dataclass to `src/backtester.py`

```python
@dataclass
class BacktestConfig:
    """Parameter set for a single backtest run.

    Allows sweeping across different threshold combinations per pair.
    """
    channel_name: str = "360_SCALP"
    atr_sl_mult: float = 1.0          # Multiplier for ATR-based SL
    tp_ratios: tuple = (0.5, 1.0, 1.5)
    min_adx: float = 20.0
    momentum_threshold_mult: float = 1.0
    slippage_pct: float = 0.02         # Execution slippage per side
    fee_pct: float = 0.04              # Taker fee per side (BPS)
    max_hold_candles: int = 50         # Maximum bars before forced close
    regime_filter: str = ""            # Only test signals in this regime ("" = all)
    pair: str = ""                     # Symbol being tested
```

### Step 2 — Add `RegimeTaggedResult` dataclass

```python
@dataclass
class RegimeTaggedResult:
    """Backtest result for a single signal with its regime context."""
    signal_id: str
    pair: str
    regime: str
    setup_class: str
    outcome: str            # "WIN", "LOSS", "PARTIAL", "EXPIRED"
    pnl_pct: float
    hold_candles: int
    entry_price: float
    sl_price: float
    tp1_price: float
    hit_tp: int             # 0=none, 1=TP1, 2=TP2, 3=TP3
    atr_at_entry: float
    atr_percentile: float
```

### Step 3 — Add `WalkForwardReport` dataclass

```python
@dataclass
class WalkForwardReport:
    """Walk-forward validation summary."""
    n_folds: int
    fold_results: list          # List of (in_sample_winrate, out_sample_winrate) tuples
    avg_in_sample_winrate: float
    avg_out_sample_winrate: float
    overfit_score: float        # (in - out) / in; > 0.15 indicates likely overfitting

    def summary(self) -> str:
        return (
            f"Walk-Forward: {self.n_folds} folds\n"
            f"In-Sample WR:  {self.avg_in_sample_winrate:.1f}%\n"
            f"Out-Sample WR: {self.avg_out_sample_winrate:.1f}%\n"
            f"Overfit Score: {self.overfit_score:.3f}"
            + (" ⚠️ OVERFITTING DETECTED" if self.overfit_score > 0.15 else " ✅ OK")
        )
```

### Step 4 — Add `detect_regime_from_arrays()` to `src/regime.py`

Expose a vectorised regime detection function that accepts numpy arrays directly (no
candle dict), suitable for historical replay:

```python
def detect_regime_from_arrays(
    closes: "np.ndarray",
    highs: "np.ndarray",
    lows: "np.ndarray",
    volumes: "np.ndarray",
    idx: int,
    lookback: int = 14,
) -> str:
    """Detect market regime at a specific bar index in a historical array.

    Parameters
    ----------
    closes, highs, lows, volumes:
        Full historical arrays (length >= idx + lookback + 1).
    idx:
        Bar index for which to detect the regime.
    lookback:
        ATR/ADX computation lookback.

    Returns
    -------
    str: regime label ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "QUIET")
    """
    import numpy as np
    from src.indicators import adx as compute_adx, atr as compute_atr, ema as compute_ema

    start = max(0, idx - lookback * 3)
    end = idx + 1
    c = closes[start:end]
    h = highs[start:end]
    l = lows[start:end]

    if len(c) < lookback * 2:
        return "RANGING"  # Default when not enough data

    adx_series = compute_adx(h, l, c, period=lookback)
    atr_series = compute_atr(h, l, c, period=lookback)
    ema9 = compute_ema(c, 9)
    ema21 = compute_ema(c, 21)

    adx_val = float(adx_series[-1]) if not np.isnan(adx_series[-1]) else 0.0
    atr_val = float(atr_series[-1]) if not np.isnan(atr_series[-1]) else 0.0
    price = float(c[-1])
    atr_pct = (atr_val / price * 100) if price > 0 else 0.5

    if adx_val >= 25:
        return "TRENDING_UP" if float(ema9[-1]) > float(ema21[-1]) else "TRENDING_DOWN"
    if atr_pct >= 1.5:
        return "VOLATILE"
    if atr_pct <= 0.3:
        return "QUIET"
    return "RANGING"
```

### Step 5 — Extend `Backtester.run()` with regime tagging

Add an optional `tag_regimes: bool = False` parameter. When True, call
`detect_regime_from_arrays()` at each signal detection bar and attach the result
to the signal detail dict:

```python
def run(
    self,
    historical_data: Dict,
    config: Optional[BacktestConfig] = None,
    tag_regimes: bool = False,
) -> BacktestResult:
    ...
    for i in range(window, len(close_arr)):
        # ... existing evaluation logic ...
        if sig is not None:
            regime_at_entry = ""
            if tag_regimes:
                regime_at_entry = detect_regime_from_arrays(
                    close_arr, high_arr, low_arr, vol_arr, i
                )
            detail = {
                ...existing fields...,
                "regime": regime_at_entry,
                "setup_class": getattr(sig, "setup_class", ""),
                "atr_at_entry": float(atr_arr[i]) if i < len(atr_arr) else 0.0,
            }
```

### Step 6 — Implement `run_per_pair_sweep()` method

```python
def run_per_pair_sweep(
    self,
    data_by_pair: Dict[str, Dict],   # symbol → historical candle dict
    configs: List[BacktestConfig],
) -> Dict[str, List[BacktestResult]]:
    """Run each config against each pair and return a nested results dict."""
    results = {}
    for pair, historical_data in data_by_pair.items():
        results[pair] = []
        for cfg in configs:
            cfg.pair = pair
            result = self.run(historical_data, config=cfg, tag_regimes=True)
            result.channel = f"{result.channel}[{pair}]"
            results[pair].append(result)
    return results
```

### Step 7 — Implement `walk_forward_validate()` method

> **Note:** `BacktestResult.win_rate` is already defined as a `float` field in the existing
> `src/backtester.py:BacktestResult` dataclass. The `walk_forward_validate()` method relies
> on this field being populated by the standard `Backtester.run()` execution path.

```python
def walk_forward_validate(
    self,
    historical_data: Dict,
    n_folds: int = 5,
    train_pct: float = 0.7,
    config: Optional[BacktestConfig] = None,
) -> WalkForwardReport:
    """Run rolling walk-forward validation over historical_data.

    The full dataset is split into n_folds segments. For each fold:
    - In-sample: first train_pct of the fold
    - Out-of-sample: remaining (1 - train_pct) of the fold
    """
    import numpy as np
    close_arr = np.asarray(historical_data.get("close", []))
    n = len(close_arr)
    fold_size = n // n_folds
    fold_results = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size
        fold_data = {k: v[start:end] for k, v in historical_data.items()
                     if isinstance(v, (list, np.ndarray))}
        split = int(fold_size * train_pct)
        in_data = {k: v[:split] for k, v in fold_data.items()
                   if isinstance(v, (list, np.ndarray))}
        out_data = {k: v[split:] for k, v in fold_data.items()
                    if isinstance(v, (list, np.ndarray))}
        in_result = self.run(in_data, config=config)
        out_result = self.run(out_data, config=config)
        fold_results.append((in_result.win_rate, out_result.win_rate))

    avg_in = float(np.mean([f[0] for f in fold_results]))
    avg_out = float(np.mean([f[1] for f in fold_results]))
    overfit = (avg_in - avg_out) / avg_in if avg_in > 0 else 0.0

    return WalkForwardReport(
        n_folds=n_folds,
        fold_results=fold_results,
        avg_in_sample_winrate=avg_in,
        avg_out_sample_winrate=avg_out,
        overfit_score=overfit,
    )
```

### Step 8 — Tests (`tests/test_backtester.py`)

```python
def test_regime_tagging_populates_regime_field():
    from src.backtester import Backtester
    import numpy as np
    bt = Backtester()
    n = 200
    hist = {
        "close": np.linspace(100, 110, n).tolist(),
        "open":  np.linspace(99, 109, n).tolist(),
        "high":  np.linspace(101, 111, n).tolist(),
        "low":   np.linspace(98, 108, n).tolist(),
        "volume": [1e6] * n,
    }
    result = bt.run(hist, tag_regimes=True)
    for detail in result.signal_details:
        assert "regime" in detail

def test_walk_forward_returns_report():
    from src.backtester import Backtester
    import numpy as np
    bt = Backtester()
    n = 500
    hist = {
        "close": (100 + np.cumsum(np.random.randn(n) * 0.5)).tolist(),
        "open":  (100 + np.cumsum(np.random.randn(n) * 0.5) - 0.1).tolist(),
        "high":  (100 + np.cumsum(np.random.randn(n) * 0.5) + 0.3).tolist(),
        "low":   (100 + np.cumsum(np.random.randn(n) * 0.5) - 0.3).tolist(),
        "volume": [1e6] * n,
    }
    report = bt.walk_forward_validate(hist, n_folds=3)
    assert report.n_folds == 3
    assert 0.0 <= report.avg_out_sample_winrate <= 100.0
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Per-pair insight | None (global metrics only) | Full per-pair win rate, drawdown, avg R:R |
| Regime performance analysis | Impossible | Fully supported (slice by regime tag) |
| Overfitting detection | None | Walk-forward overfit score reported |
| Parameter optimisation | Manual | Automated sweep via `run_per_pair_sweep()` |

---

## Dependencies

- **PR_01** — `detect_regime_from_arrays()` builds on the enhanced regime logic.
- **PR_02** — Per-pair config profiles supply the sweep parameter sets.
- **PR_07** — Dynamic SL/TP config is exercised in the sweep.
