# PR_15 — Correlation-Aware Exposure Filter

**PR Number:** PR_15  
**Branch:** `feature/pr15-correlation-aware-exposure-filter`  
**Category:** Risk & Reliability (Phase 2A)  
**Priority:** P1  
**Dependency:** PR_13 (Portfolio-Level Drawdown Circuit Breaker)  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Prevent correlated portfolio drawdowns by computing a rolling 24-hour cross-pair Pearson correlation matrix and capping total BTC-beta exposure. When a new signal would push the portfolio above the BTC-beta threshold or a sector cap, it is rejected before dispatch.

---

## Current State

`src/correlation.py` (≈3KB) exists but is minimal:
- Computes point-in-time price correlation between two pairs.
- Not integrated into the signal dispatch pipeline.
- No rolling window computation.
- No concept of sector exposure.

No portfolio-level correlation checks exist anywhere in the codebase.

---

## Proposed Changes

### Extend `src/correlation.py`

```python
"""Rolling cross-pair correlation and BTC-beta exposure management."""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

_WINDOW_HOURS = 24
_CANDLE_INTERVAL_MINUTES = 5
_WINDOW_CANDLES = int(_WINDOW_HOURS * 60 / _CANDLE_INTERVAL_MINUTES)  # 288

class CorrelationMatrix:
    """Maintains rolling 24h Pearson correlation across all active pairs."""

    def __init__(self, window: int = _WINDOW_CANDLES):
        self._window = window
        self._returns: Dict[str, deque] = {}  # symbol → deque of log-returns

    def push(self, symbol: str, close_price: float) -> None:
        """Add the latest close price for *symbol*."""
        if symbol not in self._returns:
            self._returns[symbol] = deque(maxlen=self._window + 1)
        self._returns[symbol].append(close_price)

    def correlation(self, sym_a: str, sym_b: str) -> Optional[float]:
        """Return Pearson correlation between *sym_a* and *sym_b* or None if insufficient data."""
        ra = self._log_returns(sym_a)
        rb = self._log_returns(sym_b)
        if ra is None or rb is None:
            return None
        n = min(len(ra), len(rb))
        if n < 20:
            return None
        return float(np.corrcoef(ra[-n:], rb[-n:])[0, 1])

    def btc_beta(self, symbol: str) -> Optional[float]:
        """Return the rolling BTC-beta for *symbol*."""
        return self.correlation(symbol, "BTCUSDT")

    def _log_returns(self, symbol: str) -> Optional[np.ndarray]:
        prices = list(self._returns.get(symbol, []))
        if len(prices) < 2:
            return None
        arr = np.array(prices, dtype=float)
        return np.diff(np.log(arr))
```

### Add `check_portfolio_correlation()` to `src/risk.py`

```python
from src.correlation import CorrelationMatrix
from src.sector import get_sector  # new module (see below)

BTC_BETA_THRESHOLD = 0.85
SECTOR_MAX_EXPOSURE_PCT = 0.40  # 40% of portfolio per sector

def check_portfolio_correlation(
    symbol: str,
    correlation_matrix: CorrelationMatrix,
    open_positions: Dict[str, float],    # symbol → notional USD value
    portfolio_value: float,
) -> tuple[bool, str]:
    """
    Return (True, "") if the symbol passes correlation checks.
    Return (False, reason) if adding this position would breach a threshold.
    """
    # BTC-beta guard
    beta = correlation_matrix.btc_beta(symbol)
    if beta is not None and beta > BTC_BETA_THRESHOLD:
        existing_btc_beta_notional = sum(
            v for s, v in open_positions.items()
            if (correlation_matrix.btc_beta(s) or 0) > BTC_BETA_THRESHOLD
        )
        new_position_estimate = portfolio_value * 0.01   # assume 1% risk per trade as estimate
        if (existing_btc_beta_notional + new_position_estimate) / portfolio_value > BTC_BETA_THRESHOLD:
            return False, f"BTC-beta cap breached (beta={beta:.2f})"

    # Sector cap guard
    sector = get_sector(symbol)
    sector_notional = sum(
        v for s, v in open_positions.items() if get_sector(s) == sector
    )
    if portfolio_value > 0 and (sector_notional / portfolio_value) > SECTOR_MAX_EXPOSURE_PCT:
        return False, f"Sector cap breached ({sector}: {sector_notional/portfolio_value:.1%})"

    return True, ""
```

### New file: `src/sector.py`

```python
"""Simple sector classification for portfolio exposure caps."""
_SECTOR_MAP = {
    # Layer-1
    "BTCUSDT": "L1", "ETHUSDT": "L1", "SOLUSDT": "L1", "AVAXUSDT": "L1",
    # DeFi
    "UNIUSDT": "DEFI", "AAVEUSDT": "DEFI", "LINKUSDT": "DEFI",
    # Meme
    "DOGEUSDT": "MEME", "SHIBUSDT": "MEME", "PEPEUSDT": "MEME",
    # AI / Infra
    "FETUSDT": "AI", "WLDUSDT": "AI",
}
_DEFAULT_SECTOR = "OTHER"

def get_sector(symbol: str) -> str:
    return _SECTOR_MAP.get(symbol.upper(), _DEFAULT_SECTOR)
```

### Wire into `src/signal_router.py`

```python
# Before dispatching a signal:
passed, reason = risk_manager.check_portfolio_correlation(
    signal.symbol, correlation_matrix, open_positions, portfolio_value
)
if not passed:
    logger.info("Signal for %s rejected by correlation filter: %s", signal.symbol, reason)
    return None
```

---

## Implementation Steps

1. Extend `src/correlation.py` with `CorrelationMatrix` class.
2. Create `src/sector.py` with sector classification map.
3. Add `check_portfolio_correlation()` to `src/risk.py`.
4. In `signal_router.py`, instantiate `CorrelationMatrix` (module-level singleton) and call check before dispatch.
5. Feed `CorrelationMatrix.push()` at the end of each scan cycle with the latest close price per pair.
6. Write unit tests in `tests/test_correlation_filter.py`.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/correlation.py` | Add `CorrelationMatrix` class with rolling Pearson logic |
| `src/risk.py` | Add `check_portfolio_correlation()` function |
| `src/signal_router.py` | Wire in correlation pre-dispatch check |
| `src/sector.py` | New — sector classification map |
| `tests/test_correlation_filter.py` | New test file |

---

## Testing Requirements

```python
# tests/test_correlation_filter.py
def test_perfect_positive_correlation():
    cm = CorrelationMatrix()
    prices = [100 + i for i in range(50)]
    for p in prices:
        cm.push("BTCUSDT", p)
        cm.push("ETHUSDT", p * 0.05)
    corr = cm.correlation("BTCUSDT", "ETHUSDT")
    assert corr is not None and corr > 0.99

def test_insufficient_data_returns_none():
    cm = CorrelationMatrix()
    cm.push("BTCUSDT", 100)
    assert cm.btc_beta("ETHUSDT") is None

def test_sector_cap_rejects_excess():
    # Build portfolio at 45% L1 exposure
    passed, reason = check_portfolio_correlation(
        "SOLUSDT", cm, {"BTCUSDT": 4500}, 10000
    )
    assert not passed
    assert "Sector cap" in reason

def test_btc_beta_passes_low_correlation():
    # PEPEUSDT has near-zero BTC beta in test data
    passed, reason = check_portfolio_correlation(
        "PEPEUSDT", cm, {}, 10000
    )
    assert passed
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Portfolio BTC-beta in trending market | Uncapped (can reach 0.95+) | Capped at 0.85 |
| Correlated drawdown risk | High | Reduced via sector caps |
| Signal rejection rate | 0% correlation-based | Est. 5–10% during high-correlation regimes |
| Sector diversification | No enforcement | Max 40% per sector |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Correlation matrix cold start (first 20 candles) | Return `None` from `btc_beta()`; skip correlation check, log warning |
| Sector map incomplete | Unknown symbols fall back to "OTHER" sector; OTHER cap applies |
| Performance: O(N²) matrix update | Only compute on-demand (lazy); cache for 60s |
| Over-rejection during crash where all assets correlate | Raise threshold to 0.95 during VOLATILE regime |
