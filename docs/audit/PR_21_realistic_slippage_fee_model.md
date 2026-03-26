# PR_21 — Realistic Slippage & Fee Model

**PR Number:** PR_21  
**Branch:** `feature/pr21-realistic-slippage-fee-model`  
**Category:** Backtesting Maturity (Phase 2C)  
**Priority:** P0 (required by PR_22 and PR_23)  
**Dependency:** PR_11 (Phase 1 — Backtester Per-Pair + Regime, merged as #137)  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Replace the uniform fixed-percentage slippage model in `backtester.py` with a volume-dependent market impact model, add maker/taker fee differentiation, and support configurable execution latency injection. This closes the largest single gap between backtested and live-trading results.

---

## Current State

`src/backtester.py` applies `slippage_pct` as a single fixed uniform percentage to every trade regardless of:
- Position size relative to market volume.
- Pair liquidity (BTC vs. low-cap altcoin).
- Whether the order is a limit (maker) or market (taker).
- Execution latency.

This causes the backtester to underestimate costs on large altcoin positions and overestimate costs on small BTC positions.

---

## Proposed Changes

### New file: `src/slippage_model.py`

```python
"""Volume-dependent market impact slippage and fee model for backtesting."""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Pair tier configurations
_TIER_PARAMS = {
    "MAJOR": {
        "base_slippage": 0.0001,    # 0.01%
        "impact_coef":   0.5,
        "maker_fee":     0.0002,    # 0.02%
        "taker_fee":     0.0004,    # 0.04%
    },
    "MIDCAP": {
        "base_slippage": 0.0003,
        "impact_coef":   1.0,
        "maker_fee":     0.0002,
        "taker_fee":     0.0004,
    },
    "ALTCOIN": {
        "base_slippage": 0.0005,    # 0.05%
        "impact_coef":   2.0,
        "maker_fee":     0.0002,
        "taker_fee":     0.0004,
    },
}

@dataclass
class SlippageConfig:
    pair_tier: str = "MAJOR"           # "MAJOR", "MIDCAP", "ALTCOIN"
    order_type: str = "TAKER"          # "MAKER" or "TAKER"
    latency_ms_min: int = 50           # minimum execution latency (ms)
    latency_ms_max: int = 200          # maximum execution latency (ms)
    inject_latency: bool = False       # whether to randomise latency in simulation

class SlippageModel:
    """
    Computes realistic total execution cost = slippage + fee + latency penalty.

    Slippage formula:
        slippage_pct = base_slippage + (position_notional / avg_volume_1m) × impact_coef

    Fee:
        maker_fee or taker_fee based on order_type.

    Latency penalty:
        For each 100ms of latency in a trending market, add 0.005% adverse fill.
    """

    def __init__(self, config: Optional[SlippageConfig] = None):
        self._cfg = config or SlippageConfig()
        self._params = _TIER_PARAMS.get(self._cfg.pair_tier, _TIER_PARAMS["MAJOR"])

    def total_cost_pct(
        self,
        position_notional: float,
        avg_1m_volume_usd: float,
        is_entry: bool = True,
    ) -> float:
        """
        Return total round-trip cost as a percentage of position notional.

        Args:
            position_notional: USD value of the position.
            avg_1m_volume_usd: Average 1-minute traded volume in USD for the pair.
            is_entry: True for entry fill, False for exit. Used for latency direction.

        Returns:
            Total cost as a fraction (e.g., 0.0012 = 0.12%).
        """
        slippage = self._market_impact(position_notional, avg_1m_volume_usd)
        fee = self._fee()
        latency_penalty = self._latency_penalty() if self._cfg.inject_latency else 0.0
        total = slippage + fee + latency_penalty
        logger.debug(
            "SlippageModel: slippage=%.4f%% fee=%.4f%% latency=%.4f%% total=%.4f%%",
            slippage * 100, fee * 100, latency_penalty * 100, total * 100,
        )
        return total

    def _market_impact(self, position_notional: float, avg_volume: float) -> float:
        if avg_volume <= 0:
            return self._params["base_slippage"]
        impact = (position_notional / avg_volume) * self._params["impact_coef"]
        return self._params["base_slippage"] + impact

    def _fee(self) -> float:
        key = "maker_fee" if self._cfg.order_type == "MAKER" else "taker_fee"
        return self._params[key]

    def _latency_penalty(self) -> float:
        """Randomised latency injection adds adverse fill proportional to latency."""
        latency_ms = random.uniform(self._cfg.latency_ms_min, self._cfg.latency_ms_max)
        return (latency_ms / 100) * 0.00005   # 0.005% per 100ms
```

### Integrate into `src/backtester.py`

```python
from src.slippage_model import SlippageModel, SlippageConfig

class Backtester:
    def __init__(self, config, slippage_config: Optional[SlippageConfig] = None):
        self._slippage_model = SlippageModel(slippage_config)
        # ... existing init ...

    def _apply_costs(self, price: float, position_notional: float, avg_volume: float,
                     is_entry: bool) -> float:
        """Return adjusted fill price after applying slippage and fees."""
        cost_pct = self._slippage_model.total_cost_pct(
            position_notional=position_notional,
            avg_1m_volume_usd=avg_volume,
            is_entry=is_entry,
        )
        direction = 1 if is_entry else -1   # entry costs raise price; exit costs lower it
        return price * (1 + direction * cost_pct)
```

### Config additions in `src/config/__init__.py`

```python
BACKTEST_SLIPPAGE_TIER:      str  = os.getenv("BACKTEST_SLIPPAGE_TIER", "MAJOR")
BACKTEST_ORDER_TYPE:         str  = os.getenv("BACKTEST_ORDER_TYPE",    "TAKER")
BACKTEST_INJECT_LATENCY:     bool = os.getenv("BACKTEST_INJECT_LATENCY", "false").lower() == "true"
BACKTEST_LATENCY_MS_MIN:     int  = int(os.getenv("BACKTEST_LATENCY_MS_MIN", "50"))
BACKTEST_LATENCY_MS_MAX:     int  = int(os.getenv("BACKTEST_LATENCY_MS_MAX", "500"))
```

---

## Implementation Steps

1. Create `src/slippage_model.py` with `SlippageModel` and `SlippageConfig`.
2. In `backtester.py`, replace all `price * (1 ± slippage_pct)` calls with `self._apply_costs()`.
3. Pass `avg_1m_volume_usd` from the signal or from a per-pair volume lookup table in config.
4. Add `SlippageConfig` instantiation in `Backtester.__init__` from config constants.
5. Write unit tests in `tests/test_slippage_model.py`.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/slippage_model.py` | New — `SlippageModel` and `SlippageConfig` |
| `src/backtester.py` | Replace fixed slippage with `SlippageModel.total_cost_pct()` |
| `src/config/__init__.py` | Add slippage model config constants |
| `tests/test_slippage_model.py` | New test file |

---

## Testing Requirements

```python
# tests/test_slippage_model.py
def test_major_pair_base_slippage():
    sm = SlippageModel(SlippageConfig(pair_tier="MAJOR", order_type="TAKER"))
    # Small position relative to volume → approximately base slippage + fee
    cost = sm.total_cost_pct(1_000, 1_000_000)
    assert abs(cost - (0.0001 + 0.0004)) < 1e-5

def test_altcoin_large_position_higher_impact():
    sm_major  = SlippageModel(SlippageConfig(pair_tier="MAJOR"))
    sm_alt    = SlippageModel(SlippageConfig(pair_tier="ALTCOIN"))
    cost_major = sm_major.total_cost_pct(100_000, 50_000)
    cost_alt   = sm_alt.total_cost_pct(100_000, 50_000)
    assert cost_alt > cost_major

def test_maker_cheaper_than_taker():
    sm_maker = SlippageModel(SlippageConfig(order_type="MAKER"))
    sm_taker = SlippageModel(SlippageConfig(order_type="TAKER"))
    assert sm_maker.total_cost_pct(1_000, 1_000_000) < sm_taker.total_cost_pct(1_000, 1_000_000)

def test_zero_volume_uses_base_slippage():
    sm = SlippageModel(SlippageConfig(pair_tier="ALTCOIN"))
    cost = sm.total_cost_pct(10_000, 0)
    assert cost >= 0.0005  # at least base altcoin slippage

def test_latency_injection_adds_penalty():
    sm_no_lat  = SlippageModel(SlippageConfig(inject_latency=False))
    sm_with_lat = SlippageModel(SlippageConfig(inject_latency=True,
                                               latency_ms_min=200, latency_ms_max=201))
    assert sm_with_lat.total_cost_pct(1_000, 1_000_000) > sm_no_lat.total_cost_pct(1_000, 1_000_000)
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Backtest-to-live cost gap | ~15–20% underestimation on altcoins | <5% gap |
| Slippage model accuracy | Flat uniform | Volume-dependent market impact |
| Fee accuracy | Single flat rate | Maker/taker differentiation |
| Altcoin backtest net PnL accuracy | Overstated | Realistic (more conservative) |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| avg_1m_volume_usd unavailable for all pairs | Use fallback lookup table; log warning |
| Latency injection randomness causes non-reproducible backtests | Set random seed before backtest run; document |
| Over-conservative cost model reduces all backtest results | Compare with real trade execution costs; calibrate impact_coef |
