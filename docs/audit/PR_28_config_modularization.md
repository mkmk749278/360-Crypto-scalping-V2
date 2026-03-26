# PR_28 — Config Modularization

**PR Number:** PR_28  
**Branch:** `feature/pr28-config-modularization`  
**Category:** Codebase Health (Phase 2E)  
**Priority:** P1  
**Dependency:** PR_27 (Scanner Decomposition Part 2)  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Split the 36KB monolithic `config/__init__.py` into five domain-specific modules. The existing `config/__init__.py` re-exports all symbols for full backward compatibility, so no call site outside `config/` needs to change. This makes the configuration domain discoverable, reviewable, and independently testable.

---

## Current State

`config/__init__.py` is a single ≈36KB file containing:
- Pair tier definitions (MAJOR / MIDCAP / ALTCOIN lists with per-pair config dicts).
- Market regime parameter tables (ADX, ATR, RSI thresholds per regime).
- Channel-specific settings (min_confidence, sl_pct, tp_ratios, etc. per channel).
- Risk thresholds (max_position_size, drawdown levels, Kelly parameters).
- Base settings (API keys, Telegram tokens, env-var loading, logging config).

This violates the single-responsibility principle and makes code review and auditing slow.

---

## Proposed Changes

### Target file structure

```
config/
  __init__.py       # Re-exports everything; backward-compat entry point (~30 lines)
  base.py           # API keys, env vars, logging, global constants
  pairs.py          # MAJOR/MIDCAP/ALTCOIN tier definitions and per-pair configs
  regime.py         # Regime parameter tables (thresholds per regime × pair tier)
  channels.py       # Channel-specific configs (scalp, swing, spot, gem)
  risk.py           # Risk thresholds, drawdown levels, position sizing params
```

### `config/base.py`

```python
"""Base configuration — environment variables, API keys, logging setup."""
import logging
import os

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
BINANCE_API_KEY:    str = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN:     str = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID:   str = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ADMIN_CHAT_ID: str = os.getenv("TELEGRAM_ADMIN_CHAT_ID", TELEGRAM_CHAT_ID)
REDIS_URL:          str = os.getenv("REDIS_URL", "")
ENVIRONMENT:        str = os.getenv("ENVIRONMENT", "production")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
```

### `config/pairs.py`

```python
"""Pair tier definitions and per-pair parameter overrides."""
from typing import Dict, List

MAJOR_PAIRS: List[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"]
MIDCAP_PAIRS: List[str] = ["LINKUSDT", "UNIUSDT", "AAVEUSDT", "MATICUSDT", "DOTUSDT"]
ALTCOIN_PAIRS: List[str] = ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FETUSDT", "WLDUSDT"]

ALL_PAIRS: List[str] = MAJOR_PAIRS + MIDCAP_PAIRS + ALTCOIN_PAIRS

def get_pair_tier(symbol: str) -> str:
    if symbol in MAJOR_PAIRS:
        return "MAJOR"
    if symbol in MIDCAP_PAIRS:
        return "MIDCAP"
    return "ALTCOIN"

# Per-pair overrides (merged on top of channel defaults at runtime)
PAIR_OVERRIDES: Dict[str, dict] = {
    "BTCUSDT": {"momentum_threshold": 0.12, "spread_max": 0.01, "adx_min": 18},
    "ETHUSDT": {"momentum_threshold": 0.15, "spread_max": 0.02},
    "DOGEUSDT": {"momentum_threshold": 0.30, "min_candle_persistence": 4},
}
```

### `config/regime.py`

```python
"""Market regime parameter tables."""
from typing import Dict

REGIME_ADX_MIN: Dict[str, float] = {
    "TRENDING_UP":   22.0,
    "TRENDING_DOWN": 22.0,
    "RANGING":       15.0,
    "VOLATILE":      20.0,
    "QUIET":         12.0,
}

REGIME_RSI_BOUNDS: Dict[str, tuple] = {
    "TRENDING_UP":   (40, 70),
    "TRENDING_DOWN": (30, 60),
    "RANGING":       (35, 65),
    "VOLATILE":      (30, 70),
    "QUIET":         (35, 65),
}

REGIME_ATR_MULTIPLIER: Dict[str, float] = {
    "TRENDING_UP":   1.0,
    "TRENDING_DOWN": 1.0,
    "RANGING":       0.8,
    "VOLATILE":      1.3,
    "QUIET":         0.7,
}
```

### `config/channels.py`

```python
"""Channel-specific configurations."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class ChannelConfig:
    name:            str
    min_confidence:  float
    sl_pct_range:    Tuple[float, float]
    tp_ratios:       List[float]
    trailing_atr_mult: float = 1.0
    adx_min:         float = 20.0
    risk_pct:        float = 0.01

SCALP_CONFIG = ChannelConfig(
    name="360_SCALP",
    min_confidence=60.0,
    sl_pct_range=(0.05, 0.10),
    tp_ratios=[0.5, 1.0, 1.5],
    trailing_atr_mult=0.5,
    adx_min=20.0,
    risk_pct=0.005,
)

SWING_CONFIG = ChannelConfig(
    name="360_SWING",
    min_confidence=65.0,
    sl_pct_range=(0.20, 0.50),
    tp_ratios=[1.5, 3.0, 5.0],
    trailing_atr_mult=1.0,
    adx_min=22.0,
    risk_pct=0.01,
)

SPOT_CONFIG = ChannelConfig(
    name="360_SPOT",
    min_confidence=70.0,
    sl_pct_range=(0.50, 2.00),
    tp_ratios=[2.0, 5.0, 10.0],
    trailing_atr_mult=1.5,
    adx_min=18.0,
    risk_pct=0.015,
)
```

### `config/risk.py`

```python
"""Risk management thresholds and position sizing parameters."""
import os

MAX_POSITION_SIZE_USD:    float = float(os.getenv("MAX_POSITION_SIZE_USD",  "1000"))
MAX_OPEN_POSITIONS:       int   = int(os.getenv("MAX_OPEN_POSITIONS",       "10"))
PORTFOLIO_CB_YELLOW_PCT:  float = float(os.getenv("PORTFOLIO_CB_YELLOW_PCT", "-0.03"))
PORTFOLIO_CB_RED_PCT:     float = float(os.getenv("PORTFOLIO_CB_RED_PCT",    "-0.05"))
PORTFOLIO_CB_BLACK_PCT:   float = float(os.getenv("PORTFOLIO_CB_BLACK_PCT",  "-0.08"))
BTC_BETA_THRESHOLD:       float = float(os.getenv("BTC_BETA_THRESHOLD",      "0.85"))
SECTOR_MAX_EXPOSURE_PCT:  float = float(os.getenv("SECTOR_MAX_EXPOSURE_PCT", "0.40"))
KELLY_FRACTION:           float = float(os.getenv("KELLY_FRACTION",          "0.5"))
```

### Updated `config/__init__.py` (≈30 lines)

```python
"""
Config package entry point.
Re-exports all symbols from domain modules for backward compatibility.
Any code importing from `config` continues to work unchanged.
"""
from config.base    import *   # noqa: F401, F403
from config.pairs   import *   # noqa: F401, F403
from config.regime  import *   # noqa: F401, F403
from config.channels import *  # noqa: F401, F403
from config.risk    import *   # noqa: F401, F403
```

---

## Implementation Steps

1. Create `config/base.py` and move environment/logging setup.
2. Create `config/pairs.py` and move pair tier definitions and overrides.
3. Create `config/regime.py` and move regime parameter tables.
4. Create `config/channels.py` and move channel config dataclasses.
5. Create `config/risk.py` and move risk thresholds.
6. Update `config/__init__.py` to re-export from all five modules.
7. Run full test suite to verify no import breakage.
8. Verify no single file in `config/` exceeds 12KB (except `__init__.py` which should be <2KB).

---

## Files Modified / Created

| File | Change |
|------|--------|
| `config/__init__.py` | Reduced to re-export wrapper (~30 lines) |
| `config/base.py` | New — API keys, env vars, logging |
| `config/pairs.py` | New — pair tier definitions |
| `config/regime.py` | New — regime parameter tables |
| `config/channels.py` | New — channel config dataclasses |
| `config/risk.py` | New — risk thresholds |

---

## Testing Requirements

```python
# Verify backward compatibility — all existing imports still work
def test_backward_compat_major_pairs():
    from config import MAJOR_PAIRS
    assert "BTCUSDT" in MAJOR_PAIRS

def test_backward_compat_regime_adx():
    from config import REGIME_ADX_MIN
    assert "TRENDING_UP" in REGIME_ADX_MIN

def test_backward_compat_risk():
    from config import PORTFOLIO_CB_YELLOW_PCT
    assert PORTFOLIO_CB_YELLOW_PCT < 0

def test_channel_config_dataclass():
    from config.channels import SCALP_CONFIG
    assert SCALP_CONFIG.min_confidence == 60.0
    assert len(SCALP_CONFIG.tp_ratios) == 3

def test_pair_tier_lookup():
    from config.pairs import get_pair_tier
    assert get_pair_tier("BTCUSDT") == "MAJOR"
    assert get_pair_tier("PEPEUSDT") == "ALTCOIN"
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| `config/__init__.py` size | ~36KB | ~2KB (re-export only) |
| Time to find pair config | Search 36KB file | Open `config/pairs.py` directly |
| Config PR review scope | Full 36KB diff | Only changed domain file |
| Risk threshold discoverability | Buried in monolith | Explicit in `config/risk.py` |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Import order dependencies (e.g., `base.py` must load before `channels.py`) | Use explicit imports in `__init__.py` in dependency order |
| Wildcard import hides which module a symbol comes from | Document module sources in `__init__.py` comments |
| Circular imports if domain modules reference each other | Enforce: only `base.py` can be imported by other domain modules |
| Missing symbol during split → runtime AttributeError | Run full test suite after each module extraction step |
