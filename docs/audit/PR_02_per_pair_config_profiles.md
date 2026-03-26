# PR_02 — Per-Pair Config Profiles

**Branch:** `feature/pr02-per-pair-config`  
**Priority:** 2 (Required by PR_03, PR_06, PR_07)  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Introduce pair-specific configuration profiles that override global channel thresholds
(ATR multiplier, BB width tolerance, volume minimum, spread max, momentum threshold, RSI
thresholds) on a per-pair basis. Currently every pair from BTC to SHIB uses the same
parameters set in `config/__init__.py`, causing systematic false positives on high-volatility
altcoins and false negatives on liquid majors.

Three profile tiers are defined:

| Tier | Examples | Characteristics |
|------|---------|----------------|
| `MAJOR` | BTC, ETH | Tight spreads, deep liquidity, low ATR%, high volume |
| `MIDCAP` | LINK, SOL, MATIC, AVAX | Medium volatility, moderate liquidity |
| `ALTCOIN` | DOGE, SHIB, PEPE, LYN | High ATR%, wide spreads, thin liquidity |

Each tier specifies threshold multipliers that are applied on top of the global config values,
so the base channel logic is unchanged and new pairs automatically inherit the correct profile
when they are classified.

---

## Files to Change

| File | Change type |
|------|-------------|
| `config/__init__.py` | Add `PAIR_TIER_MAP` dict and `PairProfile` dataclass |
| `src/pair_manager.py` | Add `classify_pair_tier(symbol)` function |
| `src/channels/base.py` | Accept optional `pair_profile` in `build_channel_signal()` |
| `src/channels/scalp.py` | Pass profile to `build_channel_signal()` and use for momentum threshold |
| `src/channels/swing.py` | Use profile for EMA200 buffer and ADX max |
| `src/channels/spot.py` | Use profile for volume expansion multiplier and squeeze threshold |
| `tests/test_channels.py` | Add tests verifying profile overrides are applied |

---

## Implementation Steps

### Step 1 — Define `PairProfile` in `config/__init__.py`

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class PairProfile:
    tier: str                         # "MAJOR", "MIDCAP", "ALTCOIN"
    # Multipliers applied to global config values (1.0 = no change)
    atr_mult: float = 1.0             # Multiplier for ATR-based SL distance
    momentum_threshold_mult: float = 1.0   # Multiplier for momentum threshold
    spread_max_mult: float = 1.0      # Multiplier for max spread tolerance
    volume_min_mult: float = 1.0      # Multiplier for minimum volume
    rsi_ob_level: float = 70.0        # RSI overbought level
    rsi_os_level: float = 30.0        # RSI oversold level
    adx_min_mult: float = 1.0         # Multiplier for minimum ADX
    bb_touch_pct: float = 0.002       # BB-touch proximity (0.2% default)
    momentum_persist_candles: int = 2  # Required consecutive momentum candles
    kill_zone_hard_gate: bool = False  # Hard-reject signals outside kill zones

# Tier profiles
PAIR_PROFILES: dict[str, PairProfile] = {
    "MAJOR": PairProfile(
        tier="MAJOR",
        atr_mult=1.0,
        momentum_threshold_mult=0.8,   # BTC/ETH: lower threshold (tighter moves)
        spread_max_mult=0.5,           # Tighter spread requirement
        volume_min_mult=5.0,           # Higher absolute volume floor
        rsi_ob_level=75.0,
        rsi_os_level=25.0,
        adx_min_mult=0.9,
        bb_touch_pct=0.003,            # Slightly wider tolerance for majors
        momentum_persist_candles=2,
        kill_zone_hard_gate=False,
    ),
    "MIDCAP": PairProfile(
        tier="MIDCAP",
        atr_mult=1.1,
        momentum_threshold_mult=1.0,
        spread_max_mult=1.0,
        volume_min_mult=1.0,
        rsi_ob_level=70.0,
        rsi_os_level=30.0,
        adx_min_mult=1.0,
        bb_touch_pct=0.002,
        momentum_persist_candles=2,
        kill_zone_hard_gate=False,
    ),
    "ALTCOIN": PairProfile(
        tier="ALTCOIN",
        atr_mult=1.3,
        momentum_threshold_mult=2.0,   # High-vol pairs need larger momentum moves
        spread_max_mult=2.0,           # Wider spreads acceptable
        volume_min_mult=0.3,           # Lower volume floor (smaller markets)
        rsi_ob_level=65.0,
        rsi_os_level=35.0,
        adx_min_mult=1.1,
        bb_touch_pct=0.001,            # Tighter touch requirement
        momentum_persist_candles=3,    # Extra confirmation candles
        kill_zone_hard_gate=True,      # Hard-gate: only trade in kill zones
    ),
}

# Static symbol → tier mapping (auto-classified for unlisted pairs)
PAIR_TIER_MAP: dict[str, str] = {
    "BTCUSDT": "MAJOR",
    "ETHUSDT": "MAJOR",
    "BNBUSDT": "MIDCAP",
    "SOLUSDT": "MIDCAP",
    "LINKUSDT": "MIDCAP",
    "MATICUSDT": "MIDCAP",
    "AVAXUSDT": "MIDCAP",
    "DOTUSDT": "MIDCAP",
    "DOGEUSDT": "ALTCOIN",
    "SHIBUSDT": "ALTCOIN",
    "PEPEUSDT": "ALTCOIN",
}
```

### Step 2 — Add `classify_pair_tier()` to `src/pair_manager.py`

```python
from config import PAIR_TIER_MAP, PAIR_PROFILES, PairProfile

def classify_pair_tier(symbol: str, volume_24h_usd: float = 0.0) -> PairProfile:
    """Return the PairProfile for a given symbol.

    Falls back to volume-based heuristic for unlisted pairs:
    - volume >= $500M/day → MAJOR
    - volume >= $50M/day  → MIDCAP
    - otherwise           → ALTCOIN
    """
    tier = PAIR_TIER_MAP.get(symbol.upper())
    if tier is None:
        if volume_24h_usd >= 500_000_000:
            tier = "MAJOR"
        elif volume_24h_usd >= 50_000_000:
            tier = "MIDCAP"
        else:
            tier = "ALTCOIN"
    return PAIR_PROFILES[tier]
```

### Step 3 — Thread profile through channel evaluate

In `src/scanner.py`, look up the profile once per symbol scan and attach it to the evaluate call context:

```python
from src.pair_manager import classify_pair_tier

profile = classify_pair_tier(symbol, volume_24h_usd=volume_24h_usd)
# Pass profile into evaluate via smc_data dict (no signature change required)
smc_data["pair_profile"] = profile
```

### Step 4 — Consume profile in ScalpChannel

In `_evaluate_standard()`, replace the static `momentum_threshold`:
```python
profile = smc_data.get("pair_profile")
base_momentum = max(0.10, min(0.30, atr_pct * 0.5))
if profile is not None:
    base_momentum *= profile.momentum_threshold_mult
momentum_threshold = base_momentum

# Also use profile.momentum_persist_candles for persistence check
persist = profile.momentum_persist_candles if profile else 2
if mom_arr is not None and len(mom_arr) >= persist:
    if not all(abs(float(mom_arr[-i])) >= momentum_threshold for i in range(1, persist + 1)):
        return None
```

In `_apply_kill_zone_note()`, upgrade to hard gate for ALTCOIN tier:
```python
def _apply_kill_zone_note(self, sig: Signal, profile=None, now=None):
    if not self._is_kill_zone_active(now):
        if profile is not None and profile.kill_zone_hard_gate:
            return None  # Hard reject — ALTCOIN tier outside kill zone
        sig.execution_note = "Outside kill zone — reduced conviction"
    return sig
```

### Step 5 — Consume profile in SwingChannel and SpotChannel

SwingChannel: use `profile.rsi_ob_level` / `profile.rsi_os_level` for RSI gate, and
`profile.adx_min_mult` to scale `config.adx_min`.

SpotChannel: use `profile.atr_mult` in `_bb_squeeze_threshold()` and `_volume_expansion_mult()`.

### Step 6 — Tests

In `tests/test_channels.py`:
- Assert BTCUSDT returns `PairProfile(tier="MAJOR")`.
- Assert DOGEUSDT returns `PairProfile(tier="ALTCOIN", kill_zone_hard_gate=True)`.
- Mock a channel evaluate call with ALTCOIN profile outside kill zone → assert signal is None.
- Mock a channel evaluate call with MAJOR profile → assert momentum threshold is scaled by 0.8.

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| False positives on altcoins (DOGE, SHIB) | High | Reduced by ~30% (higher momentum threshold, kill zone gate) |
| False negatives on BTC | Moderate | Reduced by ~10% (lower momentum threshold) |
| Signal frequency on altcoins outside kill zones | Uncontrolled | Eliminated for ALTCOIN tier |
| Code maintainability | Hard-coded per-channel conditionals | Centralised profile registry |

---

## Dependencies

- **PR_01**: Regime context provides `atr_percentile` which could be used to auto-adjust profiles
  dynamically in a future iteration, but is not required for this PR.
