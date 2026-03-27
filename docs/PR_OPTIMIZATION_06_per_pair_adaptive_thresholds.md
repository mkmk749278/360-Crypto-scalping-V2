# PR-OPT-06 — Per-Pair Adaptive Thresholds (Pair Profile System)

**Priority:** P3  
**Estimated Signal Increase:** 20–40% for pairs with established positive track records  
**Dependencies:** PR-OPT-05 (suppression telemetry provides the data; without it, profiles are blind)

---

## Objective

Implement a per-pair adaptive threshold system driven by historical pair statistics. Pairs that consistently spend most of their time in QUIET regime, or have wide spreads but high win rates, should have their quality gates and regime penalties adjusted automatically based on observed behaviour rather than global static thresholds.

---

## Recommended Changes

### Change 1 — Create `src/pair_profile.py`

**New file:** `src/pair_profile.py`

```python
"""
Per-pair adaptive profile system.

Tracks historical statistics per trading pair and exposes adaptive threshold
adjustments that downstream modules (scanner, signal_quality) can apply.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

try:
    import redis as _redis_lib
except ImportError:
    _redis_lib = None  # type: ignore


@dataclass
class RegimeDistribution:
    """Approximate % of time a pair spends in each regime."""
    trending_up:   float = 0.0
    trending_down: float = 0.0
    ranging:       float = 0.0
    volatile:      float = 0.0
    quiet:         float = 0.0

    @property
    def is_quiet_dominant(self) -> bool:
        """True if the pair spends >60% of its time in QUIET regime."""
        return self.quiet >= 0.60


@dataclass
class ChannelStats:
    """Per-channel win-rate statistics for a pair."""
    channel: str
    signal_count: int = 0
    win_count: int = 0
    avg_confidence: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.signal_count == 0:
            return 0.0
        return self.win_count / self.signal_count


@dataclass
class PairProfile:
    """
    Rolling statistical profile for a single trading pair.

    All metrics are computed over a configurable rolling window (default 7 days).
    """
    symbol: str
    updated_at: float = field(default_factory=time.time)

    # Liquidity metrics
    avg_spread_pct: float = 0.0         # Rolling average spread
    avg_volume_24h: float = 0.0         # Rolling average 24h volume
    avg_atr_pct: float = 0.0            # Rolling average ATR as % of price

    # Regime distribution
    regime_dist: RegimeDistribution = field(default_factory=RegimeDistribution)

    # Signal quality metrics
    channel_stats: Dict[str, ChannelStats] = field(default_factory=dict)

    # Adaptive threshold overrides (set by PairProfileManager)
    spread_threshold_override: Optional[float] = None   # None = use global default
    volume_threshold_override: Optional[float] = None
    quiet_penalty_override: Optional[float] = None      # None = use global default
    oi_penalty_weight_override: Optional[float] = None  # 0.0–1.0 multiplier

    @property
    def overall_win_rate(self) -> float:
        """Win rate across all channels."""
        total_signals = sum(s.signal_count for s in self.channel_stats.values())
        total_wins    = sum(s.win_count    for s in self.channel_stats.values())
        if total_signals == 0:
            return 0.0
        return total_wins / total_signals

    def format_telegram_summary(self) -> str:
        """Format a human-readable profile summary for the /profile command."""
        lines = [f"📊 *Pair Profile: {self.symbol}*", ""]
        lines.append(f"Avg Spread: {self.avg_spread_pct * 100:.2f} bps")
        lines.append(f"Avg Volume 24h: ${self.avg_volume_24h:,.0f}")
        lines.append(f"Avg ATR: {self.avg_atr_pct:.2f}%")
        lines.append("")
        lines.append("*Regime Distribution:*")
        d = self.regime_dist
        lines.append(f"  TRENDING: {(d.trending_up + d.trending_down) * 100:.0f}%")
        lines.append(f"  RANGING:  {d.ranging * 100:.0f}%")
        lines.append(f"  QUIET:    {d.quiet * 100:.0f}%  {'⚠️ dominant' if d.is_quiet_dominant else ''}")
        lines.append(f"  VOLATILE: {d.volatile * 100:.0f}%")
        lines.append("")
        lines.append("*Channel Win Rates:*")
        for ch, stats in self.channel_stats.items():
            if stats.signal_count > 0:
                lines.append(
                    f"  {ch}: {stats.win_rate * 100:.0f}% "
                    f"({stats.signal_count} signals, avg conf {stats.avg_confidence:.0f})"
                )
        if self.spread_threshold_override is not None:
            lines.append(f"\n🔧 Spread override: {self.spread_threshold_override * 100:.2f} bps")
        if self.quiet_penalty_override is not None:
            lines.append(f"🔧 QUIET penalty override: {self.quiet_penalty_override:.2f}")
        return "\n".join(lines)


class PairProfileManager:
    """
    Manages per-pair profiles with Redis-backed persistence.

    Reads profiles from Redis on startup and writes updates after each scan
    cycle. If Redis is unavailable, operates in memory-only mode.

    Parameters
    ----------
    redis_client:
        Optional Redis client. If None, operates in memory-only mode.
    window_days:
        Rolling window for statistical metrics (default: 7 days).
    quiet_dominant_threshold:
        Fraction of time in QUIET to trigger QUIET-relaxed treatment (default: 0.60).
    wide_spread_win_rate_threshold:
        Win rate above which a wide-spread pair gets a relaxed spread gate (default: 0.55).
    """

    REDIS_KEY_PREFIX = "pair_profile:"
    REDIS_TTL_SECONDS = 8 * 86_400   # 8 days (slightly beyond rolling window)

    def __init__(
        self,
        redis_client=None,
        window_days: int = 7,
        quiet_dominant_threshold: float = 0.60,
        wide_spread_win_rate_threshold: float = 0.55,
    ) -> None:
        self._redis = redis_client
        self._window_days = window_days
        self._quiet_threshold = quiet_dominant_threshold
        self._spread_win_rate_threshold = wide_spread_win_rate_threshold
        self._profiles: Dict[str, PairProfile] = {}

    def get_profile(self, symbol: str) -> Optional[PairProfile]:
        """Retrieve a pair profile from memory or Redis."""
        if symbol in self._profiles:
            return self._profiles[symbol]
        if self._redis is not None:
            return self._load_from_redis(symbol)
        return None

    def update_profile(self, profile: PairProfile) -> None:
        """Update a profile in memory and persist to Redis."""
        self._recompute_overrides(profile)
        self._profiles[profile.symbol] = profile
        if self._redis is not None:
            self._save_to_redis(profile)

    def _recompute_overrides(self, profile: PairProfile) -> None:
        """
        Recalculate adaptive threshold overrides based on current statistics.

        Rules applied:
        1. Quiet-dominant pairs (>60% in QUIET) → relax QUIET penalty from 1.8 to 1.2
        2. Wide-spread pairs with high win rate (>55%) → relax spread threshold by 0.02
        3. Low OI-correlation pairs → reduce OI penalty weight to 0.5
        """
        # Rule 1: QUIET-dominant pair gets relaxed QUIET penalty
        if profile.regime_dist.is_quiet_dominant:
            profile.quiet_penalty_override = 1.2   # Less strict than global 1.8
        else:
            profile.quiet_penalty_override = None  # Use global default

        # Rule 2: Wide-spread pair with good win rate gets relaxed spread gate
        is_wide_spread = profile.avg_spread_pct > 0.03
        if is_wide_spread and profile.overall_win_rate >= self._spread_win_rate_threshold:
            # Relax by adding 0.02 (2 bps) to the channel default
            profile.spread_threshold_override = profile.avg_spread_pct * 1.5
        else:
            profile.spread_threshold_override = None

        # Rule 3: If pair has low OI correlation (no channel stats dominated by OI issues),
        # reduce OI penalty weight — placeholder for future OI success rate tracking
        profile.oi_penalty_weight_override = None  # Future: set to 0.5 if oi_success_rate > 0.6

    def _load_from_redis(self, symbol: str) -> Optional[PairProfile]:
        """Load and deserialize a profile from Redis."""
        try:
            import json
            key = f"{self.REDIS_KEY_PREFIX}{symbol}"
            data = self._redis.get(key)
            if data is None:
                return None
            raw = json.loads(data)
            profile = PairProfile(symbol=symbol)
            profile.avg_spread_pct = raw.get("avg_spread_pct", 0.0)
            profile.avg_volume_24h = raw.get("avg_volume_24h", 0.0)
            profile.avg_atr_pct    = raw.get("avg_atr_pct", 0.0)
            profile.updated_at     = raw.get("updated_at", time.time())
            dist = raw.get("regime_dist", {})
            profile.regime_dist = RegimeDistribution(**dist)
            for ch, stats in raw.get("channel_stats", {}).items():
                profile.channel_stats[ch] = ChannelStats(channel=ch, **stats)
            self._profiles[symbol] = profile
            return profile
        except Exception:
            return None

    def _save_to_redis(self, profile: PairProfile) -> None:
        """Serialize and save a profile to Redis."""
        try:
            import json
            key = f"{self.REDIS_KEY_PREFIX}{profile.symbol}"
            data = json.dumps({
                "avg_spread_pct": profile.avg_spread_pct,
                "avg_volume_24h": profile.avg_volume_24h,
                "avg_atr_pct":    profile.avg_atr_pct,
                "updated_at":     profile.updated_at,
                "regime_dist": {
                    "trending_up":   profile.regime_dist.trending_up,
                    "trending_down": profile.regime_dist.trending_down,
                    "ranging":       profile.regime_dist.ranging,
                    "volatile":      profile.regime_dist.volatile,
                    "quiet":         profile.regime_dist.quiet,
                },
                "channel_stats": {
                    ch: {
                        "signal_count":    s.signal_count,
                        "win_count":       s.win_count,
                        "avg_confidence":  s.avg_confidence,
                    }
                    for ch, s in profile.channel_stats.items()
                },
            })
            self._redis.setex(key, self.REDIS_TTL_SECONDS, data)
        except Exception:
            pass  # Redis failure must never interrupt signal generation
```

### Change 2 — Apply profile overrides in scanner

**File:** `src/scanner/__init__.py` / `src/scanner.py`

```python
# After computing regime and before quality gate
profile = pair_profile_manager.get_profile(symbol)

# Apply QUIET penalty override from profile
if regime == "QUIET" and chan_name.startswith("360_SCALP"):
    if profile and profile.quiet_penalty_override is not None:
        regime_mult = profile.quiet_penalty_override
    else:
        regime_mult = _SCALP_QUIET_REGIME_PENALTY  # Global default (1.8)

# Apply spread threshold override from profile
if profile and profile.spread_threshold_override is not None:
    effective_spread_limit = profile.spread_threshold_override
else:
    effective_spread_limit = _CHANNEL_MAX_SPREAD_PCT.get(chan_name, 0.03)
```

### Change 3 — Add `/profile` Telegram command

**File:** `src/telegram_bot.py`

```python
async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /profile <SYMBOL> command."""
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /profile BTCUSDT")
        return
    symbol = args[0].upper()
    profile = pair_profile_manager.get_profile(symbol)
    if profile is None:
        await update.message.reply_text(
            f"No profile found for {symbol}. "
            "Profiles are built after 24h of scan data."
        )
        return
    await update.message.reply_text(profile.format_telegram_summary(), parse_mode="Markdown")

application.add_handler(CommandHandler("profile", cmd_profile))
```

---

## Modules Affected

| Module | Change |
|--------|--------|
| `src/pair_profile.py` | **New file** — `PairProfile`, `PairProfileManager` |
| `src/scanner/__init__.py` | Apply profile overrides for spread and QUIET penalty |
| `src/scanner.py` | Same as above |
| `src/signal_quality.py` | Accept optional profile in `assess_pair_quality` |
| `src/telegram_bot.py` | Add `/profile` command |
| `config/__init__.py` | `PAIR_PROFILE_WINDOW_DAYS`, `QUIET_DOMINANT_THRESHOLD` env vars |

---

## Test Cases

1. **`test_quiet_dominant_profile`** — Profile with 65% QUIET time gets `quiet_penalty_override = 1.2`.
2. **`test_wide_spread_high_winrate`** — Profile with avg spread 4 bps and 60% win rate gets spread override.
3. **`test_profile_redis_roundtrip`** — Profile saved to Redis and reloaded matches original.
4. **`test_profile_redis_unavailable`** — `PairProfileManager` with `redis_client=None` operates in memory.
5. **`test_profile_applied_in_scanner`** — Scanner uses profile's `quiet_penalty_override` instead of global.
6. **`test_telegram_profile_command`** — `/profile BTCUSDT` returns formatted markdown string.
7. **`test_telegram_profile_not_found`** — `/profile UNKNOWN` returns "No profile found" message.

---

## Rollback Procedure

1. Remove `src/pair_profile.py`.
2. Remove profile override code from scanner (additive path, no existing logic changed).
3. Remove `/profile` command handler.
4. Redis keys with prefix `pair_profile:` can be left or flushed: `redis-cli --scan --pattern "pair_profile:*" | xargs redis-cli del`

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Incorrect profile data causes wrong threshold for a pair | Medium | Overrides are bounded (spread limit capped at 1.5× avg; penalty floored at 1.0) |
| Redis TTL expiry causes profile loss | Low | Profile rebuilt from scan data within 24h |
| Pairs with small sample sizes get bad win-rate estimates | Medium | Min 20 signals required before override is applied |
| `/profile` command exposes internal trading metrics | Low | Existing Telegram auth middleware restricts command access |

---

## Expected Impact

- QUIET-dominant pairs (ADA, ZEC, STG) automatically get relaxed QUIET penalties based on actual behaviour data
- Wide-spread pairs with historically high win rates pass quality gates they previously failed
- `/profile SYMBOL` gives operators instant visibility into per-pair system behaviour
