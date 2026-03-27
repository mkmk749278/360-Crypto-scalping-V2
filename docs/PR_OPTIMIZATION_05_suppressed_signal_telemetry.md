# PR-OPT-05 — Suppressed Signal Telemetry

**Priority:** P2 (deploy early to establish baseline before tuning other PRs)  
**Estimated Impact:** Full visibility into signal pipeline losses; enables data-driven threshold tuning  
**Dependencies:** Can be deployed standalone; enhanced by PR-OPT-01, PR-OPT-02, PR-OPT-03

---

## Objective

Add comprehensive monitoring and telemetry for suppressed signals so that every signal that is blocked, penalised, or discarded is observable. Currently the system logs suppression events at DEBUG level, making them invisible in production. This PR makes suppressions first-class observable events with counters, rolling summaries, and a Telegram command.

---

## Recommended Changes

### Change 1 — Create `src/suppression_telemetry.py`

**New file:** `src/suppression_telemetry.py`

```python
"""
Suppression telemetry module.

Tracks all signal suppression events across the pipeline and provides:
- Per-reason counters (Prometheus-compatible)
- Rolling window of raw events for Telegram summary
- Periodic Telegram digest of suppression statistics
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


@dataclass
class SuppressionEvent:
    """A single signal suppression event."""
    timestamp: float           # Unix timestamp
    symbol: str
    channel: str
    reason: str                # e.g. "quiet_regime", "spread_gate", "oi_invalidation"
    regime: Optional[str]      # Market regime at time of suppression
    would_be_confidence: float # Confidence score the signal would have had
    extra: Dict[str, object] = field(default_factory=dict)


class SuppressionTracker:
    """
    Rolling window suppression tracker.

    Collects SuppressionEvent instances and exposes aggregated counts
    for monitoring and periodic Telegram digests.

    Parameters
    ----------
    window_seconds:
        Rolling window size for counts and event retention (default: 4h).
    max_events:
        Maximum raw events retained in memory.
    """

    REASON_QUIET_REGIME    = "quiet_regime"
    REASON_SPREAD_GATE     = "spread_gate"
    REASON_VOLUME_GATE     = "volume_gate"
    REASON_OI_INVALIDATION = "oi_invalidation"
    REASON_CLUSTER         = "cluster_suppression"
    REASON_STAT_FILTER     = "stat_filter"
    REASON_LIFESPAN        = "min_lifespan"
    REASON_CONFIDENCE      = "confidence_threshold"

    def __init__(
        self,
        window_seconds: int = 14_400,   # 4 hours
        max_events: int = 5_000,
    ) -> None:
        self._window = window_seconds
        self._events: Deque[SuppressionEvent] = deque(maxlen=max_events)
        self._counters: Dict[str, int] = defaultdict(int)

    def record(self, event: SuppressionEvent) -> None:
        """Record a suppression event and increment the relevant counter."""
        self._events.append(event)
        self._counters[event.reason] += 1

    def _prune(self) -> None:
        """Remove events outside the rolling window."""
        cutoff = time.time() - self._window
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

    def summary(self) -> Dict[str, int]:
        """Return per-reason counts within the rolling window."""
        self._prune()
        counts: Dict[str, int] = defaultdict(int)
        for event in self._events:
            counts[event.reason] += 1
        return dict(counts)

    def total_in_window(self) -> int:
        self._prune()
        return len(self._events)

    def format_telegram_digest(self) -> str:
        """Format a human-readable Telegram digest of recent suppressions."""
        self._prune()
        counts = self.summary()
        total = sum(counts.values())
        window_hours = self._window // 3600

        if total == 0:
            return f"✅ No signals suppressed in the last {window_hours}h"

        lines = [f"🔕 *Signal Suppressions — Last {window_hours}h*", f"Total: *{total}*", ""]
        reason_labels = {
            self.REASON_QUIET_REGIME:    "📉 QUIET regime",
            self.REASON_SPREAD_GATE:     "📊 Spread gate",
            self.REASON_VOLUME_GATE:     "💧 Volume gate",
            self.REASON_OI_INVALIDATION: "📈 OI invalidation",
            self.REASON_CLUSTER:         "🔗 Cluster suppression",
            self.REASON_STAT_FILTER:     "🔬 Stat filter",
            self.REASON_LIFESPAN:        "⏱ Min lifespan",
            self.REASON_CONFIDENCE:      "🎯 Confidence threshold",
        }
        for reason, label in reason_labels.items():
            count = counts.get(reason, 0)
            if count > 0:
                pct = count / total * 100
                lines.append(f"{label}: {count} ({pct:.0f}%)")

        # Other reasons not in the label map
        for reason, count in counts.items():
            if reason not in reason_labels and count > 0:
                lines.append(f"❓ {reason}: {count}")

        return "\n".join(lines)

    def get_top_suppressed_pairs(self, n: int = 5) -> list:
        """Return the N pairs with the most suppressions in the rolling window."""
        self._prune()
        pair_counts: Dict[str, int] = defaultdict(int)
        for event in self._events:
            pair_counts[event.symbol] += 1
        return sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:n]


# Module-level singleton for use across the application
_tracker: Optional[SuppressionTracker] = None


def get_tracker() -> SuppressionTracker:
    """Return the global SuppressionTracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = SuppressionTracker()
    return _tracker


def record_suppression(
    symbol: str,
    channel: str,
    reason: str,
    regime: Optional[str] = None,
    would_be_confidence: float = 0.0,
    **extra,
) -> None:
    """Convenience function to record a suppression event."""
    get_tracker().record(SuppressionEvent(
        timestamp=time.time(),
        symbol=symbol,
        channel=channel,
        reason=reason,
        regime=regime,
        would_be_confidence=would_be_confidence,
        extra=dict(extra),
    ))
```

### Change 2 — Integrate `record_suppression` into scanner pipeline

**File:** `src/scanner/__init__.py` and `src/scanner.py`

At each suppression point, call `record_suppression`:

```python
from src.suppression_telemetry import record_suppression, SuppressionTracker

# 1. QUIET regime block
if _regime_key in incompatible_regimes:
    record_suppression(
        symbol=symbol, channel=chan_name, reason=SuppressionTracker.REASON_QUIET_REGIME,
        regime=_regime_key, would_be_confidence=pre_filter_confidence,
    )
    continue

# 2. Pair quality gate failure
if not quality.passed:
    reason = SuppressionTracker.REASON_SPREAD_GATE if "spread" in quality.reason \
        else SuppressionTracker.REASON_VOLUME_GATE
    record_suppression(
        symbol=symbol, channel=chan_name, reason=reason,
        regime=_regime_key, would_be_confidence=0.0,
        spread_pct=spread_pct, quality_reason=quality.reason,
    )
    continue

# 3. OI hard rejection
if oi_eval.invalidated:
    record_suppression(
        symbol=symbol, channel=chan_name, reason=SuppressionTracker.REASON_OI_INVALIDATION,
        regime=_regime_key, would_be_confidence=pre_score,
        oi_change_pct=oi_eval.oi_change_pct,
    )
    continue

# 4. Cluster suppression
if cluster_suppressed:
    record_suppression(
        symbol=symbol, channel=chan_name, reason=SuppressionTracker.REASON_CLUSTER,
        regime=_regime_key, would_be_confidence=final_score,
    )
    continue

# 5. Stat filter
if stat_filtered:
    record_suppression(
        symbol=symbol, channel=chan_name, reason=SuppressionTracker.REASON_STAT_FILTER,
        regime=_regime_key, would_be_confidence=final_score,
    )
    continue
```

### Change 3 — Add suppression counters to `src/telemetry.py`

**File:** `src/telemetry.py`

```python
# Suppression counters — incremented by record_suppression()
suppressed_by_regime:   int = 0
suppressed_by_quality:  int = 0
suppressed_by_oi:       int = 0
suppressed_by_cluster:  int = 0
suppressed_by_stat:     int = 0
suppressed_by_lifespan: int = 0

def increment_suppression(reason: str) -> None:
    """Thread-safe suppression counter increment."""
    if reason == SuppressionTracker.REASON_QUIET_REGIME:
        _telemetry.suppressed_by_regime += 1
    elif reason in (SuppressionTracker.REASON_SPREAD_GATE, SuppressionTracker.REASON_VOLUME_GATE):
        _telemetry.suppressed_by_quality += 1
    elif reason == SuppressionTracker.REASON_OI_INVALIDATION:
        _telemetry.suppressed_by_oi += 1
    elif reason == SuppressionTracker.REASON_CLUSTER:
        _telemetry.suppressed_by_cluster += 1
    elif reason == SuppressionTracker.REASON_STAT_FILTER:
        _telemetry.suppressed_by_stat += 1
    elif reason == SuppressionTracker.REASON_LIFESPAN:
        _telemetry.suppressed_by_lifespan += 1
```

### Change 4 — Upgrade suppression log level to INFO

All scanner suppression `_log.debug(...)` calls for regime, quality, OI, and cluster should be upgraded to `_log.info(...)` for production visibility:

```python
# Before
_log.debug("regime_incompatible sym=%s chan=%s regime=%s", symbol, chan_name, _regime_key)

# After
_log.info(
    "signal_suppressed sym=%s chan=%s regime=%s reason=quiet_regime",
    symbol, chan_name, _regime_key,
)
```

### Change 5 — Add `/suppressed` Telegram command

**File:** `src/telegram_bot.py` (or `src/commands/`)

```python
async def cmd_suppressed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /suppressed command — shows recent suppression digest."""
    tracker = suppression_telemetry.get_tracker()
    digest = tracker.format_telegram_digest()

    top_pairs = tracker.get_top_suppressed_pairs(n=5)
    if top_pairs:
        digest += "\n\n📊 *Most Suppressed Pairs:*\n"
        for symbol, count in top_pairs:
            digest += f"  {symbol}: {count}\n"

    await update.message.reply_text(digest, parse_mode="Markdown")
```

Register in command handler list:

```python
application.add_handler(CommandHandler("suppressed", cmd_suppressed))
```

### Change 6 — Periodic Telegram digest

**File:** `src/scanner.py` or `src/main.py`

```python
async def _send_suppression_digest(telegram_client, tracker: SuppressionTracker) -> None:
    """Send suppression digest every 4 hours."""
    while True:
        await asyncio.sleep(4 * 3600)
        digest = tracker.format_telegram_digest()
        await telegram_client.send_message(digest)
```

---

## Modules Affected

| Module | Change |
|--------|--------|
| `src/suppression_telemetry.py` | **New file** — `SuppressionEvent`, `SuppressionTracker`, singleton |
| `src/scanner/__init__.py` | Call `record_suppression` at each suppression point |
| `src/scanner.py` | Same as above |
| `src/telemetry.py` | Add suppression counters and `increment_suppression()` |
| `src/telegram_bot.py` | Add `/suppressed` command handler |
| `src/main.py` | Start periodic digest coroutine |

---

## Test Cases

1. **`test_suppression_record_and_count`** — Record 3 QUIET events; `summary()["quiet_regime"]` == 3.
2. **`test_suppression_rolling_window`** — Events older than `window_seconds` are pruned from counts.
3. **`test_suppression_format_telegram`** — `format_telegram_digest()` includes all recorded reasons.
4. **`test_top_suppressed_pairs`** — Pair with 5 events appears first in `get_top_suppressed_pairs(3)`.
5. **`test_singleton_shared`** — Two `get_tracker()` calls return the same object.
6. **`test_telemetry_counters_incremented`** — After a scanner suppression, `telemetry.suppressed_by_regime` increments.
7. **`test_cmd_suppressed_response`** — Telegram `/suppressed` command returns formatted digest string.

---

## Rollback Procedure

1. Remove `src/suppression_telemetry.py`.
2. Remove `record_suppression` calls from scanner (they are additive — no existing logic changed).
3. Remove `/suppressed` command handler.
4. Remove suppression counters from `telemetry.py`.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Memory bloat from `deque(maxlen=5000)` | Low | 5,000 dataclass instances ≈ 2–3 MB max |
| Telegram rate limit hit by `/suppressed` spam | Low | Command accessible only to authorised users (existing auth middleware) |
| `record_suppression` in hot path adds latency | Low | All operations are O(1) append to deque; no I/O in hot path |

---

## Expected Impact

- **Full observability** into every signal that is discarded
- **Quantified signal loss** by reason, enabling data-driven threshold tuning
- **Periodic Telegram digest** gives operators actionable intelligence every 4 hours
- **Enables confidence in PRs 01–03** — suppression counts before/after deployment measure improvement directly
