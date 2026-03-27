# PR-SIG-OPT-07 — Suppressed Signal Analytics Dashboard

**Priority:** P3 — Observability; enables data-driven tuning of all prior PRs  
**Estimated Impact:** Enables admin visibility into suppression rates; allows rapid threshold adjustment without code changes  
**Dependencies:** None (standalone analytics layer); benefits most from PRs 01–06 being deployed first  
**Relates To:** Extends `src/suppression_telemetry.py` and the existing `/suppressed` Telegram command  
**Status:** 📋 Planned

---

## Objective

Add a `SuppressionAnalytics` class to `src/suppression_telemetry.py` that provides
rolling 1h/6h/24h suppression breakdowns, per-pair and per-channel suppression rates,
"would-be" confidence distribution of suppressed signals, and JSON persistence across
restarts. Add two new Telegram admin commands: `/suppression_report` and
`/suppression_pairs`. Add cycle-end suppression summary logging at INFO level.

---

## Problem Analysis

### Current State: `src/suppression_telemetry.py`

The existing `SuppressionTracker` class provides:
- Rolling 4h event window (`_DEFAULT_WINDOW_SECONDS = 4 * 3600`)
- Summary by reason, channel, symbol
- `format_telegram_digest()` for Telegram output

**Gaps:**
1. **No multi-window breakdowns**: Only 4h window; cannot compare 1h vs 6h vs 24h rates
2. **No suppression rate**: Tracks absolute counts but not `suppressed / evaluated` ratio
3. **No "would-be" confidence distribution**: We know signals were blocked, but not their confidence scores
4. **No persistence**: Data lost on restart — cannot diagnose issues that occur during low-traffic periods
5. **No auto-alert**: Admin has no automated notification when suppression rate exceeds normal bounds
6. **Logged at DEBUG**: `_suppression_counters` cycle summary is logged at DEBUG (invisible in production)

### Current Commands: `src/commands/`

Directory contains: `__init__.py`, `backtest.py`, `channels.py`, `deploy.py`,
`engine.py`, `portfolio.py`, `registry.py`, `signals.py`

No suppression-specific command exists. The suppression summary is accessible via the
existing `/suppressed` command (if it exists) or the scanner's internal counters.

### `src/scanner/__init__.py` — Cycle-End Logging

```python
# Suppression counters are tracked in self._suppression_counters (defaultdict(int))
# but cycle-end summary is logged at DEBUG level or not at all
```

---

## Required Changes

### Change 1 — Add `SuppressionAnalytics` class to `src/suppression_telemetry.py`

Add after `SuppressionTracker` class:

```python
import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Multi-window analytics
# ---------------------------------------------------------------------------

_ANALYTICS_WINDOWS: Dict[str, float] = {
    "1h":  1 * 3600.0,
    "6h":  6 * 3600.0,
    "24h": 24 * 3600.0,
}

_ANALYTICS_PERSIST_PATH: str = os.getenv(
    "SUPPRESSION_ANALYTICS_PATH", "data/suppression_analytics.json"
)

# Auto-alert threshold: if suppression rate for any channel exceeds this
# fraction for 1h, send an admin Telegram alert.
_AUTO_ALERT_SUPPRESSION_RATE: float = float(
    os.getenv("SUPPRESSION_ALERT_RATE_THRESHOLD", "0.90")
)


@dataclass
class SuppressionWindowStats:
    """Stats for a single rolling time window."""
    window_label: str
    total_suppressed: int
    by_reason: Dict[str, int]
    by_channel: Dict[str, int]
    top_pairs: List[tuple]          # [(symbol, count), ...]
    suppression_rate_by_channel: Dict[str, float]   # channel → suppressed/evaluated
    confidence_distribution: Dict[str, int]          # "50-60", "60-70", "70-80", "80+"


class SuppressionAnalytics:
    """Enhanced suppression analytics with multi-window tracking and persistence.

    Parameters
    ----------
    persist_path:
        Path to JSON file for persistence across restarts.
    admin_alert_callback:
        Optional async callable accepting a string message for Telegram admin alerts.

    Usage::

        analytics = SuppressionAnalytics(admin_alert_callback=send_admin_alert)
        analytics.record(SuppressionEvent(...), signals_evaluated=10)
        report = analytics.get_report("1h")
    """

    def __init__(
        self,
        persist_path: str = _ANALYTICS_PERSIST_PATH,
        admin_alert_callback=None,
    ) -> None:
        self._tracker_1h = SuppressionTracker(window_seconds=3600.0)
        self._tracker_6h = SuppressionTracker(window_seconds=6 * 3600.0)
        self._tracker_24h = SuppressionTracker(window_seconds=24 * 3600.0)
        self._trackers = {
            "1h": self._tracker_1h,
            "6h": self._tracker_6h,
            "24h": self._tracker_24h,
        }
        # Per-channel evaluation counters (for rate calculation)
        self._evaluated_by_channel: Dict[str, int] = defaultdict(int)
        self._persist_path = persist_path
        self._admin_alert = admin_alert_callback
        self._last_alert_ts: float = 0.0
        self._alert_cooldown: float = 3600.0  # 1h between alerts

        # Load persisted data if available
        self._load_from_disk()

    def record(self, event: SuppressionEvent, signals_evaluated: int = 0) -> None:
        """Record a suppression event across all time windows.

        Parameters
        ----------
        event:
            The suppression event to record.
        signals_evaluated:
            Number of signals evaluated for this channel in the current cycle.
            Used to compute suppression rate.
        """
        for tracker in self._trackers.values():
            tracker.record(event)
        if signals_evaluated > 0:
            self._evaluated_by_channel[event.channel] += signals_evaluated

    def record_evaluation(self, channel: str, count: int = 1) -> None:
        """Record that `count` signals were evaluated for a channel (not suppressed)."""
        self._evaluated_by_channel[channel] += count

    def get_report(self, window: str = "1h") -> SuppressionWindowStats:
        """Generate a stats report for the specified time window.

        Parameters
        ----------
        window:
            One of "1h", "6h", "24h".
        """
        tracker = self._trackers.get(window, self._tracker_1h)
        by_reason = tracker.summary()
        by_channel = tracker.by_channel()
        top_pairs = tracker.by_symbol(top_n=10)

        # Compute suppression rates
        suppression_rates: Dict[str, float] = {}
        for ch, suppressed_count in by_channel.items():
            evaluated = self._evaluated_by_channel.get(ch, 0)
            if evaluated > 0:
                suppression_rates[ch] = suppressed_count / (suppressed_count + evaluated)
            else:
                suppression_rates[ch] = 1.0 if suppressed_count > 0 else 0.0

        # Confidence distribution of suppressed signals
        conf_dist: Dict[str, int] = {"<50": 0, "50-60": 0, "60-70": 0, "70-80": 0, "80+": 0}
        for evt in tracker.recent_events(limit=10000):
            c = getattr(evt, "would_be_confidence", 0.0) or 0.0
            if c >= 80:
                conf_dist["80+"] += 1
            elif c >= 70:
                conf_dist["70-80"] += 1
            elif c >= 60:
                conf_dist["60-70"] += 1
            elif c >= 50:
                conf_dist["50-60"] += 1
            elif c > 0:
                conf_dist["<50"] += 1

        return SuppressionWindowStats(
            window_label=window,
            total_suppressed=tracker.total_in_window(),
            by_reason=by_reason,
            by_channel=by_channel,
            top_pairs=top_pairs,
            suppression_rate_by_channel=suppression_rates,
            confidence_distribution=conf_dist,
        )

    def check_auto_alert(self) -> Optional[str]:
        """Check if suppression rate exceeds alert threshold; return alert message or None."""
        now = time.monotonic()
        if now - self._last_alert_ts < self._alert_cooldown:
            return None
        report = self.get_report("1h")
        high_rate_channels = [
            (ch, rate)
            for ch, rate in report.suppression_rate_by_channel.items()
            if rate >= _AUTO_ALERT_SUPPRESSION_RATE and report.by_channel.get(ch, 0) >= 10
        ]
        if not high_rate_channels:
            return None
        self._last_alert_ts = now
        lines = [
            "🚨 *High Signal Suppression Alert*",
            f"Channels with >{_AUTO_ALERT_SUPPRESSION_RATE*100:.0f}% suppression rate (last 1h):",
        ]
        for ch, rate in high_rate_channels:
            lines.append(f"  • {ch}: {rate*100:.0f}% suppressed")
        lines.append("Use `/suppression_report` for full breakdown.")
        return "\n".join(lines)

    def format_full_report(self, window: str = "6h") -> str:
        """Format a comprehensive Telegram-ready suppression report."""
        report = self.get_report(window)
        lines = [
            f"📊 *Suppression Analytics — {window} window*",
            f"Total suppressed: *{report.total_suppressed}*",
            "",
        ]

        if report.by_reason:
            lines.append("*By suppression reason:*")
            reason_labels = {
                REASON_QUIET_REGIME: "Quiet/Ranging regime",
                REASON_SPREAD_GATE: "Spread too wide",
                REASON_VOLUME_GATE: "Volume too thin",
                REASON_OI_INVALIDATION: "OI invalidation",
                REASON_CLUSTER: "Cluster suppression",
                REASON_STAT_FILTER: "Stat filter (poor history)",
                REASON_LIFESPAN: "Signal too young",
                REASON_CONFIDENCE: "Below confidence floor",
            }
            for reason, count in sorted(report.by_reason.items(), key=lambda kv: -kv[1]):
                label = reason_labels.get(reason, reason)
                lines.append(f"  • {label}: {count}")
            lines.append("")

        if report.suppression_rate_by_channel:
            lines.append("*Suppression rate by channel:*")
            for ch, rate in sorted(report.suppression_rate_by_channel.items(), key=lambda kv: -kv[1]):
                bar = "🔴" if rate > 0.9 else "🟡" if rate > 0.7 else "🟢"
                lines.append(f"  {bar} {ch}: {rate*100:.0f}%")
            lines.append("")

        if report.top_pairs:
            lines.append("*Top 10 suppressed pairs:*")
            for sym, count in report.top_pairs[:10]:
                lines.append(f"  • {sym}: {count}")
            lines.append("")

        if any(v > 0 for v in report.confidence_distribution.values()):
            lines.append("*Would-be confidence distribution:*")
            for bucket, count in report.confidence_distribution.items():
                if count > 0:
                    lines.append(f"  • {bucket}: {count} signals")

        return "\n".join(lines)

    def format_pairs_report(self, top_n: int = 15) -> str:
        """Format a Telegram-ready report of most-suppressed pairs with reasons."""
        report_6h = self.get_report("6h")
        lines = [
            f"🔕 *Most Suppressed Pairs (6h)*",
            "",
        ]
        # Get per-pair reason breakdown
        tracker = self._tracker_6h
        tracker._prune()
        pair_reasons: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for evt in tracker._events:
            pair_reasons[evt.symbol][evt.reason] += 1

        sorted_pairs = sorted(pair_reasons.items(), key=lambda kv: -sum(kv[1].values()))
        for sym, reasons in sorted_pairs[:top_n]:
            total = sum(reasons.values())
            top_reason = max(reasons.items(), key=lambda kv: kv[1])[0]
            lines.append(f"*{sym}* — {total} suppressions")
            lines.append(f"  Top reason: {top_reason}")
        return "\n".join(lines)

    def save_to_disk(self) -> None:
        """Persist current analytics to JSON for cross-restart continuity."""
        try:
            Path(self._persist_path).parent.mkdir(parents=True, exist_ok=True)
            data = {
                "evaluated_by_channel": dict(self._evaluated_by_channel),
                "last_saved": time.time(),
            }
            with open(self._persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            log.warning("Failed to save suppression analytics: {}", exc)

    def _load_from_disk(self) -> None:
        """Load persisted analytics from JSON if available."""
        try:
            if not Path(self._persist_path).exists():
                return
            with open(self._persist_path) as f:
                data = json.load(f)
            evaluated = data.get("evaluated_by_channel", {})
            for ch, count in evaluated.items():
                self._evaluated_by_channel[ch] = int(count)
            log.info(
                "Loaded suppression analytics from disk ({} channels)", len(evaluated)
            )
        except Exception as exc:
            log.debug("Could not load suppression analytics from disk: {}", exc)
```

### Change 2 — Add `/suppression_report` and `/suppression_pairs` Commands

**File:** `src/commands/signals.py` (or create `src/commands/suppression.py`)

```python
# In the command registry, add:

async def cmd_suppression_report(update, context, analytics: SuppressionAnalytics) -> None:
    """Handler for /suppression_report admin command.

    Shows full suppression breakdown for 6h window with rates by channel.
    Admin-only command.
    """
    report = analytics.format_full_report(window="6h")
    await update.message.reply_text(report, parse_mode="Markdown")


async def cmd_suppression_pairs(update, context, analytics: SuppressionAnalytics) -> None:
    """Handler for /suppression_pairs admin command.

    Shows top-15 most suppressed pairs with their primary suppression reason.
    Admin-only command.
    """
    report = analytics.format_pairs_report(top_n=15)
    await update.message.reply_text(report, parse_mode="Markdown")
```

Register in `src/commands/registry.py`:

```python
# Add to admin command registry
registry.register("/suppression_report", cmd_suppression_report, admin_only=True)
registry.register("/suppression_pairs", cmd_suppression_pairs, admin_only=True)
```

### Change 3 — Elevate Cycle-End Suppression Summary to INFO Level

**File:** `src/scanner/__init__.py` — end of `scan_loop()` cycle

```python
# Find the location where _suppression_counters are logged (near end of scan cycle)
# Before: logged at DEBUG or not at all

# After: log top suppressors at INFO level each cycle
def _log_cycle_suppression_summary(self) -> None:
    """Log top suppression reasons at INFO level for monitoring."""
    if not self._suppression_counters:
        return
    top_reasons = sorted(
        self._suppression_counters.items(),
        key=lambda kv: -kv[1]
    )[:5]
    summary = " | ".join(f"{k}={v}" for k, v in top_reasons)
    log.info(
        "Scan cycle suppression summary (top 5): {}",
        summary,
    )
```

Call this at the end of each scan cycle in `scan_loop()`.

### Change 4 — Feed Suppression Events into `SuppressionAnalytics`

**File:** `src/scanner/__init__.py`

Replace direct `suppression_tracker.record()` calls with `analytics.record()`:

```python
# Existing pattern in _should_skip_channel():
self.suppression_tracker.record(SuppressionEvent(
    symbol=symbol,
    channel=chan_name,
    reason=_supp_reason,
    regime=ctx.regime_result.regime.value,
))

# After — also feed into analytics
_evt = SuppressionEvent(
    symbol=symbol,
    channel=chan_name,
    reason=_supp_reason,
    regime=ctx.regime_result.regime.value,
    would_be_confidence=getattr(ctx, "pre_gate_confidence", 0.0),
)
self.suppression_tracker.record(_evt)
if hasattr(self, "_suppression_analytics"):
    self._suppression_analytics.record(_evt)
```

Add `_suppression_analytics` to `Scanner.__init__()`:

```python
from src.suppression_telemetry import SuppressionAnalytics
self._suppression_analytics: SuppressionAnalytics = SuppressionAnalytics(
    admin_alert_callback=self._admin_alert_fn  # passed from main engine
)
```

### Change 5 — Auto-Alert Integration

In the scan cycle end hook, check for high suppression rates and alert:

```python
# At end of each scan_loop() cycle
if hasattr(self, "_suppression_analytics"):
    alert_msg = self._suppression_analytics.check_auto_alert()
    if alert_msg and self._admin_alert_fn:
        asyncio.create_task(self._admin_alert_fn(alert_msg))
    # Periodically persist to disk (every 10 cycles)
    if self._scan_cycle_count % 10 == 0:
        self._suppression_analytics.save_to_disk()
```

---

## Expected Impact

| Feature | Before | After |
|---------|--------|-------|
| Suppression visibility | DEBUG logs only | INFO cycle summary + Telegram commands |
| Multi-window analysis | 4h rolling only | 1h / 6h / 24h breakdowns |
| Suppression rate tracking | None | Per-channel rate (suppressed/evaluated) |
| Confidence distribution | None | Bucketed distribution of would-be signals |
| Persistence | None (lost on restart) | JSON persisted to disk |
| Auto-alerting | None | Alert when >90% suppression rate for 1h |
| Admin commands | None specific | `/suppression_report`, `/suppression_pairs` |

---

## Testing Criteria

```bash
# Run targeted tests
python -m pytest tests/test_suppression_telemetry.py -v

# Verify SuppressionAnalytics class
python -c "
from src.suppression_telemetry import SuppressionAnalytics, SuppressionEvent, REASON_QUIET_REGIME
import tempfile, os

analytics = SuppressionAnalytics(persist_path='/tmp/test_suppression.json')

# Record events
for i in range(5):
    analytics.record(SuppressionEvent(
        symbol='TESTUSDT', channel='360_SCALP',
        reason=REASON_QUIET_REGIME, regime='QUIET',
        would_be_confidence=68.5
    ))
analytics.record_evaluation('360_SCALP', count=10)

# Get 1h report
report = analytics.get_report('1h')
assert report.total_suppressed == 5, f'Expected 5, got {report.total_suppressed}'
assert '360_SCALP' in report.by_channel
assert report.suppression_rate_by_channel['360_SCALP'] == 5/15  # 5/(5+10)

# Test Telegram formatting
msg = analytics.format_full_report('1h')
assert 'Suppression Analytics' in msg
assert '360_SCALP' in msg

# Test persistence
analytics.save_to_disk()
analytics2 = SuppressionAnalytics(persist_path='/tmp/test_suppression.json')
assert analytics2._evaluated_by_channel.get('360_SCALP', 0) == 10

print('SuppressionAnalytics tests: PASS ✅')
os.unlink('/tmp/test_suppression.json')
"

# Test auto-alert threshold
python -c "
from src.suppression_telemetry import SuppressionAnalytics, SuppressionEvent, REASON_QUIET_REGIME
analytics = SuppressionAnalytics()
# Record 20 suppressions with 0 evaluations → rate = 100%
for _ in range(20):
    analytics.record(SuppressionEvent(
        symbol='TESTUSDT', channel='360_SCALP',
        reason=REASON_QUIET_REGIME, regime='QUIET',
    ))
alert = analytics.check_auto_alert()
assert alert is not None, 'Expected auto-alert but got None'
assert '360_SCALP' in alert
print(f'Auto-alert test: PASS ✅')
print(f'Alert message: {alert}')
"

# Env var: override alert threshold
SUPPRESSION_ALERT_RATE_THRESHOLD=0.5 python -m pytest tests/test_suppression_telemetry.py -v
```
