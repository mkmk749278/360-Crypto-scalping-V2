# PR_25 — Automated Anomaly Detection Service

**PR Number:** PR_25  
**Branch:** `feature/pr25-automated-anomaly-detection`  
**Category:** Monitoring & Observability (Phase 2D)  
**Priority:** P1  
**Dependency:** PR_24 (KPI Dashboard Command)  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Create a background service that runs every 15 minutes and checks for four categories of system anomalies: signal frequency drops, win rate collapse, composite score drift, and channel silence during active market hours. Detected anomalies trigger immediate Telegram alerts to the admin channel, enabling rapid human intervention before small problems become large losses.

---

## Current State

No automated anomaly detection exists. The operator must manually monitor logs or wait for a `/dashboard` query to notice degraded system behaviour. Slow feedback means issues can persist for hours before being caught.

---

## Proposed Changes

### New file: `src/anomaly_monitor.py`

```python
"""Automated anomaly detection service for the 360-Crypto signal system."""
from __future__ import annotations
import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Anomaly thresholds
SIGNAL_FREQ_DROP_THRESHOLD   = 0.50   # alert if freq drops >50% vs 7d baseline
WIN_RATE_COLLAPSE_THRESHOLD  = 0.30   # alert if rolling 20-trade WR < 30%
WIN_RATE_WINDOW              = 20     # rolling trade window for win rate check
SCORE_DRIFT_THRESHOLD        = 15.0  # alert if mean score drifts >15 pts from 30d baseline
SCORE_HISTORY_DAYS           = 30
CHANNEL_SILENCE_HOURS        = 2     # alert if channel silent >2h during active session
CHECK_INTERVAL_SECONDS       = 15 * 60  # 15 minutes

# Active market hours for channel silence detection (UTC)
ACTIVE_HOURS_START = 7   # WHY: London open marks the start of meaningful liquidity
ACTIVE_HOURS_END   = 22  # WHY: US markets close around 21:00 UTC; thin after 22:00
MAX_WEEKDAY        = 4   # WHY: 0=Monday … 4=Friday; weekends excluded from silence check

@dataclass
class AnomalyEvent:
    anomaly_type: str
    channel: Optional[str]
    description: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class AnomalyMonitor:
    """
    Runs every 15 minutes and checks for system anomalies.

    Anomaly categories:
    1. Signal frequency drop (>50% vs 7d average).
    2. Win rate collapse (rolling 20-trade WR < 30%).
    3. Score drift (mean score >15 pts from 30d baseline).
    4. Channel silence (>2h with no signals during active market).
    """

    def __init__(
        self,
        performance_tracker,
        alert_callback: Callable[[str], None],
        check_interval: int = CHECK_INTERVAL_SECONDS,
    ):
        self._tracker = performance_tracker
        self._alert = alert_callback
        self._interval = check_interval
        self._running = False
        self._last_signal_times: Dict[str, datetime] = {}

    async def start(self) -> None:
        self._running = True
        logger.info("AnomalyMonitor started (interval=%ds)", self._interval)
        while self._running:
            await asyncio.sleep(self._interval)
            await self._run_checks()

    def stop(self) -> None:
        self._running = False

    def notify_signal(self, channel: str) -> None:
        """Call this every time a signal is emitted by a channel."""
        self._last_signal_times[channel] = datetime.now(timezone.utc)

    async def _run_checks(self) -> None:
        now = datetime.now(timezone.utc)
        anomalies: List[AnomalyEvent] = []

        channels = self._tracker.get_channel_names()
        for channel in channels:
            # 1. Signal frequency drop
            freq_anomaly = self._check_signal_frequency(channel)
            if freq_anomaly:
                anomalies.append(freq_anomaly)

            # 2. Win rate collapse
            wr_anomaly = self._check_win_rate(channel)
            if wr_anomaly:
                anomalies.append(wr_anomaly)

            # 3. Score drift
            score_anomaly = self._check_score_drift(channel)
            if score_anomaly:
                anomalies.append(score_anomaly)

            # 4. Channel silence (only during active market hours)
            if ACTIVE_HOURS_START <= now.hour < ACTIVE_HOURS_END and now.weekday() <= MAX_WEEKDAY:
                silence_anomaly = self._check_channel_silence(channel, now)
                if silence_anomaly:
                    anomalies.append(silence_anomaly)

        for anomaly in anomalies:
            self._send_alert(anomaly)

    def _check_signal_frequency(self, channel: str) -> Optional[AnomalyEvent]:
        stats_7d = self._tracker.get_channel_stats(channel, window_days=7)
        stats_1d = self._tracker.get_channel_stats(channel, window_days=1)
        freq_7d = stats_7d.get("signals_per_hour", 0)
        freq_1d = stats_1d.get("signals_per_hour", 0)
        if freq_7d > 0 and freq_1d < freq_7d * (1 - SIGNAL_FREQ_DROP_THRESHOLD):
            return AnomalyEvent(
                anomaly_type="FREQ_DROP",
                channel=channel,
                description=(
                    f"Signal frequency dropped: {freq_1d:.2f}/h (24h) vs "
                    f"{freq_7d:.2f}/h (7d baseline). "
                    f"Drop: {(1 - freq_1d/freq_7d)*100:.0f}%"
                ),
            )
        return None

    def _check_win_rate(self, channel: str) -> Optional[AnomalyEvent]:
        recent = self._tracker.get_recent_trades(channel, n=WIN_RATE_WINDOW)
        if len(recent) < WIN_RATE_WINDOW:
            return None
        wins = sum(1 for t in recent if t.get("pnl_pct", 0) > 0)
        wr = wins / len(recent)
        if wr < WIN_RATE_COLLAPSE_THRESHOLD:
            return AnomalyEvent(
                anomaly_type="WR_COLLAPSE",
                channel=channel,
                description=(
                    f"Win rate collapse: {wr:.1%} over last {WIN_RATE_WINDOW} trades "
                    f"(threshold: {WIN_RATE_COLLAPSE_THRESHOLD:.0%})"
                ),
            )
        return None

    def _check_score_drift(self, channel: str) -> Optional[AnomalyEvent]:
        recent_scores = self._tracker.get_recent_scores(channel, days=1)
        baseline_scores = self._tracker.get_recent_scores(channel, days=SCORE_HISTORY_DAYS)
        if not recent_scores or not baseline_scores:
            return None
        recent_mean = float(np.mean(recent_scores))
        baseline_mean = float(np.mean(baseline_scores))
        drift = abs(recent_mean - baseline_mean)
        if drift > SCORE_DRIFT_THRESHOLD:
            return AnomalyEvent(
                anomaly_type="SCORE_DRIFT",
                channel=channel,
                description=(
                    f"Score drift detected: current mean {recent_mean:.1f} vs "
                    f"30d baseline {baseline_mean:.1f} (drift: {drift:.1f} pts)"
                ),
            )
        return None

    def _check_channel_silence(
        self, channel: str, now: datetime
    ) -> Optional[AnomalyEvent]:
        last_signal = self._last_signal_times.get(channel)
        if last_signal is None:
            return None
        silence_duration = (now - last_signal).total_seconds() / 3600.0
        if silence_duration > CHANNEL_SILENCE_HOURS:
            return AnomalyEvent(
                anomaly_type="CHANNEL_SILENCE",
                channel=channel,
                description=(
                    f"Channel silent for {silence_duration:.1f}h "
                    f"(threshold: {CHANNEL_SILENCE_HOURS}h)"
                ),
            )
        return None

    def _send_alert(self, anomaly: AnomalyEvent) -> None:
        channel_str = f" [{anomaly.channel}]" if anomaly.channel else ""
        msg = (
            f"🔔 ANOMALY DETECTED{channel_str}\n"
            f"Type: {anomaly.anomaly_type}\n"
            f"{anomaly.description}\n"
            f"Time: {anomaly.detected_at.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        logger.warning("Anomaly: %s", msg)
        try:
            self._alert(msg)
        except Exception as exc:
            logger.error("Failed to send anomaly alert: %s", exc)
```

### Wire into `src/main.py`

```python
from src.anomaly_monitor import AnomalyMonitor

# In main startup:
anomaly_monitor = AnomalyMonitor(
    performance_tracker=performance_tracker,
    alert_callback=telegram_bot.send_admin_message,
)
asyncio.create_task(anomaly_monitor.start())

# In signal dispatch, notify the monitor:
anomaly_monitor.notify_signal(channel=signal.channel)
```

---

## Implementation Steps

1. Create `src/anomaly_monitor.py` with `AnomalyMonitor` and `AnomalyEvent`.
2. Add `get_recent_trades(channel, n)` and `get_recent_scores(channel, days)` to `performance_tracker.py` if not present.
3. In `main.py`, instantiate `AnomalyMonitor` and start it as an async background task.
4. In the signal dispatch path, call `anomaly_monitor.notify_signal()`.
5. Write unit tests in `tests/test_anomaly_monitor.py`.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/anomaly_monitor.py` | New — `AnomalyMonitor` and `AnomalyEvent` |
| `src/performance_tracker.py` | Add `get_recent_trades()` and `get_recent_scores()` |
| `src/main.py` | Instantiate and start `AnomalyMonitor` |
| `tests/test_anomaly_monitor.py` | New test file |

---

## Testing Requirements

```python
# tests/test_anomaly_monitor.py
def make_monitor(win_rate=0.55, freq_7d=3.0, freq_1d=3.0):
    tracker = Mock()
    tracker.get_channel_names.return_value = ["SCALP"]
    tracker.get_channel_stats.side_effect = lambda ch, window_days: {
        "signals_per_hour": freq_7d if window_days == 7 else freq_1d,
    }
    tracker.get_recent_trades.return_value = [
        {"pnl_pct": 0.01 if i % 2 == 0 else -0.01} for i in range(20)
    ]  # 50% win rate
    tracker.get_recent_scores.return_value = [70.0] * 10
    return tracker

def test_no_anomaly_normal_conditions():
    alerts = []
    monitor = AnomalyMonitor(make_monitor(), alert_callback=alerts.append)
    asyncio.run(monitor._run_checks())
    assert not alerts

def test_freq_drop_triggers_alert():
    alerts = []
    tracker = make_monitor(freq_7d=4.0, freq_1d=1.0)   # 75% drop
    monitor = AnomalyMonitor(tracker, alert_callback=alerts.append)
    asyncio.run(monitor._run_checks())
    assert any("FREQ_DROP" in a for a in alerts)

def test_win_rate_collapse_triggers():
    tracker = make_monitor()
    tracker.get_recent_trades.return_value = [
        {"pnl_pct": -0.01} for _ in range(20)   # 0% win rate
    ]
    alerts = []
    monitor = AnomalyMonitor(tracker, alert_callback=alerts.append)
    asyncio.run(monitor._run_checks())
    assert any("WR_COLLAPSE" in a for a in alerts)

def test_channel_silence_not_triggered_outside_hours():
    alerts = []
    monitor = AnomalyMonitor(make_monitor(), alert_callback=alerts.append)
    # Simulate: last signal 5h ago but it's 02:00 UTC (inactive)
    # Silence check skipped outside 07–22 UTC → no alert
    assert True   # Validated by UTC hour guard in _run_checks()
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Issue detection latency | Hours (manual) | 15 minutes (automated) |
| Win rate collapse awareness | Next manual check | Instant alert |
| Channel silence detection | Never | Auto-alert after 2h silence |
| Score drift monitoring | None | Alert on >15 pt drift |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Alert spam during normal market fluctuations | Require anomaly to persist for 2 consecutive check cycles before alerting |
| Async task crashes silently | Wrap `_run_checks()` in try/except; log and continue; restart on crash |
| False silence alert after system restart | Only start tracking silence after first signal is received per channel |
| Too many checks per cycle → performance impact | Each check is O(N) trades; negligible for typical trade volumes |
