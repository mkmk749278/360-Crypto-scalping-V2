# PR_17 — Cross-Channel Confluence Engine

**PR Number:** PR_17  
**Branch:** `feature/pr17-cross-channel-confluence-engine`  
**Category:** Signal Intelligence (Phase 2B)  
**Priority:** P1  
**Dependency:** PR_14 (Scanner Decomposition Part 1)  
**Effort estimate:** Medium (2–3 days)

---

## Objective

Detect when two or more channels fire on the same pair in the same direction within a 5-minute window. When confluence is detected, boost the composite score by +15 points and emit a dedicated confluence alert in addition to the individual channel alerts. This replaces the previously attempted but unmerged PRs #118 and #119.

---

## Current State

PRs #118 and #119 attempted cross-channel confluence detection but were closed without merge due to architectural issues (tight coupling with the old signal dispatch path). The existing `src/confluence.py` (added in a previous session) contains partial logic but it is not fully integrated with the scoring engine or the signal router.

There is currently no mechanism to detect when multiple channels agree on a signal and amplify conviction accordingly.

---

## Proposed Changes

### `src/confluence.py` — `ConfluenceTracker` class

```python
"""Cross-channel confluence detection and score boosting."""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

CONFLUENCE_WINDOW_SECONDS = 300   # 5-minute detection window
CONFLUENCE_SCORE_BOOST    = 15    # points added to both signals when detected
CONFLUENCE_MIN_CHANNELS   = 2     # minimum distinct channels to trigger

@dataclass(frozen=True)
class _SignalKey:
    symbol: str
    direction: str   # "LONG" or "SHORT"

@dataclass
class _SignalRecord:
    channel: str
    timestamp: datetime
    score: float

class ConfluenceTracker:
    """Rolling 5-minute window tracker of signals per (symbol, direction)."""

    def __init__(
        self,
        window_seconds: int = CONFLUENCE_WINDOW_SECONDS,
        boost_pts: float = CONFLUENCE_SCORE_BOOST,
    ):
        self._window = timedelta(seconds=window_seconds)
        self._boost = boost_pts
        self._history: Dict[_SignalKey, deque] = defaultdict(deque)

    def register(
        self,
        symbol: str,
        direction: str,
        channel: str,
        score: float,
    ) -> Tuple[bool, float]:
        """
        Register a new signal.

        Returns (is_confluence, boosted_score).
        If confluence is detected the score is boosted and (True, boosted_score) returned.
        """
        key = _SignalKey(symbol=symbol.upper(), direction=direction.upper())
        now = datetime.now(timezone.utc)
        self._evict_stale(key, now)

        record = _SignalRecord(channel=channel, timestamp=now, score=score)
        self._history[key].append(record)

        distinct_channels = {r.channel for r in self._history[key]}
        if len(distinct_channels) >= CONFLUENCE_MIN_CHANNELS:
            boosted = min(100.0, score + self._boost)
            logger.info(
                "Confluence detected: %s %s — channels=%s score %.1f→%.1f",
                symbol, direction, distinct_channels, score, boosted,
            )
            return True, boosted

        return False, score

    def _evict_stale(self, key: _SignalKey, now: datetime) -> None:
        cutoff = now - self._window
        dq = self._history[key]
        while dq and dq[0].timestamp < cutoff:
            dq.popleft()
```

### Wire into `src/signal_router.py`

```python
# Module-level singleton
from src.confluence import ConfluenceTracker
_confluence_tracker = ConfluenceTracker()

# In dispatch path, after score is computed:
is_confluence, signal.post_ai_confidence = _confluence_tracker.register(
    symbol=signal.symbol,
    direction=signal.direction,
    channel=signal.channel,
    score=signal.post_ai_confidence,
)
if is_confluence:
    await _send_confluence_alert(signal)
```

### Confluence alert format

```
🔀 CONFLUENCE ALERT — BTCUSDT LONG
Channels: 360_SCALP + 360_SWING
Score: 78 → 93 (+15 boost)
Entry zone: 67,450–67,650
```

---

## Implementation Steps

1. Verify or update `src/confluence.py` to match the `ConfluenceTracker` specification above.
2. Add `CONFLUENCE_WINDOW_SECONDS` and `CONFLUENCE_SCORE_BOOST` constants to `config/__init__.py` as env-var overrides.
3. Instantiate `ConfluenceTracker` as a module-level singleton in `signal_router.py`.
4. Call `_confluence_tracker.register()` in the dispatch path before alert emission.
5. Implement `_send_confluence_alert()` with a distinct Telegram message format.
6. Write unit tests in `tests/test_confluence.py`.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/confluence.py` | Implement/update `ConfluenceTracker` class |
| `src/signal_router.py` | Wire in confluence detection and alert |
| `src/config/__init__.py` | Add confluence config constants |
| `tests/test_confluence.py` | New test file |

---

## Testing Requirements

```python
# tests/test_confluence.py
def test_single_channel_no_confluence():
    ct = ConfluenceTracker()
    is_conf, score = ct.register("BTCUSDT", "LONG", "SCALP", 70.0)
    assert not is_conf
    assert score == 70.0

def test_two_channels_same_pair_triggers_confluence():
    ct = ConfluenceTracker()
    ct.register("BTCUSDT", "LONG", "SCALP", 70.0)
    is_conf, score = ct.register("BTCUSDT", "LONG", "SWING", 72.0)
    assert is_conf
    assert score == 87.0  # 72 + 15

def test_opposite_direction_no_confluence():
    ct = ConfluenceTracker()
    ct.register("BTCUSDT", "LONG",  "SCALP", 70.0)
    is_conf, _ = ct.register("BTCUSDT", "SHORT", "SWING", 72.0)
    assert not is_conf

def test_stale_signals_evicted():
    ct = ConfluenceTracker(window_seconds=1)
    ct.register("BTCUSDT", "LONG", "SCALP", 70.0)
    import time; time.sleep(1.1)
    is_conf, _ = ct.register("BTCUSDT", "LONG", "SWING", 72.0)
    assert not is_conf  # first signal evicted

def test_score_capped_at_100():
    ct = ConfluenceTracker()
    ct.register("BTCUSDT", "LONG", "SCALP", 90.0)
    _, score = ct.register("BTCUSDT", "LONG", "SWING", 90.0)
    assert score == 100.0
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Multi-channel agreement utilisation | None | +15 pt score boost on confluence |
| Cross-channel alert visibility | Zero | Dedicated confluence alert message |
| False positive rate on confluence signals | N/A (no baseline) | Expected <15% based on prior research |
| Score inflation risk | N/A | Score cap at 100 prevents over-inflation |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Same channel firing twice on slightly different triggers | `distinct_channels` check ensures only different channels count |
| Clock drift across async tasks | Use `datetime.now(timezone.utc)` consistently; within single process, drift is negligible |
| Window too short missing genuine confluence | Window configurable via env var `CONFLUENCE_WINDOW_SECONDS` |
| Score boost too high distorting sizing | Boost configurable; starts at 15 pts with monitoring |
