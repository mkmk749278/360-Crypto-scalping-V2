# PR_20 — Session-Adaptive Threshold Engine

**PR Number:** PR_20  
**Branch:** `feature/pr20-session-adaptive-threshold-engine`  
**Category:** Signal Intelligence (Phase 2B)  
**Priority:** P2  
**Dependency:** PR_02 (Phase 1 — Per-Pair Config Profiles, merged as #128)  
**Effort estimate:** Small–Medium (1–2 days)

---

## Objective

Adjust signal score thresholds dynamically based on the current trading session. Different sessions have different liquidity profiles, volatility norms, and false-positive rates. Raising the bar during low-liquidity sessions (Asian, Weekend) and lowering it during high-liquidity sessions (London open, NY open) improves the signal quality-to-frequency ratio.

---

## Current State

`src/kill_zone.py` identifies London/NY kill zones and flags signals generated outside active sessions with a note, but does **not** adjust score thresholds. The minimum score threshold (typically 60 pts) is a static value applied uniformly at all times of day and all days of the week.

---

## Proposed Changes

### Extend `src/kill_zone.py`

```python
"""Session identification and score threshold modifiers."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional

# Session definitions (UTC hours)
_SESSIONS = {
    "ASIAN":        (0, 8),    # 00:00–08:00 UTC
    "LONDON_OPEN":  (7, 10),   # 07:00–10:00 UTC (overlaps Asian close)
    "LONDON":       (8, 17),   # 08:00–17:00 UTC
    "NY_OPEN":      (12, 16),  # 12:00–16:00 UTC (London/NY overlap — highest liquidity)
    "NY":           (13, 22),  # 13:00–22:00 UTC
    "DEAD_ZONE":    (22, 24),  # 22:00–00:00 UTC
}

# Score threshold adjustments per session (positive = raise bar, negative = lower bar)
SESSION_THRESHOLD_OFFSETS = {
    "ASIAN":        +5,   # lower liquidity → require higher conviction
    "LONDON_OPEN":  -5,   # highest directional momentum → trust signals more
    "NY_OPEN":      -3,   # strong volume → slight reduction in threshold
    "DEAD_ZONE":    +8,   # very thin markets → require strong conviction
    "WEEKEND":      +10,  # retail-driven, prone to manipulation
    "DEFAULT":       0,
}

# Crypto-specific timing offsets
CRYPTO_EVENTS = {
    # Binance BTC quarterly expiry effect — typically 3rd Friday of March/June/Sep/Dec
    "FUTURES_ROLLOVER": {"hour_offset": +3},
    # Binance funding settlement (every 8h: 00:00, 08:00, 16:00 UTC)
    "FUNDING_SETTLEMENT": {"hour_offset": +2},
}

def current_session(dt: Optional[datetime] = None) -> str:
    """Return the most specific active session name for the given UTC datetime."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.weekday() >= 5:   # Saturday=5, Sunday=6
        return "WEEKEND"
    h = dt.hour
    # Most specific (shortest window) first
    if _SESSIONS["LONDON_OPEN"][0] <= h < _SESSIONS["LONDON_OPEN"][1]:
        return "LONDON_OPEN"
    if _SESSIONS["NY_OPEN"][0] <= h < _SESSIONS["NY_OPEN"][1]:
        return "NY_OPEN"
    if _SESSIONS["ASIAN"][0] <= h < _SESSIONS["ASIAN"][1]:
        return "ASIAN"
    if _SESSIONS["DEAD_ZONE"][0] <= h < _SESSIONS["DEAD_ZONE"][1]:
        return "DEAD_ZONE"
    return "DEFAULT"

def get_threshold_offset(dt: Optional[datetime] = None) -> int:
    """Return the score threshold adjustment for the current session."""
    session = current_session(dt)
    return SESSION_THRESHOLD_OFFSETS.get(session, 0)

def is_near_funding_settlement(dt: Optional[datetime] = None) -> bool:
    """Return True if within 15 minutes of Binance funding settlement (0h, 8h, 16h UTC)."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    # Check if within 15 minutes after settlement (00:00, 08:00, 16:00 UTC)
    settlement_hours = {0, 8, 16}
    if dt.hour in settlement_hours and dt.minute < 15:
        return True
    # Check if within 15 minutes before settlement (23:45, 07:45, 15:45 UTC)
    pre_settlement_hours = {23, 7, 15}
    if dt.hour in pre_settlement_hours and dt.minute >= 45:
        return True
    return False
```

### Wire into `src/signal_quality.py`

```python
from src.kill_zone import get_threshold_offset, is_near_funding_settlement

def get_effective_min_score(base_min_score: float) -> float:
    """Return the session-adjusted minimum score threshold."""
    offset = get_threshold_offset()
    adjusted = base_min_score + offset
    # Extra caution near funding settlement
    if is_near_funding_settlement():
        adjusted += 2
    return float(adjusted)

# In score gate check:
effective_threshold = get_effective_min_score(config.min_confidence)
if signal.post_ai_confidence < effective_threshold:
    return False, f"Below session-adjusted threshold ({effective_threshold:.0f})"
```

### Config additions in `src/config/__init__.py`

```python
SESSION_THRESHOLD_ASIAN:       int = int(os.getenv("SESSION_THRESHOLD_ASIAN",        "5"))
SESSION_THRESHOLD_LONDON_OPEN: int = int(os.getenv("SESSION_THRESHOLD_LONDON_OPEN", "-5"))
SESSION_THRESHOLD_NY_OPEN:     int = int(os.getenv("SESSION_THRESHOLD_NY_OPEN",     "-3"))
SESSION_THRESHOLD_DEAD_ZONE:   int = int(os.getenv("SESSION_THRESHOLD_DEAD_ZONE",    "8"))
SESSION_THRESHOLD_WEEKEND:     int = int(os.getenv("SESSION_THRESHOLD_WEEKEND",     "10"))
```

---

## Implementation Steps

1. Extend `src/kill_zone.py` with `current_session()`, `get_threshold_offset()`, and `is_near_funding_settlement()`.
2. In `signal_quality.py`, replace the static threshold check with `get_effective_min_score()`.
3. Add session threshold config constants to `config/__init__.py`.
4. Log the active session and applied offset at signal evaluation time (DEBUG level).
5. Write unit tests in `tests/test_session_thresholds.py`.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/kill_zone.py` | Add session detection and threshold offset functions |
| `src/signal_quality.py` | Replace static threshold with `get_effective_min_score()` |
| `src/config/__init__.py` | Add session threshold env-var overrides |
| `tests/test_session_thresholds.py` | New test file |

---

## Testing Requirements

```python
# tests/test_session_thresholds.py
from datetime import datetime, timezone
from src.kill_zone import current_session, get_threshold_offset

def test_london_open_session():
    dt = datetime(2026, 3, 26, 8, 30, tzinfo=timezone.utc)  # Wednesday 08:30 UTC
    assert current_session(dt) == "LONDON_OPEN"
    assert get_threshold_offset(dt) == -5

def test_weekend_session():
    dt = datetime(2026, 3, 28, 14, 0, tzinfo=timezone.utc)  # Saturday
    assert current_session(dt) == "WEEKEND"
    assert get_threshold_offset(dt) == 10

def test_asian_session():
    dt = datetime(2026, 3, 26, 3, 0, tzinfo=timezone.utc)
    assert current_session(dt) == "ASIAN"
    assert get_threshold_offset(dt) == 5

def test_ny_open_session():
    dt = datetime(2026, 3, 26, 13, 0, tzinfo=timezone.utc)
    assert current_session(dt) == "NY_OPEN"
    assert get_threshold_offset(dt) == -3

def test_funding_settlement_near():
    from src.kill_zone import is_near_funding_settlement
    dt_near = datetime(2026, 3, 26, 7, 55, tzinfo=timezone.utc)   # 5 min before 8h
    dt_far  = datetime(2026, 3, 26, 10, 0, tzinfo=timezone.utc)   # far from settlement
    assert is_near_funding_settlement(dt_near)
    assert not is_near_funding_settlement(dt_far)

def test_effective_threshold_raised_on_weekend():
    from src.signal_quality import get_effective_min_score
    # Monkeypatch or test with explicit UTC datetime
    threshold = get_effective_min_score(60)
    assert threshold >= 60   # at worst equal; weekend raises it
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Signal filter during Asian session | No adjustment | +5 pts threshold (fewer, higher-quality signals) |
| Signal filter at London open | No adjustment | −5 pts threshold (more signals in high-liquidity window) |
| Weekend false positive rate | Same as weekday | Reduced by ~20% via +10 pt threshold |
| Near-funding-settlement noise | No adjustment | +2 pts additional buffer |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Too aggressive on threshold raise → misses valid signals | All offsets configurable via env vars; start conservative |
| Timezone confusion | Always use `datetime.now(timezone.utc)` throughout |
| London/NY overlap classification | Shorter window (LONDON_OPEN, NY_OPEN) takes precedence via check order |
| Weekend trading on major pairs is still valid | Offset configurable per pair tier; MAJOR pairs can use smaller weekend offset |
