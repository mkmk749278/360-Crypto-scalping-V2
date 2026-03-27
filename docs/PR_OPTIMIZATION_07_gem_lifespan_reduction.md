# PR-OPT-07 — GEM Channel Minimum Lifespan Reduction

**Priority:** P3  
**Estimated Impact:** Faster adverse-position response; reduced maximum drawdown exposure for GEM signals  
**Dependencies:** None (isolated config + `src/trade_monitor.py` change)

---

## Objective

Reduce the `360_GEM` channel minimum signal lifespan from 86400 seconds (24 hours) to 43200 seconds (12 hours), and add an early-exit override that allows SL evaluation before the lifespan window when confidence drops sharply. Additionally, fix misleading log messages that describe "signals skipped" when the actual behavior is "evaluation protected".

---

## Analysis of Current Code

### `config/__init__.py` — Lines 673–686

```python
MIN_SIGNAL_LIFESPAN_SECONDS: Dict[str, int] = {
    "360_SCALP": 180,
    "360_SWING": 300,
    "360_SPOT": 600,
    "360_GEM": 86400,  # 1 day — macro positions
}
```

The 86400s value was originally set to prevent macro GEM positions from being evaluated for SL/TP during initial market noise. However, this means an adverse move in the first 12–23 hours cannot trigger a protective SL exit regardless of how bad conditions become.

### `src/trade_monitor.py` (lifespan enforcement)

The lifespan check typically appears as:

```python
age_seconds = time.time() - signal.created_at
if age_seconds < MIN_SIGNAL_LIFESPAN_SECONDS.get(signal.channel, 0):
    _log.debug("signal_skipped_lifespan sym=%s age=%.0fs", signal.symbol, age_seconds)
    continue
```

**Problem 1:** The log message says `signal_skipped` but the signal is not being skipped at generation — it already exists and is being **protected from SL evaluation**. This causes confusion in production monitoring.

**Problem 2:** No early-exit path exists even if confidence has collapsed by 40+ points.

---

## Recommended Changes

### Change 1 — Reduce GEM lifespan to 43200s (12 hours)

**File:** `config/__init__.py`

```python
# Before
MIN_SIGNAL_LIFESPAN_SECONDS: Dict[str, int] = {
    "360_SCALP": 180,
    "360_SWING": 300,
    "360_SPOT": 600,
    "360_GEM": 86400,  # 1 day — macro positions
}

# After
MIN_SIGNAL_LIFESPAN_SECONDS: Dict[str, int] = {
    "360_SCALP": 180,
    "360_SWING": 300,
    "360_SPOT": 600,
    "360_GEM": int(os.getenv("MIN_GEM_LIFESPAN_SECONDS", "43200")),  # 12h default
}
```

12 hours is sufficient to protect GEM positions from intraday noise while allowing a protective SL to fire on genuinely adverse multi-hour moves.

### Change 2 — Add `GEM_EARLY_EXIT_CONFIDENCE_DROP` threshold

**File:** `config/__init__.py`

```python
# If GEM signal confidence drops by this many points within the lifespan window,
# allow early SL evaluation regardless of age.
GEM_EARLY_EXIT_CONFIDENCE_DROP: float = float(
    os.getenv("GEM_EARLY_EXIT_CONFIDENCE_DROP", "30.0")
)
```

**File:** `src/trade_monitor.py`

```python
from config import MIN_SIGNAL_LIFESPAN_SECONDS, GEM_EARLY_EXIT_CONFIDENCE_DROP

def _is_lifespan_protected(signal: Signal, current_confidence: float) -> bool:
    """
    Return True if the signal is within its minimum lifespan window AND
    the early-exit confidence-drop threshold has NOT been breached.

    A signal is NO longer protected (returns False) when:
    1. Its age exceeds MIN_SIGNAL_LIFESPAN_SECONDS for its channel, OR
    2. It is a GEM signal and confidence has dropped by >= GEM_EARLY_EXIT_CONFIDENCE_DROP
    """
    age_seconds = time.time() - signal.created_at
    min_lifespan = MIN_SIGNAL_LIFESPAN_SECONDS.get(signal.channel, 0)

    if age_seconds >= min_lifespan:
        return False  # Lifespan window expired — evaluate normally

    # GEM early exit: if confidence has collapsed, override the lifespan protection
    if signal.channel == "360_GEM":
        confidence_drop = signal.initial_confidence - current_confidence
        if confidence_drop >= GEM_EARLY_EXIT_CONFIDENCE_DROP:
            _log.info(
                "gem_early_exit_triggered sym=%s age=%.0fs "
                "initial_conf=%.1f current_conf=%.1f drop=%.1f",
                signal.symbol, age_seconds,
                signal.initial_confidence, current_confidence, confidence_drop,
            )
            return False  # Allow SL evaluation despite being within lifespan window

    return True  # Still protected
```

In the SL/TP evaluation loop:

```python
# Before
age_seconds = time.time() - signal.created_at
if age_seconds < MIN_SIGNAL_LIFESPAN_SECONDS.get(signal.channel, 0):
    _log.debug("signal_skipped_lifespan ...")
    continue

# After
current_confidence = _compute_current_confidence(signal)
if _is_lifespan_protected(signal, current_confidence):
    _log.debug(
        "sl_eval_deferred sym=%s channel=%s age=%.0fs min_lifespan=%d",
        signal.symbol, signal.channel,
        time.time() - signal.created_at,
        MIN_SIGNAL_LIFESPAN_SECONDS.get(signal.channel, 0),
    )
    continue
```

### Change 3 — Fix log messaging

Replace all existing lifespan-related log calls that say `signal_skipped` with the semantically correct `sl_eval_deferred`:

```python
# Before (misleading)
_log.debug("signal_skipped_lifespan sym=%s age=%.0fs min=%.0fs",
           signal.symbol, age_seconds, min_lifespan)

# After (accurate)
_log.debug(
    "sl_eval_deferred sym=%s channel=%s age=%.0fs lifespan_window=%ds",
    signal.symbol, signal.channel, age_seconds, min_lifespan,
)
```

This change prevents the log noise where "skipped" implies the signal wasn't generated, when it was generated and is simply being protected.

### Change 4 — Make lifespan configurable via environment variable

The env var approach (`os.getenv("MIN_GEM_LIFESPAN_SECONDS", "43200")`) in Change 1 means operators can override the lifespan without a code deploy:

```bash
# .env — reduce further for testing or aggressive drawdown management
MIN_GEM_LIFESPAN_SECONDS=21600     # 6 hours
GEM_EARLY_EXIT_CONFIDENCE_DROP=25  # Trigger on 25-point confidence drop
```

Document in `.env.example`:

```bash
# GEM channel minimum lifespan before SL evaluation (default: 43200 = 12h)
# Set lower for aggressive drawdown management; higher for macro noise tolerance
MIN_GEM_LIFESPAN_SECONDS=43200

# Confidence drop (points) that overrides lifespan protection for early SL exit
GEM_EARLY_EXIT_CONFIDENCE_DROP=30.0
```

---

## Modules Affected

| Module | Change |
|--------|--------|
| `config/__init__.py` | Reduce GEM lifespan to 43200; add `GEM_EARLY_EXIT_CONFIDENCE_DROP`; env var |
| `src/trade_monitor.py` | Add `_is_lifespan_protected()`; fix log messages |
| `.env.example` | Document new env vars |

---

## Test Cases

1. **`test_gem_lifespan_default_43200`** — `MIN_SIGNAL_LIFESPAN_SECONDS["360_GEM"]` == 43200 by default.
2. **`test_gem_env_override`** — Setting `MIN_GEM_LIFESPAN_SECONDS=21600` overrides to 21600.
3. **`test_gem_protected_within_window`** — GEM signal aged 6h with 10-point drop remains protected.
4. **`test_gem_not_protected_after_window`** — GEM signal aged 13h is not protected (window expired).
5. **`test_gem_early_exit_on_confidence_drop`** — GEM signal aged 6h with 35-point confidence drop is NOT protected (early exit triggered).
6. **`test_gem_early_exit_not_triggered_below_threshold`** — 25-point drop (below 30 threshold) does not trigger early exit.
7. **`test_scalp_lifespan_unchanged`** — `360_SCALP` still has 180s lifespan.
8. **`test_log_message_deferred_not_skipped`** — Log output uses `sl_eval_deferred`, not `signal_skipped`.
9. **`test_non_gem_no_early_exit`** — `360_SWING` signal with 35-point drop does not trigger early exit logic.

---

## Rollback Procedure

1. Restore `"360_GEM": 86400` in `MIN_SIGNAL_LIFESPAN_SECONDS`.
2. Remove `GEM_EARLY_EXIT_CONFIDENCE_DROP` from config.
3. Restore original lifespan check in `trade_monitor.py` (remove `_is_lifespan_protected`).
4. Log message changes can be left in place (semantically correct).

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 12h window too short — GEM positions hit SL on 12–24h noise | Medium | `GEM_EARLY_EXIT_CONFIDENCE_DROP` threshold provides a secondary guard; operators can set `MIN_GEM_LIFESPAN_SECONDS=86400` via env var to revert instantly |
| Early-exit confidence drop threshold triggers prematurely | Medium | Default 30-point threshold is conservative; tune via `GEM_EARLY_EXIT_CONFIDENCE_DROP` env var |
| `signal.initial_confidence` not stored on Signal object | Low | Verify `Signal` dataclass has `initial_confidence` field; add if missing |
| Log message change breaks log monitoring queries | Low | Update any Grafana/grep dashboards that query `signal_skipped_lifespan` |

---

## Expected Impact

- **Reduced maximum drawdown exposure** for GEM positions — adverse moves can trigger SL after 12h instead of 24h
- **Faster SL response** on confidence-collapsing GEM trades (early-exit path)
- **Cleaner log output** — `sl_eval_deferred` accurately describes what is happening
- **Zero signal generation impact** — this change only affects SL/TP evaluation timing
