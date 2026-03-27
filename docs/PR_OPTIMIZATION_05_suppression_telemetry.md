# PR-OPT-05 — Suppression Telemetry / Structured Logging

**Priority:** P2  
**Estimated Impact:** Enables data-driven threshold tuning; no direct signal frequency change  
**Dependencies:** PR-OPT-01  
**Status:** ✅ IMPLEMENTED

---

## Objective

Add structured, per-cycle suppression counters to the Scanner so that analysts can identify
which gates are responsible for the most signal suppression and tune thresholds accordingly.

Previously, suppression reasons were only visible as individual DEBUG log lines, making it
impossible to get an aggregate view across hundreds of pairs in a single scan cycle.

---

## Problems Addressed

- No visibility into how many signals are suppressed per reason per cycle.
- DEBUG logs exist but no aggregate counters, so it is hard to quantify the impact of each gate.
- Without telemetry, it is risky to change thresholds without knowing the baseline.

---

## Module / Strategy Affected

- `src/scanner/__init__.py` — `Scanner.__init__`, `_should_skip_channel`, `scan_loop`
- `src/scanner.py` — same

---

## Changes Made

### `Scanner.__init__`

Added `_suppression_counters: Dict[str, int] = defaultdict(int)` to track per-reason counts
within each scan cycle.

### `_should_skip_channel()`

Increments `_suppression_counters` with structured keys before returning `True`:

| Suppression Reason | Counter Key Format |
|--------------------|--------------------|
| Tier 2 excluded from SCALP | `tier2_scalp_excluded:{chan_name}` |
| Pair quality gate failure | `pair_quality:{reason}` |
| Volatile/unsuitable market state | `volatile_unsuitable:{chan_name}` |
| Paused channel | `paused_channel:{chan_name}` |
| Cooldown active | `cooldown:{chan_name}` |
| Per-symbol circuit breaker | `circuit_breaker:{chan_name}` |
| Active signal already exists | `active_signal:{chan_name}` |
| RANGING low ADX | `ranging_low_adx:{chan_name}` |
| Regime incompatibility matrix | `regime:{regime}:{chan_name}` |

### `scan_loop()`

At the end of each scan cycle, if any counters are non-zero, logs a single INFO-level summary
and then clears the counters for the next cycle:

```python
if self._suppression_counters:
    log.info("Scan cycle suppression summary: {}", dict(self._suppression_counters))
    self._suppression_counters.clear()
```

---

## Example Output

```
INFO scanner - Scan cycle suppression summary: {
    'pair_quality:spread too wide': 45,
    'pair_quality:liquidity too thin': 12,
    'cooldown:360_SCALP': 8,
    'regime:QUIET:360_SCALP_VWAP': 23,
    'active_signal:360_SWING': 5,
}
```

This single line allows operators to instantly identify that 45 pairs are failing the spread
gate (possibly requiring threshold relaxation) and 23 VWAP signals are being suppressed by
the QUIET regime hard-block.

---

## Expected Impact

- **Direct**: No change in signal frequency.
- **Indirect**: Enables data-driven threshold tuning that can recover missed signals safely.
- **Operational**: Operators can monitor suppression trends and alert on anomalies.

---

## Rollback Procedure

1. Remove `_suppression_counters` from `Scanner.__init__`.
2. Remove counter increments from `_should_skip_channel()`.
3. Remove the summary log and clear call from `scan_loop()`.
