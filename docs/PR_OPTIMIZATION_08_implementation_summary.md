# PR_OPTIMIZATION_08 — Implementation Summary

**Date:** 2026-03-27  
**Branch:** `copilot/adaptive-quiet-regime-fix`  
**Status:** ✅ Implemented — all 2360 tests passing

---

## Overview

This document summarises all changes implemented as part of the comprehensive
signal-suppression reduction initiative across PRs 1–5.  Each change is
backward-compatible, fully tested, and env-configurable via the settings
listed in `config/__init__.py`.

---

## PR 1 — Adaptive QUIET Regime

### Problem
All four scalp channels (`360_SCALP`, `360_SCALP_FVG`, `360_SCALP_CVD`,
`360_SCALP_OBI`) were hard-blocked in QUIET regime, suppressing every
mean-reversion scalp signal when BB width was ≤ 1.5 % of price.

### Changes

| File | Change |
|------|--------|
| `src/scanner/__init__.py` | Removed QUIET from `_REGIME_CHANNEL_INCOMPATIBLE` for all SCALP channels except VWAP; added `_SCALP_QUIET_REGIME_PENALTY = 1.8`; QUIET scalp block now enforces `QUIET_SCALP_MIN_CONFIDENCE` gate instead of hard-blocking |
| `src/scanner.py` | Same changes (mirror file kept in sync) |
| `src/regime.py` | Reduced `_BB_WIDTH_QUIET_PCT` from **1.5 → 1.2**, so fewer pairs are classified as QUIET |
| `config/__init__.py` | `QUIET_SCALP_MIN_CONFIDENCE` (default 72.0) and `QUIET_SCALP_VOLUME_MULTIPLIER` (default 2.5) env-configurable |

### Expected Impact
- ~30–50 % more scalp signals generated in QUIET markets
- Only top-tier mean-reversion setups (confidence ≥ 72.0) pass through
- VWAP scalp still hard-blocked in QUIET (volume anchor unreliable)

---

## PR 2 — Dynamic Pair Quality Gates

### Problem
`assess_pair_quality()` used a single 5 % spread threshold for all channels,
blocking valid SWING/SPOT/GEM pairs like KATUSDT that legitimately have
wider spreads.

### Changes

| File | Change |
|------|--------|
| `src/signal_quality.py` | `assess_pair_quality_for_channel()` already exists with per-channel spread limits: SCALP 2.5 %, SWING 5 %, SPOT 6 %, GEM 8 % |
| `src/scanner/__init__.py` | `_should_skip_channel` now calls `assess_pair_quality_for_channel` when the generic gate fails — allowing wider-spread pairs on appropriate channels |
| `src/scanner.py` | Same change (mirror) |

### Per-Channel Spread Limits

| Channel | Spread Limit | Volume Floor |
|---------|-------------|--------------|
| 360_SCALP | 2.5 % | $1 M |
| 360_SCALP_FVG/CVD/OBI/VWAP | 3.0 % | $1 M |
| 360_SWING | 5.0 % | $500 K |
| 360_SPOT | 6.0 % | $500 K |
| 360_GEM | 8.0 % | $500 K |

### Expected Impact
- KATUSDT and similar pairs now pass quality on SWING/SPOT/GEM channels
- +15–20 % more valid signals from altcoin/lower-cap pairs

---

## PR 3 — Graduated OI Validation

### Problem
`check_oi_gate()` hard-rejected all SQUEEZE/DISTRIBUTION signals above the
1 % noise threshold.  Small 1–3 % OI moves (routine market activity) were
causing spurious rejections for STGUSDT, CUSDT etc.

### Changes

| File | Change |
|------|--------|
| `src/oi_filter.py` | Added `OI_SOFT_THRESHOLD = 0.01` and `OI_HARD_THRESHOLD = 0.03`; `check_oi_gate` now implements graduated response |
| `src/order_flow.py` | `is_oi_invalidated()` already uses 1 % threshold for the rolling OI trend check |

### Graduated Response

| OI Change | Action |
|-----------|--------|
| < 1 % | Treat as noise — allow through (existing behaviour) |
| 1 % – 3 % | Soft warning — `allowed=True` with non-empty reason |
| ≥ 3 % AND LOW quality | Hard reject — `allowed=False` (previous default) |

### Expected Impact
- −40 % spurious OI invalidations for STGUSDT, CUSDT
- Only genuinely adversarial OI moves (≥ 3 %) with LOW quality are rejected

---

## PR 4 — Scanning Strategy Optimization

### Problem
- Only 10 concurrent scans allowed (under-utilising the rate limit budget)
- Tier 3 pairs not included in the cycle-based scan schedule
- `TIER3_SCAN_EVERY_N_CYCLES` config variable missing

### Changes

| File | Change |
|------|--------|
| `src/scanner/__init__.py` | `_MAX_CONCURRENT_SCANS` increased from **10 → 15**; Tier 3 included in cycle-based scan schedule every `TIER3_SCAN_EVERY_N_CYCLES` cycles |
| `src/scanner.py` | Same changes (mirror) |
| `config/__init__.py` | Added `TIER3_SCAN_EVERY_N_CYCLES: int` (default 6, env-configurable) |

### Scan Schedule

| Tier | Scan Frequency |
|------|----------------|
| Tier 1 | Every cycle |
| Tier 2 | Every `TIER2_SCAN_EVERY_N_CYCLES` cycles (default 3) |
| Tier 3 | Every `TIER3_SCAN_EVERY_N_CYCLES` cycles (default 6) OR time-based interval |

### Expected Impact
- +50 % concurrent throughput (15 vs 10 simultaneous scans)
- Tier 3 pairs now get cycle-based coverage in addition to time-based scans
- No additional rate-limit risk (paused when >85 % budget used)

---

## PR 5 — Suppressed Signal Telemetry

### Problem
No visibility into suppressed signals — impossible to tune filters without data.

### New File: `src/suppression_telemetry.py`

Implements `SuppressionTracker` with:
- Rolling 4-hour window of suppression events
- Records: `timestamp`, `symbol`, `channel`, `reason`, `regime`, `would_be_confidence`
- Methods: `summary()`, `total_in_window()`, `by_channel()`, `by_symbol()`, `recent_events()`, `format_telegram_digest()`
- Reason constants: `REASON_QUIET_REGIME`, `REASON_SPREAD_GATE`, `REASON_VOLUME_GATE`, `REASON_OI_INVALIDATION`, `REASON_CLUSTER`, `REASON_STAT_FILTER`, `REASON_LIFESPAN`, `REASON_CONFIDENCE`

### Integration Points

| File | Change |
|------|--------|
| `src/scanner/__init__.py` | `self.suppression_tracker = SuppressionTracker()` in `__init__`; records events at: spread/volume gate, regime block, stat filter, confidence gate |
| `src/scanner.py` | Same (mirror) |
| `src/commands/engine.py` | Added `/suppressed` admin command that calls `format_telegram_digest()` |

### Telegram Command

```
/suppressed
```

Returns a 4-hour rolling digest showing:
- Total suppressed count
- Breakdown by reason (quiet regime, spread gate, OI invalidation, etc.)
- Breakdown by channel
- Top 5 most-suppressed pairs

---

## New Tests

| Test File | Tests Added |
|-----------|-------------|
| `tests/test_suppression_telemetry.py` | 26 tests covering `SuppressionTracker` record, prune, summary, by_channel, by_symbol, recent_events, format_telegram_digest |
| `tests/test_advanced_filters.py` | 6 tests for graduated OI validation thresholds (soft window 1–3 %, hard reject ≥ 3 %, threshold constants) |
| `tests/test_regime_context.py` | 2 tests verifying `_BB_WIDTH_QUIET_PCT = 1.2` |

**Total test count:** 2360 (from 2334 baseline)

---

## Configuration Reference

All new settings are in `config/__init__.py` and env-configurable:

| Variable | Default | Description |
|----------|---------|-------------|
| `QUIET_SCALP_MIN_CONFIDENCE` | 72.0 | Minimum confidence for scalp signals in QUIET regime |
| `QUIET_SCALP_VOLUME_MULTIPLIER` | 2.5 | Volume multiplier gate for QUIET scalp signals |
| `TIER2_SCAN_EVERY_N_CYCLES` | 3 | Scan Tier 2 every N cycles |
| `TIER3_SCAN_EVERY_N_CYCLES` | 6 | Scan Tier 3 every N cycles (new) |
| `TIER3_SCAN_INTERVAL_MINUTES` | 30 | Time-based Tier 3 scan interval |

---

## Rollback Guide

Each change is independently revertible:

1. **QUIET scalp regime**: Add `"QUIET"` back to `_REGIME_CHANNEL_INCOMPATIBLE` for scalp channels
2. **OI thresholds**: Remove the `OI_SOFT_THRESHOLD`/`OI_HARD_THRESHOLD` branches in `check_oi_gate`
3. **BB width**: Change `_BB_WIDTH_QUIET_PCT` back to `1.5`
4. **Concurrent scans**: Set `_MAX_CONCURRENT_SCANS = 10`
5. **Tier 3 cycle scan**: Set `TIER3_SCAN_EVERY_N_CYCLES` to a large value (e.g. 9999) to effectively disable
6. **Suppression tracker**: The tracker is additive — removing it requires deleting `suppression_tracker` from Scanner and the `/suppressed` command
