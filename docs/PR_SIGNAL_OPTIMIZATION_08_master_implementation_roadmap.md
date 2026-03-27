# PR-SIG-OPT-08 — Master Implementation Roadmap

**Priority:** Reference Document — Not a code PR  
**Purpose:** Dependency graph, risk assessment, rollback procedures, and aggregate impact summary for PRs 01–07  
**Status:** 📋 Reference

---

## Overview

This document provides the master implementation guide for the 7 signal optimization
PRs (`PR_SIGNAL_OPTIMIZATION_01` through `_07`). It defines the correct deployment
order, identifies dependencies between PRs, assesses risk, and provides a testing
checklist for the full optimization set.

These PRs extend and complement the earlier optimization series (`PR_OPTIMIZATION_01`
through `_08`). The earlier series addressed: QUIET regime adaptive soft-gate
(`PR_OPT_01`), dynamic pair quality gates (`PR_OPT_02`), OI graduated validation
(`PR_OPT_03`), scanning strategy optimization (`PR_OPT_04`), suppression telemetry
(`PR_OPT_05`), lifespan channel-awareness (`PR_OPT_06`), and GEM lifespan reduction
(`PR_OPT_07`). This new series goes further by making thresholds fully env-configurable,
tier-adaptive, and regime-aware.

---

## Dependency Graph

```
PR-SIG-OPT-01 (Regime Soft-Gate)
    │
    ├── PR-SIG-OPT-06 (Adaptive Tier Thresholds)  ← depends on 01 for soft-gate system
    │
    └── [signals downstream] ────────────────────────────────────────────────────┐
                                                                                   │
PR-SIG-OPT-02 (Pair Quality Adaptive)   ── INDEPENDENT ──────────────────────────┤
                                                                                   │
PR-SIG-OPT-03 (Per-Channel Lifespan)    ── INDEPENDENT ──────────────────────────┤
                                                                                   │
PR-SIG-OPT-04 (Tiered Scan Scheduler)                                             │
    │                                                                              │
    └── PR-SIG-OPT-05 (WS Pool Optimization)  ← deploy together with 04          │
                                                                                   │
PR-SIG-OPT-07 (Suppression Analytics)   ── INDEPENDENT ──────────────────────────┘
    (Benefits from 01-06 being deployed first — more data to analyze)
```

### Deployment Order

**Phase 1 (Parallel — no dependencies between them):**
- PR-SIG-OPT-02 — Pair Quality Channel-Adaptive
- PR-SIG-OPT-03 — Per-Channel Lifespan
- PR-SIG-OPT-07 — Suppression Analytics

**Phase 2 (After Phase 1):**
- PR-SIG-OPT-01 — Regime Soft-Gate Overhaul
- PR-SIG-OPT-04 — Tiered Scan Scheduler
- PR-SIG-OPT-05 — WS Pool Optimization (deploy with 04)

**Phase 3 (After Phase 2):**
- PR-SIG-OPT-06 — Adaptive Tier Thresholds (depends on 01's soft-gate infrastructure)

---

## Risk Assessment

| PR | Impact | Risk Level | Risk Description |
|----|--------|------------|-----------------|
| SIG-OPT-01 | HIGH | 🟡 MEDIUM | Lowering confidence floor may pass more noise. Mitigated by soft-gate tags. |
| SIG-OPT-02 | HIGH | 🟢 LOW | Channel-specific thresholds only affect non-SCALP channels. SCALP unchanged. |
| SIG-OPT-03 | HIGH | 🟢 LOW | Fixes a missing config key (`360_SCALP` not in `LIFECYCLE_CHECK_INTERVAL`). |
| SIG-OPT-04 | MEDIUM | 🟡 MEDIUM | New scheduler class — regression risk for scan ordering. Isolated in new code path. |
| SIG-OPT-05 | MEDIUM | 🟢 LOW | Reduces bulk-seed limit. Mitigated by keeping `WS_FALLBACK_BULK_LIMIT` shim. |
| SIG-OPT-06 | HIGH | 🟡 MEDIUM | Module-level global patching in `AdaptiveRegimeDetector.classify()` — thread safety risk if scanner is multi-threaded. |
| SIG-OPT-07 | LOW | 🟢 LOW | Additive analytics layer — does not change signal generation. |

### SIG-OPT-06 Thread Safety Note

The `AdaptiveRegimeDetector.classify()` method in PR-SIG-OPT-06 patches module-level
globals (`_ADX_TRENDING_MIN`, etc.) in a try/finally block. This is safe in an
`asyncio` single-threaded event loop (the scanner uses `asyncio`, not threads) because
only one coroutine runs at a time when a `classify()` call is in progress.

However, if the scanner is refactored to use `asyncio.gather()` with concurrent
`_scan_symbol()` calls (as proposed in PR-SIG-OPT-04), two coroutines could
simultaneously be inside `classify()`, patching globals for different tiers.

**Mitigation:** Add a `_decide(adx_val, bb_width_pct, ema_slope, thresholds)` helper to
`MarketRegimeDetector` that accepts threshold parameters explicitly, then override it in
`AdaptiveRegimeDetector` with tier-specific values. This avoids global patching entirely
and is safe under concurrent coroutines:

```python
# In MarketRegimeDetector (base class) — extract threshold-parametric helper:
def _decide(
    self,
    adx_val: Optional[float],
    bb_width_pct: Optional[float],
    ema_slope: Optional[float],
    adx_trending_min: float = _ADX_TRENDING_MIN,
    adx_ranging_max: float = _ADX_RANGING_MAX,
    bb_width_quiet: float = _BB_WIDTH_QUIET_PCT,
    bb_width_volatile: float = _BB_WIDTH_VOLATILE_PCT,
) -> MarketRegime:
    # ... existing decision logic using the explicit threshold params ...

# In AdaptiveRegimeDetector — call _decide() with tier-specific thresholds:
def classify(self, indicators, candles=None, timeframe="5m", volume_delta=None):
    # ... compute adx_val, bb_width_pct, ema_slope ...
    regime = self._decide(
        adx_val, bb_width_pct, ema_slope,
        adx_trending_min=self._adx_trending_min,
        adx_ranging_max=self._adx_ranging_max,
        bb_width_quiet=self._bb_width_quiet,
        bb_width_volatile=self._bb_width_volatile,
    )
    # ... apply hysteresis and return RegimeResult ...
```

This design is fully thread-safe (no shared mutable state), testable in isolation, and
doesn't require `src.regime` module-level global patching.

---

## Rollback Procedures

### PR-SIG-OPT-01 (Regime Soft-Gate)
**Rollback:** Set env vars to revert to hard-blocking behavior:
```bash
RANGING_ADX_SUPPRESS_THRESHOLD=15.0    # Restores original threshold
QUIET_SCALP_MIN_CONFIDENCE=72.0         # Restores original floor
REGIME_RANGING_PENALTY=100.0            # Effectively hard-blocks (100pt penalty = fail any gate)
```

### PR-SIG-OPT-02 (Pair Quality)
**Rollback:** Set all thresholds to universal 58:
```bash
PAIR_QUALITY_THRESHOLD_SCALP=58.0
PAIR_QUALITY_THRESHOLD_SWING=58.0
PAIR_QUALITY_THRESHOLD_SPOT=58.0
PAIR_QUALITY_THRESHOLD_GEM=58.0
PAIR_QUALITY_VOLUME_FLOOR_SWING=500000
PAIR_QUALITY_VOLUME_FLOOR_SPOT=500000
PAIR_QUALITY_VOLUME_FLOOR_GEM=500000
```

### PR-SIG-OPT-03 (Lifespan)
**Rollback:** Add `360_SCALP` back to `LIFECYCLE_CHECK_INTERVAL` with 21600:
```bash
LIFECYCLE_CHECK_INTERVAL_SCALP=21600
```

### PR-SIG-OPT-04 (Tiered Scheduler)
**Rollback:** The `TieredScanScheduler` is only used if instantiated. Disable by
keeping the old counter-based path and not calling `_tiered_scheduler.populate()`.
Feature flag: `ENABLE_TIERED_SCHEDULER=false` in `.env`.

### PR-SIG-OPT-05 (WS Pool)
**Rollback:** Restore flat bulk limit:
```bash
WS_FALLBACK_BULK_LIMIT=200
WS_FALLBACK_BULK_1M=200
WS_FALLBACK_BULK_5M=200
WS_FALLBACK_BULK_4H=200
WS_CRITICAL_PAIR_COUNT=10
```

### PR-SIG-OPT-06 (Adaptive Thresholds)
**Rollback:** Disable `AdaptiveRegimeDetector` by reverting scanner to use
`MarketRegimeDetector` directly. Feature flag: `ENABLE_ADAPTIVE_REGIME=false`.

### PR-SIG-OPT-07 (Analytics)
**Rollback:** `SuppressionAnalytics` is additive — removing the instantiation
from `Scanner.__init__()` fully reverts. No functional change to signal generation.

---

## Testing Checklist

Run after each PR deployment:

```bash
# Full test suite
python -m pytest tests/ -v --tb=short 2>&1 | tail -50

# Targeted tests per PR
python -m pytest tests/test_regime_soft_penalty.py -v          # PR-01
python -m pytest tests/test_signal_quality.py -v               # PR-02
python -m pytest tests/test_signal_quality_improvements.py -v  # PR-02
python -m pytest tests/test_engine_lifecycle.py -v             # PR-03
python -m pytest tests/test_signal_execution_timing.py -v      # PR-03
python -m pytest tests/test_tiered_pairs.py -v                 # PR-04
python -m pytest tests/test_tier_manager.py -v                 # PR-04
python -m pytest tests/test_websocket_and_formatting.py -v     # PR-05
python -m pytest tests/test_regime_filters.py -v               # PR-06
python -m pytest tests/test_regime_filter_propagation.py -v    # PR-06
python -m pytest tests/test_suppression_telemetry.py -v        # PR-07
```

### 24-Hour Post-Deployment Monitoring

After each PR (especially 01, 02, 06):

1. **Monitor suppression rates** via `/suppression_report` command
   - Expected: total suppression rate decreases by 20–40% per PR
   - Flag: if any channel >95% suppression, investigate immediately

2. **Compare signal frequency** using `/stats` Telegram command
   - Expected: SCALP +30-40% (PRs 01+03), SWING/SPOT/GEM +15-25% (PR 02)

3. **Watch for noise signals** — signals with `QUIET_ADJUSTED` or `RANGING_ADJUSTED` tags
   - Monitor their hit rate in performance tracker
   - If hit rate <50% for adjusted signals, increase penalty values

4. **Rate limit monitoring**
   - Check Binance REST API usage in `rate_limiter.py` metrics
   - Expected: headroom improves after PR-05

5. **Scan latency**
   - Check scan cycle duration in logs: `"Scan cycle complete: hot=..."` INFO lines
   - Expected: hot-queue latency <15s after PR-04

---

## Configuration Migration Guide

### New Environment Variables (all PRs combined)

Add to `.env` and `.env.example`:

```bash
# PR-SIG-OPT-01: Regime Soft-Gate
RANGING_ADX_SUPPRESS_THRESHOLD=12.0     # Was hard-coded 15.0
REGIME_RANGING_PENALTY=5.0              # Confidence penalty for ranging soft-gate
REGIME_QUIET_PENALTY=8.0                # Confidence penalty for quiet soft-gate
QUIET_SCALP_MIN_CONFIDENCE=68.0         # Was 72.0
BB_WIDTH_QUIET_PCT=1.0                  # Was 1.2 (regime.py)
ADX_RANGING_MAX=18.0                    # Was 20.0 (regime.py)

# PR-SIG-OPT-02: Pair Quality
PAIR_QUALITY_THRESHOLD_SCALP=58.0
PAIR_QUALITY_THRESHOLD_SWING=50.0
PAIR_QUALITY_THRESHOLD_SPOT=45.0
PAIR_QUALITY_THRESHOLD_GEM=40.0
PAIR_QUALITY_VOLUME_FLOOR_SWING=500000
PAIR_QUALITY_VOLUME_FLOOR_SPOT=250000
PAIR_QUALITY_VOLUME_FLOOR_GEM=100000

# PR-SIG-OPT-03: Per-Channel Lifespan
LIFECYCLE_CHECK_INTERVAL_SCALP=900      # 15 min (was missing → defaulted to 21600!)
LIFECYCLE_CHECK_INTERVAL_SCALP_FVG=900
LIFECYCLE_CHECK_INTERVAL_SCALP_CVD=900
LIFECYCLE_CHECK_INTERVAL_SCALP_VWAP=900
LIFECYCLE_CHECK_INTERVAL_SCALP_OBI=900
LIFECYCLE_CHECK_INTERVAL_SWING=7200     # Was 14400 (4h), now 7200 (2h)
SCALP_MIN_LIFESPAN=900
SWING_MIN_LIFESPAN=7200
SPOT_MIN_LIFESPAN=21600
GEM_MIN_LIFESPAN=43200

# PR-SIG-OPT-04: Tiered Scan Scheduler
SCAN_TIER1_INTERVAL=30                  # Hot: futures top-50 (seconds)
SCAN_TIER2_INTERVAL=60                  # Warm: top-100 spot+futures
SCAN_TIER3_INTERVAL=180                 # Cold: remaining pairs
SCAN_HOT_CONCURRENCY=10
SCAN_COLD_CONCURRENCY=5
ENABLE_TIERED_SCHEDULER=true

# PR-SIG-OPT-05: WS Pool
WS_FALLBACK_BULK_1M=50                  # Was 200 for all TFs
WS_FALLBACK_BULK_5M=50
WS_FALLBACK_BULK_15M=50
WS_FALLBACK_BULK_1H=100
WS_FALLBACK_BULK_4H=100
WS_CRITICAL_PAIR_COUNT=20               # Was ~10
WS_RECONNECT_STAGGER_MS=500

# PR-SIG-OPT-06: Adaptive Regime Thresholds
ENABLE_ADAPTIVE_REGIME=true
MAJOR_ADX_TRENDING_MIN=28.0
MAJOR_ADX_RANGING_MAX=22.0
MIDCAP_ADX_TRENDING_MIN=25.0
MIDCAP_ADX_RANGING_MAX=20.0
ALTCOIN_ADX_TRENDING_MIN=20.0
ALTCOIN_ADX_RANGING_MAX=15.0
ALTCOIN_BB_WIDTH_QUIET=2.0

# PR-SIG-OPT-07: Suppression Analytics
SUPPRESSION_ANALYTICS_PATH=data/suppression_analytics.json
SUPPRESSION_ALERT_RATE_THRESHOLD=0.90
```

---

## Relationship to Prior Optimization Series

| Prior PR | New PR | Relationship |
|----------|--------|--------------|
| PR-OPT-01 (QUIET soft-gate) | SIG-OPT-01 | Extends: adds RANGING soft-gate, env-configurable thresholds, confidence floor reduction |
| PR-OPT-02 (Pair quality adaptive) | SIG-OPT-02 | Extends: adds per-channel composite score thresholds, regime-aware spread relaxation |
| PR-OPT-03 (OI graduated validation) | — | Unrelated — OI validation is separate |
| PR-OPT-04 (Scanning optimization) | SIG-OPT-04 | Extends: adds `TieredScanScheduler` class with wall-clock timing |
| PR-OPT-05 (Suppression telemetry) | SIG-OPT-07 | Extends: adds `SuppressionAnalytics`, multi-window, persistence, auto-alert, new commands |
| PR-OPT-06 (Lifespan channel-aware) | SIG-OPT-03 | Fixes: adds missing `360_SCALP` entries, makes all values env-configurable |
| PR-OPT-07 (GEM lifespan) | SIG-OPT-03 | Superseded: SIG-OPT-03 covers all channels including GEM |
| PR-OPT-08 (Master plan) | SIG-OPT-08 (this doc) | Parallel: separate implementation series |

---

## Expected Aggregate Impact

After all 7 PRs are deployed:

| Channel | Expected Signal Frequency Change | Quality Maintained |
|---------|----------------------------------|-------------------|
| 360_SCALP | +35–45% | Yes (confidence ≥68, tagged) |
| 360_SCALP_FVG | +20–30% | Yes |
| 360_SCALP_CVD | +20–30% | Yes |
| 360_SCALP_VWAP | +5–10% (VWAP still blocked in QUIET) | Yes |
| 360_SCALP_OBI | +20–30% | Yes |
| 360_SWING | +20–30% | Yes |
| 360_SPOT | +25–35% | Yes |
| 360_GEM | +40–60% | Yes (lower threshold by design) |

**Overall:** +40–60% signal frequency across all channels while maintaining signal
confidence above 65 (SCALP: ≥68, SWING: ≥65, SPOT/GEM: channel defaults).

**Infrastructure Improvements:**
- REST fallback activation: -60% frequency
- Scan latency for top-50 futures: 45s → 15s
- Rate limit headroom: 33% → 63% of budget
- Suppression visibility: zero → full Telegram dashboard

---

## Implementation Timeline

| Week | PRs | Focus |
|------|-----|-------|
| Week 1 | SIG-OPT-02, SIG-OPT-03, SIG-OPT-07 | Low-risk fixes; suppression visibility |
| Week 2 (early) | SIG-OPT-01 | Regime soft-gate; monitor for 24h before continuing |
| Week 2 (late) | SIG-OPT-04, SIG-OPT-05 | Scanning and WS (deploy together, after SIG-OPT-01 is stable) |
| Week 3 | SIG-OPT-06 | Adaptive regime thresholds (requires SIG-OPT-01 soft-gate infrastructure) |
| Week 4 | Tuning | Adjust env vars based on 24h analytics data from SIG-OPT-07 |
