# PR-OPT-MASTER — Signal Suppression Optimization Master Plan

**Document Version:** 2.0  
**Created:** 2026-03-27  
**Scope:** Adaptive QUIET regime, per-channel pair quality gates, graduated OI validation,
scanning strategy optimization, suppression telemetry, and channel-aware lifespan thresholds

---

## Executive Summary

Production system logs and scanner behaviour analysis revealed that the 360-Crypto-Scalping-V2
system is **over-filtering valid signals** due to overly conservative static thresholds. The
root causes and their approximate contributions to signal suppression are:

| # | Suppression Source | Approx. Share | Root Cause |
|---|-------------------|--------------|------------|
| 1 | QUIET regime hard block | ~46% | `_REGIME_CHANNEL_INCOMPATIBLE` blocks ALL scalp channels |
| 2 | Pair quality gate failures | ~23% | Fixed 3% spread limit applied uniformly across all channels |
| 3 | OI rising invalidation | ~13% | Any OI rise ≥ 0.5% triggers hard reject |
| 4 | Signal lifespan filter | ~5% | 24-hour GEM lifespan delays SL/TP evaluation |
| 5 | Other gates (cluster, stat) | ~13% | Various soft gates |

The optimizations documented across PR-OPT-01 through PR-OPT-06 address the top four
suppression causes. Together they are expected to **increase valid signal generation by
40–60%** without degrading signal quality.

---

## PR Execution Order (Sequential)

PRs are ordered by risk-adjusted impact. Deploy each PR, monitor for 24h, validate metrics,
then proceed to the next.

```
PR-OPT-05 (Telemetry)           ←── Deploy FIRST (establishes baseline)
PR-OPT-01 (Adaptive QUIET)      ←── Highest impact, lowest risk
PR-OPT-02 (Pair Quality Gates)  ←── Unblocks SWING/SPOT signals
PR-OPT-03 (Graduated OI)        ←── Reduces false OI rejections
PR-OPT-06 (Lifespan)            ←── Unblocks scalp/GEM lifespans
PR-OPT-04 (Scanning Strategy)   ←── Infrastructure change (lowest risk last)
```

| PR | Title | Priority | File | Status |
|----|-------|----------|------|--------|
| PR-OPT-01 | Adaptive QUIET Regime for Scalp Channels | P0 | `PR_OPTIMIZATION_01_adaptive_quiet_regime.md` | ✅ IMPLEMENTED |
| PR-OPT-02 | Per-Channel Pair Quality Gates | P1 | `PR_OPTIMIZATION_02_pair_quality_adaptive.md` | ✅ IMPLEMENTED |
| PR-OPT-03 | Graduated OI Validation | P1 | `PR_OPTIMIZATION_03_oi_graduated_validation.md` | ✅ IMPLEMENTED |
| PR-OPT-04 | Scanning Strategy Optimization | P2 | `PR_OPTIMIZATION_04_scanning_strategy.md` | 📋 PLANNED |
| PR-OPT-05 | Suppression Telemetry | P2 | `PR_OPTIMIZATION_05_suppression_telemetry.md` | ✅ IMPLEMENTED |
| PR-OPT-06 | Channel-Aware Signal Lifespan | P3 | `PR_OPTIMIZATION_06_lifespan_channel_aware.md` | ✅ IMPLEMENTED |

---

## 1. PR-OPT-01 — Adaptive QUIET Regime

**File:** `docs/PR_OPTIMIZATION_01_adaptive_quiet_regime.md`  
**Modules:** `src/scanner/__init__.py` (line 157), `src/scanner.py`, `config/__init__.py`

**Problem:** `_REGIME_CHANNEL_INCOMPATIBLE` hard-blocked `360_SCALP`, `360_SCALP_FVG`,
`360_SCALP_CVD`, `360_SCALP_OBI` when regime is QUIET. The `ScalpChannel` internally boosts
mean-reversion weight to 1.5× in QUIET (via regime-aware weights), but these signals were
killed before evaluation.

**Solution Implemented:**
- Removed QUIET from the incompatible list for all scalp channels except `360_SCALP_VWAP`
- Added `_SCALP_QUIET_REGIME_PENALTY = 1.8` multiplier (vs. default 0.8) for scalp channels in QUIET
- Added `QUIET_SCALP_MIN_CONFIDENCE = 72.0` minimum confidence gate (`config/__init__.py` line ~687)
- VWAP remains blocked in QUIET — VWAP signals require sufficient volume to be valid

**Expected Impact:** ~40–60% reduction in QUIET regime suppressions; 15–25% increase in
overall scalp signal frequency.

---

## 2. PR-OPT-02 — Per-Channel Pair Quality Gates

**File:** `docs/PR_OPTIMIZATION_02_pair_quality_adaptive.md`  
**Modules:** `src/signal_quality.py` (lines 319–415), `src/scanner/__init__.py`

**Problem:** `assess_pair_quality()` at line 270 used `spread_pct <= 0.03` and
`volume_24h >= 1_000_000` as hard gates for ALL channels. Pairs like KATUSDT failed with
"spread too wide" across ALL strategies.

**Solution Implemented:**
- Added `_SPREAD_LIMIT_BY_CHANNEL` dict at line 322 with per-channel limits
  (SCALP: 2.5%, SWING: 5%, SPOT: 6%, GEM: 8%)
- Added `_MIN_VOLUME_NON_SCALP = 500_000` at line 335
- Added `assess_pair_quality_for_channel()` function at line 338

**Expected Impact:** ~15–25% more pairs pass quality gate for SWING/SPOT/GEM channels.

---

## 3. PR-OPT-03 — Graduated OI Validation

**File:** `docs/PR_OPTIMIZATION_03_oi_graduated_validation.md`  
**Modules:** `src/order_flow.py` (line 162), `src/oi_filter.py` (line 65, 210)

**Problem:** `is_oi_invalidated()` returned `True` for ANY rising OI ≥ 0.5%. A 0.6% OI
rise on Binance perpetuals is noise (routine hedging/rebalancing), not meaningful new
positioning. This caused false invalidations for STGUSDT, CUSDT, and similar pairs.

**Solution Implemented:**
- Added `oi_change_pct: float = 0.0` parameter to `is_oi_invalidated()` at line 162
- Added noise floor: only invalidate when `abs(oi_change_pct) >= 0.01` (1%)
- Added `OI_NOISE_THRESHOLD = 0.01` to `src/oi_filter.py` at line 65
- `check_oi_gate()` now soft-passes OI moves below the noise threshold with debug log

**Expected Impact:** ~30–40% fewer false OI rejections for STGUSDT, CUSDT, and similar
perpetual pairs where OI fluctuates naturally.

---

## 4. PR-OPT-04 — Scanning Strategy Optimization

**File:** `docs/PR_OPTIMIZATION_04_scanning_strategy.md`  
**Modules:** `src/scanner/__init__.py`, `src/scanner.py`, `src/pair_manager.py`,
`src/rate_limiter.py`, `src/websocket_manager.py`, `config/__init__.py`

**Problem:** Sequential scan loop drops Tier 2 entirely at 85% rate limit. Scalp signals
on Tier 1 futures are delayed 60–90s due to shared budget with lower-priority scans.

**Solution Planned:**
- Refactor `scan_loop()` into three independent async coroutines per tier
- Allocate rate budget: Tier 1 = 60%, Tier 2 = 30%, Tier 3 = 10%
- Use aggregate endpoints: `/fapi/v1/ticker/24hr` (weight 40) vs. per-symbol (weight 1 each)
- Add WebSocket combined streams for Tier 1: `!miniTicker@arr`
- New config vars: `TIER1_SCAN_INTERVAL_SECONDS`, `TIER2_SCAN_INTERVAL_SECONDS`

**Expected Impact:** Tier 1 scalp signals within 15–30s; no complete Tier 2 stalls.

---

## 5. PR-OPT-05 — Suppression Telemetry

**File:** `docs/PR_OPTIMIZATION_05_suppression_telemetry.md`  
**Modules:** `src/scanner/__init__.py` (line 437–683)

**Problem:** Suppressed signals were only logged at DEBUG level, making it impossible to
analyze suppression patterns in production.

**Solution Implemented:**
- Added `self._suppression_counters: Dict[str, int]` at line 439 in `Scanner.__init__`
- Suppression increments at all skip points in `_should_skip_channel()` (line 1114):
  - `tier2_scalp_excluded:{chan_name}` — Tier 2 scalp exclusion
  - `pair_quality:{reason}` — quality gate failure with reason
  - `volatile_unsuitable:{chan_name}` — volatile regime block
  - `paused_channel:{chan_name}` — channel paused by operator
  - `cooldown:{chan_name}` — post-signal cooldown
  - `circuit_breaker:{chan_name}` — circuit breaker active
  - `active_signal:{chan_name}` — signal already active for pair
  - `ranging_low_adx:{chan_name}` — low ADX ranging rejection
  - `regime:{current_regime}:{chan_name}` — regime incompatibility
- Per-cycle INFO log at line 677 showing aggregate suppression counts
- Counters reset after each cycle to provide per-cycle visibility

**Expected Impact:** Full visibility into suppression patterns per scan cycle.

---

## 6. PR-OPT-06 — Channel-Aware Signal Lifespan

**File:** `docs/PR_OPTIMIZATION_06_lifespan_channel_aware.md`  
**Modules:** `config/__init__.py` (line 673), `src/trade_monitor.py` (line 547)

**Problem:** `MIN_SIGNAL_LIFESPAN_SECONDS["360_GEM"] = 86400` meant SL/TP evaluation was
skipped for 24 hours after any GEM signal. For volatile small-cap tokens, this is dangerous.

**Solution Implemented:**
- Reduced GEM lifespan from `86400` (24h) to `21600` (6h) in `config/__init__.py`
- Per-channel thresholds preserved: SCALP=180s, SWING=300s, SPOT=600s, GEM=21600s

**Expected Impact:** GEM signals receive SL/TP evaluation 4× more frequently.

---

## Risk Assessment

| PR | Risk Level | What Could Go Wrong | Rollback |
|----|-----------|---------------------|---------|
| PR-OPT-01 | Medium | QUIET regime signals with lower confidence may fire | Set `QUIET_SCALP_MIN_CONFIDENCE=85` to tighten |
| PR-OPT-02 | Low | Wider spread channels may trade in illiquid markets | Revert `_SPREAD_LIMIT_BY_CHANNEL` values |
| PR-OPT-03 | Medium | Legitimate OI-driven reversals may not be filtered | Set `OI_NOISE_THRESHOLD=0.02` to raise noise floor |
| PR-OPT-04 | High | Independent scan loops may cause rate limit spikes | Revert to single `scan_loop()` with env vars |
| PR-OPT-05 | None | Telemetry only — no signal logic changes | N/A |
| PR-OPT-06 | Low | Faster GEM SL/TP eval may close positions early | Increase `MIN_SIGNAL_LIFESPAN_SECONDS["360_GEM"]` |

---

## Expected Aggregate Impact

| Metric | Before | After |
|--------|--------|-------|
| Valid signal frequency | Baseline | +40–60% |
| QUIET regime suppressions | ~46% of total | ~10–15% of total |
| Pair quality failures | ~23% of total | ~8–12% of total |
| OI false rejections | ~13% of total | ~5–7% of total |
| Tier 1 signal latency | 60–90 seconds | 15–30 seconds (post PR-OPT-04) |
| GEM SL/TP evaluation gap | 24 hours | 6 hours |

---

## Step-by-Step Deployment Plan

### Phase 1 — Observability First (Day 1–2)

1. Deploy PR-OPT-05 (suppression telemetry — already in `src/scanner/__init__.py`)
2. Run 4h scan cycle and collect baseline suppression summary from INFO logs
3. Record counts by gate (regime, pair_quality, OI, etc.) as the "before" baseline

### Phase 2 — Highest Impact (Day 3–5)

4. Verify PR-OPT-01 changes in `src/scanner/__init__.py` (QUIET regime soft penalty)
5. Verify `QUIET_SCALP_MIN_CONFIDENCE = 72.0` in `config/__init__.py`
6. Run 4h cycle; compare suppression summary to Phase 1 baseline
7. Expected: `regime:QUIET:*` counts drop by 60–80%

### Phase 3 — Quality Gates + OI (Week 2)

8. Verify PR-OPT-02: `assess_pair_quality_for_channel()` in `src/signal_quality.py`
9. Verify PR-OPT-03: `is_oi_invalidated()` noise floor in `src/order_flow.py`
10. Deploy PR-OPT-06: confirm GEM lifespan = 21600 in `config/__init__.py`
11. Monitor for 48h; watch for quality regression in executed signals

### Phase 4 — Infrastructure (Week 3)

12. Implement PR-OPT-04 scanning strategy refactor (see `PR_OPTIMIZATION_04_scanning_strategy.md`)
13. Deploy to staging first — validate rate limit behaviour under load
14. Roll out to production with feature flag; monitor for 24h before removing old code path

---

## Monitoring Checklist

After each PR deployment, validate the following:

- [ ] Suppression summary shows expected reduction in the targeted gate
- [ ] Signal frequency metric in `src/telemetry.py` shows ≥ expected increase
- [ ] No increase in `false_positive_rate` from backtester feedback
- [ ] No OI-invalidation-related position reversals in `trade_monitor.py` logs
- [ ] Rate limit usage stays below 75% average (critical for PR-OPT-04)
- [ ] GEM signal SL/TP evaluations appearing in logs within 6h of signal creation

---

## File Reference Map

| Optimization | Primary File | Line Numbers |
|-------------|-------------|-------------|
| QUIET regime block | `src/scanner/__init__.py` | 151–157 (compat matrix), 1567–1570 (penalty), 2040–2046 (min confidence) |
| Per-channel spread limits | `src/signal_quality.py` | 322–333 (`_SPREAD_LIMIT_BY_CHANNEL`), 338–415 (`assess_pair_quality_for_channel`) |
| OI noise floor | `src/order_flow.py` | 162–197 (`is_oi_invalidated`) |
| OI gate noise bypass | `src/oi_filter.py` | 65 (`OI_NOISE_THRESHOLD`), 210–268 (`check_oi_gate`) |
| GEM lifespan | `config/__init__.py` | 673–680 (`MIN_SIGNAL_LIFESPAN_SECONDS`) |
| QUIET min confidence | `config/__init__.py` | 687–692 (`QUIET_SCALP_MIN_CONFIDENCE`) |
| Suppression counters | `src/scanner/__init__.py` | 437–439 (init), 677–683 (summary log), 1114–1186 (increments) |
