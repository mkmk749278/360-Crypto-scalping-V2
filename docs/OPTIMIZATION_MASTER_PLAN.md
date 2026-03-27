# OPTIMIZATION MASTER PLAN — 360-Crypto-Scalping-V2

> **Document Version:** 1.0  
> **Created:** 2026-03-27  
> **Scope:** Signal suppression recovery, pair quality improvements, scanning efficiency, and observability

---

## Executive Summary

This master plan coordinates 7 sequential optimization PRs addressing five root-cause signal suppression issues identified in production:

| # | Issue | Root Cause | PR |
|---|-------|------------|-----|
| 1 | Signals suppressed in QUIET regime | Hard block in `_REGIME_CHANNEL_INCOMPATIBLE` | PR-OPT-01 |
| 2 | Pair quality gate failures (spread) | Fixed 3 bps threshold across all channels | PR-OPT-02 |
| 3 | OI invalidation too aggressive | Binary reject at 0.5% OI change | PR-OPT-03 |
| 4 | Rate limit exposure / scan gaps | Per-symbol REST pattern, no aggregate usage | PR-OPT-04 |
| 5 | No visibility into suppression losses | All suppression events logged at DEBUG only | PR-OPT-05 |
| 6 | Static thresholds ignore pair history | No per-pair adaptive logic | PR-OPT-06 |
| 7 | GEM SL blocked for 24h | `MIN_SIGNAL_LIFESPAN_SECONDS["360_GEM"] = 86400` | PR-OPT-07 |

---

## 1. PR Priority Order and Dependencies

```
PR-OPT-05 (Telemetry) ──────────────────────────────────────────────────────────┐
PR-OPT-01 (QUIET Regime)          ←── no dependency                             │
PR-OPT-02 (Pair Quality)          ←── no dependency                             │
PR-OPT-03 (OI Validation)         ←── no dependency                             │
PR-OPT-07 (GEM Lifespan)          ←── no dependency                             │
PR-OPT-04 (Scan Strategy)         ←── no dependency (infra)                     │
                                                                                 │
PR-OPT-06 (Per-Pair Profiles) ←── depends on PR-OPT-05 (needs suppression data) ┘
```

| PR | Title | Priority | Depends On | Deploy Week |
|----|-------|----------|------------|-------------|
| PR-OPT-05 | Suppressed Signal Telemetry | P2 (deploy first) | None | Week 1 |
| PR-OPT-01 | Adaptive QUIET Regime | P0 | None | Week 1 |
| PR-OPT-02 | Dynamic Pair Quality Gates | P1 | None | Week 2 |
| PR-OPT-03 | OI Validation Refinement | P1 | None | Week 2 |
| PR-OPT-07 | GEM Lifespan Reduction | P3 | None | Week 2 |
| PR-OPT-04 | Scanning Strategy Optimization | P2 | None | Week 3 |
| PR-OPT-06 | Per-Pair Adaptive Thresholds | P3 | PR-OPT-05 | Week 4 |

> **Deploy PR-OPT-05 first** even though it is listed as P2. Telemetry provides the baseline data needed to measure the impact of all other PRs and is risk-free to deploy.

---

## 2. Implementation Timeline

### Week 1 — Observability Foundation + Highest-Impact Signal Recovery

**Day 1–2: PR-OPT-05 (Telemetry)**
- [ ] Create `src/suppression_telemetry.py` with `SuppressionEvent`, `SuppressionTracker`
- [ ] Integrate `record_suppression()` at all scanner suppression points
- [ ] Add suppression counters to `src/telemetry.py`
- [ ] Add `/suppressed` Telegram command
- [ ] Deploy and verify 4h digest appears
- [ ] Record baseline: how many signals are suppressed per reason per 4h

**Day 3–5: PR-OPT-01 (QUIET Regime)**
- [ ] Remove `"QUIET"` from `_REGIME_CHANNEL_INCOMPATIBLE` for scalp channels (keep `360_SCALP_VWAP`)
- [ ] Add `_SCALP_QUIET_REGIME_PENALTY = 1.8` override in scanner
- [ ] Add `QUIET_SCALP_MIN_CONFIDENCE = 75.0` to config
- [ ] Add `QUIET_SCALP_VOLUME_MULTIPLIER = 2.5` to config
- [ ] Reduce `_BB_WIDTH_QUIET_PCT` from 1.5 to 1.2 in `src/regime.py`
- [ ] Run tests; deploy; compare suppression digest before/after
- [ ] **Expected outcome:** QUIET regime suppression count drops by 60–80%

### Week 2 — Quality Gate Relaxation + OI Refinement

**Day 1–3: PR-OPT-02 (Pair Quality) + PR-OPT-03 (OI Validation)**
- [ ] Add `_CHANNEL_MAX_SPREAD_PCT` and `_CHANNEL_MIN_VOLUME` dicts to `signal_quality.py`
- [ ] Add `channel` parameter to `assess_pair_quality()`
- [ ] Add `spread_adjusted_confidence_delta` field to `PairQualityAssessment`
- [ ] Configure `suppressed_signals.log` handler
- [ ] Add `OITrendResult`, `OIEvaluation`, `evaluate_oi_impact()` to `order_flow.py`
- [ ] Replace `is_oi_invalidated()` calls in scanner with `evaluate_oi_impact()`
- [ ] Apply OI confidence penalty in scanner scoring
- [ ] Run tests; deploy both in same release window
- [ ] **Expected outcome:** Spread gate suppression drops; 30% of OI-rejected signals recover

**Day 4–5: PR-OPT-07 (GEM Lifespan)**
- [ ] Change `"360_GEM": 86400` to `int(os.getenv("MIN_GEM_LIFESPAN_SECONDS", "43200"))`
- [ ] Add `GEM_EARLY_EXIT_CONFIDENCE_DROP = 30.0` to config
- [ ] Implement `_is_lifespan_protected()` in `trade_monitor.py`
- [ ] Fix `sl_eval_deferred` log messages
- [ ] Update `.env.example`
- [ ] Run tests; deploy; monitor for GEM SL triggers in 6–12h window

### Week 3 — Infrastructure: Scanning Efficiency

**PR-OPT-04 (Scanning Strategy)**
- [ ] Add `fetch_all_book_tickers()` and `fetch_all_tickers_24hr()` to `src/binance.py`
- [ ] Implement `_pre_filter_pairs()` in scanner using aggregate data
- [ ] Implement 4-tier HOT/WARM/SPOT/COLD priority queue
- [ ] Add `_compute_max_concurrent()` dynamic concurrency
- [ ] Implement WS-first kline strategy with REST fallback
- [ ] Add `SCANNER_MAX_CONCURRENCY`, `WS_DEGRADED_MAX_PAIRS` env vars
- [ ] Load-test against Binance testnet to verify weight budgets
- [ ] Deploy; monitor rate limit consumption for 24h

### Week 4 — Adaptive Intelligence

**PR-OPT-06 (Per-Pair Adaptive Thresholds)**
- [ ] Create `src/pair_profile.py` with `PairProfile`, `PairProfileManager`
- [ ] Add Redis persistence with 8-day TTL
- [ ] Integrate profile overrides in scanner for QUIET penalty and spread threshold
- [ ] Add `/profile SYMBOL` Telegram command
- [ ] Deploy; profiles will begin building from scan data (full data in 24–48h)
- [ ] After 7 days: verify QUIET-dominant pairs show relaxed penalties in profile

---

## 3. Risk Assessment

| PR | Risk Level | Primary Risk | Mitigation |
|----|-----------|-------------|------------|
| PR-OPT-05 | **Low** | Memory usage from event deque | `maxlen=5000` bounds memory at ~3MB |
| PR-OPT-01 | **Medium** | Increased false-positive signals in QUIET | Confidence floor (75) + volume multiplier (2.5×) |
| PR-OPT-02 | **Low-Medium** | Wide-spread signals cause execution slippage | `spread_adjusted_confidence_delta` reduces sizing |
| PR-OPT-03 | **Medium** | Recovering OI-penalised signals adds risk | Hard reject still at >3% directional / >5% absolute |
| PR-OPT-07 | **Medium** | 12h window too short for some macro setups | `MIN_GEM_LIFESPAN_SECONDS` env var for instant revert |
| PR-OPT-04 | **Low-Medium** | Aggregate endpoint caching causes stale data | Same Binance update frequency; verify in testnet first |
| PR-OPT-06 | **Low** | Bad profile data relaxes thresholds incorrectly | Min 20 signals required before overrides applied |

### Combined Risk

Deploying PRs 01–03 together represents the highest combined signal recovery but also the highest short-term signal quality risk. Monitor the suppression telemetry digest (PR-OPT-05) for 48h after each deployment before proceeding to the next.

---

## 4. Rollback Procedures

Each PR has isolated rollback procedures documented in its individual document. Summary:

| PR | Rollback Speed | Method |
|----|---------------|--------|
| PR-OPT-01 | **Instant** | Revert `_REGIME_CHANNEL_INCOMPATIBLE` dict; restore `_BB_WIDTH_QUIET_PCT = 1.5` |
| PR-OPT-02 | **Instant** | Remove `channel` param from quality gate; restore fixed thresholds |
| PR-OPT-03 | **Instant** | Restore single-line `is_oi_invalidated()` function |
| PR-OPT-04 | **Fast** | Restore static `_MAX_CONCURRENT_SCANS = 10`; remove aggregate methods |
| PR-OPT-05 | **Instant** | Remove `src/suppression_telemetry.py`; remove `record_suppression` calls |
| PR-OPT-06 | **Instant** | Remove `src/pair_profile.py`; remove scanner override code |
| PR-OPT-07 | **Instant** | Set `MIN_GEM_LIFESPAN_SECONDS=86400` env var (no redeploy required) |

---

## 5. Testing Requirements per PR

| PR | Unit Tests Required | Integration Tests | Load Tests |
|----|---------------------|-------------------|------------|
| PR-OPT-01 | 5 (regime + volume + confidence) | 1 (24h backtest on QUIET pair) | No |
| PR-OPT-02 | 8 (spread tiers + volume tiers + delta) | 2 (KATUSDT gate; log check) | No |
| PR-OPT-03 | 9 (graduated thresholds + funding + shim) | 2 (STGUSDT / CUSDT recovery) | No |
| PR-OPT-04 | 8 (aggregate endpoint + tiers + WS fallback) | No | Yes (verify weight budget) |
| PR-OPT-05 | 7 (tracker + counters + Telegram) | No | No |
| PR-OPT-06 | 7 (profile + Redis + scanner) | No | No |
| PR-OPT-07 | 9 (lifespan + early exit + log) | No | No |

---

## 6. Before/After Metrics Framework

### Baseline Metrics (collect for 7 days before any PR)

Capture using PR-OPT-05 telemetry:

| Metric | Collection Method | Target Baseline |
|--------|-----------------|-----------------|
| Total signals suppressed per 4h | Suppression digest | Record absolute count |
| QUIET regime suppression % | `suppressed_by_regime / total` | Expected 40–60% |
| Spread gate failure % | `suppressed_by_quality / total` | Expected 10–20% |
| OI invalidation % | `suppressed_by_oi / total` | Expected 5–10% |
| Signals emitted per 4h (total) | Signal router counter | Record absolute count |
| Win rate per channel | Performance tracker | Record per-channel |
| GEM max drawdown in first 24h | Trade monitor | Record avg first-day DD |

### After-PR Target Metrics

| Metric | After PR-OPT-01 | After PR-OPT-02 | After PR-OPT-03 | After All PRs |
|--------|----------------|----------------|----------------|--------------|
| QUIET suppression % | < 20% (was 40–60%) | unchanged | unchanged | < 15% |
| Spread gate failures | unchanged | < 50% of baseline | unchanged | < 40% |
| OI hard rejections | unchanged | unchanged | < 70% of baseline | < 60% |
| Total signals emitted/4h | +15–25% | +5–10% | +5–10% | +30–45% |
| Win rate per channel | Maintained (≥ baseline) | Maintained | Maintained | Maintained |
| GEM first-24h max DD | unchanged | unchanged | unchanged | -20% (from 12h SL) |
| REST weight consumption | unchanged | unchanged | unchanged | -85% (PR-OPT-04) |

### Measuring Success

A PR is considered successful when:
1. The targeted suppression category decreases by the expected percentage
2. Win rate per affected channel does not decrease by more than 3 percentage points
3. No new error categories appear in production logs
4. Total signal volume increases (not decreases)

If win rate drops by >3pp after any PR, trigger rollback and re-evaluate thresholds.

---

## 7. Post-Deployment Monitoring Checklist

After each PR deployment, verify within 4 hours:

- [ ] `/suppressed` Telegram command returns updated digest with expected suppression reduction
- [ ] No new Python exceptions in production logs related to changed modules
- [ ] Signal emission rate is ≥ pre-deployment rate
- [ ] Win rate in signal router / performance tracker is not declining
- [ ] Rate limit usage is within budget (for PR-OPT-04: check Binance rate limit response headers)

After all PRs deployed (Week 4+), weekly review:
- [ ] Compare 7-day win rate vs. pre-optimization baseline
- [ ] Review `/profile` for top 20 pairs to verify profiles are being built correctly
- [ ] Check QUIET-dominant pair signals are passing with appropriate confidence levels
- [ ] Review `suppressed_signals.log` for emerging patterns not covered by current PRs

---

## 8. File Index

| File | PR |
|------|----|
| `docs/PR_OPTIMIZATION_01_adaptive_quiet_regime.md` | PR-OPT-01 |
| `docs/PR_OPTIMIZATION_02_dynamic_pair_quality_gates.md` | PR-OPT-02 |
| `docs/PR_OPTIMIZATION_03_oi_validation_refinement.md` | PR-OPT-03 |
| `docs/PR_OPTIMIZATION_04_scanning_strategy_optimization.md` | PR-OPT-04 |
| `docs/PR_OPTIMIZATION_05_suppressed_signal_telemetry.md` | PR-OPT-05 |
| `docs/PR_OPTIMIZATION_06_per_pair_adaptive_thresholds.md` | PR-OPT-06 |
| `docs/PR_OPTIMIZATION_07_gem_lifespan_reduction.md` | PR-OPT-07 |
| `docs/OPTIMIZATION_MASTER_PLAN.md` | This document |
