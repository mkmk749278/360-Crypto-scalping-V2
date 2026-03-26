# Master Audit Report — 360-Crypto-Scalping-V2 Signal Generation System

**Date:** 2026-03-26  
**Scope:** Full signal pipeline — channels, filters, indicators, backtester, SMC detection, regime detection  
**Auditor role:** Senior quantitative trader / software engineer

---

## Table of Contents

1. [Current Signal Logic Analysis](#1-current-signal-logic-analysis)
2. [Identified Limitations](#2-identified-limitations)
3. [Adaptive Logic Proposals per Market Regime](#3-adaptive-logic-proposals-per-market-regime)
4. [Per-Pair Strategy Recommendations](#4-per-pair-strategy-recommendations)
5. [Signal Quality Enhancement Suggestions](#5-signal-quality-enhancement-suggestions)
6. [Backtesting Gaps](#6-backtesting-gaps)
7. [Duplicate Code Identification](#7-duplicate-code-identification)
8. [Modular Design Recommendations](#8-modular-design-recommendations)
9. [Sequential PR Implementation Plan (Phase 1)](#9-sequential-pr-implementation-plan)
10. [Phase 1 Implementation Status](#10-phase-1-implementation-status)
11. [Phase 2 — Next-Phase Optimization Roadmap](#11-phase-2--next-phase-optimization-roadmap)
12. [Phase 2 Sequential PR Implementation Plan (PR_13–PR_29)](#12-phase-2-sequential-pr-implementation-plan-pr_13pr_29)

---

## 1. Current Signal Logic Analysis

### 1.1 Channel Architecture Overview

The system uses four primary channels, each targeting a distinct timeframe and trade style:

| Channel | File | Timeframes | Setup Types | SL Range | TP Ratios |
|---------|------|------------|-------------|----------|-----------|
| `360_SCALP` | `scalp.py` | M1, M5 | LIQUIDITY_SWEEP_REVERSAL, RANGE_FADE, WHALE_MOMENTUM | 0.05–0.1% | 0.5R / 1R / 1.5R |
| `360_SWING` | `swing.py` | H1, H4 | SWING_STANDARD, SWING_D1_CONFLUENCE | 0.2–0.5% | 1.5R / 3R / 5R |
| `360_SPOT` | `spot.py` | H4, D1 | BREAKOUT_INITIAL, BREAKOUT_RETEST | 0.5–2% | 2R / 5R / 10R |
| GEM Scanner | `gem_scanner.py` | D1 | Macro reversal / deep drawdown | variable | wide |

### 1.2 ScalpChannel (`src/channels/scalp.py`)

**Trigger paths (three concurrent evaluators):**

1. **Standard / LIQUIDITY_SWEEP_REVERSAL** — requires M5 liquidity sweep (SMC), ATR-adaptive momentum (threshold = `max(0.10, min(0.30, atr_pct × 0.5))`), 2-candle momentum persistence check, EMA9/EMA21 alignment, RSI extreme gate, ADX ≥ `config.adx_min`.

2. **RANGE_FADE** — requires ADX below an adaptive ceiling (18–25 depending on regime), price touching BB upper/lower band within 0.2%, RSI not yet recovered past mean-reversion window, BB squeeze guard (rejects if BB is actively expanding >10%).

3. **WHALE_MOMENTUM** — requires `whale_alert` or `volume_delta_spike`, tick-level buy/sell ratio ≥ 2:1 (`_WHALE_DELTA_MIN_RATIO`), total tick volume ≥ $500 k, OBI imbalance ≥ 1.5 on top-10 book levels.

**Regime weighting:** `_select_indicator_weights()` returns multipliers (0.3–1.5×) per path based on current regime string. All three paths are evaluated; the best regime-adjusted R-multiple wins.

**Kill zone:** Signals generated outside London (07–10 UTC) or NY (12–16 UTC) sessions receive an execution note ("outside kill zone") but are not suppressed.

### 1.3 SwingChannel (`src/channels/swing.py`)

**Trigger:** H4 ERL sweep + H1 MSS (from SMCDetector). Both must be present.

**Filters applied:**
- `check_adx_regime()` with `max_adx` cap and regime awareness
- `check_spread_adaptive()` — spreads wider in volatile regimes are tolerated
- `check_volume()` — fixed USD minimum
- EMA200 bias on H1 with ±0.5% dead zone (`_EMA200_BUFFER_PCT`)
- Bollinger band percentile position gate (`_BB_REJECTION_THRESHOLD = 0.15`)
- RSI extreme gate via `check_rsi_regime()`
- MSS candle body size minimum (`_MSS_MIN_BODY_SIZE_PCT = 0.05%`)

**Daily confluence:** Soft boost — checks whether close is within 3% of recent daily support/resistance. Marks signal as `SWING_D1_CONFLUENCE` (quality tier A+) when confirmed.

**SL/TP:** `config.sl_pct_range[0]` or ATR×1, then TP at 1.5R/3R/5R.

### 1.4 SpotChannel (`src/channels/spot.py`)

**Trigger:** H4 accumulation breakout (LONG) or distribution breakdown (SHORT) confirmed by:
- Price clearing recent 10-candle H4 high/low + ATR-adaptive breakout buffer (ATR×0.2)
- Volume expansion vs 9-candle average × regime multiplier (1.5–2.2×)
- Optional retest pattern detection (3-candle sequence)

**Filters:**
- EMA200 on H4 (bias gate) + EMA50 on D1 (trend alignment)
- ATR-normalized Bollinger squeeze threshold (scales `max(2.0, min(6.0, atr_pct × 3.0))`)
- RSI extreme gate
- SMC/MSS contradiction check

**SL/TP:** `config.sl_pct_range[0]` or ATR×1.5, then TP at 2R/5R/10R.

### 1.5 Shared Infrastructure

- **`build_channel_signal()`** in `base.py` — centralises Signal instantiation, DCA zone calc, volatility-adaptive TP ratios (BB width %), entry zone biasing (ATR×0.4 width), VWAP anchoring.
- **`lookup_signal_params()`** in `signal_params.py` — per (channel, setup_class, regime) overrides for SL multiplier, TP ratios, entry zone bias, DCA, validity window.
- **`src/filters.py`** — `check_spread`, `check_adx`, `check_ema_alignment`, `check_volume`, `check_rsi`, and regime-aware variants (`check_adx_regime`, `check_rsi_regime`, `check_ema_alignment_regime`, `check_spread_adaptive`).
- **`src/indicators.py`** — pure-compute numpy functions: EMA, SMA, ADX, ATR, Bollinger Bands, RSI, momentum, MACD (recently added).
- **`src/regime.py`** — market regime classification (`TRENDING_UP`, `TRENDING_DOWN`, `RANGING`, `VOLATILE`, `QUIET`).
- **`src/detector.py`** (`SMCDetector`) — orchestrates liquidity sweep, MSS, FVG, whale alert, OI invalidation, CVD divergence detection.
- **`src/backtester.py`** — single-pass replay with static slippage, fixed SL/TP evaluation, per-channel summary metrics.

---

## 2. Identified Limitations

### 2.1 Uniform Logic Across Pairs and Regimes

**Problem:** Despite some regime-adaptive weighting in the ScalpChannel, the core thresholds (ADX minimum, RSI overbought/oversold levels, spread max, volume minimum, SL percent range, TP ratios) are **global** values set once in `config/__init__.py`. Every pair — BTC, ETH, DOGE, mid-cap altcoins — uses the same parameters.

**Impact:**
- BTC USDT (ATR ~0.3%, tight spreads, deep liquidity) triggers false negatives: ATR-based momentum threshold often filters out valid BTC moves because BTC's ATR percentage is lower than the system's mid-curve assumptions.
- High-volatility altcoins (DOGE, SHIB — ATR ~0.8–1.5%) trigger false positives: their normal intraday noise satisfies the momentum and BB-touch conditions designed for calmer assets.
- Spread filters designed for BTC reject illiquid pairs that are profitable on longer holding periods (Spot/GEM) but have wider spreads.

### 2.2 False Positives from Static Indicator Thresholds

| Threshold | Current Value | Issue |
|-----------|--------------|-------|
| Momentum persistence | 2 candles | Insufficient for volatile regimes where 3–5 candle confirmation is needed |
| ADX minimum (scalp) | `config.adx_min` (global) | Too low in QUIET regime → weak trend signals; too high in VOLATILE → misses genuine breakouts |
| RSI overbought gate | `check_rsi_regime` | Does not scale RSI thresholds per pair (BTC oscillates more narrowly than altcoins) |
| BB expansion guard | >10% width increase | Arbitrary; does not account for pair-specific BB expansion norms |
| Swing EMA200 buffer | ±0.5% fixed | Same buffer for BTC (~$250 zone) and a $0.10 altcoin — asymmetric sensitivity |

### 2.3 Missed Trades

1. **No MACD confirmation layer** — MACD histogram direction (recently added to `indicators.py`) is computed but not gated in any channel evaluate path.
2. **No candlestick pattern recognition** — engulfing candles, pin bars, doji at key levels are classic high-probability setups not captured by any filter.
3. **Kill zone is soft-only** — signals outside high-liquidity windows are emitted with a note but not suppressed, leading to low-conviction entries being taken.
4. **No multi-timeframe confirmation gate** — the swing channel requires H4 sweep + H1 MSS, but there is no cross-channel MTF check ensuring scalp signals align with the higher-timeframe bias.
5. **Volume filter is 24-hour aggregate** — does not catch intraday session volume surges (pre-market accumulation, news-driven spikes).

### 2.4 Signal Frequency vs. Quality Trade-off

- The RANGE_FADE path in ScalpChannel is the highest-frequency emitter but has no win-rate tracking gate. In trending regimes, mean-reversion signals are counter-trend and predominantly losing.
- WHALE_MOMENTUM requires a $500k tick-level volume threshold that rarely triggers on mid-cap pairs, reducing signal frequency without compensating for quality.
- The Spot channel's retest pattern detection is a useful quality boost, but there is no "freshness" gate — a retest that occurred 20 candles ago can still pass.

### 2.5 Trailing Stop Limitations

- Trailing stop description is a static string: `f"{config.trailing_atr_mult}×ATR"`. Actual trailing stop execution is handled externally (trade monitor), but the `original_sl_distance` field is the only dynamic input.
- No partial profit locking: when TP1 is hit, position size is not reduced and trailing distance is not tightened. A full reversal can convert a winner into a breakeven or loss.
- ATR for trailing is computed at signal creation; it does not update as new candles form during the trade's life.

---

## 3. Adaptive Logic Proposals per Market Regime

### 3.1 TRENDING_UP / TRENDING_DOWN

**Recommended adjustments:**
- Disable RANGE_FADE path entirely (counter-trend in a trend = high loss rate).
- Increase WHALE_MOMENTUM weight to 1.5× (order flow more directional in trends).
- Loosen momentum persistence requirement to 2 candles (trends sustain momentum).
- Tighten RSI overbought gate to 75 (don't chase extended moves).
- Use wider TP3 target (1.5× ATR ratio) to capture full trend leg.
- EMA200 buffer can be tightened to ±0.3% (trend is confirmed, less ambiguity).

### 3.2 RANGING / QUIET

- Enable RANGE_FADE exclusively (and suppress LIQUIDITY_SWEEP and WHALE_MOMENTUM).
- Tighten BB-touch requirement to 0.1% (not 0.2%) — only trade at extreme band edges.
- RSI thresholds: require RSI ≤ 35 for LONG entry, ≥ 65 for SHORT entry.
- Use compressed TP ratios (0.7× factor already implemented in `build_channel_signal`).
- ADX ceiling: 20 for QUIET, 25 for RANGING.
- Volume filter: increase to 2.2× average (only trade on meaningful volume in quiet markets).

### 3.3 VOLATILE

- Prioritise WHALE_MOMENTUM (weight 1.5×) and LIQUIDITY_SWEEP (weight 1.3×).
- Increase momentum persistence to 3 candles.
- Widen spread tolerance by 1.5× (volatile regimes have wider natural spreads).
- Use wider SL (ATR × 1.2 rather than ATR × 0.5–0.8) to avoid stop-hunts.
- Stretch TP ratios by 1.3× (price travels further in volatile environments).
- Apply stricter kill zone gate: suppress signals entirely outside kill zones.

### 3.4 Regime Detection Enhancement

Current regime detection in `src/regime.py` should be enhanced with:
- **Volatility percentile** (14-day rolling ATR percentile) — more robust than raw ATR value.
- **ADX slope** (1-period change in ADX) — rising ADX confirms a regime is strengthening.
- **Volume profile classification** — above/below VWAP volume balance.
- These improvements are detailed in PR_01.

---

## 4. Per-Pair Strategy Recommendations

### 4.1 BTC-USDT

- ATR percentile: typically low–medium (0.2–0.5% per 5-minute candle).
- Recommended adjustments: tighten momentum threshold to 0.12%, widen BB-touch tolerance to 0.3%, set `min_volume` ≥ $50M/24h, apply tighter spread max (0.01%).
- Preferred setups: LIQUIDITY_SWEEP_REVERSAL, SWING_D1_CONFLUENCE.
- Avoid: WHALE_MOMENTUM (too many false positives from high-frequency BTC trades).

### 4.2 ETH-USDT

- ATR percentile: medium (0.3–0.7%).
- Recommended adjustments: momentum threshold 0.15%, standard parameters, enable MACD confirmation.
- Preferred setups: all three scalp paths perform well; SWING with H4 structure is reliable.
- ETH often leads BTC in altcoin rallies — consider an ETH-BTC cross-asset filter.

### 4.3 Mid-Cap Altcoins (e.g., LINK, MATIC, SOL)

- ATR percentile: medium-high (0.5–1.0%).
- Recommended adjustments: momentum threshold 0.20–0.25%, tighten ADX minimum to 22, require 3-candle persistence, enforce kill zone hard gate.
- Preferred setups: BREAKOUT_RETEST (Spot), RANGE_FADE during consolidation phases.

### 4.4 High-Volatility Altcoins (e.g., DOGE, SHIB, PEPE)

- ATR percentile: high (1.0–3.0%).
- Recommended adjustments: momentum threshold 0.30%, require 4-candle persistence, widen SL to ATR×1.0, compress TP1 to 0.4R (quick profit taking), disable WHALE_MOMENTUM (tick volume requirements too high for thin order books).
- Prefer RANGE_FADE only in confirmed QUIET/RANGING regimes.
- Apply statistical false-positive filter (PR_12) before emitting.

---

## 5. Signal Quality Enhancement Suggestions

### 5.1 MACD Histogram Confirmation (PR_04)

MACD histogram is computed in `src/indicators.py` and stored in scanner indicator dicts (`macd_histogram_last`, `macd_histogram_prev`). Adding an optional gate:
- LONG entry: histogram must be rising (last > prev) or positive.
- SHORT entry: histogram must be falling (last < prev) or negative.
- Regime-adaptive: mandatory in RANGING/QUIET, optional (soft penalty) in VOLATILE.

### 5.2 Candlestick Pattern Engine (PR_05)

Detect on M5/H1 candles:
- **Bullish engulfing**: current candle body engulfs prior candle body; direction = LONG.
- **Bearish engulfing**: inverse.
- **Pin bar / hammer**: wick > 2× body size; direction determined by wick position.
- **Doji**: body < 10% of range; signals indecision — suppress signals when doji appears at trade trigger.
- **Morning/Evening star**: 3-candle pattern at BB extremes.

Pattern presence should add to the composite confidence score (PR_09) rather than acting as a hard gate.

### 5.3 Multi-Timeframe Confirmation (PR_06)

Before emitting a M5 scalp signal, require that at least one higher timeframe (H1 or H4) shows:
- EMA alignment in the same direction, OR
- RSI in a non-extreme zone on the higher TF, OR
- ADX on H1 above threshold (for trend-following scalps).

### 5.4 Signal Scoring Engine (PR_09)

A composite 0–100 score replacing the current ad-hoc confidence approach:
- SMC confluence: 0–25 pts (sweep quality, FVG proximity)
- Regime alignment: 0–20 pts (does regime favour this setup type?)
- Volume confirmation: 0–15 pts (relative volume vs 20-period average)
- Indicator confluence: 0–20 pts (MACD + RSI + EMA all aligned)
- Candlestick pattern: 0–10 pts
- MTF confirmation: 0–10 pts
- Threshold: emit only at score ≥ 60; A+ tier at ≥ 80.

### 5.5 Statistical False-Positive Filter (PR_12)

Maintain rolling win-rate statistics per (channel, pair, regime) with a 30-signal window. When rolling win rate drops below 40% for more than 10 consecutive signals, apply a confidence penalty of –10 pts or suppress entirely until win rate recovers above 50%.

---

## 6. Backtesting Gaps

### 6.1 Missing Per-Pair Parameters

Current `Backtester.run()` uses channel-level config globally. There is no way to:
- Run a parameter sweep per pair (e.g., test different ATR multipliers for BTC vs DOGE).
- Tag backtest results with the regime active during each signal.
- Isolate performance by setup type (RANGE_FADE vs WHALE_MOMENTUM).

### 6.2 No Order-Book Simulation

Slippage is applied as a fixed percentage (`slippage_pct`) applied uniformly. A more realistic model would:
- Use bid-ask spread at entry time (already available in `Signal.spread_pct`).
- Apply depth-of-book impact for larger position sizes.
- Simulate partial fill probability for limit orders in the entry zone.

### 6.3 Static Regime During Backtest

The backtester does not compute or record the market regime at each bar. All signals are tested with a single static regime string (or no regime), ignoring how regime shifts affect signal quality during the test window.

### 6.4 No Walk-Forward Validation

The backtester runs a single full-period backtest. Walk-forward validation (rolling in-sample/out-of-sample splits) is absent, making it impossible to assess parameter overfitting.

---

## 7. Duplicate Code Identification

### 7.1 SL/TP Calculation

All three ScalpChannel paths (`_evaluate_standard`, `_evaluate_range_fade`, `_evaluate_whale_momentum`) call `self._calc_levels()` locally and then call `build_channel_signal()`. The `build_channel_signal()` function recalculates TP levels again internally when `bb_width_pct` or `params.tp_ratios` are provided. This results in **double computation** and potential inconsistency if the `_calc_levels()` result is passed as `tp1/tp2/tp3` but then overridden by `build_channel_signal()`.

**Files:** `src/channels/scalp.py:_calc_levels()`, `src/channels/base.py:build_channel_signal()`

### 7.2 Volume Expansion Logic

The volume expansion check (compare last candle USD volume to N-period average) is duplicated in:
- `src/channels/spot.py:_try_long()` and `_try_short()` (implemented as inline list comprehension)
- `src/channels/scalp.py:_evaluate_whale_momentum()` (via `smc_data["recent_ticks"]` aggregation)

These should be centralised in a `check_volume_expansion(volumes, closes, n, multiplier)` function in `src/filters.py`.

### 7.3 RSI Gate Logic

`check_rsi_regime()` is called independently in every channel `evaluate()` method. The regime-specific RSI thresholds (overbought 70/75, oversold 25/30) are computed inside `filters.py` but not accessible for configuration. If thresholds need adjustment for a specific pair, there is no mechanism to pass pair context.

### 7.4 Spread Check Duplication

`check_spread()` and `check_spread_adaptive()` both exist; the adaptive version is used in Swing and Spot but not consistently in Scalp (which calls `_pass_basic_filters()` using the non-adaptive version).

### 7.5 Trailing Stop Description

Each channel hardcodes `trailing_desc=f"{config.trailing_atr_mult}×ATR"` identically. If trailing logic changes, multiple files need updating.

---

## 8. Modular Design Recommendations

### 8.1 Extract Regime-Adaptive Threshold Registry

Create `src/regime_thresholds.py` that maps `(regime, pair_tier)` → threshold overrides (ADX, RSI, momentum, spread). Channels look up thresholds at evaluation time rather than hardcoding regime-specific logic inline.

### 8.2 Centralise Volume Analysis

Move volume expansion logic from individual channels to `src/filters.py:check_volume_expansion()`. Move tick-level aggregation to `src/volume_analysis.py` (distinct from `src/volume_divergence.py`).

### 8.3 Signal Parameter Registry

Extend `src/channels/signal_params.py` to include pair-specific rows in the lookup table, not just `(channel, setup_class, regime)` triples. This allows per-pair threshold tuning without touching channel evaluate logic.

### 8.4 Abstract Candlestick Pattern Layer

Implement `src/chart_patterns.py` (already exists as a stub) as a full registry of detected patterns per symbol/timeframe, consumed by channels as confluence inputs.

### 8.5 Unified Backtester Result Schema

Add regime tag, setup_class, and pair columns to `BacktestResult` so downstream analysis tools can slice performance by all relevant dimensions.

---

## 9. Sequential PR Implementation Plan (Phase 1)

The following 12 PRs are ordered so that each builds on the previous without circular dependencies. Earlier PRs establish foundational infrastructure (regime detection, per-pair config) that later PRs consume.

| PR | Title | Primary Dependency | GitHub PR | Status |
|----|-------|--------------------|-----------|--------|
| PR_01 | Market Regime Detector Enhancement | None | [#127](https://github.com/kishore446/360-Crypto-scalping-V2/pull/127) | ✅ Merged |
| PR_02 | Per-Pair Config Profiles | PR_01 | [#128](https://github.com/kishore446/360-Crypto-scalping-V2/pull/128) | ✅ Merged |
| PR_03 | Adaptive EMA Thresholds | PR_02 | [#129](https://github.com/kishore446/360-Crypto-scalping-V2/pull/129) | ✅ Merged |
| PR_04 | MACD Confirmation Layer | PR_01 | [#130](https://github.com/kishore446/360-Crypto-scalping-V2/pull/130) | ✅ Merged |
| PR_05 | Candlestick Pattern Engine | None | [#131](https://github.com/kishore446/360-Crypto-scalping-V2/pull/131) | ✅ Merged |
| PR_06 | Multi-Timeframe Confirmation | PR_01, PR_02 | [#132](https://github.com/kishore446/360-Crypto-scalping-V2/pull/132) | ✅ Merged |
| PR_07 | Dynamic SL/TP ATR+Regime | PR_01, PR_02 | [#133](https://github.com/kishore446/360-Crypto-scalping-V2/pull/133) | ✅ Merged |
| PR_08 | Trailing Stop Upgrade | PR_07 | [#134](https://github.com/kishore446/360-Crypto-scalping-V2/pull/134) | ✅ Merged |
| PR_09 | Signal Scoring Engine | PR_04, PR_05, PR_06 | [#135](https://github.com/kishore446/360-Crypto-scalping-V2/pull/135) | ✅ Merged |
| PR_10 | Duplicate Code Refactor | PR_01–PR_06 | [#136](https://github.com/kishore446/360-Crypto-scalping-V2/pull/136) | ✅ Merged |
| PR_11 | Backtester Per-Pair + Regime | PR_01, PR_02, PR_07 | [#137](https://github.com/kishore446/360-Crypto-scalping-V2/pull/137) | ✅ Merged |
| PR_12 | AI Statistical Filter | PR_09, PR_11 | [#138](https://github.com/kishore446/360-Crypto-scalping-V2/pull/138) | ✅ Merged |

See individual PR documents (`PR_01_*.md` through `PR_12_*.md`) for full implementation steps, expected impact, and file-level change specifications.

---

## 10. Phase 1 Implementation Status

All 12 Phase 1 PRs have been implemented and merged as of 2026-03-26. The table below summarises the final status of each PR with its corresponding GitHub pull request.

| PR | Title | GitHub PR | Merged Date | Status |
|----|-------|-----------|-------------|--------|
| PR_01 | Market Regime Detector Enhancement | [#127](https://github.com/kishore446/360-Crypto-scalping-V2/pull/127) | 2026-03-26 | ✅ Merged |
| PR_02 | Per-Pair Config Profiles | [#128](https://github.com/kishore446/360-Crypto-scalping-V2/pull/128) | 2026-03-26 | ✅ Merged |
| PR_03 | Adaptive EMA Thresholds | [#129](https://github.com/kishore446/360-Crypto-scalping-V2/pull/129) | 2026-03-26 | ✅ Merged |
| PR_04 | MACD Confirmation Layer | [#130](https://github.com/kishore446/360-Crypto-scalping-V2/pull/130) | 2026-03-26 | ✅ Merged |
| PR_05 | Candlestick Pattern Engine | [#131](https://github.com/kishore446/360-Crypto-scalping-V2/pull/131) | 2026-03-26 | ✅ Merged |
| PR_06 | Multi-Timeframe Confirmation | [#132](https://github.com/kishore446/360-Crypto-scalping-V2/pull/132) | 2026-03-26 | ✅ Merged |
| PR_07 | Dynamic SL/TP ATR+Regime | [#133](https://github.com/kishore446/360-Crypto-scalping-V2/pull/133) | 2026-03-26 | ✅ Merged |
| PR_08 | Trailing Stop Upgrade | [#134](https://github.com/kishore446/360-Crypto-scalping-V2/pull/134) | 2026-03-26 | ✅ Merged |
| PR_09 | Signal Scoring Engine | [#135](https://github.com/kishore446/360-Crypto-scalping-V2/pull/135) | 2026-03-26 | ✅ Merged |
| PR_10 | Duplicate Code Refactor | [#136](https://github.com/kishore446/360-Crypto-scalping-V2/pull/136) | 2026-03-26 | ✅ Merged |
| PR_11 | Backtester Per-Pair + Regime | [#137](https://github.com/kishore446/360-Crypto-scalping-V2/pull/137) | 2026-03-26 | ✅ Merged |
| PR_12 | AI Statistical Filter | [#138](https://github.com/kishore446/360-Crypto-scalping-V2/pull/138) | 2026-03-26 | ✅ Merged |

---

## 11. Phase 2 — Next-Phase Optimization Roadmap

With Phase 1 complete the system has per-pair adaptive logic, multi-indicator confirmation, dynamic SL/TP, a composite scoring engine, and statistical filtering. Phase 2 addresses the next tier of improvements: order-flow intelligence, portfolio-level risk controls, automated KPI monitoring, more realistic backtesting, and structural maintainability.

### 11.1 Advanced Signal Generation

#### Order Flow Microstructure Signals
- **Liquidation cascade detector** — monitor open interest drops combined with funding rate spikes as a leading indicator of forced liquidations; score +5 when detected.
- **Funding rate divergence** — price trending up while funding rate turns negative indicates over-leveraged longs being washed out; use as a bearish scoring modifier (−5) or vice versa.
- **Delta divergence enhancement** — extend the existing CVD divergence logic to detect multi-bar divergences (price higher highs, CVD lower highs) as a stronger conviction signal.

#### AI-Driven Adaptive Signal Weighting
- **Online learning layer** — replace static weight tables in `signal_params.py` with an EWMA-based logistic regression that auto-adjusts per (channel, pair, regime) after every closed trade.
- **GPT signal pre-screener** — optional: route candidate signals through a macro context summary before emission to filter news-driven reversals.
- **Feature importance tracking** — track which indicators (MACD, RSI, SMC, candlestick) have been historically predictive per pair × regime combination.

#### Cross-Channel Confluence Scoring
- PRs #118 and #119 attempted cross-channel confluence detection but were closed without merge. PR_17 will implement this correctly.
- When two or more channels fire on the same pair and direction within a 5-minute window, apply a +15 point score boost and emit a dedicated confluence alert.

#### Session-Adaptive Thresholds
- **Asian session** (+5 to minimum score threshold): lower volatility, raise bar.
- **London open** (−5): high liquidity, trust signals more.
- **NY open** (−3): strong directionality, moderate trust.
- **Weekend** (+10): thin markets, require higher conviction.

#### Market Microstructure Timing
- **Kill zone enhancement** — hard-gate (suppress) signals generated outside London/NY sessions for SCALP channel during VOLATILE regime.
- **Volatility forecast gate** — use realised volatility forecast (GARCH-lite or rolling 15-min ATR) to suppress signals when predicted volatility exceeds 3σ of historical norm.

### 11.2 Portfolio & Risk Management

#### Signal-Score-Based Position Sizing
- Wire composite score (0–100) from PR_09/PR_135 into `risk.py::_position_size()`.
- Tiered scaling: Score 80–100 → 100% of base position; Score 65–79 → 75%; Score 50–64 → 50%.
- Low-conviction signals are still emitted but with smaller capital allocation rather than being suppressed outright.

#### Correlation-Aware Exposure Limits
- Extend `correlation.py` with a rolling 24-hour Pearson correlation matrix across all active pairs.
- Cap total BTC-beta exposure at 0.85 (prevent the entire portfolio from behaving like levered BTC).
- Add per-sector caps: max 40% portfolio allocation to any single sector (DeFi, Layer-1, Meme, etc.).

#### Portfolio-Level Drawdown Protection
- Yellow at −3% daily portfolio drawdown → reduce all new position sizes by 50%.
- Red at −5% → halt new signal emission for 4 hours.
- Black at −8% → halt all trading for 24 hours and notify admin via Telegram.

#### Volatility-Regime Stop-Loss Adaptation
- After 2× ATR expansion from entry (intraday), tighten trailing stop to 0.5× ATR.
- On volatility expansion exit: if realised volatility doubles in under 15 minutes, close position at market.

#### Kelly Criterion Sizing
- Optional overlay for paper portfolio and backtester: compute fractional Kelly (`f* = (p × (b+1) − 1) / b`) using rolling win rate and average win/loss ratio per (channel, pair, regime).
- Apply half-Kelly in live sizing; full-Kelly for backtest benchmarking only.

### 11.3 Performance, Monitoring & KPIs

#### KPI Dashboard (`/dashboard` Telegram command)
Metrics per channel and aggregate:
- Win Rate (7-day rolling)
- Profit Factor
- Sharpe Ratio
- Max Drawdown
- Signal Frequency (signals/hour)
- False Positive Rate
- Avg Score: Winners vs Losers
- Avg Trade Duration

#### Automated Anomaly Detection Service
Background service running every 15 minutes checking:
- Signal frequency drop >50% vs 7-day average → alert.
- Win rate collapse: rolling 20-trade win rate < 30% → alert + optional auto-pause.
- Score drift: composite score mean deviates >15 pts from 30-day baseline → alert.
- Channel silence: any channel emits zero signals for >2 hours during active market hours → alert.

#### Regime Performance Attribution
- Tag every trade record with the `RegimeContext.label` at entry time.
- Weekly auto-generated report showing P&L breakdown by regime (`TRENDING_UP`, `TRENDING_DOWN`, `RANGE_NARROW`, `RANGE_WIDE`, `VOLATILE_EXPANSION`) per channel.

#### Signal Score Calibration Report
- Monthly report: compare predicted composite score (at signal emission) vs actual trade outcome.
- Detect score inflation — if high-score signals (>80) win at the same rate as medium-score signals (60–79), the scoring model needs recalibration.

### 11.4 Backtesting & Simulation Enhancements

#### Realistic Slippage & Fee Model
- Volume-dependent slippage: `slippage_pct = base_slippage + (position_size / avg_volume_1m) × impact_coefficient`.
- Major pairs (BTC, ETH): base = 0.01%, impact coefficient = 0.5.
- Altcoins: base = 0.05%, impact coefficient = 2.0.
- Maker/taker fee differentiation: maker = 0.02%, taker = 0.04%.
- Configurable latency injection (50ms–500ms) to simulate execution delay.

#### Scenario-Based Stress Testing
- Flash crash simulator: apply a −15% shock over 5 candles to every open position and compute portfolio recovery.
- Liquidity drought: widen spreads 5× for 30-minute windows; measure how many signals survive filters.
- Correlation breakdown: inject a period where all pair correlations temporarily converge to 0.95.

#### Monte Carlo Confidence Intervals
- After every standard backtest, run 1 000 randomised simulations:
  1. Randomly reorder trade outcomes (sequence-of-returns risk).
  2. Vary each entry/exit by ±1 ATR.
  3. Randomly drop 10% of signals (execution failure simulation).
- Report P5/P50/P95 confidence intervals for final equity, max drawdown, and Sharpe ratio.

#### Forward-Testing Framework / Paper Trading Bridge
- `PaperTradingValidator` records every live signal and its actual outcome.
- Compares rolling paper-trading performance with backtest-predicted performance.
- Alerts when divergence exceeds 2σ over a 30-trade window.

#### Walk-Forward Optimization Automation
- Weekly cron job: re-run walk-forward optimisation on the most recent 90 days of data.
- If out-of-sample Sharpe ratio > 0.5, auto-push updated parameters to config.
- Archive previous parameter set with timestamp for rollback.

### 11.5 Codebase & Maintainability

#### Scanner Decomposition
Split the 90KB `scanner.py` monolith into a `src/scanner/` subpackage:
- `src/scanner/data_fetcher.py` — kline and order-book data retrieval.
- `src/scanner/indicator_compute.py` — per-pair indicator calculation.
- `src/scanner/signal_dispatch.py` — signal creation and routing handoff.
- `src/scanner/orchestrator.py` — main scan loop and pair iteration.
- `scanner.py` becomes a thin entry-point wrapper. No single file exceeds 25KB.

#### Config Modularization
Split the 36KB `config/__init__.py` into domain-specific modules:
- `config/pairs.py` — MAJOR/MIDCAP/ALTCOIN tier definitions.
- `config/regime.py` — regime parameter tables.
- `config/channels.py` — channel-specific settings.
- `config/risk.py` — risk thresholds and circuit breaker levels.
- `config/base.py` — common settings and environment variable loading.
- Re-export everything from `config/__init__.py` for full backward compatibility.

#### Pydantic Model Validation
- Replace `SimpleNamespace` and ad-hoc `dataclass` usage with Pydantic models for `Signal`, `RegimeContext`, and `RiskAssessment`.
- Add field validators: score clamped 0–100, all numeric fields have min/max constraints.
- Validation failures log a warning and fall back to defaults rather than raising.

#### Architecture Documentation
- Create `docs/architecture.md` with signal flow diagram (Mermaid).
- Add module-level docstrings to all major source files.
- Annotate magic numbers with `# WHY:` comments explaining their derivation.

---

## 12. Phase 2 Sequential PR Implementation Plan (PR_13–PR_29)

The 17 Phase 2 PRs are grouped into five sub-phases to allow parallel development tracks where dependencies permit.

### Phase 2A — Risk & Reliability (PR_13–PR_16)

| PR | Title | Modules Affected | Priority | Dependency |
|----|-------|-----------------|----------|------------|
| PR_13 | Portfolio-Level Drawdown Circuit Breaker | `circuit_breaker.py`, `signal_router.py`, `config/` | P0 | PR_12 |
| PR_14 | Scanner Decomposition (Part 1: Data Fetching & Indicators) | `scanner.py` → `src/scanner/*.py` | P0 | None |
| PR_15 | Correlation-Aware Exposure Filter | `correlation.py`, `risk.py`, `signal_router.py` | P1 | PR_13 |
| PR_16 | Signal-Score-Weighted Position Sizing | `risk.py`, `scanner.py`, `signal_quality.py` | P1 | PR_09 (Phase 1) |

### Phase 2B — Signal Intelligence (PR_17–PR_20)

| PR | Title | Modules Affected | Priority | Dependency |
|----|-------|-----------------|----------|------------|
| PR_17 | Cross-Channel Confluence Engine | New: `src/confluence.py`, modify `signal_router.py` | P1 | PR_14 |
| PR_18 | Funding Rate Divergence Module | New: `src/funding_rate.py`, modify `signal_quality.py`, `config/` | P1 | PR_01, PR_02 (Phase 1) |
| PR_19 | Online Indicator Weight Learning | `signal_params.py`, `stat_filter.py`, `signal_quality.py` | P2 | PR_12 (Phase 1) |
| PR_20 | Session-Adaptive Threshold Engine | `kill_zone.py`, `signal_quality.py`, `config/` | P2 | PR_02 (Phase 1) |

### Phase 2C — Backtesting Maturity (PR_21–PR_23)

| PR | Title | Modules Affected | Priority | Dependency |
|----|-------|-----------------|----------|------------|
| PR_21 | Realistic Slippage & Fee Model | `backtester.py`, `config/` | P0 | PR_11 (Phase 1) |
| PR_22 | Monte Carlo Equity Simulation | `backtester.py`, new: `src/monte_carlo.py` | P1 | PR_21 |
| PR_23 | Paper Trading Reconciliation | New: `src/reconciliation.py`, modify `paper_portfolio.py` | P2 | PR_21 |

### Phase 2D — Monitoring & Observability (PR_24–PR_26)

| PR | Title | Modules Affected | Priority | Dependency |
|----|-------|-----------------|----------|------------|
| PR_24 | KPI Dashboard Command | `performance_tracker.py`, new: `src/commands/dashboard.py` | P1 | PR_13 |
| PR_25 | Automated Anomaly Detection Service | New: `src/anomaly_monitor.py`, modify `main.py` | P1 | PR_24 |
| PR_26 | Regime Performance Attribution Reports | `performance_tracker.py`, `trade_observer.py` | P2 | PR_11 (Phase 1) |

### Phase 2E — Codebase Health (PR_27–PR_29)

| PR | Title | Modules Affected | Priority | Dependency |
|----|-------|-----------------|----------|------------|
| PR_27 | Scanner Decomposition (Part 2: Signal Dispatch & Orchestration) | `scanner.py` → `src/scanner/*.py` | P0 | PR_14 |
| PR_28 | Config Modularization | `config/` | P1 | PR_27 |
| PR_29 | Pydantic Model Validation + Architecture Docs | Multiple source files, `docs/` | P2 | PR_28 |

### Recommended Implementation Schedule

```
Week 1:  PR_13 (Drawdown Breaker) → PR_14 (Scanner Split P1) → PR_21 (Slippage Model)
Week 2:  PR_15 (Correlation Filter) → PR_16 (Score-Weighted Sizing) → PR_24 (Dashboard)
Week 3:  PR_17 (Confluence) → PR_18 (Funding Rate) → PR_25 (Anomaly Monitor)
Week 4:  PR_27 (Scanner Split P2) → PR_28 (Config Split) → PR_22 (Monte Carlo)
Week 5:  PR_19 (Online Learning) → PR_20 (Session Thresholds) → PR_26 (Regime Attribution)
Week 6:  PR_23 (Reconciliation) → PR_29 (Pydantic + Docs) → Integration Testing
```

### Expected Cumulative Impact

| Metric | Current (Post Phase 1) | Target (Post Phase 2) |
|--------|----------------------|-----------------------|
| Signal Win Rate | ~45–55% | 55–65% |
| False Positive Rate | ~35–40% | 20–25% |
| Max Drawdown | Unbounded | Capped at −8% |
| Backtest-to-Live Gap | ~15–20% | <5% |
| Time to Diagnose Issues | Hours (manual) | Minutes (automated) |
| Scanner File Complexity | 90KB monolith | 4 files, max 22KB each |

See individual PR documents (`PR_13_*.md` through `PR_29_*.md`) for full implementation steps, expected impact, and file-level change specifications.

---

*End of Master Audit Report*
