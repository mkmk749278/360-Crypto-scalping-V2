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
9. [Sequential PR Implementation Plan](#9-sequential-pr-implementation-plan)

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

## 9. Sequential PR Implementation Plan

The following 12 PRs are ordered so that each builds on the previous without circular dependencies. Earlier PRs establish foundational infrastructure (regime detection, per-pair config) that later PRs consume.

| PR | Title | Primary Dependency |
|----|-------|--------------------|
| PR_01 | Market Regime Detector Enhancement | None |
| PR_02 | Per-Pair Config Profiles | PR_01 |
| PR_03 | Adaptive EMA Thresholds | PR_02 |
| PR_04 | MACD Confirmation Layer | PR_01 |
| PR_05 | Candlestick Pattern Engine | None |
| PR_06 | Multi-Timeframe Confirmation | PR_01, PR_02 |
| PR_07 | Dynamic SL/TP ATR+Regime | PR_01, PR_02 |
| PR_08 | Trailing Stop Upgrade | PR_07 |
| PR_09 | Signal Scoring Engine | PR_04, PR_05, PR_06 |
| PR_10 | Duplicate Code Refactor | PR_01–PR_06 |
| PR_11 | Backtester Per-Pair + Regime | PR_01, PR_02, PR_07 |
| PR_12 | AI Statistical Filter | PR_09, PR_11 |

See individual PR documents (`PR_01_*.md` through `PR_12_*.md`) for full implementation steps, expected impact, and file-level change specifications.

---

*End of Master Audit Report*
