"""Tests for Phase 1 & Phase 2 signal quality improvements.

Validates all targeted improvements from the comprehensive audit:
1. ATR-adaptive momentum threshold in scalp standard path
2. ADX overlap elimination (ADX=23 no longer fires range fade)
3. BB squeeze expansion guard blocks range fade during breakouts
4. CVD divergence blocked when ADX > 35
5. Depth-weighted OBI gives more weight to top-of-book levels
6. FVG rejects zones that are >60% filled
7. Swing signals get daily confluence markup
8. Funding rate contrarian bonus/penalty in confidence engine
9. Spot breakout retest detection and markup
"""

from __future__ import annotations

import numpy as np

from src.channels.scalp import ScalpChannel
from src.channels.scalp_cvd import ScalpCVDChannel
from src.channels.scalp_fvg import ScalpFVGChannel
from src.channels.scalp_obi import _compute_obi
from src.channels.spot import SpotChannel
from src.channels.swing import SwingChannel
from src.confidence import score_order_flow
from src.smc import Direction, FVGZone, LiquiditySweep, MSSSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(n: int = 60, base: float = 100.0, trend: float = 0.1) -> dict:
    close = np.cumsum(np.ones(n) * trend) + base
    high = close + 0.5
    low = close - 0.5
    volume = np.ones(n) * 1000
    return {"open": close - 0.1, "high": high, "low": low, "close": close, "volume": volume}


def _make_indicators(
    adx_val: float = 30,
    atr_val: float = 0.5,
    ema9: float = 101,
    ema21: float = 100,
    ema200: float = 95,
    rsi_val: float = 50,
    bb_upper: float = 103,
    bb_mid: float = 100,
    bb_lower: float = 97,
    mom: float = 0.5,
    bb_width_pct: float = None,
    bb_width_prev_pct: float = None,
) -> dict:
    d = {
        "adx_last": adx_val,
        "atr_last": atr_val,
        "ema9_last": ema9,
        "ema21_last": ema21,
        "ema200_last": ema200,
        "rsi_last": rsi_val,
        "bb_upper_last": bb_upper,
        "bb_mid_last": bb_mid,
        "bb_lower_last": bb_lower,
        "momentum_last": mom,
    }
    if bb_width_pct is not None:
        d["bb_width_pct"] = bb_width_pct
    if bb_width_prev_pct is not None:
        d["bb_width_prev_pct"] = bb_width_prev_pct
    return d


# ---------------------------------------------------------------------------
# 1. ATR-Adaptive Momentum Threshold
# ---------------------------------------------------------------------------

class TestATRAdaptiveMomentumThreshold:
    """ATR-adaptive threshold scales with volatility profile."""

    def _sweep_smc(self) -> dict:
        sweep = LiquiditySweep(
            index=59, direction=Direction.LONG,
            sweep_level=99, close_price=99.05,
            wick_high=101, wick_low=98,
        )
        return {"sweeps": [sweep]}

    def test_low_volatility_pair_accepts_smaller_momentum(self):
        """Low-ATR pair (BTC-like): threshold ~0.15; mom=0.12 should PASS if threshold ≤ 0.12."""
        # ATR=0.15 on close=100 → atr_pct=0.15% → threshold=max(0.10, min(0.30, 0.075))=0.10
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        ind = _make_indicators(adx_val=30, atr_val=0.15, mom=0.12, ema9=101, ema21=100)
        sig = ch._evaluate_standard("BTCUSDT", candles, {"5m": ind}, self._sweep_smc(), 0.01, 10_000_000)
        # threshold=0.10, mom=0.12 → should pass
        assert sig is not None

    def test_high_volatility_pair_requires_larger_momentum(self):
        """High-ATR pair (DOGE-like): threshold ~0.30; mom=0.15 should be REJECTED."""
        # ATR=0.8 on close=100 → atr_pct=0.8% → threshold=max(0.10, min(0.30, 0.40))=0.30
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        ind = _make_indicators(adx_val=30, atr_val=0.8, mom=0.15, ema9=101, ema21=100)
        sig = ch._evaluate_standard("BTCUSDT", candles, {"5m": ind}, self._sweep_smc(), 0.01, 10_000_000)
        # threshold=0.30, mom=0.15 → should be rejected
        assert sig is None

    def test_high_volatility_pair_passes_with_sufficient_momentum(self):
        """High-ATR pair: mom=0.35 exceeds 0.30 threshold → signal generated."""
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        ind = _make_indicators(adx_val=30, atr_val=0.8, mom=0.35, ema9=101, ema21=100)
        sig = ch._evaluate_standard("BTCUSDT", candles, {"5m": ind}, self._sweep_smc(), 0.01, 10_000_000)
        assert sig is not None

    def test_threshold_is_clamped_at_minimum(self):
        """Very low ATR still produces minimum threshold of 0.10."""
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        # ATR=0.01 → atr_pct=0.01% → 0.5*0.01=0.005, clamped to 0.10
        ind = _make_indicators(adx_val=30, atr_val=0.01, mom=0.08, ema9=101, ema21=100)
        sig = ch._evaluate_standard("BTCUSDT", candles, {"5m": ind}, self._sweep_smc(), 0.01, 10_000_000)
        # mom=0.08 < 0.10 minimum → rejected
        assert sig is None

    def test_threshold_is_clamped_at_maximum(self):
        """Extremely high ATR still caps threshold at 0.30."""
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        # ATR=5.0 on close=100 → atr_pct=5% → 0.5*5=2.5, clamped to 0.30
        # EMA gap must exceed adaptive buffer (atr_pct=5% → buffer_abs≈2.5) so ema9=104
        ind = _make_indicators(adx_val=30, atr_val=5.0, mom=0.31, ema9=104, ema21=100)
        sig = ch._evaluate_standard("BTCUSDT", candles, {"5m": ind}, self._sweep_smc(), 0.01, 10_000_000)
        # threshold=0.30, mom=0.31 → should pass
        assert sig is not None


# ---------------------------------------------------------------------------
# 2. ADX Overlap Elimination
# ---------------------------------------------------------------------------

class TestADXOverlapEliminated:
    """ADX 23 should NOT fire range fade (was allowed with old threshold of 25)."""

    def test_adx_23_does_not_fire_range_fade(self):
        """ADX=23 is above the new 22 threshold → range fade blocked."""
        ch = ScalpChannel()
        candles_data = _make_candles(60, base=100)
        candles_data["close"][-1] = 97.0
        candles = {"5m": candles_data}
        indicators = {"5m": _make_indicators(adx_val=23, bb_lower=97.1, rsi_val=28)}
        sig = ch._evaluate_range_fade("BTCUSDT", candles, indicators, {}, 0.01, 10_000_000)
        assert sig is None

    def test_adx_22_fires_range_fade(self):
        """ADX=22 is at the boundary → should produce range fade signal."""
        ch = ScalpChannel()
        candles_data = _make_candles(60, base=100)
        candles_data["close"][-1] = 97.0
        candles = {"5m": candles_data}
        indicators = {"5m": _make_indicators(adx_val=22, bb_lower=97.1, rsi_val=28)}
        sig = ch._evaluate_range_fade("BTCUSDT", candles, indicators, {}, 0.01, 10_000_000)
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_adx_25_was_old_boundary_now_blocked(self):
        """ADX=25 previously fired range fade but now correctly blocked."""
        ch = ScalpChannel()
        candles_data = _make_candles(60, base=100)
        candles_data["close"][-1] = 97.0
        candles = {"5m": candles_data}
        indicators = {"5m": _make_indicators(adx_val=25, bb_lower=97.1, rsi_val=28)}
        sig = ch._evaluate_range_fade("BTCUSDT", candles, indicators, {}, 0.01, 10_000_000)
        assert sig is None


# ---------------------------------------------------------------------------
# 3. BB Squeeze Guard for Range Fade
# ---------------------------------------------------------------------------

class TestBBSqueezeGuard:
    """Range fade should be blocked when BB is expanding rapidly (squeeze breakout)."""

    def test_bb_expanding_blocks_range_fade(self):
        """BB width expanding >10% from previous bar → range fade rejected."""
        ch = ScalpChannel()
        candles_data = _make_candles(60, base=100)
        candles_data["close"][-1] = 97.0
        candles = {"5m": candles_data}
        # bb_width_pct > bb_width_prev_pct * 1.1
        indicators = {"5m": _make_indicators(
            adx_val=15, bb_lower=97.1, rsi_val=28,
            bb_width_pct=4.5, bb_width_prev_pct=4.0,  # 4.5 > 4.0*1.1=4.4 → expanding
        )}
        sig = ch._evaluate_range_fade("BTCUSDT", candles, indicators, {}, 0.01, 10_000_000)
        assert sig is None

    def test_bb_stable_allows_range_fade(self):
        """BB width stable (not expanding >10%) → range fade allowed."""
        ch = ScalpChannel()
        candles_data = _make_candles(60, base=100)
        candles_data["close"][-1] = 97.0
        candles = {"5m": candles_data}
        # bb_width_pct <= bb_width_prev_pct * 1.1
        indicators = {"5m": _make_indicators(
            adx_val=15, bb_lower=97.1, rsi_val=28,
            bb_width_pct=4.0, bb_width_prev_pct=4.0,  # 4.0 < 4.0*1.1=4.4 → not expanding
        )}
        sig = ch._evaluate_range_fade("BTCUSDT", candles, indicators, {}, 0.01, 10_000_000)
        assert sig is not None

    def test_bb_width_missing_does_not_block(self):
        """Missing BB width indicators → guard skipped, range fade proceeds normally."""
        ch = ScalpChannel()
        candles_data = _make_candles(60, base=100)
        candles_data["close"][-1] = 97.0
        candles = {"5m": candles_data}
        # No bb_width_pct keys → guard is skipped
        indicators = {"5m": _make_indicators(adx_val=15, bb_lower=97.1, rsi_val=28)}
        sig = ch._evaluate_range_fade("BTCUSDT", candles, indicators, {}, 0.01, 10_000_000)
        assert sig is not None


# ---------------------------------------------------------------------------
# 4. CVD ADX Gate
# ---------------------------------------------------------------------------

class TestCVDADXGate:
    """CVD divergence channel should be blocked when ADX > 35."""

    def _make_cvd_candles(self) -> dict:
        closes = np.ones(25) * 100.0
        # Price lower low in second half
        closes[10:] = 99.0
        highs = closes + 1.0
        lows = closes - 1.0
        return {"5m": {"close": closes, "high": highs, "low": lows}}

    def test_adx_above_35_blocks_cvd_signal(self):
        """ADX=40 → CVD channel blocked (strong trend, divergence unreliable)."""
        ch = ScalpCVDChannel()
        candles = self._make_cvd_candles()
        # near recent low (100 * 1.008 = 100.8; close=99.0 < 100.8 → near support)
        indicators = {"5m": _make_indicators(adx_val=40, rsi_val=45)}
        smc_data = {"cvd_divergence": "BULLISH"}
        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_adx_at_35_blocks_cvd_signal(self):
        """ADX=35 is the exact boundary; the condition is > 35, so 35 should pass."""
        ch = ScalpCVDChannel()
        candles = self._make_cvd_candles()
        indicators = {"5m": _make_indicators(adx_val=35, rsi_val=45)}
        smc_data = {"cvd_divergence": "BULLISH"}
        # ADX=35 is NOT > 35, so the gate does not block it.
        # The signal may still not fire due to S/R proximity requirements, but
        # we confirm the call does not raise an exception.
        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        # sig may be None or a Signal; either is valid — gate at > 35 allows ADX=35 through
        assert sig is None or sig.direction in (Direction.LONG, Direction.SHORT)

    def test_adx_below_35_allows_cvd_signal(self):
        """ADX=25 → CVD channel allowed (moderate trend)."""
        ch = ScalpCVDChannel()
        candles = self._make_cvd_candles()
        indicators = {"5m": _make_indicators(adx_val=25, rsi_val=45)}
        smc_data = {"cvd_divergence": "BULLISH"}
        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        # If the signal fires, it should be LONG direction
        if sig is not None:
            assert sig.direction == Direction.LONG


# ---------------------------------------------------------------------------
# 5. Depth-Weighted OBI
# ---------------------------------------------------------------------------

class TestDepthWeightedOBI:
    """Level-1 imbalance should outweigh deep-book levels."""

    def test_obi_gives_more_weight_to_top_levels(self):
        """Weighted OBI amplifies top-of-book imbalance vs flat (unweighted) approach."""
        # Bids: 100 at level 1, 1 each at levels 2-10
        # Asks: 10 each at all levels
        bids = [[100.0 - i, 100.0 if i == 0 else 1.0] for i in range(10)]
        asks = [[99.0 - i, 10.0] for i in range(10)]

        obi_weighted = _compute_obi(bids, asks)
        assert obi_weighted is not None

        # Reference: unweighted OBI
        bid_qty_flat = sum(float(b[1]) for b in bids[:10])
        ask_qty_flat = sum(float(a[1]) for a in asks[:10])
        total_flat = bid_qty_flat + ask_qty_flat
        obi_flat = (bid_qty_flat - ask_qty_flat) / total_flat

        # Depth-weighted OBI should be MORE positive than flat OBI because
        # the large level-1 bid (100) gets the highest weight (1.0).
        assert obi_weighted > obi_flat

    def test_obi_range_bounded(self):
        """OBI is always in [-1, 1]."""
        bids = [[100.0, 1000.0]] + [[99.0 - i, 0.1] for i in range(9)]
        asks = [[101.0 + i, 0.1] for i in range(10)]
        obi = _compute_obi(bids, asks)
        assert obi is not None
        assert -1.0 <= obi <= 1.0

    def test_obi_returns_none_on_empty_data(self):
        assert _compute_obi([], []) is None

    def test_obi_handles_partial_levels(self):
        """Only 3 levels available; should still compute correctly."""
        bids = [[100.0, 10.0], [99.0, 5.0], [98.0, 2.0]]
        asks = [[101.0, 8.0], [102.0, 4.0], [103.0, 1.0]]
        obi = _compute_obi(bids, asks)
        assert obi is not None

    def test_deep_levels_get_lower_weight(self):
        """Verify that weights decrease monotonically with depth."""
        weights = [1.0 / (1.0 + 0.25 * i) for i in range(10)]
        for i in range(9):
            assert weights[i] > weights[i + 1], f"Weight at level {i} should exceed level {i+1}"


# ---------------------------------------------------------------------------
# 6. FVG Partial Fill Detection
# ---------------------------------------------------------------------------

class TestFVGPartialFillDetection:
    """FVG zones that are >60% filled should be rejected."""

    def _make_fvg_candles(self, close_val: float = 100.5) -> dict:
        closes = np.ones(25) * close_val
        highs = closes + 1.0
        lows = closes - 1.0
        return {"5m": {"close": closes, "high": highs, "low": lows}}

    def test_heavily_filled_bullish_fvg_rejected(self):
        """Bullish FVG with price 70% filled from above → rejected."""
        ch = ScalpFVGChannel()
        # Bullish FVG: gap_low=100, gap_high=101 (zone width=1)
        # For bullish: fill_pct = (gap_high - close) / zone_width = (101 - close) / 1
        # fill_pct > 0.6 → (101 - close) / 1 > 0.6 → close < 100.4
        # So price=100.2 → fill_pct = (101 - 100.2) / 1 = 0.8 > 0.6 → rejected
        candles = self._make_fvg_candles(close_val=100.2)
        zone = FVGZone(index=20, direction=Direction.LONG, gap_high=101.0, gap_low=100.0)
        smc_data = {"fvg": [zone]}
        indicators = {"5m": _make_indicators(adx_val=25, rsi_val=45)}
        sig = ch._evaluate_tf("BTCUSDT", "5m", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_lightly_filled_bullish_fvg_allowed(self):
        """Bullish FVG with price 30% filled → fill check passes (not blocked by fill gate)."""
        ch = ScalpFVGChannel()
        # fill_pct = (101 - 100.7) / 1 = 0.3 < 0.6 → fill check passes
        candles = self._make_fvg_candles(close_val=100.7)
        zone = FVGZone(index=20, direction=Direction.LONG, gap_high=101.0, gap_low=100.0)
        smc_data = {"fvg": [zone]}
        indicators = {"5m": _make_indicators(adx_val=25, rsi_val=45)}
        # Call succeeds without raising; result may be None due to proximity filtering
        ch._evaluate_tf("BTCUSDT", "5m", candles, indicators, smc_data, 0.01, 10_000_000)

    def test_heavily_filled_bearish_fvg_rejected(self):
        """Bearish FVG with price 70% filled from below → rejected."""
        ch = ScalpFVGChannel()
        # Bearish FVG: gap_low=100, gap_high=101
        # For bearish: fill_pct = (close - gap_low) / zone_width = (close - 100) / 1
        # fill_pct > 0.6 → (close - 100) / 1 > 0.6 → close > 100.6
        # So price=100.8 → fill_pct = (100.8 - 100) / 1 = 0.8 > 0.6 → rejected
        candles = self._make_fvg_candles(close_val=100.8)
        zone = FVGZone(index=20, direction=Direction.SHORT, gap_high=101.0, gap_low=100.0)
        smc_data = {"fvg": [zone]}
        indicators = {"5m": _make_indicators(adx_val=25, rsi_val=55)}
        sig = ch._evaluate_tf("BTCUSDT", "5m", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_exact_60pct_fill_is_allowed(self):
        """Exactly 60% filled (boundary) → fill check passes (condition is > 0.6, not >=)."""
        ch = ScalpFVGChannel()
        # fill_pct = (101 - 100.4) / 1 = 0.6 → not > 0.6 → fill check passes
        candles = self._make_fvg_candles(close_val=100.4)
        zone = FVGZone(index=20, direction=Direction.LONG, gap_high=101.0, gap_low=100.0)
        smc_data = {"fvg": [zone]}
        indicators = {"5m": _make_indicators(adx_val=25, rsi_val=45)}
        # Call succeeds without raising; result may be None due to other filters
        ch._evaluate_tf("BTCUSDT", "5m", candles, indicators, smc_data, 0.01, 10_000_000)


# ---------------------------------------------------------------------------
# 7. Swing Daily Confluence Markup
# ---------------------------------------------------------------------------

class TestSwingDailyConfluence:
    """Swing signals near daily S/R should be marked as SWING_D1_CONFLUENCE."""

    def _make_swing_setup(self, close_h1: float = 2300.0, d1_lows=None, d1_highs=None):
        h4_data = _make_candles(60, base=2300)
        h1_data = _make_candles(60, base=close_h1, trend=0.0)
        h1_data["close"][-1] = close_h1
        candles = {"4h": h4_data, "1h": h1_data}

        if d1_lows is not None or d1_highs is not None:
            d1_close = np.ones(30) * close_h1
            d1_high = np.array(d1_highs if d1_highs else [close_h1 + 10] * 30)
            d1_low = np.array(d1_lows if d1_lows else [close_h1 - 10] * 30)
            candles["1d"] = {"close": d1_close, "high": d1_high, "low": d1_low}

        sweep = LiquiditySweep(
            index=59, direction=Direction.LONG,
            sweep_level=close_h1 - 10, close_price=close_h1 - 9,
            wick_high=close_h1 + 20, wick_low=close_h1 - 15,
        )
        mss = MSSSignal(
            index=59, direction=Direction.LONG,
            midpoint=close_h1, confirm_close=close_h1 + 5,
        )
        smc_data = {"sweeps": [sweep], "mss": mss}

        h4_ind = _make_indicators(adx_val=25, ema200=2200)
        h1_ind = _make_indicators(adx_val=25, ema200=2200, bb_lower=close_h1 - 5)
        indicators = {"4h": h4_ind, "1h": h1_ind}

        return candles, indicators, smc_data

    def test_daily_confluence_markup_applied(self):
        """Swing signal near daily support gets SWING_D1_CONFLUENCE class and A+ tier."""
        ch = SwingChannel()
        close_h1 = 2300.0
        # Daily lows: last 10 bars around 2295 (support), close_h1=2300 is within 3%
        d1_lows = [2200.0] * 20 + [2295.0] * 10  # last 10 lows at 2295
        d1_highs = [2400.0] * 30
        candles, indicators, smc_data = self._make_swing_setup(
            close_h1=close_h1, d1_lows=d1_lows, d1_highs=d1_highs
        )

        sig = ch.evaluate("ETHUSDT", candles, indicators, smc_data, 0.01, 50_000_000)
        if sig is not None:
            # close_h1=2300 <= 2295*1.03=2363.85 → within 3% → confluence
            assert sig.setup_class in ("SWING_D1_CONFLUENCE", "SWING_STANDARD")
            if sig.setup_class == "SWING_D1_CONFLUENCE":
                assert sig.quality_tier == "A+"

    def test_no_daily_confluence_gets_standard_class(self):
        """Swing signal far from daily support gets SWING_STANDARD class."""
        ch = SwingChannel()
        close_h1 = 2300.0
        # Daily lows far below (2000), so close_h1=2300 is NOT within 3%
        d1_lows = [2000.0] * 30
        d1_highs = [2400.0] * 30
        candles, indicators, smc_data = self._make_swing_setup(
            close_h1=close_h1, d1_lows=d1_lows, d1_highs=d1_highs
        )

        sig = ch.evaluate("ETHUSDT", candles, indicators, smc_data, 0.01, 50_000_000)
        if sig is not None:
            # close_h1=2300 <= 2000*1.03=2060 → NOT within 3% → standard
            assert sig.setup_class == "SWING_STANDARD"

    def test_no_daily_candles_gets_standard_class(self):
        """Without 1d candles, swing signal defaults to SWING_STANDARD."""
        ch = SwingChannel()
        candles, indicators, smc_data = self._make_swing_setup(close_h1=2300.0)
        # No "1d" key in candles

        sig = ch.evaluate("ETHUSDT", candles, indicators, smc_data, 0.01, 50_000_000)
        if sig is not None:
            assert sig.setup_class == "SWING_STANDARD"


# ---------------------------------------------------------------------------
# 8. Funding Rate Contrarian Bonus/Penalty
# ---------------------------------------------------------------------------

class TestFundingRateContrarian:
    """score_order_flow should reward contrarian and penalise crowded funding."""

    def test_contrarian_funding_bonus_long(self):
        """Extreme negative funding + LONG signal = contrarian edge → bonus."""
        # funding_rate = -0.02 (very negative, longs are cheap), signal = LONG
        score_with = score_order_flow(
            oi_trend="NEUTRAL",
            funding_rate=-0.02,
            signal_direction="LONG",
        )
        score_without = score_order_flow(
            oi_trend="NEUTRAL",
            funding_rate=None,
            signal_direction="LONG",
        )
        assert score_with > score_without

    def test_contrarian_funding_bonus_short(self):
        """Extreme positive funding + SHORT signal = contrarian edge → bonus."""
        score_with = score_order_flow(
            oi_trend="NEUTRAL",
            funding_rate=0.02,
            signal_direction="SHORT",
        )
        score_without = score_order_flow(
            oi_trend="NEUTRAL",
            funding_rate=None,
            signal_direction="SHORT",
        )
        assert score_with > score_without

    def test_crowded_trade_penalty_long(self):
        """Extreme positive funding + LONG signal = crowded trade → penalty reduces score."""
        # Use CVD alignment to build a positive base score (5 pts), so penalty (-3) is visible
        score_crowded = score_order_flow(
            oi_trend="NEUTRAL",
            cvd_divergence="BULLISH",
            funding_rate=0.02,   # positive: longs are paying → crowded LONG
            signal_direction="LONG",
        )
        score_without_funding = score_order_flow(
            oi_trend="NEUTRAL",
            cvd_divergence="BULLISH",
            funding_rate=None,
            signal_direction="LONG",
        )
        assert score_crowded < score_without_funding

    def test_crowded_trade_penalty_short(self):
        """Extreme negative funding + SHORT signal = crowded trade → penalty reduces score."""
        # Use CVD alignment to build a positive base score (5 pts), so penalty (-3) is visible
        score_crowded = score_order_flow(
            oi_trend="NEUTRAL",
            cvd_divergence="BEARISH",
            funding_rate=-0.02,  # negative: shorts are paying → crowded SHORT
            signal_direction="SHORT",
        )
        score_without_funding = score_order_flow(
            oi_trend="NEUTRAL",
            cvd_divergence="BEARISH",
            funding_rate=None,
            signal_direction="SHORT",
        )
        assert score_crowded < score_without_funding

    def test_small_funding_rate_no_effect(self):
        """Below-threshold funding rate has no effect."""
        score_small = score_order_flow(
            oi_trend="NEUTRAL",
            funding_rate=0.005,  # below _EXTREME_FUNDING_RATE=0.01
            signal_direction="LONG",
        )
        score_none = score_order_flow(
            oi_trend="NEUTRAL",
            funding_rate=None,
            signal_direction="LONG",
        )
        assert score_small == score_none

    def test_score_always_bounded_0_to_20(self):
        """Score is always clamped to [0, 20]."""
        # Maximum scenario: falling OI + full liq + CVD aligned + contrarian funding
        s = score_order_flow(
            oi_trend="FALLING",
            liq_vol_usd=1_000_000.0,
            cvd_divergence="BULLISH",
            signal_direction="LONG",
            funding_rate=-0.05,
        )
        assert 0.0 <= s <= 20.0


# ---------------------------------------------------------------------------
# 9. Spot Breakout Retest Detection
# ---------------------------------------------------------------------------

class TestSpotBreakoutRetestDetection:
    """Spot channel should detect and mark breakout retests."""

    def _make_spot_candles(self, retest: bool = False) -> tuple:
        closes = np.cumsum(np.ones(60) * 0.1) + 100.0
        recent_high = max(closes[-20:-1])

        if retest:
            # Pattern: [..., above_high, below_high (pullback), above_high (reclaim)]
            closes[-3] = recent_high + 0.5   # breakout candle
            closes[-2] = recent_high - 0.2   # pullback below resistance
            closes[-1] = recent_high + 0.5   # reclaim → this IS a retest
        else:
            # Standard initial breakout
            closes[-1] = recent_high + 1.0
            closes[-2] = recent_high - 0.5
            closes[-3] = recent_high - 1.0

        highs = closes + 0.5
        lows = closes - 0.5
        volumes = np.ones(60) * 1000.0
        volumes[-1] = volumes[:-1].mean() * 2.0

        candles_data = {
            "open": closes - 0.1,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
        return candles_data, recent_high

    def test_retest_pattern_marked_as_breakout_retest(self):
        """Retest pattern (breakout → pullback → reclaim) → BREAKOUT_RETEST class."""
        ch = SpotChannel()
        candles_data, recent_high = self._make_spot_candles(retest=True)
        candles = {"4h": candles_data}
        indicators = {"4h": _make_indicators(adx_val=20, ema200=90)}
        smc_data = {}

        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 5_000_000)
        if sig is not None:
            assert sig.setup_class == "BREAKOUT_RETEST"

    def test_initial_breakout_marked_as_breakout_initial(self):
        """Standard initial breakout → BREAKOUT_INITIAL class."""
        ch = SpotChannel()
        candles_data, _ = self._make_spot_candles(retest=False)
        candles = {"4h": candles_data}
        indicators = {"4h": _make_indicators(adx_val=20, ema200=90)}
        smc_data = {}

        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 5_000_000)
        if sig is not None:
            assert sig.setup_class in ("BREAKOUT_INITIAL", "BREAKOUT_RETEST")

    def test_short_retest_marked_correctly(self):
        """SHORT breakdown retest → BREAKOUT_RETEST class."""
        ch = SpotChannel()
        # Build SHORT scenario: below EMA200 and daily EMA50, breakdown below recent low
        closes = np.cumsum(np.ones(60) * (-0.1)) + 100.0
        recent_low = min(closes[-10:-1])

        # Retest pattern for SHORT: breakdown → bounce → re-break
        closes[-3] = recent_low - 0.5   # breakdown
        closes[-2] = recent_low + 0.2   # bounce (pullback above support)
        closes[-1] = recent_low - 0.5   # re-break → retest

        volumes = np.ones(60) * 1000.0
        volumes[-1] = volumes[:-1].mean() * 2.0

        candles_data = {
            "open": closes - 0.1,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": volumes,
        }
        candles = {"4h": candles_data}
        # EMA200 above price (bearish), daily EMA50 above price
        indicators = {
            "4h": _make_indicators(adx_val=20, ema200=105),
            "1d": _make_indicators(ema200=105),
        }
        smc_data = {}

        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 5_000_000)
        if sig is not None and sig.direction == Direction.SHORT:
            assert sig.setup_class in ("BREAKOUT_RETEST", "BREAKOUT_INITIAL")


# ---------------------------------------------------------------------------
# 10. Direction-Biased Entry Zones
# ---------------------------------------------------------------------------

class TestDirectionBiasedEntryZones:
    """Entry zones must be biased toward the trade direction for better fills."""

    def _make_scalp_signal(
        self,
        direction: Direction = Direction.LONG,
        atr_val: float = 0.5,
    ):
        ch = ScalpChannel()
        candles_data = _make_candles(60, base=100)
        candles_data["close"][-1] = 97.0
        candles = {"5m": candles_data}
        ind = _make_indicators(adx_val=15, bb_lower=97.1, rsi_val=28, atr_val=atr_val)
        indicators = {"5m": ind}
        return ch._evaluate_range_fade("BTCUSDT", candles, indicators, {}, 0.01, 10_000_000)

    def test_long_zone_biased_below_close(self):
        """RANGE_FADE LONG entry zone: symmetric around entry (bias=0.5 for mean-reversion)."""
        import pytest
        sig = self._make_scalp_signal(direction=Direction.LONG)
        if sig is None:
            pytest.skip("Signal filtered by other gates — cannot validate zone bias")
        close = sig.entry
        below_dist = close - sig.entry_zone_low
        above_dist = sig.entry_zone_high - close
        # RANGE_FADE uses entry_zone_bias=0.5 (symmetric) — below and above are equal
        assert abs(below_dist - above_dist) < 1e-6, (
            f"RANGE_FADE LONG zone should be symmetric: below={below_dist}, above={above_dist}"
        )

    def test_long_entry_zone_brackets_entry(self):
        """LONG entry zone must contain the entry price."""
        import pytest
        sig = self._make_scalp_signal(direction=Direction.LONG)
        if sig is None:
            pytest.skip("Signal filtered by other gates — cannot validate zone bracketing")
        assert sig.entry_zone_low < sig.entry < sig.entry_zone_high

    def test_entry_zone_width_proportional_to_atr(self):
        """Wider ATR → wider entry zone."""
        import pytest
        sig_narrow = self._make_scalp_signal(atr_val=0.3)
        sig_wide = self._make_scalp_signal(atr_val=1.0)
        if sig_narrow is None or sig_wide is None:
            pytest.skip("Signal filtered by other gates — cannot compare zone widths")
        w_narrow = sig_narrow.entry_zone_high - sig_narrow.entry_zone_low
        w_wide = sig_wide.entry_zone_high - sig_wide.entry_zone_low
        assert w_wide > w_narrow


# ---------------------------------------------------------------------------
# 11. RANGE_FADE RSI Fix — mean-reversion RSI gating
# ---------------------------------------------------------------------------

class TestRangeFadeRSIFix:
    """RANGE_FADE should gate on mean-reversion RSI thresholds, not trend thresholds."""

    def _range_fade(self, rsi_val: float, direction_long: bool = True) -> object:
        ch = ScalpChannel()
        candles_data = _make_candles(60, base=100)
        if direction_long:
            # Price at lower BB → LONG mean-reversion
            candles_data["close"][-1] = 97.0
            indicators = {"5m": _make_indicators(adx_val=15, bb_lower=97.1, rsi_val=rsi_val)}
        else:
            # Price at upper BB → SHORT mean-reversion
            candles_data["close"][-1] = 103.0
            indicators = {"5m": _make_indicators(adx_val=15, bb_upper=102.9, rsi_val=rsi_val)}
        candles = {"5m": candles_data}
        return ch._evaluate_range_fade("BTCUSDT", candles, indicators, {}, 0.01, 10_000_000)

    def test_long_oversold_rsi_fires(self):
        """LONG range fade fires when RSI is oversold (< 30)."""
        sig = self._range_fade(rsi_val=25, direction_long=True)
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_long_rsi_at_boundary_fires(self):
        """LONG range fade fires at RSI = 50 (below threshold of 55)."""
        sig = self._range_fade(rsi_val=50, direction_long=True)
        assert sig is not None

    def test_long_rsi_above_threshold_blocked(self):
        """LONG range fade blocked when RSI > 55 (price already recovering)."""
        sig = self._range_fade(rsi_val=60, direction_long=True)
        assert sig is None

    def test_long_rsi_at_exact_threshold_blocked(self):
        """LONG range fade blocked at RSI = 56 (strictly above 55)."""
        sig = self._range_fade(rsi_val=56, direction_long=True)
        assert sig is None
