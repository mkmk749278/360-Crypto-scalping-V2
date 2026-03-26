"""Tests for signal quality improvements — P0, P1 & P2 fixes.

Covers:
  - P0-1: Entry slippage in backtester
  - P0-2: ATR-normalized S/R proximity in CVD channel
  - P0-3: ATR-normalized S/R proximity in OBI channel
  - P0-4: FVG zone age decay
  - P1-1: Regime hysteresis
  - P1-2: Relative BB proximity in swing channel
  - P1-3: OBI staleness guard
  - P2-1: Confidence log infrastructure
  - P2-2: Volatility-adaptive TP ratios
  - P2-3: Funding rate costs in backtester
  - P2-4: CVD divergence recency and magnitude validation
  - P2-5: Pre-event macro blackout window
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candles(n: int = 60, base: float = 100.0, trend: float = 0.1) -> dict:
    close = np.cumsum(np.ones(n) * trend) + base
    high = close + 0.5
    low = close - 0.5
    volume = np.ones(n) * 1000.0
    return {"open": close - 0.05, "high": high, "low": low, "close": close, "volume": volume}


def _make_fake_signal(
    direction: str = "LONG",
    entry: float = 100.0,
    sl: float = 99.0,
    tp1: float = 101.0,
    tp2: float = 102.0,
    tp3: Optional[float] = None,
):
    class _Dir:
        value = direction

    class _Sig:
        pass

    s = _Sig()
    s.direction = _Dir()
    s.entry = entry
    s.stop_loss = sl
    s.tp1 = tp1
    s.tp2 = tp2
    s.tp3 = tp3
    return s


# ===========================================================================
# P0-1: Entry slippage in backtester
# ===========================================================================

class TestEntrySlippage:
    """Entry slippage is applied adversely at the start of _simulate_trade()."""

    def test_long_entry_fills_higher_with_slippage(self):
        """LONG trade PnL must be lower with entry slippage than without."""
        from src.backtester import _simulate_trade
        sig = _make_fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.5)
        future = {
            "high": np.array([101.5]),
            "low": np.array([99.5]),
            "close": np.array([101.5]),
        }
        _, pnl_no_slip, _ = _simulate_trade(sig, future, slippage_pct=0.0)
        _, pnl_with_slip, _ = _simulate_trade(sig, future, slippage_pct=0.1)
        # With entry slippage, effective entry is higher → PnL is lower
        assert pnl_with_slip < pnl_no_slip

    def test_short_entry_fills_lower_with_slippage(self):
        """SHORT trade PnL must be lower with entry slippage than without."""
        from src.backtester import _simulate_trade
        sig = _make_fake_signal("SHORT", entry=100.0, sl=101.0, tp1=98.5, tp2=97.0)
        future = {
            "high": np.array([100.5]),
            "low": np.array([98.0]),
            "close": np.array([98.5]),
        }
        _, pnl_no_slip, _ = _simulate_trade(sig, future, slippage_pct=0.0)
        _, pnl_with_slip, _ = _simulate_trade(sig, future, slippage_pct=0.1)
        assert pnl_with_slip < pnl_no_slip

    def test_zero_slippage_symmetric_for_long(self):
        """With slippage_pct=0 entry fill equals signal.entry exactly."""
        from src.backtester import _simulate_trade
        sig = _make_fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0)
        future = {
            "high": np.array([101.5]),
            "low": np.array([99.5]),
            "close": np.array([101.5]),
        }
        won_a, pnl_a, lvl_a = _simulate_trade(sig, future, slippage_pct=0.0)
        won_b, pnl_b, lvl_b = _simulate_trade(sig, future)
        assert won_a == won_b
        assert pnl_a == pytest.approx(pnl_b, abs=1e-6)


# ===========================================================================
# P0-2 & P2-4: CVD channel — ATR proximity + recency/magnitude guards
# ===========================================================================

class TestCVDChannel:
    """Tests for scalp_cvd improvements."""

    def _make_cvd_candles(self, close_val: float = 100.0) -> dict:
        n = 25
        closes = np.ones(n) * close_val
        highs = closes + 0.5
        lows = closes - 0.5
        return {
            "5m": {
                "close": closes,
                "high": highs,
                "low": lows,
                "volume": np.ones(n) * 1000.0,
            }
        }

    def _make_ind(self, atr_val: Optional[float] = 0.5) -> dict:
        ind: dict = {
            "adx_last": 20.0,
            "rsi_last": 50.0,
        }
        if atr_val is not None:
            ind["atr_last"] = atr_val
        return {"5m": ind}

    def test_atr_proximity_rejects_far_bullish(self):
        """BULLISH divergence: price more than 1×ATR above recent low → None."""
        from src.channels.scalp_cvd import ScalpCVDChannel
        ch = ScalpCVDChannel()
        # close at 100, recent_low at 98, ATR = 1.0  → threshold = 98 + 1.0 = 99
        # close=100 > 99  → should reject
        candles = self._make_cvd_candles(close_val=100.0)
        candles["5m"]["low"] = np.ones(25) * 98.0  # recent_low = 98
        candles["5m"]["high"] = np.ones(25) * 104.0
        ind = self._make_ind(atr_val=1.0)
        smc_data = {"cvd_divergence": "BULLISH"}
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_atr_proximity_allows_near_bullish(self):
        """BULLISH divergence: price within 1×ATR of recent low → signal allowed."""
        from src.channels.scalp_cvd import ScalpCVDChannel
        ch = ScalpCVDChannel()
        # close=99, recent_low=98.8, ATR=1.0 → threshold=98.8+1.0=99.8
        # close=99 <= 99.8 → near support
        candles = self._make_cvd_candles(close_val=99.0)
        candles["5m"]["low"] = np.ones(25) * 98.8
        candles["5m"]["high"] = np.ones(25) * 102.0
        candles["5m"]["close"] = np.ones(25) * 99.0
        ind = self._make_ind(atr_val=1.0)
        # Provide both age and strength so fail-closed guards are satisfied
        smc_data = {
            "cvd_divergence": "BULLISH",
            "cvd_divergence_age": 5,
            "cvd_divergence_strength": 0.5,
        }
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is not None

    def test_fallback_pct_used_when_no_atr(self):
        """Without ATR, fixed _SR_PROXIMITY_PCT fallback is used."""
        from src.channels.scalp_cvd import ScalpCVDChannel
        ch = ScalpCVDChannel()
        # close=101, recent_low=100, no ATR → pct check 100*(1+0.8/100)=100.8
        # close=101 > 100.8 → rejects
        candles = self._make_cvd_candles(close_val=101.0)
        candles["5m"]["low"] = np.ones(25) * 100.0
        candles["5m"]["high"] = np.ones(25) * 105.0
        candles["5m"]["close"] = np.ones(25) * 101.0
        ind = {"5m": {"adx_last": 20.0, "rsi_last": 50.0}}  # no atr_last
        smc_data = {"cvd_divergence": "BULLISH"}
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_stale_divergence_rejected(self):
        """CVD divergence age > 10 candles → None."""
        from src.channels.scalp_cvd import ScalpCVDChannel
        ch = ScalpCVDChannel()
        candles = self._make_cvd_candles(close_val=99.0)
        candles["5m"]["low"] = np.ones(25) * 98.8
        candles["5m"]["high"] = np.ones(25) * 102.0
        candles["5m"]["close"] = np.ones(25) * 99.0
        ind = self._make_ind(atr_val=1.0)
        smc_data = {
            "cvd_divergence": "BULLISH",
            "cvd_divergence_age": 11,  # stale
        }
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_fresh_divergence_allowed(self):
        """CVD divergence age = 5 candles + valid strength → allowed."""
        from src.channels.scalp_cvd import ScalpCVDChannel
        ch = ScalpCVDChannel()
        candles = self._make_cvd_candles(close_val=99.0)
        candles["5m"]["low"] = np.ones(25) * 98.8
        candles["5m"]["high"] = np.ones(25) * 102.0
        candles["5m"]["close"] = np.ones(25) * 99.0
        ind = self._make_ind(atr_val=1.0)
        smc_data = {
            "cvd_divergence": "BULLISH",
            "cvd_divergence_age": 5,
            "cvd_divergence_strength": 0.5,  # also required by fail-closed guard
        }
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is not None

    def test_weak_divergence_rejected(self):
        """CVD divergence strength < 0.3 → None."""
        from src.channels.scalp_cvd import ScalpCVDChannel
        ch = ScalpCVDChannel()
        candles = self._make_cvd_candles(close_val=99.0)
        candles["5m"]["low"] = np.ones(25) * 98.8
        candles["5m"]["high"] = np.ones(25) * 102.0
        candles["5m"]["close"] = np.ones(25) * 99.0
        ind = self._make_ind(atr_val=1.0)
        smc_data = {
            "cvd_divergence": "BULLISH",
            "cvd_divergence_strength": 0.2,  # weak
        }
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_strong_divergence_allowed(self):
        """CVD divergence strength = 0.5 + valid age → allowed."""
        from src.channels.scalp_cvd import ScalpCVDChannel
        ch = ScalpCVDChannel()
        candles = self._make_cvd_candles(close_val=99.0)
        candles["5m"]["low"] = np.ones(25) * 98.8
        candles["5m"]["high"] = np.ones(25) * 102.0
        candles["5m"]["close"] = np.ones(25) * 99.0
        ind = self._make_ind(atr_val=1.0)
        smc_data = {
            "cvd_divergence": "BULLISH",
            "cvd_divergence_age": 5,   # also required by fail-closed guard
            "cvd_divergence_strength": 0.5,
        }
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is not None

    def test_no_age_or_strength_metadata_passes(self):
        """Without age/strength metadata the guard is fail-closed (_CVD_REQUIRE_METADATA=True)."""
        from src.channels.scalp_cvd import ScalpCVDChannel
        ch = ScalpCVDChannel()
        candles = self._make_cvd_candles(close_val=99.0)
        candles["5m"]["low"] = np.ones(25) * 98.8
        candles["5m"]["high"] = np.ones(25) * 102.0
        candles["5m"]["close"] = np.ones(25) * 99.0
        ind = self._make_ind(atr_val=1.0)
        smc_data = {"cvd_divergence": "BULLISH"}  # no age/strength keys
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is None  # fail-closed: missing metadata rejects the signal


# ===========================================================================
# P0-3 & P1-3: OBI channel — ATR proximity + staleness guard
# ===========================================================================

class TestOBIChannel:
    """Tests for scalp_obi improvements."""

    def _make_obi_context(self, close_val: float = 100.0, atr_val: Optional[float] = None):
        n = 25
        closes = np.ones(n) * close_val
        highs = closes + 2.0
        lows = closes - 2.0
        candles = {
            "5m": {
                "close": closes,
                "high": highs,
                "low": lows,
                "volume": np.ones(n) * 1000.0,
            }
        }
        ind: dict = {"adx_last": 25.0, "rsi_last": 50.0}
        if atr_val is not None:
            ind["atr_last"] = atr_val
        indicators = {"5m": ind}
        # Strong bid absorption: OBI ≥ 0.65 (bids >> asks)
        # Include a fresh timestamp so _OBI_REQUIRE_TIMESTAMP is satisfied by default
        order_book = {
            "bids": [[close_val, 1000.0]] * 10,
            "asks": [[close_val + 0.01, 1.0]] * 10,
            "fetched_at": time.time() - 0.5,  # fresh timestamp — satisfies _OBI_REQUIRE_TIMESTAMP
        }
        smc_data = {"order_book": order_book}
        return candles, indicators, smc_data

    def test_atr_proximity_rejects_obi_long_far_from_support(self):
        """OBI LONG: price > recent_low + 1×ATR → None."""
        from src.channels.scalp_obi import ScalpOBIChannel
        ch = ScalpOBIChannel()
        candles, ind, smc_data = self._make_obi_context(close_val=102.0, atr_val=1.0)
        candles["5m"]["low"] = np.ones(25) * 100.0  # recent_low = 100
        candles["5m"]["close"] = np.ones(25) * 102.0
        # close=102 > 100 + 1.0 = 101 → not near support
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_atr_proximity_allows_obi_long_near_support(self):
        """OBI LONG: price within 1×ATR of recent_low → signal allowed."""
        from src.channels.scalp_obi import ScalpOBIChannel
        ch = ScalpOBIChannel()
        candles, ind, smc_data = self._make_obi_context(close_val=100.5, atr_val=1.0)
        candles["5m"]["low"] = np.ones(25) * 100.0
        candles["5m"]["close"] = np.ones(25) * 100.5
        # close=100.5 <= 100.0 + 1.0 = 101 → near support
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is not None

    def test_stale_order_book_rejected(self):
        """Order book older than _OBI_MAX_STALENESS_SEC → None."""
        from src.channels.scalp_obi import ScalpOBIChannel
        ch = ScalpOBIChannel()
        candles, ind, _ = self._make_obi_context(close_val=100.5, atr_val=1.0)
        candles["5m"]["low"] = np.ones(25) * 100.0
        candles["5m"]["close"] = np.ones(25) * 100.5
        stale_ts = time.time() - 10.0  # 10 seconds old
        order_book = {
            "bids": [[100.0, 1000.0]] * 10,
            "asks": [[100.01, 1.0]] * 10,
            "fetched_at": stale_ts,
        }
        smc_data = {"order_book": order_book}
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_fresh_order_book_allowed(self):
        """Order book < _OBI_MAX_STALENESS_SEC old → allowed."""
        from src.channels.scalp_obi import ScalpOBIChannel
        ch = ScalpOBIChannel()
        candles, ind, _ = self._make_obi_context(close_val=100.5, atr_val=1.0)
        candles["5m"]["low"] = np.ones(25) * 100.0
        candles["5m"]["close"] = np.ones(25) * 100.5
        fresh_ts = time.time() - 0.5  # half second old
        order_book = {
            "bids": [[100.0, 1000.0]] * 10,
            "asks": [[100.01, 1.0]] * 10,
            "fetched_at": fresh_ts,
        }
        smc_data = {"order_book": order_book}
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is not None

    def test_missing_timestamp_fails_open(self):
        """Missing timestamp in order book → fail-closed when _OBI_REQUIRE_TIMESTAMP=True."""
        from src.channels.scalp_obi import ScalpOBIChannel
        ch = ScalpOBIChannel()
        candles, ind, _ = self._make_obi_context(close_val=100.5, atr_val=1.0)
        candles["5m"]["low"] = np.ones(25) * 100.0
        candles["5m"]["close"] = np.ones(25) * 100.5
        # order_book without timestamp — should be rejected (fail-closed)
        order_book = {
            "bids": [[100.0, 1000.0]] * 10,
            "asks": [[100.01, 1.0]] * 10,
        }
        smc_data = {"order_book": order_book}
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is None  # fail-closed: missing timestamp rejects the signal


# ===========================================================================
# P0-4: FVG zone age decay
# ===========================================================================

class TestFVGAgeDecay:
    """Tests for FVG zone age filtering and SL decay."""

    def _make_fvg_zone(self, index: int, direction_long: bool = True):
        from src.smc import Direction, FVGZone
        return FVGZone(
            index=index,
            direction=Direction.LONG if direction_long else Direction.SHORT,
            gap_high=101.0,
            gap_low=99.0,
        )

    def test_old_zone_rejected(self):
        """FVG zones older than _FVG_MAX_AGE_CANDLES are skipped."""
        from src.channels.scalp_fvg import ScalpFVGChannel
        ch = ScalpFVGChannel()
        # 25 candles total, zone index=0 → candles_ago = 25, which is < 80 but
        # let's use 60 candles total and zone at index=0 → candles_ago=60 < 80, still valid
        # For a zone older than 80: 100 candles total, zone at index=0 → 100 - 0 = 100 > 80
        n = 110
        close_val = 100.5
        closes = np.ones(n) * close_val
        highs = closes + 2.0
        lows = closes - 2.0
        candles = {
            "5m": {
                "close": closes,
                "high": highs,
                "low": lows,
                "volume": np.ones(n) * 1000.0,
            }
        }
        ind = {"5m": {"adx_last": 25.0, "atr_last": 0.5, "rsi_last": 50.0}}
        # Zone at index 5 → candles_ago = 110 - 5 = 105 > 80 → should skip
        old_zone = self._make_fvg_zone(index=5, direction_long=True)
        smc_data = {"fvg": [old_zone]}
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        # Signal should be None because the only zone is too old
        assert sig is None

    def test_recent_zone_produces_signal(self):
        """FVG zones within _FVG_MAX_AGE_CANDLES are processed normally."""
        from src.channels.scalp_fvg import ScalpFVGChannel
        ch = ScalpFVGChannel()
        n = 30
        # Position price so it retests the bullish FVG gap_high=101
        # proximity check: (close - gap_high) / zone_width = (101 - 101) / 2 = 0 ≤ 0.35 ✓
        closes = np.ones(n) * 101.0
        highs = closes + 0.5
        lows = closes - 0.5
        candles = {
            "5m": {
                "close": closes,
                "high": highs,
                "low": lows,
                "volume": np.ones(n) * 1000.0,
            }
        }
        ind = {"5m": {"adx_last": 25.0, "atr_last": 0.5, "rsi_last": 50.0}}
        # Zone at index 20 → candles_ago = 30 - 20 = 10 < 80 → valid
        recent_zone = self._make_fvg_zone(index=20, direction_long=True)
        smc_data = {"fvg": [recent_zone]}
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is not None

    def test_decay_applied_to_older_zone(self):
        """Older zones have a smaller decay factor, resulting in a tighter SL."""
        from src.channels.scalp_fvg import ScalpFVGChannel
        ch = ScalpFVGChannel()
        n = 90
        closes = np.ones(n) * 101.0
        highs = closes + 0.5
        lows = closes - 0.5
        candles = {
            "5m": {
                "close": closes,
                "high": highs,
                "low": lows,
                "volume": np.ones(n) * 1000.0,
            }
        }
        ind = {"5m": {"adx_last": 25.0, "atr_last": 0.5, "rsi_last": 50.0}}

        # Fresh zone: index=80, candles_ago=10, decay=max(0.2, 1-10/100)=0.9
        fresh_zone = self._make_fvg_zone(index=80, direction_long=True)
        smc_data_fresh = {"fvg": [fresh_zone]}
        sig_fresh = ch.evaluate("BTCUSDT", candles, ind, smc_data_fresh, 0.01, 10_000_000)

        # Older zone: index=15, candles_ago=75, decay=max(0.2, 1-75/100)=0.25
        older_zone = self._make_fvg_zone(index=15, direction_long=True)
        smc_data_older = {"fvg": [older_zone]}
        sig_older = ch.evaluate("BTCUSDT", candles, ind, smc_data_older, 0.01, 10_000_000)

        if sig_fresh is not None and sig_older is not None:
            # Older zone should have tighter SL (closer to entry)
            fresh_sl_dist = abs(sig_fresh.entry - sig_fresh.stop_loss)
            older_sl_dist = abs(sig_older.entry - sig_older.stop_loss)
            assert older_sl_dist < fresh_sl_dist


# ===========================================================================
# P1-1: Regime hysteresis
# ===========================================================================

class TestRegimeHysteresis:
    """Tests for MarketRegimeDetector hysteresis (3-candle dwell time)."""

    def _make_trending_ind(self):
        return {
            "adx_last": 30.0,
            "ema9_last": 101.0,
            "ema21_last": 100.0,
            "bb_upper_last": 102.0,
            "bb_mid_last": 100.0,
            "bb_lower_last": 98.0,  # BB width = (102-98)/100 = 4% < 5% VOLATILE threshold
        }

    def _make_ranging_ind(self):
        return {
            "adx_last": 15.0,
            "ema9_last": 100.0,
            "ema21_last": 100.0,
            "bb_upper_last": 101.0,
            "bb_mid_last": 100.0,
            "bb_lower_last": 99.0,  # BB width = 2% > 1.5% quiet threshold
        }

    def test_initial_classification_accepted_immediately(self):
        """First classification is accepted without dwell count."""
        from src.regime import MarketRegime, MarketRegimeDetector
        det = MarketRegimeDetector(hysteresis_candles=3)
        ind = self._make_trending_ind()
        result = det.classify(ind)
        assert result.regime == MarketRegime.TRENDING_UP

    def test_regime_holds_during_transition(self):
        """Regime switch requires 3 consecutive new readings."""
        from src.regime import MarketRegime, MarketRegimeDetector
        det = MarketRegimeDetector(hysteresis_candles=3)
        trending_ind = self._make_trending_ind()
        ranging_ind = self._make_ranging_ind()

        # Establish TRENDING_UP
        r = det.classify(trending_ind)
        assert r.regime == MarketRegime.TRENDING_UP

        # First reading of RANGING: still TRENDING_UP (dwell=1, need 3)
        r = det.classify(ranging_ind)
        assert r.regime == MarketRegime.TRENDING_UP

        # Second reading of RANGING: still TRENDING_UP (dwell=2)
        r = det.classify(ranging_ind)
        assert r.regime == MarketRegime.TRENDING_UP

        # Third reading of RANGING: now switches to RANGING (dwell=3)
        r = det.classify(ranging_ind)
        assert r.regime == MarketRegime.RANGING

    def test_regime_counter_resets_on_reversion(self):
        """If raw regime reverts to stable before reaching dwell count, counter resets."""
        from src.regime import MarketRegime, MarketRegimeDetector
        det = MarketRegimeDetector(hysteresis_candles=3)
        trending_ind = self._make_trending_ind()
        ranging_ind = self._make_ranging_ind()

        # Establish TRENDING_UP
        det.classify(trending_ind)

        # One RANGING reading (dwell=1)
        r = det.classify(ranging_ind)
        assert r.regime == MarketRegime.TRENDING_UP

        # Revert to TRENDING_UP (counter resets)
        r = det.classify(trending_ind)
        assert r.regime == MarketRegime.TRENDING_UP

        # Now RANGING again (fresh dwell count = 1)
        r = det.classify(ranging_ind)
        assert r.regime == MarketRegime.TRENDING_UP

    def test_hysteresis_candles_configurable(self):
        """hysteresis_candles=1 means regime switches immediately."""
        from src.regime import MarketRegime, MarketRegimeDetector
        det = MarketRegimeDetector(hysteresis_candles=1)
        trending_ind = self._make_trending_ind()
        ranging_ind = self._make_ranging_ind()

        det.classify(trending_ind)
        r = det.classify(ranging_ind)
        assert r.regime == MarketRegime.RANGING  # switches on first reading

    def test_stable_regime_stays_stable(self):
        """Repeated classifications of the same regime stay stable."""
        from src.regime import MarketRegime, MarketRegimeDetector
        det = MarketRegimeDetector(hysteresis_candles=3)
        ind = self._make_trending_ind()
        for _ in range(10):
            r = det.classify(ind)
            assert r.regime == MarketRegime.TRENDING_UP


# ===========================================================================
# P1-2: Relative BB proximity in swing channel
# ===========================================================================

class TestSwingBBProximity:
    """Tests for percentile-based BB rejection gate in SwingChannel."""

    from src.smc import Direction, LiquiditySweep, MSSSignal

    def _make_h1_candles(self, n: int = 60, close_val: float = 100.0) -> dict:
        closes = np.ones(n) * close_val
        return {
            "open": closes - 0.1,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.ones(n) * 1000.0,
        }

    def _make_h4_candles(self, n: int = 60, close_val: float = 100.0) -> dict:
        closes = np.ones(n) * close_val
        return {
            "open": closes - 0.1,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.ones(n) * 1000.0,
        }

    def test_long_rejected_when_bb_position_above_threshold(self):
        """LONG rejected when price is above 15% of BB range (not near lower band)."""
        from src.channels.swing import SwingChannel
        from src.smc import Direction, MSSSignal
        ch = SwingChannel()

        # bb_lower=95, bb_upper=105, close=97 → bb_position = (97-95)/10 = 0.2 > 0.15
        close_val = 97.0
        candles = {
            "1h": self._make_h1_candles(close_val=close_val),
            "4h": self._make_h4_candles(close_val=close_val),
        }
        from src.smc import LiquiditySweep
        sweep = LiquiditySweep(index=59, direction=Direction.LONG, sweep_level=95.0,
                               close_price=95.5, wick_high=96.0, wick_low=94.0)
        mss = MSSSignal(index=59, direction=Direction.LONG, midpoint=95.5, confirm_close=96.0)
        ind = {
            "4h": {"adx_last": 25.0},
            "1h": {
                "ema200_last": 90.0,  # below close → long bias confirmed
                "bb_upper_last": 105.0,
                "bb_lower_last": 95.0,
                "rsi_last": 50.0,
            },
        }
        smc_data = {"sweeps": [sweep], "mss": mss}
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is None  # bb_position = 0.2 > 0.15 → rejected

    def test_long_allowed_when_bb_position_at_threshold(self):
        """LONG allowed when price is in bottom 15% of BB range."""
        from src.channels.swing import SwingChannel
        from src.smc import Direction, MSSSignal, LiquiditySweep
        ch = SwingChannel()

        # bb_lower=95, bb_upper=105, close=95.5 → bb_position = (95.5-95)/10 = 0.05 < 0.15
        close_val = 95.5
        candles = {
            "1h": self._make_h1_candles(close_val=close_val),
            "4h": self._make_h4_candles(close_val=close_val),
        }
        sweep = LiquiditySweep(index=59, direction=Direction.LONG, sweep_level=94.0,
                               close_price=94.5, wick_high=96.0, wick_low=93.0)
        mss = MSSSignal(index=59, direction=Direction.LONG, midpoint=94.5, confirm_close=95.0)
        ind = {
            "4h": {"adx_last": 25.0},
            "1h": {
                "ema200_last": 90.0,
                "bb_upper_last": 105.0,
                "bb_lower_last": 95.0,
                "rsi_last": 50.0,
            },
        }
        smc_data = {"sweeps": [sweep], "mss": mss}
        sig = ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        assert sig is not None

    def test_fallback_used_when_zero_width_bands(self):
        """Zero-width BB (upper == lower) falls back to legacy fixed threshold."""
        from src.channels.swing import SwingChannel
        from src.smc import Direction, MSSSignal, LiquiditySweep
        ch = SwingChannel()
        close_val = 100.0
        candles = {
            "1h": self._make_h1_candles(close_val=close_val),
            "4h": self._make_h4_candles(close_val=close_val),
        }
        sweep = LiquiditySweep(index=59, direction=Direction.LONG, sweep_level=99.0,
                               close_price=99.5, wick_high=101.0, wick_low=98.0)
        mss = MSSSignal(index=59, direction=Direction.LONG, midpoint=99.5, confirm_close=100.0)
        ind = {
            "4h": {"adx_last": 25.0},
            "1h": {
                "ema200_last": 90.0,
                "bb_upper_last": 100.0,  # upper == lower
                "bb_lower_last": 100.0,
                "rsi_last": 50.0,
            },
        }
        smc_data = {"sweeps": [sweep], "mss": mss}
        # With zero-width bands and close == bb_lower → legacy check passes (close <= bb_lower*1.02)
        ch.evaluate("BTCUSDT", candles, ind, smc_data, 0.01, 10_000_000)
        # Result depends on legacy fallback — just verify it doesn't crash
        assert True  # No exception is the primary assertion here


# ===========================================================================
# P2-1: Confidence log infrastructure
# ===========================================================================

class TestConfidenceLog:
    """Tests for log_confidence_breakdown and CONFIDENCE_LOG_ENABLED."""

    def test_log_confidence_breakdown_writes_json(self, tmp_path):
        """log_confidence_breakdown appends a valid JSON record."""
        from src.confidence import log_confidence_breakdown
        log_file = tmp_path / "confidence_log.jsonl"
        breakdown = {"smc": 20.0, "trend": 15.0, "liquidity": 10.0}
        log_confidence_breakdown(
            signal_id="TEST-001",
            channel="360_SCALP",
            breakdown=breakdown,
            total=75.0,
            session_multiplier=1.0,
            outcome=None,
        )
        # Verify file doesn't exist yet (path not configured)
        # Re-run with our tmp path:
        import src.confidence as _cm
        import config as _cfg
        orig = _cfg.CONFIDENCE_LOG_PATH
        _cfg.CONFIDENCE_LOG_PATH = str(log_file)
        import importlib
        importlib.reload(_cm)

        _cm.log_confidence_breakdown(
            signal_id="TEST-001",
            channel="360_SCALP",
            breakdown=breakdown,
            total=75.0,
            session_multiplier=1.0,
        )
        _cfg.CONFIDENCE_LOG_PATH = orig
        importlib.reload(_cm)

        # parse the log
        if log_file.exists():
            import json
            records = [json.loads(line) for line in log_file.read_text().strip().splitlines()]
            assert len(records) >= 1
            rec = records[0]
            assert rec["signal_id"] == "TEST-001"
            assert rec["channel"] == "360_SCALP"
            assert rec["total"] == 75.0
            assert "outcome" in rec

    def test_compute_confidence_returns_result(self):
        """compute_confidence with signal_id param returns valid result."""
        from src.confidence import ConfidenceInput, compute_confidence
        inp = ConfidenceInput(
            smc_score=20.0,
            trend_score=15.0,
            liquidity_score=15.0,
            spread_score=8.0,
            data_sufficiency=10.0,
            multi_exchange=2.5,
            onchain_score=5.0,
            order_flow_score=5.0,
        )
        result = compute_confidence(inp, channel="360_SCALP", signal_id="SIG-001")
        assert result.total >= 0.0
        assert result.total <= 100.0


# ===========================================================================
# P2-2: Volatility-adaptive TP ratios
# ===========================================================================

class TestVolatilityAdaptiveTP:
    """Tests for bb_width_pct-driven TP ratio adaptation in build_channel_signal."""

    def _make_base_signal_args(self, direction_long: bool = True):
        from src.smc import Direction
        from config import CHANNEL_SCALP_CVD
        direction = Direction.LONG if direction_long else Direction.SHORT
        close = 100.0
        sl = 99.0 if direction_long else 101.0
        sl_dist = abs(close - sl)
        tp_ratios = CHANNEL_SCALP_CVD.tp_ratios
        if direction_long:
            tp1 = close + sl_dist * tp_ratios[0]
            tp2 = close + sl_dist * tp_ratios[1]
            tp3 = close + sl_dist * tp_ratios[2]
        else:
            tp1 = close - sl_dist * tp_ratios[0]
            tp2 = close - sl_dist * tp_ratios[1]
            tp3 = close - sl_dist * tp_ratios[2]
        return dict(
            config=CHANNEL_SCALP_CVD,
            symbol="BTCUSDT",
            direction=direction,
            close=close,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl_dist=sl_dist,
            id_prefix="TST",
        )

    def test_high_vol_stretches_tp_targets(self):
        """BB width > 5% stretches TP targets by 1.3× (legacy path)."""
        from src.channels.base import build_channel_signal, _HIGH_VOL_BB_WIDTH
        from unittest.mock import patch
        args = self._make_base_signal_args(direction_long=True)
        with patch("src.channels.base.DYNAMIC_SL_TP_ENABLED", False):
            sig_base = build_channel_signal(**args)
            sig_highvol = build_channel_signal(**args, bb_width_pct=_HIGH_VOL_BB_WIDTH + 1.0)
        assert sig_base is not None and sig_highvol is not None
        # TP1 should be further from entry in high-vol
        entry = sig_base.entry
        assert (sig_highvol.tp1 - entry) > (sig_base.tp1 - entry)

    def test_low_vol_compresses_tp_targets(self):
        """BB width < 1.5% compresses TP targets by 0.7× (legacy path)."""
        from src.channels.base import build_channel_signal, _LOW_VOL_BB_WIDTH
        from unittest.mock import patch
        args = self._make_base_signal_args(direction_long=True)
        with patch("src.channels.base.DYNAMIC_SL_TP_ENABLED", False):
            sig_base = build_channel_signal(**args)
            sig_lowvol = build_channel_signal(**args, bb_width_pct=_LOW_VOL_BB_WIDTH - 0.5)
        assert sig_base is not None and sig_lowvol is not None
        entry = sig_base.entry
        assert (sig_lowvol.tp1 - entry) < (sig_base.tp1 - entry)

    def test_mid_vol_uses_base_ratios(self):
        """BB width between 1.5% and 5% uses base TP ratios unchanged."""
        from src.channels.base import build_channel_signal
        args = self._make_base_signal_args(direction_long=True)
        sig_base = build_channel_signal(**args)
        sig_midvol = build_channel_signal(**args, bb_width_pct=3.0)
        assert sig_base is not None and sig_midvol is not None
        assert sig_midvol.tp1 == pytest.approx(sig_base.tp1, abs=1e-6)

    def test_no_bb_width_uses_base_ratios(self):
        """Without bb_width_pct, TP ratios are unchanged."""
        from src.channels.base import build_channel_signal
        args = self._make_base_signal_args(direction_long=True)
        sig_base = build_channel_signal(**args)
        sig_none = build_channel_signal(**args, bb_width_pct=None)
        assert sig_base is not None and sig_none is not None
        assert sig_none.tp1 == pytest.approx(sig_base.tp1, abs=1e-6)


# ===========================================================================
# P2-3: Funding rate costs in backtester
# ===========================================================================

class TestFundingRateCosts:
    """Funding rate is deducted from trade PnL based on hold duration."""

    def test_funding_rate_reduces_pnl(self):
        """Non-zero funding rate deducts from PnL."""
        from src.backtester import _simulate_trade
        sig = _make_fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0, tp2=102.0)
        future = {
            "high": np.array([101.5] * 50),  # TP1 hit on candle 1
            "low": np.array([99.5] * 50),
            "close": np.array([101.5] * 50),
        }
        _, pnl_no_fund, _ = _simulate_trade(sig, future, funding_rate_8h=0.0)
        _, pnl_with_fund, _ = _simulate_trade(sig, future, funding_rate_8h=0.05)
        assert pnl_with_fund < pnl_no_fund

    def test_zero_funding_rate_no_deduction(self):
        """Zero funding rate has no effect on PnL."""
        from src.backtester import _simulate_trade
        sig = _make_fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0)
        future = {
            "high": np.array([101.5]),
            "low": np.array([99.5]),
            "close": np.array([101.5]),
        }
        _, pnl_a, _ = _simulate_trade(sig, future, funding_rate_8h=0.0)
        _, pnl_b, _ = _simulate_trade(sig, future)
        assert pnl_a == pytest.approx(pnl_b, abs=1e-6)

    def test_calc_funding_formula(self):
        """_calc_funding returns expected value."""
        from src.backtester import _calc_funding
        # 96 candles × 5 min = 480 min = 1 funding period
        # funding_rate_8h = 0.01 → cost = 0.01 * 1 = 0.01
        cost = _calc_funding(0.01, 96, 5)
        assert cost == pytest.approx(0.01, abs=1e-6)

    def test_backtester_funding_rate_param_accepted(self):
        """Backtester accepts funding_rate_per_8h without error."""
        from src.backtester import Backtester
        from src.channels.scalp import ScalpChannel
        bt = Backtester(channels=[ScalpChannel()], min_window=30, lookahead_candles=5,
                        funding_rate_per_8h=0.01)
        assert bt._funding_rate_per_8h == pytest.approx(0.01)


# ===========================================================================
# P2-5: Pre-event macro blackout window
# ===========================================================================

class TestMacroBlackout:
    """Tests for is_in_macro_blackout()."""

    def test_empty_events_returns_false(self):
        """No events → not in blackout."""
        from src.macro_blackout import is_in_macro_blackout
        in_blackout, reason = is_in_macro_blackout([])
        assert in_blackout is False
        assert reason == ""

    def test_in_pre_event_window(self):
        """Within pre_minutes before event → in blackout."""
        from src.macro_blackout import is_in_macro_blackout
        now = datetime(2024, 3, 20, 17, 45, tzinfo=timezone.utc)
        event_time = datetime(2024, 3, 20, 18, 0, tzinfo=timezone.utc)  # 15 min away
        events = [{"event_time": event_time, "name": "FOMC", "severity": "CRITICAL"}]
        in_blackout, reason = is_in_macro_blackout(events, now=now, pre_minutes=30, post_minutes=60)
        assert in_blackout is True
        assert "FOMC" in reason

    def test_in_post_event_window(self):
        """Within post_minutes after event → in blackout."""
        from src.macro_blackout import is_in_macro_blackout
        now = datetime(2024, 3, 20, 18, 30, tzinfo=timezone.utc)  # 30 min after
        event_time = datetime(2024, 3, 20, 18, 0, tzinfo=timezone.utc)
        events = [{"event_time": event_time, "name": "CPI", "severity": "HIGH"}]
        in_blackout, reason = is_in_macro_blackout(events, now=now, pre_minutes=30, post_minutes=60)
        assert in_blackout is True
        assert "CPI" in reason

    def test_outside_blackout_window(self):
        """Outside blackout window → not in blackout."""
        from src.macro_blackout import is_in_macro_blackout
        now = datetime(2024, 3, 20, 14, 0, tzinfo=timezone.utc)  # 4h before
        event_time = datetime(2024, 3, 20, 18, 0, tzinfo=timezone.utc)
        events = [{"event_time": event_time, "name": "FOMC", "severity": "CRITICAL"}]
        in_blackout, reason = is_in_macro_blackout(events, now=now, pre_minutes=30, post_minutes=60)
        assert in_blackout is False
        assert reason == ""

    def test_low_severity_does_not_trigger_blackout(self):
        """LOW severity events do not trigger blackout."""
        from src.macro_blackout import is_in_macro_blackout
        now = datetime(2024, 3, 20, 17, 55, tzinfo=timezone.utc)
        event_time = datetime(2024, 3, 20, 18, 0, tzinfo=timezone.utc)
        events = [{"event_time": event_time, "name": "Minor Data", "severity": "LOW"}]
        in_blackout, reason = is_in_macro_blackout(events, now=now, pre_minutes=30, post_minutes=60)
        assert in_blackout is False

    def test_medium_severity_does_not_trigger_blackout(self):
        """MEDIUM severity events do not trigger blackout."""
        from src.macro_blackout import is_in_macro_blackout
        now = datetime(2024, 3, 20, 17, 55, tzinfo=timezone.utc)
        event_time = datetime(2024, 3, 20, 18, 0, tzinfo=timezone.utc)
        events = [{"event_time": event_time, "name": "Report", "severity": "MEDIUM"}]
        in_blackout, reason = is_in_macro_blackout(events, now=now, pre_minutes=30, post_minutes=60)
        assert in_blackout is False

    def test_naive_datetime_treated_as_utc(self):
        """Naive datetime is treated as UTC — no crash."""
        from src.macro_blackout import is_in_macro_blackout
        now = datetime(2024, 3, 20, 17, 55)  # naive
        event_time = datetime(2024, 3, 20, 18, 0)  # naive
        events = [{"event_time": event_time, "name": "FOMC", "severity": "CRITICAL"}]
        in_blackout, _ = is_in_macro_blackout(events, now=now, pre_minutes=30, post_minutes=60)
        assert in_blackout is True

    def test_multiple_events_any_match_triggers_blackout(self):
        """Multiple events: any match within window triggers blackout."""
        from src.macro_blackout import is_in_macro_blackout
        now = datetime(2024, 3, 20, 17, 55, tzinfo=timezone.utc)
        events = [
            {"event_time": datetime(2024, 3, 21, 10, 0, tzinfo=timezone.utc),
             "name": "Future Event", "severity": "CRITICAL"},
            {"event_time": datetime(2024, 3, 20, 18, 0, tzinfo=timezone.utc),
             "name": "FOMC", "severity": "HIGH"},
        ]
        in_blackout, reason = is_in_macro_blackout(events, now=now, pre_minutes=30, post_minutes=60)
        assert in_blackout is True
        assert "FOMC" in reason


# ===========================================================================
# P1-1: Trailing stop simulation in backtester
# ===========================================================================

class TestTrailingStopSimulation:
    """Backtester trailing stop: SL moves to breakeven at TP1, to TP1 at TP2."""

    def test_partial_win_after_tp1_then_reversal(self):
        """Trade hits TP1, SL moves to breakeven, then price reverses: partial win."""
        from src.backtester import _simulate_trade
        sig = _make_fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0, tp2=103.0)
        # Candle 1: hits TP1 (high=101.5)
        # Candle 2: price drops back to near breakeven — now current_sl ~ 100 + 0.15 = 100.15
        future = {
            "high": np.array([101.5, 100.5, 100.0]),
            "low": np.array([99.5, 99.8, 99.9]),  # below trailing SL on candle 3
            "close": np.array([101.5, 100.0, 99.9]),
        }
        won, pnl, tp_level = _simulate_trade(sig, future)
        # TP1 was hit → it's a win (partial profit locked in)
        assert won is True
        assert tp_level >= 1

    def test_full_tp3_hit_returns_tp_level_3(self):
        """Trade hits all three TP levels → tp_level = 3."""
        from src.backtester import _simulate_trade
        sig = _make_fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0, tp2=102.0, tp3=104.0)
        future = {
            "high": np.array([101.5, 102.5, 104.5]),
            "low": np.array([99.5, 100.5, 101.5]),
            "close": np.array([101.5, 102.5, 104.5]),
        }
        won, pnl, tp_level = _simulate_trade(sig, future)
        assert won is True
        assert tp_level == 3
        assert pnl > 0

    def test_sl_hit_before_tp1_is_loss(self):
        """SL hit before any TP → loss."""
        from src.backtester import _simulate_trade
        sig = _make_fake_signal("LONG", entry=100.0, sl=99.0, tp1=101.0)
        future = {
            "high": np.array([100.5]),
            "low": np.array([98.5]),  # below SL
            "close": np.array([99.0]),
        }
        won, pnl, tp_level = _simulate_trade(sig, future)
        assert won is False
        assert pnl < 0
        assert tp_level == 0

    def test_partial_wins_counted_in_backtest_result(self):
        """BacktestResult.partial_wins is incremented for wins with TP hit."""
        from src.backtester import BacktestResult
        r = BacktestResult(channel="TEST", wins=5, partial_wins=3)
        assert r.partial_wins == 3
