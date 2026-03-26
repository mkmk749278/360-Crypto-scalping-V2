"""360_SPOT – H4/D1 Spot Accumulation Channel 📈

Trigger : H4/D1 accumulation breakout with sustained volume expansion (LONG)
          OR H4/D1 distribution breakdown with sustained volume on down-move (SHORT)
Filters : EMA200, ADX, ATR, spread, volume
Risk    : SL 0.5–2 %, TP1 2R, TP2 5R, TP3 10R, Trailing 3×ATR, max hold 7 days
"""

from __future__ import annotations

from typing import Dict, Optional

from config import CHANNEL_SPOT
from src.channels.base import BaseChannel, Signal, build_channel_signal
from src.filters import check_adx_regime, check_rsi_regime, check_spread_adaptive, check_volume, check_volume_expansion
from src.smc import Direction

# Short signals require a higher minimum confidence to guard against false shorts.
_SHORT_CONFIDENCE_BOOST = 5.0


class SpotChannel(BaseChannel):
    def __init__(self) -> None:
        super().__init__(CHANNEL_SPOT)
        self._current_regime = ""
        self._current_atr_percentile = 50.0
        self._current_pair_tier = "MIDCAP"

    def evaluate(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
        regime: str = "",
    ) -> Optional[Signal]:
        self._current_regime = regime
        regime_ctx = smc_data.get("regime_context")
        self._current_atr_percentile = regime_ctx.atr_percentile if regime_ctx else 50.0
        pair_profile = smc_data.get("pair_profile")
        self._current_pair_tier = pair_profile.tier if pair_profile else "MIDCAP"
        h4 = candles.get("4h")
        if h4 is None or len(h4.get("close", [])) < 50:
            return None

        ind_h4 = indicators.get("4h", {})

        # --- Basic filters ---
        if not check_adx_regime(ind_h4.get("adx_last"), regime=regime, max_adx=100.0):
            return None
        if not check_spread_adaptive(spread_pct, self.config.spread_max, regime=regime):
            return None
        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        close_h4 = float(h4["close"][-1])
        ema200 = ind_h4.get("ema200_last")
        ind_d1 = indicators.get("1d", {})
        ema50_daily = ind_d1.get("ema50_last")
        rsi_last = ind_h4.get("rsi_last")
        mss = smc_data.get("mss")
        bb_width = ind_h4.get("bb_width_pct")
        highs = h4.get("high", [])
        lows = h4.get("low", [])
        volumes = h4.get("volume", [])
        closes_list = h4.get("close", [])
        atr_val = ind_h4.get("atr_last", close_h4 * 0.01)

        # -------------------------------------------------------------------
        # LONG path: price above EMA200, both trend filters bullish
        # -------------------------------------------------------------------
        if ema200 is not None and close_h4 >= ema200:
            # Daily EMA50 alignment: ensure the daily trend is also up
            if ema50_daily is not None and close_h4 < ema50_daily:
                pass  # fall through to SHORT check
            else:
                long_sig = self._try_long(
                    symbol, close_h4, atr_val, h4, highs, lows, volumes,
                    closes_list, bb_width, rsi_last, mss,
                )
                if long_sig is not None:
                    return long_sig

        # -------------------------------------------------------------------
        # SHORT path (feature 6): price below EMA200 AND below daily EMA50
        # Both conditions must be true to filter out mixed/neutral setups.
        # -------------------------------------------------------------------
        if ema200 is not None and close_h4 < ema200:
            if ema50_daily is not None and close_h4 < ema50_daily:
                return self._try_short(
                    symbol, close_h4, atr_val, h4, highs, lows, volumes,
                    closes_list, bb_width, rsi_last, mss,
                )

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bb_squeeze_threshold(self, atr_val: float, close: float) -> float:
        """Return the ATR-normalized Bollinger squeeze threshold.

        Scales the maximum allowed BB width with pair-specific volatility so
        that high-ATR altcoins are not blocked by BTC-centric thresholds.
        Clamped to [2.0, 6.0].
        """
        atr_pct = (atr_val / close * 100) if close > 0 else 1.0
        return max(2.0, min(6.0, atr_pct * 3.0))

    def _volume_expansion_mult(self) -> float:
        """Return the regime-adjusted volume expansion multiplier.

        * QUIET / RANGING : 2.2× — any volume expansion should be meaningful
        * VOLATILE        : 1.5× — volume is naturally elevated
        * Default         : 1.8× (TRENDING or unknown)
        """
        regime_upper = self._current_regime.upper() if self._current_regime else ""
        if regime_upper in ("QUIET", "RANGING"):
            return 2.2
        if regime_upper == "VOLATILE":
            return 1.5
        return 1.8

    # ------------------------------------------------------------------
    # LONG signal builder
    # ------------------------------------------------------------------

    def _try_long(
        self,
        symbol: str,
        close: float,
        atr_val: float,
        h4: dict,
        highs: list,
        lows: list,
        volumes: list,
        closes_list: list,
        bb_width: Optional[float],
        rsi_last: Optional[float],
        mss: object,
    ) -> Optional[Signal]:
        """Attempt to build a LONG spot signal."""
        # ATR-normalized Bollinger squeeze detection.
        # Scale the squeeze threshold with pair-specific volatility so that
        # high-ATR altcoins aren't blocked by BTC-centric thresholds.
        if bb_width is not None:
            if bb_width > self._bb_squeeze_threshold(atr_val, close):
                return None  # Not squeezing, not a real accumulation pattern

        # Accumulation breakout: price must clear recent H4 resistance
        # using an ATR-adaptive threshold instead of a fixed 0.2% proximity.
        if len(highs) < 10:
            return None
        recent_high = max(float(h) for h in highs[-10:-1])
        breakout_buffer = atr_val * 0.2
        if close < recent_high + breakout_buffer:
            return None  # No confirmed breakout — candle must close above resistance + buffer

        # Volume expansion — threshold scales with regime via _volume_expansion_mult().
        if len(volumes) < 10 or len(closes_list) < 10:
            return None
        if not check_volume_expansion(volumes, closes_list, lookback=9, multiplier=self._volume_expansion_mult()):
            return None

        # SMC: bearish structure contradicts LONG
        if mss is not None and getattr(mss, "direction", None) == Direction.SHORT:
            return None

        # RSI overbought gate
        if not check_rsi_regime(rsi_last, direction="LONG", regime=self._current_regime):
            return None

        # Detect retest pattern: breakout → pullback → reclaim
        # Previous candle was below resistance, candle before that was above = pullback then reclaim
        is_retest = (
            len(closes_list) >= 3
            and float(closes_list[-2]) < recent_high
            and float(closes_list[-3]) >= recent_high * 0.998
        )

        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val * 1.5)
        sl = close - sl_dist
        tp1 = close + sl_dist * self.config.tp_ratios[0]
        tp2 = close + sl_dist * self.config.tp_ratios[1]
        tp3 = close + sl_dist * self.config.tp_ratios[2]

        if sl >= close:
            return None

        setup_class = "BREAKOUT_RETEST" if is_retest else "BREAKOUT_INITIAL"
        sig = build_channel_signal(
            config=self.config,
            symbol=symbol,
            direction=Direction.LONG,
            close=close,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl_dist=sl_dist,
            id_prefix="SPOT",
            atr_val=atr_val,
            setup_class=setup_class,
            bb_width_pct=bb_width,
            regime=self._current_regime,
            atr_percentile=self._current_atr_percentile,
            pair_tier=self._current_pair_tier,
        )
        # Spot-specific: use DCA zone as entry zone for LONGs (accumulation zone)
        if sig is not None and sig.dca_zone_lower > 0:
            sig.entry_zone_low = sig.dca_zone_lower
            sig.entry_zone_high = sig.dca_zone_upper
        return sig

    # ------------------------------------------------------------------
    # SHORT signal builder
    # ------------------------------------------------------------------

    def _try_short(
        self,
        symbol: str,
        close: float,
        atr_val: float,
        h4: dict,
        highs: list,
        lows: list,
        volumes: list,
        closes_list: list,
        bb_width: Optional[float],
        rsi_last: Optional[float],
        mss: object,
    ) -> Optional[Signal]:
        """Attempt to build a SHORT spot signal (feature 6).

        Mirrors the LONG logic with inverted conditions:
        * Bollinger squeeze required before breakdown
        * Price breaks below recent H4 support (distribution breakdown)
        * Volume expansion on the down-move
        * SMC bullish MSS contradicts SHORT → skip
        * RSI oversold gate (< 25) prevents chasing drops
        """
        # ATR-normalized Bollinger squeeze detection (mirrors _try_long).
        if bb_width is not None:
            if bb_width > self._bb_squeeze_threshold(atr_val, close):
                return None

        # Distribution breakdown: price must breach recent H4 support
        # using an ATR-adaptive threshold instead of a fixed 0.2% proximity.
        if len(lows) < 10:
            return None
        recent_low = min(float(lo) for lo in lows[-10:-1])
        breakdown_buffer = atr_val * 0.2
        if close > recent_low - breakdown_buffer:
            return None  # No confirmed breakdown

        # Volume expansion on the down-move — regime-adjusted multiplier.
        if len(volumes) < 10 or len(closes_list) < 10:
            return None
        if not check_volume_expansion(volumes, closes_list, lookback=9, multiplier=self._volume_expansion_mult()):
            return None

        # SMC: bullish structure contradicts SHORT
        if mss is not None and getattr(mss, "direction", None) == Direction.LONG:
            return None

        # RSI oversold gate: don't short into an already oversold market
        if not check_rsi_regime(rsi_last, direction="SHORT", regime=self._current_regime):
            return None

        # Detect retest pattern for SHORT: breakdown → bounce → reclaim below support
        # Previous candle was above support (pullback/bounce), candle before was below = retest
        is_retest = (
            len(closes_list) >= 3
            and float(closes_list[-2]) > recent_low
            and float(closes_list[-3]) <= recent_low * 1.002
        )

        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val * 1.5)
        sl = close + sl_dist          # SL above entry for SHORT
        tp1 = close - sl_dist * self.config.tp_ratios[0]
        tp2 = close - sl_dist * self.config.tp_ratios[1]
        tp3 = close - sl_dist * self.config.tp_ratios[2]

        if sl <= close or tp1 >= close:
            return None

        setup_class = "BREAKOUT_RETEST" if is_retest else "BREAKOUT_INITIAL"
        sig = build_channel_signal(
            config=self.config,
            symbol=symbol,
            direction=Direction.SHORT,
            close=close,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl_dist=sl_dist,
            id_prefix="SPOT-SHORT",
            atr_val=atr_val,
            setup_class=setup_class,
            bb_width_pct=bb_width,
            regime=self._current_regime,
            atr_percentile=self._current_atr_percentile,
            pair_tier=self._current_pair_tier,
        )
        if sig is not None:
            sig.confidence += _SHORT_CONFIDENCE_BOOST
            # Spot-specific: no DCA zone for SHORT signals (accumulation is LONG-only)
            sig.dca_zone_lower = 0.0
            sig.dca_zone_upper = 0.0
        return sig

