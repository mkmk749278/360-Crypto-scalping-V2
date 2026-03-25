"""360_SPOT – H4/D1 Spot Accumulation Channel 📈

Trigger : H4/D1 accumulation breakout with sustained volume expansion (LONG)
          OR H4/D1 distribution breakdown with sustained volume on down-move (SHORT)
Filters : EMA200, ADX, ATR, spread, volume
Risk    : SL 0.5–2 %, TP1 2R, TP2 5R, TP3 10R, Trailing 3×ATR, max hold 7 days
"""

from __future__ import annotations

from typing import Dict, Optional
import uuid

from config import CHANNEL_SPOT
from src.channels.base import BaseChannel, Signal
from src.dca import compute_dca_zone
from src.filters import check_adx, check_spread, check_volume
from src.smc import Direction
from src.utils import utcnow

# Short signals require a higher minimum confidence to guard against false shorts.
_SHORT_CONFIDENCE_BOOST = 5.0


class SpotChannel(BaseChannel):
    def __init__(self) -> None:
        super().__init__(CHANNEL_SPOT)

    def evaluate(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        h4 = candles.get("4h")
        if h4 is None or len(h4.get("close", [])) < 50:
            return None

        ind_h4 = indicators.get("4h", {})

        # --- Basic filters ---
        if not check_adx(ind_h4.get("adx_last"), self.config.adx_min):
            return None
        if not check_spread(spread_pct, self.config.spread_max):
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
        # Bollinger squeeze detection: require tight BB before breakout
        if bb_width is not None and bb_width > 4.0:
            return None  # Not squeezing, not a real accumulation pattern

        # Accumulation breakout: price must clear recent H4 resistance
        # using an ATR-adaptive threshold instead of a fixed 0.2% proximity.
        if len(highs) < 10:
            return None
        recent_high = max(float(h) for h in highs[-10:-1])
        breakout_buffer = atr_val * 0.2
        if close < recent_high + breakout_buffer:
            return None  # No confirmed breakout — candle must close above resistance + buffer

        # Volume expansion
        if len(volumes) < 10 or len(closes_list) < 10:
            return None
        usd_volumes = [float(v) * float(c) for v, c in zip(volumes[-10:], closes_list[-10:])]
        avg_usd_vol = sum(usd_volumes[:-1]) / 9
        if usd_volumes[-1] < avg_usd_vol * 1.8:
            return None

        # SMC: bearish structure contradicts LONG
        if mss is not None and getattr(mss, "direction", None) == Direction.SHORT:
            return None

        # RSI overbought gate
        if rsi_last is not None and rsi_last > 75:
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

        sig = self._build_signal(
            symbol=symbol,
            direction=Direction.LONG,
            close=close,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl_dist=sl_dist,
        )
        if sig is not None:
            sig.setup_class = "BREAKOUT_RETEST" if is_retest else "BREAKOUT_INITIAL"
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
        # Bollinger squeeze
        if bb_width is not None and bb_width > 4.0:
            return None

        # Distribution breakdown: price must breach recent H4 support
        # using an ATR-adaptive threshold instead of a fixed 0.2% proximity.
        if len(lows) < 10:
            return None
        recent_low = min(float(lo) for lo in lows[-10:-1])
        breakdown_buffer = atr_val * 0.2
        if close > recent_low - breakdown_buffer:
            return None  # No confirmed breakdown

        # Volume expansion on the down-move
        if len(volumes) < 10 or len(closes_list) < 10:
            return None
        usd_volumes = [float(v) * float(c) for v, c in zip(volumes[-10:], closes_list[-10:])]
        avg_usd_vol = sum(usd_volumes[:-1]) / 9
        if usd_volumes[-1] < avg_usd_vol * 1.8:
            return None

        # SMC: bullish structure contradicts SHORT
        if mss is not None and getattr(mss, "direction", None) == Direction.LONG:
            return None

        # RSI oversold gate: don't short into an already oversold market
        if rsi_last is not None and rsi_last < 25:
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

        sig = self._build_signal(
            symbol=symbol,
            direction=Direction.SHORT,
            close=close,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl_dist=sl_dist,
            confidence_boost=_SHORT_CONFIDENCE_BOOST,
        )
        if sig is not None:
            sig.setup_class = "BREAKOUT_RETEST" if is_retest else "BREAKOUT_INITIAL"
        return sig

    # ------------------------------------------------------------------
    # Shared signal factory
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        symbol: str,
        direction: Direction,
        close: float,
        sl: float,
        tp1: float,
        tp2: float,
        tp3: float,
        sl_dist: float,
        confidence_boost: float = 0.0,
    ) -> Signal:
        """Assemble and return a :class:`Signal` instance."""
        prefix = "SPOT-SHORT" if direction == Direction.SHORT else "SPOT"
        sig = Signal(
            channel=self.config.name,
            symbol=symbol,
            direction=direction,
            entry=close,
            stop_loss=round(sl, 8),
            tp1=round(tp1, 8),
            tp2=round(tp2, 8),
            tp3=round(tp3, 8),
            trailing_active=True,
            trailing_desc=f"{self.config.trailing_atr_mult}×ATR",
            confidence=0.0 + confidence_boost,
            ai_sentiment_label="",
            ai_sentiment_summary="",
            risk_label="Conservative" if direction == Direction.LONG else "Conservative-Short",
            timestamp=utcnow(),
            signal_id=f"{prefix}-{uuid.uuid4().hex[:8].upper()}",
            current_price=close,
            original_sl_distance=sl_dist,
        )

        # DCA zone for LONG spot accumulation only (not SHORT)
        if direction == Direction.LONG and self.config.dca_enabled:
            dca_lower, dca_upper = compute_dca_zone(
                close, round(sl, 8), direction, self.config.dca_zone_range
            )
            sig.dca_zone_lower = dca_lower
            sig.dca_zone_upper = dca_upper
            sig.original_entry = close
            sig.original_tp1 = round(tp1, 8)
            sig.original_tp2 = round(tp2, 8)
            sig.original_tp3 = round(tp3, 8)
            sig.entry_zone_low = dca_lower
            sig.entry_zone_high = dca_upper

        return sig
