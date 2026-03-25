"""360_SWING – H1/H4 Institutional Swing 🏛️

Trigger : H4 ERL sweep + H1 MSS
Filters : EMA200, Bollinger rejection, ADX 20–40, ATR filter, spread < 0.02 %
Risk    : SL 0.2–0.5 %, TP1 1.5R, TP2 3R, TP3 4–5R, Trailing 2.5×ATR
"""

from __future__ import annotations

from typing import Dict, Optional
import uuid

from config import CHANNEL_SWING
from src.channels.base import BaseChannel, Signal
from src.dca import compute_dca_zone
from src.filters import check_adx, check_spread, check_volume
from src.smc import Direction
from src.utils import utcnow


class SwingChannel(BaseChannel):
    def __init__(self) -> None:
        super().__init__(CHANNEL_SWING)

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
        h1 = candles.get("1h")
        if h4 is None or h1 is None:
            return None
        if len(h4.get("close", [])) < 50 or len(h1.get("close", [])) < 50:
            return None

        # --- Filters ---
        ind_h4 = indicators.get("4h", {})
        ind_h1 = indicators.get("1h", {})
        if not check_adx(ind_h4.get("adx_last"), self.config.adx_min, self.config.adx_max):
            return None
        if not check_spread(spread_pct, self.config.spread_max):
            return None
        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        # EMA200 filter
        ema200 = ind_h1.get("ema200_last")
        close_h1 = float(h1["close"][-1])
        if ema200 is None:
            return None

        # Bollinger rejection
        bb_upper = ind_h1.get("bb_upper_last")
        bb_lower = ind_h1.get("bb_lower_last")

        # --- SMC trigger: H4 sweep + H1 MSS ---
        sweeps = smc_data.get("sweeps", [])
        mss = smc_data.get("mss")
        if not sweeps or mss is None:
            return None

        direction = mss.direction

        # RSI extreme gate: don't chase overbought LONGs or fade oversold SHORTs
        rsi_last = ind_h1.get("rsi_last")
        if rsi_last is not None:
            if direction == Direction.LONG and rsi_last > 75:
                return None
            if direction == Direction.SHORT and rsi_last < 25:
                return None

        # Validate EMA200 bias
        if direction == Direction.LONG and close_h1 < ema200:
            return None
        if direction == Direction.SHORT and close_h1 > ema200:
            return None

        # Validate Bollinger rejection
        if direction == Direction.LONG and bb_lower is not None:
            if close_h1 > bb_lower * 1.02:  # must be near lower band
                return None  # Too far from lower band — no BB rejection setup
        if direction == Direction.SHORT and bb_upper is not None:
            if close_h1 < bb_upper * 0.98:
                return None  # Too far from upper band — no BB rejection setup

        # Daily S/R confluence check (soft boost, not a hard reject)
        d1 = candles.get("1d")
        daily_confluence = False
        if d1 is not None and len(d1.get("close", [])) >= 20:
            d1_highs = [float(h) for h in list(d1.get("high", d1["close"]))[-20:]]
            d1_lows = [float(low_val) for low_val in list(d1.get("low", d1["close"]))[-20:]]
            if direction == Direction.LONG:
                nearest_daily_support = min(d1_lows[-10:])
                if close_h1 <= nearest_daily_support * 1.03:  # within 3% of daily support
                    daily_confluence = True
            elif direction == Direction.SHORT:
                nearest_daily_resistance = max(d1_highs[-10:])
                if close_h1 >= nearest_daily_resistance * 0.97:  # within 3% of daily resistance
                    daily_confluence = True

        close = close_h1
        atr_val = ind_h1.get("atr_last", close * 0.003)

        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val)
        if direction == Direction.LONG:
            sl = close - sl_dist
            tp1 = close + sl_dist * self.config.tp_ratios[0]
            tp2 = close + sl_dist * self.config.tp_ratios[1]
            tp3 = close + sl_dist * self.config.tp_ratios[2]
        else:
            sl = close + sl_dist
            tp1 = close - sl_dist * self.config.tp_ratios[0]
            tp2 = close - sl_dist * self.config.tp_ratios[1]
            tp3 = close - sl_dist * self.config.tp_ratios[2]

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
            confidence=0.0,
            ai_sentiment_label="",
            ai_sentiment_summary="",
            risk_label="Medium",
            timestamp=utcnow(),
            signal_id=f"SWING-{uuid.uuid4().hex[:8].upper()}",
            current_price=close,
            original_sl_distance=sl_dist,
        )

        # Initialise DCA zone so the trade monitor can check for Entry 2
        dca_lower, dca_upper = compute_dca_zone(
            close, round(sl, 8), direction, self.config.dca_zone_range
        )
        sig.dca_zone_lower = dca_lower
        sig.dca_zone_upper = dca_upper
        sig.original_entry = close
        sig.original_tp1 = round(tp1, 8)
        sig.original_tp2 = round(tp2, 8)
        sig.original_tp3 = round(tp3, 8)

        # Mark signal quality tier based on daily confluence
        if daily_confluence:
            sig.setup_class = "SWING_D1_CONFLUENCE"
            sig.quality_tier = "A+"
        else:
            sig.setup_class = "SWING_STANDARD"

        return sig
