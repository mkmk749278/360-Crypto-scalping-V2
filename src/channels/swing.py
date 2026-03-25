"""360_SWING – H1/H4 Institutional Swing 🏛️

Trigger : H4 ERL sweep + H1 MSS
Filters : EMA200, Bollinger rejection, ADX 20–40, ATR filter, spread < 0.02 %
Risk    : SL 0.2–0.5 %, TP1 1.5R, TP2 3R, TP3 4–5R, Trailing 2.5×ATR
"""

from __future__ import annotations

from typing import Dict, Optional

from config import CHANNEL_SWING
from src.channels.base import BaseChannel, Signal, build_channel_signal
from src.filters import check_adx, check_rsi
from src.smc import Direction

# Percentile position within the Bollinger Band range for rejection gate.
# For LONG: price must be in the bottom BB_REJECTION_THRESHOLD fraction (near lower band).
# For SHORT: price must be in the top BB_REJECTION_THRESHOLD fraction (near upper band).
_BB_REJECTION_THRESHOLD: float = 0.15


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
        if not self._pass_basic_filters(spread_pct, volume_24h_usd):
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
        if not check_rsi(ind_h1.get("rsi_last"), overbought=75, oversold=25, direction=direction.value):
            return None

        # Validate EMA200 bias
        if direction == Direction.LONG and close_h1 < ema200:
            return None
        if direction == Direction.SHORT and close_h1 > ema200:
            return None

        # Validate Bollinger rejection using percentile position within the band range.
        # This adapts to varying BB widths across regimes rather than using a fixed 2%
        # threshold that may be too permissive in tight-range environments.
        if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
            bb_position = (close_h1 - bb_lower) / (bb_upper - bb_lower)
            if direction == Direction.LONG and bb_position > _BB_REJECTION_THRESHOLD:
                return None  # Price too far from lower band — no BB rejection setup
            if direction == Direction.SHORT and bb_position < (1.0 - _BB_REJECTION_THRESHOLD):
                return None  # Price too far from upper band — no BB rejection setup
        else:
            # Fallback: zero-width bands or missing data — use original fixed threshold.
            if direction == Direction.LONG and bb_lower is not None:
                if close_h1 > bb_lower * 1.02:
                    return None
            if direction == Direction.SHORT and bb_upper is not None:
                if close_h1 < bb_upper * 0.98:
                    return None

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

        sig = build_channel_signal(
            config=self.config,
            symbol=symbol,
            direction=direction,
            close=close,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl_dist=sl_dist,
            id_prefix="SWING",
            atr_val=atr_val,
        )
        if sig is None:
            return None

        sig.risk_label = "Medium"

        # Mark signal quality tier based on daily confluence
        if daily_confluence:
            sig.setup_class = "SWING_D1_CONFLUENCE"
            sig.quality_tier = "A+"
        else:
            sig.setup_class = "SWING_STANDARD"

        return sig
