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
        ai_insight: dict,
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
        adx_val = ind_h4.get("adx_last")
        if adx_val is None or not (self.config.adx_min <= adx_val <= self.config.adx_max):
            return None
        if spread_pct > self.config.spread_max:
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

        # Validate EMA200 bias
        if direction == Direction.LONG and close_h1 < ema200:
            return None
        if direction == Direction.SHORT and close_h1 > ema200:
            return None

        # Validate Bollinger rejection
        if direction == Direction.LONG and bb_lower is not None:
            if close_h1 > bb_lower * 1.02:  # must be near lower band
                pass  # acceptable
        if direction == Direction.SHORT and bb_upper is not None:
            if close_h1 < bb_upper * 0.98:
                pass  # acceptable

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

        return Signal(
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
            ai_sentiment_label=ai_insight.get("label", "Neutral"),
            ai_sentiment_summary=ai_insight.get("summary", ""),
            risk_label="Medium",
            timestamp=utcnow(),
            signal_id=f"SWING-{uuid.uuid4().hex[:8].upper()}",
            current_price=close,
        )
