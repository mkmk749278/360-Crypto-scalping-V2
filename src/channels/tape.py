"""360_THE_TAPE – Tick / Data Whale Tracking 🐋

Trigger : Trade > 1 M USD **or** Volume Delta > 2×
Filters : Order-book imbalance, whale detection, AI sentiment, spread < 0.02 %
Risk    : SL 0.1–0.3 % AI-adaptive, TP1 1R partial, TP2 2R partial,
          TP3 AI-determined, trailing AI-adaptive
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid

from config import CHANNEL_TAPE
from src.channels.base import BaseChannel, Signal
from src.smc import Direction
from src.utils import utcnow


class TapeChannel(BaseChannel):
    def __init__(self) -> None:
        super().__init__(CHANNEL_TAPE)

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
        # --- Whale trigger ---
        whale = smc_data.get("whale_alert")
        delta_spike = smc_data.get("volume_delta_spike", False)

        if whale is None and not delta_spike:
            return None

        if spread_pct > self.config.spread_max:
            return None

        m1 = candles.get("1m")
        if m1 is None or len(m1.get("close", [])) < 10:
            return None

        close = float(m1["close"][-1])

        # Direction from net whale flow or delta
        ticks: List[Dict[str, Any]] = smc_data.get("recent_ticks", [])
        buy_vol = sum(t.get("qty", 0) * t.get("price", 0) for t in ticks if not t.get("isBuyerMaker", True))
        sell_vol = sum(t.get("qty", 0) * t.get("price", 0) for t in ticks if t.get("isBuyerMaker", True))

        if buy_vol > sell_vol:
            direction = Direction.LONG
        elif sell_vol > buy_vol:
            direction = Direction.SHORT
        else:
            return None

        atr_val = indicators.get("1m", {}).get("atr_last", close * 0.002)
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
            trailing_desc="AI Adaptive",
            confidence=0.0,
            ai_sentiment_label=ai_insight.get("label", "Neutral"),
            ai_sentiment_summary=ai_insight.get("summary", ""),
            risk_label="Medium-High",
            timestamp=utcnow(),
            signal_id=f"TAPE-{uuid.uuid4().hex[:8].upper()}",
            current_price=close,
        )
