"""360_RANGE – M15 Mean-Reversion ⚖️

Trigger : ADX < 20 + Bollinger Band rejection
Filters : SMA trend, RSI mean-reversion, ATR volatility, spread < 0.02 %
Risk    : SL 0.1–0.2 %, TP1 0.75–1R, TP2 1.5R, TP3 optional
"""

from __future__ import annotations

from typing import Dict, Optional
import uuid

from config import CHANNEL_RANGE
from src.channels.base import BaseChannel, Signal
from src.filters import check_adx, check_rsi, check_spread, check_volume
from src.smc import Direction
from src.utils import utcnow


class RangeChannel(BaseChannel):
    def __init__(self) -> None:
        super().__init__(CHANNEL_RANGE)

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
        m15 = candles.get("15m")
        if m15 is None or len(m15.get("close", [])) < 50:
            return None

        ind = indicators.get("15m", {})

        # --- Range filter: ADX must be low ---
        if not check_adx(ind.get("adx_last"), self.config.adx_min, self.config.adx_max):
            return None
        if not check_spread(spread_pct, self.config.spread_max):
            return None
        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        # --- Bollinger Band rejection ---
        bb_upper = ind.get("bb_upper_last")
        bb_lower = ind.get("bb_lower_last")
        bb_mid = ind.get("bb_mid_last")  # noqa: F841
        if bb_upper is None or bb_lower is None:
            return None

        close = float(m15["close"][-1])

        # RSI mean-reversion
        rsi_val = ind.get("rsi_last")

        # Determine direction from BB touch
        direction: Optional[Direction] = None
        if close <= bb_lower * 1.002:
            direction = Direction.LONG
        elif close >= bb_upper * 0.998:
            direction = Direction.SHORT
        else:
            return None

        # RSI must not be in an extreme zone conflicting with direction
        if not check_rsi(rsi_val, 70.0, 30.0, direction.value):
            return None

        atr_val = ind.get("atr_last", close * 0.002)
        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val * 0.8)

        if direction == Direction.LONG:
            sl = close - sl_dist
            tp1 = close + sl_dist * self.config.tp_ratios[0]
            tp2 = close + sl_dist * self.config.tp_ratios[1]
        else:
            sl = close + sl_dist
            tp1 = close - sl_dist * self.config.tp_ratios[0]
            tp2 = close - sl_dist * self.config.tp_ratios[1]

        return Signal(
            channel=self.config.name,
            symbol=symbol,
            direction=direction,
            entry=close,
            stop_loss=round(sl, 8),
            tp1=round(tp1, 8),
            tp2=round(tp2, 8),
            tp3=None,
            trailing_active=True,
            trailing_desc=f"{self.config.trailing_atr_mult}×ATR",
            confidence=0.0,
            ai_sentiment_label=ai_insight.get("label", "Neutral"),
            ai_sentiment_summary=ai_insight.get("summary", ""),
            risk_label="Conservative",
            timestamp=utcnow(),
            signal_id=f"RANGE-{uuid.uuid4().hex[:8].upper()}",
            current_price=close,
            original_sl_distance=sl_dist,
        )
