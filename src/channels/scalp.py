"""360_SCALP – M1/M5 High-Frequency Scalping ⚡

Trigger : M5 Liquidity Sweep + Momentum > 0.3 % over 3 candles
Filters : EMA alignment, ADX > 25, ATR-based volatility, spread < 0.02 %, liquidity
Risk    : SL 0.05–0.1 %, TP1 0.5–1R, TP2 1–1.5R, TP3 optional 20 %, Trailing 1.5–2×ATR
"""

from __future__ import annotations

from typing import Dict, Optional
import uuid


from config import CHANNEL_SCALP
from src.channels.base import BaseChannel, Signal
from src.filters import check_adx, check_ema_alignment, check_spread, check_volume
from src.smc import Direction
from src.utils import utcnow


class ScalpChannel(BaseChannel):
    def __init__(self) -> None:
        super().__init__(CHANNEL_SCALP)

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
        m5 = candles.get("5m")
        if m5 is None or len(m5.get("close", [])) < 50:
            return None

        # --- Filters ---
        ind = indicators.get("5m", {})
        if not check_adx(ind.get("adx_last"), self.config.adx_min):
            return None

        if not check_spread(spread_pct, self.config.spread_max):
            return None

        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        # EMA alignment check (fast > slow for LONG, opposite for SHORT)
        ema_fast = ind.get("ema9_last")
        ema_slow = ind.get("ema21_last")
        if ema_fast is None or ema_slow is None:
            return None

        # --- SMC trigger ---
        sweeps = smc_data.get("sweeps", [])
        if not sweeps:
            return None
        sweep = sweeps[0]

        # Momentum > 0.3 % over 3 candles
        mom = ind.get("momentum_last")
        if mom is None or abs(mom) < 0.3:
            return None

        direction = sweep.direction
        # EMA alignment must agree
        if not check_ema_alignment(ema_fast, ema_slow, direction.value):
            return None

        close = float(m5["close"][-1])
        atr_val = ind.get("atr_last", close * 0.001)

        # Risk levels
        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val * 0.5)
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

        sentiment_label = ai_insight.get("label", "Neutral")
        sentiment_summary = ai_insight.get("summary", "")

        # Sanity check: SL must be on the correct side of entry
        if direction == Direction.LONG and sl >= close:
            return None
        if direction == Direction.SHORT and sl <= close:
            return None

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
            confidence=0.0,  # filled later by scorer
            ai_sentiment_label=sentiment_label,
            ai_sentiment_summary=sentiment_summary,
            risk_label="Aggressive",
            timestamp=utcnow(),
            signal_id=f"SCALP-{uuid.uuid4().hex[:8].upper()}",
            current_price=close,
            original_sl_distance=sl_dist,
        )
