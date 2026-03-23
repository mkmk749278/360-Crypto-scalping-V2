"""360_SPOT – H4/D1 Spot Accumulation Channel 📈

Trigger : H4/D1 accumulation breakout with sustained volume expansion
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

        # EMA200 filter — only LONG above EMA200 (spot accumulation is buy-only)
        ema200 = ind_h4.get("ema200_last")
        if ema200 is not None and close_h4 < ema200 * 0.98:
            return None

        # --- Accumulation breakout: price must clear recent H4 resistance ---
        highs = h4.get("high", [])
        if len(highs) < 20:
            return None
        recent_high = max(float(h) for h in highs[-20:-1])
        if close_h4 <= recent_high:
            return None  # No breakout yet

        # Volume expansion: current volume must exceed 20-bar average
        volumes = h4.get("volume", [])
        if len(volumes) < 20:
            return None
        avg_vol = sum(float(v) for v in volumes[-20:-1]) / 19
        current_vol = float(volumes[-1])
        if current_vol < avg_vol * 1.2:
            return None  # Insufficient volume expansion

        # SMC trigger (optional) — check for bearish MSS that would contradict accumulation
        mss = smc_data.get("mss")

        # Determine direction — spot channel is LONG-biased accumulation
        direction = Direction.LONG
        if mss is not None and mss.direction == Direction.SHORT:
            return None  # Structural short bias contradicts accumulation setup

        close = close_h4
        atr_val = ind_h4.get("atr_last", close * 0.01)

        # Wider SL for H4/D1 timeframe
        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val * 1.5)

        sl = close - sl_dist
        tp1 = close + sl_dist * self.config.tp_ratios[0]
        tp2 = close + sl_dist * self.config.tp_ratios[1]
        tp3 = close + sl_dist * self.config.tp_ratios[2]

        # Sanity check
        if sl >= close:
            return None

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
            risk_label="Conservative",
            timestamp=utcnow(),
            signal_id=f"SPOT-{uuid.uuid4().hex[:8].upper()}",
            current_price=close,
            original_sl_distance=sl_dist,
        )

        # DCA zone for spot accumulation
        if self.config.dca_enabled:
            dca_lower, dca_upper = compute_dca_zone(
                close, round(sl, 8), direction, self.config.dca_zone_range
            )
            sig.dca_zone_lower = dca_lower
            sig.dca_zone_upper = dca_upper
            sig.original_entry = close
            sig.original_tp1 = round(tp1, 8)
            sig.original_tp2 = round(tp2, 8)
            sig.original_tp3 = round(tp3, 8)

        return sig
