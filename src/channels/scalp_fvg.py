"""360_SCALP_FVG – Fair Value Gap Retest Scalp ⚡

Trigger : Price retests an unfilled FVG zone on 5m or 15m timeframe.
Logic   : Bullish FVG retest (gap-up zone from above) → LONG
          Bearish FVG retest (gap-down zone from below) → SHORT
Filters : Same quality gates as regular scalp (ADX, spread, volume, regime)
Risk    : SL below/above FVG zone boundary, TP1 1.5R, TP2 2.5R
Signal ID prefix: "SFVG-"
"""

from __future__ import annotations

from typing import Dict, Optional
import uuid

from config import CHANNEL_SCALP_FVG
from src.channels.base import BaseChannel, Signal
from src.dca import compute_dca_zone
from src.filters import check_adx, check_spread, check_volume
from src.smc import Direction
from src.utils import utcnow

# Maximum distance from FVG zone boundary (as fraction of zone width) to be
# considered "retesting" the zone.  0.5 means price must be within 50% of the
# zone width from the zone boundary.
_FVG_RETEST_PROXIMITY: float = 0.35  # was 0.5; tighter = higher-probability retests


class ScalpFVGChannel(BaseChannel):
    """FVG Retest scalp trigger."""

    def __init__(self) -> None:
        super().__init__(CHANNEL_SCALP_FVG)

    def evaluate(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        # Try 5m first, fall back to 15m
        for tf in ("5m", "15m"):
            sig = self._evaluate_tf(
                symbol, tf, candles, indicators, smc_data, spread_pct, volume_24h_usd
            )
            if sig is not None:
                return sig
        return None

    def _evaluate_tf(
        self,
        symbol: str,
        tf: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        cd = candles.get(tf)
        if cd is None or len(cd.get("close", [])) < 20:
            return None

        ind = indicators.get(tf, {})
        if not check_adx(ind.get("adx_last"), self.config.adx_min):
            return None
        if not check_spread(spread_pct, self.config.spread_max):
            return None
        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        fvg_zones = smc_data.get("fvg", [])
        if not fvg_zones:
            return None

        close = float(cd["close"][-1])
        atr_val = ind.get("atr_last", close * 0.002)

        # Find the most recent FVG zone that price is retesting
        direction: Optional[Direction] = None
        retest_zone = None
        for zone in fvg_zones:
            gap_high = float(zone.gap_high)
            gap_low = float(zone.gap_low)
            zone_width = gap_high - gap_low
            if zone_width <= 0:
                continue

            if zone.direction == Direction.LONG:
                # Bullish FVG (gap up): price should retest from above (touching gap_high)
                # LONG entry when price is near the top of the bullish FVG
                proximity = (close - gap_high) / zone_width if zone_width > 0 else 1.0
                if abs(proximity) <= _FVG_RETEST_PROXIMITY:
                    direction = Direction.LONG
                    retest_zone = zone
                    break
            else:
                # Bearish FVG (gap down): price should retest from below (touching gap_low)
                # SHORT entry when price is near the bottom of the bearish FVG
                proximity = (gap_low - close) / zone_width if zone_width > 0 else 1.0
                if abs(proximity) <= _FVG_RETEST_PROXIMITY:
                    direction = Direction.SHORT
                    retest_zone = zone
                    break

        if direction is None or retest_zone is None:
            return None

        # FVG partial fill check: reject zones that are >60% filled
        # A heavily-filled FVG has much weaker expected bounce
        gap_high_z = float(retest_zone.gap_high)
        gap_low_z = float(retest_zone.gap_low)
        zone_width_z = gap_high_z - gap_low_z
        if zone_width_z > 0:
            if retest_zone.direction == Direction.LONG:
                # For bullish FVG: how much of the gap has price already filled from above?
                fill_pct = max(0.0, (gap_high_z - close) / zone_width_z)
            else:
                # For bearish FVG: how much has price filled from below?
                fill_pct = max(0.0, (close - gap_low_z) / zone_width_z)
            if fill_pct > 0.6:
                return None  # Zone >60% filled, weak bounce expected

        # RSI extreme gate: don't chase overbought LONGs or fade oversold SHORTs
        rsi_last = ind.get("rsi_last")
        if rsi_last is not None:
            if direction == Direction.LONG and rsi_last > 75:
                return None
            if direction == Direction.SHORT and rsi_last < 25:
                return None

        gap_high = float(retest_zone.gap_high)
        gap_low = float(retest_zone.gap_low)

        # SL: below/above FVG zone boundary
        if direction == Direction.LONG:
            sl = min(gap_low - atr_val * 0.5, close * (1 - self.config.sl_pct_range[0] / 100))
        else:
            sl = max(gap_high + atr_val * 0.5, close * (1 + self.config.sl_pct_range[0] / 100))

        sl_dist = abs(close - sl)
        if sl_dist <= 0:
            return None

        if direction == Direction.LONG:
            tp1 = close + sl_dist * self.config.tp_ratios[0]
            tp2 = close + sl_dist * self.config.tp_ratios[1]
            tp3 = close + sl_dist * self.config.tp_ratios[2]
        else:
            tp1 = close - sl_dist * self.config.tp_ratios[0]
            tp2 = close - sl_dist * self.config.tp_ratios[1]
            tp3 = close - sl_dist * self.config.tp_ratios[2]

        if direction == Direction.LONG and sl >= close:
            return None
        if direction == Direction.SHORT and sl <= close:
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
            risk_label="Aggressive",
            timestamp=utcnow(),
            signal_id=f"SFVG-{uuid.uuid4().hex[:8].upper()}",
            current_price=close,
            original_sl_distance=sl_dist,
        )

        dca_lower, dca_upper = compute_dca_zone(
            close, round(sl, 8), direction, self.config.dca_zone_range
        )
        sig.dca_zone_lower = dca_lower
        sig.dca_zone_upper = dca_upper
        sig.original_entry = close
        sig.original_tp1 = round(tp1, 8)
        sig.original_tp2 = round(tp2, 8)
        sig.original_tp3 = round(tp3, 8)
        sig.setup_class = "FVG_RETEST"

        # Entry zone: bracket around close ±ATR×0.3
        zone_half = atr_val * 0.3
        sig.entry_zone_low = round(close - zone_half, 8)
        sig.entry_zone_high = round(close + zone_half, 8)

        return sig
