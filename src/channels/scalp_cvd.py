"""360_SCALP_CVD – CVD Divergence Scalp ⚡

Trigger : Bullish or bearish CVD divergence detected on 5m timeframe.
Logic   : BULLISH divergence (price makes new low, CVD higher low) → LONG
          BEARISH divergence (price makes new high, CVD lower high) → SHORT
Filters : Must be near a support/resistance zone (recent 20-bar high/low),
          standard quality gates (ADX, spread, volume)
Risk    : SL 0.15-0.3%, TP1 1R, TP2 2R
Signal ID prefix: "SCVD-"
"""

from __future__ import annotations

from typing import Dict, Optional

from config import CHANNEL_SCALP_CVD
from src.channels.base import BaseChannel, Signal, build_channel_signal
from src.filters import check_rsi
from src.smc import Direction
from src.utils import get_logger

log = get_logger("scalp_cvd")

# Price must be within this percentage of recent 20-bar high/low to be
# considered "at support/resistance".  Used as fallback when ATR is unavailable.
_SR_PROXIMITY_PCT: float = 0.8  # was 0.5; 0.5% is too tight for many valid setups

# CVD divergence recency and magnitude guards.
# Divergences older than this many candles are considered stale and skipped.
_CVD_MAX_AGE_CANDLES: int = 10
# Divergences weaker than this magnitude are considered noise and skipped.
_CVD_MIN_STRENGTH: float = 0.3

# When True, signals are rejected (fail-closed) if cvd_divergence_age or
# cvd_divergence_strength are missing from smc_data.  Set to False only for
# backward-compatibility during migration to the new SMCDetector that populates
# these fields.
_CVD_REQUIRE_METADATA: bool = True


class ScalpCVDChannel(BaseChannel):
    """CVD Divergence scalp trigger."""

    def __init__(self) -> None:
        super().__init__(CHANNEL_SCALP_CVD)

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
        m5 = candles.get("5m")
        if m5 is None or len(m5.get("close", [])) < 21:
            return None

        if not self._pass_basic_filters(spread_pct, volume_24h_usd):
            return None

        ind = indicators.get("5m", {})

        # ADX gate: CVD divergence is unreliable in strong trends (ADX > 35)
        # divergence can persist for 20+ candles without reverting
        adx_val = ind.get("adx_last")
        if adx_val is not None and adx_val > 35:
            return None

        # Use CVD divergence from smc_data (already detected by SMCDetector)
        cvd_div = smc_data.get("cvd_divergence")
        if cvd_div is None:
            return None

        # CVD divergence recency guard: stale divergences are less reliable.
        # Fail-closed when _CVD_REQUIRE_METADATA is True and the field is missing.
        cvd_div_age = smc_data.get("cvd_divergence_age")
        if cvd_div_age is None:
            if _CVD_REQUIRE_METADATA:
                log.warning(
                    "CVD signal rejected: cvd_divergence_age missing from smc_data "
                    "(set _CVD_REQUIRE_METADATA=False to allow through during migration)"
                )
                return None
        elif cvd_div_age > _CVD_MAX_AGE_CANDLES:
            return None

        # CVD divergence magnitude guard: weak divergences are noise.
        # Fail-closed when _CVD_REQUIRE_METADATA is True and the field is missing.
        cvd_div_strength = smc_data.get("cvd_divergence_strength")
        if cvd_div_strength is None:
            if _CVD_REQUIRE_METADATA:
                log.warning(
                    "CVD signal rejected: cvd_divergence_strength missing from smc_data "
                    "(set _CVD_REQUIRE_METADATA=False to allow through during migration)"
                )
                return None
        elif cvd_div_strength < _CVD_MIN_STRENGTH:
            return None

        closes = list(m5.get("close", []))
        if len(closes) < 20:
            return None

        close = float(closes[-1])
        recent_high = max(float(h) for h in list(m5.get("high", closes))[-20:])
        recent_low = min(float(l) for l in list(m5.get("low", closes))[-20:])

        # ATR-based S/R proximity: adapts to per-asset volatility.
        # Falls back to fixed _SR_PROXIMITY_PCT when ATR is not available.
        atr_val = ind.get("atr_last")

        if cvd_div == "BULLISH":
            direction = Direction.LONG
            # Must be near recent low (support)
            if atr_val is not None and atr_val > 0:
                if close > recent_low + atr_val * 1.0:
                    return None
            elif close > recent_low * (1 + _SR_PROXIMITY_PCT / 100):
                return None
        elif cvd_div == "BEARISH":
            direction = Direction.SHORT
            # Must be near recent high (resistance)
            if atr_val is not None and atr_val > 0:
                if close < recent_high - atr_val * 1.0:
                    return None
            elif close < recent_high * (1 - _SR_PROXIMITY_PCT / 100):
                return None
        else:
            return None

        # RSI extreme gate: don't chase overbought LONGs or fade oversold SHORTs
        if not check_rsi(ind.get("rsi_last"), overbought=75, oversold=25, direction=direction.value):
            return None

        atr_for_sl = atr_val if (atr_val is not None and atr_val > 0) else close * 0.002
        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_for_sl * 0.8)

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

        if direction == Direction.LONG and sl >= close:
            return None
        if direction == Direction.SHORT and sl <= close:
            return None

        _regime_ctx = smc_data.get("regime_context")
        _pair_profile = smc_data.get("pair_profile")
        return build_channel_signal(
            config=self.config,
            symbol=symbol,
            direction=direction,
            close=close,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl_dist=sl_dist,
            id_prefix="SCVD",
            atr_val=atr_for_sl,
            setup_class="CVD_DIVERGENCE",
            regime=regime,
            atr_percentile=_regime_ctx.atr_percentile if _regime_ctx else 50.0,
            pair_tier=_pair_profile.tier if _pair_profile else "MIDCAP",
        )
