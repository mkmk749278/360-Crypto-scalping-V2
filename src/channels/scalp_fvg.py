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

from config import CHANNEL_SCALP_FVG
from src.channels.base import BaseChannel, Signal, build_channel_signal
from src.filters import check_adx, check_rsi
from src.smc import Direction

# Maximum distance from FVG zone boundary (as fraction of zone width) to be
# considered "retesting" the zone.  0.5 means price must be within 50% of the
# zone width from the zone boundary.
_FVG_RETEST_PROXIMITY: float = 0.35  # was 0.5; tighter = higher-probability retests

# FVG zone age management.
# Zones older than _FVG_MAX_AGE_CANDLES are skipped (low institutional relevance).
_FVG_MAX_AGE_CANDLES: int = 80
# Minimum decay factor applied to SL distance for very old zones.
# Decay = max(_FVG_MIN_DECAY, 1.0 - candles_ago / 100.0)
# A decay of 1.0 = full SL; 0.2 = 20% of original SL (tightest allowed).
_FVG_MIN_DECAY: float = 0.2


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
        regime: str = "",
    ) -> Optional[Signal]:
        # Try 5m first, fall back to 15m
        for tf in ("5m", "15m"):
            sig = self._evaluate_tf(
                symbol, tf, candles, indicators, smc_data, spread_pct, volume_24h_usd, regime
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
        regime: str = "",
    ) -> Optional[Signal]:
        cd = candles.get(tf)
        if cd is None or len(cd.get("close", [])) < 20:
            return None

        ind = indicators.get(tf, {})
        if not check_adx(ind.get("adx_last"), self.config.adx_min):
            return None
        if not self._pass_basic_filters(spread_pct, volume_24h_usd):
            return None

        fvg_zones = smc_data.get("fvg", [])
        if not fvg_zones:
            return None

        close = float(cd["close"][-1])
        atr_val = ind.get("atr_last", close * 0.002)

        # Total candle count in the current timeframe window (used for age calculation).
        total_candles = len(cd.get("close", []))

        # Find the most recent FVG zone that price is retesting
        direction: Optional[Direction] = None
        retest_zone = None
        zone_decay: float = 1.0  # Age decay factor for the matched zone
        for zone in fvg_zones:
            gap_high = float(zone.gap_high)
            gap_low = float(zone.gap_low)
            zone_width = gap_high - gap_low
            if zone_width <= 0:
                continue

            # Age-based filtering: zones older than _FVG_MAX_AGE_CANDLES are skipped.
            # FVGZone.index is the bar index where the gap was created; the number of
            # candles since creation is (total_candles - zone.index).
            candles_ago = total_candles - int(zone.index)
            if candles_ago > _FVG_MAX_AGE_CANDLES:
                continue  # Zone too old — skip regardless of price proximity

            if zone.direction == Direction.LONG:
                # Bullish FVG (gap up): price should retest from above (touching gap_high)
                # LONG entry when price is near the top of the bullish FVG
                proximity = (close - gap_high) / zone_width if zone_width > 0 else 1.0
                if abs(proximity) <= _FVG_RETEST_PROXIMITY:
                    direction = Direction.LONG
                    retest_zone = zone
                    zone_decay = max(_FVG_MIN_DECAY, 1.0 - candles_ago / 100.0)
                    break
            else:
                # Bearish FVG (gap down): price should retest from below (touching gap_low)
                # SHORT entry when price is near the bottom of the bearish FVG
                proximity = (gap_low - close) / zone_width if zone_width > 0 else 1.0
                if abs(proximity) <= _FVG_RETEST_PROXIMITY:
                    direction = Direction.SHORT
                    retest_zone = zone
                    zone_decay = max(_FVG_MIN_DECAY, 1.0 - candles_ago / 100.0)
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
        if not check_rsi(ind.get("rsi_last"), overbought=75, oversold=25, direction=direction.value):
            return None

        gap_high = float(retest_zone.gap_high)
        gap_low = float(retest_zone.gap_low)

        # SL: below/above FVG zone boundary, scaled by age decay factor.
        # Older zones carry less conviction (tighter SL = smaller loss if wrong).
        if direction == Direction.LONG:
            sl = min(gap_low - atr_val * 0.5, close * (1 - self.config.sl_pct_range[0] / 100))
        else:
            sl = max(gap_high + atr_val * 0.5, close * (1 + self.config.sl_pct_range[0] / 100))

        base_sl_dist = abs(close - sl)
        sl_dist = base_sl_dist * zone_decay  # Apply age decay (older zone → tighter SL)
        if sl_dist <= 0:
            return None

        # Recompute the actual SL price from the decayed distance.
        if direction == Direction.LONG:
            sl = close - sl_dist
        else:
            sl = close + sl_dist

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

        _regime_ctx = smc_data.get("regime_context")
        _pair_profile = smc_data.get("pair_profile")
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
            id_prefix="SFVG",
            atr_val=atr_val,
            setup_class="FVG_RETEST",
            regime=regime,
            atr_percentile=_regime_ctx.atr_percentile if _regime_ctx else 50.0,
            pair_tier=_pair_profile.tier if _pair_profile else "MIDCAP",
        )

        if sig is not None:
            # P2-15: annotate with zone age and decay for transparency
            sig.analyst_reason = (
                f"FVG retest {candles_ago} candles ago, decay={zone_decay:.2f}"
            )

            # P1-7: HTF FVG confluence soft-boost
            htf = "15m" if tf == "5m" else None
            if htf and self._has_htf_confluence(
                smc_data, direction, float(retest_zone.gap_high), float(retest_zone.gap_low), htf
            ):
                sig.setup_class = "FVG_RETEST_HTF_CONFLUENCE"
                sig.quality_tier = "A+"

        return sig

    def _has_htf_confluence(
        self,
        smc_data: dict,
        direction: Direction,
        gap_high: float,
        gap_low: float,
        htf: str,
    ) -> bool:
        """Return True when a higher-timeframe FVG overlaps the current zone.

        Checks whether any FVG in *smc_data* has the same direction as *direction*
        and its price range overlaps ``[gap_low, gap_high]``.  The *htf* parameter
        is informational (the caller already filtered which timeframe to check).

        Parameters
        ----------
        smc_data:
            SMC data dict passed to *evaluate()*.
        direction:
            Trade direction of the current lower-timeframe FVG.
        gap_high:
            Upper boundary of the lower-timeframe FVG zone.
        gap_low:
            Lower boundary of the lower-timeframe FVG zone.
        htf:
            Name of the higher timeframe being checked (e.g. ``"15m"``).
            Currently unused internally but kept for documentation clarity.

        Returns
        -------
        bool
            ``True`` when at least one HTF FVG zone overlaps the LTF zone.
        """
        fvg_zones = smc_data.get("fvg", [])
        for zone in fvg_zones:
            if zone.direction != direction:
                continue
            z_high = float(zone.gap_high)
            z_low = float(zone.gap_low)
            # Overlap check: ranges must intersect
            if z_low <= gap_high and z_high >= gap_low:
                return True
        return False
