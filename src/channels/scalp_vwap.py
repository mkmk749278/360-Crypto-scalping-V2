"""360_SCALP_VWAP – VWAP Band Bounce Scalp ⚡

Trigger : Price touches VWAP ±1SD band with volume confirmation.
Logic   : Price touches lower_band_1 (VWAP − 1SD) → LONG (mean reversion)
          Price touches upper_band_1 (VWAP + 1SD) → SHORT (mean reversion)
Filters : Must be in RANGING or QUIET regime (mean-reversion only, not trending)
          Current volume > 1.3× average (volume confirmation)
Risk    : SL beyond ±2SD band, TP at VWAP center
Signal ID prefix: "SVWP-"
"""

from __future__ import annotations

from typing import Dict, Optional
import uuid

from config import CHANNEL_SCALP_VWAP
from src.channels.base import BaseChannel, Signal
from src.dca import compute_dca_zone
from src.filters import check_spread, check_volume
from src.regime import MarketRegime
from src.smc import Direction
from src.utils import utcnow
from src.vwap import compute_vwap

# Minimum volume ratio (current / average) required to confirm the bounce
_MIN_VOLUME_RATIO: float = 1.3

# Regimes where VWAP bounce scalps are valid (mean-reversion only)
_VALID_REGIMES = frozenset({MarketRegime.RANGING, MarketRegime.QUIET})


class ScalpVWAPChannel(BaseChannel):
    """VWAP Band Bounce scalp trigger."""

    def __init__(self) -> None:
        super().__init__(CHANNEL_SCALP_VWAP)

    def evaluate(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        # Regime gate: only valid in RANGING or QUIET
        # Note: regime_result is not directly available in evaluate(); the
        # scanner's regime gating happens upstream.  We rely on the scanner
        # to have already filtered out trending regimes via the regime-channel
        # compatibility matrix.  For defensive validation, check ADX as a proxy.
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

        if not check_spread(spread_pct, self.config.spread_max):
            return None
        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        ind = indicators.get(tf, {})

        # ADX check: only valid in low-ADX (ranging/quiet) environment
        adx_val = ind.get("adx_last")
        if adx_val is not None and adx_val > self.config.adx_max:
            return None

        highs = list(cd.get("high", []))
        lows = list(cd.get("low", []))
        closes = list(cd.get("close", []))
        volumes = list(cd.get("volume", []))

        if len(closes) < 20 or len(volumes) < 20:
            return None

        # Compute VWAP with ±1SD bands
        vwap_result = compute_vwap(highs[-50:], lows[-50:], closes[-50:], volumes[-50:])
        if vwap_result is None:
            return None

        close = float(closes[-1])
        lower_band_1 = vwap_result.lower_band_1
        upper_band_1 = vwap_result.upper_band_1
        lower_band_2 = vwap_result.lower_band_2
        upper_band_2 = vwap_result.upper_band_2
        vwap_mid = vwap_result.vwap

        # Volume confirmation: current volume > 1.3× average
        avg_vol = sum(float(v) for v in volumes[-20:-1]) / 19
        current_vol = float(volumes[-1])
        if avg_vol <= 0 or current_vol < avg_vol * _MIN_VOLUME_RATIO:
            return None

        # Determine direction based on VWAP band touch
        direction: Optional[Direction] = None
        if close <= lower_band_1:
            direction = Direction.LONG
        elif close >= upper_band_1:
            direction = Direction.SHORT
        else:
            return None

        # RSI extreme gate: don't chase overbought LONGs or fade oversold SHORTs
        rsi_last = ind.get("rsi_last")
        if rsi_last is not None:
            if direction == Direction.LONG and rsi_last > 75:
                return None
            if direction == Direction.SHORT and rsi_last < 25:
                return None

        # SL: beyond ±2SD band
        if direction == Direction.LONG:
            sl = lower_band_2 - (vwap_result.std_dev * 0.1)
            tp1 = vwap_mid  # TP at VWAP center
        else:
            sl = upper_band_2 + (vwap_result.std_dev * 0.1)
            tp1 = vwap_mid  # TP at VWAP center

        sl_dist = abs(close - sl)
        if sl_dist <= 0:
            return None

        if direction == Direction.LONG:
            tp2 = close + sl_dist * self.config.tp_ratios[1]
            tp3 = close + sl_dist * self.config.tp_ratios[2]
        else:
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
            signal_id=f"SVWP-{uuid.uuid4().hex[:8].upper()}",
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
        sig.setup_class = "VWAP_BOUNCE"

        # Entry zone: bracket around close ±ATR×0.3
        # 0.002 (0.2%) matches the fallback already used in _evaluate_tf above
        # and is conservative enough to remain valid for ranging/VWAP setups.
        atr_val = ind.get("atr_last", close * 0.002)
        zone_half = atr_val * 0.3
        sig.entry_zone_low = round(close - zone_half, 8)
        sig.entry_zone_high = round(close + zone_half, 8)

        return sig
