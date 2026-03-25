"""360_SCALP_OBI – OBI Absorption Scalp ⚡

Trigger : Strong Order Book Imbalance (OBI) absorption pattern.
Logic   : OBI > 0.65 (strong bid absorption) + price near support → LONG
          OBI < -0.65 (strong ask absorption) + price near resistance → SHORT
          "At support/resistance" = within 0.5% of recent 20-bar low/high
Filters : Minimum volume threshold, spread gate
Risk    : Tight SL 0.1-0.2%, TP1 1R, TP2 1.5R
Signal ID prefix: "SOBI-"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid

from config import CHANNEL_SCALP_OBI
from src.channels.base import BaseChannel, Signal
from src.dca import compute_dca_zone
from src.filters import check_spread, check_volume
from src.smc import Direction
from src.utils import utcnow

# OBI thresholds for signal generation
_OBI_LONG_THRESHOLD: float = 0.65   # Strong bid absorption
_OBI_SHORT_THRESHOLD: float = -0.65  # Strong ask absorption

# Maximum distance from recent high/low to be considered at S/R
_SR_PROXIMITY_PCT: float = 0.5  # 0.5%


def _compute_obi(bids: List, asks: List) -> Optional[float]:
    """Compute depth-weighted Order Book Imbalance.

    Uses exponential depth weighting: level 1 = weight 1.0, deeper levels
    decay toward 0.  This reflects the reality that near-touch imbalance
    is far more predictive than deep-book imbalance.

    Returns OBI float in range [-1, 1], or None when data is insufficient.
    """
    try:
        weights = [1.0 / (1.0 + 0.25 * i) for i in range(10)]
        bid_qty = sum(float(b[1]) * w for b, w in zip(bids[:10], weights))
        ask_qty = sum(float(a[1]) * w for a, w in zip(asks[:10], weights))
        total = bid_qty + ask_qty
        if total <= 0:
            return None
        return (bid_qty - ask_qty) / total
    except (IndexError, TypeError, ValueError):
        return None


class ScalpOBIChannel(BaseChannel):
    """OBI Absorption scalp trigger."""

    def __init__(self) -> None:
        super().__init__(CHANNEL_SCALP_OBI)

    def evaluate(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        if not check_spread(spread_pct, self.config.spread_max):
            return None
        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        # Get order book from smc_data (set by scanner)
        order_book: Optional[Dict[str, Any]] = smc_data.get("order_book")
        if order_book is None:
            return None

        bids: List = order_book.get("bids", [])
        asks: List = order_book.get("asks", [])
        if not bids or not asks:
            return None

        obi = _compute_obi(bids, asks)
        if obi is None:
            return None

        # Get 5m candles for price context
        m5 = candles.get("5m")
        if m5 is None or len(m5.get("close", [])) < 20:
            return None

        closes = list(m5.get("close", []))
        highs = list(m5.get("high", closes))
        lows = list(m5.get("low", closes))

        close = float(closes[-1])
        recent_high = max(float(h) for h in highs[-20:])
        recent_low = min(float(l) for l in lows[-20:])

        # Determine direction based on OBI and price location
        direction: Optional[Direction] = None
        if obi >= _OBI_LONG_THRESHOLD:
            # Strong bid absorption — check if near support
            if close <= recent_low * (1 + _SR_PROXIMITY_PCT / 100):
                direction = Direction.LONG
        elif obi <= _OBI_SHORT_THRESHOLD:
            # Strong ask absorption — check if near resistance
            if close >= recent_high * (1 - _SR_PROXIMITY_PCT / 100):
                direction = Direction.SHORT

        if direction is None:
            return None

        # RSI extreme gate: don't chase overbought LONGs or fade oversold SHORTs
        ind = indicators.get("5m", {})
        rsi_last = ind.get("rsi_last")
        if rsi_last is not None:
            if direction == Direction.LONG and rsi_last > 75:
                return None
            if direction == Direction.SHORT and rsi_last < 25:
                return None

        atr_val = ind.get("atr_last", close * 0.001)
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
            signal_id=f"SOBI-{uuid.uuid4().hex[:8].upper()}",
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
        sig.setup_class = "OBI_ABSORPTION"

        # Entry zone: bracket around close ±ATR×0.3
        zone_half = atr_val * 0.3
        sig.entry_zone_low = round(close - zone_half, 8)
        sig.entry_zone_high = round(close + zone_half, 8)

        return sig
