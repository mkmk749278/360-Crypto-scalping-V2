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

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import CHANNEL_SCALP_OBI
from src.channels.base import BaseChannel, Signal, build_channel_signal
from src.filters import check_rsi
from src.smc import Direction
from src.spoof_detect import check_spoof_gate
from src.utils import get_logger

log = get_logger("scalp_obi")

# OBI thresholds for signal generation
_OBI_LONG_THRESHOLD: float = 0.65   # Strong bid absorption
_OBI_SHORT_THRESHOLD: float = -0.65  # Strong ask absorption

# Maximum distance from recent high/low to be considered at S/R.
# Used as fallback when ATR is not available.
_SR_PROXIMITY_PCT: float = 0.5  # 0.5%

# Order book data older than this threshold (seconds) is considered stale
# and will cause the OBI signal to be skipped.
_OBI_MAX_STALENESS_SEC: float = 2.0

# Flag to emit a one-time warning when the order book lacks a timestamp.
# Using a list so it can be mutated from within functions without `global`.
_obi_ts_warning_state: list = [False]  # [warned]

# When True, signals are rejected (fail-closed) when the order book lacks a
# timestamp.  Set to False only for backward-compatibility testing.
_OBI_REQUIRE_TIMESTAMP: bool = True

# Minimum total USD depth across top-10 bid+ask levels.
# Books thinner than this threshold produce unreliable OBI readings.
_MIN_OB_DEPTH_USD: float = 100_000.0


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
        regime: str = "",
    ) -> Optional[Signal]:

        if not self._pass_basic_filters(spread_pct, volume_24h_usd):
            return None

        # Get order book from smc_data (set by scanner)
        order_book: Optional[Dict[str, Any]] = smc_data.get("order_book")
        if order_book is None:
            return None

        # Staleness guard: order book data older than _OBI_MAX_STALENESS_SEC
        # is considered unreliable for scalping decisions.
        ob_ts = order_book.get("timestamp") or order_book.get("fetched_at")
        if ob_ts is not None:
            try:
                if isinstance(ob_ts, datetime):
                    ts_float = ob_ts.timestamp()
                    now_float = datetime.now(timezone.utc).timestamp()
                else:
                    ts_float = float(ob_ts)
                    now_float = time.time()
                if now_float - ts_float > _OBI_MAX_STALENESS_SEC:
                    return None
            except (TypeError, ValueError, OSError):
                pass  # Unrecognised timestamp format — fail open
        else:
            # No timestamp: emit a one-time warning.
            # When _OBI_REQUIRE_TIMESTAMP is True (default), reject the signal
            # (fail-closed) so stale order book data cannot drive a scalp entry.
            if not _obi_ts_warning_state[0]:
                log.warning(
                    "OBI order book data missing timestamp; staleness guard inactive. "
                    "Scanner should include 'fetched_at' in order_book data."
                )
                _obi_ts_warning_state[0] = True
            if _OBI_REQUIRE_TIMESTAMP:
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

        # ATR-based S/R proximity: adapts to per-asset volatility.
        # Falls back to fixed _SR_PROXIMITY_PCT when ATR is not available.
        ind = indicators.get("5m", {})
        atr_val = ind.get("atr_last")

        # Determine direction based on OBI and price location
        direction: Optional[Direction] = None
        if obi >= _OBI_LONG_THRESHOLD:
            # Strong bid absorption — check if near support
            if atr_val is not None and atr_val > 0:
                near_support = close <= recent_low + atr_val * 1.0
            else:
                near_support = close <= recent_low * (1 + _SR_PROXIMITY_PCT / 100)
            if near_support:
                direction = Direction.LONG
        elif obi <= _OBI_SHORT_THRESHOLD:
            # Strong ask absorption — check if near resistance
            if atr_val is not None and atr_val > 0:
                near_resistance = close >= recent_high - atr_val * 1.0
            else:
                near_resistance = close >= recent_high * (1 - _SR_PROXIMITY_PCT / 100)
            if near_resistance:
                direction = Direction.SHORT

        if direction is None:
            return None

        # Minimum order book depth guard: thin books produce unreliable OBI.
        bid_usd = sum(float(b[0]) * float(b[1]) for b in bids[:10])
        ask_usd = sum(float(a[0]) * float(a[1]) for a in asks[:10])
        if bid_usd + ask_usd < _MIN_OB_DEPTH_USD:
            return None  # Order book too thin for reliable OBI

        # Spoofing / layering gate: reject if the order book shows manipulation
        # patterns on the side opposing our trade direction.
        allowed, spoof_reason = check_spoof_gate(direction.value, order_book, close)
        if not allowed:
            log.debug("OBI signal rejected by spoof gate: %s", spoof_reason)
            return None

        # RSI extreme gate: don't chase overbought LONGs or fade oversold SHORTs
        if not check_rsi(ind.get("rsi_last"), overbought=75, oversold=25, direction=direction.value):
            return None

        atr_for_sl = atr_val if (atr_val is not None and atr_val > 0) else close * 0.001
        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_for_sl * 0.5)

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
            id_prefix="SOBI",
            atr_val=atr_for_sl,
            setup_class="OBI_ABSORPTION",
            regime=regime,
            atr_percentile=_regime_ctx.atr_percentile if _regime_ctx else 50.0,
            pair_tier=_pair_profile.tier if _pair_profile else "MIDCAP",
        )
        if sig is not None:
            sig.analyst_reason = f"OBI={obi:.3f} (threshold ±{_OBI_LONG_THRESHOLD})"
        return sig
