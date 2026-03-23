"""360_SCALP – M1/M5 High-Frequency Scalping ⚡

Trigger : M5 Liquidity Sweep + Momentum > 0.15 % over 3 candles
          RANGE_FADE path: BB mean-reversion (price at lower/upper BB + RSI divergence)
          WHALE_MOMENTUM path: large volume spike + OBI imbalance
Filters : EMA alignment, ADX > 20, ATR-based volatility, spread < 0.02 %, liquidity
Risk    : SL 0.05–0.1 %, TP1 0.5–1R, TP2 1–1.5R, TP3 optional 20 %, Trailing 1.5–2×ATR
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid


from config import CHANNEL_SCALP
from src.channels.base import BaseChannel, Signal
from src.dca import compute_dca_zone
from src.filters import check_adx, check_ema_alignment, check_rsi, check_spread, check_volume
from src.smc import Direction
from src.utils import utcnow

# WHALE_MOMENTUM thresholds (absorbed from former TapeChannel)
_WHALE_DELTA_MIN_RATIO: float = 2.0
_WHALE_MIN_TICK_VOLUME_USD: float = 500_000.0
_WHALE_OBI_MIN: float = 1.5


class ScalpChannel(BaseChannel):
    def __init__(self) -> None:
        super().__init__(CHANNEL_SCALP)

    def evaluate(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        # Try each setup class path in priority order
        return (
            self._evaluate_standard(symbol, candles, indicators, smc_data, spread_pct, volume_24h_usd)
            or self._evaluate_range_fade(symbol, candles, indicators, smc_data, spread_pct, volume_24h_usd)
            or self._evaluate_whale_momentum(symbol, candles, indicators, smc_data, spread_pct, volume_24h_usd)
        )

    # ------------------------------------------------------------------
    # Standard scalp path (TREND_PULLBACK / BREAKOUT / LIQUIDITY_SWEEP)
    # ------------------------------------------------------------------

    def _evaluate_standard(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        m5 = candles.get("5m")
        if m5 is None or len(m5.get("close", [])) < 50:
            return None

        ind = indicators.get("5m", {})
        if not check_adx(ind.get("adx_last"), self.config.adx_min):
            return None
        if not check_spread(spread_pct, self.config.spread_max):
            return None
        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        ema_fast = ind.get("ema9_last")
        ema_slow = ind.get("ema21_last")
        if ema_fast is None or ema_slow is None:
            return None

        sweeps = smc_data.get("sweeps", [])
        if not sweeps:
            return None
        sweep = sweeps[0]

        mom = ind.get("momentum_last")
        if mom is None or abs(mom) < 0.15:
            return None

        direction = sweep.direction
        if not check_ema_alignment(ema_fast, ema_slow, direction.value):
            return None

        close = float(m5["close"][-1])
        atr_val = ind.get("atr_last", close * 0.001)

        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val * 0.5)
        sl, tp1, tp2, tp3 = self._calc_levels(close, sl_dist, direction)

        if direction == Direction.LONG and sl >= close:
            return None
        if direction == Direction.SHORT and sl <= close:
            return None

        return self._build_signal(
            symbol, direction, close, sl, tp1, tp2, tp3, sl_dist, "SCALP"
        )

    # ------------------------------------------------------------------
    # RANGE_FADE path (absorbed from former RangeChannel)
    # BB mean-reversion: price touching lower/upper BB + RSI divergence
    # ------------------------------------------------------------------

    def _evaluate_range_fade(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        m5 = candles.get("5m")
        if m5 is None or len(m5.get("close", [])) < 50:
            return None

        ind = indicators.get("5m", {})

        # Range fade uses ADX in low-range territory (ADX < 25)
        adx_val = ind.get("adx_last")
        if adx_val is not None and adx_val > 25:
            return None

        if not check_spread(spread_pct, self.config.spread_max):
            return None
        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        bb_upper = ind.get("bb_upper_last")
        bb_lower = ind.get("bb_lower_last")
        if bb_upper is None or bb_lower is None:
            return None

        close = float(m5["close"][-1])
        rsi_val = ind.get("rsi_last")

        direction: Optional[Direction] = None
        if close <= bb_lower * 1.002:
            direction = Direction.LONG
        elif close >= bb_upper * 0.998:
            direction = Direction.SHORT
        else:
            return None

        if not check_rsi(rsi_val, 70.0, 30.0, direction.value):
            return None

        atr_val = ind.get("atr_last", close * 0.002)
        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val * 0.8)
        sl, tp1, tp2, tp3 = self._calc_levels(close, sl_dist, direction)

        sig = self._build_signal(symbol, direction, close, sl, tp1, tp2, tp3, sl_dist, "RANGE-FADE")
        if sig is not None:
            sig.setup_class = "RANGE_FADE"
        return sig

    # ------------------------------------------------------------------
    # WHALE_MOMENTUM path (absorbed from former TapeChannel)
    # Large volume spike + OBI imbalance
    # ------------------------------------------------------------------

    def _evaluate_whale_momentum(
        self,
        symbol: str,
        candles: Dict[str, dict],
        indicators: Dict[str, dict],
        smc_data: dict,
        spread_pct: float,
        volume_24h_usd: float,
    ) -> Optional[Signal]:
        whale = smc_data.get("whale_alert")
        delta_spike = smc_data.get("volume_delta_spike", False)
        if whale is None and not delta_spike:
            return None

        if not check_spread(spread_pct, self.config.spread_max):
            return None
        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        m1 = candles.get("1m")
        if m1 is None or len(m1.get("close", [])) < 10:
            return None

        close = float(m1["close"][-1])

        ticks: List[Dict[str, Any]] = smc_data.get("recent_ticks", [])
        buy_vol = sum(
            t.get("qty", 0) * t.get("price", 0)
            for t in ticks if not t.get("isBuyerMaker", True)
        )
        sell_vol = sum(
            t.get("qty", 0) * t.get("price", 0)
            for t in ticks if t.get("isBuyerMaker", True)
        )

        total_vol = buy_vol + sell_vol
        if total_vol < _WHALE_MIN_TICK_VOLUME_USD:
            return None

        if buy_vol >= sell_vol * _WHALE_DELTA_MIN_RATIO:
            direction = Direction.LONG
        elif sell_vol >= buy_vol * _WHALE_DELTA_MIN_RATIO:
            direction = Direction.SHORT
        else:
            return None

        # Order book imbalance check
        order_book = smc_data.get("order_book")
        if order_book is not None:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            bid_depth = sum(float(b[1]) * float(b[0]) for b in bids[:10])
            ask_depth = sum(float(a[1]) * float(a[0]) for a in asks[:10])
            if bid_depth > 0 and ask_depth > 0:
                imbalance_ratio = (
                    bid_depth / ask_depth if direction == Direction.LONG else ask_depth / bid_depth
                )
                if imbalance_ratio < _WHALE_OBI_MIN:
                    return None

        atr_val = indicators.get("1m", {}).get("atr_last", close * 0.002)
        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val)
        sl, tp1, tp2, tp3 = self._calc_levels(close, sl_dist, direction)

        sig = self._build_signal(symbol, direction, close, sl, tp1, tp2, tp3, sl_dist, "WHALE")
        if sig is not None:
            sig.setup_class = "WHALE_MOMENTUM"
        return sig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_levels(
        self, close: float, sl_dist: float, direction: Direction
    ):
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
        return sl, tp1, tp2, tp3

    def _build_signal(
        self,
        symbol: str,
        direction: Direction,
        close: float,
        sl: float,
        tp1: float,
        tp2: float,
        tp3: float,
        sl_dist: float,
        id_prefix: str,
    ) -> Optional[Signal]:
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
            signal_id=f"{id_prefix}-{uuid.uuid4().hex[:8].upper()}",
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

        return sig
