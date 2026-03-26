"""360_SCALP – M1/M5 High-Frequency Scalping ⚡

Trigger : M5 Liquidity Sweep + Momentum > 0.15 % over 3 candles
          RANGE_FADE path: BB mean-reversion (price at lower/upper BB + RSI divergence)
          WHALE_MOMENTUM path: large volume spike + OBI imbalance
Filters : EMA alignment, ADX > 20, ATR-based volatility, spread < 0.02 %, liquidity
Risk    : SL 0.05–0.1 %, TP1 0.5–1R, TP2 1–1.5R, TP3 optional 20 %, Trailing 1.5–2×ATR
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


from config import CHANNEL_SCALP
from src.channels.base import BaseChannel, Signal, build_channel_signal
from src.filters import (
    check_adx,
    check_macd_confirmation,
    check_rsi_regime,
    check_ema_alignment_adaptive,
    check_spread_adaptive,
    check_volume,
)
from src.mtf import mtf_gate_scalp_standard, mtf_gate_scalp_range_fade
from src.smc import Direction

# WHALE_MOMENTUM thresholds (absorbed from former TapeChannel)
_WHALE_DELTA_MIN_RATIO: float = 2.0
_WHALE_MIN_TICK_VOLUME_USD: float = 500_000.0
_WHALE_OBI_MIN: float = 1.5


class ScalpChannel(BaseChannel):
    def __init__(self) -> None:
        super().__init__(CHANNEL_SCALP)

    def _pass_basic_filters(self, spread_pct: float, volume_24h_usd: float, regime: str = "") -> bool:
        """Return True if basic spread/volume filters pass (regime-adaptive spread check)."""
        return (
            check_spread_adaptive(spread_pct, self.config.spread_max, regime=regime)
            and check_volume(volume_24h_usd, self.config.min_volume)
        )

    def _select_indicator_weights(self, regime: str) -> dict:
        """Return indicator weight multipliers for the current regime.

        The weights are applied as a confidence boost multiplier to each
        candidate signal so that regime-appropriate setups are preferred
        when multiple candidates are available.

        Parameters
        ----------
        regime:
            Current market regime string (e.g. ``"VOLATILE"``, ``"QUIET"``).

        Returns
        -------
        dict
            Keys: ``"order_flow"``, ``"trend"``, ``"mean_reversion"``,
            ``"volume"``.  Values are float multipliers (>1 boosts,
            <1 suppresses).
        """
        regime_upper = regime.upper() if regime else ""
        if regime_upper == "VOLATILE":
            # Order flow signals more reliable in volatile markets
            return {"order_flow": 1.5, "trend": 0.7, "mean_reversion": 0.5, "volume": 1.3}
        if regime_upper in ("QUIET", "RANGING"):
            # Mean-reversion signals more reliable in ranging/quiet markets
            return {"order_flow": 0.7, "trend": 0.5, "mean_reversion": 1.5, "volume": 0.8}
        if regime_upper in ("TRENDING_UP", "TRENDING_DOWN"):
            # Trend-following signals preferred in trending markets
            return {"order_flow": 1.0, "trend": 1.5, "mean_reversion": 0.3, "volume": 1.0}
        return {"order_flow": 1.0, "trend": 1.0, "mean_reversion": 1.0, "volume": 1.0}

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
        # Evaluate all three paths and return the one with the best R-multiple,
        # adjusted by regime-specific indicator weight multipliers so that the
        # most appropriate signal type is preferred for the current market regime.
        weights = self._select_indicator_weights(regime)
        # Each tuple is (signal, adjusted_r_multiple) for regime-aware selection.
        scored: List[tuple] = []
        for evaluator, weight_key in (
            (self._evaluate_standard,       "trend"),
            (self._evaluate_range_fade,     "mean_reversion"),
            (self._evaluate_whale_momentum, "order_flow"),
        ):
            sig = evaluator(symbol, candles, indicators, smc_data, spread_pct, volume_24h_usd, regime)
            if sig is not None:
                # Boost the effective R-multiple by the regime weight so that
                # regime-preferred signal types rank higher in the selection.
                adjusted_r = sig.r_multiple * weights[weight_key]
                scored.append((sig, adjusted_r))
        if not scored:
            return None
        # Return the candidate with the best regime-adjusted risk-reward
        best, _ = max(scored, key=lambda t: t[1])
        # Apply kill zone check and mark reduced-conviction signals
        profile = smc_data.get("pair_profile") if smc_data else None
        result = self._apply_kill_zone_note(best, profile=profile)
        return result

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
        regime: str = "",
    ) -> Optional[Signal]:
        m5 = candles.get("5m")
        if m5 is None or len(m5.get("close", [])) < 50:
            return None

        ind = indicators.get("5m", {})
        if not check_adx(ind.get("adx_last"), self.config.adx_min):
            return None
        if not self._pass_basic_filters(spread_pct, volume_24h_usd, regime=regime):
            return None

        ema_fast = ind.get("ema9_last")
        ema_slow = ind.get("ema21_last")
        if ema_fast is None or ema_slow is None:
            return None

        sweeps = smc_data.get("sweeps", [])
        if not sweeps:
            return None
        sweep = sweeps[0]

        close = float(m5["close"][-1])
        atr_val = ind.get("atr_last", close * 0.001)

        mom = ind.get("momentum_last")
        if mom is None:
            return None
        # ATR-adaptive momentum threshold: scales with each pair's volatility
        # BTC (ATR ~0.3%) → threshold ~0.15%, DOGE (ATR ~0.8%) → threshold ~0.30%
        atr_pct = (atr_val / close) * 100.0 if close > 0 else 0.15
        profile = smc_data.get("pair_profile")
        base_momentum = max(0.10, min(0.30, atr_pct * 0.5))
        if profile is not None:
            base_momentum *= profile.momentum_threshold_mult
        momentum_threshold = base_momentum
        if abs(mom) < momentum_threshold:
            return None

        # Momentum persistence: require momentum above threshold for consecutive
        # candles to avoid whipsaws where a single candle briefly spikes momentum.
        mom_arr = ind.get("momentum_array")
        persist = profile.momentum_persist_candles if profile else 2
        if mom_arr is not None and len(mom_arr) >= persist:
            if not all(abs(float(mom_arr[-i])) >= momentum_threshold for i in range(1, persist + 1)):
                return None  # Momentum not persistent — likely whipsaw

        direction = sweep.direction

        # RSI extreme gate: don't chase overbought LONGs or fade oversold SHORTs
        if not check_rsi_regime(ind.get("rsi_last"), direction=direction.value, regime=regime):
            return None

        # Momentum must agree with sweep direction
        if direction == Direction.LONG and mom < 0:
            return None
        if direction == Direction.SHORT and mom > 0:
            return None

        pair_tier = profile.tier if profile else "MIDCAP"
        if not check_ema_alignment_adaptive(
            ema_fast, ema_slow, direction.value,
            atr_val=atr_val, close=close,
            regime=regime, pair_tier=pair_tier,
        ):
            return None

        # MACD confirmation gate (PR_04)
        ind_macd_last = ind.get("macd_histogram_last")
        ind_macd_prev = ind.get("macd_histogram_prev")
        strict_macd = regime.upper() in ("RANGING", "QUIET") if regime else False
        macd_ok, macd_adj = check_macd_confirmation(
            ind_macd_last, ind_macd_prev, direction.value, regime=regime, strict=strict_macd
        )
        if not macd_ok:
            return None  # Hard reject in strict mode

        # MTF gate — 1h EMA/RSI must support the 5m signal direction (PR_06)
        indicators_1h = indicators.get("1h", {})
        mtf_ok, mtf_reason, mtf_adj = mtf_gate_scalp_standard(indicators_1h, direction.value, regime)
        if not mtf_ok:
            return None

        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val * 0.5)
        sl = close - sl_dist if direction == Direction.LONG else close + sl_dist

        if direction == Direction.LONG and sl >= close:
            return None
        if direction == Direction.SHORT and sl <= close:
            return None

        _regime_ctx = smc_data.get("regime_context")
        sig = build_channel_signal(
            config=self.config,
            symbol=symbol,
            direction=direction,
            close=close,
            sl=sl,
            tp1=0.0,
            tp2=0.0,
            tp3=0.0,
            sl_dist=sl_dist,
            id_prefix="SCALP",
            atr_val=atr_val,
            setup_class="LIQUIDITY_SWEEP_REVERSAL",
            regime=regime,
            atr_percentile=_regime_ctx.atr_percentile if _regime_ctx else 50.0,
            pair_tier=profile.tier if profile else "MIDCAP",
        )
        if sig is None:
            return None

        sig.trailing_atr_mult_effective = self.config.trailing_atr_mult
        sig.trailing_stage = 0
        sig.partial_close_pct = 0.0

        # Apply MACD soft penalty if applicable
        if macd_adj != 0.0:
            sig.confidence += macd_adj
            if sig.soft_gate_flags:
                sig.soft_gate_flags += ",MACD_WEAK"
            else:
                sig.soft_gate_flags = "MACD_WEAK"

        # Apply MTF soft penalty if applicable
        if mtf_adj != 0.0:
            sig.confidence += mtf_adj
            sig.soft_gate_flags = (sig.soft_gate_flags + f",MTF:{mtf_reason}").lstrip(",")

        return sig

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
        regime: str = "",
    ) -> Optional[Signal]:
        m5 = candles.get("5m")
        if m5 is None or len(m5.get("close", [])) < 50:
            return None

        ind = indicators.get("5m", {})

        # Range fade uses ADX in low-range territory
        # Ceiling adapts to regime: more permissive when ranging/quiet is confirmed,
        # stricter when trending (range-fade in trends is higher risk).
        adx_val = ind.get("adx_last")
        adx_ceiling = 22.0  # default
        if regime in ("RANGING", "QUIET"):
            adx_ceiling = 25.0  # More permissive in confirmed ranging regime
        elif regime in ("TRENDING_UP", "TRENDING_DOWN"):
            adx_ceiling = 18.0  # Stricter — shouldn't be doing range-fade in trends
        if adx_val is not None and adx_val > adx_ceiling:
            return None

        if not self._pass_basic_filters(spread_pct, volume_24h_usd, regime=regime):
            return None

        bb_upper = ind.get("bb_upper_last")
        bb_lower = ind.get("bb_lower_last")
        if bb_upper is None or bb_lower is None:
            return None

        # BB squeeze guard: if BB is expanding rapidly, don't mean-revert
        # (squeeze breaking out invalidates mean-reversion setups)
        bb_width_pct = ind.get("bb_width_pct")
        bb_width_prev_pct = ind.get("bb_width_prev_pct")
        if bb_width_pct is not None and bb_width_prev_pct is not None:
            if bb_width_pct > bb_width_prev_pct * 1.1:  # BB expanding > 10%
                return None

        close = float(m5["close"][-1])
        rsi_val = ind.get("rsi_last")

        profile = smc_data.get("pair_profile")
        bb_touch = profile.bb_touch_pct if profile else 0.002
        direction: Optional[Direction] = None
        if close <= bb_lower * (1 + bb_touch):
            direction = Direction.LONG
        elif close >= bb_upper * (1 - bb_touch):
            direction = Direction.SHORT
        else:
            return None

        # For mean-reversion LONGs we want oversold RSI; for SHORTs, overbought.
        # Reject setups where RSI has already recovered past the mean-reversion
        # entry window (i.e. the edge has been lost).
        # Thresholds adapt to regime: QUIET regime uses wider window (60/40)
        # since RSI ranges are tighter and moves are more significant.
        if rsi_val is not None:
            rsi_long_max = profile.rsi_ob_level if profile else (60.0 if regime == "QUIET" else 55.0)
            rsi_short_min = profile.rsi_os_level if profile else (40.0 if regime == "QUIET" else 45.0)
            if direction == Direction.LONG and rsi_val > rsi_long_max:
                return None  # Not oversold enough for mean-reversion LONG
            if direction == Direction.SHORT and rsi_val < rsi_short_min:
                return None  # Not overbought enough for mean-reversion SHORT

        atr_val = ind.get("atr_last", close * 0.002)
        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val * 0.8)
        sl = close - sl_dist if direction == Direction.LONG else close + sl_dist

        # MACD confirmation gate — always strict for range-fade (PR_04)
        ind_macd_last = ind.get("macd_histogram_last")
        ind_macd_prev = ind.get("macd_histogram_prev")
        macd_ok, _ = check_macd_confirmation(
            ind_macd_last, ind_macd_prev, direction.value, regime=regime, strict=True
        )
        if not macd_ok:
            return None

        # MTF gate — 15m RSI must confirm mean-reversion direction (PR_06)
        indicators_15m = indicators.get("15m", {})
        mtf_ok, mtf_reason, _ = mtf_gate_scalp_range_fade(indicators_15m, direction.value)
        if not mtf_ok:
            return None

        _regime_ctx = smc_data.get("regime_context")
        sig = build_channel_signal(
            config=self.config,
            symbol=symbol,
            direction=direction,
            close=close,
            sl=sl,
            tp1=0.0,
            tp2=0.0,
            tp3=0.0,
            sl_dist=sl_dist,
            id_prefix="RANGE-FADE",
            atr_val=atr_val,
            setup_class="RANGE_FADE",
            regime=regime,
            atr_percentile=_regime_ctx.atr_percentile if _regime_ctx else 50.0,
            pair_tier=profile.tier if profile else "MIDCAP",
        )
        if sig is not None:
            sig.trailing_atr_mult_effective = self.config.trailing_atr_mult
            sig.trailing_stage = 0
            sig.partial_close_pct = 0.0
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
        regime: str = "",
    ) -> Optional[Signal]:
        whale = smc_data.get("whale_alert")
        delta_spike = smc_data.get("volume_delta_spike", False)
        if whale is None and not delta_spike:
            return None

        if not self._pass_basic_filters(spread_pct, volume_24h_usd, regime=regime):
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

        # RSI extreme gate: don't chase overbought LONGs or fade oversold SHORTs
        if not check_rsi_regime(indicators.get("1m", {}).get("rsi_last"), direction=direction.value, regime=regime):
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
        sl = close - sl_dist if direction == Direction.LONG else close + sl_dist

        _regime_ctx = smc_data.get("regime_context")
        _pair_profile = smc_data.get("pair_profile")
        sig = build_channel_signal(
            config=self.config,
            symbol=symbol,
            direction=direction,
            close=close,
            sl=sl,
            tp1=0.0,
            tp2=0.0,
            tp3=0.0,
            sl_dist=sl_dist,
            id_prefix="WHALE",
            atr_val=atr_val,
            setup_class="WHALE_MOMENTUM",
            regime=regime,
            atr_percentile=_regime_ctx.atr_percentile if _regime_ctx else 50.0,
            pair_tier=_pair_profile.tier if _pair_profile else "MIDCAP",
        )
        if sig is not None:
            sig.trailing_atr_mult_effective = self.config.trailing_atr_mult
            sig.trailing_stage = 0
            sig.partial_close_pct = 0.0
        return sig

    # ------------------------------------------------------------------
    # Kill zone integration (P2-13)
    # ------------------------------------------------------------------

    def _is_kill_zone_active(self, now: Optional[datetime] = None) -> bool:
        """Return True if the current UTC time falls within a high-liquidity kill zone.

        Kill zones are defined as:
        * London session     : 07:00–10:00 UTC
        * NY session         : 12:00–16:00 UTC
        * London/NY overlap  : 12:00–14:00 UTC (already covered by NY range above)
        """
        if now is None:
            now = datetime.now(timezone.utc)
        hour = now.hour
        return (7 <= hour < 10) or (12 <= hour < 16)

    def _apply_kill_zone_note(self, sig: Signal, profile=None, now: Optional[datetime] = None) -> Optional[Signal]:
        """Annotate the signal with a reduced-conviction note when outside kill zones.

        For ALTCOIN tier (kill_zone_hard_gate=True), hard-rejects signals outside
        kill zones.  For other tiers, sets execution_note but still emits the signal.
        """
        if not self._is_kill_zone_active(now):
            if profile is not None and profile.kill_zone_hard_gate:
                return None  # Hard reject — ALTCOIN tier outside kill zone
            if sig.execution_note:
                sig.execution_note += "; Outside kill zone — reduced conviction"
            else:
                sig.execution_note = "Outside kill zone — reduced conviction"
        return sig
