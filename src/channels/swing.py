"""360_SWING – H1/H4 Institutional Swing 🏛️

Trigger : H4 ERL sweep + H1 MSS
Filters : EMA200, Bollinger rejection, ADX 20–40, ATR filter, spread < 0.02 %
Risk    : SL 0.2–0.5 %, TP1 1.5R, TP2 3R, TP3 4–5R, Trailing 2.5×ATR
"""

from __future__ import annotations

from typing import Dict, Optional

from config import CHANNEL_SWING
from src.channels.base import BaseChannel, Signal, build_channel_signal
from src.filters import check_adx_regime, check_macd_confirmation, check_rsi_regime, check_spread_adaptive, check_volume
from src.mtf import mtf_gate_swing
from src.smc import Direction

# Percentile position within the Bollinger Band range for rejection gate.
# For LONG: price must be in the bottom BB_REJECTION_THRESHOLD fraction (near lower band).
# For SHORT: price must be in the top BB_REJECTION_THRESHOLD fraction (near upper band).
_BB_REJECTION_THRESHOLD: float = 0.15

# EMA200 buffer zone fallback (used only when ATR data is unavailable).
# The evaluate() method computes a dynamic buffer scaled by ATR and regime.
_EMA200_BUFFER_PCT: float = 0.5

# Minimum MSS candle body size (as % of close) to avoid noise signals.
# Tiny bodies often indicate indecision candles rather than genuine structure shifts.
_MSS_MIN_BODY_SIZE_PCT: float = 0.05


class SwingChannel(BaseChannel):
    def __init__(self) -> None:
        super().__init__(CHANNEL_SWING)

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
        h4 = candles.get("4h")
        h1 = candles.get("1h")
        if h4 is None or h1 is None:
            return None
        if len(h4.get("close", [])) < 50 or len(h1.get("close", [])) < 50:
            return None

        # --- Filters ---
        ind_h4 = indicators.get("4h", {})
        ind_h1 = indicators.get("1h", {})
        if not check_adx_regime(ind_h4.get("adx_last"), regime=regime, setup_class="", max_adx=self.config.adx_max):
            return None
        if not check_spread_adaptive(spread_pct, self.config.spread_max, regime=regime):
            return None
        if not check_volume(volume_24h_usd, self.config.min_volume):
            return None

        # EMA200 filter
        ema200 = ind_h1.get("ema200_last")
        close_h1 = float(h1["close"][-1])
        if ema200 is None:
            return None

        # Bollinger rejection
        bb_upper = ind_h1.get("bb_upper_last")
        bb_lower = ind_h1.get("bb_lower_last")

        # --- SMC trigger: H4 sweep + H1 MSS ---
        sweeps = smc_data.get("sweeps", [])
        mss = smc_data.get("mss")
        if not sweeps or mss is None:
            return None

        direction = mss.direction

        # MSS body-size minimum: tiny sweep candle bodies are noise, not genuine structure shifts.
        # Use the H4 sweep candle's open_price and close_price (fields on LiquiditySweep).
        sweep = sweeps[0]
        sweep_open = getattr(sweep, "open_price", None)
        sweep_close = getattr(sweep, "close_price", None)
        if sweep_open is not None and sweep_close is not None and float(sweep_close) > 0:
            body_size_pct = abs(float(sweep_open) - float(sweep_close)) / float(sweep_close) * 100.0
            if body_size_pct < _MSS_MIN_BODY_SIZE_PCT:
                return None  # Sweep candle body too small — likely indecision, not a genuine MSS

        # RSI extreme gate: don't chase overbought LONGs or fade oversold SHORTs
        if not check_rsi_regime(ind_h1.get("rsi_last"), direction=direction.value, regime=regime):
            return None

        # MACD confirmation gate — soft penalty for swing (longer TF is smoother) (PR_04)
        ind_h1_macd_last = ind_h1.get("macd_histogram_last")
        ind_h1_macd_prev = ind_h1.get("macd_histogram_prev")
        macd_ok, macd_adj = check_macd_confirmation(
            ind_h1_macd_last, ind_h1_macd_prev, direction.value, regime=regime, strict=False
        )
        if not macd_ok:
            return None

        # Validate EMA200 bias — with a dynamic buffer zone scaled by ATR and regime.
        # Signals within the buffer are rejected because price is too close to the
        # long-term trend boundary to determine bias cleanly.
        atr_val = ind_h1.get("atr_last", close_h1 * 0.003)
        atr_pct = (atr_val / close_h1 * 100.0) if close_h1 > 0 and atr_val > 0 else 0.3
        profile = smc_data.get("pair_profile")
        pair_tier = profile.tier if profile else "MIDCAP"
        regime_ctx = smc_data.get("regime_context")
        # EMA200 floor buffers are wider than fast/slow EMA alignment buffers
        # (filters.py uses 0.1/0.2/0.3%) because EMA200 is a long-term trend
        # boundary — signals near it carry greater directional ambiguity.
        tier_floors = {"MAJOR": 0.3, "MIDCAP": 0.5, "ALTCOIN": 0.7}
        regime_mults = {
            "TRENDING_UP": 0.8, "TRENDING_DOWN": 0.8,
            "RANGING": 1.2, "QUIET": 1.2,
            "VOLATILE": 1.5,
        }
        regime_mult = regime_mults.get(regime.upper() if regime else "", 1.0)
        # EMA200 uses a 0.4 scaling factor (vs 0.5 in check_ema_alignment_adaptive)
        # because EMA200 distance is a longer-term filter that should be less
        # sensitive to short-term ATR spikes.
        ema200_buffer_pct = max(tier_floors.get(pair_tier, 0.5), atr_pct * regime_mult * 0.4)

        ema200_distance_pct = abs(close_h1 - ema200) / ema200 * 100.0
        if ema200_distance_pct < ema200_buffer_pct:
            return None  # Too close to EMA200 — ambiguous bias
        if direction == Direction.LONG and close_h1 < ema200:
            return None
        if direction == Direction.SHORT and close_h1 > ema200:
            return None

        # Validate Bollinger rejection using percentile position within the band range.
        # This adapts to varying BB widths across regimes rather than using a fixed 2%
        # threshold that may be too permissive in tight-range environments.
        if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
            bb_position = (close_h1 - bb_lower) / (bb_upper - bb_lower)
            if direction == Direction.LONG and bb_position > _BB_REJECTION_THRESHOLD:
                return None  # Price too far from lower band — no BB rejection setup
            if direction == Direction.SHORT and bb_position < (1.0 - _BB_REJECTION_THRESHOLD):
                return None  # Price too far from upper band — no BB rejection setup
        else:
            # Fallback: zero-width bands or missing data — use original fixed threshold.
            if direction == Direction.LONG and bb_lower is not None:
                if close_h1 > bb_lower * 1.02:
                    return None
            if direction == Direction.SHORT and bb_upper is not None:
                if close_h1 < bb_upper * 0.98:
                    return None

        # Daily S/R confluence check (soft boost, not a hard reject)
        d1 = candles.get("1d")
        daily_confluence = False
        if d1 is not None and len(d1.get("close", [])) >= 20:
            d1_highs = [float(h) for h in list(d1.get("high", d1["close"]))[-20:]]
            d1_lows = [float(low_val) for low_val in list(d1.get("low", d1["close"]))[-20:]]
            if direction == Direction.LONG:
                nearest_daily_support = min(d1_lows[-10:])
                if close_h1 <= nearest_daily_support * 1.03:  # within 3% of daily support
                    daily_confluence = True
            elif direction == Direction.SHORT:
                nearest_daily_resistance = max(d1_highs[-10:])
                if close_h1 >= nearest_daily_resistance * 0.97:  # within 3% of daily resistance
                    daily_confluence = True

        close = close_h1
        atr_val = ind_h1.get("atr_last", close * 0.003)

        sl_dist = max(close * self.config.sl_pct_range[0] / 100, atr_val)
        sl = close - sl_dist if direction == Direction.LONG else close + sl_dist

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
            id_prefix="SWING",
            atr_val=atr_val,
            regime=regime,
            atr_percentile=regime_ctx.atr_percentile if regime_ctx else 50.0,
            pair_tier=pair_tier,
        )
        if sig is None:
            return None

        sig.trailing_atr_mult_effective = self.config.trailing_atr_mult
        sig.trailing_stage = 0
        sig.partial_close_pct = 0.0

        # MTF gate — 4h EMA + ADX must support the 1h signal direction (PR_06)
        mtf_ok, mtf_reason, mtf_adj = mtf_gate_swing(ind_h4, direction.value)
        if not mtf_ok:
            return None
        if mtf_adj != 0.0:
            sig.confidence += mtf_adj

        # Apply MACD soft penalty if applicable
        if macd_adj != 0.0:
            sig.confidence += macd_adj
            if sig.soft_gate_flags:
                sig.soft_gate_flags += ",MACD_WEAK"
            else:
                sig.soft_gate_flags = "MACD_WEAK"

        sig.risk_label = "Medium"

        # Mark signal quality tier based on daily confluence
        if daily_confluence:
            sig.setup_class = "SWING_D1_CONFLUENCE"
            sig.quality_tier = "A+"
        else:
            sig.setup_class = "SWING_STANDARD"

        return sig
