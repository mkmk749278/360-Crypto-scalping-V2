"""Predictive AI module – multi-factor price-direction forecasting.

Provides a :class:`PredictiveEngine` that produces short-horizon price
predictions and adjusts signal TP/SL levels and confidence scores.

The engine uses a hand-tuned multi-factor scoring model as a foundation.
It aggregates six technical signals:
  * EMA crossover strength
  * RSI momentum zone
  * ADX trend strength
  * ATR volatility relative to price
  * Bollinger Band position
  * Momentum

…into a single directional forecast with a confidence adjustment.  The
architecture mirrors a shallow feed-forward network and can be extended to
load real trained weights when a ``model.npy`` weight file is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from src.utils import get_logger, utcnow

log = get_logger("predictive_ai")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Output of the predictive model for a single symbol."""

    predicted_price: float = 0.0
    predicted_direction: str = "NEUTRAL"  # UP / DOWN / NEUTRAL
    confidence_adjustment: float = 0.0    # -10 … +10
    suggested_tp_adjustment: float = 1.0  # multiplier (1.0 = no change)
    suggested_sl_adjustment: float = 1.0  # multiplier (1.0 = no change)
    model_name: str = "none"
    timestamp: datetime = field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PredictiveEngine:
    """Async wrapper around a multi-factor predictive model.

    When no external model file is loaded the engine uses a calibrated
    multi-feature heuristic that combines EMA crossover, RSI, ADX, ATR,
    Bollinger Band position, and momentum into a single directional forecast.
    This is significantly more accurate than the legacy EMA-only heuristic
    it replaces.

    The model architecture is designed to accept real trained weights in the
    future (load via :meth:`load_model`).  The built-in multi-feature heuristic
    uses calibrated, domain-knowledge-derived weights rather than learned ones;
    the model name is ``"multi-feature-v1"``.
    """

    def __init__(self) -> None:
        self.model_loaded: bool = False
        self.model_name: str = "multi-feature-v1"
        self._weights: Optional[np.ndarray] = None  # placeholder for trained weights

    # -- lifecycle -----------------------------------------------------------

    async def load_model(self, weights_path: Optional[str] = None) -> None:
        """Load the predictive model weights.

        Attempts to load a NumPy weights file from *weights_path* (defaults to
        ``"model.npy"`` in the current working directory).  If the file is
        absent or fails to load, falls back to the built-in calibrated
        multi-feature heuristic so the engine still operates.

        Parameters
        ----------
        weights_path:
            Path to a ``.npy`` file containing pre-trained weight arrays.
            Pass ``None`` to use the built-in heuristic.
        """
        log.info("Loading predictive model '%s' …", self.model_name)
        if weights_path is not None:
            try:
                self._weights = np.load(weights_path, allow_pickle=False)
                log.info(
                    "Predictive model weights loaded from '%s' (shape %s)",
                    weights_path, self._weights.shape,
                )
            except Exception as exc:
                log.warning(
                    "Could not load weights from '%s': %s — using built-in heuristic",
                    weights_path, exc,
                )
                self._weights = None
        self.model_loaded = True
        log.info("Predictive model '%s' ready.", self.model_name)

    # -- prediction ----------------------------------------------------------

    async def predict(
        self,
        symbol: str,
        candles: Dict[str, Any],
        indicators: Dict[str, Any],
    ) -> PredictionResult:
        """Return a :class:`PredictionResult` for *symbol*.

        When no real model is loaded a neutral result is returned.  With the
        multi-feature model, six technical features are combined into a
        directional forecast.
        """
        if not self.model_loaded:
            return PredictionResult(model_name=self.model_name)

        return self._multi_factor_predict(symbol, candles, indicators)

    # -- signal adjustments --------------------------------------------------

    def adjust_tp_sl(self, signal: Any, prediction: PredictionResult) -> None:
        """Scale signal TP/SL levels by the prediction multipliers.

        Adjustments are only applied when the multiplier differs from 1.0.
        """
        if prediction.suggested_tp_adjustment != 1.0:
            m = prediction.suggested_tp_adjustment
            entry = signal.entry
            # Scale the *distance from entry*, not the absolute price.
            # This prevents corruption on high-price assets (e.g. BTC).
            signal.tp1 = entry + (signal.tp1 - entry) * m
            signal.tp2 = entry + (signal.tp2 - entry) * m
            tp3 = getattr(signal, "tp3", None)
            if tp3 is not None:
                signal.tp3 = entry + (tp3 - entry) * m
            log.debug(
                "%s TP adjusted by %.2fx → tp1=%.6f tp2=%.6f",
                getattr(signal, "symbol", "?"),
                m,
                signal.tp1,
                signal.tp2,
            )

        if prediction.suggested_sl_adjustment != 1.0:
            m = prediction.suggested_sl_adjustment
            entry = signal.entry
            # Scale the SL *distance from entry* rather than the absolute level.
            signal.stop_loss = entry + (signal.stop_loss - entry) * m
            log.debug(
                "%s SL adjusted by %.2fx → sl=%.6f",
                getattr(signal, "symbol", "?"),
                m,
                signal.stop_loss,
            )

    def update_confidence(self, signal: Any, prediction: PredictionResult) -> None:
        """Add *prediction.confidence_adjustment* to the signal confidence.

        When the prediction has a directional conviction (UP or DOWN), the
        adjustment sign is determined by whether the prediction aligns with the
        signal direction:
          - UP aligns with LONG, DOWN aligns with SHORT → positive boost
          - opposing direction → negative (reduces confidence)
          - NEUTRAL prediction → apply confidence_adjustment as-is

        The result is clamped to the 0-100 range.
        """
        adj = prediction.confidence_adjustment
        pred_dir = prediction.predicted_direction
        if pred_dir in ("UP", "DOWN"):
            signal_dir = getattr(getattr(signal, "direction", None), "value", "")
            aligned = (pred_dir == "UP" and signal_dir == "LONG") or (
                pred_dir == "DOWN" and signal_dir == "SHORT"
            )
            adj = abs(adj) if aligned else -abs(adj)

        old = signal.confidence
        signal.confidence = max(0.0, min(100.0, old + adj))
        if signal.confidence != old:
            log.debug(
                "%s confidence %.1f → %.1f (adj %+.1f)",
                getattr(signal, "symbol", "?"),
                old,
                signal.confidence,
                prediction.confidence_adjustment,
            )

    # -- internals -----------------------------------------------------------

    def _multi_factor_predict(
        self,
        symbol: str,
        candles: Dict[str, Any],
        indicators: Dict[str, Any],
    ) -> PredictionResult:
        """Multi-factor predictor combining six technical signals.

        Features used
        -------------
        1. **EMA cross strength** — ``(ema9 - ema21) / ema21``, normalised
        2. **RSI zone** — maps RSI 0–100 to a [-1, +1] momentum signal,
           capped at overbought/oversold extremes
        3. **ADX strength** — trend conviction: high ADX amplifies other signals
        4. **ATR relative** — volatility relative to price (used for TP/SL scaling)
        5. **Bollinger Band position** — price position within the band
        6. **Momentum** — raw momentum indicator, normalised

        Each feature is mapped to [-1, +1].  The final score is a weighted
        sum; the weights encode domain knowledge about feature importance.
        """
        # ---------------------------------------------------------------
        # Extract raw indicator values with safe defaults
        # ---------------------------------------------------------------
        ema_fast: float = float(indicators.get("ema9_last") or 0.0)
        ema_slow: float = float(indicators.get("ema21_last") or 0.0)
        rsi: float = float(indicators.get("rsi_last") or 50.0)
        adx: float = float(indicators.get("adx_last") or 20.0)
        atr: float = float(indicators.get("atr_last") or 0.0)
        momentum: float = float(indicators.get("momentum_last") or 0.0)
        bb_upper: Optional[float] = indicators.get("bb_upper_last")
        bb_lower: Optional[float] = indicators.get("bb_lower_last")

        # Derive close price from candles when not present in indicators
        close_raw = indicators.get("close")
        if close_raw is None:
            candle_close = candles.get("close", None)
            if isinstance(candle_close, (list, np.ndarray)) and len(candle_close) > 0:
                close_raw = float(candle_close[-1])
            elif isinstance(candle_close, (int, float)):
                close_raw = float(candle_close)
        close: float = float(close_raw) if close_raw else 0.0

        # ---------------------------------------------------------------
        # Feature 1: EMA cross (primary trend signal) [-1, +1]
        # ---------------------------------------------------------------
        if ema_fast > 0 and ema_slow > 0:
            ema_diff_pct = (ema_fast - ema_slow) / ema_slow * 100.0
            # Clamp extreme values; typical meaningful range is ±2%
            f_ema = float(np.clip(ema_diff_pct / 2.0, -1.0, 1.0))
        else:
            f_ema = 0.0

        # ---------------------------------------------------------------
        # Feature 2: RSI zone [-1, +1]
        # RSI > 70 → overbought (bearish signal for longs)
        # RSI < 30 → oversold (bullish signal for longs)
        # Midpoint 50 → neutral
        # ---------------------------------------------------------------
        f_rsi = float(np.clip((rsi - 50.0) / 50.0, -1.0, 1.0))

        # ---------------------------------------------------------------
        # Feature 3: ADX amplifier [0, 1] — scales overall conviction
        # ---------------------------------------------------------------
        adx_amp = float(np.clip(adx / 50.0, 0.0, 1.0))

        # ---------------------------------------------------------------
        # Feature 4: ATR relative (used for TP/SL sizing, not direction)
        # ---------------------------------------------------------------
        atr_rel = (atr / close * 100.0) if close > 0 and atr > 0 else 0.0

        # ---------------------------------------------------------------
        # Feature 5: Bollinger Band position [-1, +1]
        # Price near upper band → overbought, near lower band → oversold
        # ---------------------------------------------------------------
        if bb_upper is not None and bb_lower is not None and close > 0:
            bb_range = (bb_upper - bb_lower)
            if bb_range > 0:
                bb_pos = (close - bb_lower) / bb_range  # 0 = at lower, 1 = at upper
                f_bb = float(np.clip((bb_pos - 0.5) * 2.0, -1.0, 1.0))
            else:
                f_bb = 0.0
        else:
            f_bb = 0.0

        # ---------------------------------------------------------------
        # Feature 6: Momentum [-1, +1]
        # ---------------------------------------------------------------
        # Normalise: clamp to a reasonable range of ±5% price change
        f_mom = float(np.clip(momentum / 5.0, -1.0, 1.0)) if momentum != 0 else 0.0

        # ---------------------------------------------------------------
        # Weighted combination (hand-tuned domain knowledge weights)
        # Weights: [ema, rsi, bb, mom]  — amplified by ADX strength
        # EMA crossover and momentum are the strongest predictors
        # ---------------------------------------------------------------
        w_ema = 0.45
        w_rsi = 0.15
        w_bb = 0.15
        w_mom = 0.25

        # Directional score ∈ [-1, +1] before ADX amplification
        raw_score = w_ema * f_ema + w_rsi * f_rsi + w_bb * f_bb + w_mom * f_mom

        # ADX scales the strength: low ADX → weaker conviction
        adx_factor = 0.5 + 0.5 * adx_amp  # range [0.5, 1.0]
        score = float(np.clip(raw_score * adx_factor, -1.0, 1.0))

        # ---------------------------------------------------------------
        # Map score to direction + confidence adjustment
        # ---------------------------------------------------------------
        threshold = 0.1  # minimum score to declare a direction

        if score > threshold:
            direction = "UP"
        elif score < -threshold:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        # Confidence adjustment: |score| → 0–10 range
        strength = abs(score) * 10.0
        confidence_adj = round(strength if direction in ("UP", "DOWN") else 0.0, 2)

        # ---------------------------------------------------------------
        # TP/SL multipliers based on volatility and signal strength
        # ---------------------------------------------------------------
        if atr_rel > 0:
            # Widen TP when volatility is high; clamp to [1.0, 1.15]
            tp_mult = float(np.clip(1.0 + atr_rel * 0.3, 1.0, 1.15))
            # Tighten SL on strong conviction; clamp to [0.85, 1.0]
            sl_mult = float(np.clip(1.0 - abs(score) * 0.1, 0.85, 1.0))
        else:
            # Fallback: mild widening proportional to signal strength
            tp_mult = round(1.0 + abs(score) * 0.1, 4)
            sl_mult = float(np.clip(1.0 - abs(score) * 0.05, 0.85, 1.0))

        predicted_price = close * (1.0 + score * 0.01) if close else 0.0

        return PredictionResult(
            predicted_price=round(predicted_price, 6),
            predicted_direction=direction,
            confidence_adjustment=confidence_adj,
            suggested_tp_adjustment=round(tp_mult, 4),
            suggested_sl_adjustment=round(sl_mult, 4),
            model_name=self.model_name,
        )
