"""Predictive AI module – LSTM/Transformer price-direction forecasting.

Provides a :class:`PredictiveEngine` that produces short-horizon price
predictions and adjusts signal TP/SL levels and confidence scores.
All heavy model logic is behind placeholders so the module works without
additional ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

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
    """Async wrapper around a predictive model (LSTM / Transformer).

    While no real model is loaded the engine falls back to a lightweight
    momentum-based heuristic so the rest of the pipeline can run unchanged.
    """

    def __init__(self) -> None:
        self.model_loaded: bool = False
        self.model_name: str = "placeholder-momentum-v0"

    # -- lifecycle -----------------------------------------------------------

    async def load_model(self) -> None:
        """Load the predictive model weights.

        Placeholder – sets *model_loaded* to ``True`` so that
        :meth:`predict` uses the heuristic path.  Replace with real
        ``torch.load`` / ``tf.saved_model.load`` when a trained model is
        available.
        """
        log.info("Loading predictive model '%s' …", self.model_name)
        # TODO: load real LSTM / Transformer weights here
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
        placeholder model a simple momentum / EMA heuristic is used.
        """
        if not self.model_loaded:
            return PredictionResult(model_name=self.model_name)

        return self._heuristic_predict(symbol, candles, indicators)

    # -- signal adjustments --------------------------------------------------

    def adjust_tp_sl(self, signal: Any, prediction: PredictionResult) -> None:
        """Scale signal TP/SL levels by the prediction multipliers.

        Adjustments are only applied when the multiplier differs from 1.0.
        """
        if prediction.suggested_tp_adjustment != 1.0:
            m = prediction.suggested_tp_adjustment
            signal.tp1 *= m
            signal.tp2 *= m
            tp3 = getattr(signal, "tp3", None)
            if tp3 is not None:
                signal.tp3 = tp3 * m
            log.debug(
                "%s TP adjusted by %.2fx → tp1=%.6f tp2=%.6f",
                getattr(signal, "symbol", "?"),
                m,
                signal.tp1,
                signal.tp2,
            )

        if prediction.suggested_sl_adjustment != 1.0:
            m = prediction.suggested_sl_adjustment
            signal.stop_loss *= m
            log.debug(
                "%s SL adjusted by %.2fx → sl=%.6f",
                getattr(signal, "symbol", "?"),
                m,
                signal.stop_loss,
            )

    def update_confidence(self, signal: Any, prediction: PredictionResult) -> None:
        """Add *prediction.confidence_adjustment* to the signal confidence.

        The result is clamped to the 0-100 range.
        """
        old = signal.confidence
        signal.confidence = max(0.0, min(100.0, old + prediction.confidence_adjustment))
        if signal.confidence != old:
            log.debug(
                "%s confidence %.1f → %.1f (adj %+.1f)",
                getattr(signal, "symbol", "?"),
                old,
                signal.confidence,
                prediction.confidence_adjustment,
            )

    # -- internals -----------------------------------------------------------

    def _heuristic_predict(
        self,
        symbol: str,
        candles: Dict[str, Any],
        indicators: Dict[str, Any],
    ) -> PredictionResult:
        """Simple momentum / EMA heuristic used as a placeholder model."""
        momentum: float = float(indicators.get("momentum", 0.0))
        ema_fast: float = float(indicators.get("ema_fast", 0.0))
        ema_slow: float = float(indicators.get("ema_slow", 0.0))
        close: float = float(indicators.get("close", 0.0)) or float(
            candles.get("close", 0.0)
        )

        # Derive direction from EMA crossover + momentum sign
        if ema_fast and ema_slow and ema_slow != 0.0:
            ema_diff_pct = (ema_fast - ema_slow) / ema_slow * 100.0
        else:
            ema_diff_pct = 0.0

        if ema_diff_pct > 0.05 and momentum > 0:
            direction = "UP"
        elif ema_diff_pct < -0.05 and momentum < 0:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        # Confidence adjustment proportional to signal strength
        strength = min(abs(ema_diff_pct) + abs(momentum) * 0.5, 10.0)
        confidence_adj = strength if direction == "UP" else (
            -strength if direction == "DOWN" else 0.0
        )

        # TP/SL multiplier: widen targets when conviction is high
        tp_mult = 1.0 + (strength / 100.0)   # e.g. 1.0 – 1.10
        sl_mult = 1.0 - (strength / 200.0)   # e.g. 1.0 – 0.95 (tighter SL)
        sl_mult = max(sl_mult, 0.8)           # floor to avoid excessively tight SL

        predicted_price = close * (1.0 + ema_diff_pct / 100.0) if close else 0.0

        return PredictionResult(
            predicted_price=predicted_price,
            predicted_direction=direction,
            confidence_adjustment=round(confidence_adj, 2),
            suggested_tp_adjustment=round(tp_mult, 4),
            suggested_sl_adjustment=round(sl_mult, 4),
            model_name=self.model_name,
        )
