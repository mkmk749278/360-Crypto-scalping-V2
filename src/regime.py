"""Market Regime Detection.

Classifies the current market regime based on technical indicators so that
channel evaluators and the confidence scorer can adapt their behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from src.utils import get_logger

log = get_logger("regime")


class MarketRegime(str, Enum):
    """Possible market regimes returned by :class:`MarketRegimeDetector`."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    QUIET = "QUIET"


@dataclass
class RegimeResult:
    """Result of a single regime classification."""

    regime: MarketRegime
    adx: Optional[float] = None
    bb_width_pct: Optional[float] = None
    ema_slope: Optional[float] = None
    note: str = ""


# Thresholds (tunable via environment variables in the future)
_ADX_TRENDING_MIN: float = 25.0
_ADX_RANGING_MAX: float = 20.0
_BB_WIDTH_VOLATILE_PCT: float = 5.0   # Bollinger width as % of price
_BB_WIDTH_QUIET_PCT: float = 1.5


class MarketRegimeDetector:
    """Classifies market regime from a set of pre-computed indicators.

    Usage::

        detector = MarketRegimeDetector()
        result = detector.classify(indicators["5m"])
        if result.regime == MarketRegime.TRENDING_UP:
            ...
    """

    def classify(
        self,
        indicators: Dict[str, Any],
        candles: Optional[Dict[str, Any]] = None,
    ) -> RegimeResult:
        """Classify market regime from *indicators* dict.

        Expected indicator keys (all optional – graceful degradation):
          - ``adx_last``        – ADX(14) value
          - ``ema9_last``       – fast EMA
          - ``ema21_last``      – slow EMA
          - ``bb_upper_last``   – Bollinger upper band
          - ``bb_mid_last``     – Bollinger middle band
          - ``bb_lower_last``   – Bollinger lower band
        """
        adx_val: Optional[float] = indicators.get("adx_last")
        ema_fast: Optional[float] = indicators.get("ema9_last")
        ema_slow: Optional[float] = indicators.get("ema21_last")
        bb_upper: Optional[float] = indicators.get("bb_upper_last")
        bb_lower: Optional[float] = indicators.get("bb_lower_last")
        bb_mid: Optional[float] = indicators.get("bb_mid_last")

        # ema_slow defaults to close price from candles when unavailable
        close: Optional[float] = None
        if candles is not None and len(candles.get("close", [])) > 0:
            close = float(candles["close"][-1])

        # Fall back to close price when EMA values are missing
        if ema_fast is None and close is not None:
            ema_fast = close
        if ema_slow is None and close is not None:
            ema_slow = close

        # EMA slope (approximation via % diff between fast and slow)
        ema_slope: Optional[float] = None
        if ema_fast is not None and ema_slow is not None and ema_slow != 0.0:
            ema_slope = (ema_fast - ema_slow) / ema_slow * 100.0

        # Bollinger Band width as % of mid price
        bb_width_pct: Optional[float] = None
        if bb_upper is not None and bb_lower is not None and bb_mid and bb_mid != 0.0:
            bb_width_pct = (bb_upper - bb_lower) / bb_mid * 100.0

        regime = self._decide(adx_val, ema_slope, bb_width_pct)

        return RegimeResult(
            regime=regime,
            adx=adx_val,
            bb_width_pct=bb_width_pct,
            ema_slope=ema_slope,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _decide(
        adx: Optional[float],
        ema_slope: Optional[float],
        bb_width_pct: Optional[float],
    ) -> MarketRegime:
        # Volatility check (Bollinger width) takes priority
        if bb_width_pct is not None:
            if bb_width_pct >= _BB_WIDTH_VOLATILE_PCT:
                return MarketRegime.VOLATILE
            if bb_width_pct <= _BB_WIDTH_QUIET_PCT:
                return MarketRegime.QUIET

        # Trending regime check
        if adx is not None and adx >= _ADX_TRENDING_MIN:
            if ema_slope is not None:
                return MarketRegime.TRENDING_UP if ema_slope > 0 else MarketRegime.TRENDING_DOWN
            return MarketRegime.TRENDING_UP  # can't determine direction without EMA

        # Range-bound
        if adx is not None and adx <= _ADX_RANGING_MAX:
            return MarketRegime.RANGING

        # Fall back to EMA slope when ADX is borderline
        if ema_slope is not None:
            if ema_slope > 0.05:
                return MarketRegime.TRENDING_UP
            if ema_slope < -0.05:
                return MarketRegime.TRENDING_DOWN

        return MarketRegime.RANGING
