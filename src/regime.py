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
    volume_delta_pct: Optional[float] = None
    note: str = ""


# Thresholds (tunable via environment variables in the future)
_ADX_TRENDING_MIN: float = 25.0
_ADX_RANGING_MAX: float = 20.0
_BB_WIDTH_VOLATILE_PCT: float = 5.0   # Bollinger width as % of price
_BB_WIDTH_QUIET_PCT: float = 1.5
# Volume-delta override: if |volume_delta_pct| >= this threshold, the regime
# is forced out of QUIET / RANGING into VOLATILE or TRENDING.
_VOLUME_DELTA_SPIKE_PCT: float = 60.0  # percent of total volume as net delta


class MarketRegimeDetector:
    """Classifies market regime from a set of pre-computed indicators.

    Parameters
    ----------
    hysteresis_candles:
        Number of consecutive classifications the new regime must be seen
        before it is officially adopted.  Prevents rapid regime flapping in
        consolidation zones near EMA crosses.  Defaults to ``3``.

    Usage::

        detector = MarketRegimeDetector()
        result = detector.classify(indicators["5m"])
        if result.regime == MarketRegime.TRENDING_UP:
            ...
    """

    def __init__(self, hysteresis_candles: int = 3) -> None:
        self._hysteresis_candles: int = hysteresis_candles
        # Stable (officially adopted) regime — None until first classification.
        self._previous_regime: Optional[MarketRegime] = None
        # Candidate regime currently accumulating dwell count.
        self._pending_regime: Optional[MarketRegime] = None
        # How many consecutive times the pending regime has been seen.
        self._regime_dwell_count: int = 0
        # Track the last timeframe used; hysteresis resets when it changes.
        self._last_timeframe: Optional[str] = None

    def classify(
        self,
        indicators: Dict[str, Any],
        candles: Optional[Dict[str, Any]] = None,
        timeframe: str = "5m",
        volume_delta: Optional[float] = None,
    ) -> RegimeResult:
        """Classify market regime from *indicators* dict.

        Expected indicator keys (all optional – graceful degradation):
          - ``adx_last``        – ADX(14) value
          - ``ema9_last``       – fast EMA
          - ``ema21_last``      – slow EMA
          - ``bb_upper_last``   – Bollinger upper band
          - ``bb_mid_last``     – Bollinger middle band
          - ``bb_lower_last``   – Bollinger lower band

        The *timeframe* parameter adjusts EMA slope thresholds: on 1-minute
        data the threshold is widened (±0.15 %) to reduce noise-driven regime
        flips that would otherwise occur every few candles.

        Parameters
        ----------
        volume_delta:
            Optional net volume delta value expressed as a percentage of total
            volume (buy_volume - sell_volume) / total_volume * 100.  When its
            absolute value exceeds :data:`_VOLUME_DELTA_SPIKE_PCT` the regime
            is forced out of ``QUIET`` or ``RANGING`` into ``VOLATILE`` (when
            the direction is ambiguous) or ``TRENDING_UP`` / ``TRENDING_DOWN``
            (when EMA slope provides directional context).  This lets the bot
            react to sudden order-book imbalances faster than ADX or EMAs can.
        """
        adx_val: Optional[float] = indicators.get("adx_last")
        ema_fast: Optional[float] = indicators.get("ema9_last")
        ema_slow: Optional[float] = indicators.get("ema21_last")
        bb_upper: Optional[float] = indicators.get("bb_upper_last")
        bb_lower: Optional[float] = indicators.get("bb_lower_last")
        bb_mid: Optional[float] = indicators.get("bb_mid_last")

        # Reset hysteresis state when the timeframe changes.  In production,
        # each scanner cycle uses the same timeframe, but tests may call the
        # same detector instance with different timeframes; resetting ensures
        # the raw _decide() output is returned directly in that case.
        if timeframe != self._last_timeframe:
            self._previous_regime = None
            self._pending_regime = None
            self._regime_dwell_count = 0
            self._last_timeframe = timeframe

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

        regime = self._decide(adx_val, ema_slope, bb_width_pct, timeframe=timeframe)

        # Apply hysteresis to the indicator-derived regime (prevents flapping near
        # EMA crosses and ADX transition zones).
        stable_regime = self._apply_hysteresis(regime)

        # Volume-delta override: a sudden order-book imbalance should push the
        # regime out of a passive (QUIET / RANGING) state before ADX and EMA
        # can catch up.  When |volume_delta_pct| exceeds the spike threshold:
        #   * Use EMA slope to determine direction if available → TRENDING_UP/DOWN
        #   * Fall back to VOLATILE when direction is unclear
        #
        # Volume-delta spikes **bypass hysteresis** — they are designed to react
        # faster than the multi-candle dwell window, so the forced regime is
        # adopted immediately and the hysteresis state is updated accordingly.
        volume_delta_pct: Optional[float] = None
        if volume_delta is not None:
            abs_volume_delta = abs(volume_delta)
            if abs_volume_delta >= _VOLUME_DELTA_SPIKE_PCT:
                volume_delta_pct = float(volume_delta)
                if stable_regime in (MarketRegime.QUIET, MarketRegime.RANGING):
                    ema_slope_threshold = 0.15 if timeframe == "1m" else 0.05
                    forced_regime: MarketRegime = MarketRegime.VOLATILE  # default
                    if ema_slope is not None and ema_slope > ema_slope_threshold:
                        forced_regime = MarketRegime.TRENDING_UP
                        log.debug(
                            "Volume-delta spike (%.1f%%) forced QUIET/RANGING → TRENDING_UP",
                            volume_delta,
                        )
                    elif ema_slope is not None and ema_slope < -ema_slope_threshold:
                        forced_regime = MarketRegime.TRENDING_DOWN
                        log.debug(
                            "Volume-delta spike (%.1f%%) forced QUIET/RANGING → TRENDING_DOWN",
                            volume_delta,
                        )
                    else:
                        forced_regime = MarketRegime.VOLATILE
                        log.debug(
                            "Volume-delta spike (%.1f%%) forced QUIET/RANGING → VOLATILE",
                            volume_delta,
                        )
                    # Bypass hysteresis: adopt the forced regime immediately and
                    # update the stable state so future calls inherit it.
                    stable_regime = forced_regime
                    self._previous_regime = forced_regime
                    self._pending_regime = forced_regime
                    self._regime_dwell_count = self._hysteresis_candles

        return RegimeResult(
            regime=stable_regime,
            adx=adx_val,
            bb_width_pct=bb_width_pct,
            ema_slope=ema_slope,
            volume_delta_pct=volume_delta_pct,
        )

    def _apply_hysteresis(self, raw_regime: MarketRegime) -> MarketRegime:
        """Apply 3-candle dwell-time hysteresis to prevent rapid regime flapping.

        A new regime is only officially adopted after it has been the raw
        classification for ``_hysteresis_candles`` consecutive calls.  Until
        then the previously stable regime is returned.

        Parameters
        ----------
        raw_regime:
            The regime produced by :meth:`_decide` for the current candle.

        Returns
        -------
        MarketRegime
            The stable (hysteresis-filtered) regime.
        """
        if self._previous_regime is None:
            # Initial state: accept the first classification immediately.
            self._previous_regime = raw_regime
            self._pending_regime = raw_regime
            self._regime_dwell_count = self._hysteresis_candles
            return raw_regime

        if raw_regime == self._previous_regime:
            # Raw regime matches the stable regime — reset any pending transition.
            self._pending_regime = raw_regime
            self._regime_dwell_count = self._hysteresis_candles
            return self._previous_regime

        # Raw regime differs from the stable regime — track as a transition candidate.
        if raw_regime == self._pending_regime:
            # Same candidate as last time: increment consecutive counter.
            self._regime_dwell_count += 1
        else:
            # New candidate: reset counter from 1.
            self._pending_regime = raw_regime
            self._regime_dwell_count = 1

        if self._regime_dwell_count >= self._hysteresis_candles:
            # Candidate has persisted long enough — officially switch regime.
            log.debug(
                "Regime switch: %s → %s (dwell=%d)",
                self._previous_regime.value,
                raw_regime.value,
                self._regime_dwell_count,
            )
            self._previous_regime = raw_regime
            self._pending_regime = raw_regime
            self._regime_dwell_count = self._hysteresis_candles
            return raw_regime

        # Not yet enough consecutive readings — return the stable regime.
        return self._previous_regime

    # ------------------------------------------------------------------

    @staticmethod
    def _decide(
        adx: Optional[float],
        ema_slope: Optional[float],
        bb_width_pct: Optional[float],
        timeframe: str = "5m",
    ) -> MarketRegime:
        # EMA slope threshold – wider for 1m data to reduce noise-driven flips
        ema_slope_threshold = 0.15 if timeframe == "1m" else 0.05
        # Volatility check (Bollinger width) takes priority
        if bb_width_pct is not None:
            if bb_width_pct >= _BB_WIDTH_VOLATILE_PCT:
                return MarketRegime.VOLATILE
            if bb_width_pct <= _BB_WIDTH_QUIET_PCT:
                return MarketRegime.QUIET

        # Weak trend zone (ADX 20-25) — use EMA slope to decide before the
        # standard trending/ranging thresholds so the ADX dead zone is resolved.
        if adx is not None and _ADX_RANGING_MAX < adx < _ADX_TRENDING_MIN:
            if ema_slope is not None and abs(ema_slope) > ema_slope_threshold:
                return MarketRegime.TRENDING_UP if ema_slope > 0 else MarketRegime.TRENDING_DOWN
            return MarketRegime.RANGING

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
            if ema_slope > ema_slope_threshold:
                return MarketRegime.TRENDING_UP
            if ema_slope < -ema_slope_threshold:
                return MarketRegime.TRENDING_DOWN

        return MarketRegime.RANGING
