"""Market Regime Detection.

Classifies the current market regime based on technical indicators so that
channel evaluators and the confidence scorer can adapt their behaviour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from src.utils import get_logger
from src.indicators import adx as _compute_adx, atr as _compute_atr, ema as _compute_ema

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


@dataclass
class RegimeContext:
    """Rich regime context for downstream signal enrichment."""

    label: str                    # TRENDING_UP / TRENDING_DOWN / RANGING / VOLATILE / QUIET
    adx_value: float              # Raw ADX
    adx_slope: float              # adx[t] - adx[t-1]; positive = strengthening
    atr_percentile: float         # 0-100 rolling percentile of current ATR vs last 200 bars
    volume_profile: str           # "ACCUMULATION", "DISTRIBUTION", "NEUTRAL"
    is_regime_strengthening: bool  # adx_slope > 0 and adx_value > 20


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------


def atr_percentile(atr_series: np.ndarray, lookback: int = 200) -> float:
    """Return rolling percentile (0-100) of the last ATR value vs prior `lookback` bars."""
    if len(atr_series) < 2:
        return 50.0
    window = atr_series[-lookback:] if len(atr_series) >= lookback else atr_series
    current = float(atr_series[-1])
    return float(np.sum(window <= current) / len(window) * 100)


def volume_profile_classify(
    volumes: np.ndarray,
    closes: np.ndarray,
    vwap: float,
    lookback: int = 20,
) -> str:
    """Classify volume profile as ACCUMULATION, DISTRIBUTION, or NEUTRAL."""
    if vwap <= 0 or len(closes) < lookback or len(volumes) < lookback:
        return "NEUTRAL"
    c = np.asarray(closes[-lookback:], dtype=float)
    v = np.asarray(volumes[-lookback:], dtype=float)
    above_vol = float(np.sum(v[c >= vwap]))
    below_vol = float(np.sum(v[c < vwap]))
    total = above_vol + below_vol
    if total == 0:
        return "NEUTRAL"
    ratio = above_vol / total
    if ratio > 0.60:
        return "ACCUMULATION"
    if ratio < 0.40:
        return "DISTRIBUTION"
    return "NEUTRAL"


# Thresholds (tunable via environment variables in the future)
_ADX_TRENDING_MIN: float = 25.0
_ADX_RANGING_MAX: float = float(os.getenv("ADX_RANGING_MAX", "18.0"))
_BB_WIDTH_VOLATILE_PCT: float = 5.0   # Bollinger width as % of price
_BB_WIDTH_QUIET_PCT: float = float(os.getenv("BB_WIDTH_QUIET_PCT", "1.2"))
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

    def build_regime_context(
        self,
        result: RegimeResult,
        candles: Optional[Dict[str, Any]] = None,
        indicators: Optional[Dict[str, Any]] = None,
        vwap: float = 0.0,
    ) -> RegimeContext:
        """Build a rich RegimeContext from a RegimeResult and raw market data."""
        adx_val = result.adx if result.adx is not None else 0.0
        adx_slope = 0.0
        atr_pct = 50.0
        vol_profile = "NEUTRAL"

        if candles is not None:
            closes = candles.get("close", [])
            highs = candles.get("high", [])
            lows = candles.get("low", [])
            volumes = candles.get("volume", [])

            # ADX slope: compute full ADX array and take last two values
            if len(closes) >= 30:
                from src.indicators import adx as compute_adx  # noqa: PLC0415
                h = np.asarray(highs, dtype=np.float64)
                lo = np.asarray(lows, dtype=np.float64)
                c = np.asarray(closes, dtype=np.float64)
                adx_arr = compute_adx(h, lo, c, 14)
                valid = adx_arr[~np.isnan(adx_arr)]
                if len(valid) >= 2:
                    adx_slope = float(valid[-1] - valid[-2])

            # ATR percentile
            if len(closes) >= 15:
                from src.indicators import atr as compute_atr  # noqa: PLC0415
                h = np.asarray(highs, dtype=np.float64)
                lo = np.asarray(lows, dtype=np.float64)
                c = np.asarray(closes, dtype=np.float64)
                atr_arr = compute_atr(h, lo, c, 14)
                valid_atr = atr_arr[~np.isnan(atr_arr)]
                if len(valid_atr) >= 2:
                    atr_pct = atr_percentile(valid_atr)

            # Volume profile
            if len(volumes) >= 20 and len(closes) >= 20 and vwap > 0:
                vol_profile = volume_profile_classify(
                    np.asarray(volumes, dtype=np.float64),
                    np.asarray(closes, dtype=np.float64),
                    vwap,
                )

        return RegimeContext(
            label=result.regime.value,
            adx_value=adx_val,
            adx_slope=adx_slope,
            atr_percentile=atr_pct,
            volume_profile=vol_profile,
            is_regime_strengthening=(adx_slope > 0 and adx_val > 20),
        )

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


# ---------------------------------------------------------------------------
# Tier-specific regime threshold profiles
# ---------------------------------------------------------------------------

_TIER_REGIME_PROFILES: Dict[str, Dict[str, float]] = {
    "MAJOR": {
        "adx_trending_min": float(os.getenv("MAJOR_ADX_TRENDING_MIN", "28.0")),
        "adx_ranging_max":  float(os.getenv("MAJOR_ADX_RANGING_MAX",  "22.0")),
        "bb_width_quiet":   float(os.getenv("MAJOR_BB_WIDTH_QUIET",    "1.0")),
        "bb_width_volatile": float(os.getenv("MAJOR_BB_WIDTH_VOLATILE", "4.0")),
    },
    "MIDCAP": {
        "adx_trending_min": float(os.getenv("MIDCAP_ADX_TRENDING_MIN", "25.0")),
        "adx_ranging_max":  float(os.getenv("MIDCAP_ADX_RANGING_MAX",  "20.0")),
        "bb_width_quiet":   float(os.getenv("MIDCAP_BB_WIDTH_QUIET",    "1.2")),
        "bb_width_volatile": float(os.getenv("MIDCAP_BB_WIDTH_VOLATILE", "5.0")),
    },
    "ALTCOIN": {
        "adx_trending_min": float(os.getenv("ALTCOIN_ADX_TRENDING_MIN", "20.0")),
        "adx_ranging_max":  float(os.getenv("ALTCOIN_ADX_RANGING_MAX",  "15.0")),
        "bb_width_quiet":   float(os.getenv("ALTCOIN_BB_WIDTH_QUIET",    "0.8")),
        "bb_width_volatile": float(os.getenv("ALTCOIN_BB_WIDTH_VOLATILE", "6.0")),
    },
}


class AdaptiveRegimeDetector(MarketRegimeDetector):
    """Regime detector with tier-specific ADX and Bollinger Band thresholds.

    Applies different classification thresholds based on the pair's volume
    tier (MAJOR, MIDCAP, ALTCOIN) to avoid misclassifying altcoins that have
    lower absolute ADX values for comparable trend strength.

    Parameters
    ----------
    pair_tier:
        Volume tier of the pair: ``"MAJOR"``, ``"MIDCAP"``, or ``"ALTCOIN"``.
    hysteresis_candles:
        Passed through to :class:`MarketRegimeDetector`.
    """

    def __init__(self, pair_tier: str = "MIDCAP", hysteresis_candles: int = 3) -> None:
        super().__init__(hysteresis_candles=hysteresis_candles)
        self._pair_tier = pair_tier
        profile = _TIER_REGIME_PROFILES.get(pair_tier, _TIER_REGIME_PROFILES["MIDCAP"])
        self._adx_trending_min: float = profile["adx_trending_min"]
        self._adx_ranging_max: float = profile["adx_ranging_max"]
        self._bb_width_quiet: float = profile["bb_width_quiet"]
        self._bb_width_volatile: float = profile["bb_width_volatile"]

    def classify(
        self,
        indicators: Dict[str, Any],
        candles: Optional[Dict[str, Any]] = None,
        timeframe: str = "5m",
        volume_delta: Optional[float] = None,
    ) -> RegimeResult:
        """Classify using tier-specific thresholds.

        Overrides :meth:`MarketRegimeDetector.classify` to use instance-level
        threshold parameters via :meth:`_decide_adaptive` instead of modifying
        the shared module-level constants (which would create a race condition
        when multiple detectors run concurrently).
        """
        adx_val: Optional[float] = indicators.get("adx_last")
        ema_fast: Optional[float] = indicators.get("ema9_last")
        ema_slow: Optional[float] = indicators.get("ema21_last")
        bb_upper: Optional[float] = indicators.get("bb_upper_last")
        bb_lower: Optional[float] = indicators.get("bb_lower_last")
        bb_mid: Optional[float] = indicators.get("bb_mid_last")

        if timeframe != self._last_timeframe:
            self._previous_regime = None
            self._pending_regime = None
            self._regime_dwell_count = 0
            self._last_timeframe = timeframe

        close: Optional[float] = None
        if candles is not None and len(candles.get("close", [])) > 0:
            close = float(candles["close"][-1])
        if ema_fast is None and close is not None:
            ema_fast = close
        if ema_slow is None and close is not None:
            ema_slow = close

        ema_slope: Optional[float] = None
        if ema_fast is not None and ema_slow is not None and ema_slow != 0.0:
            ema_slope = (ema_fast - ema_slow) / ema_slow * 100.0

        bb_width_pct: Optional[float] = None
        if bb_upper is not None and bb_lower is not None and bb_mid and bb_mid != 0.0:
            bb_width_pct = (bb_upper - bb_lower) / bb_mid * 100.0

        regime = self._decide_adaptive(adx_val, ema_slope, bb_width_pct, timeframe=timeframe)
        stable_regime = self._apply_hysteresis(regime)

        volume_delta_pct: Optional[float] = None
        if volume_delta is not None:
            abs_volume_delta = abs(volume_delta)
            if abs_volume_delta >= _VOLUME_DELTA_SPIKE_PCT:
                volume_delta_pct = float(volume_delta)
                if stable_regime in (MarketRegime.QUIET, MarketRegime.RANGING):
                    ema_slope_threshold = 0.15 if timeframe == "1m" else 0.05
                    forced_regime: MarketRegime = MarketRegime.VOLATILE
                    if ema_slope is not None and ema_slope > ema_slope_threshold:
                        forced_regime = MarketRegime.TRENDING_UP
                    elif ema_slope is not None and ema_slope < -ema_slope_threshold:
                        forced_regime = MarketRegime.TRENDING_DOWN
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

    def _decide_adaptive(
        self,
        adx: Optional[float],
        ema_slope: Optional[float],
        bb_width_pct: Optional[float],
        timeframe: str = "5m",
    ) -> MarketRegime:
        """Like :meth:`_decide` but uses instance-level tier thresholds."""
        ema_slope_threshold = 0.15 if timeframe == "1m" else 0.05
        if bb_width_pct is not None:
            if bb_width_pct >= self._bb_width_volatile:
                return MarketRegime.VOLATILE
            if bb_width_pct <= self._bb_width_quiet:
                return MarketRegime.QUIET
        if adx is not None and self._adx_ranging_max < adx < self._adx_trending_min:
            if ema_slope is not None and abs(ema_slope) > ema_slope_threshold:
                return MarketRegime.TRENDING_UP if ema_slope > 0 else MarketRegime.TRENDING_DOWN
            return MarketRegime.RANGING
        if adx is not None and adx >= self._adx_trending_min:
            if ema_slope is not None:
                return MarketRegime.TRENDING_UP if ema_slope > 0 else MarketRegime.TRENDING_DOWN
            return MarketRegime.TRENDING_UP
        if adx is not None and adx <= self._adx_ranging_max:
            return MarketRegime.RANGING
        if ema_slope is not None:
            if ema_slope > ema_slope_threshold:
                return MarketRegime.TRENDING_UP
            if ema_slope < -ema_slope_threshold:
                return MarketRegime.TRENDING_DOWN
        return MarketRegime.RANGING


# ---------------------------------------------------------------------------
# Vectorised regime detection for historical array replay
# ---------------------------------------------------------------------------


def detect_regime_from_arrays(
    closes: "np.ndarray",
    highs: "np.ndarray",
    lows: "np.ndarray",
    volumes: "np.ndarray",
    idx: int,
    lookback: int = 14,
) -> str:
    """Detect market regime at a specific bar index in a historical array.

    Parameters
    ----------
    closes, highs, lows, volumes:
        Full historical arrays (length >= idx + 1).
    idx:
        Bar index for which to detect the regime.
    lookback:
        ATR/ADX computation lookback.

    Returns
    -------
    str: regime label ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "QUIET")
    """
    start = max(0, idx - lookback * 3)
    end = idx + 1
    c = np.asarray(closes[start:end], dtype=float)
    h = np.asarray(highs[start:end], dtype=float)
    lo = np.asarray(lows[start:end], dtype=float)

    # Require at least 2 × lookback bars so indicator warm-up periods are
    # satisfied before returning a meaningful regime label.
    if len(c) < lookback * 2:
        return MarketRegime.RANGING.value

    adx_series = _compute_adx(h, lo, c, period=lookback)
    atr_series = _compute_atr(h, lo, c, period=lookback)
    ema9 = _compute_ema(c, 9)
    ema21 = _compute_ema(c, 21)

    adx_val = float(adx_series[-1]) if not np.isnan(adx_series[-1]) else 0.0
    atr_val = float(atr_series[-1]) if not np.isnan(atr_series[-1]) else 0.0
    price = float(c[-1])
    atr_pct = (atr_val / price * 100) if price > 0 else 0.5

    if adx_val >= _ADX_TRENDING_MIN:
        ema9_last = float(ema9[-1]) if not np.isnan(ema9[-1]) else float(c[-1])
        ema21_last = float(ema21[-1]) if not np.isnan(ema21[-1]) else float(c[-1])
        return (
            MarketRegime.TRENDING_UP.value
            if ema9_last > ema21_last
            else MarketRegime.TRENDING_DOWN.value
        )
    if atr_pct >= 1.5:
        return MarketRegime.VOLATILE.value
    if atr_pct <= 0.3:
        return MarketRegime.QUIET.value
    return MarketRegime.RANGING.value
