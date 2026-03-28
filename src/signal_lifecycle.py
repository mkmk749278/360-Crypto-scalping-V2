"""Signal Lifecycle Monitor.

Actively monitors every open SPOT, GEM, and SWING signal and posts
human-readable status updates to the Portfolio / Active Trading channels.

Instead of fire-and-forget, this module gives subscribers ongoing visibility
into their open positions — regime changes, momentum shifts, market structure
breaks, TP progress, and close recommendations.

Checks are spaced per channel:
  SWING  — every 4 hours  (LIFECYCLE_CHECK_INTERVAL_SWING)
  SPOT   — every 6 hours  (LIFECYCLE_CHECK_INTERVAL_SPOT)
  GEM    — every 12 hours (LIFECYCLE_CHECK_INTERVAL_GEM)

SCALP is deliberately excluded — scalp positions are too short-lived for
lifecycle updates to be meaningful.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from config import (
    CHANNEL_TELEGRAM_MAP,
    LIFECYCLE_CHECK_INTERVAL,
    LIFECYCLE_CONFIDENCE_DROP_RED,
    LIFECYCLE_CONFIDENCE_DROP_YELLOW,
)
from src.channels.base import Signal
from src.historical_data import HistoricalDataStore
from src.regime import MarketRegime, MarketRegimeDetector
from src.smc import Direction
from src.utils import get_logger, utcnow

log = get_logger("signal_lifecycle")

# Poll interval for the main loop (seconds).  The loop wakes up this often
# and checks whether any signal is due for a lifecycle check.
_POLL_INTERVAL: int = 60

# Minimum number of candles required to compute momentum indicators.
_MIN_CANDLES: int = 22

# Default lifecycle check intervals for channels not in LIFECYCLE_CHECK_INTERVAL.
_DEFAULT_SCALP_LIFECYCLE_INTERVAL: int = 900    # 15 minutes for scalp channels
_DEFAULT_LIFECYCLE_INTERVAL: int = 3600         # 1 hour generic fallback


def get_lifecycle_interval(channel_name: str) -> int:
    """Return the lifecycle check interval (seconds) for *channel_name*.

    Uses :data:`config.LIFECYCLE_CHECK_INTERVAL` as the primary lookup.
    Falls back to a 15-minute interval for all SCALP-family channels, and
    to a 1-hour interval for any other unrecognised channel name.

    Parameters
    ----------
    channel_name:
        Name of the trading channel, e.g. ``"360_SCALP"``, ``"360_SWING"``.

    Returns
    -------
    int
        Interval in seconds.
    """
    if channel_name in LIFECYCLE_CHECK_INTERVAL:
        return LIFECYCLE_CHECK_INTERVAL[channel_name]
    if "SCALP" in channel_name:
        return _DEFAULT_SCALP_LIFECYCLE_INTERVAL
    return _DEFAULT_LIFECYCLE_INTERVAL


# Primary timeframe for each lifecycle-monitored channel (used to pull
# candles for regime / momentum / structure assessment).
_PRIMARY_TIMEFRAME: Dict[str, str] = {
    "360_SWING": "4h",
    "360_SPOT":  "4h",
    "360_GEM":   "1d",
}

# Channels eligible for lifecycle monitoring.
_MONITORED_CHANNELS: frozenset = frozenset(LIFECYCLE_CHECK_INTERVAL.keys())


def _utcnow() -> datetime:
    """Return the current UTC datetime (test-injectable)."""
    return utcnow()


def _escape_md(text: str) -> str:
    """Escape Telegram MarkdownV1 special characters in dynamic text fields."""
    for ch in ("\\", "*", "_", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


def _compute_ema(prices: List[float], period: int) -> Optional[float]:
    """Compute the last EMA value for *prices* with the given *period*.

    Uses the standard smoothing factor k = 2 / (period + 1).
    Returns ``None`` when there are not enough data points.
    """
    if len(prices) < period:
        return None
    k = 2.0 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = price * k + ema * (1 - k)
    return ema


def _compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """Return RSI(period) for the given close series.

    Returns ``None`` when there are not enough data points.
    """
    if len(closes) < period + 1:
        return None
    gains, losses = [], []  # type: List[float], List[float]
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        (gains if diff > 0 else losses).append(abs(diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


class SignalLifecycleMonitor:
    """Background monitor for open SPOT/GEM/SWING signals.

    Parameters
    ----------
    router:
        The :class:`~src.signal_router.SignalRouter` instance.  Accessed for
        ``router.active_signals`` and ``router.update_signal``.
    data_store:
        :class:`~src.historical_data.HistoricalDataStore` supplying candle data.
    regime_detector:
        :class:`~src.regime.MarketRegimeDetector` for current-regime checks.
    send_telegram:
        Async callable ``(chat_id: str, text: str) → bool``.
    exchange_mgr:
        Optional exchange manager (not currently used; reserved for future
        price cross-checks against a second exchange).
    send_photo:
        Optional async callable ``(chat_id: str, photo_bytes: bytes) → bool``.
        When provided, chart images are sent alongside text lifecycle updates.
    """

    def __init__(
        self,
        router: Any,
        data_store: HistoricalDataStore,
        regime_detector: MarketRegimeDetector,
        send_telegram: Callable[[str, str], Coroutine],
        exchange_mgr: Any = None,
        send_photo: Optional[Callable[[str, bytes], Coroutine]] = None,
    ) -> None:
        self._router = router
        self._data_store = data_store
        self._regime_detector = regime_detector
        self._send_telegram = send_telegram
        self._exchange_mgr = exchange_mgr
        self._send_photo = send_photo
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Main monitoring loop.  Runs until :meth:`stop` is called."""
        self._running = True
        log.info("SignalLifecycleMonitor started")
        while self._running:
            try:
                await self._tick()
            except Exception as exc:
                log.error("Lifecycle monitor tick error: {}", exc)
            await asyncio.sleep(_POLL_INTERVAL)

    async def stop(self) -> None:
        """Signal the monitor to stop after the current tick."""
        self._running = False

    async def _tick(self) -> None:
        """Evaluate all open signals that are due for a lifecycle check."""
        signals = list(self._router.active_signals.values())
        for signal in signals:
            if signal.channel not in _MONITORED_CHANNELS:
                continue
            if self._is_due(signal):
                try:
                    await self._check_signal(signal)
                except Exception as exc:
                    log.error(
                        "Lifecycle check failed for {} {}: {}",
                        signal.symbol, signal.channel, exc,
                    )

    def _is_due(self, signal: Signal) -> bool:
        """Return True if it is time to run a lifecycle check for *signal*."""
        interval = LIFECYCLE_CHECK_INTERVAL.get(signal.channel)
        if interval is None:
            return False
        if signal.last_lifecycle_check is None:
            # Use the signal's creation time so the first check respects the
            # full interval (prevents immediate firing after signal creation).
            elapsed = (_utcnow() - signal.timestamp).total_seconds()
        else:
            elapsed = (_utcnow() - signal.last_lifecycle_check).total_seconds()
        return elapsed >= interval

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    async def _check_signal(self, signal: Signal) -> None:
        """Evaluate one open signal and post an update if warranted."""
        tf = _PRIMARY_TIMEFRAME.get(signal.channel, "4h")
        candles = self._data_store.get_candles(signal.symbol, tf)

        current_price = signal.current_price or signal.entry

        # Gather assessments
        assessments: List[str] = []

        regime_text = self._assess_regime_change(signal, candles)
        if regime_text:
            assessments.append(regime_text)

        momentum_text = self._assess_momentum(signal, candles)
        if momentum_text:
            assessments.append(momentum_text)

        structure_text = self._assess_structure(signal, candles)
        if structure_text:
            assessments.append(structure_text)

        confidence_text = self._assess_confidence_decay(signal)
        if confidence_text:
            assessments.append(confidence_text)

        tp_text = self._assess_tp_progress(signal, current_price)
        if tp_text:
            assessments.append(tp_text)

        should_close, close_reason = self._should_recommend_close(signal, assessments)

        # Determine overall alert level
        red_flags = sum(1 for a in assessments if a.startswith("🔴"))
        yellow_flags = sum(1 for a in assessments if a.startswith("🟡"))

        if should_close or red_flags >= 2:
            new_level = "RED"
        elif red_flags >= 1 or yellow_flags >= 2:
            new_level = "YELLOW"
        else:
            new_level = "GREEN"

        # Update signal state
        self._router.update_signal(
            signal.signal_id,
            last_lifecycle_check=_utcnow(),
            lifecycle_alert_level=new_level,
        )

        # Format and post message
        message = self._format_update_message(
            signal=signal,
            assessments=assessments,
            current_price=current_price,
            alert_level=new_level,
            should_close=should_close,
            close_reason=close_reason,
        )
        await self._post_update(signal, message)

        # Send chart image alongside text update when photo callback is available (feature 4)
        if self._send_photo is not None:
            await self._post_chart(signal, candles)

    # ------------------------------------------------------------------
    # Individual assessment helpers
    # ------------------------------------------------------------------

    def _assess_regime_change(
        self, signal: Signal, candles: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compare current regime to the regime when the signal was opened.

        Returns a formatted assessment string prefixed with a traffic-light
        emoji, or ``None`` when the regime is unchanged or cannot be computed.
        """
        if not candles or len(candles.get("close", [])) < _MIN_CANDLES:
            return None

        closes = list(candles["close"])
        ema9 = _compute_ema(closes, 9)
        ema21 = _compute_ema(closes, 21)
        if ema9 is None or ema21 is None or ema21 == 0.0:
            return None

        indicators = {"ema9_last": ema9, "ema21_last": ema21}
        result = self._regime_detector.classify(indicators)
        current_regime = result.regime.value

        entry_regime = signal.entry_regime or signal.market_phase
        if not entry_regime or entry_regime == "N/A":
            # First time we see this signal — record the regime for future checks
            return f"🟢 Regime: {current_regime}"

        if current_regime == entry_regime:
            return f"🟢 Regime: {current_regime} (unchanged)"

        # Regime has changed — assess severity
        bearish_regimes = {MarketRegime.TRENDING_DOWN.value, MarketRegime.VOLATILE.value}
        bullish_regimes = {MarketRegime.TRENDING_UP.value}

        if signal.direction == Direction.LONG:
            if current_regime in bearish_regimes:
                return f"🔴 Regime: flipped {entry_regime} → {current_regime}"
            return f"🟡 Regime: shifted {entry_regime} → {current_regime}"
        else:  # SHORT
            if current_regime in bullish_regimes:
                return f"🔴 Regime: flipped {entry_regime} → {current_regime}"
            return f"🟡 Regime: shifted {entry_regime} → {current_regime}"

    def _assess_momentum(
        self, signal: Signal, candles: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Assess whether momentum is intact, fading, or lost.

        Uses EMA slope (fast vs slow) and RSI direction to determine trend
        momentum relative to the position direction.
        """
        if not candles or len(candles.get("close", [])) < _MIN_CANDLES:
            return None

        closes = list(candles["close"])
        ema9 = _compute_ema(closes, 9)
        ema21 = _compute_ema(closes, 21)
        rsi = _compute_rsi(closes, period=14)

        if ema9 is None or ema21 is None or ema21 == 0.0:
            return None

        slope_pct = (ema9 - ema21) / ema21 * 100.0

        if signal.direction == Direction.LONG:
            if slope_pct > 0.1:
                if rsi is not None and rsi >= 50:
                    return "🟢 Momentum: Strong — EMA slope positive, RSI bullish"
                return "🟢 Momentum: Positive — EMA slope trending up"
            elif slope_pct > -0.1:
                return "🟡 Momentum: Fading — EMA slope flattening"
            else:
                if rsi is not None and rsi < 40:
                    return "🔴 Momentum: Lost — EMA slope negative, RSI < 40"
                return "🔴 Momentum: Lost — EMA slope negative"
        else:  # SHORT
            if slope_pct < -0.1:
                if rsi is not None and rsi <= 50:
                    return "🟢 Momentum: Strong — EMA slope negative, RSI bearish"
                return "🟢 Momentum: Negative — EMA slope trending down"
            elif slope_pct < 0.1:
                return "🟡 Momentum: Fading — EMA slope flattening"
            else:
                if rsi is not None and rsi > 60:
                    return "🔴 Momentum: Lost — EMA slope positive, RSI > 60"
                return "🔴 Momentum: Lost — EMA slope turned against position"

    def _assess_structure(
        self, signal: Signal, candles: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Check for a Break of Structure (BOS) against the position direction.

        A BOS is detected when the most recent candle's low (LONG) or high
        (SHORT) breaks through the recent swing low/high, signalling a
        structural change in the market.
        """
        if not candles:
            return None
        closes = candles.get("close", [])
        lows = candles.get("low", [])
        highs = candles.get("high", [])

        if len(closes) < 10 or len(lows) < 10 or len(highs) < 10:
            return None

        # Use the last 10 candles to detect a swing high/low reference point,
        # excluding the most recent candle (which we are checking against).
        recent_lows = list(lows[-10:-1])
        recent_highs = list(highs[-10:-1])
        current_low = float(lows[-1])
        current_high = float(highs[-1])

        if not recent_lows or not recent_highs:
            return None

        swing_low = min(recent_lows)
        swing_high = max(recent_highs)

        if signal.direction == Direction.LONG:
            if current_low < swing_low:
                return "🔴 Structure: BROKEN — lower low printed against LONG position"
            return "🟢 Structure: Intact — no BOS against position"
        else:  # SHORT
            if current_high > swing_high:
                return "🔴 Structure: BROKEN — higher high printed against SHORT position"
            return "🟢 Structure: Intact — no BOS against position"

    def _assess_confidence_decay(self, signal: Signal) -> Optional[str]:
        """Compare current confidence to entry confidence.

        Uses the signal's ``pre_ai_confidence`` as the entry baseline (it
        reflects the purely quantitative score at the time of signal creation).
        Falls back to ``confidence`` when ``pre_ai_confidence`` is zero.
        """
        entry_conf = signal.pre_ai_confidence or signal.confidence
        current_conf = signal.confidence

        if entry_conf == 0.0:
            return None

        drop = entry_conf - current_conf
        if drop <= 0:
            return None

        if drop >= LIFECYCLE_CONFIDENCE_DROP_RED:
            return (
                f"🔴 Confidence: {entry_conf:.0f} → {current_conf:.0f} "
                f"(dropped {drop:.0f}pts)"
            )
        if drop >= LIFECYCLE_CONFIDENCE_DROP_YELLOW:
            return (
                f"🟡 Confidence: {entry_conf:.0f} → {current_conf:.0f} "
                f"(dropped {drop:.0f}pts)"
            )
        return None

    def _assess_tp_progress(
        self, signal: Signal, current_price: float
    ) -> Optional[str]:
        """Report how close the current price is to TP1.

        Returns a celebratory message when TP1 has already been hit, or shows
        the percentage progress toward TP1 when it hasn't.
        """
        entry = signal.entry
        tp1 = signal.tp1
        if entry == 0 or tp1 == 0:
            return None

        if signal.best_tp_hit >= 1:
            # TP1 already hit — show progress toward TP2
            tp2 = signal.tp2
            if not tp2:
                return None
            if signal.best_tp_hit >= 2:
                return f"🎯 TP1 ✅ TP2 ✅ — riding toward TP3: ${signal.tp3 or 0:.5g}"
            dist_to_tp2 = abs(tp2 - entry)
            if dist_to_tp2 == 0:
                return None
            progress = abs(current_price - entry) / dist_to_tp2 * 100.0
            progress = min(progress, 100.0)
            return (
                f"🎯 TP1 ✅ — {progress:.0f}% toward TP2 ${tp2:.5g} "
                f"(current ${current_price:.5g})"
            )

        # TP1 not yet hit
        dist_to_tp1 = abs(tp1 - entry)
        if dist_to_tp1 == 0:
            return None

        if signal.direction == Direction.LONG:
            moved = current_price - entry
        else:
            moved = entry - current_price

        progress = moved / dist_to_tp1 * 100.0
        progress = max(0.0, min(progress, 100.0))

        if progress >= 80.0:
            return f"🎯 TP1 at ${tp1:.5g} — {progress:.0f}% there! Consider partial profits"
        return f"🎯 TP1 at ${tp1:.5g} — {progress:.0f}% of the way there"

    def _should_recommend_close(
        self, signal: Signal, assessments: List[str]
    ) -> Tuple[bool, str]:
        """Return (True, reason) when multiple red flags suggest closing.

        The threshold is: at least 3 red-flag assessments must fire, covering
        at least 2 of the 4 core concern categories (regime, momentum,
        structure, confidence).
        """
        red_assessments = [a for a in assessments if a.startswith("🔴")]
        if len(red_assessments) < 2:
            return False, ""

        has_regime = any("Regime" in a for a in red_assessments)
        has_momentum = any("Momentum" in a for a in red_assessments)
        has_structure = any("Structure" in a for a in red_assessments)
        has_confidence = any("Confidence" in a for a in red_assessments)

        distinct_categories = sum([has_regime, has_momentum, has_structure, has_confidence])
        if distinct_categories < 2:
            return False, ""

        parts = []
        if has_regime:
            parts.append("regime reversal")
        if has_structure:
            parts.append("market structure break")
        if has_momentum:
            parts.append("momentum lost")
        if has_confidence:
            parts.append("confidence collapsed")

        reason = " + ".join(parts).capitalize()
        return True, reason

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    def _format_update_message(
        self,
        signal: Signal,
        assessments: List[str],
        current_price: float,
        alert_level: str,
        should_close: bool,
        close_reason: str,
    ) -> str:
        """Format a human-readable Telegram update for *signal*."""
        entry = signal.entry
        pnl_pct = (
            (current_price - entry) / entry * 100.0
            if signal.direction == Direction.LONG
            else (entry - current_price) / entry * 100.0
        )
        pnl_sign = "+" if pnl_pct >= 0 else ""

        interval_seconds = get_lifecycle_interval(signal.channel)
        interval_hours = interval_seconds // 3600

        if should_close:
            header_prefix = "⛔ URGENT —"
        elif alert_level == "RED":
            header_prefix = "🔴"
        elif alert_level == "YELLOW":
            header_prefix = "🟡"
        else:
            header_prefix = "📊"

        lines = [
            f"{header_prefix} {signal.symbol} {signal.direction.value} "
            f"— {interval_hours}h Update",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"Entry: ${entry:.5g} → Now: ${current_price:.5g} "
            f"({pnl_sign}{pnl_pct:.1f}%)",
        ]

        for assessment in assessments:
            lines.append(_escape_md(assessment))

        if should_close:
            lines.append("⛔ CLOSE RECOMMENDED — thesis invalidated")
            lines.append(f"Reason: {close_reason}")
        elif alert_level == "RED":
            lines.append("💡 Consider closing — multiple red flags")
        elif alert_level == "YELLOW":
            lines.append("💡 Caution — consider tightening mental stop")
        else:
            lines.append("💡 Hold — thesis intact")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Telegram delivery
    # ------------------------------------------------------------------

    async def _post_update(self, signal: Signal, message: str) -> None:
        """Post a lifecycle update message to the signal's Telegram channel."""
        chat_id = CHANNEL_TELEGRAM_MAP.get(signal.channel, "")
        if not chat_id:
            log.debug(
                "No Telegram channel configured for {} — skipping lifecycle update",
                signal.channel,
            )
            return
        try:
            await self._send_telegram(chat_id, message)
            log.info(
                "Lifecycle update posted: {} {} ({})",
                signal.symbol, signal.channel, signal.lifecycle_alert_level,
            )
        except Exception as exc:
            log.error(
                "Failed to post lifecycle update for {} {}: {}",
                signal.symbol, signal.channel, exc,
            )

    async def _post_chart(
        self,
        signal: Signal,
        candles: Optional[Any],
    ) -> None:
        """Generate and post a chart image for *signal*.

        Uses :func:`~src.chart_generator.generate_gem_chart` to produce
        a PNG with overlays, then sends it via the ``_send_photo``
        callback.  Silently skips when candle data is insufficient or chart
        generation fails.
        """
        if self._send_photo is None:
            return

        chat_id = CHANNEL_TELEGRAM_MAP.get(signal.channel, "")
        if not chat_id:
            return

        if not candles:
            return

        try:
            from src.chart_generator import generate_gem_chart  # local import to avoid circular

            closes = [float(v) for v in candles.get("close", [])]
            highs = [float(v) for v in candles.get("high", [])]
            lows = [float(v) for v in candles.get("low", [])]
            volumes = [float(v) for v in candles.get("volume", [])]

            tp_levels = [t for t in [signal.tp1, signal.tp2, signal.tp3] if t]

            # Use generate_gem_chart for GEM signals, skip chart for others
            if signal.channel == "360_GEM":
                ema_20 = []  # EMA data not in candles dict; chart renders without EMA overlay
                ema_50 = []
                current_price = closes[-1] if closes else signal.entry
                ath = max(highs) if highs else current_price
                chart_bytes = generate_gem_chart(
                    symbol=signal.symbol,
                    daily_candles=candles,
                    ath=ath,
                    current_price=current_price,
                    ema_20=ema_20,
                    ema_50=ema_50,
                )
            else:
                return  # No chart generation for non-GEM channels

            if chart_bytes:
                await self._send_photo(chat_id, chart_bytes)
                log.debug(
                    "Lifecycle chart sent for {} {} ({} bytes)",
                    signal.symbol, signal.channel, len(chart_bytes),
                )
        except Exception as exc:
            log.debug(
                "Lifecycle chart generation failed for {} {}: {}",
                signal.symbol, signal.channel, exc,
            )
