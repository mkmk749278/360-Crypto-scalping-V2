"""Signal router – queue-based decoupled architecture.

Scanner → queue → Router → Telegram

The router:
  1. Consumes signals from an asyncio.Queue
  2. Enriches them with AI/predictive, confidence, risk
  3. Applies channel-specific min-confidence filter
  4. Posts to the appropriate Telegram channel
  5. Selects top 1–2 for the free channel
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import time
from datetime import date, datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from config import (
    ALL_CHANNELS,
    CHANNEL_COOLDOWN_SECONDS,
    CHANNEL_TELEGRAM_MAP,
    CHART_ENABLED_CHANNELS,
    MAX_CONCURRENT_SIGNALS_PER_CHANNEL,
    MAX_SIGNAL_HOLD_SECONDS,
    TELEGRAM_FREE_CHANNEL_ID,
)
from src.channels.base import Signal
from src.correlation import check_correlation_limit
from src.redis_client import RedisClient
from src.risk import RiskManager
from src.smc import Direction
from src.utils import get_logger
from src.ai_engine.predictor import SignalPredictor, PredictionFeatures
from src.ai_engine.scorer import AIConfidenceScorer, AIScoreResult
from src.cornix_formatter import format_cornix_signal

log = get_logger("signal_router")

# Max highlights posted to the free channel per calendar day.
_FREE_HIGHLIGHT_MAX_PER_DAY: int = 4
# Minimum TP level required to trigger a free-channel highlight.
_FREE_HIGHLIGHT_MIN_TP: int = 2

# SCALP channel names — used by the stale-signal gate and latency warnings.
_SCALP_CHANNEL_NAMES: frozenset = frozenset({
    "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD",
    "360_SCALP_VWAP", "360_SCALP_OBI",
})

# Stale-signal gate: maximum seconds a signal may spend between detection and
# posting before it is considered stale and suppressed.  For SCALP channels the
# window is tight (120 s) because micro-cap moves complete in 2-3 minutes.
# For all other channels the window is generous (used only as a safety net).
_SCALP_STALE_THRESHOLD_SECONDS: float = 120.0
_DEFAULT_STALE_THRESHOLD_SECONDS: float = 3600.0

# Latency WARNING threshold for SCALP signals (2 minutes per problem statement).
_SCALP_LATENCY_WARNING_SECONDS: float = 120.0

# Delivery-retry sleep callable – replaced in tests to avoid real waits.
async def _delivery_sleep(secs: float) -> None:
    await asyncio.sleep(secs)


def _signal_from_dict(data: dict) -> Optional[Signal]:
    """Reconstruct a Signal from a Redis-deserialized dict."""
    try:
        d = data.copy()
        if isinstance(d.get("direction"), str):
            d["direction"] = Direction(d["direction"])
        # Restore all datetime fields that were serialized as ISO strings
        for field in ("timestamp", "last_lifecycle_check", "dca_timestamp"):
            if isinstance(d.get(field), str):
                d[field] = datetime.fromisoformat(d[field])
        return Signal(**d)
    except Exception as exc:
        log.warning("Failed to reconstruct Signal from dict: {}", exc)
        return None


def _signal_to_dict(sig: Signal) -> dict:
    """Serialize a Signal to a JSON-serializable dict."""
    d = dataclasses.asdict(sig)
    d["direction"] = sig.direction.value  # Direction enum → string
    # Convert ALL datetime fields to ISO strings for JSON compatibility
    for k in list(d.keys()):
        if isinstance(d[k], datetime):
            d[k] = d[k].isoformat()
    return d


# Redis keys used for state persistence
_REDIS_KEY_SIGNALS = "signal_router:active_signals"
_REDIS_KEY_POSITION_LOCK = "signal_router:position_lock"
_REDIS_KEY_COOLDOWNS = "signal_router:cooldown_timestamps"


class SignalRouter:
    """Consumes signals from a queue, scores, filters, and dispatches."""

    def __init__(
        self,
        queue: Any,
        send_telegram: Callable[[str, str], Coroutine],
        format_signal: Callable[[Signal], str],
        redis_client: Optional[RedisClient] = None,
    ) -> None:
        self._queue = queue
        self._send_telegram = send_telegram
        self._format_signal = format_signal
        self._redis = redis_client
        self._active_signals: Dict[str, Signal] = {}
        self._daily_best: List[Signal] = []  # for free channel
        self._position_lock: Dict[str, Direction] = {}  # symbol → direction
        # (symbol, channel) → UTC timestamp of last signal completion
        self._cooldown_timestamps: Dict[Tuple[str, str], datetime] = {}
        self._running = False
        self._free_limit: int = 2  # max daily free signals
        self._risk_mgr = RiskManager()
        # Free-channel highlight rate limiting
        self._highlight_count_today: int = 0
        self._highlight_date: Optional[date] = None
        # Free-signal daily tracking: keyed by user-facing group ("active")
        self._free_signals_today: Dict[str, bool] = {}
        self._free_signal_date: Optional[date] = None
        # Detect whether queue.get() supports a timeout keyword argument
        self._queue_has_timeout = "timeout" in inspect.signature(queue.get).parameters
        # Signal enrichment (optional — set after construction)
        self.narrative_builder: Optional[Any] = None
        self.sector_comparator: Optional[Any] = None
        # AI Trade Observer (optional — set after construction in main.py)
        self.observer: Optional[Any] = None
        # AI Engine integration (PR: AI Engine Refactor)
        self._ai_predictor: Optional[SignalPredictor] = None
        self._ai_scorer: Optional[AIConfidenceScorer] = None

    # ------------------------------------------------------------------
    # AI Engine wiring
    # ------------------------------------------------------------------

    def set_ai_engine(
        self,
        predictor: Optional[Any] = None,
        scorer: Optional[Any] = None,
    ) -> None:
        """Configure AI engine components for signal enrichment.

        Parameters
        ----------
        predictor:
            :class:`~src.ai_engine.predictor.SignalPredictor` instance.
        scorer:
            :class:`~src.ai_engine.scorer.AIConfidenceScorer` instance.
        """
        self._ai_predictor = predictor
        self._ai_scorer = scorer
        log.info("AI engine configured: predictor={}, scorer={}",
                 predictor is not None, scorer is not None)

    async def _enrich_with_ai(self, signal: Signal) -> Signal:
        """Enrich a signal with AI prediction and confidence scoring.

        When the AI predictor and scorer are configured, this method:
        1. Runs the predictor to get a probability estimate
        2. Feeds the probability into the AI scorer for threshold adjustment
        3. Updates the signal's confidence metadata

        Parameters
        ----------
        signal:
            The signal to enrich.

        Returns
        -------
        Signal
            The signal with updated AI confidence fields.
        """
        if self._ai_predictor is None and self._ai_scorer is None:
            return signal

        updates: Dict[str, Any] = {}

        # Step 1: AI prediction
        if self._ai_predictor is not None:
            try:
                features = PredictionFeatures(
                    price_features={"momentum": 0.0, "ema_alignment": 0.0},
                    volume_features={"obv_trend": 0.0},
                    order_book_features={},
                    correlation_features={},
                )
                prediction = await self._ai_predictor.predict(signal.symbol, features)
                updates["pre_ai_confidence"] = signal.confidence
                log.debug(
                    "AI prediction for {}: dir={} prob={:.3f}",
                    signal.symbol, prediction.direction, prediction.probability,
                )
            except Exception as exc:
                log.debug("AI prediction failed for {}: {}", signal.symbol, exc)

        # Step 2: AI confidence scoring
        if self._ai_scorer is not None:
            try:
                score_result = self._ai_scorer.score_signal(
                    symbol=signal.symbol,
                    base_confidence=signal.confidence,
                    regime=getattr(signal, "entry_regime", ""),
                )
                updates["post_ai_confidence"] = score_result.final_confidence
                if score_result.ai_adjustment != 0.0:
                    updates["confidence"] = score_result.final_confidence
                    log.debug(
                        "AI scorer adjusted {} confidence: {:.1f} → {:.1f} (adj={:+.1f})",
                        signal.symbol, signal.confidence,
                        score_result.final_confidence, score_result.ai_adjustment,
                    )
            except Exception as exc:
                log.debug("AI scoring failed for {}: {}", signal.symbol, exc)

        if updates:
            signal = dataclasses.replace(signal, **updates)

        return signal

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def restore(self) -> None:
        """Reload active state from Redis after a process restart.

        Should be called once before :meth:`start` to resume monitoring of
        any signals that were active when the process last exited.
        """
        if self._redis is None or not self._redis.available:
            return
        try:
            client = self._redis.client
            if client is None:
                return
            # Restore active signals
            raw = await client.get(_REDIS_KEY_SIGNALS)
            if raw:
                signals_data: Dict[str, Any] = json.loads(raw)
                for sid, data in signals_data.items():
                    sig = _signal_from_dict(data)
                    if sig is not None:
                        self._active_signals[sid] = sig
                log.info(
                    "Restored {} active signal(s) from Redis",
                    len(self._active_signals),
                )

            # Restore position lock
            raw = await client.get(_REDIS_KEY_POSITION_LOCK)
            if raw:
                lock_data: Dict[str, str] = json.loads(raw)
                for sym, dir_str in lock_data.items():
                    try:
                        self._position_lock[sym] = Direction(dir_str)
                    except ValueError:
                        log.warning("Unknown direction '{}' for symbol {} – skipped", dir_str, sym)

            # Restore cooldown timestamps
            raw = await client.get(_REDIS_KEY_COOLDOWNS)
            if raw:
                cooldown_data: Dict[str, str] = json.loads(raw)
                for key, ts_str in cooldown_data.items():
                    parts = key.split("|", 1)
                    if len(parts) == 2:
                        sym, chan = parts
                        self._cooldown_timestamps[(sym, chan)] = datetime.fromisoformat(ts_str)
                log.info(
                    "Restored {} cooldown timestamp(s) from Redis",
                    len(self._cooldown_timestamps),
                )
        except Exception as exc:
            log.warning("Failed to restore state from Redis: {}", exc)

    async def _persist_state(self) -> None:
        """Serialize and save active router state to Redis.

        Persists :attr:`_active_signals`, :attr:`_position_lock`, and
        :attr:`_cooldown_timestamps` so that state can be restored after a
        process restart via :meth:`restore`.
        """
        if self._redis is None or not self._redis.available:
            return
        try:
            client = self._redis.client
            if client is None:
                return
            # Persist active signals
            signals_payload = {
                sid: _signal_to_dict(sig)
                for sid, sig in self._active_signals.items()
            }
            await client.set(_REDIS_KEY_SIGNALS, json.dumps(signals_payload))

            # Persist position lock
            lock_payload = {sym: dir_.value for sym, dir_ in self._position_lock.items()}
            await client.set(_REDIS_KEY_POSITION_LOCK, json.dumps(lock_payload))

            # Persist cooldown timestamps (tuple keys → "symbol|channel" strings)
            cooldown_payload = {
                f"{sym}|{chan}": ts.isoformat()
                for (sym, chan), ts in self._cooldown_timestamps.items()
            }
            await client.set(_REDIS_KEY_COOLDOWNS, json.dumps(cooldown_payload))
        except Exception as exc:
            log.warning("Failed to persist state to Redis: {}", exc)

    def _schedule_persist(self) -> None:
        """Fire-and-forget: schedule :meth:`_persist_state` on the running loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._persist_state())
        except RuntimeError:
            pass

    async def start(self) -> None:
        self._running = True
        log.info("Signal router started")
        _cleanup_counter = 0
        while self._running:
            try:
                if self._queue_has_timeout:
                    signal = await self._queue.get(timeout=1.0)
                    if signal is None:
                        continue
                else:
                    signal = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                _cleanup_counter += 1
                if _cleanup_counter >= 60:  # roughly every 60 seconds
                    self.cleanup_expired()
                    _cleanup_counter = 0
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Router error: {}", exc)
                continue

            # Reconstruct Signal from dict (Redis deserialization path)
            if isinstance(signal, dict):
                signal = _signal_from_dict(signal)
                if signal is None:
                    continue

            await self._process(signal)

    async def stop(self) -> None:
        self._running = False
        log.info("Signal router stopped")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    async def _process(self, signal: Signal) -> None:
        # Correlation lock – block any signal for a symbol that already has an
        # open position (regardless of direction to prevent same-dir duplicates)
        existing_dir = self._position_lock.get(signal.symbol)
        if existing_dir is not None:
            log.info(
                "Blocked {} {} – existing {} position open",
                signal.symbol, signal.direction.value, existing_dir.value,
            )
            return

        # Per-symbol + per-channel cooldown check
        cooldown_key = (signal.symbol, signal.channel)
        last_completed = self._cooldown_timestamps.get(cooldown_key)
        if last_completed is not None:
            cooldown_secs = CHANNEL_COOLDOWN_SECONDS.get(signal.channel, 60)
            elapsed = (datetime.now(timezone.utc) - last_completed).total_seconds()
            if elapsed < cooldown_secs:
                log.info(
                    "Cooldown active for {} {} – {:.1f}s remaining ({:.0f}s window)",
                    signal.symbol, signal.channel,
                    cooldown_secs - elapsed, cooldown_secs,
                )
                return

        # Per-channel concurrent position cap
        channel_count = sum(
            1 for s in self._active_signals.values() if s.channel == signal.channel
        )
        channel_max = MAX_CONCURRENT_SIGNALS_PER_CHANNEL.get(signal.channel, 5)
        if channel_count >= channel_max:
            log.info(
                "Per-channel cap reached for {} ({}/{}) – {} {} blocked",
                signal.channel, channel_count, channel_max,
                signal.symbol, signal.direction.value,
            )
            return

        # Correlation-aware position limiting
        active_positions = {
            sid: (s.symbol, s.direction.value)
            for sid, s in self._active_signals.items()
        }
        corr_allowed, corr_reason = check_correlation_limit(
            symbol=signal.symbol,
            direction=signal.direction.value,
            active_positions=active_positions,
        )
        if not corr_allowed:
            log.info(
                "Blocked {} {} – {}",
                signal.symbol, signal.direction.value, corr_reason,
            )
            return

        # TP direction sanity – reject signals where TP1 is on wrong side of entry
        if signal.direction == Direction.LONG and signal.tp1 <= signal.entry:
            log.warning(
                "Signal {} {} LONG has TP1 {:.8f} <= entry {:.8f} – rejected",
                signal.symbol, signal.channel, signal.tp1, signal.entry,
            )
            return
        if signal.direction == Direction.SHORT and signal.tp1 >= signal.entry:
            log.warning(
                "Signal {} {} SHORT has TP1 {:.8f} >= entry {:.8f} – rejected",
                signal.symbol, signal.channel, signal.tp1, signal.entry,
            )
            return

        # SL direction sanity – reject signals where SL is on wrong side of entry
        if signal.direction == Direction.LONG and signal.stop_loss >= signal.entry:
            log.warning(
                "Signal {} {} LONG has SL {:.8f} >= entry {:.8f} – rejected",
                signal.symbol, signal.channel, signal.stop_loss, signal.entry,
            )
            return
        if signal.direction == Direction.SHORT and signal.stop_loss <= signal.entry:
            log.warning(
                "Signal {} {} SHORT has SL {:.8f} <= entry {:.8f} – rejected",
                signal.symbol, signal.channel, signal.stop_loss, signal.entry,
            )
            return

        # ── Stale signal gate ───────────────────────────────────────────────
        # Check whether the signal is still actionable before posting.
        if signal.detected_at is not None:
            now_ts = time.time()
            elapsed_s = now_ts - signal.detected_at

            # Time-based staleness: signal exceeded its validity window.
            is_scalp = signal.channel in _SCALP_CHANNEL_NAMES
            stale_threshold = (
                _SCALP_STALE_THRESHOLD_SECONDS if is_scalp
                else _DEFAULT_STALE_THRESHOLD_SECONDS
            )
            if elapsed_s > stale_threshold:
                log.warning(
                    "STALE signal {} {} {}: detected→now {:.1f}s > {:.0f}s threshold – suppressed",
                    signal.channel, signal.symbol, signal.direction.value,
                    elapsed_s, stale_threshold,
                )
                return

            # Price-based staleness: check against detection-time price (current_price).
            # This catches the case where the price was already past TP1 or SL
            # at the moment the signal was detected (e.g. due to a slow scan cycle).
            if signal.current_price > 0:
                cp = signal.current_price
                if signal.direction == Direction.LONG:
                    if cp > signal.tp1:
                        log.warning(
                            "STALE signal {} {} LONG: detection-time price {:.8f} already past "
                            "TP1 {:.8f} – suppressed",
                            signal.channel, signal.symbol, cp, signal.tp1,
                        )
                        return
                    if cp < signal.stop_loss:
                        log.warning(
                            "STALE signal {} {} LONG: detection-time price {:.8f} already below "
                            "SL {:.8f} – suppressed",
                            signal.channel, signal.symbol, cp, signal.stop_loss,
                        )
                        return
                else:  # SHORT
                    if cp < signal.tp1:
                        log.warning(
                            "STALE signal {} {} SHORT: detection-time price {:.8f} already past "
                            "TP1 {:.8f} – suppressed",
                            signal.channel, signal.symbol, cp, signal.tp1,
                        )
                        return
                    if cp > signal.stop_loss:
                        log.warning(
                            "STALE signal {} {} SHORT: detection-time price {:.8f} already above "
                            "SL {:.8f} – suppressed",
                            signal.channel, signal.symbol, cp, signal.stop_loss,
                        )
                        return

        # ── AI enrichment ───────────────────────────────────────────────
        signal = await self._enrich_with_ai(signal)

        # Channel min-confidence filter
        chan_cfg = next(
            (c for c in ALL_CHANNELS if c.name == signal.channel), None
        )
        if chan_cfg and signal.confidence < chan_cfg.min_confidence:
            log.debug(
                "Signal {} {} confidence {:.1f} < min {:.1f} – skipped",
                signal.channel, signal.symbol,
                signal.confidence, chan_cfg.min_confidence,
            )
            return

        # Risk assessment: use the signal's own volume/spread fields so the risk
        # classifier has accurate data (set by the scanner before enqueuing).
        risk = self._risk_mgr.calculate_risk(
            signal, {}, volume_24h_usd=signal.volume_24h_usd,
            active_signals=self.active_signals,
        )
        if not risk.allowed:
            log.warning(
                "Signal {} {} blocked by risk manager: {}",
                signal.symbol, signal.direction.value, risk.reason,
            )
            return
        signal.risk_label = risk.risk_label

        # Format and send to premium channel
        channel_id = CHANNEL_TELEGRAM_MAP.get(signal.channel, "")
        if not channel_id:
            log.warning("No Telegram channel configured for {}", signal.channel)
            return

        text = self._format_signal(signal)

        # Append Cornix auto-execution block when enabled
        try:
            from config import CORNIX_FORMAT_ENABLED
            if CORNIX_FORMAT_ENABLED:
                cornix_block = format_cornix_signal(signal)
                if cornix_block:
                    text = text + "\n\n" + cornix_block
        except Exception as _exc:
            log.debug("Cornix format skipped: {}", _exc)

        delivered = False
        try:
            delivered = await self._send_telegram(channel_id, text)
        except Exception as exc:
            log.warning(
                "Signal delivery failed for {} {}: {}",
                signal.channel,
                signal.signal_id,
                exc,
            )
        if not delivered:
            retries = signal._delivery_retries
            if retries < 2:
                signal._delivery_retries = retries + 1
                log.info(
                    "Re-queuing {} {} (delivery attempt {}/3)",
                    signal.channel,
                    signal.signal_id,
                    retries + 2,
                )
                await _delivery_sleep(2 ** retries)  # 1 s, 2 s for retries 0, 1
                await self._queue.put(signal)
            else:
                log.error(
                    "Signal {} {} permanently lost after 3 delivery attempts",
                    signal.channel,
                    signal.signal_id,
                )
            return
        log.info(
            "Signal posted → {} | {} {}",
            signal.channel,
            signal.symbol,
            signal.direction.value,
        )

        # ── Latency tracking ─────────────────────────────────────────────────
        signal.posted_at = time.time()
        if signal.detected_at is not None:
            latency_ms = (signal.posted_at - signal.detected_at) * 1000.0
            signal.enrichment_latency_ms = latency_ms
            log.info(
                "{} {} signal: detected→posted latency = {:,.0f}ms",
                signal.symbol, signal.channel, latency_ms,
            )
            if signal.channel in _SCALP_CHANNEL_NAMES and latency_ms > _SCALP_LATENCY_WARNING_SECONDS * 1000:
                log.warning(
                    "HIGH LATENCY {} {} SCALP signal: {:.1f}s detected→posted (threshold={:.0f}s)",
                    signal.symbol, signal.channel,
                    latency_ms / 1000.0, _SCALP_LATENCY_WARNING_SECONDS,
                )

        # Register only after confirmed delivery
        self._active_signals[signal.signal_id] = signal
        self._position_lock[signal.symbol] = signal.direction
        self._schedule_persist()

        # Track for daily free-channel picks
        self._daily_best.append(signal)
        self._daily_best.sort(key=lambda s: s.confidence, reverse=True)
        self._trim_daily_best()

        # Publish a condensed version to the free channel (Phase 4)
        await self._maybe_publish_free_signal(signal)

        # Notify AI Trade Observer — capture market state at signal publish time
        if self.observer is not None:
            try:
                self.observer.capture_entry_snapshot(signal)
            except Exception as exc:
                log.debug("TradeObserver.capture_entry_snapshot failed (non-critical): {}", exc)

    # ------------------------------------------------------------------
    # Portfolio signal enrichment helpers
    # ------------------------------------------------------------------

    def _build_narrative_context(self, signal: Signal) -> Dict[str, Any]:
        """Build the context dict for NarrativeBuilder from a signal."""
        context: Dict[str, Any] = {}
        # Basic fields extractable from the signal itself
        if signal.setup_class:
            context["setup_class"] = signal.setup_class
        if signal.liquidity_info:
            context["smc_events"] = [signal.liquidity_info]
        if signal.risk_label:
            context["risk"] = signal.risk_label
        # Sector can be populated if sector_comparator is available
        if self.sector_comparator is not None:
            try:
                context["sector"] = self.sector_comparator.get_sector(signal.symbol)
            except Exception:
                pass
        # AI sentiment fields (populated for SPOT/GEM channels)
        if signal.ai_sentiment_label:
            context["sentiment_label"] = signal.ai_sentiment_label
        if signal.ai_sentiment_summary:
            context["sentiment_summary"] = signal.ai_sentiment_summary
        return context

    async def _send_photo(self, channel_id: str, photo_bytes: bytes) -> bool:
        """Send a chart image to *channel_id*.

        Uses the TelegramBot instance if available via _send_telegram, otherwise
        calls send_photo directly on a TelegramBot instance.
        """
        try:
            from src.telegram_bot import TelegramBot
            # Retrieve the bot instance bound to _send_telegram if possible
            bot = getattr(self._send_telegram, "__self__", None)
            if isinstance(bot, TelegramBot):
                return await bot.send_photo(channel_id, photo_bytes)
            # Fall back to creating a transient bot (token taken from env)
            tmp_bot = TelegramBot()
            return await tmp_bot.send_photo(channel_id, photo_bytes)
        except Exception as exc:
            log.warning("_send_photo failed: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Free-channel publication (call once/day or on demand)
    # ------------------------------------------------------------------

    def _trim_daily_best(self) -> None:
        """Trim ``_daily_best`` to the current free-signal limit."""
        self._daily_best = self._daily_best[:self._free_limit]

    def set_free_limit(self, limit: int) -> None:
        """Update the maximum number of daily free signals."""
        self._free_limit = max(0, limit)
        self._trim_daily_best()

    async def publish_free_signals(self) -> None:
        """Post the top free signals of the day to the free channel.

        .. deprecated::
            Use :meth:`publish_daily_recap` instead.  This method is kept for
            backward compatibility (tests reference it).
        """
        if not self._daily_best or not TELEGRAM_FREE_CHANNEL_ID:
            return
        for sig in self._daily_best:
            text = self._format_signal(sig)
            header = "🆓 *FREE SIGNAL OF THE DAY* 🆓\n\n"
            footer = (
                "\n\n📚 _Tip: Scalping requires discipline. "
                "Always use a stop-loss and manage risk._"
            )
            await self._send_telegram(TELEGRAM_FREE_CHANNEL_ID, header + text + footer)
        self._daily_best.clear()

    async def publish_highlight(self, sig: Signal, tp_level: int, tp_pnl_pct: float) -> None:
        """Post a winning trade highlight to the free channel.

        Called by the trade monitor when a signal hits TP2 or higher.
        Rate-limited to ``_FREE_HIGHLIGHT_MAX_PER_DAY`` highlights per day.
        """
        if not TELEGRAM_FREE_CHANNEL_ID:
            return
        if tp_level < _FREE_HIGHLIGHT_MIN_TP:
            return

        # Daily rate limit
        today = date.today()
        if self._highlight_date != today:
            self._highlight_date = today
            self._highlight_count_today = 0
        if self._highlight_count_today >= _FREE_HIGHLIGHT_MAX_PER_DAY:
            log.debug(
                "Free highlight daily limit reached ({}/{})",
                self._highlight_count_today,
                _FREE_HIGHLIGHT_MAX_PER_DAY,
            )
            return

        text = self._format_highlight(sig, tp_level, tp_pnl_pct)
        try:
            await self._send_telegram(TELEGRAM_FREE_CHANNEL_ID, text)
            self._highlight_count_today += 1
            log.info(
                "Posted free highlight: {} {} TP{} +{:.2f}%",
                sig.symbol, sig.direction.value, tp_level, tp_pnl_pct,
            )
        except Exception as exc:
            log.warning("Failed to post free highlight: {}", exc)

    def _format_highlight(self, sig: Signal, tp_level: int, tp_pnl_pct: float) -> str:
        """Delegate highlight formatting to TelegramBot."""
        from src.telegram_bot import TelegramBot
        return TelegramBot.format_highlight_message(sig, tp_level, tp_pnl_pct)

    async def publish_daily_recap(self, performance_tracker: Any) -> None:
        """Post the daily performance recap to the free channel."""
        if not TELEGRAM_FREE_CHANNEL_ID:
            return

        summary = performance_tracker.get_daily_summary(window_days=1)
        if summary["total"] == 0:
            return  # No trades today, skip

        text = self._format_daily_recap(summary)
        try:
            await self._send_telegram(TELEGRAM_FREE_CHANNEL_ID, text)
            log.info("Posted daily recap to free channel")
        except Exception as exc:
            log.warning("Failed to post daily recap: {}", exc)

    def _format_daily_recap(self, summary: Any) -> str:
        """Delegate daily recap formatting to TelegramBot."""
        from src.telegram_bot import TelegramBot
        return TelegramBot.format_daily_recap(summary)

    # ------------------------------------------------------------------
    # Free channel – condensed signal (Phase 4)
    # ------------------------------------------------------------------

    @staticmethod
    def _free_channel_group(channel: str) -> str:
        """Map a signal channel to a user-facing group name for free-signal tracking."""
        return "active"

    async def _maybe_publish_free_signal(self, signal: Signal) -> None:
        """Publish a condensed version of the signal to the free channel.

        Only posts once per calendar day, and only when confidence >= 75.
        """
        if not TELEGRAM_FREE_CHANNEL_ID:
            return

        # Reset tracking on a new day
        today = date.today()
        if self._free_signal_date != today:
            self._free_signal_date = today
            self._free_signals_today = {}

        group = self._free_channel_group(signal.channel)
        if self._free_signals_today.get(group):
            return  # Already posted for this group today
        if signal.confidence < 75:
            return  # Only show high-confidence signals for free

        text = self._format_condensed_free(signal)
        try:
            await self._send_telegram(TELEGRAM_FREE_CHANNEL_ID, text)
            self._free_signals_today[group] = True
            log.info(
                "Posted free condensed signal ({} group): {} {}",
                group, signal.symbol, signal.direction.value,
            )
        except Exception as exc:
            log.warning("Failed to post condensed free signal: {}", exc)

    def _format_condensed_free(self, signal: Signal) -> str:
        """Format a condensed free-channel version of a signal (Entry/SL/TP1 only)."""
        from src.telegram_bot import TelegramBot
        from src.utils import fmt_price

        chan_emojis = {
            "360_SCALP": "⚡",
            "360_SCALP_FVG": "⚡",
            "360_SCALP_CVD": "⚡",
            "360_SCALP_VWAP": "⚡",
            "360_SCALP_OBI": "⚡",
            "360_SWING": "🏛️",
            "360_SPOT": "📈",
            "360_GEM": "💎",
        }
        emoji = chan_emojis.get(signal.channel, "📡")
        chan_name = TelegramBot._CHANNEL_DISPLAY_NAME.get(signal.channel, signal.channel)
        dir_word = signal.direction.value

        def _pct(price: float) -> str:
            if signal.entry and signal.entry != 0:
                pct = (price - signal.entry) / signal.entry * 100
                return f"{pct:+.2f}%"
            return ""

        lines = [
            "🆓 *FREE SIGNAL PREVIEW* 🆓",
            "",
            f"{emoji} *{TelegramBot._escape_md(chan_name)}* │ *{TelegramBot._escape_md(signal.symbol)}* │ *{dir_word}*",
            TelegramBot._escape_md("━" * 24),
            "",
            f"📍 Entry: `{fmt_price(signal.entry)}`",
            f"🛑 SL: `{fmt_price(signal.stop_loss)}` ({TelegramBot._escape_md(_pct(signal.stop_loss))})",
            f"🎯 TP1: `{fmt_price(signal.tp1)}` ({TelegramBot._escape_md(_pct(signal.tp1))})",
            "",
            "🔒 _Premium members see TP2, TP3 and full analysis_",
            "📲 _Join our premium channels for real-time signals_",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Scoreboard (Phase 3)
    # ------------------------------------------------------------------

    async def publish_scoreboard(self, performance_tracker: Any) -> None:
        """Post the weekly win-rate scoreboard to the free channel."""
        if not TELEGRAM_FREE_CHANNEL_ID:
            return

        scoreboard = performance_tracker.get_channel_scoreboard(window_days=7)
        if not scoreboard:
            return

        text = self._format_scoreboard(scoreboard)
        try:
            await self._send_telegram(TELEGRAM_FREE_CHANNEL_ID, text)
            log.info("Posted weekly scoreboard to free channel")
        except Exception as exc:
            log.warning("Failed to post scoreboard: {}", exc)

    @staticmethod
    def _format_scoreboard(scoreboard: Dict[str, Any]) -> str:
        """Format the weekly scoreboard for Telegram."""
        chan_emojis = {
            "360_SCALP": "⚡",
            "360_SCALP_FVG": "⚡",
            "360_SCALP_CVD": "⚡",
            "360_SCALP_VWAP": "⚡",
            "360_SCALP_OBI": "⚡",
            "360_SWING": "🏛️",
            "360_SPOT": "📈",
            "360_GEM": "💎",
        }
        chan_labels = {
            "360_SCALP": "Scalp",
            "360_SCALP_FVG": "Scalp FVG",
            "360_SCALP_CVD": "Scalp CVD",
            "360_SCALP_VWAP": "Scalp VWAP",
            "360_SCALP_OBI": "Scalp OBI",
            "360_SWING": "Swing",
            "360_SPOT": "Spot",
            "360_GEM": "Gem",
        }
        separator = "━" * 30
        lines = [
            "📊 *360 Crypto — Weekly Performance*",
            separator,
            "",
        ]

        total_wins = 0
        total_losses = 0

        for channel in [
            "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD",
            "360_SCALP_VWAP", "360_SCALP_OBI", "360_SWING",
            "360_SPOT", "360_GEM",
        ]:
            data = scoreboard.get(channel)
            if not data:
                continue
            emoji = chan_emojis.get(channel, "📡")
            label = chan_labels.get(channel, channel)
            wins = data["wins"]
            losses = data["losses"]
            win_rate = data["win_rate"]
            avg_pnl = data["avg_pnl"]
            total_wins += wins
            total_losses += losses
            wr_str = f"({win_rate:.0f}%)"
            lines.append(
                f"{emoji} {label}:  {wins}W / {losses}L  {wr_str}  Avg {avg_pnl:+.1f}%"
            )

        # Grand total
        grand_total = total_wins + total_losses
        grand_wr = round(total_wins / grand_total * 100, 1) if grand_total > 0 else 0.0
        lines.extend([
            separator,
            f"Total: {total_wins}W / {total_losses}L ({grand_wr:.1f}%)",
            "",
            "📈 _Join our premium channels for real-time signals._",
            "⏰ _Updated every Sunday._",
        ])

        return "\n".join(lines)

    @property
    def active_signals(self) -> Dict[str, Signal]:
        return dict(self._active_signals)

    def remove_signal(self, signal_id: str) -> None:
        sig = self._active_signals.pop(signal_id, None)
        if sig:
            self._position_lock.pop(sig.symbol, None)
            # Record cooldown timestamp so we suppress rapid re-entry
            self._cooldown_timestamps[(sig.symbol, sig.channel)] = datetime.now(timezone.utc)
            self._schedule_persist()

    def update_signal(self, signal_id: str, **kwargs) -> None:
        sig = self._active_signals.get(signal_id)
        if sig:
            for k, v in kwargs.items():
                if hasattr(sig, k):
                    setattr(sig, k, v)
            self._schedule_persist()

    def cleanup_expired(self) -> int:
        """Remove signals that have exceeded their max hold duration.

        This provides a safety net to ensure :attr:`_position_lock` entries
        are always cleaned up even when the :class:`~src.trade_monitor.TradeMonitor`
        callback is not triggered (e.g. after a process restart where Redis
        state was restored but the signal is already past its TTL).

        Returns the number of signals that were expired and removed.
        """
        now = datetime.now(timezone.utc)
        expired_ids = []
        for signal_id, sig in list(self._active_signals.items()):
            max_hold = MAX_SIGNAL_HOLD_SECONDS.get(sig.channel, 86400)
            age_secs = (now - sig.timestamp).total_seconds()
            if age_secs >= max_hold:
                expired_ids.append(signal_id)

        for signal_id in expired_ids:
            sig = self._active_signals.pop(signal_id)
            self._position_lock.pop(sig.symbol, None)
            # Record cooldown timestamp so rapid re-entry is suppressed
            self._cooldown_timestamps[(sig.symbol, sig.channel)] = now
            log.info(
                "Auto-expired signal {} {} {} (exceeded max hold)",
                signal_id, sig.symbol, sig.channel,
            )

        if expired_ids:
            self._schedule_persist()

        return len(expired_ids)
