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
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from config import (
    ALL_CHANNELS,
    CHANNEL_COOLDOWN_SECONDS,
    CHANNEL_TELEGRAM_MAP,
    MAX_CONCURRENT_SIGNALS,
    TELEGRAM_FREE_CHANNEL_ID,
)
from src.channels.base import Signal
from src.correlation import check_correlation_limit
from src.redis_client import RedisClient
from src.risk import RiskManager
from src.smc import Direction
from src.utils import get_logger

log = get_logger("signal_router")


def _signal_from_dict(data: dict) -> Optional[Signal]:
    """Reconstruct a Signal from a Redis-deserialized dict."""
    try:
        d = data.copy()
        if isinstance(d.get("direction"), str):
            d["direction"] = Direction(d["direction"])
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return Signal(**d)
    except Exception as exc:
        log.warning("Failed to reconstruct Signal from dict: {}", exc)
        return None


def _signal_to_dict(sig: Signal) -> dict:
    """Serialize a Signal to a JSON-serializable dict."""
    d = dataclasses.asdict(sig)
    d["direction"] = sig.direction.value  # Direction enum → string
    d["timestamp"] = sig.timestamp.isoformat()  # datetime → ISO string
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
        # Detect whether queue.get() supports a timeout keyword argument
        self._queue_has_timeout = "timeout" in inspect.signature(queue.get).parameters

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
        while self._running:
            try:
                if self._queue_has_timeout:
                    signal = await self._queue.get(timeout=1.0)
                    if signal is None:
                        continue
                else:
                    signal = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
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

        # Global concurrent position cap
        if len(self._active_signals) >= MAX_CONCURRENT_SIGNALS:
            log.info(
                "Global position cap reached ({}/{}) – {} {} blocked",
                len(self._active_signals), MAX_CONCURRENT_SIGNALS,
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
            signal, {}, volume_24h_usd=signal.volume_24h_usd, active_signals=self.active_signals
        )
        if not risk.allowed:
            log.warning(
                "Signal {} {} blocked by risk manager: {}",
                signal.symbol, signal.direction.value, risk.reason,
            )
            return
        signal.risk_label = risk.risk_label

        # Format and send to premium channel
        text = self._format_signal(signal)
        channel_id = CHANNEL_TELEGRAM_MAP.get(signal.channel, "")
        if channel_id:
            try:
                delivered = await self._send_telegram(channel_id, text)
            except Exception as exc:
                log.warning(
                    "Signal delivery failed for {} {}: {}",
                    signal.channel,
                    signal.signal_id,
                    exc,
                )
                return
            if delivered is False:
                log.warning(
                    "Signal delivery was not confirmed for {} {}",
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
        else:
            log.warning("No Telegram channel configured for {}", signal.channel)
            return

        # Register only after confirmed delivery
        self._active_signals[signal.signal_id] = signal
        self._position_lock[signal.symbol] = signal.direction
        self._schedule_persist()

        # Track for daily free-channel picks
        self._daily_best.append(signal)
        self._daily_best.sort(key=lambda s: s.confidence, reverse=True)
        self._trim_daily_best()

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
        """Post the top free signals of the day to the free channel."""
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

    # ------------------------------------------------------------------
    # Active-signal helpers
    # ------------------------------------------------------------------

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
