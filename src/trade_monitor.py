"""Trade monitor – continuously checks active signals for TP/SL/trailing updates.

Runs as an async loop, polling the latest price for each active signal and
updating status, PnL, trailing stop, and posting updates to Telegram.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Dict, Optional

from config import CHANNEL_TELEGRAM_MAP, MIN_SIGNAL_LIFESPAN_SECONDS, MONITOR_POLL_INTERVAL
from src.channels.base import Signal
from src.historical_data import HistoricalDataStore
from src.performance_metrics import calculate_trade_pnl_pct
from src.smc import Direction
from src.utils import fmt_price, fmt_ts, get_logger, utcnow

log = get_logger("trade_monitor")

# Minimum absolute PnL (%) before SL/TP evaluation is allowed.
# Prevents false stops from stale prices or floating-point noise.
_ZERO_PNL_THRESHOLD_PCT = 0.01


class TradeMonitor:
    """Watches active signals and emits updates."""

    def __init__(
        self,
        data_store: HistoricalDataStore,
        send_telegram: Callable[[str, str], Coroutine],
        get_active_signals: Callable[[], Dict[str, Signal]],
        remove_signal: Callable[[str], None],
        update_signal: Callable[[str], None],
        performance_tracker: Optional[Any] = None,
        circuit_breaker: Optional[Any] = None,
    ) -> None:
        self._store = data_store
        self._send = send_telegram
        self._get_signals = get_active_signals
        self._remove = remove_signal
        self._update = update_signal
        self._performance_tracker = performance_tracker
        self._circuit_breaker = circuit_breaker
        self._running = False

    def _record_outcome(self, sig: Signal, hit_tp: int, hit_sl: bool) -> None:
        """Notify performance tracker and circuit breaker of a completed signal.

        Called only on final outcomes (SL_HIT or TP3_HIT).  Intermediate hits
        (TP1/TP2) and configuration-error cancellations are intentionally
        excluded because the signal is still active or was never a real trade.

        Parameters
        ----------
        sig:
            The completed :class:`src.channels.base.Signal`.
        hit_tp:
            Which TP was hit (0 if SL was hit, 3 if TP3 was hit).
        hit_sl:
            ``True`` when the stop-loss was triggered.
        """
        if self._performance_tracker is not None:
            hold_duration_sec = max((utcnow() - sig.timestamp).total_seconds(), 0.0)
            self._performance_tracker.record_outcome(
                signal_id=sig.signal_id,
                channel=sig.channel,
                symbol=sig.symbol,
                direction=sig.direction.value,
                entry=sig.entry,
                hit_tp=hit_tp,
                hit_sl=hit_sl,
                pnl_pct=sig.pnl_pct,
                confidence=sig.confidence,
                pre_ai_confidence=sig.pre_ai_confidence,
                post_ai_confidence=sig.post_ai_confidence,
                setup_class=sig.setup_class,
                market_phase=sig.market_phase,
                quality_tier=sig.quality_tier,
                spread_pct=sig.spread_pct,
                volume_24h_usd=sig.volume_24h_usd,
                hold_duration_sec=hold_duration_sec,
                max_favorable_excursion_pct=sig.max_favorable_excursion_pct,
                max_adverse_excursion_pct=sig.max_adverse_excursion_pct,
            )
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_outcome(
                signal_id=sig.signal_id,
                hit_sl=hit_sl,
                pnl_pct=sig.pnl_pct,
            )

    @staticmethod
    def _set_realized_pnl(sig: Signal, exit_price: float) -> None:
        """Freeze final trade PnL at the executed exit level."""
        sig.current_price = exit_price
        sig.pnl_pct = calculate_trade_pnl_pct(
            entry_price=sig.entry,
            exit_price=exit_price,
            direction=sig.direction.value,
        )

    async def start(self) -> None:
        self._running = True
        log.info("Trade monitor started")
        while self._running:
            try:
                await self._check_all()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Monitor error: %s", exc)
            await asyncio.sleep(MONITOR_POLL_INTERVAL)

    async def stop(self) -> None:
        self._running = False
        log.info("Trade monitor stopped")

    async def _check_all(self) -> None:
        signals = self._get_signals()
        for sid, sig in list(signals.items()):
            price = self._latest_price(sig.symbol)
            if price is None:
                continue
            sig.current_price = price
            await self._evaluate_signal(sig)

    def _latest_price(self, symbol: str) -> Optional[float]:
        # Prefer real-time tick data over (potentially stale) candle close
        ticks = self._store.ticks.get(symbol)
        if ticks:
            tick_price = ticks[-1].get("price")
            if tick_price is not None:
                return float(tick_price)
        # Fallback to last closed 1m candle
        candles = self._store.get_candles(symbol, "1m")
        if candles and len(candles.get("close", [])) > 0:
            return float(candles["close"][-1])
        return None

    async def _evaluate_signal(self, sig: Signal) -> None:
        price = sig.current_price
        is_long = sig.direction == Direction.LONG

        # Minimum lifespan guard – don't trigger SL/TP checks on very new
        # signals to protect against noise-driven instant stops
        min_lifespan = MIN_SIGNAL_LIFESPAN_SECONDS.get(sig.channel, 10)
        age_secs = (utcnow() - sig.timestamp).total_seconds()
        if age_secs < min_lifespan:
            log.debug(
                "Signal %s %s too new (%.1fs < %ds min lifespan) – skipping SL/TP eval",
                sig.symbol, sig.channel, age_secs, min_lifespan,
            )
            return

        # SL direction sanity check – catch misconfigured signals
        if is_long and sig.stop_loss > sig.entry:
            log.warning(
                "Signal %s %s has invalid SL (LONG SL %.8f > entry %.8f) – cancelling",
                sig.symbol, sig.signal_id, sig.stop_loss, sig.entry,
            )
            sig.status = "CANCELLED"
            await self._post_update(sig, "⚠️ CANCELLED (invalid SL)")
            self._remove(sig.signal_id)
            return
        if not is_long and sig.stop_loss < sig.entry:
            log.warning(
                "Signal %s %s has invalid SL (SHORT SL %.8f < entry %.8f) – cancelling",
                sig.symbol, sig.signal_id, sig.stop_loss, sig.entry,
            )
            sig.status = "CANCELLED"
            await self._post_update(sig, "⚠️ CANCELLED (invalid SL)")
            self._remove(sig.signal_id)
            return

        # PnL
        if sig.entry != 0:
            sig.pnl_pct = calculate_trade_pnl_pct(
                entry_price=sig.entry,
                exit_price=price,
                direction=sig.direction.value,
            )
        sig.max_favorable_excursion_pct = max(sig.max_favorable_excursion_pct, sig.pnl_pct)
        sig.max_adverse_excursion_pct = min(sig.max_adverse_excursion_pct, sig.pnl_pct)

        # Zero-PnL guard – don't trigger SL when price hasn't moved from entry
        # This prevents false stops from stale prices or floating-point noise
        if abs(sig.pnl_pct) < _ZERO_PNL_THRESHOLD_PCT:
            log.debug(
                "Signal %s %s PnL near zero (%.4f%%) – skipping SL/TP eval",
                sig.symbol, sig.signal_id, sig.pnl_pct,
            )
            return

        # Stop-loss hit
        if is_long and price <= sig.stop_loss:
            self._set_realized_pnl(sig, sig.stop_loss)
            sig.status = "SL_HIT"
            await self._post_update(sig, "🔴 SL HIT")
            self._record_outcome(sig, hit_tp=0, hit_sl=True)
            self._remove(sig.signal_id)
            return
        if not is_long and price >= sig.stop_loss:
            self._set_realized_pnl(sig, sig.stop_loss)
            sig.status = "SL_HIT"
            await self._post_update(sig, "🔴 SL HIT")
            self._record_outcome(sig, hit_tp=0, hit_sl=True)
            self._remove(sig.signal_id)
            return

        # TP hits (progressive)
        if is_long:
            if sig.tp3 and price >= sig.tp3 and sig.status != "TP3_HIT":
                self._set_realized_pnl(sig, sig.tp3)
                sig.status = "TP3_HIT"
                await self._post_update(sig, "🎯🎯🎯 TP3 HIT")
                self._record_outcome(sig, hit_tp=3, hit_sl=False)
                self._remove(sig.signal_id)
                return
            if price >= sig.tp2 and sig.status not in ("TP2_HIT", "TP3_HIT"):
                sig.status = "TP2_HIT"
                await self._post_update(sig, "🎯🎯 TP2 HIT")
                # Trailing: move SL to entry (break-even)
                sig.stop_loss = sig.entry
            if price >= sig.tp1 and sig.status not in ("TP1_HIT", "TP2_HIT", "TP3_HIT"):
                sig.status = "TP1_HIT"
                await self._post_update(sig, "🎯 TP1 HIT ✅")
        else:
            if sig.tp3 and price <= sig.tp3 and sig.status != "TP3_HIT":
                self._set_realized_pnl(sig, sig.tp3)
                sig.status = "TP3_HIT"
                await self._post_update(sig, "🎯🎯🎯 TP3 HIT")
                self._record_outcome(sig, hit_tp=3, hit_sl=False)
                self._remove(sig.signal_id)
                return
            if price <= sig.tp2 and sig.status not in ("TP2_HIT", "TP3_HIT"):
                sig.status = "TP2_HIT"
                await self._post_update(sig, "🎯🎯 TP2 HIT")
                sig.stop_loss = sig.entry
            if price <= sig.tp1 and sig.status not in ("TP1_HIT", "TP2_HIT", "TP3_HIT"):
                sig.status = "TP1_HIT"
                await self._post_update(sig, "🎯 TP1 HIT ✅")

        # Trailing stop adjustment
        if sig.trailing_active and sig.status in ("TP1_HIT", "TP2_HIT"):
            self._adjust_trailing(sig)

    def _adjust_trailing(self, sig: Signal) -> None:
        """Move the trailing stop behind the price."""
        price = sig.current_price
        trail_dist = abs(sig.entry - sig.stop_loss) * 0.5  # tighten on TP hits
        if sig.direction == Direction.LONG:
            new_sl = price - trail_dist
            if new_sl > sig.stop_loss:
                sig.stop_loss = round(new_sl, 8)
        else:
            new_sl = price + trail_dist
            if new_sl < sig.stop_loss:
                sig.stop_loss = round(new_sl, 8)

    async def _post_update(self, sig: Signal, event: str) -> None:
        channel_id = CHANNEL_TELEGRAM_MAP.get(sig.channel, "")
        if not channel_id:
            return

        chan_emojis = {
            "360_SCALP": "⚡",
            "360_SWING": "🏛️",
            "360_RANGE": "⚖️",
            "360_THE_TAPE": "🐋",
            "360_SELECT": "🌹",
        }
        chan_emoji = chan_emojis.get(sig.channel, "📡")
        dir_emoji = "🚀" if sig.direction == Direction.LONG else "⬇️"

        lines = [
            f"{event}",
            f"{chan_emoji} *{sig.channel}* | {sig.symbol} *{sig.direction.value}* {dir_emoji}",
            f"💰 Entry: `{fmt_price(sig.entry)}` → Current: `{fmt_price(sig.current_price)}`",
            f"📊 PnL: *{sig.pnl_pct:+.2f}%*",
            f"🛡️ SL: `{fmt_price(sig.stop_loss)}`",
            f"🤖 Confidence: *{sig.confidence:.0f}%*",
        ]
        if sig.trailing_active and sig.trailing_desc:
            lines.append(f"💹 Trailing Active ({sig.trailing_desc})")
        lines.append(f"⏰ {fmt_ts()}")

        text = "\n".join(lines)
        await self._send(channel_id, text)
