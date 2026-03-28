"""Trade monitor – continuously checks active signals for TP/SL/trailing updates.

Runs as an async loop, polling the latest price for each active signal and
updating status, PnL, trailing stop, and posting updates to Telegram.
"""

from __future__ import annotations

import asyncio
import numpy as np
from typing import Any, Callable, Coroutine, Dict, Optional

from config import (
    ALL_CHANNELS,
    CHANNEL_TELEGRAM_MAP,
    INVALIDATION_CONSECUTIVE_THRESHOLD,
    INVALIDATION_MIN_AGE_SECONDS,
    INVALIDATION_MOMENTUM_THRESHOLD,
    MAX_SIGNAL_HOLD_SECONDS,
    MIN_SIGNAL_LIFESPAN_SECONDS,
    MONITOR_POLL_INTERVAL,
    TRAILING_ATR_MULTIPLIER,
)
from src.channels.base import Signal, TrailingStopState
from src.dca import check_dca_entry, recalculate_after_dca
from src.historical_data import HistoricalDataStore
from src.indicators import atr as _compute_atr
from src.indicators import ema as _compute_ema
from src.indicators import momentum as _compute_momentum
from src.performance_metrics import calculate_trade_pnl_pct, classify_trade_outcome
from src.smc import Direction
from src.stat_filter import SignalOutcome
from src.utils import fmt_price, fmt_ts, get_logger, utcnow

log = get_logger("trade_monitor")

# Minimum absolute PnL (%) before SL/TP evaluation is allowed.
# Prevents false stops from stale prices or floating-point noise.
_ZERO_PNL_THRESHOLD_PCT = 0.01
_STOP_OUTCOME_MESSAGES = {
    "SL_HIT": "🔴 SL HIT",
    "BREAKEVEN_EXIT": "⚪ BREAKEVEN EXIT",
    "PROFIT_LOCKED": "🟢 PROFIT LOCKED",
    "EXPIRED": "⏰ EXPIRED",
}
# Seconds of grace after a DCA entry before invalidation checks are allowed.
# Gives the averaged position time to develop without being killed prematurely.
_DCA_GRACE_SECONDS = 600


def _compute_trailing_stop(
    signal: Signal,
    current_price: float,
    current_atr: float,
    trailing_state: TrailingStopState,
    atr_percentile: float = 50.0,
) -> float:
    """Compute the new trailing stop level based on current stage and ATR.

    Parameters
    ----------
    signal:
        Active signal with direction, entry, current stop_loss.
    current_price:
        Latest market price.
    current_atr:
        ATR computed from the most recent candles (updated each lifecycle poll).
    trailing_state:
        Mutable state tracking the trailing stop stage.
    atr_percentile:
        Rolling ATR percentile 0–100 (from RegimeContext).

    Returns
    -------
    float
        New stop-loss level. Will only ratchet tighter (never widen) for the
        direction of the trade.
    """
    # Update the trailing state with current ATR
    trailing_state.current_atr = current_atr

    # ATR-percentile adjustment: wider buffer in high-vol, tighter in low-vol
    if atr_percentile >= 80:
        vol_adj = 1.3
    elif atr_percentile <= 20:
        vol_adj = 0.7
    else:
        vol_adj = 1.0

    trail_dist = trailing_state.trail_distance * vol_adj

    if signal.direction == Direction.LONG:
        candidate_sl = current_price - trail_dist
        # Never move SL backwards (lower) for a long trade
        new_sl = max(signal.stop_loss, candidate_sl)
    else:
        candidate_sl = current_price + trail_dist
        # Never move SL backwards (higher) for a short trade
        new_sl = min(signal.stop_loss, candidate_sl)

    return round(new_sl, 8)


def _update_trailing_stage(
    signal: Signal,
    current_price: float,
    trailing_state: TrailingStopState,
) -> None:
    """Check if TP levels have been hit and advance trailing stage.

    Mutates both signal and trailing_state in place.
    """
    if trailing_state.stage >= 2:
        return  # Already at final stage

    if trailing_state.stage == 0:
        # Check for TP1 hit
        if signal.direction == Direction.LONG and current_price >= signal.tp1:
            trailing_state.stage = 1
            trailing_state.breakeven_set = True
            signal.trailing_stage = 1
            signal.partial_close_pct = 0.4
            signal.best_tp_hit = max(signal.best_tp_hit, 1)
            signal.execution_note += " | TP1 hit → 40% closed, SL→breakeven"
            # Move SL to breakeven (entry price)
            signal.stop_loss = signal.entry
        elif signal.direction == Direction.SHORT and current_price <= signal.tp1:
            trailing_state.stage = 1
            trailing_state.breakeven_set = True
            signal.trailing_stage = 1
            signal.partial_close_pct = 0.4
            signal.best_tp_hit = max(signal.best_tp_hit, 1)
            signal.execution_note += " | TP1 hit → 40% closed, SL→breakeven"
            signal.stop_loss = signal.entry

    if trailing_state.stage == 1:
        # Check for TP2 hit
        if signal.direction == Direction.LONG and current_price >= signal.tp2:
            trailing_state.stage = 2
            trailing_state.tight_trail_active = True
            signal.trailing_stage = 2
            signal.partial_close_pct = 0.7  # Cumulative: 40% at TP1 + 30% at TP2
            signal.best_tp_hit = max(signal.best_tp_hit, 2)
            signal.execution_note += " | TP2 hit → 70% closed, tight 0.5×ATR trail"
        elif signal.direction == Direction.SHORT and current_price <= signal.tp2:
            trailing_state.stage = 2
            trailing_state.tight_trail_active = True
            signal.trailing_stage = 2
            signal.partial_close_pct = 0.7
            signal.best_tp_hit = max(signal.best_tp_hit, 2)
            signal.execution_note += " | TP2 hit → 70% closed, tight 0.5×ATR trail"


def _escape_md(text: str) -> str:
    """Escape Telegram MarkdownV1 special characters in dynamic text fields."""
    for ch in ("\\", "*", "_", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


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
        regime_detector: Optional[Any] = None,
        indicators_fn: Optional[Callable] = None,
        order_manager: Optional[Any] = None,
        stat_filter: Optional[Any] = None,
    ) -> None:
        self._store = data_store
        self._send = send_telegram
        self._get_signals = get_active_signals
        self._remove = remove_signal
        self._update = update_signal
        self._performance_tracker = performance_tracker
        self._circuit_breaker = circuit_breaker
        self._regime_detector = regime_detector
        self._indicators_fn = indicators_fn
        # Optional OrderManager for direct exchange execution (V3 groundwork).
        # When provided and auto-execution is enabled, confirmed signals are
        # forwarded to the exchange instead of (or alongside) Telegram.
        self._order_manager = order_manager
        # Track signal IDs for which an order has already been placed to avoid
        # duplicate orders across consecutive poll cycles.
        self._order_placed_ids: set = set()
        self._running = False
        # Optional callback invoked with the symbol whenever a stop-loss is hit.
        # Set after construction (e.g. to scanner.set_symbol_sl_cooldown).
        self.on_sl_callback: Optional[Any] = None
        # Optional callback invoked with (symbol, channel, direction) on invalidation.
        # Set after construction (e.g. to scanner.set_invalidation_cooldown).
        self.on_invalidation_callback: Optional[Any] = None
        # Optional callback invoked with (symbol, channel, direction, setup_class,
        # hold_duration_seconds) when a stop-loss is hit.  Used to set thesis-based
        # cooldowns in the scanner.
        self.on_thesis_sl_callback: Optional[Any] = None
        # Optional callback invoked with (signal, tp_level, tp_pnl_pct) when TP2+ is hit.
        # Used to post highlights to the free channel.
        self.on_highlight_callback: Optional[Any] = None
        # Optional AI Trade Observer — captures full trade lifecycle data.
        # Set after construction (e.g. in main.py after router.observer is wired).
        self.observer: Optional[Any] = None
        # Optional StatisticalFilter — records resolved signal outcomes so the
        # rolling win-rate store can adapt confidence gating over time.
        self._stat_filter = stat_filter

    def _record_outcome(self, sig: Signal, hit_tp: int, hit_sl: bool) -> None:
        """Notify performance tracker and circuit breaker of a completed signal.

        Called only on final outcomes (semantic stop/TP completion). Intermediate hits
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
        # Actual PnL = the real exit price PnL (used for circuit breaker)
        actual_pnl = sig.pnl_pct

        # Signal quality PnL = best TP PnL if a TP was previously reached and is
        # better than the final outcome; otherwise same as actual PnL
        signal_quality_pnl = actual_pnl
        signal_quality_hit_tp = hit_tp
        if sig.best_tp_hit > 0 and sig.best_tp_hit > hit_tp:
            signal_quality_pnl = sig.best_tp_pnl_pct
            signal_quality_hit_tp = sig.best_tp_hit

        hold_duration_sec = max((utcnow() - sig.timestamp).total_seconds(), 0.0)
        outcome_label = classify_trade_outcome(
            pnl_pct=actual_pnl,
            hit_tp=hit_tp,
            hit_sl=hit_sl,
        )
        if self._performance_tracker is not None:
            self._performance_tracker.record_outcome(
                signal_id=sig.signal_id,
                channel=sig.channel,
                symbol=sig.symbol,
                direction=sig.direction.value,
                entry=sig.entry,
                hit_tp=hit_tp,
                hit_sl=hit_sl,
                pnl_pct=actual_pnl,
                outcome_label=outcome_label,
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
                signal_quality_pnl_pct=signal_quality_pnl,
                signal_quality_hit_tp=signal_quality_hit_tp,
            )
        # Circuit breaker ALWAYS uses actual PnL (real exit price)
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_outcome(
                signal_id=sig.signal_id,
                hit_sl=hit_sl,
                pnl_pct=actual_pnl,
                symbol=sig.symbol,
            )
        if hit_sl:
            # Notify the scanner to apply a short per-symbol cooldown so no other
            # channel fires on the same symbol immediately after a stop-loss.
            if self.on_sl_callback is not None:
                self.on_sl_callback(sig.symbol)
            # Notify the scanner about the thesis that failed so it can apply a
            # longer thesis-based cooldown to prevent repeat entries.
            if self.on_thesis_sl_callback is not None:
                self.on_thesis_sl_callback(
                    sig.symbol,
                    sig.channel,
                    sig.direction.value,
                    sig.setup_class or "",
                    hold_duration_sec,
                )
        # Release order-placement tracking for this closed signal so that the
        # set does not grow without bound across many completed signals.
        self._order_placed_ids.discard(sig.signal_id)

        # Notify the AI Trade Observer with exit analysis (fail-open)
        if self.observer is not None:
            try:
                self.observer.capture_exit_analysis(sig, outcome_label, actual_pnl)
            except Exception as exc:
                log.debug("TradeObserver.capture_exit_analysis failed (non-critical): {}", exc)

        # Statistical filter outcome recording — updates rolling win-rate store
        # so the filter can penalise or suppress future signals from poor
        # (channel, pair, regime) combinations.
        if self._stat_filter is not None:
            try:
                won = signal_quality_hit_tp >= 1
                _sf_outcome = SignalOutcome(
                    signal_id=sig.signal_id,
                    channel=sig.channel,
                    pair=sig.symbol,
                    regime=getattr(sig, "entry_regime", "") or "",
                    setup_class=sig.setup_class or "",
                    won=won,
                    pnl_pct=signal_quality_pnl,
                )
                self._stat_filter.record(_sf_outcome)
            except Exception as exc:
                log.debug("stat_filter.record failed (non-critical): {}", exc)

    @staticmethod
    def _set_realized_pnl(sig: Signal, exit_price: float) -> None:
        """Freeze final trade PnL at the executed exit level."""
        sig.current_price = exit_price
        sig.pnl_pct = calculate_trade_pnl_pct(
            entry_price=sig.entry,
            exit_price=exit_price,
            direction=sig.direction.value,
        )

    @staticmethod
    def _apply_final_outcome(sig: Signal, hit_tp: int, hit_sl: bool) -> str:
        """Apply the semantic final outcome label to the signal and return it."""
        outcome_label = classify_trade_outcome(
            pnl_pct=sig.pnl_pct,
            hit_tp=hit_tp,
            hit_sl=hit_sl,
        )
        sig.status = outcome_label
        return outcome_label

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

        async def _process_signal(sig: Signal) -> None:
            price = self._latest_price(sig.symbol)
            if price is None:
                return
            sig.current_price = price
            # Auto-execution: attempt to place an order the first time we see
            # this signal (status == "ACTIVE" and no order has been placed yet).
            # The OrderManager is a no-op when auto-execution is disabled.
            if (
                self._order_manager is not None
                and self._order_manager.is_enabled
                and sig.status == "ACTIVE"
                and sig.signal_id not in self._order_placed_ids
            ):
                try:
                    order_id = await self._order_manager.execute_signal(sig)
                    self._order_placed_ids.add(sig.signal_id)
                    if order_id:
                        log.info(
                            "Auto-execution order placed for {} {}: order_id={}",
                            sig.symbol,
                            sig.channel,
                            order_id,
                        )
                except Exception as exc:
                    log.warning(
                        "Auto-execution failed for {} {}: {}",
                        sig.symbol,
                        sig.channel,
                        exc,
                    )
            await self._evaluate_signal(sig)

        await asyncio.gather(*[_process_signal(sig) for sig in signals.values()])

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

    def _check_invalidation(self, sig: Signal) -> Optional[str]:
        """Return an invalidation reason string if the signal's thesis is no longer valid.

        Checks (in order):
        1. Market regime flip against signal direction
        2. EMA trend crossover against signal direction
        3. Momentum loss (signal is flat with no movement)

        Returns ``None`` if the signal is still valid.
        """
        is_long = sig.direction == Direction.LONG

        # Age gate — ALL invalidation checks require minimum age to avoid
        # reacting to 1m candle noise and killing trades too early.
        min_age = INVALIDATION_MIN_AGE_SECONDS.get(sig.channel, 120)
        age_secs = (utcnow() - sig.timestamp).total_seconds()
        if age_secs < min_age:
            return None  # Too young for any invalidation

        # DCA grace period — give the averaged position time to develop before
        # allowing invalidation to close it prematurely.
        if sig.entry_2_filled and sig.dca_timestamp is not None:
            dca_age = (utcnow() - sig.dca_timestamp).total_seconds()
            if dca_age < _DCA_GRACE_SECONDS:
                return None

        # Build an indicators dict for regime detection and EMA/momentum checks.
        # Priority: caller-supplied indicators_fn → data-store fallback.
        indicators: Optional[dict] = None
        if self._indicators_fn is not None:
            try:
                indicators = self._indicators_fn(sig.symbol)
            except Exception as exc:
                log.debug("indicators_fn failed for %s: %s", sig.symbol, exc)

        # Fallback: derive EMA9/EMA21 and momentum from candles in data store.
        if indicators is None and self._store is not None:
            candles = self._store.get_candles(sig.symbol, "5m")
            if not (candles and len(candles.get("close", [])) >= 21):
                candles = self._store.get_candles(sig.symbol, "1m")
            if candles and len(candles.get("close", [])) >= 21:
                closes = np.asarray(candles["close"], dtype=np.float64)
                ema9_arr = _compute_ema(closes, 9)
                ema21_arr = _compute_ema(closes, 21)
                mom_arr = _compute_momentum(closes, 3) if len(closes) >= 4 else np.array([])
                indicators = {
                    "ema9_last": float(ema9_arr[-1]) if len(ema9_arr) else None,
                    "ema21_last": float(ema21_arr[-1]) if len(ema21_arr) else None,
                    "momentum": float(mom_arr[-1]) if len(mom_arr) and not np.isnan(mom_arr[-1]) else None,
                }

        # 1. Market regime flip – use regime_detector.classify() with indicators
        if self._regime_detector is not None and indicators is not None:
            try:
                result = self._regime_detector.classify(indicators)
                regime_label = result.regime.value if result and result.regime else None
                if is_long and regime_label == "TRENDING_DOWN":
                    return f"regime shift to {regime_label} – LONG thesis no longer valid"
                if not is_long and regime_label == "TRENDING_UP":
                    return f"regime shift to {regime_label} – SHORT thesis no longer valid"
            except Exception as exc:
                log.debug("Regime detection failed for %s: %s", sig.symbol, exc)

        if indicators is None:
            return None

        ema9 = indicators.get("ema9_last")
        ema21 = indicators.get("ema21_last")
        momentum = indicators.get("momentum")

        # 2. EMA crossover against signal direction
        # After TP1 has been hit, let trailing stop manage the exit — don't kill
        # a profitable trade just because the 1m EMA crosses (common noise).
        if ema9 is not None and ema21 is not None and sig.status not in ("TP1_HIT", "TP2_HIT"):
            if is_long and ema9 < ema21:
                return "EMA bearish crossover (EMA9 < EMA21) – LONG thesis invalidated"
            if not is_long and ema9 > ema21:
                return "EMA bullish crossover (EMA9 > EMA21) – SHORT thesis invalidated"

        # 3. Momentum loss – threshold is per-channel since different timeframes have
        # different noise characteristics (TAPE 1m candles have rapid oscillation).
        # For micro-cap tokens (entry price < 0.001), scale threshold by 0.1 to
        # avoid false invalidations on tiny absolute price moves (e.g. BONKUSDT).
        mom_threshold = INVALIDATION_MOMENTUM_THRESHOLD.get(sig.channel, 0.15)
        # Prefer entry price; fall back to current_price only if entry is unset (0).
        # The current_price check guards against a zero fallback.
        entry_price = sig.entry if sig.entry > 0 else sig.current_price
        if 0 < entry_price < 0.001:
            mom_threshold *= 0.1
        if momentum is not None and abs(momentum) < mom_threshold:
            sig.momentum_invalidation_count += 1
            consecutive_required = INVALIDATION_CONSECUTIVE_THRESHOLD.get(sig.channel, 1)
            if sig.momentum_invalidation_count >= consecutive_required:
                return (
                    f"momentum loss (|momentum|={abs(momentum):.3f} < "
                    f"{mom_threshold}, {sig.momentum_invalidation_count} consecutive readings)"
                    " – signal thesis exhausted"
                )
            # Not enough consecutive readings yet — don't invalidate
        else:
            sig.momentum_invalidation_count = 0  # Reset on recovery

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

        # Max hold duration guard – auto-expire signals that have been open too long
        max_hold = MAX_SIGNAL_HOLD_SECONDS.get(sig.channel, 86400)
        if age_secs >= max_hold:
            self._set_realized_pnl(sig, price)
            sig.status = "EXPIRED"
            await self._post_update(sig, "⏰ EXPIRED (max hold time reached)")
            self._record_outcome(sig, hit_tp=0, hit_sl=False)
            self._remove(sig.signal_id)
            return

        # Notify AI Trade Observer with a mid-trade snapshot (fail-open)
        if self.observer is not None:
            try:
                self.observer.observe_trade(sig, price)
            except Exception as exc:
                log.debug("TradeObserver.observe_trade failed (non-critical): {}", exc)

        # DCA (Double Entry) check — only on ACTIVE signals before TP1 is hit
        if sig.status == "ACTIVE" and not sig.entry_2_filled:
            chan_cfg = next(
                (c for c in ALL_CHANNELS if c.name == sig.channel), None
            )
            if chan_cfg is not None and chan_cfg.dca_enabled:
                dca_price = check_dca_entry(
                    sig=sig,
                    current_price=price,
                    indicators=None,
                    smc_data=None,
                    channel_config=chan_cfg,
                )
                if dca_price is not None:
                    recalculate_after_dca(
                        sig=sig,
                        entry_2_price=dca_price,
                        tp_ratios=list(chan_cfg.tp_ratios),
                        weight_1=chan_cfg.dca_weight_1,
                        weight_2=chan_cfg.dca_weight_2,
                    )
                    await self._post_dca_update(sig)

        # SL direction sanity check – catch misconfigured signals
        protective_stop_active = sig.status in ("TP1_HIT", "TP2_HIT")
        if is_long and sig.stop_loss > sig.entry and not protective_stop_active:
            log.warning(
                "Signal %s %s has invalid SL (LONG SL %.8f > entry %.8f) – cancelling",
                sig.symbol, sig.signal_id, sig.stop_loss, sig.entry,
            )
            sig.status = "CANCELLED"
            await self._post_update(sig, "⚠️ CANCELLED (invalid SL)")
            self._remove(sig.signal_id)
            return
        if not is_long and sig.stop_loss < sig.entry and not protective_stop_active:
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

        # Stop-loss hit — checked BEFORE invalidation so that a price gap
        # through the SL is never exited at a worse price via invalidation.
        if is_long and price <= sig.stop_loss:
            self._set_realized_pnl(sig, sig.stop_loss)
            outcome_label = self._apply_final_outcome(sig, hit_tp=0, hit_sl=True)
            outcome_event = _STOP_OUTCOME_MESSAGES.get(outcome_label, "🔴 EXIT")
            await self._post_update(sig, outcome_event)
            self._record_outcome(sig, hit_tp=0, hit_sl=True)
            self._remove(sig.signal_id)
            return
        if not is_long and price >= sig.stop_loss:
            self._set_realized_pnl(sig, sig.stop_loss)
            outcome_label = self._apply_final_outcome(sig, hit_tp=0, hit_sl=True)
            outcome_event = _STOP_OUTCOME_MESSAGES.get(outcome_label, "🔴 EXIT")
            await self._post_update(sig, outcome_event)
            self._record_outcome(sig, hit_tp=0, hit_sl=True)
            self._remove(sig.signal_id)
            return

        # Market-structure invalidation – close stale signals whose thesis no
        # longer holds (regime flip, momentum loss, EMA crossover).  Checked
        # AFTER the SL check so that a price gap through the SL is always
        # caught at the SL level, not at the (potentially worse) current price.
        invalidation_reason = self._check_invalidation(sig)
        if invalidation_reason:
            # Cap the exit price — invalidation must never produce a worse exit
            # than the SL would have given.  For a LONG that gapped down, the
            # capped price is the SL; for a SHORT that gapped up, it is the SL.
            if is_long:
                capped_price = max(price, sig.stop_loss)
            else:
                capped_price = min(price, sig.stop_loss)
            self._set_realized_pnl(sig, capped_price)
            sig.status = "INVALIDATED"
            await self._post_update(sig, f"🔄 INVALIDATED ({invalidation_reason})")
            self._record_outcome(sig, hit_tp=0, hit_sl=False)
            self._remove(sig.signal_id)
            if self.on_invalidation_callback is not None:
                self.on_invalidation_callback(sig.symbol, sig.channel, sig.direction.value)
            return

        # TP hits (progressive)
        if is_long:
            if sig.tp3 and price >= sig.tp3 and sig.status != "TP3_HIT":
                tp3_pnl = calculate_trade_pnl_pct(
                    entry_price=sig.entry, exit_price=sig.tp3, direction=sig.direction.value
                )
                if self.on_highlight_callback is not None:
                    self.on_highlight_callback(sig, 3, tp3_pnl)
                # Partial TP3 execution: close 34% of original position size
                if self._order_manager is not None and self._order_manager.is_enabled:
                    try:
                        await self._order_manager.close_partial(sig, 0.34, tp_level=3)
                    except Exception as _exc:
                        log.warning("Partial TP3 close failed for {}: {}", sig.symbol, _exc)
                self._set_realized_pnl(sig, sig.tp3)
                self._apply_final_outcome(sig, hit_tp=3, hit_sl=False)
                await self._post_update(sig, "🎯🎯🎯 FULL TP HIT")
                self._record_outcome(sig, hit_tp=3, hit_sl=False)
                self._remove(sig.signal_id)
                return
            if price >= sig.tp2 and sig.status not in ("TP2_HIT", "TP3_HIT"):
                sig.status = "TP2_HIT"
                await self._post_update(sig, "🎯🎯 TP2 HIT")
                # Snapshot best-TP PnL for signal quality stats
                sig.best_tp_hit = 2
                sig.best_tp_pnl_pct = calculate_trade_pnl_pct(
                    entry_price=sig.entry, exit_price=sig.tp2, direction=sig.direction.value
                )
                if self.on_highlight_callback is not None:
                    self.on_highlight_callback(sig, 2, sig.best_tp_pnl_pct)
                # Trailing: move SL to TP1 price to protect banked profit while giving TP3 room
                sig.stop_loss = sig.tp1
                # Partial TP2 execution: close 33% of original position size
                if self._order_manager is not None and self._order_manager.is_enabled:
                    try:
                        await self._order_manager.close_partial(sig, 0.33, tp_level=2)
                    except Exception as _exc:
                        log.warning("Partial TP2 close failed for {}: {}", sig.symbol, _exc)
            if price >= sig.tp1 and sig.status not in ("TP1_HIT", "TP2_HIT", "TP3_HIT"):
                sig.status = "TP1_HIT"
                await self._post_update(sig, "🎯 TP1 HIT ✅")
                # Snapshot best-TP PnL for signal quality stats (only if TP2 not already hit)
                if sig.best_tp_hit < 1:
                    sig.best_tp_hit = 1
                    sig.best_tp_pnl_pct = calculate_trade_pnl_pct(
                        entry_price=sig.entry, exit_price=sig.tp1, direction=sig.direction.value
                    )
                # Move SL to breakeven + small buffer (15% of TP1 distance) so that a
                # retrace between TP1 and TP2 never produces a full loss after the thesis
                # has already been proven by TP1.  Only move SL upward for longs.
                tp1_dist = abs(sig.tp1 - sig.entry)
                be_buffer = tp1_dist * 0.15
                new_be_sl = sig.entry + be_buffer
                sig.stop_loss = max(sig.stop_loss, new_be_sl)
                # Partial TP1 execution: close 33% of original position size
                if self._order_manager is not None and self._order_manager.is_enabled:
                    try:
                        await self._order_manager.close_partial(sig, 0.33, tp_level=1)
                    except Exception as _exc:
                        log.warning("Partial TP1 close failed for {}: {}", sig.symbol, _exc)
        else:
            if sig.tp3 and price <= sig.tp3 and sig.status != "TP3_HIT":
                tp3_pnl = calculate_trade_pnl_pct(
                    entry_price=sig.entry, exit_price=sig.tp3, direction=sig.direction.value
                )
                if self.on_highlight_callback is not None:
                    self.on_highlight_callback(sig, 3, tp3_pnl)
                # Partial TP3 execution: close 34% of original position size
                if self._order_manager is not None and self._order_manager.is_enabled:
                    try:
                        await self._order_manager.close_partial(sig, 0.34, tp_level=3)
                    except Exception as _exc:
                        log.warning("Partial TP3 close failed for {}: {}", sig.symbol, _exc)
                self._set_realized_pnl(sig, sig.tp3)
                self._apply_final_outcome(sig, hit_tp=3, hit_sl=False)
                await self._post_update(sig, "🎯🎯🎯 FULL TP HIT")
                self._record_outcome(sig, hit_tp=3, hit_sl=False)
                self._remove(sig.signal_id)
                return
            if price <= sig.tp2 and sig.status not in ("TP2_HIT", "TP3_HIT"):
                sig.status = "TP2_HIT"
                await self._post_update(sig, "🎯🎯 TP2 HIT")
                # Snapshot best-TP PnL for signal quality stats
                sig.best_tp_hit = 2
                sig.best_tp_pnl_pct = calculate_trade_pnl_pct(
                    entry_price=sig.entry, exit_price=sig.tp2, direction=sig.direction.value
                )
                if self.on_highlight_callback is not None:
                    self.on_highlight_callback(sig, 2, sig.best_tp_pnl_pct)
                sig.stop_loss = sig.tp1
                # Partial TP2 execution: close 33% of original position size
                if self._order_manager is not None and self._order_manager.is_enabled:
                    try:
                        await self._order_manager.close_partial(sig, 0.33, tp_level=2)
                    except Exception as _exc:
                        log.warning("Partial TP2 close failed for {}: {}", sig.symbol, _exc)
            if price <= sig.tp1 and sig.status not in ("TP1_HIT", "TP2_HIT", "TP3_HIT"):
                sig.status = "TP1_HIT"
                await self._post_update(sig, "🎯 TP1 HIT ✅")
                # Snapshot best-TP PnL for signal quality stats (only if TP2 not already hit)
                if sig.best_tp_hit < 1:
                    sig.best_tp_hit = 1
                    sig.best_tp_pnl_pct = calculate_trade_pnl_pct(
                        entry_price=sig.entry, exit_price=sig.tp1, direction=sig.direction.value
                    )
                # Move SL to breakeven - small buffer (15% of TP1 distance) so that a
                # retrace between TP1 and TP2 never produces a full loss after the thesis
                # has already been proven by TP1.  Only move SL downward for shorts.
                tp1_dist = abs(sig.tp1 - sig.entry)
                be_buffer = tp1_dist * 0.15
                new_be_sl = sig.entry - be_buffer
                sig.stop_loss = min(sig.stop_loss, new_be_sl)
                # Partial TP1 execution: close 33% of original position size
                if self._order_manager is not None and self._order_manager.is_enabled:
                    try:
                        await self._order_manager.close_partial(sig, 0.33, tp_level=1)
                    except Exception as _exc:
                        log.warning("Partial TP1 close failed for {}: {}", sig.symbol, _exc)

        # Trailing stop adjustment
        if sig.trailing_active and sig.status in ("TP1_HIT", "TP2_HIT"):
            self._adjust_trailing(sig)

    def _adjust_trailing(self, sig: Signal) -> None:
        """Move the trailing stop behind the price using an ATR-based distance.

        The trailing distance is ``atr_value * atr_multiplier`` where
        ``atr_multiplier`` comes from the channel's ``trailing_atr_mult``
        config field (or the global ``TRAILING_ATR_MULTIPLIER`` constant when
        the channel config cannot be found).

        Phase-based tightening: after TP1 the multiplier is reduced to 55% of
        the base, and after TP2 to 35%, locking progressively more profit.

        Regime-aware adjustment: in trending markets the trail is kept loose
        (×1.2) to let winners run; in ranging markets it is tightened (×0.7)
        to protect profit before a range-edge reversal.

        Falls back to ``original_sl_distance * 0.75`` when ATR data is
        unavailable (e.g. candles not yet loaded for this symbol).
        """
        price = sig.current_price
        # Use the original SL distance (stored at signal creation) so that the
        # trailing buffer doesn't collapse to zero after TP2 moves SL to break-even.
        # Fall back to the live distance only for legacy signals where the field is unset.
        base_dist = sig.original_sl_distance or abs(sig.entry - sig.stop_loss)

        # ------------------------------------------------------------------
        # Attempt ATR-based trailing distance
        # ------------------------------------------------------------------
        trail_dist: Optional[float] = None
        candles = self._store.get_candles(sig.symbol, "1m")
        if candles is not None and len(candles.get("close", [])) >= 15:
            try:
                highs = np.asarray(candles["high"], dtype=np.float64)
                lows = np.asarray(candles["low"], dtype=np.float64)
                closes = np.asarray(candles["close"], dtype=np.float64)
                atr_arr = _compute_atr(highs, lows, closes, 14)
                valid = atr_arr[~np.isnan(atr_arr)]
                if len(valid) > 0:
                    atr_value = float(valid[-1])
                    # Use per-channel multiplier when available, otherwise the global default
                    chan_cfg = next(
                        (c for c in ALL_CHANNELS if c.name == sig.channel), None
                    )
                    base_mult = (
                        chan_cfg.trailing_atr_mult if chan_cfg is not None else TRAILING_ATR_MULTIPLIER
                    )

                    # Phase-based tightening: lock more profit as each TP is cleared
                    if sig.status == "TP2_HIT":
                        effective_mult = base_mult * 0.35  # Very tight – profit protection
                    elif sig.status == "TP1_HIT":
                        effective_mult = base_mult * 0.55  # Tighter after first target
                    else:
                        effective_mult = base_mult  # Default for ACTIVE signals

                    # Regime-aware adjustment: loose in trends, tight in ranges
                    if self._regime_detector is not None:
                        try:
                            indicators_for_regime: dict = {}
                            if self._indicators_fn is not None:
                                indicators_for_regime = self._indicators_fn(sig.symbol) or {}
                            elif candles and len(candles.get("close", [])) >= 21:
                                regime_closes = np.asarray(candles["close"], dtype=np.float64)
                                ema9_arr = _compute_ema(regime_closes, 9)
                                ema21_arr = _compute_ema(regime_closes, 21)
                                indicators_for_regime = {
                                    "ema9_last": float(ema9_arr[-1]) if len(ema9_arr) else None,
                                    "ema21_last": float(ema21_arr[-1]) if len(ema21_arr) else None,
                                }
                            regime_result = self._regime_detector.classify(indicators_for_regime)
                            regime_label = regime_result.regime.value if regime_result else "RANGING"
                            regime_trail_mult = {
                                "TRENDING_UP": 1.2,
                                "TRENDING_DOWN": 1.2,
                                "RANGING": 0.7,
                                "VOLATILE": 0.9,
                                "QUIET": 0.8,
                            }.get(regime_label, 1.0)
                            effective_mult *= regime_trail_mult
                        except Exception:
                            pass  # Fall back to non-regime-adjusted multiplier

                    trail_dist = atr_value * effective_mult
            except Exception:
                trail_dist = None

        # Fall back to fixed 75 % of original SL distance when ATR is unavailable
        if trail_dist is None:
            trail_dist = base_dist * 0.75

        if sig.direction == Direction.LONG:
            new_sl = price - trail_dist
            if new_sl > sig.stop_loss:
                sig.stop_loss = round(new_sl, 8)
        else:
            new_sl = price + trail_dist
            if new_sl < sig.stop_loss:
                sig.stop_loss = round(new_sl, 8)

    async def _post_dca_update(self, sig: Signal) -> None:
        """Post a Telegram notification when DCA Entry 2 is taken."""
        channel_id = CHANNEL_TELEGRAM_MAP.get(sig.channel, "")
        if not channel_id:
            return

        chan_emojis = {
            "360_SCALP": "⚡",
            "360_SWING": "🏛️",
            "360_SPOT": "📈",
            "360_GEM": "💎",
        }
        chan_emoji = chan_emojis.get(sig.channel, "📡")
        dir_emoji = "🚀" if sig.direction == Direction.LONG else "⬇️"

        # Build R-multiples string from current channel config
        chan_cfg = next((c for c in ALL_CHANNELS if c.name == sig.channel), None)
        rr_str = ""
        if chan_cfg is not None:
            rr_parts = [f"{r}R" for r in chan_cfg.tp_ratios]
            rr_str = " / ".join(rr_parts)

        lines = [
            "📊 DCA ENTRY 2",
            f"{chan_emoji} *{_escape_md(sig.channel)}* | {_escape_md(sig.symbol)} *{sig.direction.value}* {dir_emoji}",
            f"💰 Entry 1: `{fmt_price(sig.original_entry)}` → Entry 2: `{fmt_price(sig.entry_2 if sig.entry_2 is not None else 0.0)}`",
            f"📊 Avg Entry: `{fmt_price(sig.avg_entry)}`",
            f"🎯 New TP1: `{fmt_price(sig.tp1)}` | TP2: `{fmt_price(sig.tp2)}`"
            + (f" | TP3: `{fmt_price(sig.tp3)}`" if sig.tp3 is not None else ""),
            f"🛑 SL: `{fmt_price(sig.stop_loss)}` (unchanged)",
        ]
        if rr_str:
            lines.append(f"📏 New R:R preserved at {rr_str}")
        lines.append(f"⏰ {fmt_ts()}")

        text = "\n".join(lines)
        await self._send(channel_id, text)

    async def _post_update(self, sig: Signal, event: str) -> None:
        channel_id = CHANNEL_TELEGRAM_MAP.get(sig.channel, "")
        if not channel_id:
            return

        chan_emojis = {
            "360_SCALP": "⚡",
            "360_SWING": "🏛️",
            "360_SPOT": "📈",
            "360_GEM": "💎",
        }
        chan_emoji = chan_emojis.get(sig.channel, "📡")
        dir_emoji = "🚀" if sig.direction == Direction.LONG else "⬇️"

        lines = [
            f"{event}",
            f"{chan_emoji} *{_escape_md(sig.channel)}* | {_escape_md(sig.symbol)} *{sig.direction.value}* {dir_emoji}",
            f"💰 Entry: `{fmt_price(sig.entry)}` → Current: `{fmt_price(sig.current_price)}`",
            f"📊 PnL: *{sig.pnl_pct:+.2f}%*",
            f"🛡️ SL: `{fmt_price(sig.stop_loss)}`",
            f"🤖 Confidence: *{sig.confidence:.0f}%*",
        ]
        if sig.trailing_active and sig.trailing_desc:
            lines.append(f"💹 Trailing Active ({_escape_md(sig.trailing_desc)})")
        lines.append(f"⏰ {fmt_ts()}")

        text = "\n".join(lines)
        await self._send(channel_id, text)
