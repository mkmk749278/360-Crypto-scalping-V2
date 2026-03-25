"""Backtesting framework.

Replays historical candle data through channel strategies and computes
performance metrics including win rate, average R:R, and max drawdown.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.channels.base import Signal
from src.channels.scalp import ScalpChannel
from src.channels.swing import SwingChannel
from src.channels.spot import SpotChannel
from src.detector import SMCDetector
from src.indicators import adx, atr, bollinger_bands, ema, momentum, rsi
from src.utils import get_logger

log = get_logger("backtester")

# Thresholds for converting a numeric AI sentiment score ([-1, 1]) to a label.
_AI_BULLISH_THRESHOLD = 0.2
_AI_BEARISH_THRESHOLD = -0.2

# Channel name substrings used to identify SCALP channels for automatic
# execution_delay_candles assignment.  Scalp channels target M1/M5 candles
# where 0.5–3 s of live-trading latency corresponds to ~1 candle of slippage.
_SCALP_CHANNEL_NAMES = ("360_SCALP",)


@dataclass
class BacktestResult:
    """Summary metrics for a single backtest run."""

    channel: str
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    partial_wins: int = 0
    win_rate: float = 0.0
    avg_rr: float = 0.0
    max_drawdown: float = 0.0
    total_pnl_pct: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    slippage_pct: float = 0.0
    signal_details: List[Dict] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary."""
        slippage_note = (
            f"\nSlippage Assumption: {self.slippage_pct:.4f}%/trade"
            if self.slippage_pct > 0
            else ""
        )
        return (
            f"Backtest: {self.channel}\n"
            f"Signals: {self.total_signals} | Wins: {self.wins} | Losses: {self.losses}\n"
            f"Win Rate: {self.win_rate:.1f}%\n"
            f"Avg R:R: {self.avg_rr:.2f}\n"
            f"Total PnL: {self.total_pnl_pct:+.2f}%\n"
            f"Max Drawdown: {self.max_drawdown:.2f}%\n"
            f"Best: {self.best_trade:+.2f}% | Worst: {self.worst_trade:+.2f}%"
            f"{slippage_note}"
        )


def _compute_indicators(candles: Dict) -> Dict:
    """Compute the standard set of technical indicators from candle arrays."""
    h = candles.get("high", np.array([]))
    lo = candles.get("low", np.array([]))
    c = candles.get("close", np.array([]))
    ind: Dict = {}

    if len(c) >= 21:
        ind["ema9_last"] = float(ema(c, 9)[-1])
        ind["ema21_last"] = float(ema(c, 21)[-1])
    if len(c) >= 200:
        ind["ema200_last"] = float(ema(c, 200)[-1])
    if len(c) >= 30:
        a = adx(h, lo, c, 14)
        valid = a[~np.isnan(a)]
        ind["adx_last"] = float(valid[-1]) if len(valid) else None
    if len(c) >= 15:
        a = atr(h, lo, c, 14)
        valid = a[~np.isnan(a)]
        ind["atr_last"] = float(valid[-1]) if len(valid) else None
    if len(c) >= 15:
        r = rsi(c, 14)
        valid = r[~np.isnan(r)]
        ind["rsi_last"] = float(valid[-1]) if len(valid) else None
    if len(c) >= 20:
        u, m, lo_b = bollinger_bands(c, 20)
        ind["bb_upper_last"] = float(u[-1]) if not np.isnan(u[-1]) else None
        ind["bb_mid_last"] = float(m[-1]) if not np.isnan(m[-1]) else None
        ind["bb_lower_last"] = float(lo_b[-1]) if not np.isnan(lo_b[-1]) else None
    if len(c) >= 4:
        mom = momentum(c, 3)
        ind["momentum_last"] = float(mom[-1]) if not np.isnan(mom[-1]) else None

    return ind


def _simulate_trade(
    signal: Signal,
    future_candles: Dict,
    sl_multiplier: float = 1.0,
    fee_pct: float = 0.0,
    slippage_pct: float = 0.0,
    funding_rate_8h: float = 0.0,
    candle_interval_minutes: int = 5,
    execution_delay_candles: int = 0,
) -> Tuple[bool, float, int]:
    """Simulate a signal against future price data.

    Parameters
    ----------
    signal:
        The signal to simulate.
    future_candles:
        OHLCV dict of future candles.
    sl_multiplier:
        Multiplier applied to the stop-loss price.
    fee_pct:
        Round-trip fee percentage (entry + exit) deducted from PnL.
        E.g. ``0.08`` deducts 0.08 % for a typical Binance maker/taker
        round-trip.  Defaults to ``0.0`` (no fees) for backward compatibility.
    slippage_pct:
        Per-trade slippage percentage applied adversely to both entry and
        SL/TP fills.  Entry slippage is ``slippage_pct / 2`` applied
        adversely (LONG entries fill higher, SHORT entries fill lower).
        SL/TP fills receive a slight haircut in the unfavourable direction.
        Defaults to ``0.0`` (no slippage) for backward compatibility.
    funding_rate_8h:
        Funding rate per 8-hour period as a percentage (e.g. ``0.01`` for
        0.01 % per 8 h).  Deducted from PnL based on how many 8-hour periods
        the simulated trade is held open.  Defaults to ``0.0``.
    candle_interval_minutes:
        Duration of each candle in minutes (used for funding rate calculation).
        Defaults to ``5`` (5-minute candles).
    execution_delay_candles:
        Number of candles to skip before entry to simulate live-trading
        latency.  When > 0 the entry price is set to the *open* of the candle
        ``execution_delay_candles`` ahead instead of signal close.  Scalp
        channels (M1/M5) use 1 by default to account for the 0.5–3 s round-
        trip between signal detection and exchange fill.  Defaults to ``0``
        (instant execution at signal candle close).

    Returns
    -------
    (won, pnl_pct, tp_level_hit)
        ``won`` is True if TP1 was hit before SL.
        ``pnl_pct`` is the estimated PnL percentage (net of fees and slippage).
        ``tp_level_hit`` is 0 (SL), 1 (TP1), 2 (TP2), or 3 (TP3).
    """
    highs = future_candles.get("high", np.array([]))
    lows = future_candles.get("low", np.array([]))
    opens = future_candles.get("open", np.array([]))

    if len(highs) == 0:
        return False, 0.0, 0

    # Execution delay: shift entry to the open of a future candle.
    # If not enough candles remain after the delay, skip the trade.
    if execution_delay_candles > 0:
        if execution_delay_candles >= len(highs):
            return False, 0.0, 0
        delayed_open = opens[execution_delay_candles] if len(opens) > execution_delay_candles else None
        if delayed_open is not None:
            signal = dataclasses.replace(signal, entry=float(delayed_open))
        highs = highs[execution_delay_candles:]
        lows = lows[execution_delay_candles:]

    is_long = signal.direction.value == "LONG"
    sl = signal.stop_loss * sl_multiplier
    slip = slippage_pct / 100.0  # fraction

    # --- Entry slippage: applied adversely at half the per-trade rate ---
    # LONG entries fill slightly higher (worse price for buyer).
    # SHORT entries fill slightly lower (worse price for seller).
    entry_slip = slip / 2.0
    if is_long:
        entry_fill = signal.entry * (1.0 + entry_slip)
    else:
        entry_fill = signal.entry * (1.0 - entry_slip)

    targets = [signal.tp1, signal.tp2]
    if signal.tp3 is not None:
        targets.append(signal.tp3)

    # --- Trailing stop state ---
    # Tracks the moving stop level after TP1/TP2 are hit.
    current_sl = sl
    original_sl = sl
    tp1_hit = False
    tp2_hit = False
    partial_pnl_sum = 0.0  # Accumulated PnL from partial exits (33% each at TP1 & TP2)
    candles_held = 0

    for i in range(min(len(highs), len(lows))):
        h, lo = float(highs[i]), float(lows[i])
        candles_held = i + 1

        if is_long:
            # Check trailing SL first
            if lo <= current_sl:
                fill = current_sl * (1.0 - slip)
                remaining_pct = 1.0 - (0.33 if tp1_hit else 0.0) - (0.33 if tp2_hit else 0.0)
                close_pnl = (fill - entry_fill) / entry_fill * 100.0
                total_pnl = partial_pnl_sum + close_pnl * remaining_pct - fee_pct
                _deduct_funding = _calc_funding(funding_rate_8h, candles_held, candle_interval_minutes)
                return tp1_hit, total_pnl - _deduct_funding, 2 if tp2_hit else (1 if tp1_hit else 0)

            # Check TP targets
            if not tp1_hit and len(targets) >= 1 and h >= targets[0]:
                tp1_fill = targets[0] * (1.0 - slip)
                partial_pnl_sum += (tp1_fill - entry_fill) / entry_fill * 100.0 * 0.33
                tp1_hit = True
                # Move SL to breakeven + 15% buffer
                sl_buffer = (entry_fill - original_sl) * 0.15
                current_sl = entry_fill + sl_buffer

            if tp1_hit and not tp2_hit and len(targets) >= 2 and h >= targets[1]:
                tp2_fill = targets[1] * (1.0 - slip)
                partial_pnl_sum += (tp2_fill - entry_fill) / entry_fill * 100.0 * 0.33
                tp2_hit = True
                # Move SL to TP1
                current_sl = targets[0]

            if tp2_hit and len(targets) >= 3 and h >= targets[2]:
                tp3_fill = targets[2] * (1.0 - slip)
                final_pnl = partial_pnl_sum + (tp3_fill - entry_fill) / entry_fill * 100.0 * 0.34 - fee_pct
                _deduct_funding = _calc_funding(funding_rate_8h, candles_held, candle_interval_minutes)
                return True, final_pnl - _deduct_funding, 3

            if not tp2_hit and tp1_hit and len(targets) < 3 and len(targets) >= 2 and h >= targets[1]:
                # TP2 is the last target
                pass  # handled above

            # If only TP1 exists and it was hit, close out remainder now
            if tp1_hit and not tp2_hit and len(targets) == 1:
                tp1_fill = targets[0] * (1.0 - slip)
                final_pnl = (tp1_fill - entry_fill) / entry_fill * 100.0 - fee_pct
                _deduct_funding = _calc_funding(funding_rate_8h, candles_held, candle_interval_minutes)
                return True, final_pnl - _deduct_funding, 1

        else:  # SHORT
            # Check trailing SL first
            if h >= current_sl:
                fill = current_sl * (1.0 + slip)
                remaining_pct = 1.0 - (0.33 if tp1_hit else 0.0) - (0.33 if tp2_hit else 0.0)
                close_pnl = (entry_fill - fill) / entry_fill * 100.0
                total_pnl = partial_pnl_sum + close_pnl * remaining_pct - fee_pct
                _deduct_funding = _calc_funding(funding_rate_8h, candles_held, candle_interval_minutes)
                return tp1_hit, total_pnl - _deduct_funding, 2 if tp2_hit else (1 if tp1_hit else 0)

            # Check TP targets
            if not tp1_hit and len(targets) >= 1 and lo <= targets[0]:
                tp1_fill = targets[0] * (1.0 + slip)
                partial_pnl_sum += (entry_fill - tp1_fill) / entry_fill * 100.0 * 0.33
                tp1_hit = True
                # Move SL to breakeven - 15% buffer
                sl_buffer = (original_sl - entry_fill) * 0.15
                current_sl = entry_fill - sl_buffer

            if tp1_hit and not tp2_hit and len(targets) >= 2 and lo <= targets[1]:
                tp2_fill = targets[1] * (1.0 + slip)
                partial_pnl_sum += (entry_fill - tp2_fill) / entry_fill * 100.0 * 0.33
                tp2_hit = True
                # Move SL to TP1
                current_sl = targets[0]

            if tp2_hit and len(targets) >= 3 and lo <= targets[2]:
                tp3_fill = targets[2] * (1.0 + slip)
                final_pnl = partial_pnl_sum + (entry_fill - tp3_fill) / entry_fill * 100.0 * 0.34 - fee_pct
                _deduct_funding = _calc_funding(funding_rate_8h, candles_held, candle_interval_minutes)
                return True, final_pnl - _deduct_funding, 3

            # If only TP1 exists and it was hit, close out remainder now
            if tp1_hit and not tp2_hit and len(targets) == 1:
                tp1_fill = targets[0] * (1.0 + slip)
                final_pnl = (entry_fill - tp1_fill) / entry_fill * 100.0 - fee_pct
                _deduct_funding = _calc_funding(funding_rate_8h, candles_held, candle_interval_minutes)
                return True, final_pnl - _deduct_funding, 1

    # No TP or SL hit in the lookahead window — close at last available price
    last_close = float(future_candles.get("close", [signal.entry])[-1])
    remaining_pct = 1.0 - (0.33 if tp1_hit else 0.0) - (0.33 if tp2_hit else 0.0)
    if is_long:
        close_pnl = (last_close - entry_fill) / entry_fill * 100.0
    else:
        close_pnl = (entry_fill - last_close) / entry_fill * 100.0
    total_pnl = partial_pnl_sum + close_pnl * remaining_pct - fee_pct
    _deduct_funding = _calc_funding(funding_rate_8h, candles_held, candle_interval_minutes)
    total_pnl -= _deduct_funding
    won = tp1_hit or total_pnl > 0
    return won, total_pnl, 2 if tp2_hit else (1 if tp1_hit else 0)


def _calc_funding(
    funding_rate_8h: float,
    candles_held: int,
    candle_interval_minutes: int,
) -> float:
    """Calculate funding rate cost for the trade duration.

    Parameters
    ----------
    funding_rate_8h:
        Funding rate per 8-hour period as a percentage.
    candles_held:
        Number of candles the trade was open for.
    candle_interval_minutes:
        Duration of each candle in minutes.

    Returns
    -------
    float
        Funding cost as a percentage of notional to deduct from PnL.
    """
    if funding_rate_8h == 0.0:
        return 0.0
    funding_periods = candles_held * candle_interval_minutes / 480.0
    return funding_rate_8h * funding_periods


class Backtester:
    """Replays historical candle data through channel strategies.

    Parameters
    ----------
    channels:
        List of channel strategy objects to test.  Defaults to all three
        standard channels (SCALP, SWING, SPOT).
    lookahead_candles:
        Number of future candles to use for simulating trade outcomes.
    min_window:
        Minimum number of candles required before evaluation starts.
    fee_pct:
        Round-trip fee percentage deducted from every simulated trade's PnL.
        Defaults to ``0.08`` (typical Binance maker/taker round-trip).  Set to
        ``0.0`` only when comparing ideal (fee-free) scenarios.
    slippage_pct:
        Per-trade slippage percentage applied adversely to every SL/TP fill.
        Defaults to ``0.02`` for a realistic model of spread + market-impact.
        Set to ``0.0`` only when comparing ideal (no-slippage) scenarios.
    max_concurrent_positions:
        Maximum number of open simulated positions allowed at the same time.
        New signals are skipped when this limit is reached.  Defaults to ``5``.
    """

    def __init__(
        self,
        channels: Optional[List] = None,
        lookahead_candles: int = 20,
        min_window: int = 50,
        fee_pct: float = 0.08,
        slippage_pct: float = 0.02,
        funding_rate_per_8h: float = 0.01,
        max_concurrent_positions: int = 5,
    ) -> None:
        if channels is None:
            channels = [
                ScalpChannel(),
                SwingChannel(),
                SpotChannel(),
            ]
        self._channels = channels
        self._lookahead = lookahead_candles
        self._min_window = min_window
        self._smc_detector = SMCDetector()
        self._fee_pct = fee_pct
        self._slippage_pct = slippage_pct
        self._funding_rate_per_8h = funding_rate_per_8h
        self._max_concurrent_positions = max_concurrent_positions

    def run(
        self,
        candles_by_tf: Dict[str, Dict],
        symbol: str = "BTCUSDT",
        channel_name: Optional[str] = None,
        spread_pct: float = 0.01,
        volume_24h_usd: float = 10_000_000.0,
        simulated_ai_score: float = 0.0,
    ) -> List[BacktestResult]:
        """Run backtest across all (or one) channel(s).

        Parameters
        ----------
        candles_by_tf:
            Dict mapping timeframe → OHLCV dict with numpy arrays.
            E.g. ``{"5m": {"high": ..., "low": ..., "close": ..., ...}}``.
        symbol:
            Symbol name (used for SMC detection).
        channel_name:
            If provided, only backtest this specific channel.
        spread_pct:
            Simulated spread percentage.
        volume_24h_usd:
            Simulated 24h volume.
        simulated_ai_score:
            AI sentiment score passed to each channel evaluation, in the range
            ``[-1.0, 1.0]``.  Defaults to ``0.0`` (Neutral).

            **Note:** The backtester cannot replay historical AI sentiment data,
            so this value is the same for every candle window.  A value of
            ``0.0`` maps to ``score_ai_sentiment(0) ≈ 7.5/15``, which is a
            neutral mid-point — not zero.  To simulate pessimistic conditions
            (e.g. bearish news sentiment that would lower confidence in live
            trading), pass a negative value such as ``-0.5``.

        Returns
        -------
        List of :class:`BacktestResult`, one per channel tested.
        """
        channels = self._channels
        if channel_name:
            channels = [c for c in channels if c.config.name == channel_name]

        results: List[BacktestResult] = []
        for chan in channels:
            result = self._backtest_channel(
                chan, candles_by_tf, symbol, spread_pct, volume_24h_usd,
                simulated_ai_score,
            )
            results.append(result)
            log.info("Backtest %s: %s", chan.config.name, result.summary())

        return results

    def _backtest_channel(
        self,
        channel,
        candles_by_tf: Dict[str, Dict],
        symbol: str,
        spread_pct: float,
        volume_24h_usd: float,
        simulated_ai_score: float = 0.0,
    ) -> BacktestResult:
        """Run a single channel backtest across all candle windows."""
        result = BacktestResult(channel=channel.config.name, slippage_pct=self._slippage_pct)
        pnl_history: List[float] = []

        # Use the primary timeframe for the channel
        primary_tf = channel.config.timeframes[0]
        if primary_tf not in candles_by_tf:
            log.warning(
                "Backtest: timeframe %s not available for %s",
                primary_tf,
                channel.config.name,
            )
            return result

        primary_candles = candles_by_tf[primary_tf]
        total_candles = len(primary_candles.get("close", []))

        # Automatically apply 1-candle execution delay for SCALP channels to
        # simulate 0.5–3 s live-trading latency between signal detection and fill.
        is_scalp = any(channel.config.name.startswith(n) for n in _SCALP_CHANNEL_NAMES)
        execution_delay = 1 if is_scalp else 0

        # Derive a human-readable sentiment label from the numeric score so
        # channels that inspect the label field also behave consistently.
        if simulated_ai_score > _AI_BULLISH_THRESHOLD:
            ai_label = "Bullish"
        elif simulated_ai_score < _AI_BEARISH_THRESHOLD:
            ai_label = "Bearish"
        else:
            ai_label = "Neutral"
        ai_insight = {"label": ai_label, "summary": "", "score": simulated_ai_score}

        # Track open simulated positions (candle index when each trade closes)
        open_positions: List[int] = []

        for i in range(self._min_window, total_candles - self._lookahead):
            # Enforce max_concurrent_positions: prune positions that have closed
            open_positions = [end for end in open_positions if end > i]
            if len(open_positions) >= self._max_concurrent_positions:
                continue

            # Slice candles up to index i for the evaluation window
            window: Dict[str, Dict] = {}
            for tf, cd in candles_by_tf.items():
                window[tf] = {
                    k: v[:i] if hasattr(v, "__getitem__") else v
                    for k, v in cd.items()
                }

            # Compute indicators for each timeframe
            indicators: Dict[str, Dict] = {}
            for tf, cd in window.items():
                indicators[tf] = _compute_indicators(cd)

            # SMC detection
            smc_result = self._smc_detector.detect(symbol, window, [])
            smc_data = smc_result.as_dict()

            try:
                sig = channel.evaluate(
                    symbol=symbol,
                    candles=window,
                    indicators=indicators,
                    smc_data=smc_data,
                    ai_insight=ai_insight,
                    spread_pct=spread_pct,
                    volume_24h_usd=volume_24h_usd,
                )
            except Exception as exc:
                log.debug("Channel eval error at candle %d: %s", i, exc)
                continue

            if sig is None:
                continue

            # Simulate against future candles
            future: Dict[str, np.ndarray] = {}
            for k, v in primary_candles.items():
                if hasattr(v, "__getitem__"):
                    future[k] = v[i: i + self._lookahead]
            won, pnl, tp_level = _simulate_trade(
                sig, future,
                fee_pct=self._fee_pct,
                slippage_pct=self._slippage_pct,
                funding_rate_8h=self._funding_rate_per_8h,
                execution_delay_candles=execution_delay,
            )

            # Record approximate trade close candle for capacity tracking
            open_positions.append(i + self._lookahead)

            result.total_signals += 1
            if won:
                result.wins += 1
                # Count partial wins (TP1 or TP2 hit before final exit)
                if tp_level >= 1:
                    result.partial_wins += 1
            else:
                result.losses += 1
            pnl_history.append(pnl)
            result.signal_details.append({
                "candle_index": i,
                "direction": sig.direction.value,
                "entry": sig.entry,
                "won": won,
                "pnl_pct": round(pnl, 4),
                "tp_level": tp_level,
            })

        # Aggregate statistics
        if result.total_signals > 0:
            total = result.wins + result.losses
            result.win_rate = result.wins / total * 100.0 if total > 0 else 0.0
            result.total_pnl_pct = sum(pnl_history)
            # Proper avg R:R: average win / average loss magnitude
            if result.wins > 0 and result.losses > 0:
                avg_win = sum(p for p in pnl_history if p > 0) / result.wins
                avg_loss = sum(abs(p) for p in pnl_history if p <= 0) / result.losses
                result.avg_rr = avg_win / avg_loss if avg_loss > 0 else 0.0
            elif result.wins > 0:
                result.avg_rr = float("inf")
            else:
                result.avg_rr = 0.0
            result.best_trade = max(pnl_history) if pnl_history else 0.0
            result.worst_trade = min(pnl_history) if pnl_history else 0.0

            # Max drawdown
            cum = 0.0
            peak = 0.0
            dd = 0.0
            for p in pnl_history:
                cum += p
                if cum > peak:
                    peak = cum
                drop = peak - cum
                if drop > dd:
                    dd = drop
            result.max_drawdown = dd

        return result
