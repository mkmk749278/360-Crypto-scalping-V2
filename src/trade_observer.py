"""AI Trade Observer – captures full trade lifecycle data and generates
periodic AI-powered digests for the admin Telegram channel.

The observer runs as an async background task (similar to MacroWatchdog) and
hooks into 3 existing touchpoints:

1. **Signal published** → ``capture_entry_snapshot(signal)`` — called after the
   signal is successfully delivered to Telegram in ``signal_router.py``.
2. **Trade poll loop** → ``observe_trade(signal, current_price)`` — called
   during each ``TradeMonitor._evaluate_signal()`` cycle.
3. **Trade completes** → ``capture_exit_analysis(signal, outcome, pnl_pct)`` —
   called when ``TradeMonitor._record_outcome()`` fires.

Every ``OBSERVER_DIGEST_INTERVAL_SECONDS`` (default 6 h) the observer collects
completed trade data, builds a structured prompt, asks GPT-4o-mini for root-
cause analysis, and posts the formatted digest to ``TELEGRAM_ADMIN_CHAT_ID``.

Configuration (via environment variables):
  OBSERVER_ENABLED                 – "true" to enable (default: "true")
  OBSERVER_POLL_INTERVAL           – seconds between observe_trade calls (60)
  OBSERVER_DIGEST_INTERVAL         – seconds between AI digests (21600 = 6 h)
  OBSERVER_DATA_PATH               – path for JSON persistence
  OBSERVER_MAX_OBSERVATIONS        – max mid-trade snapshots per signal (120)
  OBSERVER_DIGEST_LOOKBACK         – hours of history included in digests (24)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

import aiohttp

from config import (
    OBSERVER_DIGEST_INTERVAL_SECONDS,
    OBSERVER_DIGEST_LOOKBACK_HOURS,
    OBSERVER_ENABLED,
    OBSERVER_DATA_PATH,
    OBSERVER_MAX_OBSERVATIONS_PER_TRADE,
    OBSERVER_POLL_INTERVAL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TELEGRAM_ADMIN_CHAT_ID,
)
from src.channels.base import Signal
from src.utils import get_logger

log = get_logger("trade_observer")

# OpenAI chat completions endpoint (same as openai_evaluator.py)
_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_OPENAI_TIMEOUT = 30.0

# Rolling window: discard observations older than 7 days to cap file growth.
_ROLLING_WINDOW_SECONDS: float = 7 * 24 * 3600.0

# Root-cause labels used in exit analysis
ROOT_CAUSE_LABELS = frozenset({
    "btc_correlation",
    "regime_flip",
    "momentum_loss",
    "wick_hunt",
    "spread_blowout",
    "normal_sl",
    "tp_hit",
})


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EntrySnapshot:
    """Market state captured at the moment the signal is published."""

    signal_id: str
    symbol: str
    channel: str
    direction: str
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    confidence: float
    spread_pct: float
    regime: str                         # market regime at entry
    fear_greed_value: Optional[int]     # Fear & Greed index (0-100), None if unavailable
    btc_price: Optional[float]          # BTC price at entry, None if unavailable
    eth_price: Optional[float]          # ETH price at entry, None if unavailable
    order_book_imbalance: Optional[float]  # positive → buy-side heavy
    pre_signal_momentum: Optional[float]   # EMA slope at entry (% diff)
    setup_class: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class MidTradeObservation:
    """Snapshot taken every ~60 s while a trade is active."""

    signal_id: str
    elapsed_seconds: float              # seconds since signal was published
    current_price: float
    unrealized_pnl_pct: float
    mfe_pct: float                      # max favorable excursion so far
    mae_pct: float                      # max adverse excursion so far
    btc_price: Optional[float]          # BTC price at observation time
    btc_delta_pct: Optional[float]      # BTC % change since entry
    current_regime: str                 # regime at observation time
    regime_changed: bool                # True if different from entry regime
    momentum_trajectory: str           # "improving" | "degrading" | "stable"
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExitAnalysis:
    """Root-cause analysis recorded when a trade closes."""

    signal_id: str
    symbol: str
    channel: str
    direction: str
    outcome: str                        # "SL_HIT" | "TP1_HIT" | ... | "EXPIRED"
    pnl_pct: float
    hold_duration_seconds: float
    root_cause: str                     # one of ROOT_CAUSE_LABELS
    btc_change_pct: Optional[float]     # BTC % change over the trade lifetime
    regime_transitions: int             # # of regime changes during the trade
    reached_tp1_zone: bool              # True if price touched TP1 before reversing
    time_to_tp1_seconds: Optional[float]   # None if TP1 was never reached
    time_to_sl_seconds: Optional[float]    # None if SL was never hit
    mfe_pct: float
    mae_pct: float
    entry_price: float
    entry_spread_pct: float
    entry_regime: str
    num_observations: int               # mid-trade snapshot count
    timestamp: float = field(default_factory=time.time)


@dataclass
class TradeRecord:
    """Complete lifecycle record for one trade."""

    entry: EntrySnapshot
    observations: List[MidTradeObservation] = field(default_factory=list)
    exit: Optional[ExitAnalysis] = None
    complete: bool = False              # True once exit is captured


# ---------------------------------------------------------------------------
# TradeObserver
# ---------------------------------------------------------------------------


class TradeObserver:
    """Async background service that observes trade lifecycles and produces
    AI-powered digests.

    Parameters
    ----------
    send_alert:
        Async callable that sends a string to the admin Telegram channel.
        Typically ``TelegramBot.send_admin_alert``.
    data_store:
        Optional :class:`src.historical_data.HistoricalDataStore` — used for
        candle lookbacks and BTC/ETH price access.
    regime_detector:
        Optional :class:`src.regime.MarketRegimeDetector` — used to classify
        the current regime during observations.
    """

    def __init__(
        self,
        send_alert: Callable[[str], Coroutine[Any, Any, bool]],
        data_store: Optional[Any] = None,
        regime_detector: Optional[Any] = None,
        data_path: Optional[str] = None,
    ) -> None:
        self._send_alert = send_alert
        self._data_store = data_store
        self._regime_detector = regime_detector
        # Allow the storage path to be overridden (useful in tests)
        self._data_path: str = data_path if data_path is not None else OBSERVER_DATA_PATH

        # In-memory registry: signal_id → TradeRecord
        self._records: Dict[str, TradeRecord] = {}

        # Completed records kept for digest building
        self._completed: List[TradeRecord] = []

        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

        # Load any persisted observations from disk
        self._load()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background digest loop."""
        if not OBSERVER_ENABLED:
            log.info("TradeObserver disabled by configuration – skipping start")
            return
        if self._task is not None and not self._task.done():
            log.debug("TradeObserver already running")
            return
        self._task = asyncio.create_task(self._digest_loop(), name="trade_observer")
        log.info(
            "TradeObserver started (digest_interval={}s, lookback={}h)",
            OBSERVER_DIGEST_INTERVAL_SECONDS,
            OBSERVER_DIGEST_LOOKBACK_HOURS,
        )

    async def stop(self) -> None:
        """Cancel the digest loop and close the HTTP session."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        log.info("TradeObserver stopped")

    # ------------------------------------------------------------------
    # Public hook methods (called from signal_router / trade_monitor)
    # ------------------------------------------------------------------

    def capture_entry_snapshot(self, signal: Signal) -> None:
        """Record the market state at the moment a signal is published.

        Called from ``signal_router.py`` immediately after confirmed delivery.
        Fail-open: any error is caught and logged so it never blocks signals.
        """
        if not OBSERVER_ENABLED:
            return
        try:
            self._capture_entry_snapshot_inner(signal)
        except Exception as exc:
            log.debug("TradeObserver.capture_entry_snapshot failed (non-critical): {}", exc)

    def observe_trade(self, signal: Signal, current_price: float) -> None:
        """Record a mid-trade snapshot.

        Called from ``TradeMonitor._evaluate_signal()`` on every poll cycle.
        Fail-open: any error is caught and logged.
        """
        if not OBSERVER_ENABLED:
            return
        try:
            self._observe_trade_inner(signal, current_price)
        except Exception as exc:
            log.debug("TradeObserver.observe_trade failed (non-critical): {}", exc)

    def capture_exit_analysis(
        self, signal: Signal, outcome: str, pnl_pct: float
    ) -> None:
        """Classify the exit and record root-cause analysis.

        Called from ``TradeMonitor._record_outcome()``.
        Fail-open: any error is caught and logged.
        """
        if not OBSERVER_ENABLED:
            return
        try:
            self._capture_exit_analysis_inner(signal, outcome, pnl_pct)
        except Exception as exc:
            log.debug("TradeObserver.capture_exit_analysis failed (non-critical): {}", exc)

    # ------------------------------------------------------------------
    # Inner implementation helpers
    # ------------------------------------------------------------------

    def _capture_entry_snapshot_inner(self, signal: Signal) -> None:
        if signal.signal_id in self._records:
            return  # already captured

        btc_price = self._get_reference_price("BTCUSDT")
        eth_price = self._get_reference_price("ETHUSDT")
        regime = self._get_current_regime(signal.symbol)
        spread_pct = getattr(signal, "spread_pct", 0.0) or 0.0
        order_book_imbalance = getattr(signal, "order_book_imbalance", None)

        entry = EntrySnapshot(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            channel=signal.channel,
            direction=signal.direction.value,
            entry_price=signal.entry,
            stop_loss=signal.stop_loss,
            tp1=signal.tp1,
            tp2=signal.tp2,
            tp3=signal.tp3,
            confidence=signal.confidence,
            spread_pct=spread_pct,
            regime=regime,
            fear_greed_value=None,  # populated async by _fetch_fear_greed_async
            btc_price=btc_price,
            eth_price=eth_price,
            order_book_imbalance=order_book_imbalance,
            pre_signal_momentum=getattr(signal, "entry_momentum_slope", None),
            setup_class=getattr(signal, "setup_class", "") or "",
        )

        record = TradeRecord(entry=entry)
        self._records[signal.signal_id] = record
        log.debug(
            "TradeObserver: entry snapshot captured for {} {} (regime={})",
            signal.symbol, signal.signal_id[:8], regime,
        )

    def _observe_trade_inner(self, signal: Signal, current_price: float) -> None:
        record = self._records.get(signal.signal_id)
        if record is None or record.complete:
            return

        # Throttle: don't add more than one observation per OBSERVER_POLL_INTERVAL
        obs = record.observations
        if obs:
            last_ts = obs[-1].timestamp
            if time.time() - last_ts < OBSERVER_POLL_INTERVAL * 0.8:
                return

        # Respect max per-trade observation cap
        if len(obs) >= OBSERVER_MAX_OBSERVATIONS_PER_TRADE:
            return

        entry = record.entry
        elapsed = time.time() - entry.timestamp

        # BTC correlation
        btc_price_now = self._get_reference_price("BTCUSDT")
        btc_delta_pct: Optional[float] = None
        if btc_price_now is not None and entry.btc_price is not None and entry.btc_price != 0:
            btc_delta_pct = (btc_price_now - entry.btc_price) / entry.btc_price * 100.0

        # PnL
        if signal.direction.value == "LONG":
            unrealized_pnl_pct = (current_price - entry.entry_price) / entry.entry_price * 100.0
        else:
            unrealized_pnl_pct = (entry.entry_price - current_price) / entry.entry_price * 100.0

        # Regime
        current_regime = self._get_current_regime(signal.symbol)
        regime_changed = current_regime != entry.regime and bool(current_regime) and bool(entry.regime)

        # Momentum trajectory from the signal's running MFE/MAE
        mfe = signal.max_favorable_excursion_pct
        mae = signal.max_adverse_excursion_pct
        prev_pnl = obs[-1].unrealized_pnl_pct if obs else 0.0
        if unrealized_pnl_pct > prev_pnl + 0.05:
            trajectory = "improving"
        elif unrealized_pnl_pct < prev_pnl - 0.05:
            trajectory = "degrading"
        else:
            trajectory = "stable"

        observation = MidTradeObservation(
            signal_id=signal.signal_id,
            elapsed_seconds=elapsed,
            current_price=current_price,
            unrealized_pnl_pct=unrealized_pnl_pct,
            mfe_pct=mfe,
            mae_pct=mae,
            btc_price=btc_price_now,
            btc_delta_pct=btc_delta_pct,
            current_regime=current_regime,
            regime_changed=regime_changed,
            momentum_trajectory=trajectory,
        )
        obs.append(observation)

    def _capture_exit_analysis_inner(
        self, signal: Signal, outcome: str, pnl_pct: float
    ) -> None:
        record = self._records.get(signal.signal_id)
        if record is None or record.complete:
            return

        entry = record.entry
        obs = record.observations
        hold_duration = time.time() - entry.timestamp

        # BTC change over trade lifetime
        btc_price_now = self._get_reference_price("BTCUSDT")
        btc_change_pct: Optional[float] = None
        if btc_price_now is not None and entry.btc_price is not None and entry.btc_price != 0:
            btc_change_pct = (btc_price_now - entry.btc_price) / entry.btc_price * 100.0

        # Count regime transitions
        regime_transitions = sum(1 for o in obs if o.regime_changed)

        # Did the price ever reach the TP1 zone?
        tp1 = entry.tp1
        entry_p = entry.entry_price
        reached_tp1_zone = False
        time_to_tp1_seconds: Optional[float] = None
        time_to_sl_seconds: Optional[float] = None

        if entry.direction == "LONG":
            for o in obs:
                if o.current_price >= tp1:
                    reached_tp1_zone = True
                    time_to_tp1_seconds = o.elapsed_seconds
                    break
        else:
            for o in obs:
                if o.current_price <= tp1:
                    reached_tp1_zone = True
                    time_to_tp1_seconds = o.elapsed_seconds
                    break

        if "SL" in outcome or outcome == "EXPIRED":
            time_to_sl_seconds = hold_duration

        # Root-cause classification
        root_cause = self._classify_root_cause(
            outcome=outcome,
            pnl_pct=pnl_pct,
            btc_change_pct=btc_change_pct,
            regime_transitions=regime_transitions,
            entry_spread_pct=entry.spread_pct,
            observations=obs,
        )

        exit_analysis = ExitAnalysis(
            signal_id=signal.signal_id,
            symbol=entry.symbol,
            channel=entry.channel,
            direction=entry.direction,
            outcome=outcome,
            pnl_pct=pnl_pct,
            hold_duration_seconds=hold_duration,
            root_cause=root_cause,
            btc_change_pct=btc_change_pct,
            regime_transitions=regime_transitions,
            reached_tp1_zone=reached_tp1_zone,
            time_to_tp1_seconds=time_to_tp1_seconds,
            time_to_sl_seconds=time_to_sl_seconds,
            mfe_pct=signal.max_favorable_excursion_pct,
            mae_pct=signal.max_adverse_excursion_pct,
            entry_price=entry_p,
            entry_spread_pct=entry.spread_pct,
            entry_regime=entry.regime,
            num_observations=len(obs),
        )

        record.exit = exit_analysis
        record.complete = True

        # Move to completed list for digest building
        self._completed.append(record)

        # Prune stale completed records (rolling 7-day window)
        self._prune_completed()

        # Persist (best-effort, using a running loop if available)
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon(self._save)
        except RuntimeError:
            # No running event loop — save synchronously (e.g. in tests)
            self._save()

        log.debug(
            "TradeObserver: exit captured for {} {} – outcome={} root_cause={}",
            entry.symbol, signal.signal_id[:8], outcome, root_cause,
        )

    # ------------------------------------------------------------------
    # Root-cause classifier
    # ------------------------------------------------------------------

    def _classify_root_cause(
        self,
        outcome: str,
        pnl_pct: float,
        btc_change_pct: Optional[float],
        regime_transitions: int,
        entry_spread_pct: float,
        observations: List[MidTradeObservation],
    ) -> str:
        """Heuristic root-cause classification for SL hits.

        Returns one of the labels in ROOT_CAUSE_LABELS.
        """
        # Winning outcomes
        if "TP" in outcome:
            return "tp_hit"

        # Spread blowout: very wide spread at entry suggests illiquid market
        if entry_spread_pct > 0.5:
            return "spread_blowout"

        # BTC correlation: large BTC move against trade direction
        if btc_change_pct is not None and abs(btc_change_pct) > 1.5:
            return "btc_correlation"

        # Regime flip: regime changed at least once during the trade
        if regime_transitions >= 1:
            return "regime_flip"

        # Wick hunt: MAE significantly exceeded the final loss (price spiked
        # through SL then came back) — use the ratio of MAE to final PnL
        if observations:
            final_mae = min(o.mae_pct for o in observations) if observations else pnl_pct
            if pnl_pct != 0 and final_mae != 0 and final_mae / pnl_pct > 1.5:
                return "wick_hunt"

        # Momentum loss: majority of observations showed degrading momentum
        if observations:
            degrading = sum(1 for o in observations if o.momentum_trajectory == "degrading")
            if len(observations) > 3 and degrading / len(observations) > 0.6:
                return "momentum_loss"

        return "normal_sl"

    # ------------------------------------------------------------------
    # Digest loop
    # ------------------------------------------------------------------

    async def _digest_loop(self) -> None:
        """Infinite loop: sleep → build digest → send to admin."""
        while True:
            try:
                await asyncio.sleep(OBSERVER_DIGEST_INTERVAL_SECONDS)
                await self._run_digest()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("TradeObserver digest loop error: {}", exc)

    async def run_digest_on_demand(self, lookback_hours: Optional[int] = None) -> str:
        """Build an AI digest and return it as a string (does not send).

        Parameters
        ----------
        lookback_hours:
            Number of hours of completed trades to include.  Defaults to
            ``OBSERVER_DIGEST_LOOKBACK_HOURS`` when ``None``.

        Returns
        -------
        str
            A formatted digest message, or a "no trades" notice.
        """
        hours = lookback_hours if lookback_hours is not None else OBSERVER_DIGEST_LOOKBACK_HOURS
        cutoff = time.time() - hours * 3600.0
        window = [r for r in self._completed if r.exit is not None and r.exit.timestamp >= cutoff]

        if not window:
            return f"ℹ️ No completed trades in the last {hours}h — nothing to analyse."

        prompt = self._build_digest_prompt(window)
        ai_text = await self._call_openai(prompt)
        return self._format_digest_message(window, ai_text)

    async def _run_digest(self) -> None:
        """Build and send one AI digest covering the last OBSERVER_DIGEST_LOOKBACK_HOURS."""
        if not TELEGRAM_ADMIN_CHAT_ID:
            log.debug("TradeObserver: TELEGRAM_ADMIN_CHAT_ID not configured – skipping digest")
            return

        cutoff = time.time() - OBSERVER_DIGEST_LOOKBACK_HOURS * 3600.0
        window = [r for r in self._completed if r.exit is not None and r.exit.timestamp >= cutoff]

        if not window:
            log.debug("TradeObserver: no completed trades in the last {}h – skipping digest", OBSERVER_DIGEST_LOOKBACK_HOURS)
            return

        prompt = self._build_digest_prompt(window)
        ai_text = await self._call_openai(prompt)
        message = self._format_digest_message(window, ai_text)

        try:
            await self._send_alert(message)
            log.info(
                "TradeObserver: AI digest sent to admin ({} trades analysed)",
                len(window),
            )
        except Exception as exc:
            log.warning("TradeObserver: failed to send digest alert: {}", exc)

    # ------------------------------------------------------------------
    # AI digest helpers
    # ------------------------------------------------------------------

    def _build_digest_prompt(self, records: List[TradeRecord]) -> str:
        """Build a structured GPT prompt from completed trade records."""
        lines: List[str] = [
            "You are an expert crypto trading analyst. Analyse the following completed trades "
            "and provide actionable insights.\n",
            f"Period: last {OBSERVER_DIGEST_LOOKBACK_HOURS} hours",
            f"Total trades: {len(records)}\n",
        ]

        wins = [r for r in records if r.exit and "TP" in r.exit.outcome]
        losses = [r for r in records if r.exit and ("SL" in r.exit.outcome or r.exit.pnl_pct < 0)]

        lines.append(f"Wins: {len(wins)}, Losses: {len(losses)}")

        # Root cause breakdown
        from collections import Counter
        root_causes = Counter(r.exit.root_cause for r in records if r.exit)
        lines.append("\nRoot cause breakdown:")
        for cause, count in root_causes.most_common():
            lines.append(f"  {cause}: {count}")

        # Per-channel summary
        channels: dict = {}
        for r in records:
            if r.exit is None:
                continue
            ch = r.exit.channel
            if ch not in channels:
                channels[ch] = {"wins": 0, "losses": 0, "pnl": []}
            if "TP" in r.exit.outcome:
                channels[ch]["wins"] += 1
            else:
                channels[ch]["losses"] += 1
            channels[ch]["pnl"].append(r.exit.pnl_pct)

        lines.append("\nChannel breakdown:")
        for ch, stats in channels.items():
            avg_pnl = sum(stats["pnl"]) / len(stats["pnl"]) if stats["pnl"] else 0.0
            lines.append(f"  {ch}: {stats['wins']}W/{stats['losses']}L, avg_pnl={avg_pnl:.2f}%")

        # BTC correlation analysis
        btc_corr_losses = [r for r in losses if r.exit and r.exit.root_cause == "btc_correlation"]
        lines.append(f"\nBTC correlation losses: {len(btc_corr_losses)}")

        # Sample trade details (up to 10 most recent)
        lines.append("\nRecent trade details (up to 10):")
        for r in records[-10:]:
            if r.exit is None:
                continue
            e = r.exit
            detail = (
                f"  {e.symbol} {e.direction} {e.channel}: outcome={e.outcome} "
                f"pnl={e.pnl_pct:.2f}% root_cause={e.root_cause} "
                f"hold={e.hold_duration_seconds:.0f}s regime_flips={e.regime_transitions}"
            )
            if e.btc_change_pct is not None:
                detail += f" btc_chg={e.btc_change_pct:.1f}%"
            lines.append(detail)

        lines.append(
            "\nRespond with a JSON object with these exact keys:\n"
            '{\n'
            '  "summary": "<2-3 sentence overview>",\n'
            '  "top_root_causes": ["<cause1>", "<cause2>", "<cause3>"],\n'
            '  "btc_correlation_note": "<1-2 sentences>",\n'
            '  "best_channel": "<channel name>",\n'
            '  "worst_channel": "<channel name>",\n'
            '  "recommendations": ["<rec1>", "<rec2>", "<rec3>"]\n'
            '}'
        )
        return "\n".join(lines)

    async def _call_openai(self, prompt: str) -> Optional[str]:
        """POST to OpenAI and return the raw response text, or None on failure."""
        if not OPENAI_API_KEY:
            log.debug("TradeObserver: OPENAI_API_KEY not set – skipping AI digest")
            return None

        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional crypto trading analyst. "
                        "Analyse trade lifecycle data and respond only with valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 600,
        }
        try:
            timeout = aiohttp.ClientTimeout(total=_OPENAI_TIMEOUT)
            async with session.post(_OPENAI_CHAT_URL, headers=headers, json=body, timeout=timeout) as resp:
                if resp.status != 200:
                    log.warning("TradeObserver OpenAI returned status {}", resp.status)
                    return None
                data = await resp.json(content_type=None)
            return str(data["choices"][0]["message"]["content"])
        except Exception as exc:
            log.warning("TradeObserver OpenAI call failed: {}", exc)
            return None

    def _format_digest_message(
        self, records: List[TradeRecord], ai_text: Optional[str]
    ) -> str:
        """Build the Telegram message for the admin digest."""
        wins = sum(1 for r in records if r.exit and "TP" in r.exit.outcome)
        losses = len(records) - wins
        win_rate = wins / len(records) * 100.0 if records else 0.0
        avg_pnl = (
            sum(r.exit.pnl_pct for r in records if r.exit) / len(records)
            if records else 0.0
        )

        header = (
            f"🤖 *AI Trade Observer Digest*\n"
            f"_{OBSERVER_DIGEST_LOOKBACK_HOURS}h window · {len(records)} trades analysed_\n\n"
            f"📊 *Results:* {wins}W / {losses}L ({win_rate:.0f}% win rate)\n"
            f"💰 *Avg PnL:* {avg_pnl:+.2f}%\n\n"
        )

        if ai_text:
            try:
                parsed = json.loads(ai_text)
                summary = parsed.get("summary", "")
                top_causes = parsed.get("top_root_causes", [])
                btc_note = parsed.get("btc_correlation_note", "")
                best_ch = parsed.get("best_channel", "")
                worst_ch = parsed.get("worst_channel", "")
                recs = parsed.get("recommendations", [])

                body_parts = []
                if summary:
                    body_parts.append(f"📝 *Analysis:*\n{summary}")
                if top_causes:
                    causes_str = " → ".join(f"`{c}`" for c in top_causes[:3])
                    body_parts.append(f"🔍 *Top SL causes:* {causes_str}")
                if btc_note:
                    body_parts.append(f"₿ *BTC correlation:* {btc_note}")
                if best_ch and worst_ch:
                    body_parts.append(
                        f"🏆 *Best channel:* {best_ch}\n"
                        f"⚠️ *Worst channel:* {worst_ch}"
                    )
                if recs:
                    rec_lines = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(recs[:3]))
                    body_parts.append(f"✅ *Recommendations:*\n{rec_lines}")

                body = "\n\n".join(body_parts)
            except (json.JSONDecodeError, KeyError):
                # Fall back to raw AI text if JSON parsing fails
                body = f"📝 *AI Analysis:*\n{ai_text[:800]}"
        else:
            # No AI response — produce a pure statistical digest
            from collections import Counter
            causes = Counter(r.exit.root_cause for r in records if r.exit)
            top = "\n".join(f"  • `{c}`: {n}" for c, n in causes.most_common(3))
            body = f"🔍 *Top SL root causes:*\n{top}"

        return header + body

    # ------------------------------------------------------------------
    # Price / regime helpers
    # ------------------------------------------------------------------

    def _get_reference_price(self, symbol: str) -> Optional[float]:
        """Return the latest close price for *symbol* from the data store."""
        if self._data_store is None:
            return None
        try:
            candles = self._data_store.get_candles(symbol, "1m")
            if candles and "close" in candles and candles["close"]:
                return float(candles["close"][-1])
        except Exception:
            pass
        return None

    def _get_current_regime(self, symbol: str) -> str:
        """Return a string description of the current market regime."""
        if self._regime_detector is None:
            return ""
        try:
            candles = None
            if self._data_store is not None:
                candles = self._data_store.get_candles(symbol, "1h")
            if candles is None:
                return ""
            result = self._regime_detector.detect(candles)
            if result is None:
                return ""
            return str(getattr(result, "regime", result))
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist completed records to disk (best-effort)."""
        try:
            path = Path(self._data_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = [self._record_to_dict(r) for r in self._completed if r.exit is not None]
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            log.debug("TradeObserver persist failed: {}", exc)

    def _load(self) -> None:
        """Load previously persisted records from disk (best-effort)."""
        try:
            path = Path(self._data_path)
            if not path.exists():
                return
            raw = json.loads(path.read_text(encoding="utf-8"))
            for item in raw:
                record = self._record_from_dict(item)
                if record is not None:
                    self._completed.append(record)
            self._prune_completed()
            log.debug("TradeObserver: loaded {} completed records from disk", len(self._completed))
        except Exception as exc:
            log.debug("TradeObserver load failed (non-critical): {}", exc)

    def _prune_completed(self) -> None:
        """Discard completed records older than the rolling window."""
        cutoff = time.time() - _ROLLING_WINDOW_SECONDS
        self._completed = [r for r in self._completed if r.exit is None or r.exit.timestamp >= cutoff]

    @staticmethod
    def _record_to_dict(record: TradeRecord) -> dict:
        return {
            "entry": asdict(record.entry),
            "observations": [asdict(o) for o in record.observations],
            "exit": asdict(record.exit) if record.exit else None,
            "complete": record.complete,
        }

    @staticmethod
    def _record_from_dict(data: dict) -> Optional[TradeRecord]:
        try:
            entry = EntrySnapshot(**data["entry"])
            observations = [MidTradeObservation(**o) for o in data.get("observations", [])]
            exit_data = data.get("exit")
            exit_analysis = ExitAnalysis(**exit_data) if exit_data else None
            return TradeRecord(
                entry=entry,
                observations=observations,
                exit=exit_analysis,
                complete=data.get("complete", False),
            )
        except Exception:
            return None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
