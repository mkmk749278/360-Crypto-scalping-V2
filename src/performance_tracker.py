"""Performance Tracker – records and analyses completed signal outcomes.

Persists data to ``data/signal_performance.json`` and exposes per-channel
stats with rolling 7-day and 30-day windows.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.performance_metrics import (
    calculate_drawdown_metrics,
    classify_trade_outcome,
    is_breakeven_pnl,
    normalize_pnl_pct,
)
from src.utils import get_logger

log = get_logger("performance_tracker")

_DEFAULT_STORAGE_PATH = "data/signal_performance.json"


@dataclass
class SignalRecord:
    """A single completed signal record."""

    signal_id: str
    channel: str
    symbol: str
    direction: str
    entry: float
    hit_tp: int        # 0 = none, 1 = TP1, 2 = TP2, 3 = TP3
    hit_sl: bool
    pnl_pct: float
    confidence: float
    outcome_label: str = ""
    pre_ai_confidence: float = 0.0
    post_ai_confidence: float = 0.0
    setup_class: str = ""
    market_phase: str = ""
    quality_tier: str = ""
    spread_pct: float = 0.0
    volume_24h_usd: float = 0.0
    hold_duration_sec: float = 0.0
    max_favorable_excursion_pct: float = 0.0
    max_adverse_excursion_pct: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChannelStats:
    """Aggregated statistics for a channel."""

    channel: str
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    total_signals: int = 0


class PerformanceTracker:
    """Records completed signal outcomes and computes performance statistics.

    Parameters
    ----------
    storage_path:
        Path to the JSON file used for persistence.
    """

    def __init__(self, storage_path: str = _DEFAULT_STORAGE_PATH) -> None:
        self._path = Path(storage_path)
        self._records: List[SignalRecord] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        signal_id: str,
        channel: str,
        symbol: str,
        direction: str,
        entry: float,
        hit_tp: int,
        hit_sl: bool,
        pnl_pct: float,
        outcome_label: str = "",
        confidence: float = 0.0,
        pre_ai_confidence: float = 0.0,
        post_ai_confidence: float = 0.0,
        setup_class: str = "",
        market_phase: str = "",
        quality_tier: str = "",
        spread_pct: float = 0.0,
        volume_24h_usd: float = 0.0,
        hold_duration_sec: float = 0.0,
        max_favorable_excursion_pct: float = 0.0,
        max_adverse_excursion_pct: float = 0.0,
    ) -> None:
        """Record the outcome of a completed signal."""
        record = SignalRecord(
            signal_id=signal_id,
            channel=channel,
            symbol=symbol,
            direction=direction,
            entry=entry,
            hit_tp=hit_tp,
            hit_sl=hit_sl,
            pnl_pct=normalize_pnl_pct(pnl_pct),
            outcome_label=(
                outcome_label
                or classify_trade_outcome(pnl_pct=pnl_pct, hit_tp=hit_tp, hit_sl=hit_sl)
            ),
            confidence=confidence,
            pre_ai_confidence=pre_ai_confidence,
            post_ai_confidence=post_ai_confidence,
            setup_class=setup_class,
            market_phase=market_phase,
            quality_tier=quality_tier,
            spread_pct=spread_pct,
            volume_24h_usd=volume_24h_usd,
            hold_duration_sec=hold_duration_sec,
            max_favorable_excursion_pct=max_favorable_excursion_pct,
            max_adverse_excursion_pct=max_adverse_excursion_pct,
        )
        self._records.append(record)
        self._save()
        log.debug(
            "Recorded outcome for %s: pnl=%.2f%% hit_sl=%s",
            signal_id,
            pnl_pct,
            hit_sl,
        )

    def get_stats(
        self,
        channel: Optional[str] = None,
        window_days: Optional[int] = None,
    ) -> ChannelStats:
        """Compute stats for a channel (or all channels if channel is None).

        Parameters
        ----------
        channel:
            Filter to a specific channel name.  Pass ``None`` for global stats.
        window_days:
            Rolling window in days.  Pass ``None`` for all-time stats.
        """
        records = self._filter(channel=channel, window_days=window_days)
        return self._compute_stats(channel or "ALL", records)

    def format_stats_message(
        self,
        channel: Optional[str] = None,
        window_days: Optional[int] = None,
    ) -> str:
        """Return a Telegram-ready performance summary.

        Parameters
        ----------
        channel:
            Optional channel name filter.
        window_days:
            Optional rolling window (7 or 30 days).
        """
        label = channel or "All Channels"
        window_label = f" (last {window_days}d)" if window_days else " (all time)"
        stats = self.get_stats(channel=channel, window_days=window_days)

        return (
            f"📊 *Performance Stats – {label}{window_label}*\n"
            f"Total signals: {stats.total_signals}\n"
            f"Wins: {stats.win_count} | Losses: {stats.loss_count} | "
            f"Breakeven: {stats.breakeven_count}\n"
            f"Win rate: {stats.win_rate:.1f}%\n"
            f"Avg PnL: {stats.avg_pnl_pct:+.2f}%\n"
            f"Best trade: {stats.best_trade:+.2f}%\n"
            f"Worst trade: {stats.worst_trade:+.2f}%\n"
            f"Max drawdown: {stats.max_drawdown:.2f}%"
        )

    def all_channel_stats(self, window_days: Optional[int] = None) -> Dict[str, ChannelStats]:
        """Return a dict of channel → ChannelStats."""
        channels = {r.channel for r in self._records}
        result: Dict[str, ChannelStats] = {}
        for ch in channels:
            result[ch] = self.get_stats(channel=ch, window_days=window_days)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _filter(
        self,
        channel: Optional[str] = None,
        window_days: Optional[int] = None,
    ) -> List[SignalRecord]:
        """Return records filtered by channel and/or time window."""
        records = self._records
        if channel:
            records = [r for r in records if r.channel == channel]
        if window_days:
            cutoff = time.time() - window_days * 86_400.0
            records = [r for r in records if r.timestamp >= cutoff]
        return records

    @staticmethod
    def _compute_stats(channel: str, records: List[SignalRecord]) -> ChannelStats:
        """Compute aggregate stats from a list of records."""
        stats = ChannelStats(channel=channel)
        if not records:
            return stats

        stats.total_signals = len(records)
        for record in records:
            if is_breakeven_pnl(record.pnl_pct):
                stats.breakeven_count += 1
            elif record.pnl_pct > 0:
                stats.win_count += 1
            else:
                stats.loss_count += 1
        total = stats.win_count + stats.loss_count
        stats.win_rate = (stats.win_count / total * 100.0) if total > 0 else 0.0

        pnls = [r.pnl_pct for r in records]
        stats.avg_pnl_pct = sum(pnls) / len(pnls) if pnls else 0.0
        stats.best_trade = max(pnls) if pnls else 0.0
        stats.worst_trade = min(pnls) if pnls else 0.0

        _, stats.max_drawdown = calculate_drawdown_metrics(pnls)

        return stats

    def _save(self) -> None:
        """Persist records to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(r) for r in self._records]
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            log.warning("Failed to save performance data: %s", exc)

    def _load(self) -> None:
        """Load records from disk if the file exists."""
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data: List[Dict[str, Any]] = json.load(fh)
            self._records = [SignalRecord(**item) for item in data]
            for record in self._records:
                record.pnl_pct = normalize_pnl_pct(record.pnl_pct)
                if not record.outcome_label:
                    record.outcome_label = classify_trade_outcome(
                        pnl_pct=record.pnl_pct,
                        hit_tp=record.hit_tp,
                        hit_sl=record.hit_sl,
                    )
            log.info("Loaded %d performance records from %s", len(self._records), self._path)
        except Exception as exc:
            log.warning("Failed to load performance data: %s", exc)
