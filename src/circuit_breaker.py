"""Circuit Breaker for rapid-loss protection."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Optional

from src.performance_metrics import calculate_drawdown_metrics, normalize_pnl_pct
from src.utils import get_logger

log = get_logger("circuit_breaker")

# Default thresholds (overridable via constructor)
_DEFAULT_MAX_CONSECUTIVE_SL: int = 3
_DEFAULT_MAX_HOURLY_SL: int = 5
_DEFAULT_MAX_DAILY_DRAWDOWN_PCT: float = 10.0
_DEFAULT_COOLDOWN_SECONDS: int = 900
_DEFAULT_PER_SYMBOL_MAX_SL: int = 3
_DEFAULT_PER_SYMBOL_COOLDOWN_SECONDS: int = 3600
_HOURLY_WINDOW_SECONDS: float = 3600.0
_DAILY_WINDOW_SECONDS: float = 86_400.0


@dataclass
class OutcomeRecord:
    """A single recorded signal outcome."""

    signal_id: str
    hit_sl: bool
    pnl_pct: float
    timestamp: float = field(default_factory=time.monotonic)


class CircuitBreaker:
    """Tracks signal outcomes and trips when loss thresholds are exceeded.

    Parameters
    ----------
    max_consecutive_sl:
        Maximum consecutive SL hits before tripping.
    max_hourly_sl:
        Maximum SL hits within a rolling 1-hour window before tripping.
    max_daily_drawdown_pct:
        Maximum cumulative PnL loss (%) within a rolling 24-hour window
        before tripping.
    alert_callback:
        Optional async callable that receives a message string.  Used to
        send Telegram alerts when the circuit trips.
    """

    def __init__(
        self,
        max_consecutive_sl: int = _DEFAULT_MAX_CONSECUTIVE_SL,
        max_hourly_sl: int = _DEFAULT_MAX_HOURLY_SL,
        max_daily_drawdown_pct: float = _DEFAULT_MAX_DAILY_DRAWDOWN_PCT,
        cooldown_seconds: int = _DEFAULT_COOLDOWN_SECONDS,
        per_symbol_max_sl: int = _DEFAULT_PER_SYMBOL_MAX_SL,
        per_symbol_cooldown_seconds: int = _DEFAULT_PER_SYMBOL_COOLDOWN_SECONDS,
        alert_callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.max_consecutive_sl = max_consecutive_sl
        self.max_hourly_sl = max_hourly_sl
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.cooldown_seconds = cooldown_seconds
        self.per_symbol_max_sl = per_symbol_max_sl
        self.per_symbol_cooldown_seconds = per_symbol_cooldown_seconds
        self._alert_callback = alert_callback

        # Rolling outcome history (keep last 1000 entries)
        self._outcomes: Deque[OutcomeRecord] = deque(maxlen=1000)

        self._tripped: bool = False
        self._trip_reason: str = ""
        self._trip_time: Optional[float] = None
        self._consecutive_sl: int = 0
        self._status_mode: str = "healthy"
        self._last_resume_time: Optional[float] = None
        self._last_resume_reason: str = ""
        self._monitoring_started_at: float = time.monotonic()

        # Per-symbol consecutive SL counters and cooldown expiry times.
        # After per_symbol_max_sl consecutive SL hits on the same symbol, that
        # symbol is suppressed for per_symbol_cooldown_seconds.
        self._per_symbol_consecutive_sl: Dict[str, int] = {}
        self._per_symbol_tripped_until: Dict[str, float] = {}


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        signal_id: str,
        hit_sl: bool,
        pnl_pct: float,
        symbol: Optional[str] = None,
    ) -> None:
        """Record the outcome of a completed signal.

        Parameters
        ----------
        signal_id:
            Unique identifier of the signal.
        hit_sl:
            ``True`` if the stop-loss was triggered (a loss).
        pnl_pct:
            Realized PnL as a percentage (negative for losses).
        symbol:
            Optional trading pair symbol.  When provided, consecutive SL hits
            on the same symbol are tracked independently and can trigger a
            per-symbol suppression even if the global breaker has not tripped.
        """
        self._prune_outcomes()
        normalized_pnl = normalize_pnl_pct(pnl_pct)
        loss_hit = hit_sl and normalized_pnl < 0.0
        record = OutcomeRecord(
            signal_id=signal_id,
            hit_sl=loss_hit,
            pnl_pct=normalized_pnl,
            timestamp=time.monotonic(),
        )
        self._outcomes.append(record)

        if loss_hit:
            self._consecutive_sl += 1
            if symbol:
                self._per_symbol_consecutive_sl[symbol] = (
                    self._per_symbol_consecutive_sl.get(symbol, 0) + 1
                )
                if self._per_symbol_consecutive_sl[symbol] >= self.per_symbol_max_sl:
                    expiry = time.monotonic() + self.per_symbol_cooldown_seconds
                    self._per_symbol_tripped_until[symbol] = expiry
                    log.warning(
                        "Per-symbol circuit breaker tripped for %s "
                        "(%d consecutive SL hits) – suppressed for %ds",
                        symbol,
                        self._per_symbol_consecutive_sl[symbol],
                        self.per_symbol_cooldown_seconds,
                    )
                    self._emit_alert(
                        f"⚠️ *Per-Symbol Circuit Breaker TRIPPED*\n"
                        f"Symbol: {symbol}\n"
                        f"Consecutive SL hits: {self._per_symbol_consecutive_sl[symbol]}\n"
                        f"Suppressed for: {self.per_symbol_cooldown_seconds}s"
                    )
        else:
            self._consecutive_sl = 0
            if symbol:
                self._per_symbol_consecutive_sl[symbol] = 0

        self._refresh_state()
        self._evaluate()

    def is_tripped(self) -> bool:
        """Return ``True`` when the circuit breaker is active."""
        self._refresh_state()
        return self._tripped

    def is_symbol_tripped(self, symbol: str) -> bool:
        """Return ``True`` when *symbol* is under a per-symbol suppression.

        The suppression expires automatically based on wall-clock time; no
        manual reset is required.
        """
        expiry = self._per_symbol_tripped_until.get(symbol)
        if expiry is None:
            return False
        if time.monotonic() < expiry:
            return True
        # Suppression expired – clean up
        del self._per_symbol_tripped_until[symbol]
        self._per_symbol_consecutive_sl[symbol] = 0
        return False

    def reset(self) -> None:
        """Manually reset the circuit breaker and clear all rolling state."""
        resumed_at = time.monotonic()
        self._outcomes.clear()
        self._tripped = False
        self._trip_reason = ""
        self._trip_time = None
        self._consecutive_sl = 0
        self._status_mode = "resumed"
        self._last_resume_time = resumed_at
        self._monitoring_started_at = resumed_at
        self._last_resume_reason = (
            "Manual reset cleared breaker history and restarted the monitoring window."
        )
        self._per_symbol_consecutive_sl.clear()
        self._per_symbol_tripped_until.clear()
        log.info("Circuit breaker reset manually and rolling history cleared.")

    def status_text(self) -> str:
        """Return a human-readable status string."""
        self._refresh_state()
        hourly = self._hourly_sl_count()
        daily_dd = self._daily_drawdown_pct()

        if self._status_mode == "cooldown":
            tripped_ago = (
                f"{time.monotonic() - self._trip_time:.0f}s ago"
                if self._trip_time is not None
                else "unknown"
            )
            return (
                f"⚠️ *Circuit Breaker TRIPPED* ({tripped_ago})\n"
                "State: Cooling down\n"
                f"Reason: {self._trip_reason}\n"
                f"Cooldown remaining: {self._cooldown_remaining():.0f}s\n"
                "Automatic resume is blocked until the cooldown period completes "
                "and rolling losses normalize."
            )
        if self._status_mode == "recovery_pending":
            return (
                "🟡 *Circuit Breaker AUTO-RESUME PENDING*\n"
                "State: Recovery pending\n"
                f"Trip reason: {self._trip_reason}\n"
                "Cooldown finished, but current rolling losses are still above safe limits.\n"
                f"Hourly SL hits in active window: {hourly}/{self.max_hourly_sl}\n"
                f"Daily drawdown in active window: "
                f"{daily_dd:.2f}% / {self.max_daily_drawdown_pct:.2f}%\n"
                "Signal generation stays paused until losses normalize or a manual reset is issued."
            )

        healthy_label = "✅ *Circuit Breaker: RESUMED & HEALTHY*" if self._last_resume_time else "✅ *Circuit Breaker: HEALTHY*"
        resume_line = ""
        if self._last_resume_time is not None and self._last_resume_reason:
            resume_line = (
                f"\nLast resume: {time.monotonic() - self._last_resume_time:.0f}s ago"
                f" ({self._last_resume_reason})"
            )
        return (
            f"{healthy_label}\n"
            "State: Resumed monitoring window active\n"
            f"Consecutive SL hits: {self._consecutive_sl}/{self.max_consecutive_sl}\n"
            f"Hourly SL hits in active window: {hourly}/{self.max_hourly_sl}\n"
            f"Daily drawdown in active window: "
            f"{daily_dd:.2f}% / {self.max_daily_drawdown_pct:.2f}%"
            f"{resume_line}"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate(self) -> None:
        """Check thresholds and trip if any are exceeded."""
        self._refresh_state()
        if self._tripped:
            return

        # 1. Consecutive SL hits
        if self._consecutive_sl >= self.max_consecutive_sl:
            self._trip(
                f"{self._consecutive_sl} consecutive SL hits "
                f"(max={self.max_consecutive_sl})"
            )
            return

        # 2. Hourly SL rate
        hourly = self._hourly_sl_count()
        if hourly >= self.max_hourly_sl:
            self._trip(
                f"{hourly} SL hits in the last hour "
                f"(max={self.max_hourly_sl})"
            )
            return

        # 3. Daily drawdown
        daily_dd = self._daily_drawdown_pct()
        if daily_dd >= self.max_daily_drawdown_pct:
            self._trip(
                f"Daily drawdown {daily_dd:.2f}% exceeded "
                f"threshold {self.max_daily_drawdown_pct}%"
            )

    def _trip(self, reason: str) -> None:
        """Set the tripped state and fire the optional alert."""
        self._tripped = True
        self._trip_reason = reason
        self._trip_time = time.monotonic()
        self._status_mode = "cooldown"
        log.warning("Circuit breaker TRIPPED: %s", reason)

        self._emit_alert(
            f"🚨 *Circuit Breaker TRIPPED*\n"
            f"Reason: {reason}\n"
            f"Cooldown: {self.cooldown_seconds}s\n"
            "Signal generation paused until cooldown and recovery checks pass."
        )

    def _refresh_state(self) -> None:
        """Advance cooldown/recovery state based on current rolling conditions."""
        self._prune_outcomes()
        if not self._tripped:
            if self._status_mode not in ("healthy", "resumed"):
                self._status_mode = "healthy"
            return

        if self._cooldown_remaining() > 0:
            self._status_mode = "cooldown"
            return

        self._consecutive_sl = 0
        if self._loss_conditions_active():
            self._status_mode = "recovery_pending"
            return

        self._resume(
            "Automatic resume after cooldown; rolling losses normalized and "
            "a fresh monitoring window was started."
        )

    def _resume(self, reason: str) -> None:
        """Resume signal generation after a successful recovery check."""
        resumed_at = time.monotonic()
        self._tripped = False
        self._trip_reason = ""
        self._trip_time = None
        self._consecutive_sl = 0
        self._status_mode = "resumed"
        self._last_resume_time = resumed_at
        self._monitoring_started_at = resumed_at
        self._last_resume_reason = reason
        log.info("Circuit breaker resumed automatically: %s", reason)
        self._emit_alert(
            "✅ *Circuit Breaker RESUMED*\n"
            f"{reason}\n"
            "Signal generation is active again."
        )

    def _emit_alert(self, message: str) -> None:
        """Fire the optional async alert callback."""
        if self._alert_callback is None:
            return

        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._alert_callback(message))
        except Exception as exc:
            log.warning("Alert callback error (circuit breaker): %s", exc)

    def _loss_conditions_active(self) -> bool:
        """Return True when rolling hourly/drawdown conditions are still elevated."""
        return (
            self._hourly_sl_count() >= self.max_hourly_sl
            or self._daily_drawdown_pct() >= self.max_daily_drawdown_pct
        )

    def _cooldown_remaining(self) -> float:
        """Return remaining cooldown seconds."""
        if self._trip_time is None:
            return 0.0
        return max(0.0, self.cooldown_seconds - (time.monotonic() - self._trip_time))

    def _prune_outcomes(self) -> None:
        """Drop outcomes older than the longest rolling window."""
        cutoff = time.monotonic() - _DAILY_WINDOW_SECONDS
        while self._outcomes and self._outcomes[0].timestamp < cutoff:
            self._outcomes.popleft()

    def _hourly_sl_count(self) -> int:
        """Count SL hits in the last 3600 seconds."""
        cutoff = max(
            time.monotonic() - _HOURLY_WINDOW_SECONDS,
            self._monitoring_started_at,
        )
        return sum(1 for r in self._outcomes if r.hit_sl and r.timestamp >= cutoff)

    def _daily_drawdown_pct(self) -> float:
        """Current 24h drawdown (%) on a compounded equity curve."""
        cutoff = max(
            time.monotonic() - _DAILY_WINDOW_SECONDS,
            self._monitoring_started_at,
        )
        pnls = [r.pnl_pct for r in self._outcomes if r.timestamp >= cutoff]
        current_drawdown, _ = calculate_drawdown_metrics(pnls)
        return current_drawdown
