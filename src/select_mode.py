"""Select Mode – ultra-selective 360_SELECT channel filter.

When :attr:`SelectModeFilter.enabled` is ``True``, every signal that passes
its regular channel filters is also evaluated against 11 stricter criteria.
Signals that pass **all** criteria get a copy published to the ``360_SELECT``
Telegram channel.  Regular channels are completely unaffected.

Usage::

    from src.select_mode import SelectModeFilter

    select_filter = SelectModeFilter()
    select_filter.enable()

    allowed, reason = select_filter.should_publish(
        signal=sig,
        confidence=92.0,
        indicators=indicators_dict,
        smc_data=smc_data_dict,
        ai_sentiment={"label": "Bullish", "score": 0.8},
        cross_exchange_verified=True,
        volume_24h=15_000_000.0,
        spread_pct=0.008,
    )
    if allowed:
        # enqueue SELECT copy
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional, Tuple

from src.utils import get_logger

log = get_logger("select_mode")


@dataclass
class SelectModeConfig:
    """Configuration for the ultra-selective 360_SELECT filter.

    All values can be updated at runtime via
    :meth:`SelectModeFilter.update_config`.
    """

    enabled: bool = False
    """Whether the select-mode filter is active."""

    min_confidence: float = 80.0
    """Minimum confidence score (0–100) required to pass."""

    max_daily_signals: int = 5
    """Maximum number of SELECT signals published per channel per day."""

    min_confluence_timeframes: int = 2
    """Number of timeframes where EMA9 vs EMA21 must align with direction."""

    max_spread_pct: float = 0.015
    """Maximum bid-ask spread percentage allowed."""

    min_volume_24h: float = 10_000_000.0
    """Minimum 24-hour USD volume required."""

    min_adx: float = 25.0
    """Minimum ADX value required (trend strength)."""

    rsi_min: float = 30.0
    """Minimum RSI value; signals below this are rejected (oversold on entry)."""

    rsi_max: float = 70.0
    """Maximum RSI value; signals above this are rejected (overbought on entry)."""

    require_smc_event: bool = True
    """Require at least one SMC event (sweep, MSS, or FVG)."""

    require_ai_sentiment_match: bool = True
    """Require AI sentiment to align with the signal direction."""

    require_cross_exchange: bool = True
    """Require cross-exchange verification to pass (``True``) or be neutral
    (``None``).  A result of ``False`` will always reject the signal."""


class SelectModeFilter:
    """Ultra-selective filter that gates signals for the 360_SELECT channel.

    Parameters
    ----------
    config:
        Optional :class:`SelectModeConfig` instance.  A default configuration
        (with ``enabled=False``) is created if not provided.
    """

    def __init__(self, config: Optional[SelectModeConfig] = None) -> None:
        self._config: SelectModeConfig = config or SelectModeConfig()
        # Daily counter: channel_name -> (date, count)
        self._daily_counts: Dict[str, Tuple[date, int]] = {}

    # ------------------------------------------------------------------
    # Properties / toggle
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return ``True`` when the select-mode filter is active."""
        return self._config.enabled

    def enable(self) -> None:
        """Activate the select-mode filter."""
        self._config.enabled = True
        log.info("Select mode ENABLED")

    def disable(self) -> None:
        """Deactivate the select-mode filter."""
        self._config.enabled = False
        log.info("Select mode DISABLED")

    # ------------------------------------------------------------------
    # Main filter
    # ------------------------------------------------------------------

    def should_publish(
        self,
        signal: Any,
        confidence: float,
        indicators: Dict[str, Dict],
        smc_data: Dict,
        ai_sentiment: Dict,
        cross_exchange_verified: Optional[bool],
        volume_24h: float,
        spread_pct: float,
    ) -> Tuple[bool, str]:
        """Evaluate whether a signal should be published to 360_SELECT.

        When select mode is **disabled** this method always returns
        ``(True, "")`` so that it has zero impact on normal operation.

        Parameters
        ----------
        signal:
            The :class:`src.channels.base.Signal` being evaluated.
        confidence:
            Computed confidence score (0–100).
        indicators:
            Per-timeframe indicator dict (e.g. ``{"5m": {...}, "1h": {...}}``).
        smc_data:
            SMC detection result dict with keys ``sweeps``, ``mss``, ``fvg``.
        ai_sentiment:
            AI sentiment dict with keys ``label`` (str) and ``score`` (float).
        cross_exchange_verified:
            ``True`` → verified, ``None`` → neutral/unavailable, ``False`` →
            contradicted.
        volume_24h:
            24-hour USD trading volume.
        spread_pct:
            Current bid-ask spread as a percentage.

        Returns
        -------
        tuple[bool, str]
            ``(allowed, reason)`` — when *allowed* is ``False``, *reason*
            describes which filter rejected the signal.
        """
        if not self._config.enabled:
            return True, ""

        direction = signal.direction.value  # "LONG" or "SHORT"
        channel = signal.channel

        # 1. Minimum confidence
        if confidence < self._config.min_confidence:
            return False, (
                f"confidence {confidence:.1f} < {self._config.min_confidence}"
            )

        # 2. SMC event required
        if self._config.require_smc_event:
            has_smc = (
                bool(smc_data.get("sweeps"))
                or smc_data.get("mss") is not None
                or bool(smc_data.get("fvg"))
            )
            if not has_smc:
                return False, "no SMC event (sweep/MSS/FVG)"

        # 3. ADX >= min_adx (primary TF: 5m with 1m fallback)
        primary_ind = indicators.get("5m", indicators.get("1m", {}))
        adx_val = primary_ind.get("adx_last")
        if adx_val is not None and adx_val < self._config.min_adx:
            return False, f"ADX {adx_val:.1f} < {self._config.min_adx}"

        # 4. Spread <= max_spread_pct
        if spread_pct > self._config.max_spread_pct:
            return False, (
                f"spread {spread_pct:.4f}% > {self._config.max_spread_pct:.4f}%"
            )

        # 5. Volume >= min_volume_24h
        if volume_24h < self._config.min_volume_24h:
            return False, (
                f"volume ${volume_24h:,.0f} < ${self._config.min_volume_24h:,.0f}"
            )

        # 6. Multi-TF EMA confluence
        ema_aligned_count = 0
        for tf_ind in indicators.values():
            ema9 = tf_ind.get("ema9_last")
            ema21 = tf_ind.get("ema21_last")
            if ema9 is None or ema21 is None:
                continue
            if direction == "LONG" and ema9 > ema21:
                ema_aligned_count += 1
            elif direction == "SHORT" and ema9 < ema21:
                ema_aligned_count += 1
        if ema_aligned_count < self._config.min_confluence_timeframes:
            return False, (
                f"EMA confluence {ema_aligned_count} TF(s) < "
                f"{self._config.min_confluence_timeframes} required"
            )

        # 7. Multi-TF momentum agreement (at least 2 TFs)
        mom_agree_count = 0
        for tf_ind in indicators.values():
            mom = tf_ind.get("momentum_last")
            if mom is None:
                continue
            if direction == "LONG" and mom > 0:
                mom_agree_count += 1
            elif direction == "SHORT" and mom < 0:
                mom_agree_count += 1
        if mom_agree_count < 2:
            return False, (
                f"momentum agreement only on {mom_agree_count} TF(s) (need ≥2)"
            )

        # 8. RSI band check (pass if RSI data unavailable)
        rsi_val = primary_ind.get("rsi_last")
        if rsi_val is not None:
            if rsi_val < self._config.rsi_min or rsi_val > self._config.rsi_max:
                return False, (
                    f"RSI {rsi_val:.1f} outside [{self._config.rsi_min}–"
                    f"{self._config.rsi_max}]"
                )

        # 9. AI sentiment matches direction
        if self._config.require_ai_sentiment_match:
            ai_label = (ai_sentiment.get("label") or "Neutral").lower()
            if ai_label == "bullish" and direction != "LONG":
                return False, "AI bullish but signal is SHORT"
            if ai_label == "bearish" and direction != "SHORT":
                return False, "AI bearish but signal is LONG"

        # 10. Cross-exchange verification
        if self._config.require_cross_exchange and cross_exchange_verified is False:
            return False, "cross-exchange verification failed"

        # 11. Daily cap
        self._maybe_reset_daily(channel)
        today_date, today_count = self._daily_counts.get(channel, (date.today(), 0))
        if today_count >= self._config.max_daily_signals:
            return False, (
                f"daily cap reached ({today_count}/{self._config.max_daily_signals})"
            )
        self._daily_counts[channel] = (today_date, today_count + 1)

        return True, ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self, channel: str) -> None:
        """Reset the daily counter for *channel* if the date has rolled over."""
        today = date.today()
        entry = self._daily_counts.get(channel)
        if entry is not None and entry[0] != today:
            self._daily_counts[channel] = (today, 0)

    def status_text(self) -> str:
        """Return a Telegram-formatted status string."""
        cfg = self._config
        status = "🌹 *ON* ✅" if cfg.enabled else "🔘 *OFF*"
        lines = [
            f"🌹 *360\\_SELECT Mode* — {status}",
            "",
            "⚙️ *Current Config:*",
            f"• Min confidence: `{cfg.min_confidence}`",
            f"• Max daily signals: `{cfg.max_daily_signals}`",
            f"• Min EMA confluence TFs: `{cfg.min_confluence_timeframes}`",
            f"• Max spread: `{cfg.max_spread_pct:.3f}%`",
            f"• Min 24h volume: `${cfg.min_volume_24h:,.0f}`",
            f"• Min ADX: `{cfg.min_adx}`",
            f"• RSI band: `{cfg.rsi_min}–{cfg.rsi_max}`",
            f"• Require SMC event: `{cfg.require_smc_event}`",
            f"• Require AI match: `{cfg.require_ai_sentiment_match}`",
            f"• Require cross-exchange: `{cfg.require_cross_exchange}`",
        ]
        if self._daily_counts:
            lines.append("")
            lines.append("📊 *Today's counts:*")
            for ch, (d, cnt) in self._daily_counts.items():
                lines.append(f"  • {ch}: {cnt}/{cfg.max_daily_signals}")
        return "\n".join(lines)

    def update_config(self, key: str, value: str) -> Tuple[bool, str]:
        """Update a single config field dynamically.

        Parameters
        ----------
        key:
            Config field name (e.g. ``"min_confidence"``).
        value:
            New value as a string (will be cast to the correct type).

        Returns
        -------
        tuple[bool, str]
            ``(success, message)``
        """
        cfg = self._config
        try:
            if key == "min_confidence":
                cfg.min_confidence = float(value)
            elif key == "max_daily_signals":
                cfg.max_daily_signals = int(value)
            elif key == "min_confluence_timeframes":
                cfg.min_confluence_timeframes = int(value)
            elif key == "max_spread_pct":
                cfg.max_spread_pct = float(value)
            elif key == "min_volume_24h":
                cfg.min_volume_24h = float(value)
            elif key == "min_adx":
                cfg.min_adx = float(value)
            elif key == "rsi_min":
                cfg.rsi_min = float(value)
            elif key == "rsi_max":
                cfg.rsi_max = float(value)
            elif key == "require_smc_event":
                cfg.require_smc_event = value.lower() in ("true", "1", "yes")
            elif key == "require_ai_sentiment_match":
                cfg.require_ai_sentiment_match = value.lower() in ("true", "1", "yes")
            elif key == "require_cross_exchange":
                cfg.require_cross_exchange = value.lower() in ("true", "1", "yes")
            else:
                return False, f"Unknown config key: `{key}`"
        except (ValueError, TypeError) as exc:
            return False, f"Invalid value for `{key}`: {exc}"

        log.info("SelectMode config updated: %s = %s", key, value)
        return True, f"✅ `{key}` set to `{value}`"
