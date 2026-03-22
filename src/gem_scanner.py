"""Gem Scanner — macro-reversal detection for deeply discounted altcoins.

Scans for tokens that have:
1. Massive drawdown from ATH (≥70%)
2. Formed an accumulation base (low volatility sideways period)
3. Early reversal signals (volume surge + MA crossover on daily)

Publishes to the 360_GEM Telegram channel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    GEM_MAX_DAILY_SIGNALS,
    GEM_MAX_RANGE_PCT,
    GEM_MIN_DRAWDOWN_PCT,
    GEM_MIN_VOLUME_RATIO,
    GEM_SCAN_INTERVAL_HOURS,
    GEM_SCANNER_ENABLED,
)
from src.indicators import ema
from src.utils import get_logger

log = get_logger("gem_scanner")


@dataclass
class GemSignal:
    """A detected gem/moonshot opportunity."""

    symbol: str
    current_price: float
    ath: float
    drawdown_pct: float
    x_potential: float
    accumulation_days: int
    volume_ratio: float
    ma_crossover: bool
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class GemScannerConfig:
    """Runtime-adjustable configuration for the gem scanner."""

    enabled: bool = True
    min_drawdown_pct: float = 70.0
    max_range_pct: float = 40.0
    min_volume_ratio: float = 1.5
    max_daily_signals: int = 3
    scan_interval_hours: int = 6


class GemScanner:
    """Scans for deeply discounted tokens showing macro reversal patterns."""

    def __init__(self, config: Optional[GemScannerConfig] = None) -> None:
        self._config = config or GemScannerConfig(
            enabled=GEM_SCANNER_ENABLED,
            min_drawdown_pct=GEM_MIN_DRAWDOWN_PCT,
            max_range_pct=GEM_MAX_RANGE_PCT,
            min_volume_ratio=GEM_MIN_VOLUME_RATIO,
            max_daily_signals=GEM_MAX_DAILY_SIGNALS,
            scan_interval_hours=GEM_SCAN_INTERVAL_HOURS,
        )
        self._daily_counts: Dict[str, Tuple[date, int]] = {}
        self._last_scan: float = 0.0
        self._gem_pairs: List[str] = []

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def set_gem_pairs(self, symbols: List[str]) -> None:
        """Store the gem-specific pair list being tracked."""
        self._gem_pairs = list(symbols)

    def get_scan_pair_count(self) -> int:
        """Return how many pairs the gem scanner is currently tracking."""
        return len(self._gem_pairs)

    def enable(self) -> None:
        self._config.enabled = True
        log.info("Gem scanner ENABLED")

    def disable(self) -> None:
        self._config.enabled = False
        log.info("Gem scanner DISABLED")

    def scan(
        self,
        symbol: str,
        daily_candles: Dict[str, list],
        weekly_candles: Optional[Dict[str, list]] = None,
    ) -> Optional[GemSignal]:
        """Evaluate a single symbol for gem/moonshot potential.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. "TOKENUSDT").
        daily_candles:
            Dict with keys "open", "high", "low", "close", "volume" — daily OHLCV.
        weekly_candles:
            Optional weekly OHLCV for ATH detection. Falls back to daily if None.

        Returns
        -------
        Optional[GemSignal]
            A GemSignal if the token passes all filters, else None.
        """
        if not self._config.enabled:
            return None

        closes = daily_candles.get("close", [])
        highs = daily_candles.get("high", [])
        volumes = daily_candles.get("volume", [])

        if len(closes) < 50:
            return None  # Need enough history

        # Use weekly candles for ATH if available, else daily
        ath_source = weekly_candles if weekly_candles else daily_candles
        ath_highs = ath_source.get("high", highs)
        if not ath_highs:
            return None

        current_price = float(closes[-1])
        ath = float(max(ath_highs))

        if ath <= 0 or current_price <= 0:
            return None

        # Step 1: Check drawdown from ATH
        drawdown_pct = (ath - current_price) / ath * 100.0
        if drawdown_pct < self._config.min_drawdown_pct:
            return None

        # Step 2: Check for accumulation base (tight range in last 30 candles)
        recent_30 = min(30, len(highs))
        recent_highs = [float(h) for h in highs[-recent_30:]]
        recent_lows = [float(lo) for lo in daily_candles.get("low", [])[-recent_30:]]

        if not recent_highs or not recent_lows:
            return None

        range_high = max(recent_highs)
        range_low = min(recent_lows)
        if range_low <= 0:
            return None

        range_pct = (range_high - range_low) / range_low * 100.0
        if range_pct > self._config.max_range_pct:
            return None  # Still too volatile, no base formed

        # Count accumulation days (consecutive days within the base range)
        base_mid = (range_high + range_low) / 2.0
        base_tolerance = (range_high - range_low) * 1.5
        accum_days = 0
        for c in reversed([float(x) for x in closes]):
            if abs(c - base_mid) <= base_tolerance:
                accum_days += 1
            else:
                break

        # Step 3: Volume surge detection
        if len(volumes) < 90:
            avg_vol_period = len(volumes)
        else:
            avg_vol_period = 90

        float_volumes = [float(v) for v in volumes]
        avg_vol = np.mean(float_volumes[-avg_vol_period:]) if float_volumes else 0
        recent_vol = np.mean(float_volumes[-7:]) if len(float_volumes) >= 7 else 0

        if avg_vol <= 0:
            return None

        vol_ratio = float(recent_vol / avg_vol)

        if vol_ratio < self._config.min_volume_ratio:
            return None

        # Step 4: Moving average crossover on daily
        float_closes = [float(c) for c in closes]
        np_closes = np.array(float_closes, dtype=np.float64)

        ma_crossover = False
        if len(np_closes) >= 50:
            ema_20 = ema(np_closes, 20)
            ema_50 = ema(np_closes, 50)

            # Check for fresh golden cross (EMA20 crossed above EMA50 in last 5 days)
            if len(ema_20) >= 5 and len(ema_50) >= 5:
                current_above = ema_20[-1] > ema_50[-1]
                was_below = any(ema_20[-i] <= ema_50[-i] for i in range(2, min(6, len(ema_20))))
                ma_crossover = current_above and was_below

            # Also check if price is reclaiming EMA20 from below
            if not ma_crossover and len(ema_20) >= 3:
                price_above_now = current_price > ema_20[-1]
                price_below_recent = any(
                    float(closes[-i]) < ema_20[-i] for i in range(2, min(4, len(ema_20)))
                )
                ma_crossover = price_above_now and price_below_recent

        if not ma_crossover:
            return None

        # Step 5: Calculate x-potential and confidence
        x_potential = ath / current_price

        # Confidence scoring for gems
        confidence = 50.0  # Base
        if drawdown_pct >= 85:
            confidence += 15.0
        elif drawdown_pct >= 75:
            confidence += 10.0
        else:
            confidence += 5.0

        if accum_days >= 20:
            confidence += 10.0
        elif accum_days >= 10:
            confidence += 5.0

        if vol_ratio >= 3.0:
            confidence += 15.0
        elif vol_ratio >= 2.0:
            confidence += 10.0
        else:
            confidence += 5.0

        if x_potential >= 10:
            confidence += 10.0
        elif x_potential >= 5:
            confidence += 5.0

        confidence = min(100.0, max(0.0, confidence))

        # Daily cap check
        today = date.today()
        day_key = "360_GEM"
        entry = self._daily_counts.get(day_key)
        if entry is not None:
            d, count = entry
            if d == today and count >= self._config.max_daily_signals:
                return None
            if d != today:
                self._daily_counts[day_key] = (today, 0)

        return GemSignal(
            symbol=symbol,
            current_price=current_price,
            ath=ath,
            drawdown_pct=drawdown_pct,
            x_potential=x_potential,
            accumulation_days=accum_days,
            volume_ratio=vol_ratio,
            ma_crossover=ma_crossover,
            confidence=confidence,
        )

    def record_published(self) -> None:
        """Increment daily counter after a gem signal is published."""
        today = date.today()
        day_key = "360_GEM"
        entry = self._daily_counts.get(day_key)
        if entry is None or entry[0] != today:
            self._daily_counts[day_key] = (today, 1)
        else:
            self._daily_counts[day_key] = (today, entry[1] + 1)

    def status_text(self) -> str:
        """Return a Telegram-formatted status string."""
        cfg = self._config
        status = "💎 *ON* ✅" if cfg.enabled else "🔘 *OFF*"
        today = date.today()
        day_key = "360_GEM"
        entry = self._daily_counts.get(day_key)
        today_count = entry[1] if entry and entry[0] == today else 0
        pair_count = self.get_scan_pair_count()
        lines = [
            f"💎 *360\\_GEM Scanner* — {status}",
            "",
            "⚙️ *Current Config:*",
            f"• Min drawdown from ATH: `{cfg.min_drawdown_pct:.0f}%`",
            f"• Max 30d range: `{cfg.max_range_pct:.0f}%`",
            f"• Min volume surge: `{cfg.min_volume_ratio:.1f}x`",
            f"• Max daily signals: `{cfg.max_daily_signals}`",
            f"• Scan interval: `{cfg.scan_interval_hours}h`",
            f"• Pairs tracked: `{pair_count}`",
            "",
            f"📊 *Today:* {today_count}/{cfg.max_daily_signals} signals",
        ]
        return "\n".join(lines)

    def update_config(self, key: str, value: str) -> Tuple[bool, str]:
        """Update a single config field dynamically."""
        cfg = self._config
        try:
            if key == "min_drawdown_pct":
                cfg.min_drawdown_pct = float(value)
            elif key == "max_range_pct":
                cfg.max_range_pct = float(value)
            elif key == "min_volume_ratio":
                cfg.min_volume_ratio = float(value)
            elif key == "max_daily_signals":
                cfg.max_daily_signals = int(value)
            elif key == "scan_interval_hours":
                cfg.scan_interval_hours = int(value)
            else:
                return False, f"Unknown config key: `{key}`"
        except (ValueError, TypeError) as exc:
            return False, f"Invalid value for `{key}`: {exc}"

        log.info("GemScanner config updated: %s = %s", key, value)
        return True, f"✅ `{key}` set to `{value}`"
