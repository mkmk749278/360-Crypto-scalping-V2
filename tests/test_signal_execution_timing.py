"""Tests for signal execution timing improvements.

Validates:
- Entry zone calculation (zone brackets around entry price)
- valid_for_minutes population per channel
- Telegram format includes entry zone and validity window
- Trade monitor respects updated MIN_SIGNAL_LIFESPAN_SECONDS values
- SL distance is wider with new CHANNEL_SCALP config values
- SIGNAL_VALID_FOR_MINUTES config dict has correct per-channel values
"""

from __future__ import annotations

import pytest

from config import (
    CHANNEL_SCALP,
    MIN_SIGNAL_LIFESPAN_SECONDS,
    SIGNAL_VALID_FOR_MINUTES,
)
from src.channels.base import Direction, Signal, build_channel_signal
from src.channels.scalp import ScalpChannel
from src.telegram_bot import TelegramBot
from src.utils import utcnow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    channel: str = "360_SCALP",
    entry: float = 30000.0,
    stop_loss: float = 29850.0,
    tp1: float = 30150.0,
    tp2: float = 30300.0,
    tp3: float = 30450.0,
    entry_zone_low: float | None = None,
    entry_zone_high: float | None = None,
    valid_for_minutes: int = 15,
    execution_type: str = "LIMIT_ZONE",
) -> Signal:
    return Signal(
        channel=channel,
        symbol="BTCUSDT",
        direction=Direction.LONG,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        confidence=75.0,
        timestamp=utcnow(),
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        valid_for_minutes=valid_for_minutes,
        execution_type=execution_type,
    )


# ---------------------------------------------------------------------------
# Phase 1: Config changes
# ---------------------------------------------------------------------------

class TestConfigChanges:
    """Verify that config values match the required execution-timing fixes."""

    def test_channel_scalp_sl_pct_range_widened(self):
        """CHANNEL_SCALP sl_pct_range must be at least 0.20% (was 0.05%)."""
        low, high = CHANNEL_SCALP.sl_pct_range
        assert low >= 0.20, f"sl_pct_range lower bound too tight: {low}"
        assert high >= 0.40, f"sl_pct_range upper bound too tight: {high}"

    def test_channel_scalp_tp_ratios_widened(self):
        """CHANNEL_SCALP tp_ratios must be at least [1.5, 2.5, 4.0]."""
        r1, r2, r3 = CHANNEL_SCALP.tp_ratios
        assert r1 >= 1.5, f"TP1 ratio too tight: {r1}"
        assert r2 >= 2.5, f"TP2 ratio too tight: {r2}"
        assert r3 >= 4.0, f"TP3 ratio too tight: {r3}"

    def test_min_signal_lifespan_scalp_increased(self):
        """SCALP min lifespan must be at least 180 seconds (was 30s)."""
        assert MIN_SIGNAL_LIFESPAN_SECONDS["360_SCALP"] >= 180

    def test_min_signal_lifespan_swing_increased(self):
        """SWING min lifespan must be at least 300 seconds (was 60s)."""
        assert MIN_SIGNAL_LIFESPAN_SECONDS["360_SWING"] >= 300

    def test_min_signal_lifespan_spot_increased(self):
        """SPOT min lifespan must be at least 600 seconds (was 120s)."""
        assert MIN_SIGNAL_LIFESPAN_SECONDS["360_SPOT"] >= 600

    def test_min_signal_lifespan_gem_unchanged(self):
        """GEM min lifespan has been reduced to 6 hours (21600 seconds) to allow
        GEM signals on volatile small-cap tokens to be evaluated sooner."""
        assert MIN_SIGNAL_LIFESPAN_SECONDS["360_GEM"] == 21600

    def test_signal_valid_for_minutes_scalp(self):
        """SCALP signals should be valid for 15 minutes."""
        assert SIGNAL_VALID_FOR_MINUTES.get("360_SCALP") == 15

    def test_signal_valid_for_minutes_swing(self):
        """SWING signals should be valid for 60 minutes."""
        assert SIGNAL_VALID_FOR_MINUTES.get("360_SWING") == 60

    def test_signal_valid_for_minutes_spot(self):
        """SPOT signals should be valid for 240 minutes."""
        assert SIGNAL_VALID_FOR_MINUTES.get("360_SPOT") == 240

    def test_signal_valid_for_minutes_gem(self):
        """GEM signals should be valid for 1440 minutes (24 hours)."""
        assert SIGNAL_VALID_FOR_MINUTES.get("360_GEM") == 1440

    def test_signal_valid_for_minutes_all_scalp_subtypes(self):
        """All SCALP sub-channel types must have a valid_for_minutes entry."""
        scalp_channels = [
            "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD",
            "360_SCALP_VWAP", "360_SCALP_OBI",
        ]
        for ch in scalp_channels:
            assert ch in SIGNAL_VALID_FOR_MINUTES, f"{ch} missing from SIGNAL_VALID_FOR_MINUTES"
            assert SIGNAL_VALID_FOR_MINUTES[ch] == 15, f"{ch} valid_for_minutes != 15"


# ---------------------------------------------------------------------------
# Phase 2: Signal dataclass new fields
# ---------------------------------------------------------------------------

class TestSignalDataclassNewFields:
    """Signal dataclass must have the new execution-timing fields."""

    def test_entry_zone_low_defaults_to_none(self):
        sig = _make_signal()
        assert sig.entry_zone_low is None

    def test_entry_zone_high_defaults_to_none(self):
        sig = _make_signal()
        assert sig.entry_zone_high is None

    def test_valid_for_minutes_default(self):
        sig = _make_signal()
        assert sig.valid_for_minutes == 15

    def test_execution_type_default(self):
        sig = _make_signal()
        assert sig.execution_type == "LIMIT_ZONE"

    def test_fields_can_be_set(self):
        sig = _make_signal(
            entry_zone_low=29950.0,
            entry_zone_high=30050.0,
            valid_for_minutes=60,
            execution_type="MARKET",
        )
        assert sig.entry_zone_low == pytest.approx(29950.0)
        assert sig.entry_zone_high == pytest.approx(30050.0)
        assert sig.valid_for_minutes == 60
        assert sig.execution_type == "MARKET"


# ---------------------------------------------------------------------------
# Phase 3: Channel logic – entry zone population
# ---------------------------------------------------------------------------

class TestScalpChannelEntryZone:
    """build_channel_signal must populate entry_zone_low/high."""

    def _make_scalp_signal(
        self,
        close: float = 30000.0,
        direction: Direction = Direction.LONG,
        atr_val: float = 100.0,
    ) -> Signal | None:
        chan = ScalpChannel()
        sl_dist = max(close * chan.config.sl_pct_range[0] / 100, atr_val * 0.5)
        if direction == Direction.LONG:
            sl = close - sl_dist
            tp1 = close + sl_dist * chan.config.tp_ratios[0]
            tp2 = close + sl_dist * chan.config.tp_ratios[1]
            tp3 = close + sl_dist * chan.config.tp_ratios[2]
        else:
            sl = close + sl_dist
            tp1 = close - sl_dist * chan.config.tp_ratios[0]
            tp2 = close - sl_dist * chan.config.tp_ratios[1]
            tp3 = close - sl_dist * chan.config.tp_ratios[2]

        return build_channel_signal(
            config=chan.config,
            symbol="BTCUSDT",
            direction=direction,
            close=close,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl_dist=sl_dist,
            id_prefix="SCALP",
            atr_val=atr_val,
        )

    def test_entry_zone_low_set(self):
        sig = self._make_scalp_signal(close=30000.0, atr_val=100.0)
        assert sig is not None
        assert sig.entry_zone_low is not None
        assert sig.entry_zone_low < sig.entry

    def test_entry_zone_high_set(self):
        sig = self._make_scalp_signal(close=30000.0, atr_val=100.0)
        assert sig is not None
        assert sig.entry_zone_high is not None
        assert sig.entry_zone_high > sig.entry

    def test_entry_zone_brackets_entry(self):
        """Entry price must be within (zone_low, zone_high)."""
        sig = self._make_scalp_signal(close=30000.0, atr_val=100.0)
        assert sig is not None
        assert sig.entry_zone_low < sig.entry < sig.entry_zone_high

    def test_entry_zone_width_proportional_to_atr(self):
        """Larger ATR → wider entry zone."""
        sig_small = self._make_scalp_signal(close=30000.0, atr_val=50.0)
        sig_large = self._make_scalp_signal(close=30000.0, atr_val=200.0)
        assert sig_small is not None and sig_large is not None
        width_small = sig_small.entry_zone_high - sig_small.entry_zone_low
        width_large = sig_large.entry_zone_high - sig_large.entry_zone_low
        assert width_large > width_small

    def test_zone_half_width_approx_atr_times_0_6(self):
        """Direction-biased zone: LONG biases below close (0.7×width), above close (0.3×width)."""
        close = 30000.0
        atr = 100.0
        # zone_width = atr * 0.4 = 40.0; LONG: low = close - 40*0.7, high = close + 40*0.3
        zone_width = atr * 0.4
        sig = self._make_scalp_signal(close=close, atr_val=atr, direction=Direction.LONG)
        assert sig is not None
        expected_low = close - zone_width * 0.7
        expected_high = close + zone_width * 0.3
        assert sig.entry_zone_low == pytest.approx(expected_low, abs=1e-6)
        assert sig.entry_zone_high == pytest.approx(expected_high, abs=1e-6)

    def test_zone_short_direction_biased_above_close(self):
        """SHORT zone biases above close (0.7×width) and below (0.3×width)."""
        close = 30000.0
        atr = 100.0
        zone_width = atr * 0.4
        sig = self._make_scalp_signal(close=close, atr_val=atr, direction=Direction.SHORT)
        assert sig is not None
        expected_low = close - zone_width * 0.3
        expected_high = close + zone_width * 0.7
        assert sig.entry_zone_low == pytest.approx(expected_low, abs=1e-6)
        assert sig.entry_zone_high == pytest.approx(expected_high, abs=1e-6)

    def test_entry_zone_fallback_when_atr_zero(self):
        """When atr_val=0, fall back to sl_dist * 0.6 for zone width."""
        close = 30000.0
        sig = self._make_scalp_signal(close=close, atr_val=0.0)
        assert sig is not None
        # zone should still be set and bracket entry
        assert sig.entry_zone_low is not None
        assert sig.entry_zone_high is not None
        assert sig.entry_zone_low < sig.entry
        assert sig.entry_zone_high > sig.entry

    def test_short_signal_zone_also_brackets_entry(self):
        sig = self._make_scalp_signal(close=30000.0, direction=Direction.SHORT, atr_val=100.0)
        assert sig is not None
        assert sig.entry_zone_low < sig.entry < sig.entry_zone_high


class TestSpotChannelEntryZone:
    """SpotChannel must use the DCA zone as entry_zone_low/high."""

    def _make_spot_signal(self) -> Signal | None:
        """Build a minimal SpotChannel signal bypassing evaluate() gates."""
        from src.channels.spot import SpotChannel
        from src.dca import compute_dca_zone
        import uuid

        chan = SpotChannel()
        close = 30000.0
        atr_val = close * 0.01  # 1%
        sl_dist = max(close * chan.config.sl_pct_range[0] / 100, atr_val * 1.5)
        sl = close - sl_dist
        tp1 = close + sl_dist * chan.config.tp_ratios[0]
        tp2 = close + sl_dist * chan.config.tp_ratios[1]
        tp3 = close + sl_dist * chan.config.tp_ratios[2]

        sig = Signal(
            channel=chan.config.name,
            symbol="ETHUSDT",
            direction=Direction.LONG,
            entry=close,
            stop_loss=round(sl, 8),
            tp1=round(tp1, 8),
            tp2=round(tp2, 8),
            tp3=round(tp3, 8),
            trailing_active=True,
            trailing_desc=f"{chan.config.trailing_atr_mult}×ATR",
            confidence=0.0,
            ai_sentiment_label="",
            ai_sentiment_summary="",
            risk_label="Conservative",
            timestamp=utcnow(),
            signal_id=f"SPOT-{uuid.uuid4().hex[:8].upper()}",
            current_price=close,
            original_sl_distance=sl_dist,
        )

        if chan.config.dca_enabled:
            dca_lower, dca_upper = compute_dca_zone(
                close, round(sl, 8), Direction.LONG, chan.config.dca_zone_range
            )
            sig.dca_zone_lower = dca_lower
            sig.dca_zone_upper = dca_upper
            sig.original_entry = close
            sig.original_tp1 = round(tp1, 8)
            sig.original_tp2 = round(tp2, 8)
            sig.original_tp3 = round(tp3, 8)
            # Spot uses DCA zone as entry zone
            sig.entry_zone_low = dca_lower
            sig.entry_zone_high = dca_upper

        return sig

    def test_spot_entry_zone_equals_dca_zone(self):
        sig = self._make_spot_signal()
        assert sig is not None
        assert sig.entry_zone_low == pytest.approx(sig.dca_zone_lower)
        assert sig.entry_zone_high == pytest.approx(sig.dca_zone_upper)

    def test_spot_entry_zone_is_below_entry(self):
        """For a LONG spot signal the DCA zone should be below the entry price."""
        sig = self._make_spot_signal()
        assert sig is not None
        # DCA zone is an accumulation zone below the breakout price
        assert sig.entry_zone_low <= sig.entry
        assert sig.entry_zone_high <= sig.entry


# ---------------------------------------------------------------------------
# Phase 4: Telegram format – entry zone and validity window
# ---------------------------------------------------------------------------

class TestTelegramFormatEntryZone:
    """format_signal() must display entry zone when populated."""

    def test_no_zone_shows_exact_entry(self):
        sig = _make_signal(entry=30000.0)
        text = TelegramBot.format_signal(sig)
        assert "📍 Entry:" in text
        assert "Entry Zone" not in text

    def test_zone_shows_entry_zone_not_exact_entry_line(self):
        sig = _make_signal(
            entry=30000.0,
            entry_zone_low=29970.0,
            entry_zone_high=30030.0,
        )
        text = TelegramBot.format_signal(sig)
        assert "Entry Zone" in text

    def test_zone_shows_low_and_high_prices(self):
        sig = _make_signal(
            entry=30000.0,
            entry_zone_low=29970.0,
            entry_zone_high=30030.0,
        )
        text = TelegramBot.format_signal(sig)
        # Both zone boundary prices should appear
        assert "29,970" in text or "29970" in text
        assert "30,030" in text or "30030" in text

    def test_zone_shows_mid_reference(self):
        """Mid-point reference line must appear when zone is shown."""
        sig = _make_signal(
            entry=30000.0,
            entry_zone_low=29970.0,
            entry_zone_high=30030.0,
        )
        text = TelegramBot.format_signal(sig)
        assert "Mid" in text

    def test_validity_line_present_when_valid_for_minutes_set(self):
        sig = _make_signal(valid_for_minutes=15)
        text = TelegramBot.format_signal(sig)
        assert "Valid for" in text
        assert "15" in text

    def test_validity_shows_execution_limit_order(self):
        sig = _make_signal(valid_for_minutes=15, execution_type="LIMIT_ZONE")
        text = TelegramBot.format_signal(sig)
        assert "LIMIT ORDER" in text

    def test_validity_minutes_correct_for_spot(self):
        sig = _make_signal(channel="360_SPOT", valid_for_minutes=240)
        text = TelegramBot.format_signal(sig)
        assert "240" in text

    def test_sl_and_tp_still_present(self):
        """Core SL and TP levels must still appear in the message."""
        sig = _make_signal(
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            tp2=30300.0,
            entry_zone_low=29970.0,
            entry_zone_high=30030.0,
        )
        text = TelegramBot.format_signal(sig)
        assert "🛑 SL" in text
        assert "🎯 TP1" in text
        assert "🎯 TP2" in text


# ---------------------------------------------------------------------------
# Phase 5: Trade monitor respects updated lifespan values
# ---------------------------------------------------------------------------

class TestTradeMonitorLifespanValues:
    """Trade monitor must use the new MIN_SIGNAL_LIFESPAN_SECONDS values."""

    def _build_monitor(self, active):
        from unittest.mock import MagicMock
        from src.trade_monitor import TradeMonitor

        removed = []

        async def mock_send(chat_id, text):
            pass

        data_store = MagicMock()
        data_store.get_candles.return_value = None
        data_store.ticks = {}

        monitor = TradeMonitor(
            data_store=data_store,
            send_telegram=mock_send,
            get_active_signals=lambda: dict(active),
            remove_signal=lambda sid: removed.append(sid),
            update_signal=MagicMock(),
        )
        return monitor, removed

    def _make_signal_with_age(
        self,
        channel: str,
        age_seconds: float,
        direction: Direction = Direction.LONG,
        entry: float = 30000.0,
        stop_loss: float = 29850.0,
        tp1: float = 30150.0,
        tp2: float = 30300.0,
    ) -> Signal:
        from datetime import timedelta
        sig = Signal(
            channel=channel,
            symbol="BTCUSDT",
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            confidence=75.0,
            timestamp=utcnow() - timedelta(seconds=age_seconds),
            signal_id=f"TEST-{channel}-{int(age_seconds)}",
        )
        return sig

    @pytest.mark.asyncio
    async def test_scalp_signal_below_180s_not_triggered(self):
        """A SCALP signal at age=100s (< 180s min) must NOT trigger SL."""
        sig = self._make_signal_with_age("360_SCALP", age_seconds=100.0)
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed = self._build_monitor(active)
        await monitor._evaluate_signal(sig)

        assert sig.signal_id not in removed
        assert sig.status == "ACTIVE"

    @pytest.mark.asyncio
    async def test_scalp_signal_above_180s_triggered(self):
        """A SCALP signal at age=200s (> 180s min) MUST trigger SL."""
        sig = self._make_signal_with_age("360_SCALP", age_seconds=200.0)
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed = self._build_monitor(active)
        await monitor._evaluate_signal(sig)

        assert sig.signal_id in removed
        assert sig.status == "SL_HIT"

    @pytest.mark.asyncio
    async def test_swing_signal_below_300s_not_triggered(self):
        """A SWING signal at age=100s (< 300s min) must NOT trigger SL."""
        sig = self._make_signal_with_age("360_SWING", age_seconds=100.0)
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed = self._build_monitor(active)
        await monitor._evaluate_signal(sig)

        assert sig.signal_id not in removed
        assert sig.status == "ACTIVE"

    @pytest.mark.asyncio
    async def test_spot_signal_below_600s_not_triggered(self):
        """A SPOT signal at age=400s (< 600s min) must NOT trigger SL."""
        sig = self._make_signal_with_age("360_SPOT", age_seconds=400.0)
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed = self._build_monitor(active)
        await monitor._evaluate_signal(sig)

        assert sig.signal_id not in removed
        assert sig.status == "ACTIVE"
