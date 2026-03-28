"""Tests for Phase 4: free channel condensed signal logic (_maybe_publish_free_signal)."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.channels.base import Signal
from src.smc import Direction
from src.signal_router import SignalRouter
from src.utils import utcnow


def _make_signal(channel="360_SCALP", symbol="BTCUSDT", confidence=80.0) -> Signal:
    return Signal(
        channel=channel,
        symbol=symbol,
        direction=Direction.LONG,
        entry=50000.0,
        stop_loss=49800.0,
        tp1=50200.0,
        tp2=50400.0,
        confidence=confidence,
        timestamp=utcnow(),
    )


def _make_router(free_channel="free_id"):
    q = MagicMock()
    router = SignalRouter(
        queue=q,
        send_telegram=AsyncMock(return_value=True),
        format_signal=MagicMock(return_value="formatted"),
    )
    return router


class TestMaybePublishFreeSignal:
    @pytest.mark.asyncio
    async def test_posts_first_active_signal_today(self, monkeypatch):
        """First high-confidence SCALP/SWING signal of the day is posted to free channel."""
        monkeypatch.setattr("src.signal_router.TELEGRAM_FREE_CHANNEL_ID", "free_id")
        router = _make_router()
        sig = _make_signal(channel="360_SCALP", confidence=80.0)

        await router._maybe_publish_free_signal(sig)

        router._send_telegram.assert_called_once()
        call_args = router._send_telegram.call_args[0]
        assert call_args[0] == "free_id"
        assert router._free_signals_today.get("active") is True

    @pytest.mark.asyncio
    async def test_posts_first_spot_signal_today(self, monkeypatch):
        """First high-confidence SPOT/GEM signal of the day is posted to free channel."""
        monkeypatch.setattr("src.signal_router.TELEGRAM_FREE_CHANNEL_ID", "free_id")
        router = _make_router()
        sig = _make_signal(channel="360_SPOT", confidence=80.0)

        await router._maybe_publish_free_signal(sig)

        router._send_telegram.assert_called_once()
        assert router._free_signals_today.get("active") is True

    @pytest.mark.asyncio
    async def test_does_not_post_second_active_signal_same_day(self, monkeypatch):
        """Only one free signal per group per day; second signal is suppressed."""
        monkeypatch.setattr("src.signal_router.TELEGRAM_FREE_CHANNEL_ID", "free_id")
        router = _make_router()

        # Post first
        sig1 = _make_signal(channel="360_SCALP", confidence=80.0)
        await router._maybe_publish_free_signal(sig1)

        # Second call should be suppressed
        router._send_telegram.reset_mock()
        sig2 = _make_signal(channel="360_SWING", confidence=90.0)
        await router._maybe_publish_free_signal(sig2)

        router._send_telegram.assert_not_called()

    @pytest.mark.asyncio
    async def test_active_and_spot_share_same_group(self, monkeypatch):
        """Active and SPOT/GEM signals share the same group, so only first is posted."""
        monkeypatch.setattr("src.signal_router.TELEGRAM_FREE_CHANNEL_ID", "free_id")
        router = _make_router()

        sig_active = _make_signal(channel="360_SCALP", confidence=80.0)
        sig_spot = _make_signal(channel="360_SPOT", confidence=80.0)

        await router._maybe_publish_free_signal(sig_active)
        await router._maybe_publish_free_signal(sig_spot)

        # Both map to "active" group, so only first should be posted
        assert router._send_telegram.call_count == 1

    @pytest.mark.asyncio
    async def test_low_confidence_signal_not_posted(self, monkeypatch):
        """Signals with confidence < 75 are NOT posted to free channel."""
        monkeypatch.setattr("src.signal_router.TELEGRAM_FREE_CHANNEL_ID", "free_id")
        router = _make_router()
        sig = _make_signal(channel="360_SCALP", confidence=70.0)  # below 75

        await router._maybe_publish_free_signal(sig)

        router._send_telegram.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_nothing_when_no_free_channel(self, monkeypatch):
        """If TELEGRAM_FREE_CHANNEL_ID is empty, nothing is posted."""
        monkeypatch.setattr("src.signal_router.TELEGRAM_FREE_CHANNEL_ID", "")
        router = _make_router()
        sig = _make_signal(channel="360_SCALP", confidence=90.0)

        await router._maybe_publish_free_signal(sig)

        router._send_telegram.assert_not_called()

    @pytest.mark.asyncio
    async def test_daily_reset_logic(self, monkeypatch):
        """After a new day starts, tracking resets and posting is allowed again."""
        monkeypatch.setattr("src.signal_router.TELEGRAM_FREE_CHANNEL_ID", "free_id")
        router = _make_router()

        # Simulate having already posted today
        router._free_signal_date = date.today()
        router._free_signals_today = {"active": True}

        # Simulate a new day
        new_day = date.today() + timedelta(days=1)
        with patch("src.signal_router.date") as mock_date:
            mock_date.today.return_value = new_day
            sig = _make_signal(channel="360_SCALP", confidence=80.0)
            await router._maybe_publish_free_signal(sig)

        router._send_telegram.assert_called_once()


class TestFreeChannelGroup:
    def test_scalp_channels_are_active_group(self):
        for ch in ("360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD", "360_SCALP_VWAP", "360_SCALP_OBI", "360_SWING"):
            assert SignalRouter._free_channel_group(ch) == "active"

    def test_spot_gem_are_active_group(self):
        for ch in ("360_SPOT", "360_GEM"):
            assert SignalRouter._free_channel_group(ch) == "active"


class TestFormatCondensedFree:
    def test_condensed_format_has_free_header(self):
        q = MagicMock()
        router = SignalRouter(queue=q, send_telegram=AsyncMock(), format_signal=MagicMock())
        sig = _make_signal()
        text = router._format_condensed_free(sig)
        assert "FREE SIGNAL PREVIEW" in text

    def test_condensed_format_has_entry_sl_tp1(self):
        q = MagicMock()
        router = SignalRouter(queue=q, send_telegram=AsyncMock(), format_signal=MagicMock())
        sig = _make_signal()
        text = router._format_condensed_free(sig)
        assert "Entry" in text
        assert "SL" in text
        assert "TP1" in text

    def test_condensed_format_no_tp2_or_tp3(self):
        q = MagicMock()
        router = SignalRouter(queue=q, send_telegram=AsyncMock(), format_signal=MagicMock())
        sig = _make_signal()
        text = router._format_condensed_free(sig)
        # Should not show TP2/TP3 as signal lines (only TP1)
        assert "🎯 TP2" not in text
        assert "🎯 TP3" not in text

    def test_condensed_format_has_premium_footer(self):
        q = MagicMock()
        router = SignalRouter(queue=q, send_telegram=AsyncMock(), format_signal=MagicMock())
        sig = _make_signal()
        text = router._format_condensed_free(sig)
        assert "Premium" in text or "premium" in text
