"""Tests for CommandHandler – command parsing and routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.commands import CommandHandler


def _make_handler(**kwargs) -> CommandHandler:
    """Create a minimal CommandHandler with mocked dependencies."""
    telegram = MagicMock()
    telegram.send_message = AsyncMock()

    defaults = dict(
        telegram=telegram,
        telemetry=MagicMock(),
        pair_mgr=MagicMock(),
        router=MagicMock(),
        data_store=MagicMock(),
        signal_queue=MagicMock(),
        signal_history=[],
        paused_channels=set(),
        confidence_overrides={},
        scanner=MagicMock(),
        ws_spot=None,
        ws_futures=None,
        tasks=[],
        boot_time=0.0,
        free_channel_limit=2,
        alert_subscribers=set(),
    )
    defaults.update(kwargs)
    return CommandHandler(**defaults)


ADMIN_CHAT_ID = "710718010"
USER_CHAT_ID = "999999"


class TestAdminGuard:
    @pytest.mark.asyncio
    async def test_admin_command_blocked_for_non_admin(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/view_dashboard", USER_CHAT_ID)
        handler._telegram.send_message.assert_called_once()
        args = handler._telegram.send_message.call_args[0]
        assert "restricted" in args[1].lower()

    @pytest.mark.asyncio
    async def test_user_command_allowed_for_non_admin(self):
        handler = _make_handler()
        handler._router.active_signals = {}
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/signals", USER_CHAT_ID)
        handler._telegram.send_message.assert_called_once()
        # Should NOT say "restricted"
        args = handler._telegram.send_message.call_args[0]
        assert "restricted" not in args[1].lower()


class TestAdminCommands:
    @pytest.mark.asyncio
    async def test_view_dashboard(self):
        handler = _make_handler()
        handler._telemetry.dashboard_text.return_value = "📊 Dashboard"
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/view_dashboard", ADMIN_CHAT_ID)
        handler._telemetry.dashboard_text.assert_called_once()
        handler._telegram.send_message.assert_called_once_with(ADMIN_CHAT_ID, "📊 Dashboard")

    @pytest.mark.asyncio
    async def test_force_scan(self):
        scanner = MagicMock()
        scanner.force_scan = False
        handler = _make_handler(scanner=scanner)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/force_scan", ADMIN_CHAT_ID)
        assert scanner.force_scan is True

    @pytest.mark.asyncio
    async def test_pause_channel(self):
        paused = set()
        handler = _make_handler(paused_channels=paused)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/pause_channel 360_SCALP", ADMIN_CHAT_ID)
        assert "360_SCALP" in paused

    @pytest.mark.asyncio
    async def test_resume_channel(self):
        paused = {"360_SCALP"}
        handler = _make_handler(paused_channels=paused)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/resume_channel 360_SCALP", ADMIN_CHAT_ID)
        assert "360_SCALP" not in paused

    @pytest.mark.asyncio
    async def test_set_confidence_threshold(self):
        overrides: dict = {}
        handler = _make_handler(confidence_overrides=overrides)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command(
                "/set_confidence_threshold 360_SCALP 75.0", ADMIN_CHAT_ID
            )
        assert "360_SCALP" in overrides
        assert overrides["360_SCALP"] == 75.0

    @pytest.mark.asyncio
    async def test_set_confidence_threshold_invalid_value(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command(
                "/set_confidence_threshold 360_SCALP abc", ADMIN_CHAT_ID
            )
        call_args = handler._telegram.send_message.call_args[0]
        assert "number" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_subscribe_alerts(self):
        subs: set = set()
        handler = _make_handler(alert_subscribers=subs)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/subscribe_alerts", ADMIN_CHAT_ID)
        assert ADMIN_CHAT_ID in subs

    @pytest.mark.asyncio
    async def test_set_free_channel_limit_allows_zero(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/set_free_channel_limit 0", ADMIN_CHAT_ID)
        assert handler.free_channel_limit == 0
        handler._router.set_free_limit.assert_called_once_with(0)


class TestUserCommands:
    @pytest.mark.asyncio
    async def test_signals_empty(self):
        handler = _make_handler()
        handler._router.active_signals = {}
        await handler._handle_command("/signals", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "No active signals" in call_args[1]

    @pytest.mark.asyncio
    async def test_subscribe(self):
        handler = _make_handler()
        await handler._handle_command("/subscribe", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "subscribed" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        handler = _make_handler()
        await handler._handle_command("/unsubscribe", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "unsubscribed" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_signal_history_empty(self):
        handler = _make_handler(signal_history=[])
        await handler._handle_command("/signal_history", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "No completed" in call_args[1]

    @pytest.mark.asyncio
    async def test_free_signals_empty(self):
        handler = _make_handler()
        handler._router.active_signals = {}
        await handler._handle_command("/free_signals", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "No free signals" in call_args[1]

    @pytest.mark.asyncio
    async def test_unknown_command_returns_help(self):
        handler = _make_handler()
        await handler._handle_command("/unknown_cmd_xyz", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "Available commands" in call_args[1]


class TestCircuitBreakerCommands:
    @pytest.mark.asyncio
    async def test_circuit_breaker_status_no_cb(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/circuit_breaker_status", ADMIN_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "not enabled" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_status_with_cb(self):
        cb = MagicMock()
        cb.status_text.return_value = "✅ Circuit Breaker: OK"
        handler = _make_handler(circuit_breaker=cb)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/circuit_breaker_status", ADMIN_CHAT_ID)
        cb.status_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker(self):
        cb = MagicMock()
        handler = _make_handler(circuit_breaker=cb)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/reset_circuit_breaker", ADMIN_CHAT_ID)
        cb.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_stats_no_tracker(self):
        handler = _make_handler()
        await handler._handle_command("/stats", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "not enabled" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_stats_with_tracker(self):
        tracker = MagicMock()
        tracker.format_stats_message.return_value = "📊 Stats"
        handler = _make_handler(performance_tracker=tracker)
        await handler._handle_command("/stats", USER_CHAT_ID)
        tracker.format_stats_message.assert_called_once()


class TestCommandAliases:
    @pytest.mark.asyncio
    async def test_status_alias_calls_engine_status(self):
        handler = _make_handler()
        handler._signal_queue.qsize = AsyncMock(return_value=0)
        handler._pair_mgr.pairs = {}
        handler._router.active_signals = {}
        handler._tasks = []
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/status", ADMIN_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "Engine Status" in call_args[1]
