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


class TestWelcomeCommands:
    """Tests for /start and /help welcome message commands."""

    @pytest.mark.asyncio
    async def test_start_sends_welcome_message(self):
        handler = _make_handler()
        await handler._handle_command("/start", USER_CHAT_ID)
        handler._telegram.send_message.assert_called_once()
        args = handler._telegram.send_message.call_args[0]
        assert args[0] == USER_CHAT_ID
        assert "360 Crypto Eye" in args[1]

    @pytest.mark.asyncio
    async def test_help_sends_welcome_message(self):
        handler = _make_handler()
        await handler._handle_command("/help", USER_CHAT_ID)
        handler._telegram.send_message.assert_called_once()
        args = handler._telegram.send_message.call_args[0]
        assert "360 Crypto Eye" in args[1]

    @pytest.mark.asyncio
    async def test_start_not_admin_gated(self):
        """Even non-admin users must receive the welcome message for /start."""
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/start", USER_CHAT_ID)
        args = handler._telegram.send_message.call_args[0]
        assert "restricted" not in args[1].lower()

    def test_get_welcome_message_returns_string(self):
        handler = _make_handler()
        msg = handler.get_welcome_message()
        assert isinstance(msg, str)
        assert len(msg) <= 4096
        assert "360 Crypto Eye" in msg


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
        assert "Commands" in call_args[1]


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
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/stats", ADMIN_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "not enabled" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_stats_with_tracker(self):
        tracker = MagicMock()
        tracker.format_stats_message.return_value = "📊 Stats"
        handler = _make_handler(performance_tracker=tracker)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/stats", ADMIN_CHAT_ID)
        tracker.format_stats_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_stats_no_tracker(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/reset_stats", ADMIN_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "not enabled" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_reset_stats_all(self):
        tracker = MagicMock()
        tracker.reset_stats.return_value = 5
        handler = _make_handler(performance_tracker=tracker)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/reset_stats", ADMIN_CHAT_ID)
        tracker.reset_stats.assert_called_once_with(channel=None)
        call_args = handler._telegram.send_message.call_args[0]
        assert "5 records cleared" in call_args[1]

    @pytest.mark.asyncio
    async def test_reset_stats_channel(self):
        tracker = MagicMock()
        tracker.reset_stats.return_value = 3
        handler = _make_handler(performance_tracker=tracker)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/reset_stats 360_SCALP", ADMIN_CHAT_ID)
        tracker.reset_stats.assert_called_once_with(channel="360_SCALP")
        call_args = handler._telegram.send_message.call_args[0]
        assert "3 records cleared" in call_args[1]
        assert "360_SCALP" in call_args[1]


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


class TestSignalStatsCommands:
    """Tests for /signal_stats (user), /real_stats (admin), and /stats (admin) commands."""

    @pytest.mark.asyncio
    async def test_signal_stats_accessible_to_non_admin(self):
        """Non-admin users must be able to use /signal_stats."""
        tracker = MagicMock()
        tracker.format_signal_quality_stats_message.return_value = "🎯 Signal Quality Stats"
        handler = _make_handler(performance_tracker=tracker)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/signal_stats", USER_CHAT_ID)
        # Must NOT say "restricted"
        call_args = handler._telegram.send_message.call_args[0]
        assert "restricted" not in call_args[1].lower()
        assert "Signal Quality Stats" in call_args[1]
        tracker.format_signal_quality_stats_message.assert_called_once_with(channel=None)

    @pytest.mark.asyncio
    async def test_signal_stats_with_channel_arg(self):
        """Non-admin users can pass a channel arg to /signal_stats."""
        tracker = MagicMock()
        tracker.format_signal_quality_stats_message.return_value = "🎯 Stats for 360_SCALP"
        handler = _make_handler(performance_tracker=tracker)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/signal_stats 360_SCALP", USER_CHAT_ID)
        tracker.format_signal_quality_stats_message.assert_called_once_with(channel="360_SCALP")

    @pytest.mark.asyncio
    async def test_signal_stats_no_tracker(self):
        """Non-admin user with no performance tracker gets informative message."""
        handler = _make_handler()  # no performance_tracker
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/signal_stats", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "not enabled" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_real_stats_blocked_for_non_admin(self):
        """Non-admin users must be blocked from /real_stats."""
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/real_stats", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "restricted" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_real_stats_admin_shows_account_pnl(self):
        """Admin user can use /real_stats and gets the account PnL stats."""
        tracker = MagicMock()
        tracker.format_stats_message.return_value = "📊 Account PnL Stats"
        handler = _make_handler(performance_tracker=tracker)
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/real_stats", ADMIN_CHAT_ID)
        tracker.format_stats_message.assert_called_once_with(channel=None)
        call_args = handler._telegram.send_message.call_args[0]
        assert "Account PnL Stats" in call_args[1]

    @pytest.mark.asyncio
    async def test_stats_blocked_for_non_admin(self):
        """/stats must remain blocked for non-admin users."""
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/stats", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "restricted" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_unknown_command_help_includes_signal_stats(self):
        """Unknown command fallback help text must include /signal_stats in the User section."""
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/unknown_xyz", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "signal\\_stats" in call_args[1]

    @pytest.mark.asyncio
    async def test_unknown_command_help_includes_real_stats(self):
        """Unknown command fallback help text must include /stats (which covers real_stats) in the Admin section."""
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/unknown_xyz", ADMIN_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "stats" in call_args[1].lower()


class TestBacktestCommands:
    """Tests for /backtest, /backtest_all, and /backtest_config commands."""

    @pytest.mark.asyncio
    async def test_backtest_blocked_for_non_admin(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest BTCUSDT", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "restricted" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_backtest_no_args_shows_usage(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest", ADMIN_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "Usage" in call_args[1]

    @pytest.mark.asyncio
    async def test_backtest_no_data_returns_error(self):
        handler = _make_handler()
        handler._data_store.candles = {}
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest BTCUSDT", ADMIN_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "No candle data" in call_args[1]

    @pytest.mark.asyncio
    async def test_backtest_runs_and_sends_results(self):
        import numpy as np
        from src.backtester import BacktestResult

        handler = _make_handler()
        candles = {
            "open": np.ones(200) * 100.0,
            "high": np.ones(200) * 101.0,
            "low": np.ones(200) * 99.0,
            "close": np.ones(200) * 100.0,
            "volume": np.ones(200) * 1000.0,
        }
        handler._data_store.candles = {"BTCUSDT": {"5m": candles, "1m": candles}}

        fake_result = BacktestResult(
            channel="360_SCALP",
            total_signals=10,
            wins=7,
            losses=3,
            win_rate=70.0,
            avg_rr=1.5,
            total_pnl_pct=5.0,
            max_drawdown=1.2,
        )

        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID), \
             patch("asyncio.to_thread", return_value=[fake_result]):
            await handler._handle_command("/backtest BTCUSDT", ADMIN_CHAT_ID)

        # Acknowledgement + result messages
        assert handler._telegram.send_message.call_count >= 2
        result_call = handler._telegram.send_message.call_args_list[-1][0]
        assert "BTCUSDT" in result_call[1]
        assert "360_SCALP" in result_call[1]

    @pytest.mark.asyncio
    async def test_backtest_error_sends_error_message(self):
        import numpy as np

        handler = _make_handler()
        candles = {
            "open": np.ones(200) * 100.0,
            "high": np.ones(200) * 101.0,
            "low": np.ones(200) * 99.0,
            "close": np.ones(200) * 100.0,
            "volume": np.ones(200) * 1000.0,
        }
        handler._data_store.candles = {"BTCUSDT": {"5m": candles}}

        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID), \
             patch("asyncio.to_thread", side_effect=RuntimeError("test error")):
            await handler._handle_command("/backtest BTCUSDT", ADMIN_CHAT_ID)

        last_call = handler._telegram.send_message.call_args_list[-1][0]
        assert "failed" in last_call[1].lower() or "error" in last_call[1].lower()

    @pytest.mark.asyncio
    async def test_backtest_all_blocked_for_non_admin(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest_all", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "restricted" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_backtest_all_no_data_returns_error(self):
        handler = _make_handler()
        handler._data_store.candles = {}
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest_all", ADMIN_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "No candle data" in call_args[1]

    @pytest.mark.asyncio
    async def test_backtest_all_aggregates_results(self):
        import numpy as np
        from src.backtester import BacktestResult

        handler = _make_handler()
        candles = {
            "open": np.ones(200) * 100.0,
            "high": np.ones(200) * 101.0,
            "low": np.ones(200) * 99.0,
            "close": np.ones(200) * 100.0,
            "volume": np.ones(200) * 1000.0,
        }
        handler._data_store.candles = {
            "BTCUSDT": {"5m": candles, "1m": candles},
            "ETHUSDT": {"5m": candles, "1m": candles},
        }

        fake_result = BacktestResult(
            channel="360_SCALP",
            total_signals=8,
            wins=5,
            losses=3,
            win_rate=62.5,
            total_pnl_pct=3.0,
            max_drawdown=1.0,
        )

        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID), \
             patch("asyncio.to_thread", return_value=[fake_result]):
            await handler._handle_command("/backtest_all", ADMIN_CHAT_ID)

        # At least acknowledgement + summary message
        assert handler._telegram.send_message.call_count >= 2
        last_call = handler._telegram.send_message.call_args_list[-1][0]
        assert "360_SCALP" in last_call[1]

    @pytest.mark.asyncio
    async def test_backtest_config_show_defaults(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest_config", ADMIN_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "Fee" in call_args[1]
        assert "Slippage" in call_args[1]
        assert "Lookahead" in call_args[1]
        assert "Min Window" in call_args[1]

    @pytest.mark.asyncio
    async def test_backtest_config_update_fee(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest_config fee 0.10", ADMIN_CHAT_ID)
        assert handler._bt_fee_pct == pytest.approx(0.10)
        call_args = handler._telegram.send_message.call_args[0]
        assert "fee" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_backtest_config_update_slippage(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest_config slippage 0.05", ADMIN_CHAT_ID)
        assert handler._bt_slippage_pct == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_backtest_config_update_lookahead(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest_config lookahead 30", ADMIN_CHAT_ID)
        assert handler._bt_lookahead == 30

    @pytest.mark.asyncio
    async def test_backtest_config_update_min_window(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest_config min_window 100", ADMIN_CHAT_ID)
        assert handler._bt_min_window == 100

    @pytest.mark.asyncio
    async def test_backtest_config_invalid_key(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest_config badkey 1.0", ADMIN_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "Unknown config key" in call_args[1]

    @pytest.mark.asyncio
    async def test_backtest_config_invalid_value(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest_config fee notanumber", ADMIN_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "Invalid value" in call_args[1]

    @pytest.mark.asyncio
    async def test_backtest_config_blocked_for_non_admin(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/backtest_config", USER_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "restricted" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_backtest_config_defaults_set_on_init(self):
        handler = _make_handler()
        assert handler._bt_fee_pct == pytest.approx(0.08)
        assert handler._bt_slippage_pct == pytest.approx(0.02)
        assert handler._bt_lookahead == 20
        assert handler._bt_min_window == 50

    @pytest.mark.asyncio
    async def test_help_text_includes_backtest_commands(self):
        handler = _make_handler()
        with patch("src.commands.TELEGRAM_ADMIN_CHAT_ID", ADMIN_CHAT_ID):
            await handler._handle_command("/unknown_xyz", ADMIN_CHAT_ID)
        call_args = handler._telegram.send_message.call_args[0]
        assert "backtest" in call_args[1].lower()
