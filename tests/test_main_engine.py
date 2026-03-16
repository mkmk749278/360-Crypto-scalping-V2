"""Tests for CryptoSignalEngine initialization and structure."""

from __future__ import annotations

from unittest.mock import patch


from src.commands import CommandHandler
from src.scanner import Scanner
from src.bootstrap import Bootstrap
from src.circuit_breaker import CircuitBreaker
from src.performance_tracker import PerformanceTracker


class TestCryptoSignalEngineImport:
    def test_main_module_importable(self):
        """main.py should be importable without errors."""
        import src.main  # noqa: F401

    def test_engine_class_exists(self):
        from src.main import CryptoSignalEngine
        assert CryptoSignalEngine is not None

    def test_entry_point_functions_exist(self):
        from src.main import main, _run
        assert callable(main)
        assert callable(_run)


class TestCryptoSignalEngineInit:
    def _make_engine(self):
        """Create engine with all network calls mocked."""
        with patch("src.main.TelegramBot"), \
             patch("src.main.TelemetryCollector"), \
             patch("src.main.RedisClient"), \
             patch("src.main.SignalQueue"), \
             patch("src.main.StateCache"), \
             patch("src.main.SignalRouter"), \
             patch("src.main.TradeMonitor"), \
             patch("src.main.PairManager"), \
             patch("src.main.HistoricalDataStore"), \
             patch("src.main.PredictiveEngine"), \
             patch("src.main.ExchangeManager"), \
             patch("src.main.SMCDetector"), \
             patch("src.main.MarketRegimeDetector"):
            from src.main import CryptoSignalEngine
            return CryptoSignalEngine()

    def test_engine_has_scanner(self):
        engine = self._make_engine()
        assert isinstance(engine._scanner, Scanner)

    def test_engine_has_command_handler(self):
        engine = self._make_engine()
        assert isinstance(engine._command_handler, CommandHandler)

    def test_engine_has_bootstrap(self):
        engine = self._make_engine()
        assert isinstance(engine._bootstrap, Bootstrap)

    def test_engine_has_circuit_breaker(self):
        engine = self._make_engine()
        assert isinstance(engine._circuit_breaker, CircuitBreaker)

    def test_engine_has_performance_tracker(self):
        engine = self._make_engine()
        assert isinstance(engine._performance_tracker, PerformanceTracker)

    def test_scanner_paused_channels_shared(self):
        engine = self._make_engine()
        # Paused channels set should be the same object
        assert engine._scanner.paused_channels is engine._paused_channels

    def test_scanner_confidence_overrides_shared(self):
        engine = self._make_engine()
        assert engine._scanner.confidence_overrides is engine._confidence_overrides

    def test_scanner_has_circuit_breaker(self):
        engine = self._make_engine()
        assert engine._scanner.circuit_breaker is engine._circuit_breaker

    def test_engine_channels_count(self):
        engine = self._make_engine()
        assert len(engine._channels) == 4  # SCALP, SWING, RANGE, TAPE

    def test_signal_history_starts_empty(self):
        engine = self._make_engine()
        assert engine._signal_history == []

    def test_tasks_starts_empty(self):
        engine = self._make_engine()
        assert engine._tasks == []

    def test_signal_queue_receives_admin_alert_callback(self):
        with patch("src.main.TelegramBot") as telegram_cls, \
             patch("src.main.TelemetryCollector"), \
             patch("src.main.RedisClient"), \
             patch("src.main.SignalQueue") as signal_queue_cls, \
             patch("src.main.StateCache"), \
             patch("src.main.SignalRouter"), \
             patch("src.main.TradeMonitor"), \
             patch("src.main.PairManager"), \
             patch("src.main.HistoricalDataStore"), \
             patch("src.main.PredictiveEngine"), \
             patch("src.main.ExchangeManager"), \
             patch("src.main.SMCDetector"), \
             patch("src.main.MarketRegimeDetector"):
            from src.main import CryptoSignalEngine
            engine = CryptoSignalEngine()

        assert engine is not None
        assert signal_queue_cls.call_args.kwargs["alert_callback"] is telegram_cls.return_value.send_admin_alert


class TestBootstrapInterface:
    def test_bootstrap_has_preflight_check(self):
        assert hasattr(Bootstrap, "preflight_check")

    def test_bootstrap_has_boot(self):
        assert hasattr(Bootstrap, "boot")

    def test_bootstrap_has_shutdown(self):
        assert hasattr(Bootstrap, "shutdown")

    def test_bootstrap_has_start_websockets(self):
        assert hasattr(Bootstrap, "start_websockets")


class TestScannerInterface:
    def test_scanner_has_scan_loop(self):
        assert hasattr(Scanner, "scan_loop")

    def test_scanner_has_cooldown_methods(self):
        assert hasattr(Scanner, "_is_in_cooldown")
        assert hasattr(Scanner, "_set_cooldown")

    def test_scanner_has_scan_symbol(self):
        assert hasattr(Scanner, "_scan_symbol")


class TestCommandHandlerInterface:
    def test_command_handler_has_handle_command(self):
        assert hasattr(CommandHandler, "_handle_command")
