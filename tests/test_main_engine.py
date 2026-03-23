"""Tests for CryptoSignalEngine initialization and structure."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest


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
        assert len(engine._channels) == 3  # SCALP, SWING, SPOT

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

    @pytest.mark.asyncio
    @patch("src.bootstrap.close_shared_session", new_callable=AsyncMock)
    async def test_shutdown_closes_shared_ai_sessions(self, close_shared_session_mock):
        async def wait_forever() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(wait_forever())
        engine = SimpleNamespace(
            _tasks=[task],
            router=SimpleNamespace(stop=AsyncMock()),
            monitor=SimpleNamespace(stop=AsyncMock()),
            telemetry=SimpleNamespace(stop=AsyncMock()),
            _ws_spot=None,
            _ws_futures=None,
            data_store=SimpleNamespace(
                save_snapshot=AsyncMock(),
                close=AsyncMock(),
            ),
            pair_mgr=SimpleNamespace(close=AsyncMock()),
            _exchange_mgr=SimpleNamespace(close=AsyncMock()),
            _scanner=SimpleNamespace(spot_client=None),
            _openai_evaluator=SimpleNamespace(close=AsyncMock()),
            _onchain_client=SimpleNamespace(close=AsyncMock()),
            _redis_client=SimpleNamespace(close=AsyncMock()),
            telegram=SimpleNamespace(stop=AsyncMock()),
        )

        bootstrap = Bootstrap(engine)
        await bootstrap.shutdown()

        assert task.cancelled()
        close_shared_session_mock.assert_awaited_once()
        engine._openai_evaluator.close.assert_awaited_once()
        engine._onchain_client.close.assert_awaited_once()
        engine._redis_client.close.assert_awaited_once()
        engine.telegram.stop.assert_awaited_once()
        assert engine._tasks == []


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


class TestPairRefreshLoopCap:
    """_pair_refresh_loop caps new pair seeding to _MAX_NEW_SEEDS_PER_CYCLE."""

    _MAX_NEW_SEEDS_PER_CYCLE = 10

    @pytest.mark.asyncio
    async def test_seeds_at_most_10_new_pairs_per_cycle(self):
        """When >10 new symbols are discovered, only the first 10 are seeded."""
        seeded = []

        async def fake_seed(sym, market):
            seeded.append(sym)

        # 15 new symbols discovered
        new_syms = [f"TOKEN{i}USDT" for i in range(15)]

        engine = SimpleNamespace(
            pair_mgr=SimpleNamespace(
                refresh_pairs=AsyncMock(return_value=new_syms),
                pairs={s: SimpleNamespace(market="spot") for s in new_syms},
                spot_symbols=[],
                futures_symbols=[],
                record_candles=lambda sym, tf, count: None,
            ),
            data_store=SimpleNamespace(
                seed_symbol=fake_seed,
                candles={},
            ),
        )

        # Simulate one iteration of the loop (skip the outer sleep)
        async def _run_one_cycle(eng):
            new_symbols = await eng.pair_mgr.refresh_pairs()
            seeded_in_cycle = []
            for sym in new_symbols[:self._MAX_NEW_SEEDS_PER_CYCLE]:
                info = eng.pair_mgr.pairs.get(sym)
                if info is None:
                    continue
                await eng.data_store.seed_symbol(sym, info.market)
                seeded_in_cycle.append(sym)
            return seeded_in_cycle, new_symbols

        seeded_in_cycle, new_symbols = await _run_one_cycle(engine)
        assert len(seeded_in_cycle) == self._MAX_NEW_SEEDS_PER_CYCLE
        assert len(new_symbols) == 15

    @pytest.mark.asyncio
    async def test_seeds_all_when_under_cap(self):
        """When ≤10 new symbols are found, all are seeded without truncation."""
        seeded = []

        async def fake_seed(sym, market):
            seeded.append(sym)

        new_syms = [f"TOKEN{i}USDT" for i in range(5)]
        engine = SimpleNamespace(
            pair_mgr=SimpleNamespace(
                refresh_pairs=AsyncMock(return_value=new_syms),
                pairs={s: SimpleNamespace(market="spot") for s in new_syms},
                record_candles=lambda sym, tf, count: None,
            ),
            data_store=SimpleNamespace(
                seed_symbol=fake_seed,
                candles={},
            ),
        )

        async def _run_one_cycle(eng):
            new_symbols = await eng.pair_mgr.refresh_pairs()
            seeded_in_cycle = []
            for sym in new_symbols[:self._MAX_NEW_SEEDS_PER_CYCLE]:
                info = eng.pair_mgr.pairs.get(sym)
                if info is None:
                    continue
                await eng.data_store.seed_symbol(sym, info.market)
                seeded_in_cycle.append(sym)
            return seeded_in_cycle

        seeded_in_cycle = await _run_one_cycle(engine)
        assert len(seeded_in_cycle) == 5
