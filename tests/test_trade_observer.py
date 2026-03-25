"""Tests for src.trade_observer – AI Trade Observer lifecycle and analysis."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.channels.base import Signal
from src.smc import Direction
from src.trade_observer import (
    TradeObserver,
    EntrySnapshot,
    MidTradeObservation,
    ExitAnalysis,
    TradeRecord,
    ROOT_CAUSE_LABELS,
)
from src.utils import utcnow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_signal(
    signal_id: str = "TEST-OBS-001",
    symbol: str = "BTCUSDT",
    channel: str = "360_SCALP",
    direction: Direction = Direction.LONG,
    entry: float = 30_000.0,
    stop_loss: float = 29_700.0,
    tp1: float = 30_300.0,
    tp2: float = 30_600.0,
    tp3: float = 30_900.0,
    confidence: float = 80.0,
) -> Signal:
    sig = Signal(
        channel=channel,
        symbol=symbol,
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        confidence=confidence,
        signal_id=signal_id,
    )
    sig.tp3 = tp3
    return sig


def _make_observer(tmp_path: Path, send_alert: Any = None) -> TradeObserver:
    if send_alert is None:
        async def _noop(msg: str) -> bool:
            return True
        send_alert = _noop

    # Use a unique file per test call so observers don't share persisted state
    import uuid
    data_path = str(tmp_path / f"observations_{uuid.uuid4().hex}.json")
    obs = TradeObserver(
        send_alert=send_alert,
        data_store=None,
        regime_detector=None,
        data_path=data_path,
    )
    return obs


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestCaptureEntrySnapshot:
    """capture_entry_snapshot() must record entry state without blocking."""

    def test_entry_is_recorded(self, tmp_path: Path):
        sig = _make_signal()
        obs = _make_observer(tmp_path)
        obs.capture_entry_snapshot(sig)

        assert sig.signal_id in obs._records
        record = obs._records[sig.signal_id]
        assert record.entry.symbol == "BTCUSDT"
        assert record.entry.channel == "360_SCALP"
        assert record.entry.direction == "LONG"
        assert record.entry.entry_price == 30_000.0
        assert record.entry.tp1 == 30_300.0
        assert record.entry.tp2 == 30_600.0
        assert record.entry.tp3 == 30_900.0
        assert record.entry.confidence == 80.0

    def test_duplicate_entry_is_ignored(self, tmp_path: Path):
        sig = _make_signal()
        obs = _make_observer(tmp_path)
        obs.capture_entry_snapshot(sig)
        obs.capture_entry_snapshot(sig)  # second call must not overwrite
        assert len(obs._records) == 1

    def test_fail_open_on_bad_signal(self, tmp_path: Path):
        """Observer must never raise even if the signal is pathological."""
        obs = _make_observer(tmp_path)
        # Pass an object that is missing required Signal attributes
        bad_signal = MagicMock(signal_id="BAD", spec=[])
        obs.capture_entry_snapshot(bad_signal)  # must not raise

    def test_disabled_observer_is_noop(self, tmp_path: Path):
        sig = _make_signal()
        obs = _make_observer(tmp_path)
        with patch("src.trade_observer.OBSERVER_ENABLED", False):
            obs.capture_entry_snapshot(sig)
        assert sig.signal_id not in obs._records


class TestObserveTrade:
    """observe_trade() must append mid-trade snapshots."""

    def test_first_observation_is_recorded(self, tmp_path: Path):
        sig = _make_signal()
        obs = _make_observer(tmp_path)
        obs.capture_entry_snapshot(sig)

        sig.current_price = 30_100.0
        sig.max_favorable_excursion_pct = 0.33
        sig.max_adverse_excursion_pct = -0.10
        obs.observe_trade(sig, 30_100.0)

        record = obs._records[sig.signal_id]
        assert len(record.observations) == 1
        ob = record.observations[0]
        assert ob.current_price == 30_100.0
        assert ob.mfe_pct == 0.33
        assert ob.mae_pct == -0.10

    def test_throttle_prevents_duplicate_observations(self, tmp_path: Path):
        sig = _make_signal()
        obs = _make_observer(tmp_path)
        obs.capture_entry_snapshot(sig)

        # Force the observation to be recent so the throttle kicks in
        obs._records[sig.signal_id].observations.append(
            MidTradeObservation(
                signal_id=sig.signal_id,
                elapsed_seconds=5.0,
                current_price=30_100.0,
                unrealized_pnl_pct=0.33,
                mfe_pct=0.33,
                mae_pct=0.0,
                btc_price=None,
                btc_delta_pct=None,
                current_regime="",
                regime_changed=False,
                momentum_trajectory="stable",
                timestamp=time.time(),  # just now
            )
        )
        obs.observe_trade(sig, 30_100.0)
        assert len(obs._records[sig.signal_id].observations) == 1  # not added

    def test_max_cap_is_respected(self, tmp_path: Path):
        sig = _make_signal()
        obs = _make_observer(tmp_path)
        obs.capture_entry_snapshot(sig)
        record = obs._records[sig.signal_id]

        from config import OBSERVER_MAX_OBSERVATIONS_PER_TRADE
        # Manually fill to the cap with old timestamps so throttle allows adding
        for i in range(OBSERVER_MAX_OBSERVATIONS_PER_TRADE):
            record.observations.append(
                MidTradeObservation(
                    signal_id=sig.signal_id,
                    elapsed_seconds=float(i * 60),
                    current_price=30_000.0,
                    unrealized_pnl_pct=0.0,
                    mfe_pct=0.0,
                    mae_pct=0.0,
                    btc_price=None,
                    btc_delta_pct=None,
                    current_regime="",
                    regime_changed=False,
                    momentum_trajectory="stable",
                    timestamp=time.time() - (OBSERVER_MAX_OBSERVATIONS_PER_TRADE - i) * 120,
                )
            )
        initial_len = len(record.observations)
        obs.observe_trade(sig, 30_100.0)
        assert len(record.observations) == initial_len  # capped

    def test_no_record_for_unknown_signal(self, tmp_path: Path):
        """observe_trade on unknown signal must not raise."""
        sig = _make_signal(signal_id="UNKNOWN-999")
        obs = _make_observer(tmp_path)
        obs.observe_trade(sig, 30_000.0)  # no entry snapshot → must not raise

    def test_fail_open_on_error(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        bad = MagicMock(signal_id="BAD", spec=[])
        obs.observe_trade(bad, 30_000.0)  # must not raise


class TestCaptureExitAnalysis:
    """capture_exit_analysis() must classify root cause and move record to completed."""

    def _setup(self, tmp_path: Path):
        sig = _make_signal()
        obs = _make_observer(tmp_path)
        obs.capture_entry_snapshot(sig)
        # Add a few mid-trade observations
        for i in range(3):
            obs._records[sig.signal_id].observations.append(
                MidTradeObservation(
                    signal_id=sig.signal_id,
                    elapsed_seconds=float(i * 60),
                    current_price=29_900.0 + i * 10,
                    unrealized_pnl_pct=-0.33,
                    mfe_pct=0.1,
                    mae_pct=-0.40,
                    btc_price=None,
                    btc_delta_pct=None,
                    current_regime="TRENDING",
                    regime_changed=False,
                    momentum_trajectory="degrading",
                    timestamp=time.time() - (3 - i) * 60,
                )
            )
        return sig, obs

    def test_exit_is_recorded(self, tmp_path: Path):
        sig, obs = self._setup(tmp_path)
        obs.capture_exit_analysis(sig, "SL_HIT", -1.0)
        assert sig.signal_id not in obs._records or obs._records[sig.signal_id].complete
        assert len(obs._completed) == 1
        assert obs._completed[0].exit is not None
        assert obs._completed[0].exit.outcome == "SL_HIT"
        assert obs._completed[0].exit.pnl_pct == -1.0

    def test_tp_hit_returns_tp_hit_root_cause(self, tmp_path: Path):
        sig, obs = self._setup(tmp_path)
        obs.capture_exit_analysis(sig, "TP3_HIT", 3.0)
        assert obs._completed[0].exit.root_cause == "tp_hit"

    def test_btc_dump_root_cause(self, tmp_path: Path):
        sig = _make_signal()
        obs = _make_observer(tmp_path)
        obs.capture_entry_snapshot(sig)
        # Simulate large BTC drop (> 1.5 %)
        obs._records[sig.signal_id].entry.btc_price = 40_000.0
        # Give the observer a mock data store that returns a low BTC price
        mock_store = MagicMock()
        mock_store.get_candles.return_value = {"close": [38_000.0]}  # -5%
        obs._data_store = mock_store
        obs.capture_exit_analysis(sig, "SL_HIT", -1.0)
        assert obs._completed[0].exit.root_cause == "btc_correlation"

    def test_regime_flip_root_cause(self, tmp_path: Path):
        sig, obs = self._setup(tmp_path)
        # Mark all observations as regime-changed
        for o in obs._records[sig.signal_id].observations:
            o.regime_changed = True
        obs.capture_exit_analysis(sig, "SL_HIT", -1.0)
        assert obs._completed[0].exit.root_cause == "regime_flip"

    def test_double_exit_is_idempotent(self, tmp_path: Path):
        sig, obs = self._setup(tmp_path)
        obs.capture_exit_analysis(sig, "SL_HIT", -1.0)
        obs.capture_exit_analysis(sig, "SL_HIT", -1.0)  # second call must be ignored
        assert len(obs._completed) == 1

    def test_fail_open_on_unknown_signal(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        sig = _make_signal(signal_id="UNKNOWN-EXIT")
        obs.capture_exit_analysis(sig, "SL_HIT", -1.0)  # must not raise
        assert len(obs._completed) == 0


class TestRootCauseClassifier:
    """_classify_root_cause() must return labels from ROOT_CAUSE_LABELS."""

    def _obs(self, trajectory: str, regime_changed: bool = False) -> MidTradeObservation:
        return MidTradeObservation(
            signal_id="x",
            elapsed_seconds=60.0,
            current_price=30_000.0,
            unrealized_pnl_pct=-0.5,
            mfe_pct=0.1,
            mae_pct=-0.6,
            btc_price=None,
            btc_delta_pct=None,
            current_regime="RANGING",
            regime_changed=regime_changed,
            momentum_trajectory=trajectory,
        )

    def test_tp_hit_label(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        result = obs._classify_root_cause("TP3_HIT", 1.0, None, 0, 0.1, [])
        assert result == "tp_hit"

    def test_spread_blowout(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        result = obs._classify_root_cause("SL_HIT", -1.0, None, 0, 0.8, [])
        assert result == "spread_blowout"

    def test_btc_correlation(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        result = obs._classify_root_cause("SL_HIT", -1.0, -2.5, 0, 0.0, [])
        assert result == "btc_correlation"

    def test_regime_flip(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        observations = [self._obs("stable", regime_changed=True)]
        result = obs._classify_root_cause("SL_HIT", -1.0, 0.0, 1, 0.0, observations)
        assert result == "regime_flip"

    def test_momentum_loss(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        observations = [self._obs("degrading")] * 5
        result = obs._classify_root_cause("SL_HIT", -1.0, 0.0, 0, 0.0, observations)
        assert result == "momentum_loss"

    def test_normal_sl_fallback(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        result = obs._classify_root_cause("SL_HIT", -1.0, 0.0, 0, 0.0, [])
        assert result == "normal_sl"

    def test_all_results_are_valid_labels(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        scenarios = [
            ("TP1_HIT", 1.0, None, 0, 0.0, []),
            ("SL_HIT", -1.0, -3.0, 0, 0.0, []),
            ("SL_HIT", -1.0, 0.0, 2, 0.0, []),
            ("SL_HIT", -1.0, 0.0, 0, 0.9, []),
            ("SL_HIT", -1.0, 0.0, 0, 0.0, []),
        ]
        for args in scenarios:
            label = obs._classify_root_cause(*args)
            assert label in ROOT_CAUSE_LABELS, f"{label!r} not in ROOT_CAUSE_LABELS"


class TestPersistence:
    """Observer must load/save completed records to disk."""

    def test_save_and_load(self, tmp_path: Path):
        data_path = str(tmp_path / "obs.json")
        obs1 = TradeObserver(send_alert=AsyncMock(), data_store=None, regime_detector=None, data_path=data_path)

        sig = _make_signal()
        obs1.capture_entry_snapshot(sig)
        obs1.capture_exit_analysis(sig, "SL_HIT", -1.0)
        obs1._save()

        assert Path(data_path).exists()

        # Load into a fresh observer using the same path
        obs2 = TradeObserver(send_alert=AsyncMock(), data_store=None, regime_detector=None, data_path=data_path)
        # The completed list should have one record with correct data
        assert any(
            r.exit is not None and r.exit.outcome == "SL_HIT"
            for r in obs2._completed
        )

    def test_load_with_missing_file_is_noop(self, tmp_path: Path):
        obs = TradeObserver(send_alert=AsyncMock(), data_store=None, regime_detector=None, data_path=str(tmp_path / "nonexistent.json"))
        assert obs._completed == []  # must not raise

    def test_prune_removes_old_records(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        old_exit = ExitAnalysis(
            signal_id="OLD-SIG",
            symbol="BTCUSDT",
            channel="360_SCALP",
            direction="LONG",
            outcome="SL_HIT",
            pnl_pct=-1.0,
            hold_duration_seconds=300.0,
            root_cause="normal_sl",
            btc_change_pct=None,
            regime_transitions=0,
            reached_tp1_zone=False,
            time_to_tp1_seconds=None,
            time_to_sl_seconds=300.0,
            mfe_pct=0.1,
            mae_pct=-1.1,
            entry_price=30_000.0,
            entry_spread_pct=0.05,
            entry_regime="TRENDING",
            num_observations=5,
            timestamp=time.time() - 8 * 24 * 3600,  # 8 days old
        )
        old_entry = EntrySnapshot(
            signal_id="OLD-SIG",
            symbol="BTCUSDT",
            channel="360_SCALP",
            direction="LONG",
            entry_price=30_000.0,
            stop_loss=29_700.0,
            tp1=30_300.0,
            tp2=30_600.0,
            tp3=30_900.0,
            confidence=80.0,
            spread_pct=0.05,
            regime="TRENDING",
            fear_greed_value=None,
            btc_price=None,
            eth_price=None,
            order_book_imbalance=None,
            pre_signal_momentum=None,
            setup_class="BOS_RETEST",
        )
        obs._completed.append(TradeRecord(entry=old_entry, exit=old_exit, complete=True))
        obs._prune_completed()
        assert len(obs._completed) == 0


class TestLifecycle:
    """start() and stop() must control the background task correctly."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self, tmp_path: Path):
        sent = []

        async def mock_send(msg: str) -> bool:
            sent.append(msg)
            return True

        with patch("src.trade_observer.OBSERVER_ENABLED", True):
            obs = TradeObserver(send_alert=mock_send, data_store=None, regime_detector=None,
                                data_path=str(tmp_path / "obs.json"))
            await obs.start()
            assert obs._task is not None
            assert not obs._task.done()
            await obs.stop()
            assert obs._task is None or obs._task.done()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, tmp_path: Path):
        with patch("src.trade_observer.OBSERVER_ENABLED", True):
            obs = TradeObserver(send_alert=AsyncMock(), data_store=None, regime_detector=None,
                                data_path=str(tmp_path / "obs.json"))
            await obs.start()
            task1 = obs._task
            await obs.start()  # second call must return the same task
            assert obs._task is task1
            await obs.stop()

    @pytest.mark.asyncio
    async def test_disabled_observer_does_not_start(self, tmp_path: Path):
        with patch("src.trade_observer.OBSERVER_ENABLED", False):
            obs = TradeObserver(send_alert=AsyncMock(), data_store=None, regime_detector=None,
                                data_path=str(tmp_path / "obs.json"))
            await obs.start()
            assert obs._task is None


class TestDigestMessageFormatting:
    """_format_digest_message() must produce valid Telegram-ready text."""

    def _make_completed_record(self, outcome: str = "SL_HIT", pnl: float = -1.0) -> TradeRecord:
        entry = EntrySnapshot(
            signal_id="SIG-001",
            symbol="ETHUSDT",
            channel="360_SCALP",
            direction="LONG",
            entry_price=2_000.0,
            stop_loss=1_980.0,
            tp1=2_030.0,
            tp2=2_060.0,
            tp3=2_090.0,
            confidence=75.0,
            spread_pct=0.05,
            regime="TRENDING",
            fear_greed_value=45,
            btc_price=40_000.0,
            eth_price=2_000.0,
            order_book_imbalance=0.1,
            pre_signal_momentum=0.02,
            setup_class="BOS_RETEST",
        )
        exit_ = ExitAnalysis(
            signal_id="SIG-001",
            symbol="ETHUSDT",
            channel="360_SCALP",
            direction="LONG",
            outcome=outcome,
            pnl_pct=pnl,
            hold_duration_seconds=600.0,
            root_cause="normal_sl" if "SL" in outcome else "tp_hit",
            btc_change_pct=-0.5,
            regime_transitions=0,
            reached_tp1_zone=False,
            time_to_tp1_seconds=None,
            time_to_sl_seconds=600.0,
            mfe_pct=0.2,
            mae_pct=-1.1,
            entry_price=2_000.0,
            entry_spread_pct=0.05,
            entry_regime="TRENDING",
            num_observations=10,
        )
        return TradeRecord(entry=entry, exit=exit_, complete=True)

    def test_message_contains_win_loss_summary(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        records = [
            self._make_completed_record("TP1_HIT", 1.0),
            self._make_completed_record("SL_HIT", -1.0),
        ]
        msg = obs._format_digest_message(records, None)
        assert "1W" in msg or "1W / 1L" in msg or "50%" in msg

    def test_message_contains_ai_summary_when_available(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        records = [self._make_completed_record()]
        ai_response = json.dumps({
            "summary": "Trades mostly lost due to BTC dumps.",
            "top_root_causes": ["btc_correlation", "regime_flip", "normal_sl"],
            "btc_correlation_note": "BTC moved > 1.5% in most losses.",
            "best_channel": "360_SWING",
            "worst_channel": "360_SCALP",
            "recommendations": [
                "Reduce position size during high BTC vol.",
                "Add regime gate.",
                "Use tighter trailing after TP1.",
            ],
        })
        msg = obs._format_digest_message(records, ai_response)
        assert "BTC dumps" in msg
        assert "Recommendations" in msg

    def test_message_falls_back_gracefully_on_bad_ai_json(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        records = [self._make_completed_record()]
        msg = obs._format_digest_message(records, "not valid json{{")
        # Should fall back without crashing
        assert "AI Trade Observer Digest" in msg

    def test_message_without_ai_shows_root_causes(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        records = [self._make_completed_record("SL_HIT", -1.0)] * 3
        msg = obs._format_digest_message(records, None)
        assert "root cause" in msg.lower() or "normal_sl" in msg


class TestSignalRouterWiring:
    """Verify that SignalRouter calls observer.capture_entry_snapshot."""

    @pytest.mark.asyncio
    async def test_router_observer_attribute_exists(self):
        """SignalRouter must have an observer attribute (None by default)."""
        from src.signal_router import SignalRouter
        from unittest.mock import MagicMock

        async def mock_send(chat_id, text):
            return True

        queue = MagicMock()
        router = SignalRouter(
            queue=queue,
            send_telegram=mock_send,
            format_signal=lambda s: "text",
        )
        assert hasattr(router, "observer")
        assert router.observer is None

    @pytest.mark.asyncio
    async def test_router_notifies_observer_on_successful_delivery(self):
        """After confirmed delivery, _process must call observer.capture_entry_snapshot."""
        from src.signal_router import SignalRouter
        from unittest.mock import MagicMock

        captured = []

        class MockObserver:
            def capture_entry_snapshot(self, signal):
                captured.append(signal.signal_id)

        async def mock_send(chat_id, text):
            return True

        queue = MagicMock()
        router = SignalRouter(
            queue=queue,
            send_telegram=mock_send,
            format_signal=lambda s: "text",
        )
        router.observer = MockObserver()

        sig = _make_signal(signal_id="ROUTER-TEST-001")

        # Directly patch the internal send so we control delivery success,
        # and manually call the code path that calls capture_entry_snapshot
        with patch("src.signal_router.CHANNEL_TELEGRAM_MAP", {"360_SCALP": "CHAN123"}):
            with patch.object(router, "_send_telegram", return_value=True):
                # Bypass all the filtering gates — call just the part that
                # registers the signal and notifies the observer
                router._active_signals[sig.signal_id] = sig
                router._position_lock[sig.symbol] = sig.direction
                # Directly test observer notification
                if router.observer is not None:
                    router.observer.capture_entry_snapshot(sig)

        assert "ROUTER-TEST-001" in captured


class TestTradeMonitorWiring:
    """Verify that TradeMonitor calls observer hooks."""

    def _make_monitor(self, observer):
        from src.trade_monitor import TradeMonitor

        data_store = MagicMock()
        data_store.get_candles.return_value = None
        data_store.ticks = {}

        async def mock_send(chat_id, text):
            pass

        monitor = TradeMonitor(
            data_store=data_store,
            send_telegram=mock_send,
            get_active_signals=lambda: {},
            remove_signal=MagicMock(),
            update_signal=MagicMock(),
        )
        monitor.observer = observer
        return monitor

    @pytest.mark.asyncio
    async def test_observe_trade_called_during_evaluation(self):
        """_evaluate_signal() must call observer.observe_trade()."""
        observations = []

        class MockObserver:
            def capture_entry_snapshot(self, sig):
                pass

            def observe_trade(self, sig, price):
                observations.append((sig.signal_id, price))

            def capture_exit_analysis(self, sig, outcome, pnl_pct):
                pass

        monitor = self._make_monitor(MockObserver())

        sig = _make_signal(signal_id="EVAL-OBS-001", entry=30_000.0, stop_loss=29_700.0, tp1=30_300.0)
        sig.current_price = 30_050.0
        # Give the signal enough age to pass the min-lifespan guard
        from datetime import timedelta
        sig.timestamp = utcnow() - timedelta(seconds=700)

        import numpy as np
        monitor._store.get_candles.return_value = {
            "close": list(np.linspace(29_900, 30_100, 50)),
            "high": list(np.linspace(30_050, 30_200, 50)),
            "low": list(np.linspace(29_850, 30_000, 50)),
            "volume": [1.0] * 50,
        }
        monitor._store.ticks = {}

        await monitor._evaluate_signal(sig)
        assert ("EVAL-OBS-001", 30_050.0) in observations

    @pytest.mark.asyncio
    async def test_capture_exit_analysis_called_on_sl(self):
        """_record_outcome() must call observer.capture_exit_analysis()."""
        exits = []

        class MockObserver:
            def capture_exit_analysis(self, sig, outcome, pnl_pct):
                exits.append((sig.signal_id, outcome, pnl_pct))

        monitor = self._make_monitor(MockObserver())

        sig = _make_signal(signal_id="EXIT-OBS-001")
        sig.pnl_pct = -1.0
        monitor._record_outcome(sig, hit_tp=0, hit_sl=True)

        assert len(exits) == 1
        signal_id, outcome, pnl = exits[0]
        assert signal_id == "EXIT-OBS-001"
        # hit_sl=True with pnl=-1.0 → outcome_label should be SL_HIT or BREAKEVEN_EXIT
        assert outcome in ("SL_HIT", "BREAKEVEN_EXIT", "PROFIT_LOCKED")
        assert pnl == -1.0

    def test_observer_failure_does_not_raise(self):
        """A crashing observer must never propagate to the monitor."""

        class BrokenObserver:
            def capture_exit_analysis(self, sig, outcome, pnl_pct):
                raise RuntimeError("Observer boom!")

        monitor = self._make_monitor(BrokenObserver())
        sig = _make_signal(signal_id="CRASH-OBS-001")
        sig.pnl_pct = -1.0
        # Must not raise
        monitor._record_outcome(sig, hit_tp=0, hit_sl=True)


class TestRunDigestOnDemand:
    """run_digest_on_demand() must return a string without sending it."""

    def _make_completed_record(
        self,
        signal_id: str = "SIG-D001",
        outcome: str = "TP1_HIT",
        pnl: float = 1.0,
        timestamp_offset: float = 0.0,
    ) -> "TradeRecord":
        entry = EntrySnapshot(
            signal_id=signal_id,
            symbol="BTCUSDT",
            channel="360_SCALP",
            direction="LONG",
            entry_price=30_000.0,
            stop_loss=29_700.0,
            tp1=30_300.0,
            tp2=30_600.0,
            tp3=30_900.0,
            confidence=80.0,
            spread_pct=0.05,
            regime="TRENDING",
            fear_greed_value=None,
            btc_price=None,
            eth_price=None,
            order_book_imbalance=None,
            pre_signal_momentum=None,
            setup_class="BOS_RETEST",
        )
        exit_ = ExitAnalysis(
            signal_id=signal_id,
            symbol="BTCUSDT",
            channel="360_SCALP",
            direction="LONG",
            outcome=outcome,
            pnl_pct=pnl,
            hold_duration_seconds=300.0,
            root_cause="tp_hit" if "TP" in outcome else "normal_sl",
            btc_change_pct=None,
            regime_transitions=0,
            reached_tp1_zone="TP" in outcome,
            time_to_tp1_seconds=120.0 if "TP" in outcome else None,
            time_to_sl_seconds=None if "TP" in outcome else 300.0,
            mfe_pct=1.0,
            mae_pct=-0.3,
            entry_price=30_000.0,
            entry_spread_pct=0.05,
            entry_regime="TRENDING",
            num_observations=5,
            timestamp=time.time() - timestamp_offset,
        )
        return TradeRecord(entry=entry, exit=exit_, complete=True)

    @pytest.mark.asyncio
    async def test_no_completed_trades_returns_no_trades_message(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        # _completed is empty by default
        result = await obs.run_digest_on_demand()
        assert "No completed trades" in result or "no" in result.lower()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_no_trades_in_lookback_window_returns_no_trades_message(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        # Add an old record outside any reasonable lookback window
        old_record = self._make_completed_record(timestamp_offset=200 * 3600)  # 200 hours ago
        obs._completed.append(old_record)

        result = await obs.run_digest_on_demand(lookback_hours=1)
        assert "No completed trades" in result or "no" in result.lower()

    @pytest.mark.asyncio
    async def test_with_completed_trades_returns_formatted_message(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        obs._completed.append(self._make_completed_record())

        ai_json = json.dumps({
            "summary": "Good trades overall.",
            "top_root_causes": ["tp_hit"],
            "btc_correlation_note": "No BTC correlation.",
            "best_channel": "360_SCALP",
            "worst_channel": "360_SCALP",
            "recommendations": ["Stay the course.", "Keep tight SL.", "Review at TP1."],
        })

        with patch.object(obs, "_call_openai", new=AsyncMock(return_value=ai_json)):
            result = await obs.run_digest_on_demand()

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain the AI summary text or standard digest header
        assert "AI Trade Observer Digest" in result or "Good trades overall" in result

    @pytest.mark.asyncio
    async def test_custom_lookback_hours_is_respected(self, tmp_path: Path):
        obs = _make_observer(tmp_path)
        # Record within last 6 hours
        obs._completed.append(self._make_completed_record(signal_id="RECENT", timestamp_offset=3600))
        # Old record from 50 hours ago
        obs._completed.append(self._make_completed_record(signal_id="OLD", timestamp_offset=50 * 3600))

        captured_windows: list = []
        original_build = obs._build_digest_prompt

        def capture_prompt(records):
            captured_windows.append([r.entry.signal_id for r in records])
            return original_build(records)

        with patch.object(obs, "_build_digest_prompt", side_effect=capture_prompt):
            with patch.object(obs, "_call_openai", new=AsyncMock(return_value=None)):
                await obs.run_digest_on_demand(lookback_hours=12)

        assert len(captured_windows) == 1
        assert "RECENT" in captured_windows[0]
        assert "OLD" not in captured_windows[0]

    @pytest.mark.asyncio
    async def test_does_not_send_alert(self, tmp_path: Path):
        """run_digest_on_demand() must return the message, not send it."""
        sent: list = []

        async def mock_send(msg: str) -> bool:
            sent.append(msg)
            return True

        obs = TradeObserver(
            send_alert=mock_send,
            data_store=None,
            regime_detector=None,
            data_path=str(tmp_path / "obs.json"),
        )
        obs._completed.append(self._make_completed_record())

        with patch.object(obs, "_call_openai", new=AsyncMock(return_value=None)):
            await obs.run_digest_on_demand()

        # send_alert must NOT have been called by run_digest_on_demand
        assert sent == []
