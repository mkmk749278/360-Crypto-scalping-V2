"""Tests for src.trade_monitor – minimum lifespan and SL/TP evaluation."""

from __future__ import annotations

from datetime import timedelta
from typing import Dict
from unittest.mock import MagicMock

import pytest

from src.channels.base import Signal
from src.smc import Direction
from src.trade_monitor import TradeMonitor
from src.utils import utcnow


def _make_signal(
    channel: str = "360_SCALP",
    symbol: str = "BTCUSDT",
    direction: Direction = Direction.LONG,
    entry: float = 30000.0,
    stop_loss: float = 29850.0,
    tp1: float = 30150.0,
    tp2: float = 30300.0,
    tp3: float = 30450.0,
    signal_id: str = "TEST-SIG-001",
    age_seconds: float = 0.0,
) -> Signal:
    sig = Signal(
        channel=channel,
        symbol=symbol,
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        confidence=85.0,
        signal_id=signal_id,
    )
    sig.tp3 = tp3
    # Backdate the timestamp to simulate a signal of `age_seconds` old
    if age_seconds > 0:
        sig.timestamp = utcnow() - timedelta(seconds=age_seconds)
    return sig


class TestMinimumLifespan:
    """The monitor must NOT trigger SL/TP checks for very new signals."""

    def _build_monitor(self, active: Dict[str, Signal]):
        removed = []
        sent = []

        async def mock_send(chat_id, text):
            sent.append((chat_id, text))

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
        return monitor, removed, sent

    @pytest.mark.asyncio
    async def test_sl_not_triggered_within_min_lifespan(self):
        """Brand-new SCALP signal (age=0) below SL should NOT be removed."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=0.0,  # just created
        )
        # Set current price below stop loss to simulate SL condition
        sig.current_price = 29800.0

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        await monitor._evaluate_signal(sig)

        # Signal must NOT be removed because the min lifespan hasn't passed
        assert sig.signal_id not in removed
        assert sig.status == "ACTIVE"

    @pytest.mark.asyncio
    async def test_sl_triggered_after_min_lifespan(self):
        """A SCALP signal older than 30s whose price is below SL SHOULD be removed."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=35.0,  # past the 30s SCALP minimum
        )
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        await monitor._evaluate_signal(sig)

        assert sig.signal_id in removed
        assert sig.status == "SL_HIT"
        assert sig.current_price == pytest.approx(29850.0)

    @pytest.mark.asyncio
    async def test_swing_min_lifespan_is_longer(self):
        """A SWING signal at age=15s (< 60s min) should NOT trigger SL."""
        sig = _make_signal(
            channel="360_SWING",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=15.0,  # below the 60s SWING minimum
        )
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        await monitor._evaluate_signal(sig)

        assert sig.signal_id not in removed
        assert sig.status == "ACTIVE"

    @pytest.mark.asyncio
    async def test_tp_not_triggered_within_min_lifespan(self):
        """TP1 should NOT fire on a brand-new signal even if price reaches TP."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            tp2=30300.0,
            age_seconds=0.0,
        )
        sig.current_price = 30200.0  # above TP1

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        await monitor._evaluate_signal(sig)

        assert sig.status == "ACTIVE"


class TestOutcomeRecording:
    """TradeMonitor must call performance_tracker and circuit_breaker on final outcomes."""

    def _build_monitor_with_mocks(self, active: Dict[str, Signal]):
        """Build a TradeMonitor wired with mock performance_tracker and circuit_breaker."""
        removed = []
        sent = []

        async def mock_send(chat_id, text):
            sent.append((chat_id, text))

        data_store = MagicMock()
        data_store.get_candles.return_value = None
        data_store.ticks = {}

        performance_tracker = MagicMock()
        circuit_breaker = MagicMock()

        monitor = TradeMonitor(
            data_store=data_store,
            send_telegram=mock_send,
            get_active_signals=lambda: dict(active),
            remove_signal=lambda sid: removed.append(sid),
            update_signal=MagicMock(),
            performance_tracker=performance_tracker,
            circuit_breaker=circuit_breaker,
        )
        return monitor, removed, sent, performance_tracker, circuit_breaker

    @pytest.mark.asyncio
    async def test_sl_hit_calls_performance_tracker(self):
        """Losing stop exits must record a semantic SL_HIT outcome."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=35.0,
        )
        sig.setup_class = "BREAKOUT_RETEST"
        sig.market_phase = "STRONG_TREND"
        sig.quality_tier = "A"
        sig.pre_ai_confidence = 78.0
        sig.post_ai_confidence = 84.0
        sig.spread_pct = 0.008
        sig.volume_24h_usd = 12_000_000.0
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        assert sig.status == "SL_HIT"
        pt.record_outcome.assert_called_once()
        call_kwargs = pt.record_outcome.call_args.kwargs
        assert call_kwargs["hit_sl"] is True
        assert call_kwargs["hit_tp"] == 0
        assert call_kwargs["signal_id"] == sig.signal_id
        assert call_kwargs["pnl_pct"] == pytest.approx(-0.5)
        assert call_kwargs["outcome_label"] == "SL_HIT"
        assert call_kwargs["setup_class"] == "BREAKOUT_RETEST"
        assert call_kwargs["market_phase"] == "STRONG_TREND"
        assert call_kwargs["quality_tier"] == "A"
        assert call_kwargs["pre_ai_confidence"] == 78.0
        assert call_kwargs["post_ai_confidence"] == 84.0

    @pytest.mark.asyncio
    async def test_sl_hit_calls_circuit_breaker(self):
        """SL_HIT must also notify circuit_breaker.record_outcome."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=35.0,
        )
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        cb.record_outcome.assert_called_once()
        call_kwargs = cb.record_outcome.call_args.kwargs
        assert call_kwargs["hit_sl"] is True
        assert call_kwargs["signal_id"] == sig.signal_id

    @pytest.mark.asyncio
    async def test_tp3_hit_calls_performance_tracker(self):
        """Full TP completion must record a semantic FULL_TP_HIT outcome."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            tp2=30300.0,
            tp3=30450.0,
            age_seconds=35.0,
        )
        sig.current_price = 30500.0  # above TP3

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        assert sig.status == "FULL_TP_HIT"
        pt.record_outcome.assert_called_once()
        call_kwargs = pt.record_outcome.call_args.kwargs
        assert call_kwargs["hit_sl"] is False
        assert call_kwargs["hit_tp"] == 3
        assert call_kwargs["pnl_pct"] == pytest.approx(1.5)
        assert call_kwargs["outcome_label"] == "FULL_TP_HIT"
        assert sig.current_price == pytest.approx(30450.0)

    @pytest.mark.asyncio
    async def test_tp1_hit_does_not_call_record_outcome(self):
        """TP1_HIT must NOT call record_outcome — signal is still active."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            tp2=30300.0,
            tp3=30450.0,
            age_seconds=35.0,
        )
        sig.current_price = 30200.0  # above TP1 but below TP2

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        assert sig.status == "TP1_HIT"
        pt.record_outcome.assert_not_called()
        cb.record_outcome.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancelled_invalid_sl_does_not_call_record_outcome(self):
        """CANCELLED (invalid SL) must NOT call record_outcome — not a real trade outcome."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=30100.0,  # invalid: SL above entry for LONG
            age_seconds=35.0,
        )
        sig.current_price = 30000.0

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        assert sig.status == "CANCELLED"
        pt.record_outcome.assert_not_called()
        cb.record_outcome.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_performance_tracker_does_not_raise(self):
        """Monitor without performance_tracker/circuit_breaker must not raise on SL_HIT."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=35.0,
        )
        sig.current_price = 29800.0

        removed = []
        sent = []

        async def mock_send(chat_id, text):
            sent.append((chat_id, text))

        data_store = MagicMock()
        data_store.get_candles.return_value = None
        data_store.ticks = {}

        monitor = TradeMonitor(
            data_store=data_store,
            send_telegram=mock_send,
            get_active_signals=lambda: {sig.signal_id: sig},
            remove_signal=lambda sid: removed.append(sid),
            update_signal=MagicMock(),
            # No performance_tracker or circuit_breaker — must not raise
        )

        await monitor._evaluate_signal(sig)

        assert sig.status == "SL_HIT"
        assert sig.signal_id in removed

    @pytest.mark.asyncio
    async def test_short_sl_uses_stop_price_for_realized_pnl(self):
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.SHORT,
            entry=30000.0,
            stop_loss=30150.0,
            tp1=29850.0,
            tp2=29700.0,
            tp3=29550.0,
            age_seconds=35.0,
        )
        sig.current_price = 30250.0

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        call_kwargs = pt.record_outcome.call_args.kwargs
        assert call_kwargs["pnl_pct"] == pytest.approx(-0.5)
        assert sig.current_price == pytest.approx(30150.0)
        assert sig.status == "SL_HIT"
        assert call_kwargs["outcome_label"] == "SL_HIT"

    @pytest.mark.asyncio
    async def test_short_tp3_uses_take_profit_price_for_realized_pnl(self):
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.SHORT,
            entry=30000.0,
            stop_loss=30150.0,
            tp1=29850.0,
            tp2=29700.0,
            tp3=29550.0,
            age_seconds=35.0,
        )
        sig.current_price = 29400.0

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        call_kwargs = pt.record_outcome.call_args.kwargs
        assert call_kwargs["pnl_pct"] == pytest.approx(1.5)
        assert call_kwargs["outcome_label"] == "FULL_TP_HIT"
        assert sig.current_price == pytest.approx(29550.0)
        assert sig.status == "FULL_TP_HIT"

    @pytest.mark.asyncio
    async def test_trailing_stop_break_even_records_zero_pnl(self):
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=35.0,
        )
        sig.status = "TP2_HIT"
        sig.stop_loss = sig.entry
        sig.current_price = 29900.0

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        call_kwargs = pt.record_outcome.call_args.kwargs
        assert call_kwargs["hit_sl"] is True
        assert call_kwargs["pnl_pct"] == pytest.approx(0.0)
        assert call_kwargs["outcome_label"] == "BREAKEVEN_EXIT"
        assert sig.status == "BREAKEVEN_EXIT"

    @pytest.mark.asyncio
    async def test_trailing_stop_profit_lock_is_not_reported_as_sl_hit(self):
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=35.0,
        )
        sig.status = "TP2_HIT"
        sig.stop_loss = 30120.0
        sig.current_price = 30090.0

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        call_kwargs = pt.record_outcome.call_args.kwargs
        assert call_kwargs["hit_sl"] is True
        assert call_kwargs["pnl_pct"] == pytest.approx(0.4)
        assert call_kwargs["outcome_label"] == "PROFIT_LOCKED"
        assert sig.status == "PROFIT_LOCKED"


class TestTrailingStopAfterTP2:
    """Trailing stop must continue to advance after TP2 moves SL to break-even."""

    def _build_monitor(self, active: Dict[str, Signal]):
        removed = []
        sent = []

        async def mock_send(chat_id, text):
            sent.append((chat_id, text))

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

    @pytest.mark.asyncio
    async def test_trailing_stop_advances_after_tp2(self):
        """After TP2 sets SL to entry, the trailing stop should still move up with price."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,  # original SL → original_sl_distance = 150
            tp1=30150.0,
            tp2=30300.0,
            tp3=30450.0,
            age_seconds=60.0,
        )
        # Simulate what happens after TP2 is hit: SL moves to entry
        sig.status = "TP2_HIT"
        sig.stop_loss = sig.entry  # break-even
        sig.original_sl_distance = 150.0  # 30000 - 29850
        sig.trailing_active = True

        # Price has moved up to 30400 (between TP2 and TP3)
        sig.current_price = 30400.0

        active = {sig.signal_id: sig}
        monitor, removed = self._build_monitor(active)

        # Invoke trailing adjustment directly
        monitor._adjust_trailing(sig)

        # trail_dist = 150 * 0.5 = 75
        # new_sl = 30400 - 75 = 30325
        # 30325 > 30000 (break-even), so stop should advance
        assert sig.stop_loss == pytest.approx(30325.0)

    @pytest.mark.asyncio
    async def test_trailing_stop_does_not_regress(self):
        """Trailing stop should never move backwards (lower for LONG)."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=60.0,
        )
        sig.status = "TP2_HIT"
        sig.stop_loss = 30200.0  # already advanced above break-even
        sig.original_sl_distance = 150.0
        sig.trailing_active = True
        # Price dips slightly – trailing should NOT regress
        sig.current_price = 30250.0  # new_sl would be 30175, below current 30200

        monitor, _ = self._build_monitor({sig.signal_id: sig})
        monitor._adjust_trailing(sig)

        assert sig.stop_loss == pytest.approx(30200.0)  # unchanged

    @pytest.mark.asyncio
    async def test_on_sl_callback_triggered_on_sl_hit(self):
        """on_sl_callback must be called with the symbol when a stop-loss is hit."""
        sl_callbacks: list = []

        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=35.0,
        )
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        removed = []
        sent = []

        async def mock_send(chat_id, text):
            sent.append((chat_id, text))

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
        monitor.on_sl_callback = sl_callbacks.append

        await monitor._evaluate_signal(sig)

        assert sig.status == "SL_HIT"
        assert sl_callbacks == ["BTCUSDT"]


class TestSignalExpiry:
    """Auto-expiry: signals older than MAX_SIGNAL_HOLD_SECONDS are closed at market."""

    def _build_monitor(self, active: Dict[str, Signal]):
        removed = []
        sent = []

        async def mock_send(chat_id, text):
            sent.append((chat_id, text))

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
        return monitor, removed, sent

    @pytest.mark.asyncio
    async def test_scalp_signal_expired_after_3600s(self):
        """A SCALP signal older than 3600s must be auto-expired at market price."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            tp2=30300.0,
            tp3=30450.0,
            age_seconds=3601.0,  # just over 1 hour
        )
        market_price = 30100.0
        sig.current_price = market_price

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        await monitor._evaluate_signal(sig)

        assert sig.signal_id in removed
        assert sig.status == "EXPIRED"
        # PnL should reflect the market exit price
        assert sig.current_price == pytest.approx(market_price)

    @pytest.mark.asyncio
    async def test_scalp_signal_not_expired_before_3600s(self):
        """A SCALP signal younger than 3600s must NOT be auto-expired."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            tp2=30300.0,
            age_seconds=3599.0,  # just under 1 hour
        )
        sig.current_price = 30050.0  # price in range (no TP/SL triggered)

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        await monitor._evaluate_signal(sig)

        assert sig.signal_id not in removed
        assert sig.status != "EXPIRED"

    @pytest.mark.asyncio
    async def test_expiry_records_correct_pnl(self):
        """On expiry, PnL must be calculated at the current market price."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            tp2=30300.0,
            age_seconds=3700.0,
        )
        market_price = 30200.0  # price moved up, expect positive PnL
        sig.current_price = market_price

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        await monitor._evaluate_signal(sig)

        assert sig.signal_id in removed
        assert sig.status == "EXPIRED"
        expected_pnl = (market_price - 30000.0) / 30000.0 * 100.0
        assert sig.pnl_pct == pytest.approx(expected_pnl, rel=1e-4)

    @pytest.mark.asyncio
    async def test_expiry_posts_telegram_update(self):
        """An expired signal must attempt to post a Telegram update with EXPIRED text."""
        from unittest.mock import AsyncMock, patch

        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            tp2=30300.0,
            age_seconds=4000.0,
        )
        sig.current_price = 30050.0

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        with patch.object(monitor, "_post_update", new_callable=AsyncMock) as mock_post:
            await monitor._evaluate_signal(sig)
            mock_post.assert_called_once()
            # The event argument (second positional arg) must contain "EXPIRED"
            call_args = mock_post.call_args
            event_text = call_args[0][1] if call_args[0] else call_args.kwargs.get("event", "")
            assert "EXPIRED" in event_text
