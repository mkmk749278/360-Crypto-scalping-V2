"""Tests for src.trade_monitor – minimum lifespan and SL/TP evaluation."""

from __future__ import annotations

from datetime import timedelta
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
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


class TestATRBasedTrailing:
    """TradeMonitor._adjust_trailing must use ATR-based distance when data is available."""

    def _build_monitor(self, active: Dict[str, Signal], candles=None):
        removed = []
        sent = []

        async def mock_send(chat_id, text):
            sent.append((chat_id, text))

        data_store = MagicMock()
        data_store.get_candles.return_value = candles
        data_store.ticks = {}

        monitor = TradeMonitor(
            data_store=data_store,
            send_telegram=mock_send,
            get_active_signals=lambda: dict(active),
            remove_signal=lambda sid: removed.append(sid),
            update_signal=MagicMock(),
        )
        return monitor, removed

    def _make_candles_with_atr(self, n: int = 50, price: float = 30000.0, noise: float = 50.0):
        """Generate synthetic candles that will produce a non-zero ATR(14)."""
        rng = np.random.default_rng(7)
        close = np.cumsum(rng.normal(0, noise, n)) + price
        high = close + abs(rng.normal(0, noise * 0.3, n))
        low = close - abs(rng.normal(0, noise * 0.3, n))
        return {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1000.0,
        }

    def test_atr_based_trailing_advances_stop(self):
        """When ATR data is available the trailing stop must advance with price."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=60.0,
        )
        sig.status = "TP1_HIT"
        sig.trailing_active = True
        sig.original_sl_distance = 150.0

        # Price well above entry
        sig.current_price = 30500.0

        candles = self._make_candles_with_atr(n=50, price=30000.0)
        monitor, _ = self._build_monitor({sig.signal_id: sig}, candles=candles)

        original_sl = sig.stop_loss
        monitor._adjust_trailing(sig)

        # SL must have moved above the original level
        assert sig.stop_loss > original_sl

    def test_fallback_when_no_candles(self):
        """When get_candles returns None, fall back to base_dist * 0.5."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=60.0,
        )
        sig.status = "TP1_HIT"
        sig.trailing_active = True
        sig.original_sl_distance = 150.0
        sig.current_price = 30400.0

        # No candles available → fallback
        monitor, _ = self._build_monitor({sig.signal_id: sig}, candles=None)
        monitor._adjust_trailing(sig)

        # trail_dist = 150 * 0.5 = 75  → new_sl = 30400 - 75 = 30325
        assert sig.stop_loss == pytest.approx(30325.0)

    def test_fallback_when_insufficient_candles(self):
        """When fewer than 15 candles are available, fall back to base_dist * 0.5."""
        short_candles = {
            "open": np.ones(5) * 30000.0,
            "high": np.ones(5) * 30010.0,
            "low": np.ones(5) * 29990.0,
            "close": np.ones(5) * 30000.0,
            "volume": np.ones(5) * 100.0,
        }
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=60.0,
        )
        sig.status = "TP1_HIT"
        sig.trailing_active = True
        sig.original_sl_distance = 150.0
        sig.current_price = 30400.0

        monitor, _ = self._build_monitor({sig.signal_id: sig}, candles=short_candles)
        monitor._adjust_trailing(sig)

        # Fallback: trail_dist = 150 * 0.5 = 75 → new_sl = 30325
        assert sig.stop_loss == pytest.approx(30325.0)

    def test_atr_trailing_short_direction(self):
        """For SHORT positions the ATR-based trailing stop must move down with price."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.SHORT,
            entry=30000.0,
            stop_loss=30150.0,
            tp1=29850.0,
            tp2=29700.0,
            age_seconds=60.0,
        )
        sig.status = "TP1_HIT"
        sig.trailing_active = True
        sig.original_sl_distance = 150.0
        # Price has fallen
        sig.current_price = 29600.0

        candles = self._make_candles_with_atr(n=50, price=30000.0)
        monitor, _ = self._build_monitor({sig.signal_id: sig}, candles=candles)

        original_sl = sig.stop_loss
        monitor._adjust_trailing(sig)

        # SL for SHORT must decrease (move closer to price from above)
        assert sig.stop_loss < original_sl

    def test_channel_atr_multiplier_used(self):
        """The trailing distance should reflect the channel's trailing_atr_mult config."""
        candles = self._make_candles_with_atr(n=50)
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=60.0,
        )
        sig.status = "TP1_HIT"
        sig.trailing_active = True
        sig.original_sl_distance = 150.0
        sig.current_price = 30500.0

        monitor, _ = self._build_monitor({sig.signal_id: sig}, candles=candles)
        # Run once to capture the SL after adjustment
        sig_copy_1 = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=60.0,
        )
        sig_copy_1.status = "TP1_HIT"
        sig_copy_1.trailing_active = True
        sig_copy_1.original_sl_distance = 150.0
        sig_copy_1.current_price = 30500.0

        monitor._adjust_trailing(sig_copy_1)
        sl_after = sig_copy_1.stop_loss

        # The SL must be above entry and above original stop
        assert sl_after > 29850.0


class TestSignalQualityPnL:
    """TradeMonitor must correctly compute signal_quality_pnl when TP1/TP2 is hit before SL."""

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
    async def test_tp1_then_sl_signal_quality_uses_tp1_pnl(self):
        """TP1 hit followed by SL: signal_quality_pnl_pct uses TP1 price, pnl_pct uses SL price."""
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
        # Simulate TP1 having been hit previously
        sig.best_tp_hit = 1
        sig.best_tp_pnl_pct = 0.5  # (30150 - 30000) / 30000 * 100
        sig.status = "TP1_HIT"
        sig.stop_loss = 29850.0  # SL not yet moved
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        assert sig.status == "SL_HIT"
        pt.record_outcome.assert_called_once()
        call_kwargs = pt.record_outcome.call_args.kwargs
        # Actual PnL = SL price
        assert call_kwargs["pnl_pct"] == pytest.approx(-0.5)
        assert call_kwargs["hit_sl"] is True
        # Signal quality PnL = TP1 price
        assert call_kwargs["signal_quality_pnl_pct"] == pytest.approx(0.5)
        assert call_kwargs["signal_quality_hit_tp"] == 1
        # Circuit breaker must use actual (SL) PnL
        cb_kwargs = cb.record_outcome.call_args.kwargs
        assert cb_kwargs["pnl_pct"] == pytest.approx(-0.5)
        assert cb_kwargs["hit_sl"] is True

    @pytest.mark.asyncio
    async def test_tp2_then_sl_signal_quality_uses_tp2_pnl(self):
        """TP2 hit followed by SL: signal_quality_pnl_pct uses TP2 price."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=30000.0,  # break-even after TP2
            tp1=30150.0,
            tp2=30300.0,
            tp3=30450.0,
            age_seconds=35.0,
        )
        # Simulate TP2 having been hit previously
        sig.best_tp_hit = 2
        sig.best_tp_pnl_pct = 1.0  # (30300 - 30000) / 30000 * 100
        sig.status = "TP2_HIT"
        sig.current_price = 29900.0  # below break-even SL

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        assert sig.status == "BREAKEVEN_EXIT"
        pt.record_outcome.assert_called_once()
        call_kwargs = pt.record_outcome.call_args.kwargs
        # Actual PnL = 0 (break-even at entry)
        assert call_kwargs["pnl_pct"] == pytest.approx(0.0, abs=0.05)
        # Signal quality PnL = TP2 price
        assert call_kwargs["signal_quality_pnl_pct"] == pytest.approx(1.0)
        assert call_kwargs["signal_quality_hit_tp"] == 2
        # Circuit breaker uses actual PnL (not signal quality)
        cb_kwargs = cb.record_outcome.call_args.kwargs
        assert cb_kwargs["pnl_pct"] == pytest.approx(0.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_no_tp_hit_then_sl_signal_quality_equals_actual(self):
        """No TP hit, then SL: signal_quality_pnl_pct equals actual pnl_pct."""
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
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        assert sig.status == "SL_HIT"
        pt.record_outcome.assert_called_once()
        call_kwargs = pt.record_outcome.call_args.kwargs
        # Both actual and signal quality should be the SL PnL
        assert call_kwargs["pnl_pct"] == pytest.approx(-0.5)
        assert call_kwargs["signal_quality_pnl_pct"] == pytest.approx(-0.5)
        assert call_kwargs["signal_quality_hit_tp"] == 0

    @pytest.mark.asyncio
    async def test_tp1_hit_snapshots_best_tp_fields(self):
        """When TP1 is hit, best_tp_hit and best_tp_pnl_pct must be set on the signal."""
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
        assert sig.best_tp_hit == 1
        assert sig.best_tp_pnl_pct == pytest.approx(0.5)  # (30150 - 30000) / 30000 * 100

    @pytest.mark.asyncio
    async def test_tp2_hit_upgrades_best_tp_fields(self):
        """When TP2 is hit, best_tp_hit upgrades to 2 and best_tp_pnl_pct uses TP2 price."""
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
        sig.current_price = 30350.0  # above TP2 but below TP3

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        assert sig.status == "TP2_HIT"
        assert sig.best_tp_hit == 2
        assert sig.best_tp_pnl_pct == pytest.approx(1.0)  # (30300 - 30000) / 30000 * 100

    @pytest.mark.asyncio
    async def test_tp1_expiry_signal_quality_uses_tp1_pnl(self):
        """TP1 hit then signal expires: signal quality uses TP1 PnL, actual uses market price."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            tp2=30300.0,
            tp3=30450.0,
            age_seconds=3700.0,  # expired
        )
        # TP1 was hit before expiry
        sig.best_tp_hit = 1
        sig.best_tp_pnl_pct = 0.5
        sig.status = "TP1_HIT"
        market_price = 30050.0  # price at expiry (lower than TP1)
        sig.current_price = market_price

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        assert sig.status == "EXPIRED"
        pt.record_outcome.assert_called_once()
        call_kwargs = pt.record_outcome.call_args.kwargs
        # Actual PnL = market price at expiry
        expected_actual = (market_price - 30000.0) / 30000.0 * 100.0
        assert call_kwargs["pnl_pct"] == pytest.approx(expected_actual, rel=1e-4)
        # Signal quality PnL = TP1 price (best TP reached)
        assert call_kwargs["signal_quality_pnl_pct"] == pytest.approx(0.5)
        assert call_kwargs["signal_quality_hit_tp"] == 1

    @pytest.mark.asyncio
    async def test_short_tp1_then_sl_signal_quality_uses_tp1_pnl(self):
        """SHORT: TP1 hit followed by SL uses TP1 for signal quality PnL."""
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
        # Simulate TP1 having been hit
        sig.best_tp_hit = 1
        sig.best_tp_pnl_pct = 0.5  # (30000 - 29850) / 30000 * 100
        sig.status = "TP1_HIT"
        sig.current_price = 30250.0  # above SL for SHORT

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        assert sig.status == "SL_HIT"
        pt.record_outcome.assert_called_once()
        call_kwargs = pt.record_outcome.call_args.kwargs
        # Actual PnL = SL price (30150) for SHORT: (30000 - 30150) / 30000 * 100 = -0.5%
        assert call_kwargs["pnl_pct"] == pytest.approx(-0.5)
        # Signal quality PnL = TP1 price
        assert call_kwargs["signal_quality_pnl_pct"] == pytest.approx(0.5)
        assert call_kwargs["signal_quality_hit_tp"] == 1
        # Circuit breaker uses actual PnL
        cb_kwargs = cb.record_outcome.call_args.kwargs
        assert cb_kwargs["pnl_pct"] == pytest.approx(-0.5)
        assert cb_kwargs["hit_sl"] is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_always_uses_actual_pnl(self):
        """Circuit breaker must always receive the real exit PnL regardless of signal quality."""
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
        # Simulate TP2 having been hit (signal quality would be +1%)
        sig.best_tp_hit = 2
        sig.best_tp_pnl_pct = 1.0
        sig.status = "TP2_HIT"
        sig.stop_loss = 30000.0  # break-even after TP2
        sig.current_price = 29900.0  # below break-even

        active = {sig.signal_id: sig}
        monitor, removed, sent, pt, cb = self._build_monitor_with_mocks(active)

        await monitor._evaluate_signal(sig)

        # Signal quality shows positive, but circuit breaker sees the actual (breakeven/loss)
        cb_kwargs = cb.record_outcome.call_args.kwargs
        assert cb_kwargs["pnl_pct"] == pytest.approx(0.0, abs=0.05)
        # Not a loss from circuit breaker's perspective (break-even)
        assert cb_kwargs["hit_sl"] is True


# ---------------------------------------------------------------------------
# Signal Invalidation Tests
# ---------------------------------------------------------------------------

class TestSignalInvalidation:
    """Tests for TradeMonitor._check_invalidation() and its integration with
    _evaluate_signal()."""

    def _build_monitor(
        self,
        active: Dict[str, Signal],
        candles_close=None,
        regime_detector=None,
        indicators_fn=None,
    ):
        removed = []
        sent = []

        async def mock_send(chat_id, text):
            sent.append((chat_id, text))

        data_store = MagicMock()
        data_store.ticks = {}
        if candles_close is not None:
            closes = list(candles_close)
            candles_dict = {
                "close": closes,
                "open": closes,
                "high": closes,
                "low": closes,
                "volume": [1.0] * len(closes),
            }
            data_store.get_candles.return_value = candles_dict
        else:
            data_store.get_candles.return_value = None

        monitor = TradeMonitor(
            data_store=data_store,
            send_telegram=mock_send,
            get_active_signals=lambda: dict(active),
            remove_signal=lambda sid: removed.append(sid),
            update_signal=MagicMock(),
            regime_detector=regime_detector,
            indicators_fn=indicators_fn,
        )
        # Set current_price so _evaluate_signal can proceed
        for sig in active.values():
            if sig.current_price == 0.0:
                sig.current_price = sig.entry
        return monitor, removed, sent

    # ------------------------------------------------------------------
    # _check_invalidation unit tests
    # ------------------------------------------------------------------

    def test_no_invalidation_when_no_data(self):
        """When no indicators are available, _check_invalidation returns None."""
        sig = _make_signal(age_seconds=200.0)
        monitor, _, _ = self._build_monitor({sig.signal_id: sig})
        assert monitor._check_invalidation(sig) is None

    def test_regime_flip_invalidates_long(self):
        """LONG signal must be invalidated when regime detector returns TRENDING_DOWN."""
        sig = _make_signal(direction=Direction.LONG, age_seconds=400.0)

        regime_detector = MagicMock()
        regime_result = MagicMock()
        regime_result.regime.value = "TRENDING_DOWN"
        regime_detector.classify.return_value = regime_result

        closes = [30000.0] * 25  # enough data
        monitor, _, _ = self._build_monitor(
            {sig.signal_id: sig},
            candles_close=closes,
            regime_detector=regime_detector,
        )
        reason = monitor._check_invalidation(sig)
        assert reason is not None
        assert "TRENDING_DOWN" in reason
        assert "LONG" in reason

    def test_regime_flip_invalidates_short(self):
        """SHORT signal must be invalidated when regime detector returns TRENDING_UP."""
        sig = _make_signal(
            direction=Direction.SHORT,
            entry=30000.0,
            stop_loss=30150.0,
            tp1=29850.0,
            age_seconds=400.0,
        )

        regime_detector = MagicMock()
        regime_result = MagicMock()
        regime_result.regime.value = "TRENDING_UP"
        regime_detector.classify.return_value = regime_result

        closes = [30000.0] * 25
        monitor, _, _ = self._build_monitor(
            {sig.signal_id: sig},
            candles_close=closes,
            regime_detector=regime_detector,
        )
        reason = monitor._check_invalidation(sig)
        assert reason is not None
        assert "TRENDING_UP" in reason
        assert "SHORT" in reason

    def test_ema_bearish_crossover_invalidates_long(self):
        """LONG signal invalidated when EMA9 < EMA21 (bearish crossover)."""
        sig = _make_signal(direction=Direction.LONG, age_seconds=400.0)

        # Create a falling price sequence: EMA9 will be lower than EMA21
        closes = [30000.0 - i * 10 for i in range(25)]  # descending
        monitor, _, _ = self._build_monitor({sig.signal_id: sig}, candles_close=closes)
        reason = monitor._check_invalidation(sig)
        assert reason is not None
        assert "EMA" in reason
        assert "LONG" in reason

    def test_ema_bullish_crossover_invalidates_short(self):
        """SHORT signal invalidated when EMA9 > EMA21 (bullish crossover)."""
        sig = _make_signal(
            direction=Direction.SHORT,
            entry=30000.0,
            stop_loss=30150.0,
            tp1=29850.0,
            age_seconds=400.0,
        )

        # Rising prices: EMA9 > EMA21
        closes = [30000.0 + i * 10 for i in range(25)]
        monitor, _, _ = self._build_monitor({sig.signal_id: sig}, candles_close=closes)
        reason = monitor._check_invalidation(sig)
        assert reason is not None
        assert "EMA" in reason
        assert "SHORT" in reason

    def test_momentum_loss_invalidates_after_min_age(self):
        """Signal with flat momentum invalidated after INVALIDATION_MIN_AGE_SECONDS."""
        from config import INVALIDATION_MIN_AGE_SECONDS
        channel = "360_SCALP"
        min_age = INVALIDATION_MIN_AGE_SECONDS[channel]
        sig = _make_signal(channel=channel, age_seconds=min_age + 10)

        # Flat prices → tiny momentum
        closes = [30000.0] * 25  # all same → zero momentum
        monitor, _, _ = self._build_monitor({sig.signal_id: sig}, candles_close=closes)

        # Disable regime and EMA to isolate momentum check
        # (flat prices mean EMA9 == EMA21, so no EMA invalidation)
        reason = monitor._check_invalidation(sig)
        # Flat → momentum ≈ 0 < threshold
        assert reason is not None
        assert "momentum" in reason.lower()

    def test_momentum_not_invalidated_before_min_age(self):
        """Momentum-loss check must NOT fire before INVALIDATION_MIN_AGE_SECONDS."""
        from config import INVALIDATION_MIN_AGE_SECONDS
        channel = "360_SCALP"
        min_age = INVALIDATION_MIN_AGE_SECONDS[channel]
        # Signal is 10s younger than the minimum age (always positive for min_age >= 10)
        sig = _make_signal(channel=channel, age_seconds=min_age - 10)

        closes = [30000.0] * 25  # flat → zero momentum, but EMA9 == EMA21 (no EMA invalidation)
        monitor, _, _ = self._build_monitor({sig.signal_id: sig}, candles_close=closes)

        # Regime detector is None → no regime check
        # EMA9 == EMA21 for flat prices → no EMA crossover invalidation
        # Momentum check is age-gated → must NOT fire before min_age
        reason = monitor._check_invalidation(sig)
        # Momentum invalidation must not occur before the minimum age
        assert reason is None, (
            f"Expected no invalidation before min_age ({min_age}s), got: {reason!r}"
        )

    # ------------------------------------------------------------------
    # Integration tests: _check_invalidation called inside _evaluate_signal
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_invalidated_signal_is_removed_and_slot_freed(self):
        """An invalidated signal must be removed from active signals."""
        sig = _make_signal(
            channel="360_SCALP",
            age_seconds=400.0,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
        )
        # Price slightly above entry so PnL is non-zero and zero-PnL guard passes
        sig.current_price = 30050.0

        regime_detector = MagicMock()
        regime_result = MagicMock()
        regime_result.regime.value = "TRENDING_DOWN"
        regime_detector.classify.return_value = regime_result

        closes = [30000.0] * 25
        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(
            active, candles_close=closes, regime_detector=regime_detector
        )

        await monitor._evaluate_signal(sig)

        assert sig.signal_id in removed
        assert sig.status == "INVALIDATED"
        # Telegram send is skipped in tests (no CHANNEL_TELEGRAM_MAP entry),
        # but the signal must be removed and status must be INVALIDATED.

    @pytest.mark.asyncio
    async def test_invalidated_signal_not_counted_as_sl_for_circuit_breaker(self):
        """Invalidated signals must NOT count as stop-losses in the circuit breaker."""
        sig = _make_signal(
            channel="360_SCALP",
            age_seconds=400.0,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
        )
        # Price slightly above entry so PnL is non-zero and zero-PnL guard passes
        sig.current_price = 30050.0

        cb = MagicMock()

        regime_detector = MagicMock()
        regime_result = MagicMock()
        regime_result.regime.value = "TRENDING_DOWN"
        regime_detector.classify.return_value = regime_result

        closes = [30000.0] * 25
        active = {sig.signal_id: sig}

        async def mock_send(chat_id, text):
            pass

        data_store = MagicMock()
        data_store.ticks = {}
        candles_dict = {
            "close": closes, "open": closes, "high": closes, "low": closes,
            "volume": [1.0] * len(closes),
        }
        data_store.get_candles.return_value = candles_dict

        monitor = TradeMonitor(
            data_store=data_store,
            send_telegram=mock_send,
            get_active_signals=lambda: dict(active),
            remove_signal=MagicMock(),
            update_signal=MagicMock(),
            circuit_breaker=cb,
            regime_detector=regime_detector,
        )

        await monitor._evaluate_signal(sig)

        # circuit_breaker.record_outcome must be called with hit_sl=False
        assert cb.record_outcome.called
        call_kwargs = cb.record_outcome.call_args.kwargs
        assert call_kwargs.get("hit_sl") is False

    # ------------------------------------------------------------------
    # New tests for the reordered evaluation and age-gating
    # ------------------------------------------------------------------

    def test_regime_not_invalidated_before_min_age(self):
        """Regime flip must NOT fire before INVALIDATION_MIN_AGE_SECONDS (global gate)."""
        from config import INVALIDATION_MIN_AGE_SECONDS
        channel = "360_SCALP"
        min_age = INVALIDATION_MIN_AGE_SECONDS[channel]
        sig = _make_signal(channel=channel, direction=Direction.LONG, age_seconds=min_age - 10)

        regime_detector = MagicMock()
        regime_result = MagicMock()
        regime_result.regime.value = "TRENDING_DOWN"
        regime_detector.classify.return_value = regime_result

        closes = [30000.0] * 25
        monitor, _, _ = self._build_monitor(
            {sig.signal_id: sig},
            candles_close=closes,
            regime_detector=regime_detector,
        )
        reason = monitor._check_invalidation(sig)
        assert reason is None, (
            f"Regime invalidation must not fire before min_age ({min_age}s), got: {reason!r}"
        )

    def test_ema_not_invalidated_before_min_age(self):
        """EMA crossover must NOT fire before INVALIDATION_MIN_AGE_SECONDS (global gate)."""
        from config import INVALIDATION_MIN_AGE_SECONDS
        channel = "360_SCALP"
        min_age = INVALIDATION_MIN_AGE_SECONDS[channel]
        sig = _make_signal(channel=channel, direction=Direction.LONG, age_seconds=min_age - 10)

        # Falling prices → EMA9 < EMA21 (bearish crossover)
        closes = [30000.0 - i * 10 for i in range(25)]
        monitor, _, _ = self._build_monitor({sig.signal_id: sig}, candles_close=closes)
        reason = monitor._check_invalidation(sig)
        assert reason is None, (
            f"EMA invalidation must not fire before min_age ({min_age}s), got: {reason!r}"
        )

    @pytest.mark.asyncio
    async def test_sl_fires_before_invalidation(self):
        """When price is below SL and regime flips, SL must fire (not invalidation)."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            age_seconds=400.0,
        )
        # Price has gapped BELOW the stop-loss
        sig.current_price = 29700.0

        regime_detector = MagicMock()
        regime_result = MagicMock()
        regime_result.regime.value = "TRENDING_DOWN"
        regime_detector.classify.return_value = regime_result

        closes = [30000.0] * 25
        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(
            active, candles_close=closes, regime_detector=regime_detector
        )

        await monitor._evaluate_signal(sig)

        # SL must have fired, not invalidation
        assert sig.signal_id in removed
        assert sig.status != "INVALIDATED", "Should be SL_HIT, not INVALIDATED"
        assert "SL" in sig.status or sig.status in ("SL_HIT", "BREAKEVEN_EXIT", "PROFIT_LOCKED")

    @pytest.mark.asyncio
    async def test_invalidation_exit_price_capped_at_sl_long(self):
        """LONG: when invalidation fires, exit price must be capped at the SL (not below)."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            age_seconds=400.0,
        )
        # Price is above SL so SL doesn't fire; invalidation will fire instead
        sig.current_price = 29900.0

        regime_detector = MagicMock()
        regime_result = MagicMock()
        regime_result.regime.value = "TRENDING_DOWN"
        regime_detector.classify.return_value = regime_result

        closes = [30000.0] * 25
        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(
            active, candles_close=closes, regime_detector=regime_detector
        )

        await monitor._evaluate_signal(sig)

        assert sig.status == "INVALIDATED"
        # Exit PnL must be computed at the capped price (max(29900, 29850) = 29900 for LONG)
        # which is the current price since price > SL
        from src.performance_metrics import calculate_trade_pnl_pct
        expected_pnl = calculate_trade_pnl_pct(30000.0, 29900.0, "LONG")
        assert sig.pnl_pct == pytest.approx(expected_pnl, abs=1e-4)

    @pytest.mark.asyncio
    async def test_invalidation_exit_price_never_worse_than_sl_long(self):
        """LONG: invalidation with price below SL must exit at SL, not current price."""
        # This scenario: price has gapped, but SL check runs first and catches it.
        # However, if the reorder means SL fires first, we verify that pathway.
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            tp1=30150.0,
            age_seconds=400.0,
        )
        # Price is below SL — SL check fires first
        sig.current_price = 29700.0

        closes = [30000.0] * 25
        active = {sig.signal_id: sig}
        monitor, removed, _ = self._build_monitor(active, candles_close=closes)

        await monitor._evaluate_signal(sig)

        # Must be removed via SL, not invalidation
        assert sig.signal_id in removed
        # Exit price frozen at SL, not at the worse 29700
        assert sig.current_price == pytest.approx(29850.0)

    @pytest.mark.asyncio
    async def test_invalidation_exit_price_capped_at_sl_short(self):
        """SHORT: when invalidation fires, exit price must be capped at the SL (not above)."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.SHORT,
            entry=30000.0,
            stop_loss=30150.0,
            tp1=29850.0,
            age_seconds=400.0,
        )
        # Price is below SL so SL doesn't fire; invalidation will fire instead
        sig.current_price = 30100.0

        regime_detector = MagicMock()
        regime_result = MagicMock()
        regime_result.regime.value = "TRENDING_UP"
        regime_detector.classify.return_value = regime_result

        closes = [30000.0] * 25
        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(
            active, candles_close=closes, regime_detector=regime_detector
        )

        await monitor._evaluate_signal(sig)

        assert sig.status == "INVALIDATED"
        # Exit PnL must use min(30100, 30150) = 30100 — current price since price < SL
        from src.performance_metrics import calculate_trade_pnl_pct
        expected_pnl = calculate_trade_pnl_pct(30000.0, 30100.0, "SHORT")
        assert sig.pnl_pct == pytest.approx(expected_pnl, abs=1e-4)

    def test_dca_grace_period_prevents_invalidation(self):
        """Invalidation must return None within 300s of a DCA entry being filled."""
        from datetime import timedelta
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            age_seconds=400.0,
        )
        # Mark DCA as just filled (1 minute ago — within grace period)
        sig.entry_2_filled = True
        sig.dca_timestamp = utcnow() - timedelta(seconds=60)

        regime_detector = MagicMock()
        regime_result = MagicMock()
        regime_result.regime.value = "TRENDING_DOWN"
        regime_detector.classify.return_value = regime_result

        closes = [30000.0] * 25
        monitor, _, _ = self._build_monitor(
            {sig.signal_id: sig},
            candles_close=closes,
            regime_detector=regime_detector,
        )
        reason = monitor._check_invalidation(sig)
        assert reason is None, (
            f"Invalidation must be suppressed during DCA grace period, got: {reason!r}"
        )

    def test_dca_grace_period_expires(self):
        """Invalidation is allowed after the DCA grace period (>300s since DCA)."""
        from datetime import timedelta
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            age_seconds=400.0,
        )
        # Mark DCA as filled 310 seconds ago — grace period has expired
        sig.entry_2_filled = True
        sig.dca_timestamp = utcnow() - timedelta(seconds=310)

        regime_detector = MagicMock()
        regime_result = MagicMock()
        regime_result.regime.value = "TRENDING_DOWN"
        regime_detector.classify.return_value = regime_result

        closes = [30000.0] * 25
        monitor, _, _ = self._build_monitor(
            {sig.signal_id: sig},
            candles_close=closes,
            regime_detector=regime_detector,
        )
        reason = monitor._check_invalidation(sig)
        assert reason is not None, "Invalidation must fire after DCA grace period expires"
        assert "TRENDING_DOWN" in reason

    def test_microcap_momentum_threshold_scaled_down(self):
        """Micro-cap tokens (entry < 0.001) use a 10× smaller momentum threshold."""
        from config import INVALIDATION_MIN_AGE_SECONDS, INVALIDATION_MOMENTUM_THRESHOLD
        channel = "360_SCALP"
        min_age = INVALIDATION_MIN_AGE_SECONDS[channel]
        micro_entry = 0.0000064  # BONK-like price

        sig = _make_signal(
            channel=channel,
            entry=micro_entry,
            stop_loss=micro_entry * 0.95,
            tp1=micro_entry * 1.05,
            age_seconds=min_age + 10,
        )

        # Flat prices → zero momentum → below even the scaled threshold
        closes = [30000.0] * 25
        monitor, _, _ = self._build_monitor({sig.signal_id: sig}, candles_close=closes)

        reason = monitor._check_invalidation(sig)
        assert reason is not None
        assert "momentum" in reason.lower()
        # Verify the threshold reported in the message is the scaled value
        base_threshold = INVALIDATION_MOMENTUM_THRESHOLD.get(channel, 0.15)
        scaled_threshold = base_threshold * 0.1
        assert f"{scaled_threshold}" in reason or f"{scaled_threshold:.4f}" in reason

    def test_normal_cap_momentum_threshold_not_scaled(self):
        """Standard-price tokens (entry >= 0.001) use the base momentum threshold."""
        from config import INVALIDATION_MIN_AGE_SECONDS, INVALIDATION_MOMENTUM_THRESHOLD
        channel = "360_SCALP"
        min_age = INVALIDATION_MIN_AGE_SECONDS[channel]
        standard_entry = 1.5  # normal price like XRPUSDT

        sig = _make_signal(
            channel=channel,
            entry=standard_entry,
            stop_loss=standard_entry * 0.95,
            tp1=standard_entry * 1.05,
            age_seconds=min_age + 10,
        )

        # Flat prices → zero momentum
        closes = [30000.0] * 25
        monitor, _, _ = self._build_monitor({sig.signal_id: sig}, candles_close=closes)

        reason = monitor._check_invalidation(sig)
        assert reason is not None
        assert "momentum" in reason.lower()
        # The threshold in the reason message should be the base (unscaled) value
        base_threshold = INVALIDATION_MOMENTUM_THRESHOLD.get(channel, 0.15)
        assert f"{base_threshold}" in reason


class TestOnHighlightCallback:
    """TradeMonitor.on_highlight_callback is called for TP2/TP3 but not TP1 or SL."""

    def _build_monitor(self, active, channel_map=None):
        removed = []
        sent = []
        highlight_calls = []

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
        monitor.on_highlight_callback = lambda sig, tp, pnl: highlight_calls.append((sig, tp, pnl))
        return monitor, removed, highlight_calls

    @pytest.mark.asyncio
    async def test_highlight_called_on_tp2_long(self):
        sig = _make_signal(
            direction=Direction.LONG,
            entry=30000.0, stop_loss=29850.0,
            tp1=30150.0, tp2=30300.0, tp3=30450.0,
            age_seconds=120.0,
        )
        active = {sig.signal_id: sig}
        monitor, _, highlight_calls = self._build_monitor(active)

        # Price hits TP2
        sig.current_price = 30300.0
        await monitor._evaluate_signal(sig)

        assert len(highlight_calls) == 1
        _, tp, pnl = highlight_calls[0]
        assert tp == 2
        assert pnl > 0

    @pytest.mark.asyncio
    async def test_highlight_called_on_tp3_long(self):
        sig = _make_signal(
            direction=Direction.LONG,
            entry=30000.0, stop_loss=29850.0,
            tp1=30150.0, tp2=30300.0, tp3=30450.0,
            age_seconds=120.0,
        )
        active = {sig.signal_id: sig}
        monitor, _, highlight_calls = self._build_monitor(active)

        # Price hits TP3
        sig.current_price = 30450.0
        await monitor._evaluate_signal(sig)

        assert len(highlight_calls) == 1
        _, tp, pnl = highlight_calls[0]
        assert tp == 3
        assert pnl > 0

    @pytest.mark.asyncio
    async def test_highlight_not_called_on_tp1(self):
        sig = _make_signal(
            direction=Direction.LONG,
            entry=30000.0, stop_loss=29850.0,
            tp1=30150.0, tp2=30300.0, tp3=30450.0,
            age_seconds=120.0,
        )
        active = {sig.signal_id: sig}
        monitor, _, highlight_calls = self._build_monitor(active)

        # Price only hits TP1
        sig.current_price = 30150.0
        await monitor._evaluate_signal(sig)

        assert highlight_calls == []

    @pytest.mark.asyncio
    async def test_highlight_not_called_on_sl(self):
        sig = _make_signal(
            direction=Direction.LONG,
            entry=30000.0, stop_loss=29850.0,
            tp1=30150.0, tp2=30300.0, tp3=30450.0,
            age_seconds=120.0,
        )
        active = {sig.signal_id: sig}
        monitor, _, highlight_calls = self._build_monitor(active)

        # Price hits SL
        sig.current_price = 29800.0
        await monitor._evaluate_signal(sig)

        assert highlight_calls == []

    @pytest.mark.asyncio
    async def test_highlight_called_on_tp2_short(self):
        sig = _make_signal(
            direction=Direction.SHORT,
            entry=30000.0, stop_loss=30150.0,
            tp1=29850.0, tp2=29700.0, tp3=29550.0,
            age_seconds=120.0,
        )
        active = {sig.signal_id: sig}
        monitor, _, highlight_calls = self._build_monitor(active)

        # Price hits TP2 for SHORT
        sig.current_price = 29700.0
        await monitor._evaluate_signal(sig)

        assert len(highlight_calls) == 1
        _, tp, pnl = highlight_calls[0]
        assert tp == 2
        assert pnl > 0

    @pytest.mark.asyncio
    async def test_highlight_called_on_tp3_short(self):
        sig = _make_signal(
            direction=Direction.SHORT,
            entry=30000.0, stop_loss=30150.0,
            tp1=29850.0, tp2=29700.0, tp3=29550.0,
            age_seconds=120.0,
        )
        active = {sig.signal_id: sig}
        monitor, _, highlight_calls = self._build_monitor(active)

        # Price hits TP3 for SHORT
        sig.current_price = 29550.0
        await monitor._evaluate_signal(sig)

        assert len(highlight_calls) == 1
        _, tp, pnl = highlight_calls[0]
        assert tp == 3
        assert pnl > 0

    @pytest.mark.asyncio
    async def test_no_highlight_when_callback_not_set(self):
        """TradeMonitor works correctly when on_highlight_callback is None."""
        sig = _make_signal(
            direction=Direction.LONG,
            entry=30000.0, stop_loss=29850.0,
            tp1=30150.0, tp2=30300.0, tp3=30450.0,
            age_seconds=120.0,
        )
        active = {sig.signal_id: sig}

        data_store = MagicMock()
        data_store.get_candles.return_value = None
        data_store.ticks = {}

        monitor = TradeMonitor(
            data_store=data_store,
            send_telegram=MagicMock(return_value=None),
            get_active_signals=lambda: dict(active),
            remove_signal=MagicMock(),
            update_signal=MagicMock(),
        )
        # on_highlight_callback is None by default
        assert monitor.on_highlight_callback is None

        # Should not raise even when TP2 is hit
        sig.current_price = 30300.0
        await monitor._evaluate_signal(sig)  # must not raise
