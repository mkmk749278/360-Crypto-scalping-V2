"""Tests for src.trade_monitor – minimum lifespan and SL/TP evaluation."""

from __future__ import annotations

import asyncio
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
        """A SCALP signal older than 10s whose price is below SL SHOULD be removed."""
        sig = _make_signal(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=15.0,  # past the 10s SCALP minimum
        )
        sig.current_price = 29800.0  # below SL

        active = {sig.signal_id: sig}
        monitor, removed, sent = self._build_monitor(active)

        await monitor._evaluate_signal(sig)

        assert sig.signal_id in removed
        assert sig.status == "SL_HIT"

    @pytest.mark.asyncio
    async def test_swing_min_lifespan_is_longer(self):
        """A SWING signal at age=15s (< 30s min) should NOT trigger SL."""
        sig = _make_signal(
            channel="360_SWING",
            direction=Direction.LONG,
            entry=30000.0,
            stop_loss=29850.0,
            age_seconds=15.0,  # below the 30s SWING minimum
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
