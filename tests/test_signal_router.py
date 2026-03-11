"""Tests for src.signal_router – queue-based signal routing."""

import asyncio

import pytest
import pytest_asyncio

from src.channels.base import Signal
from src.signal_router import SignalRouter
from src.smc import Direction
from src.utils import utcnow


@pytest.fixture
def sent_messages():
    """Collects (chat_id, text) tuples sent by the router."""
    return []


@pytest.fixture
def queue():
    return asyncio.Queue()


@pytest.fixture
def router(queue, sent_messages):
    async def mock_send(chat_id: str, text: str):
        sent_messages.append((chat_id, text))

    def mock_format(sig: Signal) -> str:
        return f"Signal: {sig.channel} {sig.symbol} {sig.direction.value}"

    return SignalRouter(queue=queue, send_telegram=mock_send, format_signal=mock_format)


def _make_signal(channel="360_SCALP", symbol="BTCUSDT", direction=Direction.LONG, confidence=85):
    return Signal(
        channel=channel,
        symbol=symbol,
        direction=direction,
        entry=32000,
        stop_loss=31900,
        tp1=32100,
        tp2=32200,
        confidence=confidence,
        signal_id=f"TEST-{symbol}-001",
        timestamp=utcnow(),
    )


class TestSignalRouter:
    @pytest.mark.asyncio
    async def test_signal_processed_and_sent(self, queue, router, sent_messages):
        sig = _make_signal(confidence=90)
        await queue.put(sig)
        # Run router briefly
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert sig.signal_id in router.active_signals

    @pytest.mark.asyncio
    async def test_low_confidence_filtered(self, queue, router, sent_messages):
        sig = _make_signal(confidence=30)  # below min 70
        await queue.put(sig)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.2)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert sig.signal_id not in router.active_signals

    @pytest.mark.asyncio
    async def test_correlation_lock(self, queue, router, sent_messages):
        sig1 = _make_signal(symbol="BTCUSDT", direction=Direction.LONG, confidence=90)
        sig1.signal_id = "TEST-BTC-001"
        sig2 = _make_signal(symbol="BTCUSDT", direction=Direction.SHORT, confidence=90)
        sig2.signal_id = "TEST-BTC-002"

        await queue.put(sig1)
        await queue.put(sig2)
        task = asyncio.create_task(router.start())
        await asyncio.sleep(0.3)
        await router.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Only the first should be active (second blocked by correlation lock)
        assert "TEST-BTC-001" in router.active_signals
        assert "TEST-BTC-002" not in router.active_signals

    @pytest.mark.asyncio
    async def test_remove_signal(self, router):
        sig = _make_signal()
        router._active_signals[sig.signal_id] = sig
        router._position_lock[sig.symbol] = sig.direction

        router.remove_signal(sig.signal_id)
        assert sig.signal_id not in router.active_signals
        assert sig.symbol not in router._position_lock
