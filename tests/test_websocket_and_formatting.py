"""Tests for WebSocket REST fallback and enhanced signal formatting."""

import asyncio

import pytest

from src.channels.base import Signal
from src.smc import Direction
from src.telegram_bot import TelegramBot
from src.utils import utcnow
from src.websocket_manager import WebSocketManager


class TestWebSocketFallback:
    def test_set_critical_pairs(self):
        msgs = []

        async def handler(data):
            msgs.append(data)

        ws = WebSocketManager(handler, market="spot")
        ws.set_critical_pairs(["BTCUSDT", "ETHUSDT"])
        assert ws._critical_pairs == {"BTCUSDT", "ETHUSDT"}

    def test_fallback_not_active_initially(self):
        async def handler(data):
            pass

        ws = WebSocketManager(handler, market="spot")
        assert ws._rest_fallback_active is False
        assert ws._fallback_task is None

    def test_start_rest_fallback_no_pairs(self):
        async def handler(data):
            pass

        ws = WebSocketManager(handler, market="spot")
        ws._start_rest_fallback()
        assert ws._rest_fallback_active is False  # no critical pairs

    def test_stop_rest_fallback_noop(self):
        async def handler(data):
            pass

        ws = WebSocketManager(handler, market="spot")
        ws._stop_rest_fallback()  # should not raise
        assert ws._rest_fallback_active is False


class TestFormatFreeSignal:
    def test_free_signal_has_header_and_footer(self):
        sig = Signal(
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32150,
            stop_loss=32120,
            tp1=32200,
            tp2=32300,
            tp3=32400,
            trailing_active=True,
            trailing_desc="1.5×ATR",
            confidence=87,
            ai_sentiment_label="Positive",
            ai_sentiment_summary="Whale Activity",
            risk_label="Aggressive",
            market_phase="Bullish",
            liquidity_info="High",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_free_signal(sig)
        assert "FREE SIGNAL OF THE DAY" in text
        assert "BTCUSDT" in text
        assert "Tip:" in text
        assert "Premium gets all signals!" in text

    def test_format_signal_includes_market_phase(self):
        sig = Signal(
            channel="360_SCALP",
            symbol="ETHUSDT",
            direction=Direction.SHORT,
            entry=2350,
            stop_loss=2380,
            tp1=2320,
            tp2=2300,
            confidence=80,
            market_phase="Bearish",
            liquidity_info="Low",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "Market Phase: Bearish" in text
        assert "Liquidity Pool: Low" in text

    def test_format_signal_default_market_phase(self):
        sig = Signal(
            channel="360_RANGE",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32100,
            stop_loss=32050,
            tp1=32150,
            tp2=32200,
            confidence=75,
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "Market Phase: N/A" in text
        assert "Liquidity Pool: Standard" in text


class TestSignalDataclass:
    def test_new_fields_default(self):
        sig = Signal(
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32000,
            stop_loss=31900,
            tp1=32100,
            tp2=32200,
            confidence=85,
            timestamp=utcnow(),
        )
        assert sig.market_phase == "N/A"
        assert sig.liquidity_info == "Standard"

    def test_new_fields_custom(self):
        sig = Signal(
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32000,
            stop_loss=31900,
            tp1=32100,
            tp2=32200,
            confidence=85,
            market_phase="Accumulation",
            liquidity_info="Deep",
            timestamp=utcnow(),
        )
        assert sig.market_phase == "Accumulation"
        assert sig.liquidity_info == "Deep"
