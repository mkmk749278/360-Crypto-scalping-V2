"""Tests for src.telegram_bot – signal formatting."""

import pytest

from src.channels.base import Signal
from src.smc import Direction
from src.telegram_bot import TelegramBot
from src.utils import utcnow


class TestFormatSignal:
    def test_scalp_long_format(self):
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
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "⚡" in text
        assert "360_SCALP" in text
        assert "BTCUSDT" in text
        assert "LONG" in text
        assert "32,150" in text
        assert "87%" in text
        assert "Whale Activity" in text
        assert "Aggressive" in text
        assert "Trailing Active" in text

    def test_swing_short_format(self):
        sig = Signal(
            channel="360_SWING",
            symbol="ETHUSDT",
            direction=Direction.SHORT,
            entry=2350,
            stop_loss=2380,
            tp1=2320,
            tp2=2300,
            tp3=2270,
            trailing_active=True,
            trailing_desc="2.5×ATR",
            confidence=92,
            ai_sentiment_label="Neutral",
            ai_sentiment_summary="Moderate Volume Spike",
            risk_label="Medium",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "🏛️" in text
        assert "SHORT" in text
        assert "⬇️" in text
        assert "92%" in text

    def test_tape_format_with_ai_adaptive(self):
        sig = Signal(
            channel="360_THE_TAPE",
            symbol="ETHUSDT",
            direction=Direction.LONG,
            entry=2355,
            stop_loss=2340,
            tp1=2370,
            tp2=2390,
            tp3=None,
            trailing_active=True,
            trailing_desc="AI Adaptive",
            confidence=95,
            ai_sentiment_label="Bullish",
            ai_sentiment_summary="Whale Trade Confirmed",
            risk_label="Medium-High",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "🐋" in text
        assert "Dynamic/trailing" in text
        assert "AI Adaptive" in text
        assert "95%" in text

    def test_range_format(self):
        sig = Signal(
            channel="360_RANGE",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32100,
            stop_loss=32050,
            tp1=32150,
            tp2=32200,
            tp3=None,
            trailing_active=True,
            trailing_desc="1×ATR",
            confidence=80,
            ai_sentiment_label="Positive",
            ai_sentiment_summary="",
            risk_label="Conservative",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "⚖️" in text
        assert "Conservative" in text
        assert "80%" in text
