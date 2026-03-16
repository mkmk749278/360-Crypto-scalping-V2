"""Tests for WebSocket REST fallback and enhanced signal formatting."""

import asyncio
import time

import pytest


from src.channels.base import Signal
from src.smc import Direction
from src.telegram_bot import TelegramBot
from src.utils import utcnow
from src.websocket_manager import WSConnection, WebSocketManager


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


class TestEscapeMdFunction:
    """Verify the _escape_md helper escapes all Markdown V1 special characters."""

    def test_escape_asterisk(self):
        assert TelegramBot._escape_md("*bold*") == "\\*bold\\*"

    def test_escape_underscore(self):
        assert TelegramBot._escape_md("_italic_") == "\\_italic\\_"

    def test_escape_backtick(self):
        assert TelegramBot._escape_md("`code`") == "\\`code\\`"

    def test_escape_bracket(self):
        assert TelegramBot._escape_md("[text]") == "\\[text]"

    def test_escape_backslash(self):
        assert TelegramBot._escape_md("a\\b") == "a\\\\b"

    def test_escape_all_special_chars(self):
        raw = "*_`[\\"
        escaped = TelegramBot._escape_md(raw)
        assert escaped == "\\*\\_\\`\\[\\\\"

    def test_plain_text_unchanged(self):
        text = "Sweep SHORT at 0.3572 | FVG 0.3543-0.3538"
        assert TelegramBot._escape_md(text) == text

    def test_empty_string(self):
        assert TelegramBot._escape_md("") == ""


class TestWebSocketLastPongOnText:
    """Verify that last_pong is updated when TEXT messages arrive."""

    def test_last_pong_updated_on_text_message(self):
        """is_healthy should remain True after TEXT messages (not just PONG frames)."""

        received = []

        async def handler(data):
            received.append(data)

        ws = WebSocketManager(handler, market="spot")

        # Simulate a connection that received a TEXT message recently
        conn = WSConnection()
        conn.last_pong = time.monotonic()  # fresh timestamp

        # Immediately after connect the connection should be healthy
        ws._connections = [conn]

        # Monkey-patch ws so is_healthy thinks the socket is open
        mock_ws = type("FakeWS", (), {"closed": False})()
        conn.ws = mock_ws

        assert ws.is_healthy is True

        # Simulate staleness beyond the 3× heartbeat window
        conn.last_pong = time.monotonic() - 200  # 200 s ago → stale
        assert ws.is_healthy is False

        # Simulate a TEXT message arriving and updating last_pong
        conn.last_pong = time.monotonic()
        assert ws.is_healthy is True

    def test_closed_connections_are_not_reported_healthy(self):
        async def handler(data):
            return None

        ws = WebSocketManager(handler, market="spot")
        closed_ws = type("FakeWS", (), {"closed": True})()
        ws._connections = [WSConnection(ws=closed_ws, last_pong=time.monotonic())]

        assert ws.is_healthy is False


class TestWebSocketLifecycle:
    @pytest.mark.asyncio
    async def test_stop_awaits_cancelled_tasks(self):
        cancelled = []

        async def handler(data):
            return None

        async def sleeper(name: str):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.append(name)
                raise

        ws = WebSocketManager(handler, market="spot")
        conn_task = asyncio.create_task(sleeper("conn"))
        fallback_task = asyncio.create_task(sleeper("fallback"))
        watchdog_task = asyncio.create_task(sleeper("watchdog"))
        fake_ws = type(
            "FakeWS",
            (),
            {"closed": False, "close": lambda self: asyncio.sleep(0)},
        )()
        ws._connections = [WSConnection(ws=fake_ws, streams=["btcusdt@kline_1m"], task=conn_task)]
        ws._fallback_task = fallback_task
        ws._watchdog_task = watchdog_task
        ws._session = type(
            "FakeSession",
            (),
            {"closed": False, "close": lambda self: asyncio.sleep(0)},
        )()
        await asyncio.sleep(0)

        await ws.stop()

        assert {"conn", "fallback", "watchdog"} <= set(cancelled)
        assert conn_task.done() and fallback_task.done() and watchdog_task.done()

    def test_fallback_stays_active_until_all_degraded_connections_recover(self):
        async def handler(data):
            return None

        ws = WebSocketManager(handler, market="spot")
        ws.set_critical_pairs(["BTCUSDT", "ETHUSDT"])
        conn_a = WSConnection(streams=["btcusdt@kline_1m"])
        conn_b = WSConnection(streams=["ethusdt@kline_1m"])
        ws._connections = [conn_a, conn_b]
        ws._start_rest_fallback = lambda: setattr(ws, "_rest_fallback_active", True)
        ws._stop_rest_fallback = lambda: setattr(ws, "_rest_fallback_active", False)

        ws._set_connection_degraded(conn_a, True)
        ws._set_connection_degraded(conn_b, True)
        assert ws._rest_fallback_active is True

        ws._set_connection_degraded(conn_a, False)
        assert ws._rest_fallback_active is True

        ws._set_connection_degraded(conn_b, False)
        assert ws._rest_fallback_active is False
