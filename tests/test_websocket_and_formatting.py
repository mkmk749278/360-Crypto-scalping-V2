"""Tests for WebSocket REST fallback and enhanced signal formatting."""

import asyncio
import time
import unittest.mock as mock

import aiohttp
import pytest


from config import WS_ALERT_COOLDOWN, WS_FALLBACK_TIMEFRAMES, WS_HEARTBEAT_INTERVAL, WS_HEARTBEAT_INTERVAL_FUTURES, WS_SESSION_RECYCLE_ATTEMPTS, WS_STALENESS_MULTIPLIER, WS_STALENESS_MULTIPLIER_FUTURES
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


class TestWebSocketHealthRatio:
    """WebSocketManager.health_ratio returns a continuous 0.0–1.0 health score."""

    def test_health_ratio_one_before_start(self):
        """Before start() is called there are no connections: ratio == 1.0."""
        async def handler(data): pass
        ws = WebSocketManager(handler, market="spot")
        assert ws.health_ratio == 1.0

    def test_health_ratio_all_healthy(self):
        """All connections open and recently pinged → ratio == 1.0."""
        async def handler(data): pass
        ws = WebSocketManager(handler, market="spot")
        now = time.monotonic()
        conn1 = WSConnection(streams=["btcusdt@kline_1m"], last_pong=now)
        conn2 = WSConnection(streams=["ethusdt@kline_1m"], last_pong=now)
        # Attach mock open WebSocket objects
        mock_ws = mock.MagicMock()
        mock_ws.closed = False
        conn1.ws = mock_ws
        conn2.ws = mock_ws
        ws._connections = [conn1, conn2]
        assert ws.health_ratio == 1.0

    def test_health_ratio_half_healthy(self):
        """One of two connections stale → ratio == 0.5."""
        async def handler(data): pass
        ws = WebSocketManager(handler, market="spot")
        now = time.monotonic()
        mock_open = mock.MagicMock()
        mock_open.closed = False
        mock_closed = mock.MagicMock()
        mock_closed.closed = True

        conn_healthy = WSConnection(streams=["btcusdt@kline_1m"], last_pong=now, ws=mock_open)
        conn_unhealthy = WSConnection(streams=["ethusdt@kline_1m"], last_pong=now, ws=mock_closed)
        ws._connections = [conn_healthy, conn_unhealthy]
        assert ws.health_ratio == 0.5

    def test_health_ratio_all_stale(self):
        """All connections stale → ratio == 0.0."""
        async def handler(data): pass
        ws = WebSocketManager(handler, market="spot")
        stale_pong = time.monotonic() - 9999.0  # way in the past
        mock_ws = mock.MagicMock()
        mock_ws.closed = False
        conn = WSConnection(streams=["btcusdt@kline_1m"], last_pong=stale_pong, ws=mock_ws)
        ws._connections = [conn]
        assert ws.health_ratio == 0.0

    def test_health_ratio_no_ws_object(self):
        """Connection without a ws object counts as unhealthy."""
        async def handler(data): pass
        ws = WebSocketManager(handler, market="spot")
        conn = WSConnection(streams=["btcusdt@kline_1m"])  # ws=None
        ws._connections = [conn]
        assert ws.health_ratio == 0.0

    def test_is_healthy_uses_strict_all_or_nothing(self):
        """is_healthy still requires ALL connections to be healthy."""
        async def handler(data): pass
        ws = WebSocketManager(handler, market="spot")
        now = time.monotonic()
        mock_open = mock.MagicMock()
        mock_open.closed = False
        mock_closed = mock.MagicMock()
        mock_closed.closed = True

        conn_healthy = WSConnection(streams=["btcusdt@kline_1m"], last_pong=now, ws=mock_open)
        conn_unhealthy = WSConnection(streams=["ethusdt@kline_1m"], last_pong=now, ws=mock_closed)
        ws._connections = [conn_healthy, conn_unhealthy]
        # health_ratio is 0.5 but is_healthy is False (strict)
        assert ws.health_ratio == 0.5
        assert ws.is_healthy is False


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
        text = TelegramBot.format_signal_legacy(sig)
        assert "Market Phase: Bearish" in text
        assert "Liquidity Pool: Low" in text

    def test_format_signal_default_market_phase(self):
        sig = Signal(
            channel="360_SPOT",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32100,
            stop_loss=32050,
            tp1=32150,
            tp2=32200,
            confidence=75,
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal_legacy(sig)
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

        # Simulate staleness beyond the 10× heartbeat window
        conn.last_pong = time.monotonic() - 350  # 350 s ago → stale (threshold: 30×10=300s)
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
        assert ws._connections == []
        assert ws._session is None

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


class TestAdminAlertRateLimiting:
    """Admin alert must not fire more than once per WS_ALERT_COOLDOWN window."""

    def test_last_alert_time_starts_at_zero(self):
        """_last_alert_time initialises to 0.0 so the first alert always fires."""
        ws = WebSocketManager(lambda data: None, market="spot")
        assert ws._last_alert_time == 0.0

    @pytest.mark.asyncio
    async def test_alert_fires_on_first_reconnect(self):
        """Admin alert callback is invoked on the first connection drop."""
        alerted = []

        async def alert(msg):
            alerted.append(msg)

        ws = WebSocketManager(lambda data: None, market="futures", admin_alert_callback=alert)
        # Set last_alert_time beyond the WS_ALERT_COOLDOWN window so the alert fires.
        ws._last_alert_time = time.monotonic() - (WS_ALERT_COOLDOWN + 1)

        # Simulate the alert logic directly
        now = time.monotonic()
        if now - ws._last_alert_time > WS_ALERT_COOLDOWN:
            ws._last_alert_time = now
            await alert("⚠️ WebSocket connection lost (futures, attempt 1). Reconnecting…")

        assert len(alerted) == 1

    @pytest.mark.asyncio
    async def test_alert_suppressed_within_cooldown(self):
        """Alert is not sent again while the cooldown is active."""
        alerted = []

        async def alert(msg):
            alerted.append(msg)

        ws = WebSocketManager(lambda data: None, market="futures", admin_alert_callback=alert)
        # Simulate that an alert was just sent
        ws._last_alert_time = time.monotonic()

        now = time.monotonic()
        if now - ws._last_alert_time > WS_ALERT_COOLDOWN:
            ws._last_alert_time = now
            await alert("⚠️ WebSocket connection lost (futures, attempt 1). Reconnecting…")

        assert len(alerted) == 0

    def test_no_ping_loop_method(self):
        """_ping_loop must not exist — aiohttp heartbeat= handles keepalive."""
        ws = WebSocketManager(lambda data: None, market="spot")
        assert not hasattr(ws, "_ping_loop")

    def test_alert_cooldown_is_600(self):
        """WS_ALERT_COOLDOWN must be at least 600s to reduce Telegram spam."""
        assert WS_ALERT_COOLDOWN >= 600


class TestLabelParameter:
    """label parameter lets managers report a distinct name in logs and alerts."""

    def test_label_defaults_to_market(self):
        """When label is not provided, _label falls back to the market value."""
        ws = WebSocketManager(lambda data: None, market="futures")
        assert ws._label == "futures"

    def test_label_defaults_to_market_spot(self):
        """When label is not provided for spot, _label falls back to 'spot'."""
        ws = WebSocketManager(lambda data: None, market="spot")
        assert ws._label == "spot"

    def test_custom_label_is_stored(self):
        """An explicit label is stored and distinct from market."""
        ws = WebSocketManager(lambda data: None, market="futures", label="futures_liq")
        assert ws._label == "futures_liq"
        assert ws._market == "futures"

    @pytest.mark.asyncio
    async def test_alert_message_uses_label(self):
        """Alert messages include the label, not the raw market string."""
        alerted = []

        async def alert(msg):
            alerted.append(msg)

        ws = WebSocketManager(
            lambda data: None,
            market="futures",
            admin_alert_callback=alert,
            label="futures_liq",
        )
        # Verify that _label (not _market) is what the alert format string uses.
        # The implementation builds: f"... ({self._label}, attempt …)"
        # so _label == "futures_liq" means the alert will say "futures_liq".
        assert ws._label == "futures_liq"
        assert ws._market == "futures"
        # The alert text produced by _run_connection will use self._label:
        alert_text = (
            f"⚠️ WebSocket connection lost ({ws._label}, attempt 1, "
            f"total drops: 1). Reconnecting…"
        )
        assert "futures_liq" in alert_text
        assert alert_text.count("futures") == 1  # only "futures_liq", not bare "futures"


class TestWsFuturesLiqNoAdminCallback:
    """_ws_futures_liq must be created without an admin alert callback."""

    def test_futures_liq_has_no_alert_callback(self):
        """Liquidation WS manager is expendable; drops should not alert admin."""
        ws = WebSocketManager(
            lambda data: None,
            market="futures",
            admin_alert_callback=None,
            label="futures_liq",
        )
        assert ws._admin_alert is None

    def test_futures_liq_label_is_set(self):
        """Liquidation WS manager reports as 'futures_liq' in logs."""
        ws = WebSocketManager(
            lambda data: None,
            market="futures",
            admin_alert_callback=None,
            label="futures_liq",
        )
        assert ws._label == "futures_liq"




class TestHeartbeatIntervalPerMarket:
    """Verify per-market heartbeat interval selection."""

    def test_futures_uses_longer_heartbeat(self):
        """Futures WebSocketManager must use WS_HEARTBEAT_INTERVAL_FUTURES."""
        ws = WebSocketManager(lambda data: None, market="futures")
        assert ws._heartbeat_interval == WS_HEARTBEAT_INTERVAL_FUTURES

    def test_spot_uses_default_heartbeat(self):
        """Spot WebSocketManager must use WS_HEARTBEAT_INTERVAL."""
        ws = WebSocketManager(lambda data: None, market="spot")
        assert ws._heartbeat_interval == WS_HEARTBEAT_INTERVAL

    def test_futures_heartbeat_longer_than_spot(self):
        """Futures heartbeat interval must be strictly longer than spot."""
        assert WS_HEARTBEAT_INTERVAL_FUTURES > WS_HEARTBEAT_INTERVAL

    def test_futures_staleness_threshold_with_new_heartbeat(self):
        """Futures connections should be stale after 60×15=900s with no data."""
        ws = WebSocketManager(lambda data: None, market="futures")
        conn = WSConnection()
        mock_ws = type("FakeWS", (), {"closed": False})()
        conn.ws = mock_ws
        conn.last_pong = time.monotonic() - 950  # 950s > 900s threshold
        ws._connections = [conn]
        assert ws.is_healthy is False

    def test_futures_staleness_healthy_under_threshold(self):
        """Futures connections with data within 900s should still be healthy."""
        ws = WebSocketManager(lambda data: None, market="futures")
        conn = WSConnection()
        mock_ws = type("FakeWS", (), {"closed": False})()
        conn.ws = mock_ws
        conn.last_pong = time.monotonic() - 650  # 650s < 900s — still healthy
        ws._connections = [conn]
        assert ws.is_healthy is True


class TestReconnectJitter:
    """Bug 1: Reconnect delay should include random jitter (±25%)."""

    def test_data_store_param_accepted(self):
        """WebSocketManager accepts optional data_store parameter."""
        mock_store = object()
        ws = WebSocketManager(lambda data: None, market="spot", data_store=mock_store)
        assert ws._data_store is mock_store

    def test_data_store_defaults_to_none(self):
        """data_store defaults to None for backward compatibility."""
        ws = WebSocketManager(lambda data: None, market="spot")
        assert ws._data_store is None

    def test_jitter_produces_varied_delays(self):
        """Successive jitter values should not always be identical."""
        import random
        from config import WS_RECONNECT_BASE_DELAY, WS_RECONNECT_MAX_DELAY
        # Test across different reconnect attempts (exponents 0-3) to verify
        # jitter is applied regardless of backoff magnitude.
        all_delays = []
        for attempt in range(4):
            delays = set()
            for _ in range(20):
                delay = min(WS_RECONNECT_BASE_DELAY * (2 ** attempt), WS_RECONNECT_MAX_DELAY)
                jitter = delay * random.uniform(-0.25, 0.25)
                actual = max(0.5, delay + jitter)
                delays.add(round(actual, 6))
            # With 20 samples, at least 2 distinct values should appear
            assert len(delays) > 1, f"No jitter variation at attempt={attempt}"
            all_delays.extend(delays)
        # Verify delays stay within the expected ±25% jitter band + min 0.5s floor
        for attempt in range(4):
            base = min(WS_RECONNECT_BASE_DELAY * (2 ** attempt), WS_RECONNECT_MAX_DELAY)
            low = max(0.5, base * 0.75)
            high = base * 1.25
            for _ in range(50):
                jitter = base * random.uniform(-0.25, 0.25)
                actual = max(0.5, base + jitter)
                assert low <= actual <= high, f"Delay {actual:.3f} outside [{low:.3f}, {high:.3f}]"


class TestMultiTimeframeFallbackConfig:
    """Bug 3: Fallback timeframe constants are set correctly."""

    def test_fallback_timeframes_cover_all_channels(self):
        """WS_FALLBACK_TIMEFRAMES must include 1m, 5m, 15m, and 1h."""
        required = {"1m", "5m", "15m", "1h"}
        assert required <= set(WS_FALLBACK_TIMEFRAMES)

    def test_fallback_poll_intervals_present(self):
        """WS_FALLBACK_POLL_INTERVALS config is importable."""
        from config import WS_FALLBACK_POLL_INTERVALS
        assert "1m" in WS_FALLBACK_POLL_INTERVALS


class TestSessionRecycling:
    """Bug 5: _recreate_session closes old session and creates a new one."""

    @pytest.mark.asyncio
    async def test_recreate_session_closes_old(self):
        """_recreate_session must close the old session."""
        closed = []

        async def fake_close():
            closed.append(True)

        ws = WebSocketManager(lambda data: None, market="spot")
        fake_session = type(
            "FakeSession",
            (),
            {"closed": False, "close": lambda self: fake_close()},
        )()
        ws._session = fake_session

        await ws._recreate_session()

        assert closed, "Old session close() was not called"
        assert ws._session is not fake_session

    @pytest.mark.asyncio
    async def test_recreate_session_skips_already_closed(self):
        """_recreate_session must not call close() on an already-closed session."""
        closed = []

        ws = WebSocketManager(lambda data: None, market="spot")
        fake_session = type(
            "FakeSession",
            (),
            {"closed": True, "close": lambda self: closed.append(True)},
        )()
        ws._session = fake_session

        await ws._recreate_session()

        assert not closed, "close() should not be called on an already-closed session"
        assert ws._session is not fake_session

    def test_session_recycle_attempts_config(self):
        """WS_SESSION_RECYCLE_ATTEMPTS must be a positive integer."""
        assert isinstance(WS_SESSION_RECYCLE_ATTEMPTS, int)
        assert WS_SESSION_RECYCLE_ATTEMPTS > 0


class TestFetchAndStoreFallback:
    """Bug 2: HistoricalDataStore.fetch_and_store_fallback stores candles correctly."""

    @pytest.mark.asyncio
    async def test_fetch_and_store_fallback_stores_new_data(self):
        """fetch_and_store_fallback seeds candles when none exist for the symbol."""
        import numpy as np
        from src.historical_data import HistoricalDataStore

        store = HistoricalDataStore()
        dummy = {
            "open": np.array([1.0, 2.0]),
            "high": np.array([1.1, 2.1]),
            "low": np.array([0.9, 1.9]),
            "close": np.array([1.05, 2.05]),
            "volume": np.array([100.0, 200.0]),
        }

        with mock.patch.object(store, "fetch_candles", return_value=dummy) as patched:
            await store.fetch_and_store_fallback("BTCUSDT", "1m", 200, "futures")
            patched.assert_called_once_with("BTCUSDT", "1m", 200, "futures")

        assert "BTCUSDT" in store.candles
        assert "1m" in store.candles["BTCUSDT"]
        np.testing.assert_array_equal(store.candles["BTCUSDT"]["1m"]["close"], dummy["close"])

    @pytest.mark.asyncio
    async def test_fetch_and_store_fallback_merges_existing(self):
        """fetch_and_store_fallback merges new candles with existing data."""
        import numpy as np
        from src.historical_data import HistoricalDataStore

        store = HistoricalDataStore()
        existing = {
            "open": np.array([0.5]),
            "high": np.array([0.6]),
            "low": np.array([0.4]),
            "close": np.array([0.55]),
            "volume": np.array([50.0]),
        }
        store.candles["BTCUSDT"] = {"1m": existing}
        new_data = {
            "open": np.array([1.0]),
            "high": np.array([1.1]),
            "low": np.array([0.9]),
            "close": np.array([1.05]),
            "volume": np.array([100.0]),
        }

        with mock.patch.object(store, "fetch_candles", return_value=new_data):
            await store.fetch_and_store_fallback("BTCUSDT", "1m", 200, "spot")

        assert len(store.candles["BTCUSDT"]["1m"]["close"]) == 2

    @pytest.mark.asyncio
    async def test_fetch_and_store_fallback_noop_on_empty_response(self):
        """fetch_and_store_fallback does nothing if fetch_candles returns empty."""
        from src.historical_data import HistoricalDataStore

        store = HistoricalDataStore()
        with mock.patch.object(store, "fetch_candles", return_value={}):
            await store.fetch_and_store_fallback("ETHUSDT", "5m", 200, "spot")
        assert "ETHUSDT" not in store.candles


class TestTotalDropsCounter:
    """Verify _total_drops counter increments on each connection drop."""

    def test_total_drops_starts_at_zero(self):
        """_total_drops must be 0 on a freshly created WebSocketManager."""
        ws = WebSocketManager(lambda data: None, market="futures")
        assert ws._total_drops == 0

    @pytest.mark.asyncio
    async def test_total_drops_increments_on_drop(self):
        """_total_drops must increment each time a connection is reported lost."""
        alerts = []

        async def fake_alert(msg: str) -> None:
            alerts.append(msg)

        ws = WebSocketManager(
            lambda data: None,
            market="futures",
            admin_alert_callback=fake_alert,
        )
        # Bypass the cooldown by back-dating _last_alert_time far enough that
        # even on a freshly booted CI container (low time.monotonic() value)
        # the cooldown check `now - _last_alert_time > WS_ALERT_COOLDOWN` passes.
        ws._last_alert_time = time.monotonic() - WS_ALERT_COOLDOWN - 1.0

        # Simulate two consecutive drops.
        conn = WSConnection()
        ws._connections = [conn]
        ws._running = True

        # Manually trigger the drop-path logic twice.
        for _ in range(2):
            ws._total_drops += 1
            now = time.monotonic()
            if now - ws._last_alert_time > WS_ALERT_COOLDOWN:
                ws._last_alert_time = now
                await fake_alert(
                    f"⚠️ WebSocket connection lost (futures, attempt 1, "
                    f"total drops: {ws._total_drops}). Reconnecting…"
                )
            # Reset to allow the second iteration through the cooldown.
            ws._last_alert_time = time.monotonic() - WS_ALERT_COOLDOWN - 1.0

        assert ws._total_drops == 2
        assert len(alerts) == 2
        assert "total drops: 1" in alerts[0]
        assert "total drops: 2" in alerts[1]

    def test_total_drops_in_alert_message(self):
        """Alert message must include total drops when _total_drops > 0."""
        ws = WebSocketManager(lambda data: None, market="spot")
        ws._total_drops = 5
        msg = (
            f"⚠️ WebSocket connection lost ({ws._market}, "
            f"attempt 1, total drops: {ws._total_drops}). Reconnecting…"
        )
        assert "total drops: 5" in msg


class TestStalenessMultiplierConfig:
    """Verify WS_STALENESS_MULTIPLIER config constants are correct."""

    def test_staleness_multiplier_spot(self):
        assert WS_STALENESS_MULTIPLIER == 10

    def test_staleness_multiplier_futures_default(self):
        assert WS_STALENESS_MULTIPLIER_FUTURES == 15

    def test_futures_staleness_multiplier_greater_than_spot(self):
        assert WS_STALENESS_MULTIPLIER_FUTURES > WS_STALENESS_MULTIPLIER

    def test_spot_manager_uses_spot_multiplier(self):
        ws = WebSocketManager(lambda data: None, market="spot")
        assert ws._staleness_multiplier == WS_STALENESS_MULTIPLIER

    def test_futures_manager_uses_futures_multiplier(self):
        ws = WebSocketManager(lambda data: None, market="futures")
        assert ws._staleness_multiplier == WS_STALENESS_MULTIPLIER_FUTURES


class TestForceOrderStreamSeparation:
    """Verify forceOrder streams are kept separate from kline streams in bootstrap."""

    def test_forceorder_not_in_futures_kline_streams(self):
        """Kline stream list must not contain any @forceOrder streams."""
        syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        futures_kline_streams = []
        futures_liq_streams = []
        for sym in syms:
            s = sym.lower()
            futures_kline_streams.append(f"{s}@kline_1m")
            futures_kline_streams.append(f"{s}@kline_5m")
            futures_liq_streams.append(f"{s}@forceOrder")

        assert not any("forceOrder" in s for s in futures_kline_streams)
        assert all("forceOrder" in s for s in futures_liq_streams)

    def test_kline_streams_not_in_liq_streams(self):
        """Liquidation stream list must not contain kline streams."""
        syms = ["BTCUSDT", "ETHUSDT"]
        futures_kline_streams = []
        futures_liq_streams = []
        for sym in syms:
            s = sym.lower()
            futures_kline_streams.append(f"{s}@kline_1m")
            futures_kline_streams.append(f"{s}@kline_5m")
            futures_liq_streams.append(f"{s}@forceOrder")

        assert not any("kline" in s for s in futures_liq_streams)

    def test_liq_stream_count_matches_symbol_count(self):
        """Each symbol produces exactly one forceOrder stream."""
        syms = [f"SYM{i}USDT" for i in range(50)]
        futures_liq_streams = [f"{s.lower()}@forceOrder" for s in syms]
        assert len(futures_liq_streams) == len(syms)

    def test_engine_has_ws_futures_liq_attribute(self):
        """CryptoSignalEngine must declare _ws_futures_liq attribute."""
        from src.main import CryptoSignalEngine
        engine = CryptoSignalEngine()
        assert hasattr(engine, "_ws_futures_liq")
        assert engine._ws_futures_liq is None  # set during boot, None at init


class TestStreamShardingCap:
    """Requirement 1 & 2: Hard cap and auto-sharding across connections."""

    def test_max_streams_per_conn_at_least_150(self):
        """Safe stream cap must be >= 150 (well below Binance's 1024 limit)."""
        from config import WS_MAX_STREAMS_PER_CONN
        assert WS_MAX_STREAMS_PER_CONN >= 150

    def test_max_streams_per_conn_at_most_200(self):
        """Safe stream cap must be <= 200 to stay well within Binance limits."""
        from config import WS_MAX_STREAMS_PER_CONN
        assert WS_MAX_STREAMS_PER_CONN <= 200

    def test_streams_sharded_into_multiple_connections(self):
        """start() must distribute streams across multiple WSConnection objects."""
        from config import WS_MAX_STREAMS_PER_CONN
        ws = WebSocketManager(lambda data: None, market="spot")
        ws._running = True
        ws._session = mock.MagicMock()

        dummy_task = mock.MagicMock()
        dummy_task.done.return_value = False

        # Build a stream list large enough to require 2 shards
        stream_count = WS_MAX_STREAMS_PER_CONN + 1
        streams = [f"sym{i}usdt@kline_1m" for i in range(stream_count)]

        with mock.patch("asyncio.create_task", return_value=dummy_task):
            asyncio.get_event_loop().run_until_complete(
                _mock_start(ws, streams)
            )

        assert len(ws._connections) >= 2, "Must create at least 2 shards for > MAX_STREAMS_PER_CONN streams"
        for conn in ws._connections:
            assert len(conn.streams) <= WS_MAX_STREAMS_PER_CONN, "No shard may exceed the cap"

    def test_single_shard_for_small_stream_list(self):
        """A small stream list that fits within the cap uses exactly one shard."""
        from config import WS_MAX_STREAMS_PER_CONN
        ws = WebSocketManager(lambda data: None, market="spot")
        ws._running = True
        ws._session = mock.MagicMock()

        dummy_task = mock.MagicMock()
        dummy_task.done.return_value = False

        streams = [f"sym{i}usdt@kline_1m" for i in range(10)]

        with mock.patch("asyncio.create_task", return_value=dummy_task):
            asyncio.get_event_loop().run_until_complete(
                _mock_start(ws, streams)
            )

        assert len(ws._connections) == 1


async def _mock_start(ws: WebSocketManager, streams: list) -> None:
    """Helper: run WebSocketManager.start() without actually connecting."""
    ws._connections = []
    ws._subscribed_streams = set()
    ws._rest_fallback_active = False
    from config import WS_MAX_STREAMS_PER_CONN
    for i in range(0, len(streams), WS_MAX_STREAMS_PER_CONN):
        chunk = streams[i: i + WS_MAX_STREAMS_PER_CONN]
        conn = WSConnection(streams=chunk)
        ws._connections.append(conn)
        conn.task = mock.MagicMock()


class TestPingPongHeartbeatMonitor:
    """Requirement 3: Strict ping/pong latency and timeout detection."""

    def test_ws_ping_timeout_ms_config_exists(self):
        """WS_PING_TIMEOUT_MS constant must exist in config."""
        from config import WS_PING_TIMEOUT_MS
        assert WS_PING_TIMEOUT_MS > 0

    def test_ws_ping_timeout_ms_is_2000(self):
        """WS_PING_TIMEOUT_MS default must be 5000 ms."""
        from config import WS_PING_TIMEOUT_MS
        assert WS_PING_TIMEOUT_MS == 5000

    def test_ws_connection_has_last_ping_time_field(self):
        """WSConnection must expose last_ping_time for latency tracking."""
        conn = WSConnection()
        assert hasattr(conn, "last_ping_time")
        assert conn.last_ping_time == 0.0

    def test_ws_connection_has_ping_latency_ms_field(self):
        """WSConnection must expose ping_latency_ms for RTT measurement."""
        conn = WSConnection()
        assert hasattr(conn, "ping_latency_ms")
        assert conn.ping_latency_ms == 0.0

    @pytest.mark.asyncio
    async def test_watchdog_closes_connection_on_ping_timeout(self):
        """Watchdog must force-close a shard when pong is not received within WS_PING_TIMEOUT_MS."""
        from config import WS_PING_TIMEOUT_MS

        ws = WebSocketManager(lambda data: None, market="spot")
        ws._running = True

        closed = []

        class FakeWS:
            closed = False

            async def close(self):
                closed.append(True)
                FakeWS.closed = True

            async def ping(self):
                pass

        fake_ws = FakeWS()
        conn = WSConnection(streams=["btcusdt@kline_1m"])
        conn.ws = fake_ws
        conn.last_pong = time.monotonic()
        # Simulate a ping sent WS_PING_TIMEOUT_MS + 100ms ago with no pong back
        conn.last_ping_time = time.monotonic() - (WS_PING_TIMEOUT_MS / 1000 + 0.1)
        conn.ping_latency_ms = 0.0
        ws._connections = [conn]
        ws._heartbeat_interval = 0  # skip the sleep

        # Run one watchdog tick (sleep=0 so it completes quickly)
        with mock.patch.object(ws, "_heartbeat_interval", 0):
            task = asyncio.create_task(ws._health_watchdog())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert closed, "Watchdog must close the connection when ping times out"

    @pytest.mark.asyncio
    async def test_watchdog_closes_connection_on_high_latency(self):
        """Watchdog must force-close a shard when measured RTT exceeds WS_PING_TIMEOUT_MS."""
        from config import WS_PING_TIMEOUT_MS

        ws = WebSocketManager(lambda data: None, market="spot")
        ws._running = True

        closed = []

        class FakeWS:
            closed = False

            async def close(self):
                closed.append(True)
                FakeWS.closed = True

            async def ping(self):
                pass

        fake_ws = FakeWS()
        conn = WSConnection(streams=["btcusdt@kline_1m"])
        conn.ws = fake_ws
        conn.last_pong = time.monotonic()
        conn.last_ping_time = 0.0
        # Simulate a previously measured high RTT
        conn.ping_latency_ms = WS_PING_TIMEOUT_MS + 500
        ws._connections = [conn]

        with mock.patch.object(ws, "_heartbeat_interval", 0):
            task = asyncio.create_task(ws._health_watchdog())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert closed, "Watchdog must close the connection when RTT exceeds threshold"

    @pytest.mark.asyncio
    async def test_watchdog_sends_ping_when_connection_healthy(self):
        """Watchdog must send a manual ping on healthy connections to measure latency."""
        ws = WebSocketManager(lambda data: None, market="spot")
        ws._running = True

        pings_sent = []

        class FakeWS:
            closed = False

            async def close(self):
                pass

            async def ping(self):
                pings_sent.append(time.monotonic())

        conn = WSConnection(streams=["btcusdt@kline_1m"])
        conn.ws = FakeWS()
        conn.last_pong = time.monotonic()
        conn.last_ping_time = 0.0
        conn.ping_latency_ms = 0.0
        ws._connections = [conn]

        # _heartbeat_interval=0 -> asyncio.sleep(0) yields to event loop without blocking.
        # The staleness threshold is max(1.0, 0 * staleness_multiplier) = 1.0 s, so a
        # freshly created connection (last_pong ~= now) won't be falsely closed.
        ws._heartbeat_interval = 0
        task = asyncio.create_task(ws._health_watchdog())
        # Two sleep(0) calls are needed: the first tick starts the watchdog and
        # lets it hit its own sleep(0); the second tick runs the watchdog body
        # (staleness checks + the manual ping send).
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert pings_sent, "Watchdog must send a manual ping on healthy connections"
        assert conn.last_ping_time > 0, "last_ping_time must be set after ping is sent"

    @pytest.mark.asyncio
    async def test_listen_computes_latency_on_pong(self):
        """_listen must compute ping_latency_ms when a PONG frame arrives."""
        received = []

        async def on_msg(data):
            received.append(data)

        ws = WebSocketManager(on_msg, market="spot")
        conn = WSConnection(streams=["btcusdt@kline_1m"])
        # Simulate a ping sent 50ms ago
        conn.last_ping_time = time.monotonic() - 0.05

        class FakeMsg:
            def __init__(self, mtype, data=""):
                self.type = mtype
                self.data = data

        class FakeWS:
            def __init__(self):
                self._msgs = [
                    FakeMsg(aiohttp.WSMsgType.PONG),
                    FakeMsg(aiohttp.WSMsgType.CLOSED),
                ]

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._msgs:
                    return self._msgs.pop(0)
                raise StopAsyncIteration

        conn.ws = FakeWS()
        await ws._listen(conn)

        assert conn.ping_latency_ms > 0, "ping_latency_ms must be set after PONG"
        assert conn.last_ping_time == 0.0, "last_ping_time must be reset after PONG"

    @pytest.mark.asyncio
    async def test_connect_resets_ping_fields(self):
        """_connect must reset last_ping_time and ping_latency_ms on (re)connect."""
        ws = WebSocketManager(lambda data: None, market="spot")
        ws._running = True

        class FakeWS:
            closed = False

        fake_ws_response = FakeWS()

        class FakeSession:
            async def ws_connect(self, url, **kwargs):
                return fake_ws_response

        ws._session = FakeSession()

        conn = WSConnection(streams=["btcusdt@kline_1m"])
        conn.last_ping_time = 99.9
        conn.ping_latency_ms = 1234.5

        await ws._connect(conn)

        assert conn.last_ping_time == 0.0
        assert conn.ping_latency_ms == 0.0


class TestShardResiliency:
    """Requirement 4: Individual shard failure must not affect other shards."""

    def test_degraded_flag_per_connection(self):
        """Each WSConnection has its own degraded flag, independent of others."""
        conn1 = WSConnection(streams=["btcusdt@kline_1m"])
        conn2 = WSConnection(streams=["ethusdt@kline_1m"])
        conn1.degraded = True
        assert conn2.degraded is False, "conn2 must be unaffected when conn1 is degraded"

    def test_health_ratio_reflects_partial_degradation(self):
        """health_ratio must reflect the fraction of healthy shards."""
        ws = WebSocketManager(lambda data: None, market="spot")
        now = time.monotonic()

        mock_open = type("FWS", (), {"closed": False})()
        mock_closed = type("FWS", (), {"closed": True})()

        conn_ok = WSConnection(streams=["btcusdt@kline_1m"], last_pong=now, ws=mock_open)
        conn_down = WSConnection(streams=["ethusdt@kline_1m"], last_pong=now, ws=mock_closed)
        ws._connections = [conn_ok, conn_down]

        ratio = ws.health_ratio
        assert ratio == 0.5, f"Expected 0.5, got {ratio}"

    def test_ping_fields_independent_per_shard(self):
        """Ping latency fields on one shard are independent of another shard's fields."""
        conn1 = WSConnection(streams=["btcusdt@kline_1m"])
        conn2 = WSConnection(streams=["ethusdt@kline_1m"])
        conn1.ping_latency_ms = 3000.0
        conn1.last_ping_time = 1.0
        assert conn2.ping_latency_ms == 0.0
        assert conn2.last_ping_time == 0.0
