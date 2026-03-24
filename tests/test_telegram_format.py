"""Tests for src.telegram_bot – signal formatting."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

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
            setup_class="BREAKOUT_RETEST",
            quality_tier="A",
            entry_zone="32,120 - 32,150",
            invalidation_summary="Below reclaim low + ATR buffer",
            analyst_reason="Breakout retest holding above value.",
            execution_note="Retest confirmed; do not chase.",
            component_scores={"market": 19.0, "setup": 21.0, "execution": 16.0, "risk": 15.0, "context": 8.0},
            timestamp=utcnow(),
        )
        # Test the new compact format
        text = TelegramBot.format_signal(sig)
        assert "⚡" in text
        assert "SCALP" in text
        assert "BTCUSDT" in text
        assert "LONG" in text
        assert "32,150" in text
        assert "Aggressive" in text

        # Test the legacy format still exposes the old fields
        legacy_text = TelegramBot.format_signal_legacy(sig)
        assert "⚡" in legacy_text
        assert r"360\_SCALP" in legacy_text
        assert "BTCUSDT" in legacy_text
        assert "LONG" in legacy_text
        assert "32,150" in legacy_text
        assert "87%" in legacy_text
        assert "Whale Activity" in legacy_text
        assert "Aggressive" in legacy_text
        assert "Trailing Active" in legacy_text
        assert "Breakout Retest" in legacy_text
        assert "A" in legacy_text
        assert "Thesis" in legacy_text
        assert "Execution" in legacy_text

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
        # New format
        text = TelegramBot.format_signal(sig)
        assert "🏛️" in text
        assert "SHORT" in text
        assert "92.0" in text

        # Legacy format keeps old fields
        legacy_text = TelegramBot.format_signal_legacy(sig)
        assert "🏛️" in legacy_text
        assert "SHORT" in legacy_text
        assert "⬇️" in legacy_text
        assert "92%" in legacy_text

    def test_spot_format_with_ai_adaptive(self):
        sig = Signal(
            channel="360_SPOT",
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
            ai_sentiment_summary="Spot Accumulation",
            risk_label="Conservative",
            timestamp=utcnow(),
        )
        # New format
        text = TelegramBot.format_signal(sig)
        assert "📈" in text
        assert "Dynamic/trailing" in text
        assert "95.0" in text

        # Legacy format includes trailing_desc
        legacy_text = TelegramBot.format_signal_legacy(sig)
        assert "📈" in legacy_text
        assert "Dynamic/trailing" in legacy_text
        assert "AI Adaptive" in legacy_text
        assert "95%" in legacy_text

    def test_spot_format(self):
        sig = Signal(
            channel="360_SPOT",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32100,
            stop_loss=32050,
            tp1=32150,
            tp2=32200,
            tp3=None,
            trailing_active=True,
            trailing_desc="3×ATR",
            confidence=80,
            ai_sentiment_label="Positive",
            ai_sentiment_summary="",
            risk_label="Conservative",
            timestamp=utcnow(),
        )
        # New format
        text = TelegramBot.format_signal(sig)
        assert "📈" in text
        assert "Conservative" in text
        assert "80.0" in text

        # Legacy format
        legacy_text = TelegramBot.format_signal_legacy(sig)
        assert "📈" in legacy_text
        assert "Conservative" in legacy_text
        assert "80%" in legacy_text


class TestEscapeMd:
    def test_escape_backtick(self):
        assert TelegramBot._escape_md("price `0.175700`") == "price \\`0.175700\\`"

    def test_escape_asterisk(self):
        assert TelegramBot._escape_md("*bold*") == "\\*bold\\*"

    def test_escape_underscore(self):
        assert TelegramBot._escape_md("_italic_") == "\\_italic\\_"

    def test_escape_bracket(self):
        assert TelegramBot._escape_md("[link]") == "\\[link]"

    def test_escape_backslash(self):
        assert TelegramBot._escape_md("a\\b") == "a\\\\b"

    def test_escape_combined(self):
        result = TelegramBot._escape_md("Sweep SHORT at 0.3572 | FVG 0.3543-0.3538")
        # Pipe and digits should pass through unchanged; no special MD chars here
        assert result == "Sweep SHORT at 0.3572 | FVG 0.3543-0.3538"

    def test_escape_with_backtick_in_liquidity(self):
        raw = "Sweep `SHORT` at 0.3572 | FVG 0.3543-0.3538"
        escaped = TelegramBot._escape_md(raw)
        assert "\\`" in escaped
        assert "`" not in escaped.replace("\\`", "")

    def test_plain_text_unmodified(self):
        assert TelegramBot._escape_md("No special chars here") == "No special chars here"


class TestFormatSignalEscaping:
    def test_liquidity_info_with_pipe_and_fvg(self):
        """Liquidity info containing | and decimals should appear escaped in output."""
        sig = Signal(
            channel="360_SCALP",
            symbol="PIPPINUSDT",
            direction=Direction.SHORT,
            entry=0.35599,
            stop_loss=0.35642,
            tp1=0.35671,
            tp2=0.35592,
            tp3=0.35512,
            trailing_active=True,
            trailing_desc="1.5×ATR",
            confidence=78,
            ai_sentiment_label="Neutral",
            ai_sentiment_summary="No API key",
            risk_label="Low",
            market_phase="VOLATILE",
            liquidity_info="Sweep SHORT at 0.3572 | FVG 0.3543-0.3538",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        # The raw pipe and text should still appear (no MD special chars in this string)
        assert "Sweep SHORT at 0.3572 | FVG 0.3543-0.3538" in text

    def test_liquidity_info_with_backtick_escaped(self):
        """Backticks in liquidity_info must be escaped to prevent Markdown parse errors."""
        sig = Signal(
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=69640,
            stop_loss=69508,
            tp1=69848,
            tp2=69880,
            tp3=69911,
            trailing_active=False,
            confidence=73,
            ai_sentiment_label="Neutral",
            ai_sentiment_summary="",
            risk_label="Low",
            market_phase="QUIET",
            liquidity_info="Sweep LONG at `69594` | FVG 69790-69786",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal(sig)
        assert "\\`69594\\`" in text

    def test_ai_sentiment_summary_escaped(self):
        """AI sentiment summary with Markdown chars should be escaped (legacy format)."""
        sig = Signal(
            channel="360_SCALP",
            symbol="ETHUSDT",
            direction=Direction.LONG,
            entry=2042,
            stop_loss=2037,
            tp1=2049,
            tp2=2050,
            tp3=2051,
            trailing_active=False,
            confidence=73,
            ai_sentiment_label="Neutral",
            ai_sentiment_summary="Price *near* support_level",
            risk_label="Low",
            market_phase="QUIET",
            liquidity_info="Standard",
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal_legacy(sig)
        assert "\\*near\\*" in text
        assert "support\\_level" in text

    def test_trailing_desc_escaped(self):
        """trailing_desc with × should pass through; * would be escaped (legacy format)."""
        sig = Signal(
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32000,
            stop_loss=31900,
            tp1=32100,
            tp2=32200,
            trailing_active=True,
            trailing_desc="1.5*ATR",
            confidence=80,
            timestamp=utcnow(),
        )
        text = TelegramBot.format_signal_legacy(sig)
        assert "1.5\\*ATR" in text

    def test_premium_fields_render_markdown_safe(self):
        sig = Signal(
            channel="360_GEM",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32000,
            stop_loss=31900,
            tp1=32150,
            tp2=32300,
            confidence=91,
            quality_tier="A+",
            setup_class="LIQUIDITY_SWEEP_REVERSAL",
            entry_zone="31_980 - 32_020",
            invalidation_summary="Below sweep_low *with* buffer",
            analyst_reason="Sweep reclaim [confirmed]",
            execution_note="Reclaim only; don't chase _extension_.",
            component_scores={"market": 22.0, "setup": 22.0, "execution": 17.0, "risk": 17.0, "context": 9.0},
            timestamp=utcnow(),
        )
        # New format: shows quality tier, setup, risk; escapes narrative fields
        text = TelegramBot.format_signal(sig)
        assert "A+" in text
        assert "\\*with\\*" in text  # from invalidation_summary in narrative

        # Legacy format: shows all fields including entry_zone and execution_note
        legacy_text = TelegramBot.format_signal_legacy(sig)
        assert "A+" in legacy_text
        assert "31\\_980 - 32\\_020" in legacy_text
        assert "\\*with\\*" in legacy_text
        assert "\\[confirmed]" in legacy_text
        assert "don't" in legacy_text
        assert "\\_extension\\_" in legacy_text


class TestSendMessageFallback:
    def test_plain_text_retry_on_markdown_parse_error(self):
        """send_message retries without parse_mode when Telegram returns 400 parse error."""
        bot = TelegramBot()
        bot._token = "fake-token"

        call_count = 0

        async def _run():
            nonlocal call_count

            # First response: 400 with parse entities error
            first_resp = MagicMock()
            first_resp.status = 400
            first_resp.text = AsyncMock(
                return_value='{"ok":false,"error_code":400,"description":"Bad Request: can\'t parse entities: Can\'t find end of the entity starting at byte offset 309"}'
            )
            first_resp.__aenter__ = AsyncMock(return_value=first_resp)
            first_resp.__aexit__ = AsyncMock(return_value=False)

            # Second response (retry): 200 OK
            second_resp = MagicMock()
            second_resp.status = 200
            second_resp.__aenter__ = AsyncMock(return_value=second_resp)
            second_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = MagicMock()
            mock_session.closed = False
            mock_session.post = MagicMock(side_effect=[first_resp, second_resp])

            bot._session = mock_session

            result = await bot.send_message("123456", "test *message* `broken")
            call_count = mock_session.post.call_count
            return result

        result = asyncio.new_event_loop().run_until_complete(_run())
        assert result is True
        assert call_count == 2  # initial attempt + retry

    def test_no_retry_on_other_400_errors(self):
        """send_message does NOT retry on 400 errors unrelated to Markdown parsing."""
        bot = TelegramBot()
        bot._token = "fake-token"

        call_count = 0

        async def _run():
            nonlocal call_count

            resp = MagicMock()
            resp.status = 400
            resp.text = AsyncMock(
                return_value='{"ok":false,"error_code":400,"description":"Bad Request: chat not found"}'
            )
            resp.__aenter__ = AsyncMock(return_value=resp)
            resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = MagicMock()
            mock_session.closed = False
            mock_session.post = MagicMock(return_value=resp)

            bot._session = mock_session

            result = await bot.send_message("123456", "hello")
            call_count = mock_session.post.call_count
            return result

        result = asyncio.new_event_loop().run_until_complete(_run())
        assert result is False
        assert call_count == 1  # no retry


class TestFormatHighlightMessage:
    """Tests for TelegramBot.format_highlight_message()."""

    def _make_signal(self, tp3=32400.0, setup_class="BREAKOUT_RETEST", quality_tier="A"):
        sig = Signal(
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=32000.0,
            stop_loss=31900.0,
            tp1=32100.0,
            tp2=32200.0,
            confidence=88.0,
        )
        sig.tp3 = tp3
        sig.setup_class = setup_class
        sig.quality_tier = quality_tier
        sig.market_phase = "STRONG_TREND"
        return sig

    def test_highlight_contains_winner_header(self):
        sig = self._make_signal()
        text = TelegramBot.format_highlight_message(sig, 2, 0.62)
        assert "WINNING TRADE HIGHLIGHT" in text

    def test_highlight_tp2_emoji(self):
        sig = self._make_signal()
        text = TelegramBot.format_highlight_message(sig, 2, 0.62)
        assert "TP2 HIT" in text
        assert "✅✅" in text

    def test_highlight_tp3_emoji(self):
        sig = self._make_signal()
        text = TelegramBot.format_highlight_message(sig, 3, 1.25)
        assert "TP3 HIT" in text
        assert "✅✅✅" in text

    def test_highlight_contains_pnl(self):
        sig = self._make_signal()
        text = TelegramBot.format_highlight_message(sig, 2, 1.23)
        assert "+1.23%" in text

    def test_highlight_contains_cta(self):
        sig = self._make_signal()
        text = TelegramBot.format_highlight_message(sig, 2, 0.62)
        assert "Premium" in text

    def test_highlight_contains_confidence(self):
        sig = self._make_signal()
        text = TelegramBot.format_highlight_message(sig, 2, 0.62)
        assert "88%" in text

    def test_highlight_contains_quality_tier(self):
        sig = self._make_signal(quality_tier="A+")
        text = TelegramBot.format_highlight_message(sig, 2, 0.62)
        assert "A+" in text

    def test_highlight_no_setup_label_for_unclassified(self):
        sig = self._make_signal(setup_class="UNCLASSIFIED")
        text = TelegramBot.format_highlight_message(sig, 2, 0.62)
        assert "Setup" not in text

    def test_highlight_short_direction(self):
        sig = Signal(
            channel="360_SWING",
            symbol="ETHUSDT",
            direction=Direction.SHORT,
            entry=2000.0,
            stop_loss=2050.0,
            tp1=1950.0,
            tp2=1900.0,
            confidence=82.0,
        )
        text = TelegramBot.format_highlight_message(sig, 2, 2.5)
        assert "SHORT" in text
        assert "⬇️" in text

    def test_highlight_tp3_uses_tp3_price(self):
        sig = self._make_signal(tp3=32400.0)
        text = TelegramBot.format_highlight_message(sig, 3, 1.25)
        assert "32,400" in text

    def test_highlight_tp2_uses_tp2_price(self):
        sig = self._make_signal()
        text = TelegramBot.format_highlight_message(sig, 2, 0.62)
        assert "32,200" in text


class TestFormatDailyRecap:
    """Tests for TelegramBot.format_daily_recap()."""

    def _make_summary(self, total=10, wins=7, losses=2, breakeven=1,
                      win_rate=77.8, avg_pnl=1.5, best=None, top=None):
        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "best_trade": best,
            "top_trades": top or [],
        }

    def test_recap_contains_header(self):
        text = TelegramBot.format_daily_recap(self._make_summary())
        assert "DAILY PERFORMANCE RECAP" in text

    def test_recap_shows_total(self):
        text = TelegramBot.format_daily_recap(self._make_summary(total=15))
        assert "15" in text

    def test_recap_shows_win_rate(self):
        text = TelegramBot.format_daily_recap(self._make_summary(win_rate=75.0))
        assert "75%" in text

    def test_recap_shows_wins_losses(self):
        text = TelegramBot.format_daily_recap(self._make_summary(wins=5, losses=3))
        assert "5" in text
        assert "3" in text

    def test_recap_contains_cta(self):
        text = TelegramBot.format_daily_recap(self._make_summary())
        assert "Premium" in text

    def test_recap_top_trades_section(self):
        from src.performance_tracker import SignalRecord
        record = SignalRecord(
            signal_id="X",
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction="LONG",
            entry=50000.0,
            hit_tp=2,
            hit_sl=False,
            pnl_pct=2.0,
            confidence=85.0,
            signal_quality_pnl_pct=2.0,
        )
        summary = self._make_summary(top=[record], best=record)
        text = TelegramBot.format_daily_recap(summary)
        assert "Top 3 Trades" in text
        assert "BTCUSDT" in text

    def test_recap_no_top_trades_when_empty(self):
        text = TelegramBot.format_daily_recap(self._make_summary(top=[]))
        assert "Top 3 Trades" not in text


# ---------------------------------------------------------------------------
# send_message retry logic tests
# ---------------------------------------------------------------------------


class TestSendMessageRetry:
    """Tests for send_message retry behaviour on various HTTP error responses."""

    def _make_bot(self) -> TelegramBot:
        bot = TelegramBot()
        bot._token = "test-token"
        return bot

    @pytest.mark.asyncio
    async def test_retries_on_429_with_retry_after(self, monkeypatch):
        """send_message honours retry_after from a 429 response and retries."""
        import json as _json

        bot = self._make_bot()
        calls = []
        sleep_args = []

        async def instant_sleep(secs):
            sleep_args.append(secs)

        monkeypatch.setattr(asyncio, "sleep", instant_sleep)

        # First response: 429 with retry_after=3; second: 200 success
        responses = [
            (429, _json.dumps({"ok": False, "parameters": {"retry_after": 3}})),
            (200, "{}"),
        ]

        class FakeResp:
            def __init__(self, status, body):
                self.status = status
                self._body = body

            async def text(self):
                return self._body

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        class FakeSession:
            closed = False

            def post(self, url, **kwargs):
                status, body = responses.pop(0)
                calls.append(status)
                return FakeResp(status, body)

            async def close(self):
                pass

        bot._session = FakeSession()

        result = await bot.send_message("chat123", "hello")
        assert result is True
        assert calls == [429, 200]
        assert sleep_args == [3.0]

    @pytest.mark.asyncio
    async def test_retries_on_500_with_exponential_backoff(self, monkeypatch):
        """send_message retries on 5xx errors with exponential back-off."""
        bot = self._make_bot()
        sleep_args = []

        async def instant_sleep(secs):
            sleep_args.append(secs)

        monkeypatch.setattr(asyncio, "sleep", instant_sleep)

        responses = [
            (500, "Internal Server Error"),
            (500, "Internal Server Error"),
            (200, "{}"),
        ]

        class FakeResp:
            def __init__(self, status, body):
                self.status = status
                self._body = body

            async def text(self):
                return self._body

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        class FakeSession:
            closed = False

            def post(self, url, **kwargs):
                status, body = responses.pop(0)
                return FakeResp(status, body)

            async def close(self):
                pass

        bot._session = FakeSession()

        result = await bot.send_message("chat123", "hello")
        assert result is True
        # Back-off: 2**0=1, 2**1=2
        assert sleep_args == [1, 2]

    @pytest.mark.asyncio
    async def test_no_retry_on_403(self, monkeypatch):
        """send_message does NOT retry on non-recoverable 4xx errors (e.g. 403)."""
        bot = self._make_bot()
        call_count = [0]

        class FakeResp:
            status = 403
            _body = "Forbidden"

            async def text(self):
                return self._body

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        class FakeSession:
            closed = False

            def post(self, url, **kwargs):
                call_count[0] += 1
                return FakeResp()

            async def close(self):
                pass

        bot._session = FakeSession()

        result = await bot.send_message("chat123", "hello")
        assert result is False
        assert call_count[0] == 1  # no retries
