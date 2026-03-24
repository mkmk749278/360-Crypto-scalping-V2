"""Tests for Phase 2: new compact format_signal() and legacy format_signal_legacy()."""

from __future__ import annotations

import pytest

from src.channels.base import Signal
from src.smc import Direction
from src.telegram_bot import TelegramBot
from src.utils import utcnow


def _make_signal(**kwargs) -> Signal:
    defaults = dict(
        channel="360_SCALP",
        symbol="BTCUSDT",
        direction=Direction.LONG,
        entry=67234.50,
        stop_loss=66980.00,
        tp1=67520.00,
        tp2=67800.00,
        tp3=68100.00,
        confidence=82.4,
        risk_label="LOW",
        quality_tier="A+",
        setup_class="SWEEP_RECLAIM",
        liquidity_info="BTC sweep at 67200",
        invalidation_summary="bullish FVG",
        timestamp=utcnow(),
    )
    defaults.update(kwargs)
    return Signal(**defaults)


class TestNewFormatSignal:
    def test_compact_header_format(self):
        """New format: emoji │ chan_name │ SYMBOL │ DIRECTION on first line."""
        sig = _make_signal()
        text = TelegramBot.format_signal(sig)
        assert "⚡" in text
        assert "SCALP" in text
        assert "BTCUSDT" in text
        assert "LONG" in text

    def test_entry_sl_tp_present(self):
        """New format shows Entry, SL, TP1, TP2."""
        sig = _make_signal()
        text = TelegramBot.format_signal(sig)
        assert "Entry" in text
        assert "SL" in text
        assert "TP1" in text
        assert "TP2" in text

    def test_percentage_calculations_sl(self):
        """SL percentage is negative for a LONG signal."""
        sig = _make_signal(entry=67234.50, stop_loss=66980.00)
        text = TelegramBot.format_signal(sig)
        # SL is below entry → negative percentage
        assert "-" in text

    def test_percentage_calculations_tp(self):
        """TP percentages are positive for a LONG signal."""
        sig = _make_signal(entry=67234.50, tp1=67520.00)
        text = TelegramBot.format_signal(sig)
        # TP1 is above entry → positive percentage
        assert "+" in text

    def test_tp3_shown_when_present(self):
        sig = _make_signal(tp3=68100.00)
        text = TelegramBot.format_signal(sig)
        assert "TP3" in text
        assert "Dynamic/trailing" not in text

    def test_tp3_dynamic_when_none(self):
        sig = _make_signal(tp3=None)
        text = TelegramBot.format_signal(sig)
        assert "Dynamic/trailing" in text

    def test_setup_and_confidence_line(self):
        sig = _make_signal(setup_class="SWEEP_RECLAIM", confidence=82.4, quality_tier="A+")
        text = TelegramBot.format_signal(sig)
        assert "SWEEP RECLAIM" in text
        assert "82.4" in text
        assert "A+" in text

    def test_estimated_hold_time_scalp(self):
        sig = _make_signal(channel="360_SCALP")
        text = TelegramBot.format_signal(sig)
        assert "~1-2h" in text

    def test_estimated_hold_time_swing(self):
        sig = _make_signal(channel="360_SWING", risk_label="LOW")
        text = TelegramBot.format_signal(sig)
        assert "~1-2d" in text

    def test_estimated_hold_time_spot(self):
        sig = _make_signal(channel="360_SPOT", risk_label="LOW")
        text = TelegramBot.format_signal(sig)
        assert "~3-7d" in text

    def test_estimated_hold_time_gem(self):
        sig = _make_signal(channel="360_GEM", risk_label="LOW")
        text = TelegramBot.format_signal(sig)
        assert "~2-4w" in text

    def test_risk_reward_ratio(self):
        """R:R is computed from SL distance and TP1 distance."""
        sig = _make_signal(entry=100.0, stop_loss=90.0, tp1=120.0)
        text = TelegramBot.format_signal(sig)
        # TP1 dist = 20, SL dist = 10 → R:R 1:2.0
        assert "1:2.0" in text

    def test_risk_and_quality_labels_shown(self):
        sig = _make_signal(risk_label="LOW", quality_tier="PREMIUM")
        text = TelegramBot.format_signal(sig)
        assert "LOW" in text
        assert "PREMIUM" in text

    def test_narrative_from_liquidity_and_invalidation(self):
        sig = _make_signal(
            liquidity_info="BTC sweep at 67200",
            invalidation_summary="bullish FVG",
        )
        text = TelegramBot.format_signal(sig)
        assert "BTC sweep at 67200" in text
        assert "bullish FVG" in text

    def test_channel_display_name_no_360_prefix(self):
        """Channel name in header should use short display name (no 360_ prefix)."""
        for channel, display in [
            ("360_SCALP", "SCALP"),
            ("360_SCALP_FVG", "SCALP FVG"),
            ("360_SWING", "SWING"),
            ("360_SPOT", "SPOT"),
            ("360_GEM", "GEM"),
        ]:
            sig = _make_signal(channel=channel, risk_label="LOW")
            text = TelegramBot.format_signal(sig)
            assert display in text
            # Should NOT include the 360_ prefix as-is
            assert "360_" not in text.split("\n")[0]

    def test_short_direction_emoji_not_in_new_format(self):
        """New format does not use 🚀/⬇️ direction emojis; direction is plain text."""
        sig = _make_signal(direction=Direction.SHORT, stop_loss=67500.00, tp1=67000.00)
        text = TelegramBot.format_signal(sig)
        assert "SHORT" in text
        # The new format does not include ⬇️
        assert "⬇️" not in text


class TestLegacyFormatSignal:
    def test_legacy_contains_channel_name_with_prefix(self):
        sig = _make_signal()
        text = TelegramBot.format_signal_legacy(sig)
        assert r"360\_SCALP" in text

    def test_legacy_contains_confidence_pct(self):
        sig = _make_signal(confidence=87)
        text = TelegramBot.format_signal_legacy(sig)
        assert "87%" in text

    def test_legacy_contains_direction_emoji(self):
        sig = _make_signal(direction=Direction.SHORT, stop_loss=67500.00, tp1=67000.00)
        text = TelegramBot.format_signal_legacy(sig)
        assert "⬇️" in text

    def test_legacy_contains_ai_sentiment(self):
        sig = _make_signal()
        sig.ai_sentiment_label = "Positive"
        sig.ai_sentiment_summary = "Whale Activity"
        text = TelegramBot.format_signal_legacy(sig)
        assert "AI Sentiment" in text
        assert "Whale Activity" in text

    def test_legacy_contains_market_phase(self):
        sig = _make_signal()
        sig.market_phase = "STRONG_TREND"
        text = TelegramBot.format_signal_legacy(sig)
        assert "Market Phase" in text
        assert "STRONG\\_TREND" in text  # underscores are escaped in Markdown

    def test_legacy_and_new_both_callable(self):
        """Both format methods should produce non-empty strings."""
        sig = _make_signal()
        new_text = TelegramBot.format_signal(sig)
        old_text = TelegramBot.format_signal_legacy(sig)
        assert len(new_text) > 50
        assert len(old_text) > 50

    def test_legacy_watchlist_routing(self):
        """WATCHLIST signals route to format_watchlist_signal in legacy too."""
        sig = _make_signal()
        sig.signal_tier = "WATCHLIST"
        text = TelegramBot.format_signal_legacy(sig)
        assert "WATCHLIST" in text


class TestEstimatedHold:
    @pytest.mark.parametrize("channel,expected", [
        ("360_SCALP", "~1-2h"),
        ("360_SCALP_FVG", "~1-2h"),
        ("360_SCALP_CVD", "~1-2h"),
        ("360_SCALP_VWAP", "~1-2h"),
        ("360_SCALP_OBI", "~1-2h"),
        ("360_SWING", "~1-2d"),
        ("360_SPOT", "~3-7d"),
        ("360_GEM", "~2-4w"),
    ])
    def test_hold_time_per_channel(self, channel, expected):
        assert TelegramBot._ESTIMATED_HOLD.get(channel) == expected


class TestChannelDisplayName:
    @pytest.mark.parametrize("channel,expected", [
        ("360_SCALP", "SCALP"),
        ("360_SCALP_FVG", "SCALP FVG"),
        ("360_SCALP_CVD", "SCALP CVD"),
        ("360_SCALP_VWAP", "SCALP VWAP"),
        ("360_SCALP_OBI", "SCALP OBI"),
        ("360_SWING", "SWING"),
        ("360_SPOT", "SPOT"),
        ("360_GEM", "GEM"),
    ])
    def test_display_name(self, channel, expected):
        assert TelegramBot._CHANNEL_DISPLAY_NAME.get(channel) == expected
