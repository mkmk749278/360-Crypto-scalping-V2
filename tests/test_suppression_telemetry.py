"""Tests for src.suppression_telemetry – SuppressionTracker."""

from __future__ import annotations

import time

import pytest

from src.suppression_telemetry import (
    REASON_CLUSTER,
    REASON_CONFIDENCE,
    REASON_OI_INVALIDATION,
    REASON_QUIET_REGIME,
    REASON_SPREAD_GATE,
    REASON_STAT_FILTER,
    REASON_VOLUME_GATE,
    SuppressionEvent,
    SuppressionTracker,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _event(
    symbol: str = "BTCUSDT",
    channel: str = "360_SCALP",
    reason: str = REASON_QUIET_REGIME,
    regime: str = "QUIET",
    would_be_confidence: float = 65.0,
    timestamp: float | None = None,
) -> SuppressionEvent:
    evt = SuppressionEvent(
        symbol=symbol,
        channel=channel,
        reason=reason,
        regime=regime,
        would_be_confidence=would_be_confidence,
    )
    if timestamp is not None:
        object.__setattr__(evt, "timestamp", timestamp)
    return evt


# ---------------------------------------------------------------------------
# Basic record / total_in_window
# ---------------------------------------------------------------------------


class TestBasicRecording:
    def test_empty_tracker_returns_zero(self):
        tracker = SuppressionTracker()
        assert tracker.total_in_window() == 0

    def test_single_event_counted(self):
        tracker = SuppressionTracker()
        tracker.record(_event())
        assert tracker.total_in_window() == 1

    def test_multiple_events_counted(self):
        tracker = SuppressionTracker()
        for _ in range(5):
            tracker.record(_event())
        assert tracker.total_in_window() == 5

    def test_old_events_are_pruned(self):
        tracker = SuppressionTracker(window_seconds=1.0)
        # Manually create events with old timestamps
        old_event = _event(timestamp=time.monotonic() - 10.0)
        recent_event = _event(timestamp=time.monotonic())
        tracker._events.append(old_event)
        tracker._events.append(recent_event)
        # Trigger pruning via total_in_window
        assert tracker.total_in_window() == 1

    def test_all_events_pruned_returns_zero(self):
        tracker = SuppressionTracker(window_seconds=0.001)
        tracker.record(_event())
        time.sleep(0.01)
        assert tracker.total_in_window() == 0


# ---------------------------------------------------------------------------
# summary() – by reason
# ---------------------------------------------------------------------------


class TestSummaryByReason:
    def test_single_reason(self):
        tracker = SuppressionTracker()
        tracker.record(_event(reason=REASON_QUIET_REGIME))
        tracker.record(_event(reason=REASON_QUIET_REGIME))
        summary = tracker.summary()
        assert summary == {REASON_QUIET_REGIME: 2}

    def test_multiple_reasons(self):
        tracker = SuppressionTracker()
        tracker.record(_event(reason=REASON_QUIET_REGIME))
        tracker.record(_event(reason=REASON_SPREAD_GATE))
        tracker.record(_event(reason=REASON_CONFIDENCE))
        tracker.record(_event(reason=REASON_SPREAD_GATE))
        summary = tracker.summary()
        assert summary[REASON_QUIET_REGIME] == 1
        assert summary[REASON_SPREAD_GATE] == 2
        assert summary[REASON_CONFIDENCE] == 1

    def test_empty_returns_empty_dict(self):
        tracker = SuppressionTracker()
        assert tracker.summary() == {}


# ---------------------------------------------------------------------------
# by_channel()
# ---------------------------------------------------------------------------


class TestByChannel:
    def test_groups_by_channel(self):
        tracker = SuppressionTracker()
        tracker.record(_event(channel="360_SCALP"))
        tracker.record(_event(channel="360_SCALP"))
        tracker.record(_event(channel="360_SWING"))
        counts = tracker.by_channel()
        assert counts["360_SCALP"] == 2
        assert counts["360_SWING"] == 1


# ---------------------------------------------------------------------------
# by_symbol()
# ---------------------------------------------------------------------------


class TestBySymbol:
    def test_top_n_most_suppressed(self):
        tracker = SuppressionTracker()
        for _ in range(3):
            tracker.record(_event(symbol="ZECUSDT"))
        for _ in range(5):
            tracker.record(_event(symbol="ADAUSDT"))
        tracker.record(_event(symbol="ETHUSDT"))
        top = tracker.by_symbol(top_n=2)
        assert top[0] == ("ADAUSDT", 5)
        assert top[1] == ("ZECUSDT", 3)

    def test_empty_tracker(self):
        tracker = SuppressionTracker()
        assert tracker.by_symbol() == []


# ---------------------------------------------------------------------------
# recent_events()
# ---------------------------------------------------------------------------


class TestRecentEvents:
    def test_returns_newest_first(self):
        tracker = SuppressionTracker()
        tracker.record(_event(symbol="A"))
        tracker.record(_event(symbol="B"))
        tracker.record(_event(symbol="C"))
        recent = tracker.recent_events(limit=3)
        assert recent[0].symbol == "C"
        assert recent[1].symbol == "B"
        assert recent[2].symbol == "A"

    def test_limit_respected(self):
        tracker = SuppressionTracker()
        for i in range(10):
            tracker.record(_event(symbol=f"SYM{i}"))
        recent = tracker.recent_events(limit=5)
        assert len(recent) == 5


# ---------------------------------------------------------------------------
# format_telegram_digest()
# ---------------------------------------------------------------------------


class TestFormatTelegramDigest:
    def test_empty_digest_total_zero(self):
        tracker = SuppressionTracker()
        digest = tracker.format_telegram_digest()
        assert "Total suppressed: *0*" in digest

    def test_digest_contains_reason_label(self):
        tracker = SuppressionTracker()
        tracker.record(_event(reason=REASON_QUIET_REGIME))
        digest = tracker.format_telegram_digest()
        assert "Quiet regime" in digest

    def test_digest_contains_channel(self):
        tracker = SuppressionTracker()
        tracker.record(_event(channel="360_SCALP", reason=REASON_SPREAD_GATE))
        digest = tracker.format_telegram_digest()
        assert "360_SCALP" in digest

    def test_digest_contains_symbol(self):
        tracker = SuppressionTracker()
        tracker.record(_event(symbol="ZECUSDT"))
        digest = tracker.format_telegram_digest()
        assert "ZECUSDT" in digest

    def test_custom_window_hours_label(self):
        tracker = SuppressionTracker(window_seconds=3600)
        digest = tracker.format_telegram_digest(window_hours=1)
        assert "last 1h" in digest

    def test_all_reason_constants_are_strings(self):
        """All public reason constants must be non-empty strings."""
        reasons = [
            REASON_QUIET_REGIME,
            REASON_SPREAD_GATE,
            REASON_VOLUME_GATE,
            REASON_OI_INVALIDATION,
            REASON_CLUSTER,
            REASON_STAT_FILTER,
            REASON_CONFIDENCE,
        ]
        for r in reasons:
            assert isinstance(r, str)
            assert len(r) > 0
