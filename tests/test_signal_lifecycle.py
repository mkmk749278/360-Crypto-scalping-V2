"""Tests for src.signal_lifecycle — Signal Lifecycle Monitor."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.channels.base import Signal
from src.signal_lifecycle import (
    SignalLifecycleMonitor,
    _compute_ema,
    _compute_rsi,
)
from src.smc import Direction
from src.utils import utcnow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    channel: str = "360_SPOT",
    symbol: str = "ZETAUSDT",
    direction: Direction = Direction.LONG,
    entry: float = 1.0,
    confidence: float = 80.0,
    status: str = "ACTIVE",
) -> Signal:
    sig = Signal(
        channel=channel,
        symbol=symbol,
        direction=direction,
        entry=entry,
        stop_loss=entry * 0.95,
        tp1=entry * 1.10,
        tp2=entry * 1.20,
        confidence=confidence,
        signal_id=f"{symbol}-TEST",
        timestamp=utcnow(),
    )
    sig.pre_ai_confidence = confidence
    sig.current_price = entry
    return sig


def _make_candles(
    n: int = 30,
    trend: str = "up",  # "up", "down", "flat"
    base: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Build a minimal OHLCV dict for testing."""
    closes: List[float] = []
    for i in range(n):
        if trend == "up":
            closes.append(base + i * 0.01)
        elif trend == "down":
            closes.append(base - i * 0.01)
        else:
            closes.append(base + (i % 3) * 0.001)
    arr = np.array(closes)
    # lows slightly below close, highs slightly above
    return {
        "open":   arr - 0.002,
        "high":   arr + 0.005,
        "low":    arr - 0.005,
        "close":  arr,
        "volume": np.ones(n) * 1000.0,
    }


class _FakeRouter:
    """Minimal router stub exposing active_signals and update_signal."""

    def __init__(self, signals: Optional[Dict[str, Signal]] = None) -> None:
        self._signals: Dict[str, Signal] = signals or {}
        self.updates: List[Dict[str, Any]] = []

    @property
    def active_signals(self) -> Dict[str, Signal]:
        return dict(self._signals)

    def update_signal(self, signal_id: str, **kwargs: Any) -> None:
        self.updates.append({"signal_id": signal_id, **kwargs})
        sig = self._signals.get(signal_id)
        if sig:
            for k, v in kwargs.items():
                if hasattr(sig, k):
                    setattr(sig, k, v)


class _FakeDataStore:
    """Minimal HistoricalDataStore stub."""

    def __init__(self, candles: Optional[Dict] = None) -> None:
        self._candles = candles or {}

    def get_candles(
        self, symbol: str, interval: str
    ) -> Optional[Dict[str, np.ndarray]]:
        return self._candles.get(symbol, {}).get(interval)


class _FakeRegimeDetector:
    """Always returns a fixed regime."""

    def __init__(self, regime_value: str = "TRENDING_UP") -> None:
        self._regime = regime_value

    def classify(self, indicators: dict, **kwargs: Any) -> Any:
        result = MagicMock()
        result.regime = MagicMock()
        result.regime.value = self._regime
        return result


def _make_monitor(
    signal: Signal,
    candles: Optional[Dict] = None,
    regime: str = "TRENDING_UP",
) -> SignalLifecycleMonitor:
    sent: List = []

    async def mock_send(chat_id: str, text: str) -> bool:
        sent.append((chat_id, text))
        return True

    router = _FakeRouter({signal.signal_id: signal})
    data_store = _FakeDataStore(
        candles={signal.symbol: {"4h": candles}} if candles else {}
    )
    regime_detector = _FakeRegimeDetector(regime)
    monitor = SignalLifecycleMonitor(
        router=router,
        data_store=data_store,
        regime_detector=regime_detector,
        send_telegram=mock_send,
    )
    monitor._sent = sent
    monitor._router_stub = router
    return monitor


# ---------------------------------------------------------------------------
# Unit tests: pure helpers
# ---------------------------------------------------------------------------


class TestComputeEma:
    def test_basic_uptrend(self):
        prices = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        ema = _compute_ema(prices, period=5)
        assert ema is not None
        assert ema > 1.0

    def test_returns_none_when_insufficient_data(self):
        assert _compute_ema([1.0, 2.0], period=5) is None

    def test_single_price_period(self):
        result = _compute_ema([3.0], period=1)
        assert result == pytest.approx(3.0)


class TestComputeRsi:
    def test_all_gains_returns_high_rsi(self):
        closes = [float(i) for i in range(20)]
        rsi = _compute_rsi(closes, period=14)
        assert rsi is not None
        assert rsi > 70.0

    def test_all_losses_returns_low_rsi(self):
        closes = [float(20 - i) for i in range(20)]
        rsi = _compute_rsi(closes, period=14)
        assert rsi is not None
        assert rsi < 30.0

    def test_returns_none_when_insufficient_data(self):
        assert _compute_rsi([1.0, 2.0, 3.0], period=14) is None

    def test_no_losses_returns_100(self):
        closes = [1.0 + i * 0.1 for i in range(16)]
        rsi = _compute_rsi(closes, period=14)
        assert rsi == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Unit tests: assessment helpers
# ---------------------------------------------------------------------------


class TestAssessRegimeChange:
    def test_unchanged_regime_returns_green(self):
        sig = _make_signal()
        sig.entry_regime = "TRENDING_UP"
        candles = _make_candles(n=30, trend="up")
        monitor = _make_monitor(sig, candles, regime="TRENDING_UP")
        result = monitor._assess_regime_change(sig, candles)
        assert result is not None
        assert result.startswith("🟢")
        assert "unchanged" in result

    def test_regime_flip_to_down_on_long_returns_red(self):
        sig = _make_signal(direction=Direction.LONG)
        sig.entry_regime = "TRENDING_UP"
        candles = _make_candles(n=30, trend="down")
        monitor = _make_monitor(sig, candles, regime="TRENDING_DOWN")
        result = monitor._assess_regime_change(sig, candles)
        assert result is not None
        assert result.startswith("🔴")

    def test_regime_shift_to_ranging_on_long_returns_yellow(self):
        sig = _make_signal(direction=Direction.LONG)
        sig.entry_regime = "TRENDING_UP"
        candles = _make_candles(n=30, trend="flat")
        monitor = _make_monitor(sig, candles, regime="RANGING")
        result = monitor._assess_regime_change(sig, candles)
        assert result is not None
        assert result.startswith("🟡")

    def test_missing_candles_returns_none(self):
        sig = _make_signal()
        monitor = _make_monitor(sig, candles=None, regime="TRENDING_UP")
        result = monitor._assess_regime_change(sig, None)
        assert result is None

    def test_too_few_candles_returns_none(self):
        sig = _make_signal()
        candles = _make_candles(n=5, trend="up")
        monitor = _make_monitor(sig, candles, regime="TRENDING_UP")
        result = monitor._assess_regime_change(sig, candles)
        assert result is None


class TestAssessMomentum:
    def test_strong_uptrend_long_returns_green(self):
        sig = _make_signal(direction=Direction.LONG)
        candles = _make_candles(n=30, trend="up")
        monitor = _make_monitor(sig, candles, regime="TRENDING_UP")
        result = monitor._assess_momentum(sig, candles)
        assert result is not None
        assert result.startswith("🟢")

    def test_downtrend_long_returns_red(self):
        sig = _make_signal(direction=Direction.LONG)
        candles = _make_candles(n=30, trend="down", base=2.0)
        monitor = _make_monitor(sig, candles, regime="TRENDING_DOWN")
        result = monitor._assess_momentum(sig, candles)
        assert result is not None
        assert result.startswith("🔴")

    def test_downtrend_short_returns_green(self):
        sig = _make_signal(direction=Direction.SHORT)
        candles = _make_candles(n=30, trend="down", base=2.0)
        monitor = _make_monitor(sig, candles, regime="TRENDING_DOWN")
        result = monitor._assess_momentum(sig, candles)
        assert result is not None
        assert result.startswith("🟢")

    def test_missing_candles_returns_none(self):
        sig = _make_signal()
        monitor = _make_monitor(sig, candles=None)
        result = monitor._assess_momentum(sig, None)
        assert result is None


class TestAssessStructure:
    def test_intact_structure_long(self):
        sig = _make_signal(direction=Direction.LONG)
        # Uptrending candles — no lower low
        candles = _make_candles(n=15, trend="up", base=1.0)
        monitor = _make_monitor(sig, candles)
        result = monitor._assess_structure(sig, candles)
        assert result is not None
        assert result.startswith("🟢")
        assert "Intact" in result

    def test_bos_against_long_returns_red(self):
        sig = _make_signal(direction=Direction.LONG)
        # Build candles where the last candle prints a new lower low
        candles = _make_candles(n=15, trend="flat", base=1.0)
        # Force a break-of-structure: current low below all previous lows
        candles["low"][-1] = float(candles["low"][:-1].min()) - 0.1
        monitor = _make_monitor(sig, candles)
        result = monitor._assess_structure(sig, candles)
        assert result is not None
        assert result.startswith("🔴")

    def test_too_few_candles_returns_none(self):
        sig = _make_signal(direction=Direction.LONG)
        candles = _make_candles(n=5)
        monitor = _make_monitor(sig, candles)
        result = monitor._assess_structure(sig, candles)
        assert result is None


class TestAssessConfidenceDecay:
    def test_no_decay_returns_none(self):
        sig = _make_signal(confidence=80.0)
        sig.pre_ai_confidence = 80.0
        monitor = _make_monitor(sig)
        assert monitor._assess_confidence_decay(sig) is None

    def test_small_drop_below_yellow_threshold_returns_none(self):
        sig = _make_signal(confidence=68.0)
        sig.pre_ai_confidence = 80.0  # drop = 12, below YELLOW=15
        monitor = _make_monitor(sig)
        assert monitor._assess_confidence_decay(sig) is None

    def test_yellow_threshold_drop_returns_yellow(self):
        sig = _make_signal(confidence=63.0)
        sig.pre_ai_confidence = 80.0  # drop = 17, >= YELLOW=15
        monitor = _make_monitor(sig)
        result = monitor._assess_confidence_decay(sig)
        assert result is not None
        assert result.startswith("🟡")

    def test_red_threshold_drop_returns_red(self):
        sig = _make_signal(confidence=50.0)
        sig.pre_ai_confidence = 80.0  # drop = 30, >= RED=25
        monitor = _make_monitor(sig)
        result = monitor._assess_confidence_decay(sig)
        assert result is not None
        assert result.startswith("🔴")

    def test_zero_entry_confidence_returns_none(self):
        sig = _make_signal(confidence=0.0)
        sig.pre_ai_confidence = 0.0
        monitor = _make_monitor(sig)
        assert monitor._assess_confidence_decay(sig) is None


class TestAssessTpProgress:
    def test_no_progress_shows_zero_percent(self):
        sig = _make_signal(entry=1.0)
        sig.tp1 = 1.10
        sig.current_price = 1.0
        monitor = _make_monitor(sig)
        result = monitor._assess_tp_progress(sig, current_price=1.0)
        assert result is not None
        assert "0%" in result

    def test_halfway_shows_50_percent(self):
        sig = _make_signal(entry=1.0)
        sig.tp1 = 1.10
        sig.current_price = 1.05
        monitor = _make_monitor(sig)
        result = monitor._assess_tp_progress(sig, current_price=1.05)
        assert result is not None
        assert "50%" in result

    def test_tp1_hit_shows_celebration(self):
        sig = _make_signal(entry=1.0)
        sig.tp1 = 1.10
        sig.tp2 = 1.20
        sig.best_tp_hit = 1
        sig.current_price = 1.10
        monitor = _make_monitor(sig)
        result = monitor._assess_tp_progress(sig, current_price=1.10)
        assert result is not None
        assert "TP1 ✅" in result


class TestShouldRecommendClose:
    def test_single_red_flag_does_not_recommend_close(self):
        sig = _make_signal()
        monitor = _make_monitor(sig)
        assessments = ["🔴 Regime: flipped TRENDING_UP → TRENDING_DOWN"]
        should_close, reason = monitor._should_recommend_close(sig, assessments)
        assert not should_close

    def test_two_reds_from_same_category_does_not_recommend_close(self):
        sig = _make_signal()
        monitor = _make_monitor(sig)
        # Both are "Regime" category — only 1 distinct category
        assessments = [
            "🔴 Regime: flipped TRENDING_UP → TRENDING_DOWN",
            "🔴 Regime: something else",
        ]
        should_close, reason = monitor._should_recommend_close(sig, assessments)
        assert not should_close

    def test_two_reds_from_different_categories_recommend_close(self):
        sig = _make_signal()
        monitor = _make_monitor(sig)
        assessments = [
            "🔴 Regime: flipped TRENDING_UP → TRENDING_DOWN",
            "🔴 Momentum: Lost — EMA slope negative",
        ]
        should_close, reason = monitor._should_recommend_close(sig, assessments)
        assert should_close
        assert reason  # should contain explanation

    def test_full_red_sweep_recommends_close_with_all_reasons(self):
        sig = _make_signal()
        monitor = _make_monitor(sig)
        assessments = [
            "🔴 Regime: flipped TRENDING_UP → TRENDING_DOWN",
            "🔴 Momentum: Lost — EMA slope negative, RSI < 40",
            "🔴 Structure: BROKEN — lower low printed",
            "🔴 Confidence: 80 → 45 (dropped 35pts)",
        ]
        should_close, reason = monitor._should_recommend_close(sig, assessments)
        assert should_close
        assert "regime reversal" in reason.lower() or "market structure" in reason.lower()


# ---------------------------------------------------------------------------
# Integration tests: _check_signal and _tick
# ---------------------------------------------------------------------------


class TestCheckSignal:
    @pytest.mark.asyncio
    async def test_check_signal_posts_update_and_updates_state(self):
        sig = _make_signal(channel="360_SPOT")
        sig.entry_regime = "TRENDING_UP"
        candles = _make_candles(n=30, trend="up")
        monitor = _make_monitor(sig, candles, regime="TRENDING_UP")
        # Patch CHANNEL_TELEGRAM_MAP so a message can be "sent"
        import src.signal_lifecycle as lc_module
        orig_map = lc_module.CHANNEL_TELEGRAM_MAP
        lc_module.CHANNEL_TELEGRAM_MAP = {"360_SPOT": "spot-chat-id"}
        try:
            await monitor._check_signal(sig)
        finally:
            lc_module.CHANNEL_TELEGRAM_MAP = orig_map

        # A Telegram message must have been sent
        assert len(monitor._sent) == 1
        _, text = monitor._sent[0]
        assert "ZETAUSDT" in text

        # Signal state must be updated
        assert any(u["signal_id"] == sig.signal_id for u in monitor._router_stub.updates)
        updated = {u["signal_id"]: u for u in monitor._router_stub.updates}
        assert "last_lifecycle_check" in updated[sig.signal_id]
        assert "lifecycle_alert_level" in updated[sig.signal_id]

    @pytest.mark.asyncio
    async def test_check_signal_no_chat_id_skips_send(self):
        sig = _make_signal(channel="360_SPOT")
        candles = _make_candles(n=30, trend="up")
        monitor = _make_monitor(sig, candles, regime="TRENDING_UP")
        import src.signal_lifecycle as lc_module
        orig_map = lc_module.CHANNEL_TELEGRAM_MAP
        lc_module.CHANNEL_TELEGRAM_MAP = {}  # no channel configured
        try:
            await monitor._check_signal(sig)
        finally:
            lc_module.CHANNEL_TELEGRAM_MAP = orig_map

        # No message sent when channel not configured
        assert len(monitor._sent) == 0


class TestIsDue:
    def test_new_signal_not_due_immediately(self):
        """A brand-new signal (last_lifecycle_check=None) must not fire immediately."""
        sig = _make_signal(channel="360_SPOT")
        sig.last_lifecycle_check = None
        # timestamp defaults to utcnow() in _make_signal — elapsed ≈ 0s
        monitor = _make_monitor(sig)
        assert not monitor._is_due(sig)

    def test_old_signal_with_no_check_is_due(self):
        """A signal created more than one interval ago with no check should fire."""
        sig = _make_signal(channel="360_SPOT")
        sig.last_lifecycle_check = None
        # Back-date the creation timestamp by 7h (SPOT interval is 6h)
        sig.timestamp = utcnow() - timedelta(hours=7)
        monitor = _make_monitor(sig)
        assert monitor._is_due(sig)

    def test_recently_checked_is_not_due(self):
        sig = _make_signal(channel="360_SPOT")
        sig.last_lifecycle_check = utcnow()
        monitor = _make_monitor(sig)
        assert not monitor._is_due(sig)

    def test_overdue_signal_is_due(self):
        sig = _make_signal(channel="360_SPOT")
        # SPOT interval is 21600s (6h) — set last check to 7 hours ago
        sig.last_lifecycle_check = utcnow() - timedelta(hours=7)
        monitor = _make_monitor(sig)
        assert monitor._is_due(sig)

    def test_scalp_channel_never_due(self):
        sig = _make_signal(channel="360_SCALP")
        sig.last_lifecycle_check = None
        monitor = _make_monitor(sig)
        # SCALP is not in LIFECYCLE_CHECK_INTERVAL
        assert not monitor._is_due(sig)


class TestFormatUpdateMessage:
    def test_healthy_update_contains_hold(self):
        sig = _make_signal(channel="360_SPOT", entry=1.0)
        monitor = _make_monitor(sig)
        msg = monitor._format_update_message(
            signal=sig,
            assessments=["🟢 Regime: TRENDING_UP (unchanged)", "🟢 Momentum: Strong"],
            current_price=1.05,
            alert_level="GREEN",
            should_close=False,
            close_reason="",
        )
        assert "Hold" in msg
        assert "ZETAUSDT" in msg
        assert "+5.0%" in msg

    def test_close_recommendation_shows_close_signal(self):
        sig = _make_signal(channel="360_SPOT", entry=1.0)
        monitor = _make_monitor(sig)
        msg = monitor._format_update_message(
            signal=sig,
            assessments=["🔴 Regime: flipped", "🔴 Momentum: Lost"],
            current_price=0.90,
            alert_level="RED",
            should_close=True,
            close_reason="regime reversal + momentum lost",
        )
        assert "CLOSE RECOMMENDED" in msg
        assert "thesis invalidated" in msg

    def test_regime_underscores_escaped_in_message(self):
        """Regime names like TRENDING_UP must have underscores escaped for Telegram Markdown."""
        sig = _make_signal(channel="360_SPOT", entry=1.0)
        monitor = _make_monitor(sig)
        msg = monitor._format_update_message(
            signal=sig,
            assessments=["🔴 Regime: flipped TRENDING_UP → TRENDING_DOWN"],
            current_price=0.95,
            alert_level="RED",
            should_close=False,
            close_reason="",
        )
        # Underscores in regime names must be escaped for Markdown safety
        assert "TRENDING\\_UP" in msg
        assert "TRENDING\\_DOWN" in msg


# ---------------------------------------------------------------------------
# Lifecycle config test
# ---------------------------------------------------------------------------


class TestLifecycleConfig:
    def test_spot_gem_caps_are_unlimited(self):
        from config import MAX_CONCURRENT_SIGNALS_PER_CHANNEL
        assert MAX_CONCURRENT_SIGNALS_PER_CHANNEL["360_SPOT"] >= 999
        assert MAX_CONCURRENT_SIGNALS_PER_CHANNEL["360_GEM"] >= 999

    def test_swing_cap_raised(self):
        from config import MAX_CONCURRENT_SIGNALS_PER_CHANNEL
        assert MAX_CONCURRENT_SIGNALS_PER_CHANNEL["360_SWING"] >= 10

    def test_scalp_caps_unchanged(self):
        from config import MAX_CONCURRENT_SIGNALS_PER_CHANNEL
        assert MAX_CONCURRENT_SIGNALS_PER_CHANNEL["360_SCALP"] == 5
        assert MAX_CONCURRENT_SIGNALS_PER_CHANNEL["360_SCALP_FVG"] == 3

    def test_lifecycle_check_intervals_present(self):
        from config import LIFECYCLE_CHECK_INTERVAL
        assert "360_SWING" in LIFECYCLE_CHECK_INTERVAL
        assert "360_SPOT" in LIFECYCLE_CHECK_INTERVAL
        assert "360_GEM" in LIFECYCLE_CHECK_INTERVAL
        # SCALP must not be in lifecycle monitoring
        assert "360_SCALP" not in LIFECYCLE_CHECK_INTERVAL

    def test_lifecycle_intervals_values(self):
        from config import LIFECYCLE_CHECK_INTERVAL
        assert LIFECYCLE_CHECK_INTERVAL["360_SWING"] == 14400   # 4 hours
        assert LIFECYCLE_CHECK_INTERVAL["360_SPOT"] == 21600    # 6 hours
        assert LIFECYCLE_CHECK_INTERVAL["360_GEM"] == 43200     # 12 hours

    def test_confidence_thresholds_present(self):
        from config import LIFECYCLE_CONFIDENCE_DROP_YELLOW, LIFECYCLE_CONFIDENCE_DROP_RED
        assert LIFECYCLE_CONFIDENCE_DROP_YELLOW == pytest.approx(15.0)
        assert LIFECYCLE_CONFIDENCE_DROP_RED == pytest.approx(25.0)
        assert LIFECYCLE_CONFIDENCE_DROP_RED > LIFECYCLE_CONFIDENCE_DROP_YELLOW
