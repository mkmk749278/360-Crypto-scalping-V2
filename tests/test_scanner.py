"""Tests for Scanner – cooldown logic and regime-aware gating."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.channels.base import Signal
from src.regime import MarketRegime
from src.scanner import Scanner, _RANGING_ADX_SUPPRESS_THRESHOLD
from src.smc import Direction
from src.utils import utcnow


def _make_scanner(**kwargs) -> Scanner:
    """Create a minimal Scanner instance with mocked dependencies."""
    defaults = dict(
        pair_mgr=MagicMock(),
        data_store=MagicMock(),
        channels=[],
        smc_detector=MagicMock(),
        regime_detector=MagicMock(),
        predictive=MagicMock(),
        exchange_mgr=MagicMock(),
        spot_client=None,
        telemetry=MagicMock(),
        signal_queue=MagicMock(),
        router=MagicMock(),
    )
    defaults.update(kwargs)
    return Scanner(**defaults)


def _candles(length: int = 40) -> dict:
    base = [float(i + 1) for i in range(length)]
    return {
        "high": base,
        "low": [max(v - 0.5, 0.1) for v in base],
        "close": base,
        "volume": [100.0 for _ in base],
    }


class TestScannerCooldown:
    def test_no_cooldown_initially(self):
        scanner = _make_scanner()
        assert scanner._is_in_cooldown("BTCUSDT", "360_SCALP") is False

    def test_cooldown_active_after_set(self):
        scanner = _make_scanner()
        scanner._set_cooldown("BTCUSDT", "360_SCALP")
        assert scanner._is_in_cooldown("BTCUSDT", "360_SCALP") is True

    def test_cooldown_expires(self):
        scanner = _make_scanner()
        # Manually set an already-expired cooldown
        scanner._cooldown_until[("BTCUSDT", "360_SCALP")] = (
            time.monotonic() - 1  # 1 second in the past
        )
        assert scanner._is_in_cooldown("BTCUSDT", "360_SCALP") is False

    def test_cooldown_expires_cleans_up(self):
        scanner = _make_scanner()
        scanner._cooldown_until[("BTCUSDT", "360_SCALP")] = (
            time.monotonic() - 1
        )
        scanner._is_in_cooldown("BTCUSDT", "360_SCALP")
        assert ("BTCUSDT", "360_SCALP") not in scanner._cooldown_until

    def test_cooldown_separate_per_channel(self):
        scanner = _make_scanner()
        scanner._set_cooldown("BTCUSDT", "360_SCALP")
        assert scanner._is_in_cooldown("BTCUSDT", "360_SCALP") is True
        assert scanner._is_in_cooldown("BTCUSDT", "360_SWING") is False

    def test_cooldown_separate_per_symbol(self):
        scanner = _make_scanner()
        scanner._set_cooldown("BTCUSDT", "360_SCALP")
        assert scanner._is_in_cooldown("ETHUSDT", "360_SCALP") is False

    def test_cooldown_duration_from_config(self):
        from config import SIGNAL_SCAN_COOLDOWN_SECONDS
        scanner = _make_scanner()
        scanner._set_cooldown("BTCUSDT", "360_SCALP")
        expiry = scanner._cooldown_until[("BTCUSDT", "360_SCALP")]
        expected_duration = SIGNAL_SCAN_COOLDOWN_SECONDS.get("360_SCALP", 300)
        actual_duration = expiry - time.monotonic()
        assert abs(actual_duration - expected_duration) < 2  # within 2 seconds


class TestScannerCircuitBreaker:
    def test_circuit_breaker_not_set_by_default(self):
        scanner = _make_scanner()
        assert scanner.circuit_breaker is None

    @pytest.mark.asyncio
    async def test_scan_loop_skips_when_tripped(self):
        """Scan loop should skip evaluation when circuit breaker is tripped."""
        scanner = _make_scanner()
        cb = MagicMock()
        cb.is_tripped.return_value = True
        scanner.circuit_breaker = cb

        # Patch asyncio.sleep to avoid infinite loop
        sleep_count = 0

        async def mock_sleep(n):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError

        with patch("src.scanner.asyncio.sleep", side_effect=mock_sleep):
            try:
                await scanner.scan_loop()
            except asyncio.CancelledError:
                pass

        # pair_mgr should NOT have been accessed (scan was skipped)
        scanner.pair_mgr.pairs.items.assert_not_called()


class TestScannerRegimeGating:
    def test_ranging_adx_threshold_constant(self):
        assert _RANGING_ADX_SUPPRESS_THRESHOLD == 15.0

    def test_scanner_has_paused_channels_attribute(self):
        scanner = _make_scanner()
        assert isinstance(scanner.paused_channels, set)

    def test_scanner_has_confidence_overrides_attribute(self):
        scanner = _make_scanner()
        assert isinstance(scanner.confidence_overrides, dict)

    def test_scanner_paused_channels_shared_with_external_set(self):
        shared = set()
        scanner = _make_scanner()
        scanner.paused_channels = shared
        shared.add("360_SCALP")
        assert "360_SCALP" in scanner.paused_channels


class TestScannerAttributes:
    def test_force_scan_starts_false(self):
        scanner = _make_scanner()
        assert scanner.force_scan is False

    def test_force_scan_can_be_set(self):
        scanner = _make_scanner()
        scanner.force_scan = True
        assert scanner.force_scan is True

    def test_ws_spot_starts_none(self):
        scanner = _make_scanner()
        assert scanner.ws_spot is None

    def test_ws_futures_starts_none(self):
        scanner = _make_scanner()
        assert scanner.ws_futures is None


class TestScannerConfidencePipeline:
    @pytest.mark.asyncio
    async def test_adjustments_persist_and_threshold_applied_last(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_RANGE", min_confidence=60.0)
        channel.evaluate.return_value = Signal(
            channel="360_RANGE",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=100.0,
            stop_loss=95.0,
            tp1=105.0,
            tp2=110.0,
            confidence=10.0,
            signal_id="SIG-001",
            timestamp=utcnow(),
        )

        smc_result = SimpleNamespace(
            sweeps=[],
            fvg=[],
            mss=None,
            as_dict=lambda: {"sweeps": [], "fvg": [], "mss": None},
        )
        regime_result = SimpleNamespace(regime=MarketRegime.RANGING)
        predictive = MagicMock()
        predictive.predict = AsyncMock(
            return_value=SimpleNamespace(
                confidence_adjustment=7.0,
                predicted_direction="UP",
                suggested_tp_adjustment=1.0,
                suggested_sl_adjustment=1.0,
            )
        )
        predictive.adjust_tp_sl = MagicMock()
        predictive.update_confidence = MagicMock(
            side_effect=lambda signal, _prediction: setattr(
                signal, "confidence", signal.confidence + 7.0
            )
        )
        openai_evaluator = MagicMock()
        openai_evaluator.enabled = True
        openai_evaluator.evaluate = AsyncMock(
            return_value=SimpleNamespace(
                adjustment=3.0,
                recommended=True,
                reasoning="aligned",
            )
        )
        signal_queue = MagicMock()
        signal_queue.put_nowait.return_value = True
        pair_mgr = MagicMock()
        pair_mgr.has_enough_history.return_value = True
        data_store = MagicMock()
        data_store.get_candles.side_effect = lambda _symbol, _interval: _candles()
        data_store.ticks = {"BTCUSDT": []}

        scanner = _make_scanner(
            pair_mgr=pair_mgr,
            data_store=data_store,
            channels=[channel],
            smc_detector=MagicMock(detect=MagicMock(return_value=smc_result)),
            regime_detector=MagicMock(classify=MagicMock(return_value=regime_result)),
            predictive=predictive,
            exchange_mgr=MagicMock(
                verify_signal_cross_exchange=AsyncMock(return_value=True)
            ),
            signal_queue=signal_queue,
            router=MagicMock(active_signals={}),
            openai_evaluator=openai_evaluator,
            onchain_client=MagicMock(get_exchange_flow=AsyncMock(return_value=None)),
        )

        with patch("src.scanner.get_ai_insight", AsyncMock(return_value=SimpleNamespace(label="Neutral", summary="", score=0.0))), \
             patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=50.0, blocked=False)):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        queued_signal = signal_queue.put_nowait.call_args[0][0]
        assert queued_signal.confidence == 65.0
        openai_evaluator.evaluate.assert_awaited_once()
        assert openai_evaluator.evaluate.await_args.kwargs["confidence_before"] == 62.0

    @pytest.mark.asyncio
    async def test_openai_skip_prevents_enqueue(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = Signal(
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=100.0,
            stop_loss=95.0,
            tp1=105.0,
            tp2=110.0,
            confidence=10.0,
            signal_id="SIG-002",
            timestamp=utcnow(),
        )
        smc_result = SimpleNamespace(
            sweeps=[],
            fvg=[],
            mss=None,
            as_dict=lambda: {"sweeps": [], "fvg": [], "mss": None},
        )
        scanner = _make_scanner(
            pair_mgr=MagicMock(has_enough_history=MagicMock(return_value=True)),
            data_store=MagicMock(
                get_candles=MagicMock(side_effect=lambda _symbol, _interval: _candles()),
                ticks={"BTCUSDT": []},
            ),
            channels=[channel],
            smc_detector=MagicMock(detect=MagicMock(return_value=smc_result)),
            regime_detector=MagicMock(
                classify=MagicMock(return_value=SimpleNamespace(regime=MarketRegime.TRENDING_UP))
            ),
            predictive=MagicMock(
                predict=AsyncMock(return_value=SimpleNamespace(
                    confidence_adjustment=0.0,
                    predicted_direction="NEUTRAL",
                    suggested_tp_adjustment=1.0,
                    suggested_sl_adjustment=1.0,
                )),
                adjust_tp_sl=MagicMock(),
                update_confidence=MagicMock(),
            ),
            exchange_mgr=MagicMock(
                verify_signal_cross_exchange=AsyncMock(return_value=True)
            ),
            signal_queue=MagicMock(put_nowait=MagicMock(return_value=True)),
            router=MagicMock(active_signals={}),
            openai_evaluator=MagicMock(
                enabled=True,
                evaluate=AsyncMock(return_value=SimpleNamespace(
                    adjustment=0.0,
                    recommended=False,
                    reasoning="reject",
                )),
            ),
        )

        with patch("src.scanner.get_ai_insight", AsyncMock(return_value=SimpleNamespace(label="Neutral", summary="", score=0.0))), \
             patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=55.0, blocked=False)):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        scanner.signal_queue.put_nowait.assert_not_called()
