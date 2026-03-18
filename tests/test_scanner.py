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
from src.signal_quality import (
    ExecutionAssessment,
    RiskAssessment,
    SetupAssessment,
    SetupClass,
)
from src.smc import Direction
from src.utils import utcnow


def _make_scanner(**kwargs) -> Scanner:
    """Create a minimal Scanner instance with mocked dependencies."""
    signal_queue = MagicMock()
    signal_queue.put = AsyncMock(return_value=True)
    router_mock = MagicMock(active_signals={})
    router_mock.cleanup_expired.return_value = 0
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
        signal_queue=signal_queue,
        router=router_mock,
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


def _make_signal(
    *,
    channel: str = "360_SCALP",
    signal_id: str = "SIG-001",
    confidence: float = 10.0,
) -> Signal:
    return Signal(
        channel=channel,
        symbol="BTCUSDT",
        direction=Direction.LONG,
        entry=100.0,
        stop_loss=95.0,
        tp1=105.0,
        tp2=110.0,
        confidence=confidence,
        signal_id=signal_id,
        timestamp=utcnow(),
    )


def _make_scan_ready_scanner(
    *,
    channel: MagicMock,
    signal_queue: MagicMock,
    predictive: MagicMock | None = None,
    openai_evaluator: MagicMock | None = None,
    regime: MarketRegime = MarketRegime.TRENDING_UP,
) -> Scanner:
    smc_result = SimpleNamespace(
        sweeps=[SimpleNamespace(direction=Direction.LONG, sweep_level=95.0)],
        fvg=[],
        mss=SimpleNamespace(direction=Direction.LONG, midpoint=98.0),
        as_dict=lambda: {
            "sweeps": [SimpleNamespace(direction=Direction.LONG, sweep_level=95.0)],
            "fvg": [],
            "mss": SimpleNamespace(direction=Direction.LONG, midpoint=98.0),
        },
    )
    if predictive is None:
        predictive = MagicMock(
            predict=AsyncMock(
                return_value=SimpleNamespace(
                    confidence_adjustment=0.0,
                    predicted_direction="NEUTRAL",
                    suggested_tp_adjustment=1.0,
                    suggested_sl_adjustment=1.0,
                )
            ),
            adjust_tp_sl=MagicMock(),
            update_confidence=MagicMock(),
        )

    return _make_scanner(
        pair_mgr=MagicMock(has_enough_history=MagicMock(return_value=True)),
        data_store=MagicMock(
            get_candles=MagicMock(side_effect=lambda _symbol, _interval: _candles()),
            ticks={"BTCUSDT": []},
        ),
        channels=[channel],
        smc_detector=MagicMock(detect=MagicMock(return_value=smc_result)),
        regime_detector=MagicMock(
            classify=MagicMock(return_value=SimpleNamespace(regime=regime))
        ),
        predictive=predictive,
        exchange_mgr=MagicMock(
            verify_signal_cross_exchange=AsyncMock(return_value=True)
        ),
        spot_client=MagicMock(
            fetch_order_book=AsyncMock(
                return_value={"bids": [["100.0", "1"]], "asks": [["100.01", "1"]]}
            )
        ),
        signal_queue=signal_queue,
        router=MagicMock(active_signals={}, cleanup_expired=MagicMock(return_value=0)),
        openai_evaluator=openai_evaluator,
        onchain_client=MagicMock(get_exchange_flow=AsyncMock(return_value=None)),
    )


def _setup_pass() -> SetupAssessment:
    return SetupAssessment(
        setup_class=SetupClass.BREAKOUT_RETEST,
        thesis="Breakout Retest",
        channel_compatible=True,
        regime_compatible=True,
    )


def _execution_pass() -> ExecutionAssessment:
    return ExecutionAssessment(
        passed=True,
        trigger_confirmed=True,
        extension_ratio=0.6,
        anchor_price=99.0,
        entry_zone="99.0000 – 100.0000",
        execution_note="Retest hold confirmed.",
    )


def _risk_pass() -> RiskAssessment:
    return RiskAssessment(
        passed=True,
        stop_loss=95.0,
        tp1=106.5,
        tp2=111.5,
        tp3=117.0,
        r_multiple=1.3,
        invalidation_summary="Below 96.0000 structure + volatility buffer",
    )


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
    async def test_adjustments_persist_and_final_clamp_applies_last(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_RANGE", min_confidence=60.0)
        channel.evaluate.return_value = _make_signal(channel="360_RANGE", signal_id="SIG-001")

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
        predictive.update_confidence = MagicMock()

        def _update_confidence(signal, _prediction):
            # Base confidence (55) plus the RANGE ranging boost (+5) must be in
            # place before predictive adjustments run.
            assert signal.confidence == 60.0
            signal.confidence += 7.0

        predictive.update_confidence.side_effect = _update_confidence

        openai_evaluator = MagicMock()
        openai_evaluator.enabled = True
        openai_evaluator.evaluate = AsyncMock(
            return_value=SimpleNamespace(
                adjustment=50.0,
                recommended=True,
                reasoning="aligned",
            )
        )
        signal_queue = MagicMock()
        signal_queue.put = AsyncMock(return_value=True)

        scanner = _make_scan_ready_scanner(
            channel=channel,
            signal_queue=signal_queue,
            predictive=predictive,
            openai_evaluator=openai_evaluator,
            regime=MarketRegime.RANGING,
        )

        with patch("src.scanner.get_ai_insight", AsyncMock(return_value=SimpleNamespace(label="Neutral", summary="", score=0.0))), \
             patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=55.0, blocked=False)), \
             patch.object(scanner, "_evaluate_setup", return_value=_setup_pass()), \
             patch.object(scanner, "_evaluate_execution", return_value=_execution_pass()), \
             patch.object(scanner, "_evaluate_risk", return_value=_risk_pass()):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        queued_signal = signal_queue.put.await_args.args[0]
        assert queued_signal.confidence == 100.0
        openai_evaluator.evaluate.assert_awaited_once()
        assert predictive.adjust_tp_sl.called
        assert predictive.update_confidence.called
        assert openai_evaluator.evaluate.await_args.kwargs["confidence_before"] == queued_signal.pre_ai_confidence
        assert queued_signal.post_ai_confidence == 100.0
        assert queued_signal.setup_class == SetupClass.BREAKOUT_RETEST.value

    @pytest.mark.asyncio
    async def test_signals_below_final_min_confidence_are_rejected_after_all_adjustments(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=80.0)
        channel.evaluate.return_value = _make_signal(channel="360_SCALP", signal_id="SIG-LOW")

        predictive = MagicMock(
            predict=AsyncMock(
                return_value=SimpleNamespace(
                    confidence_adjustment=-5.0,
                    predicted_direction="DOWN",
                    suggested_tp_adjustment=1.0,
                    suggested_sl_adjustment=1.0,
                )
            ),
            adjust_tp_sl=MagicMock(),
            update_confidence=MagicMock(
                side_effect=lambda signal, _prediction: setattr(
                    signal, "confidence", signal.confidence - 5.0
                )
            ),
        )
        openai_evaluator = MagicMock(
            enabled=True,
            evaluate=AsyncMock(
                return_value=SimpleNamespace(
                    adjustment=-10.0,
                    recommended=True,
                    reasoning="weak setup",
                )
            ),
        )
        signal_queue = MagicMock()
        signal_queue.put = AsyncMock(return_value=True)
        scanner = _make_scan_ready_scanner(
            channel=channel,
            signal_queue=signal_queue,
            predictive=predictive,
            openai_evaluator=openai_evaluator,
        )

        with patch("src.scanner.get_ai_insight", AsyncMock(return_value=SimpleNamespace(label="Neutral", summary="", score=0.0))), \
             patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=50.0, blocked=False)), \
             patch.object(scanner, "_evaluate_setup", return_value=_setup_pass()), \
             patch.object(scanner, "_evaluate_execution", return_value=_execution_pass()), \
             patch.object(scanner, "_evaluate_risk", return_value=_risk_pass()):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        assert openai_evaluator.evaluate.await_args.kwargs["confidence_before"] > 70.0
        signal_queue.put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_openai_skip_prevents_enqueue(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal(channel="360_SCALP", signal_id="SIG-002")
        signal_queue = MagicMock()
        signal_queue.put = AsyncMock(return_value=True)
        scanner = _make_scan_ready_scanner(
            channel=channel,
            signal_queue=signal_queue,
            openai_evaluator=MagicMock(
                enabled=True,
                evaluate=AsyncMock(
                    return_value=SimpleNamespace(
                        adjustment=0.0,
                        recommended=False,
                        reasoning="reject",
                    )
                ),
            ),
        )

        with patch("src.scanner.get_ai_insight", AsyncMock(return_value=SimpleNamespace(label="Neutral", summary="", score=0.0))), \
             patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=55.0, blocked=False)), \
             patch.object(scanner, "_evaluate_setup", return_value=_setup_pass()), \
             patch.object(scanner, "_evaluate_execution", return_value=_execution_pass()), \
             patch.object(scanner, "_evaluate_risk", return_value=_risk_pass()):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        scanner.signal_queue.put.assert_not_awaited()


class TestScannerEnqueueSemantics:
    @pytest.mark.asyncio
    async def test_cooldown_not_started_when_enqueue_fails(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal(channel="360_SCALP", signal_id="SIG-DROP")
        signal_queue = MagicMock()
        signal_queue.put = AsyncMock(return_value=False)
        scanner = _make_scan_ready_scanner(channel=channel, signal_queue=signal_queue)

        with patch("src.scanner.get_ai_insight", AsyncMock(return_value=SimpleNamespace(label="Neutral", summary="", score=0.0))), \
             patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=80.0, blocked=False)), \
             patch.object(scanner, "_evaluate_setup", return_value=_setup_pass()), \
             patch.object(scanner, "_evaluate_execution", return_value=_execution_pass()), \
             patch.object(scanner, "_evaluate_risk", return_value=_risk_pass()):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        assert ("BTCUSDT", "360_SCALP") not in scanner._cooldown_until

    @pytest.mark.asyncio
    async def test_failed_enqueue_does_not_suppress_later_signal(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.side_effect = [
            _make_signal(channel="360_SCALP", signal_id="SIG-FIRST"),
            _make_signal(channel="360_SCALP", signal_id="SIG-SECOND"),
        ]
        signal_queue = MagicMock()
        signal_queue.put = AsyncMock(side_effect=[False, True])
        scanner = _make_scan_ready_scanner(channel=channel, signal_queue=signal_queue)

        with patch("src.scanner.get_ai_insight", AsyncMock(return_value=SimpleNamespace(label="Neutral", summary="", score=0.0))), \
             patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=80.0, blocked=False)), \
             patch.object(scanner, "_evaluate_setup", return_value=_setup_pass()), \
             patch.object(scanner, "_evaluate_execution", return_value=_execution_pass()), \
             patch.object(scanner, "_evaluate_risk", return_value=_risk_pass()):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        assert signal_queue.put.await_count == 2
        assert scanner._cooldown_until.get(("BTCUSDT", "360_SCALP")) is not None


class TestComputeIndicatorsArrayShape:
    """_compute_indicators must tolerate 2-D (non-flat) candle arrays."""

    def test_2d_arrays_do_not_raise(self):
        """Candle data stored as 2-D arrays must be flattened without error."""
        import numpy as np
        n = 40
        flat = np.arange(1.0, n + 1.0)
        # Wrap flat 1-D arrays into 2-D column vectors (simulates bad storage)
        candles = {
            "5m": {
                "high": flat.reshape(-1, 1),
                "low": (flat - 0.5).reshape(-1, 1),
                "close": flat.reshape(-1, 1),
                "volume": np.ones((n, 1)) * 100,
            }
        }
        scanner = _make_scanner()
        # Should not raise ValueError about truth value of array
        indicators = scanner._compute_indicators(candles)
        assert "5m" in indicators
        # EMA values must be scalar floats
        assert isinstance(indicators["5m"].get("ema9_last"), float)
        assert isinstance(indicators["5m"].get("ema21_last"), float)

    def test_1d_arrays_still_work(self):
        """Normal 1-D candle arrays continue to produce correct indicators."""
        import numpy as np
        n = 40
        flat = np.arange(1.0, n + 1.0)
        candles = {
            "5m": {
                "high": flat,
                "low": flat - 0.5,
                "close": flat,
                "volume": np.ones(n) * 100,
            }
        }
        scanner = _make_scanner()
        indicators = scanner._compute_indicators(candles)
        assert isinstance(indicators["5m"].get("ema9_last"), float)


class TestSpreadCacheFailureTTL:
    """Failed order-book fetches (e.g. HTTP 400) must be cached long enough
    to avoid hammering the endpoint on every scan cycle."""

    @pytest.mark.asyncio
    async def test_failed_fetch_not_retried_within_fail_ttl(self):
        """When fetch_order_book returns None the fallback is cached for
        _SPREAD_FAIL_CACHE_TTL seconds, not the shorter _SPREAD_CACHE_TTL."""
        scanner = _make_scanner()
        mock_client = MagicMock()
        mock_client.fetch_order_book = AsyncMock(return_value=None)
        scanner.spot_client = mock_client

        spread1 = await scanner._get_spread_pct("EURUSDT")
        assert spread1 == 0.01  # fallback

        # Second call within the fail-TTL window should hit cache, not the client
        spread2 = await scanner._get_spread_pct("EURUSDT")
        assert spread2 == 0.01
        # fetch_order_book must only have been called once despite two calls
        assert mock_client.fetch_order_book.await_count == 1

    @pytest.mark.asyncio
    async def test_successful_fetch_uses_normal_ttl(self):
        """Successful fetches are cached and returned on the next call."""
        scanner = _make_scanner()
        mock_client = MagicMock()
        mock_client.fetch_order_book = AsyncMock(
            return_value={"bids": [["100.0", "1"]], "asks": [["100.01", "1"]]}
        )
        scanner.spot_client = mock_client

        spread1 = await scanner._get_spread_pct("BTCUSDT")
        spread2 = await scanner._get_spread_pct("BTCUSDT")
        assert spread1 == spread2
        # Client called only once due to caching
        assert mock_client.fetch_order_book.await_count == 1

    @pytest.mark.asyncio
    async def test_fail_cache_expires_and_retries(self):
        """After _SPREAD_FAIL_CACHE_TTL elapses the endpoint is retried."""
        from src.scanner import _SPREAD_FAIL_CACHE_TTL
        scanner = _make_scanner()
        mock_client = MagicMock()
        mock_client.fetch_order_book = AsyncMock(return_value=None)
        scanner.spot_client = mock_client

        await scanner._get_spread_pct("EURUSDT")
        assert mock_client.fetch_order_book.await_count == 1

        # Simulate the fail-TTL expiring by backdating the cached expiry
        symbol = "EURUSDT"
        cached_spread, expiry = scanner._order_book_cache[symbol]
        scanner._order_book_cache[symbol] = (cached_spread, expiry - _SPREAD_FAIL_CACHE_TTL - 1)

        # Reset per-cycle fetch counter so the cap doesn't block the retry
        scanner._order_book_fetches_this_cycle = 0

        await scanner._get_spread_pct("EURUSDT")
        # Endpoint must have been called a second time after expiry
        assert mock_client.fetch_order_book.await_count == 2

    @pytest.mark.asyncio
    async def test_fail_ttl_longer_than_success_ttl(self):
        """_SPREAD_FAIL_CACHE_TTL must be greater than _SPREAD_CACHE_TTL."""
        from src.scanner import _SPREAD_CACHE_TTL, _SPREAD_FAIL_CACHE_TTL
        assert _SPREAD_FAIL_CACHE_TTL > _SPREAD_CACHE_TTL

    @pytest.mark.asyncio
    async def test_futures_market_uses_futures_client(self):
        """When market='futures', the futures client is used instead of spot."""
        scanner = _make_scanner()
        mock_futures = MagicMock()
        mock_futures.fetch_order_book = AsyncMock(
            return_value={"bids": [["2000.0", "1"]], "asks": [["2001.0", "1"]]}
        )
        mock_spot = MagicMock()
        mock_spot.fetch_order_book = AsyncMock(return_value=None)
        scanner.futures_client = mock_futures
        scanner.spot_client = mock_spot

        spread = await scanner._get_spread_pct("XAUUSDT", market="futures")
        # Futures client must have been called, spot client must not
        assert mock_futures.fetch_order_book.await_count == 1
        assert mock_spot.fetch_order_book.await_count == 0
        assert spread > 0

    @pytest.mark.asyncio
    async def test_spot_market_uses_spot_client(self):
        """When market='spot' (default), the spot client is used."""
        scanner = _make_scanner()
        mock_spot = MagicMock()
        mock_spot.fetch_order_book = AsyncMock(
            return_value={"bids": [["100.0", "1"]], "asks": [["100.01", "1"]]}
        )
        mock_futures = MagicMock()
        mock_futures.fetch_order_book = AsyncMock(return_value=None)
        scanner.spot_client = mock_spot
        scanner.futures_client = mock_futures

        spread = await scanner._get_spread_pct("BTCUSDT", market="spot")
        assert mock_spot.fetch_order_book.await_count == 1
        assert mock_futures.fetch_order_book.await_count == 0
        assert spread > 0
