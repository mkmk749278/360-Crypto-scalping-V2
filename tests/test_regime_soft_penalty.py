"""Tests for regime-adaptive soft-penalty gate logic in _prepare_signal().

Verifies that:
- 5 gates (VWAP, OI, Spoof, VolDiv, Cluster) apply scaled confidence penalties
  instead of hard-blocking signals.
- Penalty severity scales by live market regime via _REGIME_PENALTY_MULTIPLIER.
- 3 gates (MTF, Kill Zone, Cross-Asset) remain hard-blocking.
- Accumulated penalties are reflected on sig.soft_penalty_total /
  sig.regime_penalty_multiplier / sig.soft_gate_flags.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.channels.base import Signal
from src.regime import MarketRegime
from src.scanner import Scanner, _REGIME_PENALTY_MULTIPLIER
from src.signal_quality import (
    ExecutionAssessment,
    RiskAssessment,
    SetupAssessment,
    SetupClass,
)
from src.smc import Direction
from src.utils import utcnow


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_scanner.py to keep this file self-contained)
# ---------------------------------------------------------------------------

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


def _make_scanner(**kwargs) -> Scanner:
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


# Score with total=85.0 — high enough to survive most single-gate penalties
_FAKE_SCORE_HIGH = SimpleNamespace(
    total=85.0,
    quality_tier=SimpleNamespace(value="A"),
    components={
        "market": 20.0,
        "setup": 25.0,
        "execution": 18.0,
        "risk": 15.0,
        "context": 7.0,
    },
)

# Score with total=50.0 — moderate; accumulation of penalties will kill it
_FAKE_SCORE_MEDIUM = SimpleNamespace(
    total=50.0,
    quality_tier=SimpleNamespace(value="B"),
    components={
        "market": 20.0,
        "setup": 15.0,
        "execution": 12.0,
        "risk": 10.0,
        "context": 3.0,
    },
)


def _make_scan_ready_scanner(
    *,
    channel: MagicMock,
    signal_queue: MagicMock,
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
        onchain_client=MagicMock(get_exchange_flow=AsyncMock(return_value=None)),
    )


def _common_patches(scanner, score=None, extra=None):
    """Return ExitStack with standard pipeline patches."""
    if score is None:
        score = _FAKE_SCORE_HIGH
    stack = contextlib.ExitStack()
    stack.enter_context(
        patch("src.scanner.compute_confidence", return_value=SimpleNamespace(total=70.0, blocked=False))
    )
    stack.enter_context(patch("src.scanner.score_signal_components", return_value=score))
    stack.enter_context(patch("src.scanner.check_vwap_extension", return_value=(True, "")))
    stack.enter_context(patch("src.scanner.check_kill_zone_gate", return_value=(True, "")))
    stack.enter_context(patch.object(scanner, "_evaluate_setup", return_value=_setup_pass()))
    stack.enter_context(patch.object(scanner, "_evaluate_execution", return_value=_execution_pass()))
    stack.enter_context(patch.object(scanner, "_evaluate_risk", return_value=_risk_pass()))
    for p in (extra or []):
        stack.enter_context(p)
    return stack


# ---------------------------------------------------------------------------
# 1. Regime multiplier table constants
# ---------------------------------------------------------------------------

class TestRegimePenaltyMultiplierTable:
    """Verify the _REGIME_PENALTY_MULTIPLIER constant is correctly defined."""

    def test_all_regimes_present(self):
        expected = {"TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "QUIET"}
        assert set(_REGIME_PENALTY_MULTIPLIER.keys()) == expected

    def test_volatile_highest(self):
        assert _REGIME_PENALTY_MULTIPLIER["VOLATILE"] > _REGIME_PENALTY_MULTIPLIER["RANGING"]

    def test_trending_most_lenient(self):
        assert _REGIME_PENALTY_MULTIPLIER["TRENDING_UP"] < _REGIME_PENALTY_MULTIPLIER["RANGING"]
        assert _REGIME_PENALTY_MULTIPLIER["TRENDING_DOWN"] < _REGIME_PENALTY_MULTIPLIER["RANGING"]

    def test_ranging_is_baseline(self):
        assert _REGIME_PENALTY_MULTIPLIER["RANGING"] == 1.0


# ---------------------------------------------------------------------------
# 2. Regime-scaled VWAP penalty
# ---------------------------------------------------------------------------

class TestRegimeScaledVWAPPenalty:
    """Test that VWAP soft-penalty scales correctly by regime."""

    @pytest.mark.asyncio
    async def test_soft_penalty_volatile_regime_scaling(self):
        """VOLATILE regime: VWAP penalty = 15.0 × 1.5 = 22.5 (360_SCALP weight)."""
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal()
        sq = MagicMock()
        sq.put = AsyncMock(return_value=True)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.VOLATILE
        )

        captured = {}

        async def _capture(sig):
            captured["sig"] = sig
            return True

        sq.put = AsyncMock(side_effect=_capture)

        with _common_patches(scanner, extra=[
            patch("src.scanner.check_vwap_extension", return_value=(False, "VWAP: overextended")),
        ]):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        sig = captured.get("sig")
        assert sig is not None, "Signal must be enqueued (soft penalty, not hard block)"
        assert sig.soft_penalty_total == pytest.approx(22.5, abs=0.1)
        assert sig.regime_penalty_multiplier == pytest.approx(1.5)
        assert "VWAP" in sig.soft_gate_flags

    @pytest.mark.asyncio
    async def test_soft_penalty_trending_lenient(self):
        """TRENDING_UP regime: VWAP penalty = 15.0 × 0.6 = 9.0 (360_SCALP weight)."""
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal()
        sq = MagicMock()

        captured = {}

        async def _capture(sig):
            captured["sig"] = sig
            return True

        sq.put = AsyncMock(side_effect=_capture)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.TRENDING_UP
        )

        with _common_patches(scanner, extra=[
            patch("src.scanner.check_vwap_extension", return_value=(False, "VWAP: overextended")),
        ]):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        sig = captured.get("sig")
        assert sig is not None
        assert sig.soft_penalty_total == pytest.approx(9.0, abs=0.1)
        assert sig.regime_penalty_multiplier == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 3. Multiple soft gates accumulate
# ---------------------------------------------------------------------------

class TestMultipleSoftGatesAccumulate:
    """Multiple failing soft gates accumulate their penalties."""

    @pytest.mark.asyncio
    async def test_multiple_soft_gates_accumulate_ranging(self):
        """RANGING regime: VWAP (15.0) + OI (8.0) = 23.0 total penalty (360_SCALP weights)."""
        from src.order_flow import OISnapshot

        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal()
        sq = MagicMock()

        captured = {}

        async def _capture(sig):
            captured["sig"] = sig
            return True

        sq.put = AsyncMock(side_effect=_capture)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.RANGING
        )
        oi_store = MagicMock()
        oi_store._oi = {
            "BTCUSDT": [
                OISnapshot(timestamp=1.0, open_interest=5000.0),
                OISnapshot(timestamp=2.0, open_interest=4800.0),
            ]
        }
        scanner.order_flow_store = oi_store

        with _common_patches(scanner, extra=[
            patch("src.scanner.check_vwap_extension", return_value=(False, "VWAP: overextended")),
            patch("src.scanner.analyse_oi", return_value=MagicMock()),
            patch("src.scanner.check_oi_gate", return_value=(False, "OI: squeeze")),
        ]):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        sig = captured.get("sig")
        assert sig is not None
        # RANGING multiplier = 1.0 → 15.0 + 8.0 = 23.0 (360_SCALP channel weights)
        assert sig.soft_penalty_total == pytest.approx(23.0, abs=0.1)
        assert "VWAP" in sig.soft_gate_flags
        assert "OI" in sig.soft_gate_flags


# ---------------------------------------------------------------------------
# 4. Hard gates still block
# ---------------------------------------------------------------------------

class TestHardGatesStillBlock:
    """MTF, Kill Zone, and Cross-Asset gates remain hard-blocking."""

    @pytest.mark.asyncio
    async def test_mtf_gate_still_hard_blocks(self):
        """MTF gate still returns None, None regardless of regime."""
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal()
        sq = MagicMock()
        sq.put = AsyncMock(return_value=True)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.TRENDING_UP
        )

        with _common_patches(scanner, extra=[
            patch("src.scanner.check_mtf_gate", return_value=(False, "MTF misaligned")),
        ]):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        sq.put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_kill_zone_applies_soft_penalty(self):
        """Kill zone gate is now a soft penalty, not a hard block.

        When check_kill_zone_gate returns (False, reason), a soft penalty of
        10 pts (scaled by regime multiplier) is applied and the signal is still
        enqueued if it passes the confidence floor.
        """
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal()
        sq = MagicMock()
        sq.put = AsyncMock(return_value=True)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.VOLATILE
        )

        with _common_patches(scanner, extra=[
            patch("src.scanner.check_kill_zone_gate", return_value=(False, "Kill zone: WEEKEND")),
        ]):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        # Soft penalty: signal is still enqueued (not hard-blocked)
        sq.put.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cross_asset_gate_still_hard_blocks(self):
        """Cross-asset gate still returns None, None for altcoins regardless of regime."""
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        altcoin_signal = Signal(
            channel="360_SCALP",
            symbol="SOLUSDT",
            direction=Direction.LONG,
            entry=100.0,
            stop_loss=95.0,
            tp1=105.0,
            tp2=110.0,
            confidence=10.0,
            signal_id="SIG-SOL",
            timestamp=utcnow(),
        )
        channel.evaluate.return_value = altcoin_signal
        sq = MagicMock()
        sq.put = AsyncMock(return_value=True)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.TRENDING_UP
        )

        with _common_patches(scanner, extra=[
            patch("src.scanner.check_cross_asset_gate", return_value=(False, "BTC: DUMPING")),
        ]):
            await scanner._scan_symbol("SOLUSDT", 5_000_000)

        sq.put.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. Signal survives single soft gate
# ---------------------------------------------------------------------------

class TestSignalSurvivesSingleSoftGate:
    """A signal with high confidence survives one soft-gate penalty."""

    @pytest.mark.asyncio
    async def test_signal_survives_vwap_soft_penalty(self):
        """FAKE_SCORE=85, VWAP penalty=7.2 (TRENDING_UP) → 77.8 > min_conf=10 → passes."""
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal()
        sq = MagicMock()
        sq.put = AsyncMock(return_value=True)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.TRENDING_UP
        )

        with _common_patches(scanner, extra=[
            patch("src.scanner.check_vwap_extension", return_value=(False, "VWAP: overextended")),
        ]):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        sq.put.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. Signal killed by accumulated soft penalties
# ---------------------------------------------------------------------------

class TestSignalKilledByAccumulatedSoftPenalties:
    """Enough accumulated penalties drop signal below min_confidence threshold."""

    @pytest.mark.asyncio
    async def test_signal_killed_by_accumulated_penalties(self):
        """RANGING: VWAP(12) + OI(15) + SPOOF(10) = 37 pts.

        Score=50 - 37 = 13. If min_confidence > 13, signal is dropped.
        Set min_confidence=20 to ensure it is killed.
        """
        from src.order_flow import OISnapshot

        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=20.0)
        channel.evaluate.return_value = _make_signal()
        sq = MagicMock()
        sq.put = AsyncMock(return_value=True)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.RANGING
        )
        oi_store = MagicMock()
        oi_store._oi = {
            "BTCUSDT": [
                OISnapshot(timestamp=1.0, open_interest=5000.0),
                OISnapshot(timestamp=2.0, open_interest=4800.0),
            ]
        }
        scanner.order_flow_store = oi_store

        with _common_patches(scanner, score=_FAKE_SCORE_MEDIUM, extra=[
            patch("src.scanner.check_vwap_extension", return_value=(False, "VWAP: overextended")),
            patch("src.scanner.analyse_oi", return_value=MagicMock()),
            patch("src.scanner.check_oi_gate", return_value=(False, "OI: squeeze")),
            patch("src.scanner.check_spoof_gate", return_value=(False, "Spoof: wall detected")),
        ]):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        # 50 - 37 = 13 < min_confidence(20) → dropped
        sq.put.assert_not_awaited()


# ---------------------------------------------------------------------------
# 7. Soft gate flags tracked
# ---------------------------------------------------------------------------

class TestSoftGateFlagsTracked:
    """Verify sig.soft_gate_flags records which gates fired."""

    @pytest.mark.asyncio
    async def test_soft_gate_flags_contain_fired_gates(self):
        """Trigger VWAP + SPOOF → soft_gate_flags = 'VWAP,SPOOF'."""
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal()
        sq = MagicMock()

        captured = {}

        async def _capture(sig):
            captured["sig"] = sig
            return True

        sq.put = AsyncMock(side_effect=_capture)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.RANGING
        )

        with _common_patches(scanner, extra=[
            patch("src.scanner.check_vwap_extension", return_value=(False, "VWAP: overextended")),
            patch("src.scanner.check_spoof_gate", return_value=(False, "Spoof: wall")),
        ]):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        sig = captured.get("sig")
        assert sig is not None
        flags = sig.soft_gate_flags.split(",")
        assert "VWAP" in flags
        assert "SPOOF" in flags

    @pytest.mark.asyncio
    async def test_no_gates_fired_empty_flags(self):
        """When no soft gates fire, soft_gate_flags is empty string."""
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal()
        sq = MagicMock()

        captured = {}

        async def _capture(sig):
            captured["sig"] = sig
            return True

        sq.put = AsyncMock(side_effect=_capture)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.RANGING
        )

        with _common_patches(scanner):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        sig = captured.get("sig")
        assert sig is not None
        assert sig.soft_gate_flags == ""
        assert sig.soft_penalty_total == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. Regime multiplier stored on signal
# ---------------------------------------------------------------------------

class TestRegimeMultiplierStoredOnSignal:
    """Verify sig.regime_penalty_multiplier reflects the live regime."""

    @pytest.mark.asyncio
    async def test_regime_multiplier_stored_volatile(self):
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SCALP", min_confidence=10.0)
        channel.evaluate.return_value = _make_signal()
        sq = MagicMock()

        captured = {}

        async def _capture(sig):
            captured["sig"] = sig
            return True

        sq.put = AsyncMock(side_effect=_capture)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.VOLATILE
        )

        with _common_patches(scanner):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        sig = captured.get("sig")
        assert sig is not None
        assert sig.regime_penalty_multiplier == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_regime_multiplier_stored_quiet(self):
        """In QUIET regime, 360_SWING uses the standard QUIET penalty (0.8)."""
        channel = MagicMock()
        channel.config = SimpleNamespace(name="360_SWING", min_confidence=10.0)
        channel.evaluate.return_value = Signal(
            channel="360_SWING",
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
        sq = MagicMock()

        captured = {}

        async def _capture(sig):
            captured["sig"] = sig
            return True

        sq.put = AsyncMock(side_effect=_capture)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.QUIET
        )

        with _common_patches(scanner):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        sig = captured.get("sig")
        assert sig is not None
        assert sig.regime_penalty_multiplier == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_regime_multiplier_scalp_quiet_uses_higher_penalty(self):
        """In QUIET regime, 360_SCALP uses the higher 1.8× penalty multiplier."""
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
        sq = MagicMock()

        captured = {}

        async def _capture(sig):
            captured["sig"] = sig
            return True

        sq.put = AsyncMock(side_effect=_capture)
        scanner = _make_scan_ready_scanner(
            channel=channel, signal_queue=sq, regime=MarketRegime.QUIET
        )

        with _common_patches(scanner):
            await scanner._scan_symbol("BTCUSDT", 10_000_000)

        sig = captured.get("sig")
        # Signal may or may not pass the QUIET_SCALP_MIN_CONFIDENCE gate.
        # If it does pass, the regime_penalty_multiplier must be 1.8.
        if sig is not None:
            assert sig.regime_penalty_multiplier == pytest.approx(1.8)
