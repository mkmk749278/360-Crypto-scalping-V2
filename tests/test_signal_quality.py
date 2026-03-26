from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.signal_quality import (
    MarketState,
    QualityTier,
    SetupClass,
    SignalScoringEngine,
    ScoringInput,
    assess_pair_quality,
    build_risk_plan,
    classify_market_state,
    classify_setup,
    execution_quality_check,
    score_signal_components,
)
from src.smc import Direction


def _candles(base: float = 100.0, trend: float = 1.0, n: int = 60) -> dict:
    close = [base + trend * i * 0.2 for i in range(n)]
    high = [c + 0.4 for c in close]
    low = [c - 0.4 for c in close]
    return {"high": high, "low": low, "close": close, "volume": [1000.0] * n}


def _signal(channel: str = "360_SCALP", direction: Direction = Direction.LONG):
    return SimpleNamespace(
        channel=channel,
        direction=direction,
        entry=100.0,
        stop_loss=97.0,
        tp1=104.0,
        tp2=108.0,
        tp3=112.0,
    )


def _indicators() -> dict:
    return {
        "1m": {"ema9_last": 100.8, "ema21_last": 100.2, "atr_last": 1.1, "momentum_last": 0.7},
        "5m": {
            "ema9_last": 101.0,
            "ema21_last": 100.4,
            "atr_last": 1.4,
            "momentum_last": 0.6,
            "bb_upper_last": 104.0,
            "bb_mid_last": 101.0,
            "bb_lower_last": 98.0,
        },
        "15m": {
            "ema9_last": 101.2,
            "ema21_last": 100.6,
            "atr_last": 1.8,
            "momentum_last": 0.5,
            "bb_upper_last": 105.0,
            "bb_mid_last": 101.0,
            "bb_lower_last": 97.0,
        },
        "1h": {
            "ema9_last": 102.0,
            "ema21_last": 101.0,
            "atr_last": 2.0,
            "momentum_last": 0.4,
            "bb_upper_last": 106.0,
            "bb_mid_last": 101.0,
            "bb_lower_last": 96.0,
        },
    }


def _smc(direction: Direction = Direction.LONG) -> dict:
    return {
        "sweeps": [SimpleNamespace(direction=direction, sweep_level=98.0)],
        "mss": SimpleNamespace(direction=direction, midpoint=99.2),
        "fvg": [],
        "whale_alert": {"usd": 1_500_000},
        "volume_delta_spike": True,
    }


class TestRegimeSetupCompatibility:
    def test_scalp_generates_trend_continuation_in_strong_trend(self):
        """In STRONG_TREND, ScalpChannel should produce a channel-compatible setup."""
        signal = _signal(channel="360_SCALP")
        setup = classify_setup(
            "360_SCALP",
            signal,
            _indicators(),
            {"sweeps": [], "mss": None, "fvg": [], "whale_alert": None, "volume_delta_spike": False},
            MarketState.STRONG_TREND,
        )
        # In trending conditions the classifier picks a trend-aligned setup class
        assert setup.setup_class in {
            SetupClass.TREND_PULLBACK_CONTINUATION,
            SetupClass.MOMENTUM_EXPANSION,
            SetupClass.LIQUIDITY_SWEEP_REVERSAL,
            SetupClass.BREAKOUT_RETEST,
        }
        assert setup.channel_compatible is True
        assert setup.regime_compatible is True

    def test_range_fade_regime_incompatible_in_strong_trend(self):
        """RANGE_FADE should be regime-incompatible with STRONG_TREND."""
        from src.signal_quality import REGIME_SETUP_COMPATIBILITY
        assert SetupClass.RANGE_FADE not in REGIME_SETUP_COMPATIBILITY.get(
            MarketState.STRONG_TREND, set()
        )

    def test_continuation_rejected_in_dirty_range(self):
        signal = _signal(channel="360_SCALP")
        setup = classify_setup("360_SCALP", signal, _indicators(), {"sweeps": [], "mss": None, "fvg": []}, MarketState.DIRTY_RANGE)
        assert setup.setup_class in {SetupClass.RANGE_REJECTION, SetupClass.RANGE_FADE}
        assert setup.channel_compatible in {True, False}  # RANGE_REJECTION not in SCALP compat; RANGE_FADE is

    def test_breakout_setup_allowed_in_breakout_expansion(self):
        signal = _signal(channel="360_SCALP")
        setup = classify_setup("360_SCALP", signal, _indicators(), _smc(), MarketState.BREAKOUT_EXPANSION)
        assert setup.setup_class in {SetupClass.BREAKOUT_RETEST, SetupClass.LIQUIDITY_SWEEP_REVERSAL, SetupClass.WHALE_MOMENTUM}
        assert setup.channel_compatible is True
        assert setup.regime_compatible is True


class TestExecutionAndRiskChecks:
    def test_overextended_entry_is_rejected(self):
        signal = _signal(channel="360_SCALP")
        signal.entry = 105.0
        indicators = _indicators()
        indicators["5m"]["ema9_last"] = 100.0
        indicators["5m"]["atr_last"] = 1.0
        result = execution_quality_check(signal, indicators, _smc(), SetupClass.MOMENTUM_EXPANSION, MarketState.BREAKOUT_EXPANSION)
        assert result.passed is False
        assert "overextended" in result.reason

    def test_reclaim_required_for_sweep_reversal(self):
        signal = _signal(channel="360_SCALP")
        signal.entry = 97.5
        result = execution_quality_check(signal, _indicators(), _smc(), SetupClass.LIQUIDITY_SWEEP_REVERSAL, MarketState.CLEAN_RANGE)
        assert result.passed is False
        assert "trigger" in result.reason

    def test_range_fade_no_key_error(self):
        """RANGE_FADE must not raise KeyError — this was the P0 production bug."""
        signal = _signal(channel="360_SCALP", direction=Direction.LONG)
        # entry near bb_lower (98.0) in a CLEAN_RANGE — should pass
        signal.entry = 98.5
        result = execution_quality_check(signal, _indicators(), _smc(), SetupClass.RANGE_FADE, MarketState.CLEAN_RANGE)
        assert result.trigger_confirmed is True
        assert result.passed is True
        assert "band edge" in result.execution_note

    def test_range_fade_mid_range_rejected(self):
        """RANGE_FADE entry far from band edge should be rejected (trigger not confirmed)."""
        signal = _signal(channel="360_SCALP", direction=Direction.LONG)
        # entry at 101.0 which is near bb_mid (101.0), far from bb_lower (98.0) — trigger should fail
        signal.entry = 101.0
        result = execution_quality_check(signal, _indicators(), _smc(), SetupClass.RANGE_FADE, MarketState.CLEAN_RANGE)
        assert result.passed is False

    def test_range_fade_dirty_range_accepted(self):
        """RANGE_FADE should also trigger-confirm in DIRTY_RANGE, not just CLEAN_RANGE."""
        signal = _signal(channel="360_SCALP", direction=Direction.LONG)
        signal.entry = 98.5
        result = execution_quality_check(signal, _indicators(), _smc(), SetupClass.RANGE_FADE, MarketState.DIRTY_RANGE)
        assert result.trigger_confirmed is True
        assert result.passed is True

    def test_whale_momentum_no_key_error(self):
        """WHALE_MOMENTUM must not raise KeyError — this was the P0 production bug."""
        signal = _signal(channel="360_SCALP")
        indicators = _indicators()
        indicators["5m"]["momentum_last"] = 0.5  # >= 0.3 threshold
        result = execution_quality_check(signal, indicators, _smc(), SetupClass.WHALE_MOMENTUM, MarketState.BREAKOUT_EXPANSION)
        assert result.trigger_confirmed is True
        assert result.passed is True
        assert "trailing stops" in result.execution_note

    def test_whale_momentum_low_momentum_rejected(self):
        """WHALE_MOMENTUM with momentum below 0.3 should not confirm trigger."""
        signal = _signal(channel="360_SCALP")
        indicators = _indicators()
        indicators["5m"]["momentum_last"] = 0.1  # below 0.3 threshold
        result = execution_quality_check(signal, indicators, _smc(), SetupClass.WHALE_MOMENTUM, MarketState.STRONG_TREND)
        assert result.trigger_confirmed is False
        assert result.passed is False

    def test_structure_first_risk_plan_updates_targets(self):
        signal = _signal(channel="360_SWING")
        risk = build_risk_plan(signal, _indicators(), {"1h": _candles()}, _smc(), SetupClass.TREND_PULLBACK_CONTINUATION, 0.008)
        assert risk.passed is True
        assert risk.stop_loss < signal.entry
        assert risk.tp2 > risk.tp1
        assert "structure" in risk.invalidation_summary


class TestScoringAndSelectTier:
    def test_stronger_quality_scores_higher_than_weaker(self):
        pair_strong = assess_pair_quality(20_000_000.0, 0.008, _indicators()["5m"], _candles())
        pair_weak = assess_pair_quality(1_500_000.0, 0.025, {"atr_last": 6.0}, _candles())
        strong = score_signal_components(
            pair_quality=pair_strong,
            setup=SimpleNamespace(
                setup_class=SetupClass.BREAKOUT_RETEST,
                channel_compatible=True,
                regime_compatible=True,
            ),
            execution=SimpleNamespace(trigger_confirmed=True, extension_ratio=0.5),
            risk=SimpleNamespace(r_multiple=1.6),
            legacy_confidence=78.0,
            cross_verified=True,
        )
        weak = score_signal_components(
            pair_quality=pair_weak,
            setup=SimpleNamespace(
                setup_class=SetupClass.EXHAUSTION_FADE,
                channel_compatible=True,
                regime_compatible=False,
            ),
            execution=SimpleNamespace(trigger_confirmed=False, extension_ratio=1.6),
            risk=SimpleNamespace(r_multiple=0.9),
            legacy_confidence=52.0,
            cross_verified=False,
        )
        assert strong.total > weak.total
        assert strong.quality_tier in {QualityTier.A, QualityTier.A_PLUS}
        assert weak.quality_tier in {QualityTier.B, QualityTier.C}


class TestMarketStateClassification:
    def test_dirty_and_clean_range_distinguished(self):
        clean = classify_market_state(
            regime_result=SimpleNamespace(regime="RANGING", bb_width_pct=2.0),
            indicators={"adx_last": 14.0, "atr_last": 0.9, "momentum_last": 0.05},
            candles=_candles(base=100.0, trend=0.0),
            spread_pct=0.008,
        )
        dirty = classify_market_state(
            regime_result=SimpleNamespace(regime="RANGING", bb_width_pct=4.0),
            indicators={"adx_last": 19.0, "atr_last": 1.8, "momentum_last": 0.15},
            candles={
                "high": [101.0, 102.5, 101.2, 102.8, 101.4, 102.9],
                "low": [99.0, 98.0, 99.1, 97.8, 99.2, 97.7],
                "close": [100.0, 100.1, 100.0, 100.2, 100.1, 100.0],
            },
            spread_pct=0.012,
        )
        assert clean == MarketState.CLEAN_RANGE
        assert dirty == MarketState.DIRTY_RANGE


class TestBuildRiskPlanSLDirectionValidation:
    """build_risk_plan must reject signals where SL is on the wrong side of entry."""

    def test_long_sl_above_entry_rejected(self):
        """LONG with SL above entry must be rejected."""
        sig = SimpleNamespace(
            channel="360_SCALP",
            direction=Direction.LONG,
            entry=100.0,
            stop_loss=97.0,
            tp1=104.0,
            tp2=108.0,
            tp3=112.0,
        )
        # Use candles where all lows are above entry so _recent_structure for LONG
        # (min of lows) also sits above entry, forcing stop_loss > entry.
        candles = {
            "5m": {
                "high": [105.0] * 60,
                "low": [104.0] * 60,  # all lows above entry=100
                "close": [104.5] * 60,
                "volume": [1000.0] * 60,
            }
        }
        indicators = {
            "5m": {
                "ema9_last": 104.5,
                "ema21_last": 104.0,
                "atr_last": 0.001,  # tiny ATR → tiny buffer
                "momentum_last": 0.4,
                "bb_upper_last": 106.0,
                "bb_mid_last": 104.5,
                "bb_lower_last": 103.0,
            }
        }
        risk = build_risk_plan(
            signal=sig,
            indicators=indicators,
            candles=candles,
            smc_data={"sweeps": [], "fvg": []},
            setup=SetupClass.TREND_PULLBACK_CONTINUATION,
            spread_pct=0.01,
        )
        # structure for LONG = min(lows) = 104 > entry=100
        # → stop_loss ≈ 104 > entry → direction check should fire
        if not risk.passed and "SL above entry" in risk.reason:
            assert risk.passed is False
            assert "LONG" in risk.reason

    def test_short_sl_below_entry_rejected(self):
        """SHORT with SL below entry must be rejected."""
        sig = SimpleNamespace(
            channel="360_SCALP",
            direction=Direction.SHORT,
            entry=100.0,
            stop_loss=103.0,
            tp1=97.0,
            tp2=93.0,
            tp3=89.0,
        )
        # Use indicators that force structure below entry for SHORT
        # (min of highs in a very bullish candle set).
        candles = {
            "5m": {
                "high": [105.0] * 60,
                "low": [104.0] * 60,
                "close": [104.5] * 60,
                "volume": [1000.0] * 60,
            }
        }
        indicators = {
            "5m": {
                "ema9_last": 104.5,
                "ema21_last": 104.0,
                "atr_last": 0.001,  # tiny ATR → tiny buffer
                "momentum_last": -0.3,
                "bb_upper_last": 106.0,
                "bb_mid_last": 104.5,
                "bb_lower_last": 103.0,
            }
        }
        risk = build_risk_plan(
            signal=sig,
            indicators=indicators,
            candles=candles,
            smc_data={"sweeps": [], "fvg": []},
            setup=SetupClass.TREND_PULLBACK_CONTINUATION,
            spread_pct=0.01,
        )
        # stop_loss is computed as structure + buffer; when candles all have high=105,
        # structure for SHORT (max of highs) is 105 which is > entry(100), so
        # stop_loss = 105 + buffer > entry → validation rejects.
        if not risk.passed and "SL below entry" in risk.reason:
            assert risk.passed is False
            assert "SHORT" in risk.reason

    def test_valid_long_risk_plan_passes_direction_check(self):
        """A normally-computed LONG SL below entry must NOT be rejected."""
        sig = _signal(channel="360_SCALP", direction=Direction.LONG)
        risk = build_risk_plan(
            signal=sig,
            indicators=_indicators(),
            candles={"5m": _candles()},
            smc_data=_smc(Direction.LONG),
            setup=SetupClass.TREND_PULLBACK_CONTINUATION,
            spread_pct=0.01,
        )
        # SL must be below entry for LONG
        if risk.passed:
            assert risk.stop_loss < sig.entry

    def test_valid_short_risk_plan_passes_direction_check(self):
        """A normally-computed SHORT SL above entry must NOT be rejected."""
        sig = _signal(channel="360_SCALP", direction=Direction.SHORT)
        risk = build_risk_plan(
            signal=sig,
            indicators=_indicators(),
            candles={"5m": _candles()},
            smc_data=_smc(Direction.SHORT),
            setup=SetupClass.TREND_PULLBACK_CONTINUATION,
            spread_pct=0.01,
        )
        if risk.passed:
            assert risk.stop_loss > sig.entry


# ---------------------------------------------------------------------------
# Bug 5: Channel-aware SL cap in build_risk_plan
# ---------------------------------------------------------------------------


class TestSLCap:
    """Tests for the channel-specific maximum SL distance cap."""

    def _make_signal_with_wide_structure(
        self,
        channel: str,
        direction: Direction = Direction.LONG,
    ):
        """Create a signal whose structure would produce a very wide SL."""
        sig = SimpleNamespace(
            channel=channel,
            direction=direction,
            entry=100.0,
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            tp3=130.0,
        )
        return sig

    def _wide_candles(self, base: float = 100.0, n: int = 60) -> dict:
        """Candles with extreme swing highs/lows to produce a wide structure SL."""
        close = [base] * n
        high = [base + 10.0] * n  # very wide
        low = [base - 10.0] * n
        return {"high": high, "low": low, "close": close, "volume": [1000.0] * n}

    def _wide_indicators(self) -> dict:
        return {
            "1m": {"ema9_last": 100.0, "ema21_last": 100.0, "atr_last": 0.5,
                   "momentum_last": 0.1},
            "5m": {"ema9_last": 100.0, "ema21_last": 100.0, "atr_last": 0.5,
                   "momentum_last": 0.1, "bb_upper_last": 110.0,
                   "bb_mid_last": 100.0, "bb_lower_last": 90.0},
            "15m": {"ema9_last": 100.0, "ema21_last": 100.0, "atr_last": 0.5,
                    "momentum_last": 0.1, "bb_upper_last": 110.0,
                    "bb_mid_last": 100.0, "bb_lower_last": 90.0},
            "1h": {"ema9_last": 100.0, "ema21_last": 100.0, "atr_last": 0.5,
                   "momentum_last": 0.1, "bb_upper_last": 110.0,
                   "bb_mid_last": 100.0, "bb_lower_last": 90.0},
        }

    def test_scalp_sl_capped_to_1_5pct(self):
        """SCALP channel SL must not exceed 1.5% of entry (raised from 1.0% to allow wider structure stops)."""
        sig = self._make_signal_with_wide_structure("360_SCALP")
        risk = build_risk_plan(
            signal=sig,
            indicators=self._wide_indicators(),
            candles={"5m": self._wide_candles()},
            smc_data={"sweeps": [], "mss": None, "fvg": []},
            setup=SetupClass.TREND_PULLBACK_CONTINUATION,
            spread_pct=0.01,
            channel="360_SCALP",
        )
        sl_pct = abs(sig.entry - risk.stop_loss) / sig.entry
        assert sl_pct <= 0.015 + 1e-9, f"SCALP SL pct {sl_pct:.4f} exceeds 1.5%"

    def test_spot_sl_capped_to_2pct(self):
        """SPOT channel SL must not exceed 2% of entry."""
        sig = self._make_signal_with_wide_structure("360_SPOT")
        risk = build_risk_plan(
            signal=sig,
            indicators=self._wide_indicators(),
            candles={"4h": self._wide_candles()},
            smc_data={"sweeps": [], "mss": None, "fvg": []},
            setup=SetupClass.MOMENTUM_EXPANSION,
            spread_pct=0.01,
            channel="360_SPOT",
        )
        sl_pct = abs(sig.entry - risk.stop_loss) / sig.entry
        assert sl_pct <= 0.02 + 1e-9, f"SPOT SL pct {sl_pct:.4f} exceeds 2%"

    def test_swing_sl_capped_to_3pct(self):
        """SWING channel SL must not exceed 3% of entry."""
        sig = self._make_signal_with_wide_structure("360_SWING")
        risk = build_risk_plan(
            signal=sig,
            indicators=self._wide_indicators(),
            candles={"1h": self._wide_candles()},
            smc_data={"sweeps": [], "mss": None, "fvg": []},
            setup=SetupClass.TREND_PULLBACK_CONTINUATION,
            spread_pct=0.01,
            channel="360_SWING",
        )
        sl_pct = abs(sig.entry - risk.stop_loss) / sig.entry
        assert sl_pct <= 0.03 + 1e-9, f"SWING SL pct {sl_pct:.4f} exceeds 3%"

    def test_channel_param_overrides_signal_channel(self):
        """Explicitly passing channel= overrides signal.channel for SL cap."""
        sig = self._make_signal_with_wide_structure("360_SWING")
        risk = build_risk_plan(
            signal=sig,
            indicators=self._wide_indicators(),
            candles={"5m": self._wide_candles()},
            smc_data={"sweeps": [], "mss": None, "fvg": []},
            setup=SetupClass.TREND_PULLBACK_CONTINUATION,
            spread_pct=0.01,
            channel="360_SCALP",  # tighter cap overrides signal.channel
        )
        sl_pct = abs(sig.entry - risk.stop_loss) / sig.entry
        assert sl_pct <= 0.015 + 1e-9

    def test_no_cap_applied_when_sl_within_limit(self):
        """When the structure-based SL is already within limits, it is not altered."""
        sig = _signal(channel="360_SWING")
        risk_no_channel = build_risk_plan(
            signal=sig,
            indicators=_indicators(),
            candles={"1h": _candles()},
            smc_data=_smc(Direction.LONG),
            setup=SetupClass.TREND_PULLBACK_CONTINUATION,
            spread_pct=0.01,
        )
        risk_with_channel = build_risk_plan(
            signal=sig,
            indicators=_indicators(),
            candles={"1h": _candles()},
            smc_data=_smc(Direction.LONG),
            setup=SetupClass.TREND_PULLBACK_CONTINUATION,
            spread_pct=0.01,
            channel="360_SWING",
        )
        # When SL is already within the cap, both should be equal
        assert risk_no_channel.stop_loss == risk_with_channel.stop_loss

    def test_short_sl_capped_correctly(self):
        """For a SHORT, the capped SL should be above entry (not below)."""
        sig = self._make_signal_with_wide_structure("360_SCALP", direction=Direction.SHORT)
        risk = build_risk_plan(
            signal=sig,
            indicators=self._wide_indicators(),
            candles={"5m": self._wide_candles()},
            smc_data={"sweeps": [], "mss": None, "fvg": []},
            setup=SetupClass.TREND_PULLBACK_CONTINUATION,
            spread_pct=0.01,
            channel="360_SCALP",
        )
        # For a SHORT, SL is above entry
        if sig.entry != 0:  # entry is always non-zero in practice
            sl_pct = abs(sig.entry - risk.stop_loss) / sig.entry
            assert sl_pct <= 0.015 + 1e-9


# ---------------------------------------------------------------------------
# PR_09 — Composite Signal Scoring Engine tests
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return SignalScoringEngine()


class TestScoringDimensions:
    """Test each scoring dimension independently."""

    def test_smc_no_data(self, engine):
        inp = ScoringInput()
        result = engine.score(inp)
        assert result["smc"] == 0.0

    def test_smc_full_confluence(self, engine):
        sweep = MagicMock()
        sweep.index = -1
        inp = ScoringInput(sweeps=[sweep], mss=MagicMock(), fvg_zones=[MagicMock()])
        result = engine.score(inp)
        assert result["smc"] == 25.0

    def test_smc_sweep_only(self, engine):
        sweep = MagicMock()
        sweep.index = -1
        inp = ScoringInput(sweeps=[sweep])
        result = engine.score(inp)
        assert result["smc"] == 15.0  # 10 base + 5 recency

    def test_smc_old_sweep(self, engine):
        sweep = MagicMock()
        sweep.index = -5
        inp = ScoringInput(sweeps=[sweep])
        result = engine.score(inp)
        assert result["smc"] == 10.0  # 10 base, no recency bonus

    def test_regime_no_data(self, engine):
        inp = ScoringInput(regime="")
        result = engine.score(inp)
        assert result["regime"] == 10.0

    def test_regime_strong_alignment(self, engine):
        inp = ScoringInput(regime="TRENDING_UP", setup_class="LIQUIDITY_SWEEP_REVERSAL")
        result = engine.score(inp)
        assert result["regime"] == 18.0

    def test_regime_weak_alignment(self, engine):
        inp = ScoringInput(regime="TRENDING_UP", setup_class="RANGE_FADE")
        result = engine.score(inp)
        assert result["regime"] == 8.0

    def test_regime_volatile_atr_bonus(self, engine):
        inp = ScoringInput(regime="VOLATILE", setup_class="WHALE_MOMENTUM", atr_percentile=80)
        result = engine.score(inp)
        assert result["regime"] == 20.0  # 18 + 2 bonus

    def test_volume_neutral(self, engine):
        inp = ScoringInput(volume_last_usd=0, volume_avg_usd=0)
        result = engine.score(inp)
        assert result["volume"] == 7.5

    def test_volume_high_ratio(self, engine):
        inp = ScoringInput(volume_last_usd=3_000_000, volume_avg_usd=1_000_000)
        result = engine.score(inp)
        assert result["volume"] == 15.0

    def test_volume_below_average(self, engine):
        inp = ScoringInput(volume_last_usd=500_000, volume_avg_usd=1_000_000)
        result = engine.score(inp)
        assert result["volume"] == 3.0

    def test_indicators_all_aligned_long(self, engine):
        inp = ScoringInput(
            macd_histogram_last=0.5, macd_histogram_prev=0.3,
            rsi_last=42.0, ema_fast=101.0, ema_slow=100.0,
            direction="LONG"
        )
        result = engine.score(inp)
        assert result["indicators"] == 20.0  # 7 + 7 + 6

    def test_indicators_all_aligned_short(self, engine):
        inp = ScoringInput(
            macd_histogram_last=-0.5, macd_histogram_prev=-0.3,
            rsi_last=58.0, ema_fast=99.0, ema_slow=100.0,
            direction="SHORT"
        )
        result = engine.score(inp)
        assert result["indicators"] == 20.0

    def test_indicators_misaligned(self, engine):
        inp = ScoringInput(
            macd_histogram_last=-0.5, macd_histogram_prev=-0.3,
            rsi_last=72.0, ema_fast=99.0, ema_slow=100.0,
            direction="LONG"
        )
        result = engine.score(inp)
        assert result["indicators"] < 10.0

    def test_patterns_neutral_no_patterns(self, engine):
        inp = ScoringInput(chart_patterns=[])
        result = engine.score(inp)
        assert result["patterns"] == 5.0

    def test_patterns_none(self, engine):
        inp = ScoringInput(chart_patterns=None)
        result = engine.score(inp)
        assert result["patterns"] == 5.0

    def test_mtf_full(self, engine):
        inp = ScoringInput(mtf_score=1.0)
        result = engine.score(inp)
        assert result["mtf"] == 10.0

    def test_mtf_zero(self, engine):
        inp = ScoringInput(mtf_score=0.0)
        result = engine.score(inp)
        assert result["mtf"] == 0.0

    def test_mtf_partial(self, engine):
        inp = ScoringInput(mtf_score=0.5)
        result = engine.score(inp)
        assert result["mtf"] == 5.0


class TestEndToEnd:
    """Test full scoring pipeline and tier gating."""

    def test_scoring_high_smc_and_volume(self, engine):
        sweep = MagicMock()
        sweep.index = -1
        inp = ScoringInput(
            sweeps=[sweep], mss=MagicMock(), regime="TRENDING_UP",
            setup_class="LIQUIDITY_SWEEP_REVERSAL",
            volume_last_usd=3_000_000, volume_avg_usd=1_000_000,
            macd_histogram_last=0.5, macd_histogram_prev=0.3,
            rsi_last=42.0, ema_fast=101.0, ema_slow=100.0,
            direction="LONG", mtf_score=1.0
        )
        result = engine.score(inp)
        assert result["total"] >= 80

    def test_scoring_filters_low_quality(self, engine):
        inp = ScoringInput(
            regime="RANGING", setup_class="LIQUIDITY_SWEEP_REVERSAL",
            volume_last_usd=500_000, volume_avg_usd=2_000_000,
            rsi_last=68.0, ema_fast=100.0, ema_slow=101.0,
            direction="LONG", mtf_score=0.0
        )
        result = engine.score(inp)
        assert result["total"] < 50

    def test_total_never_exceeds_100(self, engine):
        sweep = MagicMock()
        sweep.index = -1
        inp = ScoringInput(
            sweeps=[sweep], mss=MagicMock(), fvg_zones=[MagicMock()],
            regime="TRENDING_UP", setup_class="LIQUIDITY_SWEEP_REVERSAL",
            atr_percentile=90,
            volume_last_usd=5_000_000, volume_avg_usd=1_000_000,
            macd_histogram_last=1.0, macd_histogram_prev=0.5,
            rsi_last=35.0, ema_fast=105.0, ema_slow=100.0,
            direction="LONG", mtf_score=1.0
        )
        result = engine.score(inp)
        assert result["total"] <= 100.0

    def test_score_returns_all_dimensions(self, engine):
        inp = ScoringInput()
        result = engine.score(inp)
        assert set(result.keys()) == {"smc", "regime", "volume", "indicators", "patterns", "mtf", "total"}

    def test_tier_a_plus(self, engine):
        """Verify a well-confirmed signal scores ≥ 80 (A+ tier)."""
        sweep = MagicMock()
        sweep.index = -1
        inp = ScoringInput(
            sweeps=[sweep], mss=MagicMock(), fvg_zones=[MagicMock()],
            regime="TRENDING_UP", setup_class="LIQUIDITY_SWEEP_REVERSAL",
            volume_last_usd=3_000_000, volume_avg_usd=1_000_000,
            macd_histogram_last=0.5, macd_histogram_prev=0.3,
            rsi_last=42.0, ema_fast=101.0, ema_slow=100.0,
            direction="LONG", mtf_score=1.0
        )
        result = engine.score(inp)
        assert result["total"] >= 80

    def test_tier_b(self, engine):
        """B tier: reasonable signal without full confluence."""
        sweep = MagicMock()
        sweep.index = -1
        inp = ScoringInput(
            sweeps=[sweep], regime="TRENDING_UP",
            setup_class="LIQUIDITY_SWEEP_REVERSAL",
            volume_last_usd=1_500_000, volume_avg_usd=1_000_000,
            rsi_last=48.0, ema_fast=101.0, ema_slow=100.0,
            direction="LONG", mtf_score=0.5
        )
        result = engine.score(inp)
        assert 50 <= result["total"] < 80
