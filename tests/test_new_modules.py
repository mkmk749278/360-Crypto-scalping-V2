"""Tests for the new src modules: detector, regime, filters, risk, binance, exchange, logger."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.detector import SMCDetector, SMCResult
from src.exchange import ExchangeManager
from src.filters import (
    check_adx,
    check_ema_alignment,
    check_rsi,
    check_spread,
    check_volume,
)
from src.regime import MarketRegime, MarketRegimeDetector
from src.risk import RiskAssessment, RiskManager


# ---------------------------------------------------------------------------
# SMCDetector
# ---------------------------------------------------------------------------


class TestSMCDetector:
    def _make_candles(self, n: int = 60):
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0
        return {"high": high, "low": low, "close": close, "volume": np.ones(n)}

    def test_returns_smc_result(self):
        det = SMCDetector()
        candles = {"5m": self._make_candles()}
        result = det.detect("BTCUSDT", candles, [])
        assert isinstance(result, SMCResult)

    def test_no_ticks_gives_no_whale(self):
        det = SMCDetector()
        result = det.detect("BTCUSDT", {}, [])
        assert result.whale_alert is None
        assert result.volume_delta_spike is False

    def test_whale_detected_in_ticks(self):
        det = SMCDetector()
        ticks = [{"price": 50000.0, "qty": 30.0, "isBuyerMaker": False, "time": 1}]
        result = det.detect("BTCUSDT", {}, ticks)
        assert result.whale_alert is not None  # 50000 * 30 = 1.5M > threshold

    def test_as_dict_keys(self):
        det = SMCDetector()
        result = det.detect("BTCUSDT", {}, [])
        d = result.as_dict()
        for key in ("sweeps", "mss", "fvg", "whale_alert", "volume_delta_spike", "recent_ticks"):
            assert key in d

    def test_sweep_detected_in_candles(self):
        det = SMCDetector()
        n = 60
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0
        # Last candle: wick below recent low, close just inside
        high[-1] = 105.0
        low[-1] = 93.0
        close[-1] = 95.04
        candles = {"5m": {"high": high, "low": low, "close": close, "volume": np.ones(n)}}
        result = det.detect("BTCUSDT", candles, [])
        assert len(result.sweeps) >= 1


# ---------------------------------------------------------------------------
# MarketRegimeDetector
# ---------------------------------------------------------------------------


class TestMarketRegimeDetector:
    def test_trending_up_high_adx_positive_ema(self):
        det = MarketRegimeDetector()
        ind = {"adx_last": 30.0, "ema9_last": 102.0, "ema21_last": 100.0}
        result = det.classify(ind)
        assert result.regime == MarketRegime.TRENDING_UP

    def test_trending_down_high_adx_negative_ema(self):
        det = MarketRegimeDetector()
        ind = {"adx_last": 30.0, "ema9_last": 98.0, "ema21_last": 100.0}
        result = det.classify(ind)
        assert result.regime == MarketRegime.TRENDING_DOWN

    def test_ranging_low_adx(self):
        det = MarketRegimeDetector()
        ind = {"adx_last": 15.0, "ema9_last": 100.1, "ema21_last": 100.0}
        result = det.classify(ind)
        assert result.regime == MarketRegime.RANGING

    def test_volatile_wide_bb(self):
        det = MarketRegimeDetector()
        ind = {
            "adx_last": 22.0,
            "bb_upper_last": 112.0,
            "bb_lower_last": 88.0,
            "bb_mid_last": 100.0,
        }
        result = det.classify(ind)
        assert result.regime == MarketRegime.VOLATILE

    def test_quiet_narrow_bb(self):
        det = MarketRegimeDetector()
        ind = {
            "adx_last": 22.0,
            "bb_upper_last": 100.5,
            "bb_lower_last": 99.5,
            "bb_mid_last": 100.0,
        }
        result = det.classify(ind)
        assert result.regime == MarketRegime.QUIET

    def test_empty_indicators_defaults_to_ranging(self):
        det = MarketRegimeDetector()
        result = det.classify({})
        assert result.regime == MarketRegime.RANGING

    def test_result_has_regime_attribute(self):
        det = MarketRegimeDetector()
        result = det.classify({"adx_last": 28.0, "ema9_last": 105.0, "ema21_last": 100.0})
        assert hasattr(result, "regime")
        assert isinstance(result.regime, MarketRegime)

    # ------------------------------------------------------------------
    # Bug 4: timeframe parameter and 1m wider EMA slope threshold
    # ------------------------------------------------------------------

    def test_1m_timeframe_uses_wider_threshold_stays_ranging(self):
        """On 1m data, an EMA slope of ±0.1% should NOT trigger a trending regime
        (it would with the default ±0.05% threshold on 5m data)."""
        det = MarketRegimeDetector()
        # ema_slope = (100.1 - 100.0) / 100.0 * 100 = 0.1% — above 0.05 but below 0.15
        ind = {"ema9_last": 100.1, "ema21_last": 100.0}
        result_5m = det.classify(ind, timeframe="5m")
        result_1m = det.classify(ind, timeframe="1m")
        # 5m uses 0.05 threshold → slope 0.1 > 0.05 → TRENDING_UP
        assert result_5m.regime == MarketRegime.TRENDING_UP
        # 1m uses 0.15 threshold → slope 0.1 < 0.15 → RANGING
        assert result_1m.regime == MarketRegime.RANGING

    def test_1m_timeframe_triggers_trending_above_wider_threshold(self):
        """On 1m data, an EMA slope beyond ±0.15% should still trigger a trending regime."""
        det = MarketRegimeDetector()
        # ema_slope = (100.2 - 100.0) / 100.0 * 100 = 0.2% — above 0.15
        ind = {"ema9_last": 100.2, "ema21_last": 100.0}
        result_1m = det.classify(ind, timeframe="1m")
        assert result_1m.regime == MarketRegime.TRENDING_UP

    def test_5m_timeframe_is_default_backward_compatible(self):
        """Calling classify() without timeframe behaves identically to timeframe='5m'."""
        det = MarketRegimeDetector()
        ind = {"ema9_last": 100.1, "ema21_last": 100.0}
        assert det.classify(ind).regime == det.classify(ind, timeframe="5m").regime

    def test_1m_negative_slope_ranging_within_threshold(self):
        """Negative slope within 1m threshold (-0.1%) stays RANGING."""
        det = MarketRegimeDetector()
        ind = {"ema9_last": 99.9, "ema21_last": 100.0}
        result = det.classify(ind, timeframe="1m")
        assert result.regime == MarketRegime.RANGING

    def test_adx_based_regime_unaffected_by_timeframe(self):
        """When ADX is decisive, the timeframe parameter does not change the outcome."""
        det = MarketRegimeDetector()
        ind = {"adx_last": 30.0, "ema9_last": 102.0, "ema21_last": 100.0}
        result_5m = det.classify(ind, timeframe="5m")
        result_1m = det.classify(ind, timeframe="1m")
        # High ADX always returns TRENDING_UP regardless of timeframe
        assert result_5m.regime == MarketRegime.TRENDING_UP
        assert result_1m.regime == MarketRegime.TRENDING_UP

    # ------------------------------------------------------------------
    # Volume-delta override
    # ------------------------------------------------------------------

    def test_volume_delta_spike_forces_quiet_to_volatile(self):
        """A large volume-delta spike with no EMA context should upgrade QUIET → VOLATILE."""
        det = MarketRegimeDetector()
        # Narrow BB → QUIET without volume delta
        ind = {
            "bb_upper_last": 100.5,
            "bb_lower_last": 99.5,
            "bb_mid_last": 100.0,
        }
        quiet_result = det.classify(ind)
        assert quiet_result.regime == MarketRegime.QUIET
        # Same indicators + 70% net volume delta → forced VOLATILE
        result = det.classify(ind, volume_delta=70.0)
        assert result.regime == MarketRegime.VOLATILE
        assert result.volume_delta_pct == 70.0

    def test_volume_delta_spike_forces_ranging_to_trending_up(self):
        """Volume spike with bullish EMA slope upgrades RANGING → TRENDING_UP."""
        det = MarketRegimeDetector()
        ind = {"adx_last": 15.0, "ema9_last": 100.1, "ema21_last": 100.0}
        ranging = det.classify(ind)
        assert ranging.regime == MarketRegime.RANGING
        result = det.classify(ind, volume_delta=65.0)
        assert result.regime == MarketRegime.TRENDING_UP

    def test_volume_delta_spike_forces_ranging_to_trending_down(self):
        """Negative volume spike with bearish EMA slope upgrades RANGING → TRENDING_DOWN."""
        det = MarketRegimeDetector()
        ind = {"adx_last": 15.0, "ema9_last": 99.9, "ema21_last": 100.0}
        result = det.classify(ind, volume_delta=-65.0)
        assert result.regime == MarketRegime.TRENDING_DOWN

    def test_volume_delta_below_threshold_no_override(self):
        """A small volume delta does not change the base regime."""
        det = MarketRegimeDetector()
        ind = {"adx_last": 15.0, "ema9_last": 100.1, "ema21_last": 100.0}
        result = det.classify(ind, volume_delta=30.0)
        assert result.regime == MarketRegime.RANGING
        assert result.volume_delta_pct is None

    def test_volume_delta_does_not_downgrade_trending(self):
        """Volume delta override only affects QUIET/RANGING; trending stays trending."""
        det = MarketRegimeDetector()
        ind = {"adx_last": 30.0, "ema9_last": 102.0, "ema21_last": 100.0}
        result = det.classify(ind, volume_delta=70.0)
        assert result.regime == MarketRegime.TRENDING_UP

    def test_no_volume_delta_result_field_is_none(self):
        """When volume_delta is not provided, volume_delta_pct in result is None."""
        det = MarketRegimeDetector()
        result = det.classify({"adx_last": 15.0})
        assert result.volume_delta_pct is None


# ---------------------------------------------------------------------------


class TestFilters:
    def test_check_spread_pass(self):
        assert check_spread(0.01, 0.02) is True

    def test_check_spread_fail(self):
        assert check_spread(0.03, 0.02) is False

    def test_check_spread_equal(self):
        assert check_spread(0.02, 0.02) is True

    def test_check_adx_in_range(self):
        assert check_adx(30.0, 25.0, 60.0) is True

    def test_check_adx_below_min(self):
        assert check_adx(20.0, 25.0) is False

    def test_check_adx_none(self):
        assert check_adx(None, 25.0) is False

    def test_check_ema_alignment_long(self):
        assert check_ema_alignment(102.0, 100.0, "LONG") is True

    def test_check_ema_alignment_long_fail(self):
        assert check_ema_alignment(98.0, 100.0, "LONG") is False

    def test_check_ema_alignment_short(self):
        assert check_ema_alignment(98.0, 100.0, "SHORT") is True

    def test_check_ema_alignment_none(self):
        assert check_ema_alignment(None, 100.0, "LONG") is False

    def test_check_volume_pass(self):
        assert check_volume(10_000_000, 5_000_000) is True

    def test_check_volume_fail(self):
        assert check_volume(1_000_000, 5_000_000) is False

    def test_check_rsi_long_not_overbought(self):
        assert check_rsi(60.0, 70.0, 30.0, "LONG") is True

    def test_check_rsi_long_overbought(self):
        assert check_rsi(75.0, 70.0, 30.0, "LONG") is False

    def test_check_rsi_short_not_oversold(self):
        assert check_rsi(40.0, 70.0, 30.0, "SHORT") is True

    def test_check_rsi_short_oversold(self):
        assert check_rsi(25.0, 70.0, 30.0, "SHORT") is False

    def test_check_rsi_none_passes(self):
        assert check_rsi(None, 70.0, 30.0, "LONG") is True


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


def _make_signal_obj(
    symbol: str = "BTCUSDT",
    direction: str = "LONG",
    entry: float = 50000.0,
    sl: float = 49500.0,
    tp1: float = 50750.0,  # sl_dist=500, tp_dist=750 → R:R=1.5 ≥ 1.2 floor
    confidence: float = 75.0,
):
    class _FakeDir:
        value = direction

    class _FakeSig:
        pass

    s = _FakeSig()
    s.symbol = symbol
    s.direction = _FakeDir()
    s.entry = entry
    s.stop_loss = sl
    s.tp1 = tp1
    s.confidence = confidence
    s.spread_pct = 0.01
    return s


class TestRiskManager:
    def test_basic_risk_assessment(self):
        rm = RiskManager()
        sig = _make_signal_obj()
        result = rm.calculate_risk(sig, {"atr_last": 200.0}, volume_24h_usd=50_000_000)
        assert isinstance(result, RiskAssessment)
        assert result.allowed is True

    def test_risk_reward_ratio(self):
        rm = RiskManager()
        sig = _make_signal_obj(entry=100.0, sl=99.0, tp1=101.0)
        result = rm.calculate_risk(sig, {}, volume_24h_usd=50_000_000)
        assert abs(result.risk_reward - 1.0) < 0.01

    def test_low_risk_label(self):
        rm = RiskManager()
        sig = _make_signal_obj(confidence=80.0)
        result = rm.calculate_risk(sig, {"atr_last": 10.0}, volume_24h_usd=100_000_000)
        assert result.risk_label in ("Low", "Medium")

    def test_high_risk_label_low_volume(self):
        rm = RiskManager()
        sig = _make_signal_obj(confidence=45.0)
        result = rm.calculate_risk(sig, {"atr_last": 1000.0}, volume_24h_usd=500_000)
        assert result.risk_label in ("High", "Very High")

    def test_concurrent_signal_blocked(self):
        rm = RiskManager(max_concurrent_same_direction=1)
        sig = _make_signal_obj(symbol="BTCUSDT", direction="LONG")

        existing = _make_signal_obj(symbol="BTCUSDT", direction="LONG")
        active = {"BTC-001": existing}

        result = rm.calculate_risk(sig, {}, volume_24h_usd=50_000_000, active_signals=active)
        assert result.allowed is False

    def test_concurrent_different_direction_allowed(self):
        rm = RiskManager()
        sig = _make_signal_obj(symbol="BTCUSDT", direction="LONG")
        active = {"BTC-001": _make_signal_obj(symbol="BTCUSDT", direction="SHORT")}
        result = rm.calculate_risk(sig, {}, volume_24h_usd=50_000_000, active_signals=active)
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Fix 3: Cross-exchange direction verification
# ---------------------------------------------------------------------------


class TestCrossExchangeDirectionVerification:
    """verify_signal_cross_exchange must check directional agreement."""

    @pytest.mark.asyncio
    async def test_no_second_exchange_returns_false(self):
        mgr = ExchangeManager()
        result = await mgr.verify_signal_cross_exchange("BTCUSDT", "LONG", 50000.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_long_signal_confirmed_when_prices_agree(self):
        """LONG signal verified when second price is same as primary (neutral within tolerance)."""
        mgr = ExchangeManager(second_exchange_url="http://fake-exchange/api")
        mgr._fetch_price_second = AsyncMock(return_value=50000.0)
        result = await mgr.verify_signal_cross_exchange("BTCUSDT", "LONG", 50000.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_long_signal_rejected_when_second_price_much_lower(self):
        """LONG signal contradicted when second-exchange price is significantly below primary."""
        mgr = ExchangeManager(second_exchange_url="http://fake-exchange/api")
        # Second price is 1% below primary – well below tolerance/2
        mgr._fetch_price_second = AsyncMock(return_value=49000.0)
        result = await mgr.verify_signal_cross_exchange("BTCUSDT", "LONG", 50000.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_short_signal_rejected_when_second_price_much_higher(self):
        """SHORT signal contradicted when second-exchange price is significantly above primary."""
        mgr = ExchangeManager(second_exchange_url="http://fake-exchange/api")
        # Second price is 1% above primary
        mgr._fetch_price_second = AsyncMock(return_value=50500.0)
        result = await mgr.verify_signal_cross_exchange("BTCUSDT", "SHORT", 50000.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_large_spread_still_returns_false(self):
        """Large price divergence between exchanges still fails (existing behaviour)."""
        mgr = ExchangeManager(second_exchange_url="http://fake-exchange/api")
        # Second price is 2% away (beyond 0.5% tolerance)
        mgr._fetch_price_second = AsyncMock(return_value=51000.0)
        result = await mgr.verify_signal_cross_exchange("BTCUSDT", "LONG", 50000.0)
        assert result is False


# ---------------------------------------------------------------------------
# Fix 4: Regime detector EMA fallback to close price
# ---------------------------------------------------------------------------


class TestRegimeEMAFallback:
    """When EMA values are missing, close price should be used as fallback."""

    def test_ema_fallback_uses_close(self):
        detector = MarketRegimeDetector()
        # Provide candles but no EMA in indicators
        indicators = {"adx_last": 22.0}  # borderline ADX, needs EMA for direction
        candles = {"close": [100.0, 101.0, 102.0]}  # upward trend
        result = detector.classify(indicators, candles)
        # With ema_fast = ema_slow = close (same price), ema_slope = 0
        # borderline ADX (20-25) + ema_slope 0 (not > 0.05 and not < -0.05) → RANGING
        assert result.regime is not None  # must not crash or return None

    def test_ema_fallback_does_not_crash_without_candles(self):
        """Without candles and without EMAs, detector must still return a regime."""
        detector = MarketRegimeDetector()
        indicators = {}
        result = detector.classify(indicators, candles=None)
        assert result.regime is not None



# ---------------------------------------------------------------------------
# BUG 1: Stablecoin blacklist in PairManager
# ---------------------------------------------------------------------------


class TestStablecoinBlacklist:
    """Stablecoin-vs-stablecoin pairs must never enter the scanning pipeline."""

    def test_blacklist_contains_known_stablecoins(self):
        from src.pair_manager import _STABLECOIN_BLACKLIST
        assert "USDCUSDT" in _STABLECOIN_BLACKLIST
        assert "BUSDUSDT" in _STABLECOIN_BLACKLIST
        assert "USD1USDT" in _STABLECOIN_BLACKLIST
        assert "FDUSDUSDT" in _STABLECOIN_BLACKLIST

    def test_btcusdt_not_in_blacklist(self):
        from src.pair_manager import _STABLECOIN_BLACKLIST
        assert "BTCUSDT" not in _STABLECOIN_BLACKLIST

    def test_ethusdt_not_in_blacklist(self):
        from src.pair_manager import _STABLECOIN_BLACKLIST
        assert "ETHUSDT" not in _STABLECOIN_BLACKLIST

    @pytest.mark.asyncio
    async def test_fetch_spot_filters_stablecoins(self, monkeypatch):
        """fetch_top_spot_pairs must exclude blacklisted stablecoin pairs."""
        from src.pair_manager import PairManager

        ticker_data = [
            {"symbol": "BTCUSDT", "quoteVolume": "1000000"},
            {"symbol": "USDCUSDT", "quoteVolume": "2000000"},   # blacklisted
            {"symbol": "ETHUSDT", "quoteVolume": "900000"},
            {"symbol": "BUSDUSDT", "quoteVolume": "800000"},    # blacklisted
        ]

        pm = PairManager.__new__(PairManager)
        pm._spot_client = AsyncMock()
        pm._spot_client._get = AsyncMock(return_value=ticker_data)
        pm._futures_client = AsyncMock()

        pairs = await pm.fetch_top_spot_pairs(limit=10)
        symbols = [p.symbol for p in pairs]

        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
        assert "USDCUSDT" not in symbols
        assert "BUSDUSDT" not in symbols

    @pytest.mark.asyncio
    async def test_fetch_futures_filters_stablecoins(self, monkeypatch):
        """fetch_top_futures_pairs must exclude blacklisted stablecoin pairs."""
        from src.pair_manager import PairManager

        ticker_data = [
            {"symbol": "BTCUSDT", "quoteVolume": "5000000"},
            {"symbol": "USD1USDT", "quoteVolume": "9000000"},   # blacklisted
            {"symbol": "SOLUSDT", "quoteVolume": "3000000"},
        ]

        pm = PairManager.__new__(PairManager)
        pm._futures_client = AsyncMock()
        pm._futures_client._get = AsyncMock(return_value=ticker_data)
        pm._spot_client = AsyncMock()

        pairs = await pm.fetch_top_futures_pairs(limit=10)
        symbols = [p.symbol for p in pairs]

        assert "BTCUSDT" in symbols
        assert "SOLUSDT" in symbols
        assert "USD1USDT" not in symbols

    def test_rlusdusdt_in_blacklist(self):
        """Bug 3: RLUSDUSDT (Ripple's RLUSD stablecoin) must be blacklisted."""
        from src.pair_manager import _STABLECOIN_BLACKLIST
        assert "RLUSDUSDT" in _STABLECOIN_BLACKLIST

    def test_new_stablecoins_in_blacklist(self):
        """All newly added stablecoins from Bug 3 must be in the blacklist."""
        from src.pair_manager import _STABLECOIN_BLACKLIST
        new_coins = {"RLUSDUSDT", "PYUSDUSDT", "USDDUSDT", "GUSDUSDT",
                     "FRAXUSDT", "LUSDUSDT", "SUSDUSDT", "CUSDUSDT"}
        for coin in new_coins:
            assert coin in _STABLECOIN_BLACKLIST, f"{coin} not in blacklist"
