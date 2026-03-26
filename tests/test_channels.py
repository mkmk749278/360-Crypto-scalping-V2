"""Tests for channel strategies – evaluate() logic."""

import numpy as np

from src.channels.scalp import ScalpChannel
from src.channels.swing import SwingChannel
from src.channels.spot import SpotChannel
from src.smc import Direction, LiquiditySweep, MSSSignal


def _make_candles(n=60, base=100.0, trend=0.1):
    """Create synthetic OHLCV data."""
    close = np.cumsum(np.ones(n) * trend) + base
    high = close + 0.5
    low = close - 0.5
    volume = np.ones(n) * 1000
    return {"open": close - 0.1, "high": high, "low": low, "close": close, "volume": volume}


def _make_indicators(adx_val=30, atr_val=0.5, ema9=101, ema21=100, ema200=95,
                      rsi_val=50, bb_upper=103, bb_mid=100, bb_lower=97, mom=0.5):
    return {
        "adx_last": adx_val,
        "atr_last": atr_val,
        "ema9_last": ema9,
        "ema21_last": ema21,
        "ema200_last": ema200,
        "rsi_last": rsi_val,
        "bb_upper_last": bb_upper,
        "bb_mid_last": bb_mid,
        "bb_lower_last": bb_lower,
        "momentum_last": mom,
    }


class TestScalpChannel:
    def test_signal_generated_on_valid_conditions(self):
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        sweep = LiquiditySweep(
            index=59, direction=Direction.LONG,
            sweep_level=99, close_price=99.05,
            wick_high=101, wick_low=98,
        )
        indicators = {"5m": _make_indicators(adx_val=30, mom=0.5, ema9=101, ema21=100)}
        smc_data = {"sweeps": [sweep]}

        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is not None
        assert sig.channel == "360_SCALP"
        assert sig.direction == Direction.LONG
        assert sig.entry > 0

    def test_no_signal_when_adx_low_standard_path(self):
        """Standard scalp path requires ADX >= 20; directly tests that path."""
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        sweep = LiquiditySweep(
            index=59, direction=Direction.LONG,
            sweep_level=99, close_price=99.05,
            wick_high=101, wick_low=98,
        )
        indicators = {"5m": _make_indicators(adx_val=10)}  # below 20
        smc_data = {"sweeps": [sweep]}
        # Standard path should return None (ADX too low), test it directly
        sig = ch._evaluate_standard("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_no_signal_without_sweep(self):
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        indicators = {"5m": _make_indicators()}
        sig = ch.evaluate("BTCUSDT", candles, indicators, {"sweeps": []}, 0.01, 10_000_000)
        assert sig is None

    def test_range_fade_long_signal_at_lower_bb(self):
        """RANGE_FADE path: price at lower BB + low RSI + low ADX → LONG signal."""
        ch = ScalpChannel()
        candles_data = _make_candles(60, base=100)
        candles_data["close"][-1] = 97.0  # at bb_lower
        candles = {"5m": candles_data}
        indicators = {"5m": _make_indicators(adx_val=15, bb_lower=97.1, rsi_val=28)}
        smc_data = {}

        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.setup_class == "RANGE_FADE"

    def test_range_fade_no_signal_when_adx_high(self):
        """RANGE_FADE path: ADX > 22 means trending → no range fade signal."""
        ch = ScalpChannel()
        candles_data = _make_candles(60, base=100)
        candles_data["close"][-1] = 97.0
        candles = {"5m": candles_data}
        indicators = {"5m": _make_indicators(adx_val=30, bb_lower=97.1, rsi_val=28, ema9=None, ema21=None)}
        smc_data = {}
        sig = ch._evaluate_range_fade("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_whale_momentum_long_signal_on_buy_flow(self):
        """WHALE_MOMENTUM path: strong buy tick flow → LONG signal."""
        ch = ScalpChannel()
        candles = {"1m": _make_candles(20)}
        indicators = {"1m": _make_indicators()}
        ticks = [
            {"price": 100.0, "qty": 15000, "isBuyerMaker": False},  # buy: $1.5M
            {"price": 100.0, "qty": 5000, "isBuyerMaker": True},    # sell: $0.5M
        ]
        smc_data = {
            "whale_alert": {"amount_usd": 1_500_000},
            "volume_delta_spike": True,
            "recent_ticks": ticks,
        }

        sig = ch.evaluate("ETHUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.setup_class == "WHALE_MOMENTUM"

    def test_whale_momentum_no_signal_without_whale(self):
        """WHALE_MOMENTUM path: no whale alert and no delta spike → no signal."""
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        indicators = {"5m": _make_indicators()}
        smc_data = {"whale_alert": None, "volume_delta_spike": False}
        sig = ch._evaluate_whale_momentum("ETHUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_whale_momentum_no_signal_when_flow_ambiguous(self):
        """Buy/sell ratio < 2× should return None (ambiguous flow)."""
        ch = ScalpChannel()
        candles = {"1m": _make_candles(20)}
        indicators = {"1m": _make_indicators()}
        ticks = [
            {"price": 100.0, "qty": 10000, "isBuyerMaker": False},  # buy: $1M
            {"price": 100.0, "qty": 8000, "isBuyerMaker": True},    # sell: $0.8M (ratio 1.25×)
        ]
        smc_data = {
            "whale_alert": {"amount_usd": 1_500_000},
            "volume_delta_spike": True,
            "recent_ticks": ticks,
        }
        sig = ch._evaluate_whale_momentum("ETHUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is None

    def test_range_fade_signal_has_dca_zone(self):
        """RANGE_FADE signals should include DCA zone fields."""
        ch = ScalpChannel()
        candles_data = _make_candles(60, base=100)
        candles_data["close"][-1] = 97.0
        candles = {"5m": candles_data}
        indicators = {"5m": _make_indicators(adx_val=15, bb_lower=97.1, rsi_val=28)}
        smc_data = {}

        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is not None
        assert sig.dca_zone_lower is not None and sig.dca_zone_lower > 0
        assert sig.dca_zone_upper is not None and sig.dca_zone_upper > 0
        assert sig.dca_zone_lower < sig.dca_zone_upper


class TestSwingChannel:
    def test_signal_with_sweep_and_mss(self):
        ch = SwingChannel()
        candles = {
            "4h": _make_candles(60, base=2300),
            "1h": _make_candles(60, base=2300),
        }
        sweep = LiquiditySweep(
            index=59, direction=Direction.LONG,
            sweep_level=2290, close_price=2291,
            wick_high=2360, wick_low=2285,
        )
        mss = MSSSignal(
            index=59, direction=Direction.LONG,
            midpoint=2322.5, confirm_close=2330,
        )
        indicators = {
            "4h": _make_indicators(adx_val=25, ema9=2320, ema21=2290, ema200=2200),
            "1h": _make_indicators(adx_val=25, ema200=2200, bb_lower=2290),
        }
        smc_data = {"sweeps": [sweep], "mss": mss}

        sig = ch.evaluate("ETHUSDT", candles, indicators, smc_data, 0.01, 50_000_000)
        assert sig is not None
        assert sig.channel == "360_SWING"

    def test_no_signal_without_mss(self):
        ch = SwingChannel()
        candles = {"4h": _make_candles(60), "1h": _make_candles(60)}
        indicators = {"4h": _make_indicators(adx_val=25), "1h": _make_indicators(adx_val=25)}
        sweep = LiquiditySweep(
            index=59, direction=Direction.LONG,
            sweep_level=99, close_price=99.05,
            wick_high=101, wick_low=98,
        )
        smc_data = {"sweeps": [sweep], "mss": None}
        sig = ch.evaluate("ETHUSDT", candles, indicators, smc_data, 0.01, 50_000_000)
        assert sig is None


class TestSpotChannel:
    def test_signal_on_h4_breakout(self):
        """SpotChannel should fire on H4 breakout above recent 20-bar high with volume."""
        ch = SpotChannel()
        # Build candles where last close breaks above recent high
        closes = np.cumsum(np.ones(60) * 0.1) + 100.0
        # Make last close clearly above the previous 19 bars' high
        closes[-1] = max(closes[-20:-1]) + 1.0
        highs = closes + 0.5
        lows = closes - 0.5
        # Volume: last bar is 2.0× the 19-bar average (above 1.8× expansion threshold)
        volumes = np.ones(60) * 1000.0
        volumes[-1] = volumes[:-1].mean() * 2.0
        candles_data = {
            "open": closes - 0.1,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
        candles = {"4h": candles_data}
        indicators = {"4h": _make_indicators(adx_val=20, ema200=90)}
        smc_data = {}

        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 5_000_000)
        assert sig is not None
        assert sig.channel == "360_SPOT"
        assert sig.direction == Direction.LONG

    def test_no_signal_without_h4_data(self):
        ch = SpotChannel()
        sig = ch.evaluate("BTCUSDT", {}, {}, {}, 0.01, 5_000_000)
        assert sig is None

    def test_no_signal_without_breakout(self):
        """Price below recent 20-bar high → no signal."""
        ch = SpotChannel()
        candles_data = _make_candles(60, base=100.0, trend=0.0)
        candles = {"4h": candles_data}
        indicators = {"4h": _make_indicators(adx_val=20, ema200=90)}
        smc_data = {}
        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 5_000_000)
        assert sig is None

    def test_no_signal_when_bearish_mss(self):
        """SpotChannel is LONG-biased; SHORT MSS should block the signal."""
        ch = SpotChannel()
        closes = np.cumsum(np.ones(60) * 0.1) + 100.0
        closes[-1] = max(closes[-20:-1]) + 1.0
        volumes = np.ones(60) * 1000.0
        volumes[-1] = volumes[:-1].mean() * 1.5
        candles_data = {
            "open": closes - 0.1,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": volumes,
        }
        candles = {"4h": candles_data}
        indicators = {"4h": _make_indicators(adx_val=20, ema200=90)}
        mss = MSSSignal(index=59, direction=Direction.SHORT, midpoint=99.0, confirm_close=98.0)
        smc_data = {"mss": mss}
        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 5_000_000)
        assert sig is None


# ---------------------------------------------------------------------------
# PR_10 refactor verification tests
# ---------------------------------------------------------------------------

def test_volume_expansion_returns_false_when_below_threshold():
    from src.filters import check_volume_expansion
    volumes = [1000.0] * 10 + [800.0]   # Last candle is below average
    closes  = [100.0] * 11
    assert not check_volume_expansion(volumes, closes, lookback=9, multiplier=1.8)


def test_volume_expansion_returns_true_when_above():
    from src.filters import check_volume_expansion
    volumes = [1000.0] * 10 + [2500.0]  # Last candle is 2.5× average
    closes  = [100.0] * 11
    assert check_volume_expansion(volumes, closes, lookback=9, multiplier=1.8)


def test_scalp_channel_no_calc_levels_method():
    """After refactor, ScalpChannel should not have _calc_levels."""
    from src.channels.scalp import ScalpChannel
    ch = ScalpChannel()
    assert not hasattr(ch, "_calc_levels"), \
        "_calc_levels should be removed; TP is computed by build_channel_signal()"

