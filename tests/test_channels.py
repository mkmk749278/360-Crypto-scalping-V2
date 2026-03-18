"""Tests for channel strategies – evaluate() logic."""

import numpy as np

from src.channels.scalp import ScalpChannel
from src.channels.swing import SwingChannel
from src.channels.range_channel import RangeChannel
from src.channels.tape import TapeChannel
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
        ai = {"label": "Positive", "summary": "Whale activity", "score": 0.5}

        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, ai, 0.01, 10_000_000)
        assert sig is not None
        assert sig.channel == "360_SCALP"
        assert sig.direction == Direction.LONG
        assert sig.entry > 0

    def test_no_signal_when_adx_low(self):
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        sweep = LiquiditySweep(
            index=59, direction=Direction.LONG,
            sweep_level=99, close_price=99.05,
            wick_high=101, wick_low=98,
        )
        indicators = {"5m": _make_indicators(adx_val=10)}  # below 20
        smc_data = {"sweeps": [sweep]}
        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, {}, 0.01, 10_000_000)
        assert sig is None

    def test_no_signal_without_sweep(self):
        ch = ScalpChannel()
        candles = {"5m": _make_candles(60)}
        indicators = {"5m": _make_indicators()}
        sig = ch.evaluate("BTCUSDT", candles, indicators, {"sweeps": []}, {}, 0.01, 10_000_000)
        assert sig is None


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
            "4h": _make_indicators(adx_val=25, ema200=2200),
            "1h": _make_indicators(adx_val=25, ema200=2200, bb_lower=2290),
        }
        smc_data = {"sweeps": [sweep], "mss": mss}

        sig = ch.evaluate("ETHUSDT", candles, indicators, smc_data, {}, 0.01, 50_000_000)
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
        sig = ch.evaluate("ETHUSDT", candles, indicators, smc_data, {}, 0.01, 50_000_000)
        assert sig is None


class TestRangeChannel:
    def test_long_signal_at_lower_bb(self):
        ch = RangeChannel()
        # Close at lower BB
        candles_data = _make_candles(60, base=100)
        candles_data["close"][-1] = 97.0  # at bb_lower
        candles = {"15m": candles_data}
        indicators = {"15m": _make_indicators(adx_val=15, bb_lower=97.1, rsi_val=28)}
        smc_data = {}

        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, {}, 0.01, 10_000_000)
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_no_signal_when_adx_high(self):
        ch = RangeChannel()
        candles = {"15m": _make_candles(60)}
        indicators = {"15m": _make_indicators(adx_val=30)}
        sig = ch.evaluate("BTCUSDT", candles, indicators, {}, {}, 0.01, 10_000_000)
        assert sig is None


class TestTapeChannel:
    def test_signal_on_whale_alert(self):
        ch = TapeChannel()
        candles = {"1m": _make_candles(20)}
        indicators = {"1m": _make_indicators()}
        ticks = [
            {"price": 100.0, "qty": 15000, "isBuyerMaker": False},  # buy
            {"price": 100.0, "qty": 5000, "isBuyerMaker": True},    # sell
        ]
        smc_data = {
            "whale_alert": {"amount_usd": 1_500_000},
            "volume_delta_spike": True,
            "recent_ticks": ticks,
        }
        ai = {"label": "Bullish", "summary": "Whale confirmed"}

        sig = ch.evaluate("ETHUSDT", candles, indicators, smc_data, ai, 0.01, 50_000_000)
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.channel == "360_THE_TAPE"

    def test_no_signal_without_whale(self):
        ch = TapeChannel()
        candles = {"1m": _make_candles(20)}
        indicators = {"1m": _make_indicators()}
        smc_data = {"whale_alert": None, "volume_delta_spike": False}
        sig = ch.evaluate("ETHUSDT", candles, indicators, smc_data, {}, 0.01, 50_000_000)
        assert sig is None
