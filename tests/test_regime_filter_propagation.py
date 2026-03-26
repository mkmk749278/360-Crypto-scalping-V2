"""Tests verifying that SwingChannel and SpotChannel use regime-aware filters."""

import numpy as np


class TestSwingChannelRegimeFilters:
    def test_swing_uses_regime_aware_adx(self):
        """SwingChannel should accept lower ADX in QUIET regime via check_adx_regime."""
        from src.channels.swing import SwingChannel
        from src.smc import LiquiditySweep, MSSSignal, Direction

        ch = SwingChannel()
        n = 60
        base = 100.0
        close = np.cumsum(np.ones(n) * 0.1) + base
        candles_arr = {
            "open": close - 0.05,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(n) * 1000,
        }
        candles = {"4h": candles_arr, "1h": candles_arr}
        mss = MSSSignal(
            index=59, direction=Direction.LONG,
            midpoint=close[-1], confirm_close=close[-1],
        )
        sweep = LiquiditySweep(
            index=59, direction=Direction.LONG,
            sweep_level=99, close_price=99.05,
            wick_high=101, wick_low=98,
        )
        # ADX 14 — below default min (20) but above QUIET min (12)
        indicators = {
            "4h": {"adx_last": 14, "atr_last": 1.0},
            "1h": {
                "adx_last": 14,
                "atr_last": 0.5,
                "ema200_last": close[-1] - 5,
                "bb_upper_last": close[-1] + 5,
                "bb_lower_last": close[-1] - 0.2,
            },
        }
        smc_data = {"sweeps": [sweep], "mss": mss}

        # Without regime → ADX 14 < 20 → should fail (return None)
        ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 15_000_000)
        # With QUIET regime → ADX 14 > 12 → may pass ADX filter
        ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 15_000_000, regime="QUIET")


class TestSpotChannelRegimeFilters:
    def test_spot_uses_regime_aware_rsi(self):
        """SpotChannel should use check_rsi_regime instead of inline RSI checks."""
        from src.channels import spot
        import inspect
        source = inspect.getsource(spot)
        assert "check_rsi_regime" in source
        # Inline threshold checks should have been replaced by check_rsi_regime
        assert "rsi_last > 75" not in source
        assert "rsi_last < 25" not in source

    def test_spot_uses_adaptive_spread(self):
        """SpotChannel should import and use check_spread_adaptive."""
        from src.channels import spot
        import inspect
        source = inspect.getsource(spot)
        assert "check_spread_adaptive" in source
