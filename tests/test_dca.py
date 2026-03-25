"""Comprehensive tests for the DCA (Double Entry) system.

Tests cover:
- compute_dca_zone — correct bounds for LONG and SHORT
- recalculate_after_dca — avg_entry, TP1/2/3 recalc, SL unchanged
- check_dca_entry — zone check, momentum, structure, already-filled guard
- Breakeven and R:R properties
- Weight variations
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config import CHANNEL_SWING, CHANNEL_SCALP
from src.channels.base import Signal
from src.dca import check_dca_entry, compute_dca_zone, recalculate_after_dca
from src.smc import Direction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_long_signal(
    entry: float = 2300.0,
    stop_loss: float = 2280.0,
    tp1: float = 2330.0,
    tp2: float = 2360.0,
    tp3: float = 2400.0,
    channel: str = "360_SWING",
) -> Signal:
    """Create a LONG swing signal for DCA testing."""
    sl_dist = entry - stop_loss
    sig = Signal(
        channel=channel,
        symbol="ETHUSDT",
        direction=Direction.LONG,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        signal_id="TEST-LONG-001",
        original_sl_distance=sl_dist,
    )
    # Pre-populate DCA zone (30–70 % of SL distance below entry)
    sig.dca_zone_lower = entry - 0.70 * sl_dist
    sig.dca_zone_upper = entry - 0.30 * sl_dist
    sig.original_entry = entry
    sig.original_tp1 = tp1
    sig.original_tp2 = tp2
    sig.original_tp3 = tp3
    return sig


def _make_short_signal(
    entry: float = 2300.0,
    stop_loss: float = 2320.0,
    tp1: float = 2270.0,
    tp2: float = 2240.0,
    tp3: float = 2200.0,
    channel: str = "360_SWING",
) -> Signal:
    """Create a SHORT swing signal for DCA testing."""
    sl_dist = stop_loss - entry
    sig = Signal(
        channel=channel,
        symbol="ETHUSDT",
        direction=Direction.SHORT,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        signal_id="TEST-SHORT-001",
        original_sl_distance=sl_dist,
    )
    # Pre-populate DCA zone (30–70 % of SL distance above entry)
    sig.dca_zone_lower = entry + 0.30 * sl_dist
    sig.dca_zone_upper = entry + 0.70 * sl_dist
    sig.original_entry = entry
    sig.original_tp1 = tp1
    sig.original_tp2 = tp2
    sig.original_tp3 = tp3
    return sig


# ---------------------------------------------------------------------------
# compute_dca_zone tests
# ---------------------------------------------------------------------------

class TestComputeDcaZone:
    def test_compute_dca_zone_long(self):
        """LONG zone must be below entry, within 30–70 % of SL dist."""
        entry = 100.0
        sl = 90.0
        sl_dist = 10.0

        lower, upper = compute_dca_zone(entry, sl, Direction.LONG)

        assert upper == pytest.approx(entry - 0.30 * sl_dist)  # 97.0
        assert lower == pytest.approx(entry - 0.70 * sl_dist)  # 93.0
        # Zone must be below entry and above SL
        assert lower < upper < entry
        assert lower > sl

    def test_compute_dca_zone_short(self):
        """SHORT zone must be above entry, within 30–70 % of SL dist."""
        entry = 100.0
        sl = 110.0
        sl_dist = 10.0

        lower, upper = compute_dca_zone(entry, sl, Direction.SHORT)

        assert lower == pytest.approx(entry + 0.30 * sl_dist)  # 103.0
        assert upper == pytest.approx(entry + 0.70 * sl_dist)  # 107.0
        # Zone must be above entry and below SL
        assert entry < lower < upper
        assert upper < sl

    def test_compute_dca_zone_custom_range(self):
        """Custom zone_range parameters are respected."""
        lower, upper = compute_dca_zone(100.0, 80.0, Direction.LONG, (0.20, 0.50))
        assert lower == pytest.approx(100.0 - 0.50 * 20.0)  # 90.0
        assert upper == pytest.approx(100.0 - 0.20 * 20.0)  # 96.0


# ---------------------------------------------------------------------------
# recalculate_after_dca tests
# ---------------------------------------------------------------------------

class TestRecalculateAfterDca:
    def test_recalculate_after_dca_long(self):
        """LONG: avg_entry and new TPs calculated correctly; SL unchanged."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        entry_2 = 2294.0  # within DCA zone

        recalculate_after_dca(sig, entry_2, list(CHANNEL_SWING.tp_ratios))

        # Weighted average: 2300 * 0.6 + 2294 * 0.4 = 1380 + 917.6 = 2297.6
        assert sig.avg_entry == pytest.approx(2297.6, abs=1e-4)
        assert sig.entry == pytest.approx(2297.6, abs=1e-4)

        # SL must be unchanged
        assert sig.stop_loss == pytest.approx(2280.0)

        # New SL distance from avg_entry to original SL
        new_sl_dist = 2297.6 - 2280.0  # 17.6
        assert sig.tp1 == pytest.approx(2297.6 + 1.5 * new_sl_dist, abs=1e-4)
        assert sig.tp2 == pytest.approx(2297.6 + 3.0 * new_sl_dist, abs=1e-4)
        assert sig.tp3 == pytest.approx(2297.6 + 5.0 * new_sl_dist, abs=1e-4)

    def test_recalculate_after_dca_short(self):
        """SHORT: avg_entry and new TPs calculated correctly; SL unchanged."""
        sig = _make_short_signal(entry=2300.0, stop_loss=2320.0)
        entry_2 = 2306.0  # within DCA zone (above entry for SHORT)

        recalculate_after_dca(sig, entry_2, list(CHANNEL_SWING.tp_ratios))

        # Weighted average: 2300 * 0.6 + 2306 * 0.4 = 1380 + 922.4 = 2302.4
        assert sig.avg_entry == pytest.approx(2302.4, abs=1e-4)
        assert sig.entry == pytest.approx(2302.4, abs=1e-4)

        # SL must be unchanged
        assert sig.stop_loss == pytest.approx(2320.0)

        # New SL distance from avg_entry to original SL
        new_sl_dist = 2320.0 - 2302.4  # 17.6
        assert sig.tp1 == pytest.approx(2302.4 - 1.5 * new_sl_dist, abs=1e-4)
        assert sig.tp2 == pytest.approx(2302.4 - 3.0 * new_sl_dist, abs=1e-4)
        assert sig.tp3 == pytest.approx(2302.4 - 5.0 * new_sl_dist, abs=1e-4)

    def test_sl_unchanged_after_dca(self):
        """Stop-loss must NOT be modified by recalculate_after_dca."""
        sig = _make_long_signal(entry=100.0, stop_loss=90.0)
        original_sl = sig.stop_loss
        recalculate_after_dca(sig, 95.0, [1.0, 2.0, 3.0])
        assert sig.stop_loss == original_sl

    def test_r_ratio_preserved(self):
        """R:R multiples for each TP must be preserved after DCA recalc."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        tp_ratios = list(CHANNEL_SWING.tp_ratios)  # [1.5, 3.0, 5.0]

        recalculate_after_dca(sig, 2294.0, tp_ratios)

        new_sl_dist = sig.avg_entry - sig.stop_loss
        assert new_sl_dist > 0

        r1 = (sig.tp1 - sig.avg_entry) / new_sl_dist
        r2 = (sig.tp2 - sig.avg_entry) / new_sl_dist
        r3 = (sig.tp3 - sig.avg_entry) / new_sl_dist

        assert r1 == pytest.approx(tp_ratios[0], abs=1e-6)
        assert r2 == pytest.approx(tp_ratios[1], abs=1e-6)
        assert r3 == pytest.approx(tp_ratios[2], abs=1e-6)

    def test_tps_closer_after_dca(self):
        """All TPs must move closer to the new avg_entry (easier to hit)."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        old_tp1, old_tp2, old_tp3 = sig.tp1, sig.tp2, sig.tp3

        recalculate_after_dca(sig, 2294.0, list(CHANNEL_SWING.tp_ratios))

        # New TPs should be lower than original (closer to current price)
        assert sig.tp1 < old_tp1
        assert sig.tp2 < old_tp2
        assert sig.tp3 is not None and old_tp3 is not None and sig.tp3 < old_tp3

    def test_original_values_preserved(self):
        """original_entry, original_tp1/2/3 must be stored before modification."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        # Already set by _make_long_signal; verify they're not overwritten
        expected_orig_tp1 = sig.original_tp1

        recalculate_after_dca(sig, 2294.0, list(CHANNEL_SWING.tp_ratios))

        assert sig.original_entry == pytest.approx(2300.0)
        assert sig.original_tp1 == pytest.approx(expected_orig_tp1)

    def test_entry_2_filled_flag_set(self):
        """entry_2_filled must be True after DCA."""
        sig = _make_long_signal()
        assert sig.entry_2_filled is False
        recalculate_after_dca(sig, 2294.0, list(CHANNEL_SWING.tp_ratios))
        assert sig.entry_2_filled is True

    def test_dca_weight_variations(self):
        """Different weight splits (50/50, 60/40, 70/30) all produce correct avg."""
        for w1, w2 in [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]:
            sig = _make_long_signal(entry=100.0, stop_loss=90.0)
            recalculate_after_dca(sig, 95.0, [1.0, 2.0, 3.0], weight_1=w1, weight_2=w2)
            expected_avg = 100.0 * w1 + 95.0 * w2
            assert sig.avg_entry == pytest.approx(expected_avg, abs=1e-6)
            assert sig.entry == pytest.approx(expected_avg, abs=1e-6)


# ---------------------------------------------------------------------------
# check_dca_entry tests
# ---------------------------------------------------------------------------

class TestCheckDcaEntry:
    def test_check_dca_entry_in_zone(self):
        """Valid DCA when price is in zone without indicators provided."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        # DCA zone: 2300 - 0.70*20 = 2286 to 2300 - 0.30*20 = 2294
        price_in_zone = 2290.0
        result = check_dca_entry(sig, price_in_zone)
        assert result == pytest.approx(price_in_zone)

    def test_check_dca_entry_outside_zone_too_close(self):
        """Returns None when price is too close to entry (< 30% into zone)."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        # Price is only 1 unit below entry (< 30% of 20 = 6)
        result = check_dca_entry(sig, 2299.0)
        assert result is None

    def test_check_dca_entry_outside_zone_too_far(self):
        """Returns None when price is too close to SL (> 70% into zone)."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        # Price is 15 below entry (> 70% of 20 = 14)
        result = check_dca_entry(sig, 2284.0)
        assert result is None

    def test_check_dca_entry_already_filled(self):
        """Returns None when entry_2_filled is True."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        sig.entry_2_filled = True
        result = check_dca_entry(sig, 2290.0)
        assert result is None

    def test_check_dca_entry_no_momentum(self):
        """Returns None when momentum fades below threshold."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        low_momentum_indicators = {"5m": {"momentum_last": 0.05}}  # below 0.2
        result = check_dca_entry(sig, 2290.0, indicators=low_momentum_indicators)
        assert result is None

    def test_check_dca_entry_good_momentum(self):
        """Returns entry price when momentum is above threshold."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        good_indicators = {"5m": {"momentum_last": 0.5}}  # above 0.2
        result = check_dca_entry(sig, 2290.0, indicators=good_indicators)
        assert result == pytest.approx(2290.0)

    def test_check_dca_entry_no_mss(self):
        """Returns None when MSS is no longer present in smc_data."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        smc_no_mss = {"mss": None, "sweeps": []}
        result = check_dca_entry(sig, 2290.0, smc_data=smc_no_mss)
        assert result is None

    def test_check_dca_entry_mss_present(self):
        """Returns entry price when MSS is present."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        smc_with_mss = {"mss": MagicMock(), "sweeps": []}
        result = check_dca_entry(sig, 2290.0, smc_data=smc_with_mss)
        assert result == pytest.approx(2290.0)

    def test_check_dca_entry_zone_not_configured(self):
        """Returns None when DCA zone bounds are zero (not configured)."""
        sig = Signal(
            channel="360_SWING",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=100.0,
            stop_loss=90.0,
            tp1=110.0,
            tp2=120.0,
        )
        # dca_zone_lower and dca_zone_upper default to 0.0
        result = check_dca_entry(sig, 95.0)
        assert result is None

    def test_check_dca_entry_swing_ema200_bias_long(self):
        """SWING LONG: rejected when price is below EMA200."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0, channel="360_SWING")
        # Price in DCA zone but below EMA200 — should be rejected
        indicators = {"1h": {"ema200_last": 2295.0, "momentum_last": 0.5}}
        result = check_dca_entry(
            sig, 2290.0, indicators=indicators, channel_config=CHANNEL_SWING
        )
        assert result is None

    def test_check_dca_entry_swing_ema200_bias_long_ok(self):
        """SWING LONG: passes when price is above EMA200."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0, channel="360_SWING")
        # EMA200 below the DCA zone — bias intact
        indicators = {"1h": {"ema200_last": 2285.0, "momentum_last": 0.5}}
        result = check_dca_entry(
            sig, 2290.0, indicators=indicators, channel_config=CHANNEL_SWING
        )
        assert result == pytest.approx(2290.0)


# ---------------------------------------------------------------------------
# Breakeven and profit math
# ---------------------------------------------------------------------------

class TestBreakevenAndProfit:
    def test_breakeven_at_entry1_gives_profit_long(self):
        """When price returns to Entry 1 after DCA, PnL is positive from avg_entry."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        recalculate_after_dca(sig, 2294.0, list(CHANNEL_SWING.tp_ratios))

        # Price returns to Entry 1 level
        price_at_entry1 = 2300.0
        pnl = (price_at_entry1 - sig.avg_entry) / sig.avg_entry * 100

        assert pnl > 0, "PnL must be positive when price returns to Entry 1"

    def test_breakeven_at_entry1_gives_profit_short(self):
        """SHORT: When price returns to Entry 1, PnL is positive from avg_entry."""
        sig = _make_short_signal(entry=2300.0, stop_loss=2320.0)
        recalculate_after_dca(sig, 2306.0, list(CHANNEL_SWING.tp_ratios))

        # Price returns to Entry 1 level
        price_at_entry1 = 2300.0
        pnl = (sig.avg_entry - price_at_entry1) / sig.avg_entry * 100

        assert pnl > 0, "PnL must be positive when price returns to Entry 1"

    def test_deeper_dca_gives_more_profit_at_entry1(self):
        """A deeper Entry 2 dip produces more profit when price recovers to Entry 1."""
        # Shallow DCA
        sig_shallow = _make_long_signal(entry=100.0, stop_loss=90.0)
        recalculate_after_dca(sig_shallow, 97.0, [1.0, 2.0, 3.0])
        pnl_shallow = (100.0 - sig_shallow.avg_entry) / sig_shallow.avg_entry * 100

        # Deeper DCA
        sig_deep = _make_long_signal(entry=100.0, stop_loss=90.0)
        recalculate_after_dca(sig_deep, 93.0, [1.0, 2.0, 3.0])
        pnl_deep = (100.0 - sig_deep.avg_entry) / sig_deep.avg_entry * 100

        assert pnl_deep > pnl_shallow


# ---------------------------------------------------------------------------
# Channel signal DCA zone initialisation
# ---------------------------------------------------------------------------

class TestChannelDcaZoneInit:
    def test_scalp_channel_initialises_dca_zone(self):
        """ScalpChannel.evaluate() must populate DCA zone fields on the signal."""
        from src.channels.scalp import ScalpChannel
        from src.smc import LiquiditySweep
        import numpy as np

        ch = ScalpChannel()
        n = 60
        base = 100.0
        close = np.cumsum(np.ones(n) * 0.1) + base
        candles = {
            "5m": {
                "open": close - 0.05,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.ones(n) * 1000,
            }
        }
        sweep = LiquiditySweep(
            index=59, direction=Direction.LONG,
            sweep_level=99, close_price=99.05,
            wick_high=101, wick_low=98,
        )
        indicators = {
            "5m": {
                "adx_last": 30,
                "atr_last": 0.5,
                "ema9_last": 101,
                "ema21_last": 100,
                "momentum_last": 0.5,
            }
        }
        smc_data = {"sweeps": [sweep]}

        sig = ch.evaluate("BTCUSDT", candles, indicators, smc_data, 0.01, 10_000_000)
        assert sig is not None
        assert sig.dca_zone_lower > 0
        assert sig.dca_zone_upper > 0
        assert sig.dca_zone_lower < sig.dca_zone_upper
        assert sig.original_entry == sig.entry
        assert sig.original_tp1 == sig.tp1

    def test_swing_channel_initialises_dca_zone(self):
        """SwingChannel.evaluate() must populate DCA zone fields on the signal."""
        from src.channels.swing import SwingChannel
        from src.smc import LiquiditySweep, MSSSignal
        import numpy as np

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
            index=59,
            direction=Direction.LONG,
            midpoint=close[-1],
            confirm_close=close[-1],
        )
        sweep = LiquiditySweep(
            index=59, direction=Direction.LONG,
            sweep_level=99, close_price=99.05,
            wick_high=101, wick_low=98,
        )
        indicators = {
            "4h": {"adx_last": 30, "atr_last": 1.0},
            "1h": {
                "adx_last": 25,
                "atr_last": 0.5,
                "ema200_last": close[-1] - 5,
                "bb_upper_last": close[-1] + 5,
                "bb_lower_last": close[-1] - 0.2,  # close very near lower BB → bb_position ≈ 0.04 < 0.15
            },
        }
        smc_data = {"sweeps": [sweep], "mss": mss}

        sig = ch.evaluate("ETHUSDT", candles, indicators, smc_data, 0.01, 15_000_000)
        assert sig is not None
        assert sig.dca_zone_lower > 0
        assert sig.dca_zone_upper > 0
        assert sig.original_entry == sig.entry


# ---------------------------------------------------------------------------
# DCA config tests
# ---------------------------------------------------------------------------

class TestDcaConfig:
    def test_swing_dca_enabled(self):
        assert CHANNEL_SWING.dca_enabled is True

    def test_scalp_dca_enabled(self):
        assert CHANNEL_SCALP.dca_enabled is True

    def test_swing_default_weights(self):
        assert CHANNEL_SWING.dca_weight_1 == pytest.approx(0.6)
        assert CHANNEL_SWING.dca_weight_2 == pytest.approx(0.4)

    def test_swing_dca_zone_range(self):
        assert CHANNEL_SWING.dca_zone_range == (0.30, 0.70)

    def test_spot_dca_enabled(self):
        from config import CHANNEL_SPOT
        assert CHANNEL_SPOT.dca_enabled is True

    def test_spot_dca_zone_range(self):
        from config import CHANNEL_SPOT
        assert CHANNEL_SPOT.dca_zone_range == (0.30, 0.70)

    def test_spot_dca_weights(self):
        from config import CHANNEL_SPOT
        assert CHANNEL_SPOT.dca_weight_1 == pytest.approx(0.6)
        assert CHANNEL_SPOT.dca_weight_2 == pytest.approx(0.4)

    def test_gem_dca_disabled(self):
        from config import CHANNEL_GEM
        assert CHANNEL_GEM.dca_enabled is False


# ---------------------------------------------------------------------------
# dca_timestamp tests
# ---------------------------------------------------------------------------

class TestDcaTimestamp:
    def test_dca_timestamp_set_after_recalculate(self):
        """dca_timestamp must be set to a recent UTC time after recalculate_after_dca."""
        from src.utils import utcnow
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        assert sig.dca_timestamp is None  # not set before DCA

        before = utcnow()
        recalculate_after_dca(sig, 2294.0, list(CHANNEL_SWING.tp_ratios))
        after = utcnow()

        assert sig.dca_timestamp is not None
        assert before <= sig.dca_timestamp <= after

    def test_dca_timestamp_not_overwritten_on_second_call(self):
        """dca_timestamp is updated on each DCA call (recalculate stamps with current time)."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        recalculate_after_dca(sig, 2294.0, list(CHANNEL_SWING.tp_ratios))
        first_ts = sig.dca_timestamp

        # Reset flag to allow a second call (testing timestamp behaviour)
        sig.entry_2_filled = False
        recalculate_after_dca(sig, 2292.0, list(CHANNEL_SWING.tp_ratios))

        # Each DCA call stamps the current time, so the second stamp >= first
        assert sig.dca_timestamp is not None
        assert sig.dca_timestamp >= first_ts


# ---------------------------------------------------------------------------
# Regime-aware DCA zones
# ---------------------------------------------------------------------------


class TestRegimeAwareDcaZone:
    """compute_dca_zone must adjust zones based on market regime."""

    def test_default_zone_no_regime(self):
        """Without regime, default zone (0.30, 0.70) is used."""
        lower, upper = compute_dca_zone(100.0, 90.0, Direction.LONG)
        assert upper == pytest.approx(100.0 - 0.30 * 10.0)  # 97.0
        assert lower == pytest.approx(100.0 - 0.70 * 10.0)  # 93.0

    def test_volatile_regime_pushes_zone_deeper(self):
        """VOLATILE regime must use (0.50, 0.85) zone range."""
        lower, upper = compute_dca_zone(100.0, 90.0, Direction.LONG, regime="VOLATILE")
        assert upper == pytest.approx(100.0 - 0.50 * 10.0)  # 95.0
        assert lower == pytest.approx(100.0 - 0.85 * 10.0)  # 91.5

    def test_trending_regime_pushes_zone_deeper(self):
        """TRENDING regime must also use (0.50, 0.85) zone range."""
        lower, upper = compute_dca_zone(100.0, 90.0, Direction.LONG, regime="TRENDING")
        assert upper == pytest.approx(100.0 - 0.50 * 10.0)  # 95.0
        assert lower == pytest.approx(100.0 - 0.85 * 10.0)  # 91.5

    def test_ranging_regime_tightens_zone(self):
        """RANGING regime must use (0.30, 0.60) zone range."""
        lower, upper = compute_dca_zone(100.0, 90.0, Direction.LONG, regime="RANGING")
        assert upper == pytest.approx(100.0 - 0.30 * 10.0)  # 97.0
        assert lower == pytest.approx(100.0 - 0.60 * 10.0)  # 94.0

    def test_unknown_regime_keeps_default(self):
        """Unknown regime strings must leave zone_range unchanged."""
        lower, upper = compute_dca_zone(100.0, 90.0, Direction.LONG, regime="UNKNOWN")
        assert upper == pytest.approx(100.0 - 0.30 * 10.0)  # default 97.0
        assert lower == pytest.approx(100.0 - 0.70 * 10.0)  # default 93.0

    def test_regime_case_insensitive(self):
        """Regime string matching is case-insensitive."""
        lower_upper, upper_upper = compute_dca_zone(100.0, 90.0, Direction.LONG, regime="volatile")
        lower_mixed, upper_mixed = compute_dca_zone(100.0, 90.0, Direction.LONG, regime="Volatile")
        assert lower_upper == pytest.approx(lower_mixed)
        assert upper_upper == pytest.approx(upper_mixed)

    def test_volatile_zone_short(self):
        """VOLATILE regime produces deeper zone for SHORT trades too."""
        lower, upper = compute_dca_zone(100.0, 110.0, Direction.SHORT, regime="VOLATILE")
        assert lower == pytest.approx(100.0 + 0.50 * 10.0)  # 105.0
        assert upper == pytest.approx(100.0 + 0.85 * 10.0)  # 108.5

    def test_custom_zone_range_overridden_by_regime(self):
        """When regime is provided, zone_range default is overridden."""
        # Custom zone (0.20, 0.50) with VOLATILE regime → should become (0.50, 0.85)
        lower, upper = compute_dca_zone(
            100.0, 90.0, Direction.LONG, zone_range=(0.20, 0.50), regime="VOLATILE"
        )
        assert upper == pytest.approx(100.0 - 0.50 * 10.0)  # 95.0 not 96.0
        assert lower == pytest.approx(100.0 - 0.85 * 10.0)  # 91.5 not 90.0


# ---------------------------------------------------------------------------
# Volume delta check in check_dca_entry
# ---------------------------------------------------------------------------


class TestVolumeDeltaCheck:
    """check_dca_entry must reject DCA when volume_delta indicates heavy
    counter-directional pressure."""

    def test_long_rejected_on_heavy_selling(self):
        """LONG DCA must be rejected when volume_delta <= -0.7 (heavy selling)."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        indicators = {"5m": {"momentum_last": 0.5, "volume_delta": -0.8}}
        result = check_dca_entry(sig, 2290.0, indicators=indicators)
        assert result is None

    def test_short_rejected_on_heavy_buying(self):
        """SHORT DCA must be rejected when volume_delta >= 0.7 (heavy buying)."""
        sig = _make_short_signal(entry=2300.0, stop_loss=2320.0)
        indicators = {"5m": {"momentum_last": -0.5, "volume_delta": 0.8}}
        result = check_dca_entry(sig, 2306.0, indicators=indicators)
        assert result is None

    def test_long_allowed_on_neutral_delta(self):
        """LONG DCA must be allowed when volume_delta is neutral (between -0.7 and 0)."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        indicators = {"5m": {"momentum_last": 0.5, "volume_delta": -0.5}}
        result = check_dca_entry(sig, 2290.0, indicators=indicators)
        assert result == pytest.approx(2290.0)

    def test_short_allowed_on_neutral_delta(self):
        """SHORT DCA must be allowed when volume_delta is neutral (between 0 and 0.7)."""
        sig = _make_short_signal(entry=2300.0, stop_loss=2320.0)
        indicators = {"5m": {"momentum_last": -0.5, "volume_delta": 0.5}}
        result = check_dca_entry(sig, 2306.0, indicators=indicators)
        assert result == pytest.approx(2306.0)

    def test_no_volume_delta_key_skips_check(self):
        """When volume_delta is not in indicators, the check is skipped."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        # No volume_delta key → check skipped, DCA proceeds normally
        indicators = {"5m": {"momentum_last": 0.5}}
        result = check_dca_entry(sig, 2290.0, indicators=indicators)
        assert result == pytest.approx(2290.0)

    def test_long_boundary_delta_at_threshold_allowed(self):
        """volume_delta exactly at -0.7 satisfies strict inequality (< -0.7 is False) — allowed."""
        sig = _make_long_signal(entry=2300.0, stop_loss=2280.0)
        indicators = {"5m": {"momentum_last": 0.5, "volume_delta": -0.7}}
        # -0.7 < -0.7 evaluates to False (strict less-than), so check passes
        result = check_dca_entry(sig, 2290.0, indicators=indicators)
        assert result == pytest.approx(2290.0)
