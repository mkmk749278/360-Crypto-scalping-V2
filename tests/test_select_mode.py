"""Tests for src.select_mode – SelectModeFilter."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.select_mode import SelectModeConfig, SelectModeFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(direction: str = "LONG", channel: str = "360_SCALP") -> MagicMock:
    sig = MagicMock()
    sig.direction.value = direction
    sig.channel = channel
    return sig


def _good_indicators() -> dict:
    """Indicators that pass all EMA-confluence and momentum checks."""
    return {
        "5m": {
            "ema9_last": 100.0,
            "ema21_last": 99.0,   # EMA9 > EMA21 → LONG aligned
            "adx_last": 30.0,
            "rsi_last": 55.0,
            "momentum_last": 0.5,
        },
        "15m": {
            "ema9_last": 100.0,
            "ema21_last": 99.0,
            "adx_last": 28.0,
            "rsi_last": 50.0,
            "momentum_last": 0.3,
        },
    }


def _good_smc() -> dict:
    return {"sweeps": [{"level": 100}], "mss": None, "fvg": []}


def _good_ai(ai_label: str = "Bullish") -> dict:
    return {"label": ai_label, "score": 0.8}


def _call_should_publish(
    sf: SelectModeFilter,
    direction: str = "LONG",
    channel: str = "360_SCALP",
    confidence: float = 85.0,
    indicators: dict | None = None,
    smc_data: dict | None = None,
    ai_sentiment: dict | None = None,
    cross_exchange_verified: bool | None = True,
    volume_24h: float = 15_000_000.0,
    spread_pct: float = 0.008,
):
    signal = _make_signal(direction, channel)
    return sf.should_publish(
        signal=signal,
        confidence=confidence,
        indicators=indicators or _good_indicators(),
        smc_data=smc_data or _good_smc(),
        ai_sentiment=ai_sentiment or _good_ai(),
        cross_exchange_verified=cross_exchange_verified,
        volume_24h=volume_24h,
        spread_pct=spread_pct,
    )


# ---------------------------------------------------------------------------
# Tests: disabled mode
# ---------------------------------------------------------------------------

class TestSelectModeDisabled:
    """When select mode is OFF, should_publish always returns (True, '')."""

    def test_disabled_by_default(self):
        sf = SelectModeFilter()
        assert sf.enabled is False

    def test_disabled_always_passes(self):
        sf = SelectModeFilter()  # disabled by default
        allowed, reason = _call_should_publish(sf, confidence=10.0)
        assert allowed is True
        assert reason == ""

    def test_disabled_ignores_all_bad_params(self):
        sf = SelectModeFilter()
        # Terrible inputs – should still pass when disabled
        allowed, _ = _call_should_publish(
            sf,
            confidence=0.0,
            smc_data={"sweeps": [], "mss": None, "fvg": []},
            ai_sentiment={"label": "Bearish", "score": -1.0},
            cross_exchange_verified=False,
            volume_24h=0.0,
            spread_pct=1.0,
        )
        assert allowed is True


# ---------------------------------------------------------------------------
# Tests: confidence filter
# ---------------------------------------------------------------------------

class TestConfidenceFilter:
    def test_rejects_low_confidence(self):
        sf = SelectModeFilter()
        sf.enable()
        allowed, reason = _call_should_publish(sf, confidence=70.0)
        assert allowed is False
        assert "confidence" in reason

    def test_accepts_high_confidence(self):
        sf = SelectModeFilter()
        sf.enable()
        allowed, _ = _call_should_publish(sf, confidence=85.0)
        assert allowed is True


# ---------------------------------------------------------------------------
# Tests: daily cap
# ---------------------------------------------------------------------------

class TestDailyCap:
    def test_daily_cap_enforced(self):
        cfg = SelectModeConfig(enabled=True, max_daily_signals=2)
        sf = SelectModeFilter(config=cfg)

        # First two should pass
        for i in range(2):
            ok, _ = _call_should_publish(sf, channel="360_SCALP")
            assert ok is True, f"Expected pass on call {i+1}"

        # Third should be blocked
        ok, reason = _call_should_publish(sf, channel="360_SCALP")
        assert ok is False
        assert "daily cap" in reason

    def test_daily_cap_per_channel(self):
        """Cap is tracked independently per channel."""
        cfg = SelectModeConfig(enabled=True, max_daily_signals=1)
        sf = SelectModeFilter(config=cfg)

        ok1, _ = _call_should_publish(sf, channel="360_SCALP")
        assert ok1 is True

        # Different channel – should have its own counter
        ok2, _ = _call_should_publish(sf, channel="360_SWING")
        assert ok2 is True

        # Same channel again – blocked
        ok3, reason = _call_should_publish(sf, channel="360_SCALP")
        assert ok3 is False
        assert "daily cap" in reason


# ---------------------------------------------------------------------------
# Tests: EMA confluence
# ---------------------------------------------------------------------------

class TestEMAConfluence:
    def test_rejects_insufficient_confluence(self):
        sf = SelectModeFilter()
        sf.enable()
        # Only one timeframe aligned with LONG (5m), the other contradicts
        bad_indicators = {
            "5m": {"ema9_last": 100.0, "ema21_last": 99.0, "momentum_last": 0.5},
            "15m": {"ema9_last": 98.0, "ema21_last": 99.0, "momentum_last": 0.3},
        }
        allowed, reason = _call_should_publish(sf, indicators=bad_indicators)
        assert allowed is False
        assert "EMA confluence" in reason

    def test_accepts_two_aligned_timeframes(self):
        sf = SelectModeFilter()
        sf.enable()
        # Both 5m and 15m aligned for LONG
        good = {
            "5m": {"ema9_last": 100.0, "ema21_last": 99.0, "momentum_last": 0.5},
            "15m": {"ema9_last": 100.0, "ema21_last": 99.0, "momentum_last": 0.3},
        }
        allowed, _ = _call_should_publish(sf, indicators=good)
        assert allowed is True


# ---------------------------------------------------------------------------
# Tests: RSI band
# ---------------------------------------------------------------------------

class TestRSIBand:
    def test_passes_rsi_inside_band(self):
        sf = SelectModeFilter()
        sf.enable()
        ind = {
            "5m": {"ema9_last": 100.0, "ema21_last": 99.0,
                   "rsi_last": 55.0, "momentum_last": 0.5},
            "15m": {"ema9_last": 100.0, "ema21_last": 99.0,
                    "rsi_last": 50.0, "momentum_last": 0.3},
        }
        allowed, _ = _call_should_publish(sf, indicators=ind)
        assert allowed is True

    def test_rejects_rsi_above_max(self):
        sf = SelectModeFilter()
        sf.enable()
        ind = {
            "5m": {"ema9_last": 100.0, "ema21_last": 99.0,
                   "rsi_last": 75.0, "momentum_last": 0.5},
            "15m": {"ema9_last": 100.0, "ema21_last": 99.0,
                    "rsi_last": 50.0, "momentum_last": 0.3},
        }
        allowed, reason = _call_should_publish(sf, indicators=ind)
        assert allowed is False
        assert "RSI" in reason

    def test_rejects_rsi_below_min(self):
        sf = SelectModeFilter()
        sf.enable()
        ind = {
            "5m": {"ema9_last": 100.0, "ema21_last": 99.0,
                   "rsi_last": 25.0, "momentum_last": 0.5},
            "15m": {"ema9_last": 100.0, "ema21_last": 99.0,
                    "rsi_last": 50.0, "momentum_last": 0.3},
        }
        allowed, reason = _call_should_publish(sf, indicators=ind)
        assert allowed is False
        assert "RSI" in reason

    def test_passes_when_rsi_unavailable(self):
        """If RSI data is absent, the filter must not reject the signal."""
        sf = SelectModeFilter()
        sf.enable()
        ind = {
            "5m": {"ema9_last": 100.0, "ema21_last": 99.0,
                   "rsi_last": None, "momentum_last": 0.5},
            "15m": {"ema9_last": 100.0, "ema21_last": 99.0,
                    "momentum_last": 0.3},
        }
        allowed, _ = _call_should_publish(sf, indicators=ind)
        assert allowed is True


# ---------------------------------------------------------------------------
# Tests: AI sentiment direction match
# ---------------------------------------------------------------------------

class TestAISentimentMatch:
    def test_bullish_allows_long(self):
        sf = SelectModeFilter()
        sf.enable()
        allowed, _ = _call_should_publish(
            sf, direction="LONG", ai_sentiment={"label": "Bullish", "score": 0.8}
        )
        assert allowed is True

    def test_bullish_rejects_short(self):
        sf = SelectModeFilter()
        sf.enable()
        ind = {
            "5m": {"ema9_last": 99.0, "ema21_last": 100.0, "momentum_last": -0.5},
            "15m": {"ema9_last": 99.0, "ema21_last": 100.0, "momentum_last": -0.3},
        }
        allowed, reason = _call_should_publish(
            sf,
            direction="SHORT",
            ai_sentiment={"label": "Bullish", "score": 0.8},
            indicators=ind,
        )
        assert allowed is False
        assert "AI" in reason

    def test_neutral_allows_any_direction(self):
        sf = SelectModeFilter()
        sf.enable()
        allowed_long, _ = _call_should_publish(
            sf, direction="LONG", ai_sentiment={"label": "Neutral", "score": 0.0}
        )
        assert allowed_long is True

        ind_short = {
            "5m": {"ema9_last": 99.0, "ema21_last": 100.0, "momentum_last": -0.5},
            "15m": {"ema9_last": 99.0, "ema21_last": 100.0, "momentum_last": -0.3},
        }
        allowed_short, _ = _call_should_publish(
            sf,
            direction="SHORT",
            ai_sentiment={"label": "Neutral", "score": 0.0},
            indicators=ind_short,
        )
        assert allowed_short is True


# ---------------------------------------------------------------------------
# Tests: config update
# ---------------------------------------------------------------------------

class TestConfigUpdate:
    def test_update_min_confidence(self):
        sf = SelectModeFilter()
        ok, msg = sf.update_config("min_confidence", "90")
        assert ok is True
        assert sf._config.min_confidence == 90.0

    def test_update_max_daily_signals(self):
        sf = SelectModeFilter()
        ok, _ = sf.update_config("max_daily_signals", "3")
        assert ok is True
        assert sf._config.max_daily_signals == 3

    def test_update_boolean_field(self):
        sf = SelectModeFilter()
        ok, _ = sf.update_config("require_smc_event", "false")
        assert ok is True
        assert sf._config.require_smc_event is False

    def test_update_unknown_key(self):
        sf = SelectModeFilter()
        ok, msg = sf.update_config("nonexistent_key", "42")
        assert ok is False
        assert "Unknown" in msg

    def test_update_invalid_value(self):
        sf = SelectModeFilter()
        ok, msg = sf.update_config("min_confidence", "not_a_number")
        assert ok is False

    def test_update_rsi_range(self):
        sf = SelectModeFilter()
        sf.update_config("rsi_min", "35")
        sf.update_config("rsi_max", "65")
        assert sf._config.rsi_min == 35.0
        assert sf._config.rsi_max == 65.0


# ---------------------------------------------------------------------------
# Tests: status_text()
# ---------------------------------------------------------------------------

class TestStatusText:
    def test_status_text_contains_mode_state(self):
        sf = SelectModeFilter()
        text = sf.status_text()
        assert "OFF" in text
        assert "360_SELECT" in text or "SELECT" in text

    def test_status_text_enabled(self):
        sf = SelectModeFilter()
        sf.enable()
        text = sf.status_text()
        assert "ON" in text

    def test_status_text_contains_config_fields(self):
        sf = SelectModeFilter()
        text = sf.status_text()
        assert "confidence" in text.lower()
        assert "daily" in text.lower()
        assert "ADX" in text or "adx" in text.lower()
