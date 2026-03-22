"""Tests for src.chart_generator — generate_gem_chart."""

from __future__ import annotations

import pytest

from src.chart_generator import generate_gem_chart, _MPF_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_candles(n: int = 120) -> dict:
    """Build synthetic OHLCV daily candles for testing."""
    base = 1.0
    closes = [base + 0.001 * i for i in range(n)]
    highs = [c + 0.05 for c in closes]
    lows = [c - 0.05 for c in closes]
    volumes = [1_000_000.0] * n
    return {
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


def _make_ema(n: int, start: float = 1.0, step: float = 0.001) -> list:
    return [start + step * i for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateGemChartInvalidData:
    def test_returns_none_for_empty_candles(self):
        result = generate_gem_chart(
            symbol="TESTUSDT",
            daily_candles={"open": [], "high": [], "low": [], "close": [], "volume": []},
            ath=10.0,
            current_price=1.0,
            ema_20=[],
            ema_50=[],
        )
        assert result is None

    def test_returns_none_for_insufficient_candles(self):
        """Fewer than 10 candles should return None."""
        n = 5
        candles = _make_daily_candles(n)
        result = generate_gem_chart(
            symbol="TESTUSDT",
            daily_candles=candles,
            ath=10.0,
            current_price=1.0,
            ema_20=_make_ema(n),
            ema_50=_make_ema(n),
        )
        assert result is None


@pytest.mark.skipif(not _MPF_AVAILABLE, reason="mplfinance not installed")
class TestGenerateGemChartValid:
    def test_returns_png_bytes_for_valid_input(self):
        """Valid 120-candle input should return PNG bytes."""
        n = 120
        candles = _make_daily_candles(n)
        result = generate_gem_chart(
            symbol="LYNUSDT",
            daily_candles=candles,
            ath=10.0,
            current_price=1.0,
            ema_20=_make_ema(n, start=1.0, step=0.001),
            ema_50=_make_ema(n, start=0.98, step=0.001),
        )
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 1000  # sanity: a real PNG is at least a few KB

    def test_returns_png_magic_bytes(self):
        """Output should start with the PNG magic bytes \\x89PNG."""
        n = 120
        candles = _make_daily_candles(n)
        result = generate_gem_chart(
            symbol="TAOUSDT",
            daily_candles=candles,
            ath=5.0,
            current_price=1.0,
            ema_20=_make_ema(n),
            ema_50=_make_ema(n, start=0.95),
        )
        assert result is not None
        assert result[:4] == b"\x89PNG"

    def test_works_with_fewer_ema_values_than_candles(self):
        """EMA arrays shorter than the candle window are padded gracefully."""
        n = 120
        candles = _make_daily_candles(n)
        result = generate_gem_chart(
            symbol="DOGEUSDT",
            daily_candles=candles,
            ath=8.0,
            current_price=1.0,
            ema_20=_make_ema(20),   # Only 20 EMA values
            ema_50=_make_ema(50),   # Only 50 EMA values
        )
        assert result is not None
        assert isinstance(result, bytes)

    def test_works_with_exactly_10_candles(self):
        """Exactly 10 candles (minimum) should still produce output."""
        n = 10
        candles = _make_daily_candles(n)
        result = generate_gem_chart(
            symbol="GEMTEST",
            daily_candles=candles,
            ath=2.0,
            current_price=1.0,
            ema_20=_make_ema(n),
            ema_50=_make_ema(n),
        )
        assert result is not None
        assert isinstance(result, bytes)


class TestGenerateGemChartNomplfinance:
    """These tests verify graceful degradation when mplfinance is unavailable."""

    def test_returns_none_when_mpf_unavailable(self, monkeypatch):
        """Patch _MPF_AVAILABLE to False and verify None is returned."""
        import src.chart_generator as cg
        monkeypatch.setattr(cg, "_MPF_AVAILABLE", False)
        n = 120
        candles = _make_daily_candles(n)
        result = cg.generate_gem_chart(
            symbol="TESTUSDT",
            daily_candles=candles,
            ath=10.0,
            current_price=1.0,
            ema_20=_make_ema(n),
            ema_50=_make_ema(n),
        )
        assert result is None
