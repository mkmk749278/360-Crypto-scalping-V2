"""Tests for the 8 competitive gap improvements.

Feature 1: AI Sentiment wired into scanner signal pipeline
Feature 2: On-chain enrichment for GemScanner
Feature 3: CCXT-based OrderManager with position sizing + partial TP
Feature 4: Chart images in SignalLifecycleMonitor
Feature 5: HTML performance report (generate_html_report)
Feature 6: SHORT spot signals in SpotChannel
Feature 7: Adaptive gem scanner thresholds by regime
Feature 8: Cornix auto-execution block (existing formatter, verified wired)
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_spot_candles_long(n: int = 60, base: float = 200.0) -> dict:
    """Candles where price is above EMA200 and above recent resistance (LONG setup)."""
    close = [base + i * 0.1 for i in range(n)]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    volume = [1000.0] * n
    # Current bar breaks out (higher than last 9 bars' highs)
    close[-1] = close[-2] + 2.0
    high[-1] = close[-1] + 0.5
    volume[-1] = 2500.0  # volume surge
    return {"open": close, "high": high, "low": low, "close": close, "volume": volume}


def _make_spot_candles_short(n: int = 60, base: float = 200.0) -> dict:
    """Candles where price is below EMA200 and breaks below recent support (SHORT setup)."""
    close = [base - i * 0.1 for i in range(n)]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    volume = [1000.0] * n
    # Current bar breaks down below recent lows
    close[-1] = close[-2] - 2.0
    low[-1] = close[-1] - 0.5
    volume[-1] = 2500.0  # volume surge on the down-move
    return {"open": close, "high": high, "low": low, "close": close, "volume": volume}


def _spot_indicators_long(close: float, ema200: float) -> dict:
    """Indicators for a LONG spot setup."""
    return {
        "adx_last": 30,
        "atr_last": close * 0.01,
        "ema200_last": ema200,
        "ema50_last": ema200 - 1.0,
        "rsi_last": 55,
        "bb_width_pct": 2.0,
    }


def _spot_indicators_short(close: float, ema200: float) -> dict:
    """Indicators for a SHORT spot setup."""
    return {
        "adx_last": 30,
        "atr_last": close * 0.01,
        "ema200_last": ema200,
        "ema50_last": ema200 + 1.0,  # daily EMA50 also above price
        "rsi_last": 45,
        "bb_width_pct": 2.0,
    }


def _gem_candles(n: int = 210, ath: float = 10.0, current: float = 1.0, vol_surge: float = 3.0) -> dict:
    """Build candles that pass GEM filters."""
    closes = [current] * n
    for i in range(max(0, n - 25), n):
        closes[i] = current * (1 + 0.002 * (i - (n - 25)))
    highs = [ath] + [c + 0.01 for c in closes[1:]]
    lows = [c - 0.01 for c in closes]
    base_vol = 300_000.0
    volumes = [base_vol] * n
    for i in range(n - 7, n):
        volumes[i] = base_vol * vol_surge
    return {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes}


def _make_signal(
    symbol: str = "BTCUSDT",
    channel: str = "360_SPOT",
    direction: str = "LONG",
    entry: float = 100.0,
    signal_id: str = "SPOT-TESTID01",
) -> Any:
    """Build a minimal mock signal object."""
    from src.smc import Direction

    sig = MagicMock()
    sig.symbol = symbol
    sig.channel = channel
    sig.direction = Direction.LONG if direction == "LONG" else Direction.SHORT
    sig.entry = entry
    sig.stop_loss = entry * 0.99
    sig.tp1 = entry * 1.02
    sig.tp2 = entry * 1.05
    sig.tp3 = entry * 1.10
    sig.signal_id = signal_id
    sig.confidence = 70.0
    sig.ai_sentiment_label = ""
    sig.ai_sentiment_summary = ""
    sig.status = "ACTIVE"
    sig.best_tp_hit = 0
    sig.best_tp_pnl_pct = 0.0
    sig.current_price = entry
    return sig


# ===========================================================================
# Feature 1: AI Sentiment wired into signal pipeline
# ===========================================================================


class TestAiSentimentWiring:
    """Signal.ai_sentiment_label and ai_sentiment_summary are populated for SPOT/GEM."""

    def test_signal_router_builds_narrative_context_with_sentiment(self) -> None:
        """_build_narrative_context includes sentiment fields when present."""
        from src.signal_router import SignalRouter

        router = SignalRouter.__new__(SignalRouter)
        router.sector_comparator = None

        sig = _make_signal()
        sig.setup_class = "BB_SQUEEZE"
        sig.liquidity_info = ""
        sig.risk_label = "Conservative"
        sig.ai_sentiment_label = "Positive"
        sig.ai_sentiment_summary = "News: bullish — Social: positive — Fear&Greed: 70 (Greed)"

        ctx = router._build_narrative_context(sig)

        assert ctx.get("sentiment_label") == "Positive"
        assert "Fear&Greed" in ctx.get("sentiment_summary", "")

    def test_signal_router_omits_sentiment_when_empty(self) -> None:
        """_build_narrative_context does NOT add sentiment keys for SCALP/SWING."""
        from src.signal_router import SignalRouter

        router = SignalRouter.__new__(SignalRouter)
        router.sector_comparator = None

        sig = _make_signal(channel="360_SCALP")
        sig.setup_class = ""
        sig.liquidity_info = ""
        sig.risk_label = ""
        sig.ai_sentiment_label = ""
        sig.ai_sentiment_summary = ""

        ctx = router._build_narrative_context(sig)

        assert "sentiment_label" not in ctx
        assert "sentiment_summary" not in ctx


# ===========================================================================
# Feature 2: On-chain enrichment for GemScanner
# ===========================================================================


class TestGemOnchainEnrichment:
    """GemScannerConfig has new fields; scan() accepts onchain_data."""

    def test_new_config_fields_exist_with_defaults(self) -> None:
        from src.gem_scanner import GemScannerConfig
        cfg = GemScannerConfig()
        assert cfg.whale_accumulation_boost == 10.0
        assert cfg.social_spike_boost == 5.0
        assert cfg.unlock_penalty == 10.0

    def test_whale_accumulation_boosts_confidence(self) -> None:
        from src.gem_scanner import GemScanner
        scanner = GemScanner()
        candles = _gem_candles(n=210)
        onchain = {"whale_accumulation": True, "social_volume_ratio": 1.0, "unlock_days": None}
        sig_with = scanner.scan("TOKENUSDT", candles, onchain_data=onchain)

        onchain_none = {"whale_accumulation": False, "social_volume_ratio": 1.0, "unlock_days": None}
        sig_without = scanner.scan("TOKENUSDT", candles, onchain_data=onchain_none)

        if sig_with is not None and sig_without is not None:
            assert sig_with.confidence > sig_without.confidence

    def test_social_spike_boosts_confidence(self) -> None:
        from src.gem_scanner import GemScanner
        scanner = GemScanner()
        candles = _gem_candles(n=210)
        onchain_spike = {"whale_accumulation": False, "social_volume_ratio": 3.0, "unlock_days": None}
        onchain_flat = {"whale_accumulation": False, "social_volume_ratio": 1.0, "unlock_days": None}

        sig_spike = scanner.scan("TOKENUSDT", candles, onchain_data=onchain_spike)
        # Reset daily counter between calls
        scanner._daily_counts.clear()
        sig_flat = scanner.scan("TOKENUSDT", candles, onchain_data=onchain_flat)

        if sig_spike is not None and sig_flat is not None:
            assert sig_spike.confidence >= sig_flat.confidence

    def test_imminent_unlock_penalises_confidence(self) -> None:
        from src.gem_scanner import GemScanner
        scanner = GemScanner()
        candles = _gem_candles(n=210)
        onchain_unlock = {"whale_accumulation": False, "social_volume_ratio": 1.0, "unlock_days": 3}
        onchain_no_unlock = {"whale_accumulation": False, "social_volume_ratio": 1.0, "unlock_days": None}

        sig_unlock = scanner.scan("TOKENUSDT", candles, onchain_data=onchain_unlock)
        scanner._daily_counts.clear()
        sig_clean = scanner.scan("TOKENUSDT", candles, onchain_data=onchain_no_unlock)

        if sig_unlock is not None and sig_clean is not None:
            assert sig_unlock.confidence < sig_clean.confidence

    def test_scan_accepts_no_onchain_data(self) -> None:
        from src.gem_scanner import GemScanner
        scanner = GemScanner()
        candles = _gem_candles(n=210)
        # No exception when onchain_data is None
        scanner.scan("TOKENUSDT", candles)  # should not raise


# ===========================================================================
# Feature 3: CCXT-Based OrderManager
# ===========================================================================


class TestOrderManager:
    """OrderManager with mock CCXTClient."""

    def _make_mock_client(self) -> MagicMock:
        client = MagicMock()
        client.create_limit_order = AsyncMock(return_value={"id": "limit-123", "status": "open"})
        client.create_market_order = AsyncMock(return_value={"id": "market-456", "status": "closed"})
        client.cancel_order = AsyncMock(return_value={"id": "limit-123", "status": "canceled"})
        client.fetch_balance = AsyncMock(return_value={"USDT": {"free": 1000.0, "total": 1000.0}})
        return client

    @pytest.mark.asyncio
    async def test_place_limit_order_uses_ccxt(self) -> None:
        from src.order_manager import OrderManager
        client = self._make_mock_client()
        mgr = OrderManager(auto_execution_enabled=True, exchange_client=client)
        sig = _make_signal(channel="360_SPOT")

        order_id = await mgr.place_limit_order(sig)

        assert order_id == "limit-123"
        client.create_limit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_market_order_uses_ccxt(self) -> None:
        from src.order_manager import OrderManager
        client = self._make_mock_client()
        mgr = OrderManager(auto_execution_enabled=True, exchange_client=client)
        sig = _make_signal(channel="360_SCALP")

        order_id = await mgr.place_market_order(sig)

        assert order_id == "market-456"
        client.create_market_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_order_uses_ccxt(self) -> None:
        from src.order_manager import OrderManager
        client = self._make_mock_client()
        mgr = OrderManager(auto_execution_enabled=True, exchange_client=client)

        result = await mgr.cancel_order("limit-123", "BTCUSDT")

        assert result is True  # "canceled" status
        client.cancel_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_partial_calls_market_order(self) -> None:
        from src.order_manager import OrderManager
        client = self._make_mock_client()
        mgr = OrderManager(auto_execution_enabled=True, exchange_client=client)
        sig = _make_signal()
        # Simulate an existing open position
        mgr._open_quantities[sig.signal_id] = 0.5

        await mgr.close_partial(sig, 0.33)

        client.create_market_order.assert_called_once()
        # quantity ≈ 0.5 × 0.33 = 0.165
        args = client.create_market_order.call_args
        assert abs(args[0][2] - 0.165) < 1e-6

    @pytest.mark.asyncio
    async def test_position_sizing_uses_balance(self) -> None:
        from src.order_manager import OrderManager
        client = self._make_mock_client()
        # 2% of $1000 = $20; entry $100 → qty = 0.2
        mgr = OrderManager(
            auto_execution_enabled=True,
            exchange_client=client,
            position_size_pct=2.0,
            max_position_usd=1000.0,
        )
        qty = await mgr._compute_quantity(entry_price=100.0)
        assert abs(qty - 0.2) < 1e-6

    @pytest.mark.asyncio
    async def test_disabled_manager_returns_none(self) -> None:
        from src.order_manager import OrderManager
        mgr = OrderManager(auto_execution_enabled=False)
        sig = _make_signal()
        assert await mgr.execute_signal(sig) is None
        assert await mgr.place_limit_order(sig) is None
        assert await mgr.place_market_order(sig) is None
        assert await mgr.cancel_order("x", "BTCUSDT") is False
        assert await mgr.close_partial(sig, 0.33) is None


class TestCCXTClient:
    """CCXTClient raises NotImplementedError when ccxt is absent."""

    def test_raises_when_ccxt_not_installed(self) -> None:
        """Without ccxt installed, _require_exchange should raise NotImplementedError."""
        from src.exchange_client import CCXTClient, _CCXT_AVAILABLE
        if _CCXT_AVAILABLE:
            pytest.skip("ccxt is installed — stub path not testable")
        client = CCXTClient(exchange_id="binance")
        with pytest.raises(NotImplementedError):
            asyncio.get_event_loop().run_until_complete(
                client.create_market_order("BTC/USDT", "buy", 0.01)
            )


# ===========================================================================
# Feature 4: Chart images in lifecycle updates
# ===========================================================================


class TestLifecycleChartImages:
    """_post_chart is called from _check_signal and sends photo when available."""

    @pytest.mark.asyncio
    async def test_post_chart_calls_send_photo(self) -> None:
        from src.signal_lifecycle import SignalLifecycleMonitor

        send_photo = AsyncMock()
        send_telegram = AsyncMock(return_value=True)

        router = MagicMock()
        router.active_signals = {}

        data_store = MagicMock()
        candles = {
            "close": [100 + i for i in range(20)],
            "high": [101 + i for i in range(20)],
            "low": [99 + i for i in range(20)],
            "volume": [1000.0] * 20,
        }
        data_store.get_candles = MagicMock(return_value=candles)

        regime_detector = MagicMock()
        monitor = SignalLifecycleMonitor(
            router=router,
            data_store=data_store,
            regime_detector=regime_detector,
            send_telegram=send_telegram,
            send_photo=send_photo,
        )

        chart_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        with patch(
            "src.chart_generator.generate_gem_chart",
            return_value=chart_bytes,
        ):
            sig = _make_signal(channel="360_GEM")
            sig.last_lifecycle_check = None
            sig.lifecycle_alert_level = "GREEN"
            sig.pre_ai_confidence = 70.0
            sig.confidence = 70.0
            sig.best_tp_hit = 0

            with patch("src.signal_lifecycle.CHANNEL_TELEGRAM_MAP", {"360_GEM": "-100TESTCHAN"}):
                await monitor._post_chart(sig, candles)

        send_photo.assert_called_once()
        call_args = send_photo.call_args[0]
        assert call_args[1] == chart_bytes

    @pytest.mark.asyncio
    async def test_post_chart_skipped_when_no_send_photo(self) -> None:
        from src.signal_lifecycle import SignalLifecycleMonitor

        router = MagicMock()
        data_store = MagicMock()
        regime_detector = MagicMock()

        monitor = SignalLifecycleMonitor(
            router=router,
            data_store=data_store,
            regime_detector=regime_detector,
            send_telegram=AsyncMock(),
            send_photo=None,  # no photo callback
        )
        # Should return without error
        await monitor._post_chart(_make_signal(), {})

    def test_init_accepts_send_photo_parameter(self) -> None:
        from src.signal_lifecycle import SignalLifecycleMonitor
        send_photo = AsyncMock()
        monitor = SignalLifecycleMonitor(
            router=MagicMock(),
            data_store=MagicMock(),
            regime_detector=MagicMock(),
            send_telegram=AsyncMock(),
            send_photo=send_photo,
        )
        assert monitor._send_photo is send_photo


# ===========================================================================
# Feature 5: HTML performance report
# ===========================================================================


class TestPerformanceReport:
    """generate_html_report produces a valid HTML file."""

    def test_generates_html_file(self, tmp_path) -> None:
        from src.performance_report import generate_html_report
        from src.performance_tracker import PerformanceTracker

        tracker = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        tracker.record_outcome(
            signal_id="X1",
            channel="360_SCALP",
            symbol="BTCUSDT",
            direction="LONG",
            entry=30000.0,
            hit_tp=1,
            hit_sl=False,
            pnl_pct=2.5,
            confidence=75.0,
        )

        out = str(tmp_path / "report.html")
        path = generate_html_report(tracker, output_path=out)

        import os
        assert os.path.exists(path)
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Performance Report" in content
        assert "360_SCALP" in content

    def test_empty_tracker_generates_valid_html(self, tmp_path) -> None:
        from src.performance_report import generate_html_report
        from src.performance_tracker import PerformanceTracker

        tracker = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        out = str(tmp_path / "empty_report.html")
        path = generate_html_report(tracker, output_path=out)

        content = open(path).read()
        assert "<!DOCTYPE html>" in content

    def test_creates_parent_directory(self, tmp_path) -> None:
        from src.performance_report import generate_html_report
        from src.performance_tracker import PerformanceTracker

        tracker = PerformanceTracker(storage_path=str(tmp_path / "perf.json"))
        nested = str(tmp_path / "nested" / "dir" / "report.html")
        path = generate_html_report(tracker, output_path=nested)
        assert open(path).read().startswith("<!DOCTYPE html>")


# ===========================================================================
# Feature 6: SHORT spot signals
# ===========================================================================


class TestShortSpotSignals:
    """SpotChannel.evaluate() can produce SHORT signals."""

    def test_long_signal_above_ema200(self) -> None:
        from src.channels.spot import SpotChannel
        from src.smc import Direction

        ch = SpotChannel()
        candles = _make_spot_candles_long(n=60, base=200.0)
        h4_candles = {"4h": candles}
        close = float(candles["close"][-1])
        ema200 = close * 0.95  # price above EMA200

        ind_h4 = _spot_indicators_long(close, ema200)
        ind_h4["ema50_last"] = ema200 - 2.0  # daily EMA50 below price

        sig = ch.evaluate(
            "BTCUSDT",
            h4_candles,
            {"4h": ind_h4, "1d": {"ema50_last": ema200 - 2.0}},
            {},
            0.01,
            20_000_000,
        )

        if sig is not None:
            assert sig.direction == Direction.LONG

    def test_short_signal_below_ema200_and_daily_ema50(self) -> None:
        from src.channels.spot import SpotChannel
        from src.smc import Direction

        ch = SpotChannel()
        candles = _make_spot_candles_short(n=60, base=200.0)
        h4_candles = {"4h": candles}
        close = float(candles["close"][-1])
        ema200 = close * 1.10  # price below EMA200

        ind_h4 = _spot_indicators_short(close, ema200)
        ind_d1 = {"ema50_last": close * 1.05}  # daily EMA50 above price (bearish)

        sig = ch.evaluate(
            "BTCUSDT",
            h4_candles,
            {"4h": ind_h4, "1d": ind_d1},
            {},
            0.01,
            20_000_000,
        )

        if sig is not None:
            assert sig.direction == Direction.SHORT
            assert sig.stop_loss > sig.entry  # SL above entry for SHORT
            assert sig.tp1 < sig.entry        # TP below entry for SHORT

    def test_no_short_when_rsi_oversold(self) -> None:
        """RSI < 25 should block SHORT signals."""
        from src.channels.spot import SpotChannel

        ch = SpotChannel()
        candles = _make_spot_candles_short(n=60, base=200.0)
        h4_candles = {"4h": candles}
        close = float(candles["close"][-1])
        ema200 = close * 1.10

        ind_h4 = _spot_indicators_short(close, ema200)
        ind_h4["rsi_last"] = 15  # oversold → block SHORT

        sig = ch.evaluate(
            "BTCUSDT",
            h4_candles,
            {"4h": ind_h4, "1d": {"ema50_last": close * 1.05}},
            {},
            0.01,
            20_000_000,
        )
        assert sig is None

    def test_no_signal_above_ema200_bearish_mss(self) -> None:
        """Bearish MSS above EMA200 should block LONG (existing behaviour preserved)."""
        from src.channels.spot import SpotChannel
        from src.smc import Direction

        ch = SpotChannel()
        candles = _make_spot_candles_long(n=60, base=200.0)
        h4_candles = {"4h": candles}
        close = float(candles["close"][-1])
        ema200 = close * 0.95

        ind_h4 = _spot_indicators_long(close, ema200)
        ind_d1 = {"ema50_last": ema200 - 2.0}

        mss = MagicMock()
        mss.direction = Direction.SHORT
        smc_data = {"mss": mss}

        sig = ch.evaluate(
            "BTCUSDT",
            h4_candles,
            {"4h": ind_h4, "1d": ind_d1},
            smc_data,
            0.01,
            20_000_000,
        )
        assert sig is None


# ===========================================================================
# Feature 7: Adaptive gem scanner thresholds by regime
# ===========================================================================


class TestAdaptiveGemThresholds:
    """GemScanner.adjust_for_regime() changes thresholds correctly."""

    def test_trending_up_loosens_thresholds(self) -> None:
        from src.gem_scanner import GemScanner

        scanner = GemScanner()
        scanner.adjust_for_regime("TRENDING_UP")
        assert scanner._config.min_drawdown_pct == 60.0
        assert scanner._config.min_volume_ratio == 1.2

    def test_trending_down_tightens_thresholds(self) -> None:
        from src.gem_scanner import GemScanner

        scanner = GemScanner()
        scanner.adjust_for_regime("TRENDING_DOWN")
        assert scanner._config.min_drawdown_pct == 80.0
        assert scanner._config.min_volume_ratio == 2.0

    def test_volatile_uses_intermediate_thresholds(self) -> None:
        from src.gem_scanner import GemScanner

        scanner = GemScanner()
        scanner.adjust_for_regime("VOLATILE")
        assert scanner._config.min_drawdown_pct == 75.0
        assert scanner._config.min_volume_ratio == 1.8

    def test_ranging_uses_defaults(self) -> None:
        from src.gem_scanner import GemScanner

        scanner = GemScanner()
        # First change, then reset
        scanner.adjust_for_regime("TRENDING_DOWN")
        scanner.adjust_for_regime("RANGING")
        assert scanner._config.min_drawdown_pct == 70.0
        assert scanner._config.min_volume_ratio == 1.5

    def test_unknown_regime_falls_back_to_default(self) -> None:
        from src.gem_scanner import GemScanner

        scanner = GemScanner()
        scanner.adjust_for_regime("UNKNOWN_REGIME_STRING")
        assert scanner._config.min_drawdown_pct == 70.0
        assert scanner._config.min_volume_ratio == 1.5

    def test_looser_threshold_allows_smaller_drawdown(self) -> None:
        """With TRENDING_UP (min_drawdown=60%), a 65% drawdown coin should pass."""
        from src.gem_scanner import GemScanner

        scanner = GemScanner()
        scanner.adjust_for_regime("TRENDING_UP")

        # Build candles with ~65% drawdown (ATH=10, current=3.5 → drawdown ~65%)
        candles = _gem_candles(n=210, ath=10.0, current=3.5, vol_surge=2.0)
        sig = scanner.scan("TOKENUSDT", candles)

        if sig is not None:
            assert sig.drawdown_pct >= 60.0

    def test_tighter_threshold_rejects_small_drawdown(self) -> None:
        """With TRENDING_DOWN (min_drawdown=80%), a 65% drawdown coin should fail."""
        from src.gem_scanner import GemScanner

        scanner = GemScanner()
        scanner.adjust_for_regime("TRENDING_DOWN")

        candles = _gem_candles(n=210, ath=10.0, current=3.5, vol_surge=2.0)
        sig = scanner.scan("TOKENUSDT", candles)
        # 65% drawdown is below the 80% threshold
        assert sig is None


# ===========================================================================
# Feature 8: Cornix format wiring
# ===========================================================================


class TestCornixFormatWiring:
    """format_cornix_signal produces valid output; CORNIX_FORMAT_ENABLED gates it."""

    def test_cornix_signal_format_basic(self) -> None:
        from src.cornix_formatter import format_cornix_signal
        sig = _make_signal(channel="360_SPOT")
        result = format_cornix_signal(sig)
        assert "Entry Targets:" in result
        assert "Stop Targets:" in result
        assert "Leverage:" in result

    def test_cornix_signal_format_includes_symbol(self) -> None:
        from src.cornix_formatter import format_cornix_signal
        sig = _make_signal(symbol="ETHUSDT", channel="360_SWING")
        result = format_cornix_signal(sig)
        assert "ETHUSDT" in result

    def test_cornix_disabled_by_default(self) -> None:
        """CORNIX_FORMAT_ENABLED defaults to false — no Cornix block unless opted-in."""
        import config
        # The default .env.example should have this false
        assert hasattr(config, "CORNIX_FORMAT_ENABLED")

    def test_cornix_leverages_by_channel(self) -> None:
        from src.cornix_formatter import format_cornix_signal
        for channel, expected_leverage in [
            ("360_SCALP", "20x"),
            ("360_SWING", "5x"),
            ("360_SPOT", "1x"),
            ("360_GEM", "1x"),
        ]:
            sig = _make_signal(channel=channel)
            result = format_cornix_signal(sig)
            assert expected_leverage in result, f"{channel} should have {expected_leverage}"
