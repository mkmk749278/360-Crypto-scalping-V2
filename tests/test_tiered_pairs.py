"""Tests for the tiered pair universe (Part 2 of the SMC/sweep improvements).

Covers:
- PairTier enum classification
- PairInfo.tier field default
- refresh_pairs() tier assignment based on volume rank
- Pair pruning (removed symbols)
- Tier properties (tier1_symbols, tier2_symbols, tier3_symbols)
- check_promotions() — Tier 3 auto-promotion on volume surge
- Scanner: Tier 2 scan frequency (every N cycles)
- Scanner: Tier 2 pairs excluded from SCALP channel
- SMCDetector.detect() accepts lookback/tolerance_pct overrides
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.pair_manager import PairInfo, PairManager, PairTier
from src.detector import SMCDetector


# ---------------------------------------------------------------------------
# PairTier enum and PairInfo defaults
# ---------------------------------------------------------------------------


class TestPairTierEnum:
    def test_tier_values_exist(self):
        assert PairTier.TIER1 == "TIER1"
        assert PairTier.TIER2 == "TIER2"
        assert PairTier.TIER3 == "TIER3"

    def test_pairinfo_default_tier_is_tier1(self):
        p = PairInfo(symbol="BTCUSDT", market="futures")
        assert p.tier == PairTier.TIER1

    def test_pairinfo_tier_can_be_overridden(self):
        p = PairInfo(symbol="XYZUSDT", market="spot", tier=PairTier.TIER2)
        assert p.tier == PairTier.TIER2


# ---------------------------------------------------------------------------
# Tier properties
# ---------------------------------------------------------------------------


class TestTierProperties:
    def _make_pm_with_tiers(self) -> PairManager:
        """Return a PairManager with pre-populated pairs across tiers."""
        pm = PairManager.__new__(PairManager)
        pm.pairs = {
            "BTCUSDT": PairInfo("BTCUSDT", "futures", tier=PairTier.TIER1),
            "ETHUSDT": PairInfo("ETHUSDT", "futures", tier=PairTier.TIER1),
            "SOLUSDT": PairInfo("SOLUSDT", "spot", tier=PairTier.TIER2),
            "XRPUSDT": PairInfo("XRPUSDT", "spot", tier=PairTier.TIER2),
            "LTCUSDT": PairInfo("LTCUSDT", "spot", tier=PairTier.TIER3),
        }
        pm._prev_volumes = {}
        pm._spot_client = MagicMock()
        pm._futures_client = MagicMock()
        return pm

    def test_tier1_symbols(self):
        pm = self._make_pm_with_tiers()
        assert set(pm.tier1_symbols) == {"BTCUSDT", "ETHUSDT"}

    def test_tier2_symbols(self):
        pm = self._make_pm_with_tiers()
        assert set(pm.tier2_symbols) == {"SOLUSDT", "XRPUSDT"}

    def test_tier3_symbols(self):
        pm = self._make_pm_with_tiers()
        assert set(pm.tier3_symbols) == {"LTCUSDT"}

    def test_tier1_spot_symbols(self):
        pm = self._make_pm_with_tiers()
        # BTCUSDT and ETHUSDT are futures — no tier1 spot in this fixture
        assert pm.tier1_spot_symbols == []

    def test_tier1_futures_symbols(self):
        pm = self._make_pm_with_tiers()
        assert set(pm.tier1_futures_symbols) == {"BTCUSDT", "ETHUSDT"}

    def test_spot_symbols_includes_all_markets(self):
        pm = self._make_pm_with_tiers()
        assert set(pm.spot_symbols) == {"SOLUSDT", "XRPUSDT", "LTCUSDT"}

    def test_futures_symbols_includes_all_tiers(self):
        pm = self._make_pm_with_tiers()
        assert set(pm.futures_symbols) == {"BTCUSDT", "ETHUSDT"}


# ---------------------------------------------------------------------------
# refresh_pairs() — tier classification
# ---------------------------------------------------------------------------


def _make_ticker_data(symbols_with_vol: List[tuple]) -> List[dict]:
    """Build a minimal ticker response list from (symbol, volume) tuples."""
    return [{"symbol": sym, "quoteVolume": str(vol)} for sym, vol in symbols_with_vol]


class TestRefreshPairsTierClassification:
    @pytest.mark.asyncio
    async def test_first_pairs_are_tier1(self, monkeypatch):
        """The top TIER1_PAIR_COUNT symbols get assigned TIER1."""
        import src.pair_manager as pm_mod
        monkeypatch.setattr(pm_mod, "TIER1_PAIR_COUNT", 2)
        monkeypatch.setattr(pm_mod, "TIER2_PAIR_COUNT", 4)
        monkeypatch.setattr(pm_mod, "PAIR_PRUNE_ENABLED", False)

        ticker = _make_ticker_data([
            ("AAUSDT", 1_000_000),
            ("BBUSDT",   900_000),
            ("CCUSDT",   800_000),
            ("DDUSDT",   700_000),
            ("EEUSDT",   600_000),
        ])

        pm = pm_mod.PairManager.__new__(pm_mod.PairManager)
        pm.pairs = {}
        pm._prev_volumes = {}
        pm._spot_client = MagicMock()
        pm._futures_client = MagicMock()
        pm._spot_client._get = AsyncMock(return_value=ticker)
        pm._futures_client._get = AsyncMock(return_value=[])

        new_syms, removed_syms = await pm.refresh_pairs(market="spot")
        assert removed_syms == []
        assert "AAUSDT" in new_syms
        assert pm.pairs["AAUSDT"].tier == pm_mod.PairTier.TIER1
        assert pm.pairs["BBUSDT"].tier == pm_mod.PairTier.TIER1
        assert pm.pairs["CCUSDT"].tier == pm_mod.PairTier.TIER2
        assert pm.pairs["DDUSDT"].tier == pm_mod.PairTier.TIER2
        assert pm.pairs["EEUSDT"].tier == pm_mod.PairTier.TIER3

    @pytest.mark.asyncio
    async def test_returns_tuple(self):
        """refresh_pairs() must return (new_symbols, removed_symbols) tuple."""
        pm = PairManager.__new__(PairManager)
        pm.pairs = {}
        pm._prev_volumes = {}
        pm._spot_client = MagicMock()
        pm._futures_client = MagicMock()
        pm._spot_client._get = AsyncMock(return_value=[])
        pm._futures_client._get = AsyncMock(return_value=[])

        result = await pm.refresh_pairs()
        assert isinstance(result, tuple)
        assert len(result) == 2
        new_syms, removed_syms = result
        assert isinstance(new_syms, list)
        assert isinstance(removed_syms, list)

    @pytest.mark.asyncio
    async def test_new_symbols_added(self):
        """Symbols not in self.pairs are returned as new_symbols."""
        ticker = _make_ticker_data([("BTCUSDT", 5_000_000)])
        pm = PairManager.__new__(PairManager)
        pm.pairs = {}
        pm._prev_volumes = {}
        pm._spot_client = MagicMock()
        pm._futures_client = MagicMock()
        pm._spot_client._get = AsyncMock(return_value=ticker)
        pm._futures_client._get = AsyncMock(return_value=[])

        new_syms, _ = await pm.refresh_pairs(market="spot")
        assert "BTCUSDT" in new_syms

    @pytest.mark.asyncio
    async def test_existing_symbols_not_in_new_symbols(self):
        """Already-tracked symbols are NOT listed as new_symbols."""
        ticker = _make_ticker_data([("BTCUSDT", 5_000_000)])
        pm = PairManager.__new__(PairManager)
        pm.pairs = {"BTCUSDT": PairInfo("BTCUSDT", "spot", volume_24h_usd=4_000_000)}
        pm._prev_volumes = {}
        pm._spot_client = MagicMock()
        pm._futures_client = MagicMock()
        pm._spot_client._get = AsyncMock(return_value=ticker)
        pm._futures_client._get = AsyncMock(return_value=[])

        new_syms, _ = await pm.refresh_pairs(market="spot")
        assert "BTCUSDT" not in new_syms


# ---------------------------------------------------------------------------
# Pair pruning
# ---------------------------------------------------------------------------


class TestPairPruning:
    @pytest.mark.asyncio
    async def test_stale_pair_removed_when_pruning_enabled(self, monkeypatch):
        """Pairs absent from the exchange response are pruned when PAIR_PRUNE_ENABLED."""
        import src.pair_manager as pm_mod
        monkeypatch.setattr(pm_mod, "PAIR_PRUNE_ENABLED", True)
        monkeypatch.setattr(pm_mod, "TIER1_PAIR_COUNT", 10)
        monkeypatch.setattr(pm_mod, "TIER2_PAIR_COUNT", 20)

        # Previously tracked: BTCUSDT + DEADUSDT (no longer on exchange)
        pm = pm_mod.PairManager.__new__(pm_mod.PairManager)
        pm.pairs = {
            "BTCUSDT": pm_mod.PairInfo("BTCUSDT", "spot", volume_24h_usd=1_000_000),
            "DEADUSDT": pm_mod.PairInfo("DEADUSDT", "spot", volume_24h_usd=500_000),
        }
        pm._prev_volumes = {}
        pm._spot_client = MagicMock()
        pm._futures_client = MagicMock()
        # Exchange only returns BTCUSDT — DEADUSDT is "delisted"
        pm._spot_client._get = AsyncMock(return_value=[{"symbol": "BTCUSDT", "quoteVolume": "1000000"}])
        pm._futures_client._get = AsyncMock(return_value=[])

        new_syms, removed_syms = await pm.refresh_pairs()
        assert "DEADUSDT" in removed_syms
        assert "DEADUSDT" not in pm.pairs
        assert "BTCUSDT" in pm.pairs

    @pytest.mark.asyncio
    async def test_no_pruning_when_disabled(self, monkeypatch):
        """Stale pairs are preserved when PAIR_PRUNE_ENABLED=False."""
        import src.pair_manager as pm_mod
        monkeypatch.setattr(pm_mod, "PAIR_PRUNE_ENABLED", False)
        monkeypatch.setattr(pm_mod, "TIER1_PAIR_COUNT", 10)
        monkeypatch.setattr(pm_mod, "TIER2_PAIR_COUNT", 20)

        pm = pm_mod.PairManager.__new__(pm_mod.PairManager)
        pm.pairs = {
            "BTCUSDT": pm_mod.PairInfo("BTCUSDT", "spot", volume_24h_usd=1_000_000),
            "OLDUSDT": pm_mod.PairInfo("OLDUSDT", "spot", volume_24h_usd=500_000),
        }
        pm._prev_volumes = {}
        pm._spot_client = MagicMock()
        pm._futures_client = MagicMock()
        pm._spot_client._get = AsyncMock(return_value=[{"symbol": "BTCUSDT", "quoteVolume": "1000000"}])
        pm._futures_client._get = AsyncMock(return_value=[])

        _, removed_syms = await pm.refresh_pairs()
        assert removed_syms == []
        assert "OLDUSDT" in pm.pairs  # not removed


# ---------------------------------------------------------------------------
# check_promotions() — Tier 3 auto-promotion
# ---------------------------------------------------------------------------


class TestCheckPromotions:
    def _make_pm(self) -> PairManager:
        pm = PairManager.__new__(PairManager)
        pm.pairs = {}
        pm._prev_volumes = {}
        pm._spot_client = MagicMock()
        pm._futures_client = MagicMock()
        return pm

    def test_tier3_promoted_on_volume_surge(self, monkeypatch):
        """A Tier 3 pair with volume >= surge_multiplier × prev_vol is promoted."""
        import src.pair_manager as pm_mod
        monkeypatch.setattr(pm_mod, "TIER3_VOLUME_SURGE_MULTIPLIER", 3.0)

        pm = pm_mod.PairManager.__new__(pm_mod.PairManager)
        pm._spot_client = MagicMock()
        pm._futures_client = MagicMock()
        pm.pairs = {
            "GEMTOKEN": pm_mod.PairInfo(
                "GEMTOKEN", "spot",
                volume_24h_usd=3_000_000,   # 3× the previous volume
                tier=pm_mod.PairTier.TIER3,
            )
        }
        pm._prev_volumes = {"GEMTOKEN": 1_000_000}

        promoted = pm.check_promotions()
        assert "GEMTOKEN" in promoted
        assert pm.pairs["GEMTOKEN"].tier == pm_mod.PairTier.TIER2

    def test_tier3_not_promoted_below_threshold(self, monkeypatch):
        """Tier 3 pair with insufficient volume surge stays in Tier 3."""
        import src.pair_manager as pm_mod
        monkeypatch.setattr(pm_mod, "TIER3_VOLUME_SURGE_MULTIPLIER", 3.0)

        pm = pm_mod.PairManager.__new__(pm_mod.PairManager)
        pm._spot_client = MagicMock()
        pm._futures_client = MagicMock()
        pm.pairs = {
            "NORMALUSDT": pm_mod.PairInfo(
                "NORMALUSDT", "spot",
                volume_24h_usd=1_500_000,   # only 1.5× — below 3.0× threshold
                tier=pm_mod.PairTier.TIER3,
            )
        }
        pm._prev_volumes = {"NORMALUSDT": 1_000_000}

        promoted = pm.check_promotions()
        assert "NORMALUSDT" not in promoted
        assert pm.pairs["NORMALUSDT"].tier == pm_mod.PairTier.TIER3

    def test_tier1_not_considered_for_promotion(self):
        """Only Tier 3 pairs are eligible for promotion."""
        pm = PairManager.__new__(PairManager)
        pm._spot_client = MagicMock()
        pm._futures_client = MagicMock()
        pm.pairs = {
            "BTCUSDT": PairInfo("BTCUSDT", "futures", volume_24h_usd=100_000_000, tier=PairTier.TIER1),
        }
        pm._prev_volumes = {"BTCUSDT": 1_000}  # extreme surge — but it's Tier 1
        promoted = pm.check_promotions()
        assert promoted == []

    def test_no_prev_volume_no_promotion(self):
        """Pairs without a recorded previous volume are not promoted."""
        pm = PairManager.__new__(PairManager)
        pm._spot_client = MagicMock()
        pm._futures_client = MagicMock()
        pm.pairs = {
            "NEWTOKEN": PairInfo("NEWTOKEN", "spot", volume_24h_usd=5_000_000, tier=PairTier.TIER3),
        }
        pm._prev_volumes = {}  # no previous volume recorded
        promoted = pm.check_promotions()
        assert promoted == []


# ---------------------------------------------------------------------------
# Scanner: Tier 2 scan frequency (every N cycles)
# ---------------------------------------------------------------------------


class TestScannerTier2Frequency:
    """Verify that Tier 2 pairs are only included in the scan on every Nth cycle."""

    def _make_minimal_scanner(self):
        """Create a minimal Scanner-like object for testing tiered scan logic."""
        from src.scanner import Scanner
        scanner = Scanner.__new__(Scanner)
        scanner._scan_cycle_count = 0
        scanner._last_tier3_scan_time = 0.0
        scanner.pair_mgr = MagicMock()
        scanner.pair_mgr.pairs = {
            "BTCUSDT": PairInfo("BTCUSDT", "futures", volume_24h_usd=10_000_000, tier=PairTier.TIER1),
            "SOLUSDT": PairInfo("SOLUSDT", "spot", volume_24h_usd=2_000_000, tier=PairTier.TIER2),
            "XYZUSDT": PairInfo("XYZUSDT", "spot", volume_24h_usd=100_000, tier=PairTier.TIER3),
        }
        return scanner

    def test_tier2_excluded_on_non_nth_cycle(self, monkeypatch):
        """On cycles that are NOT divisible by TIER2_SCAN_EVERY_N_CYCLES,
        Tier 2 pairs are excluded from the scan."""
        import config as cfg
        monkeypatch.setattr(cfg, "TIER2_SCAN_EVERY_N_CYCLES", 3)

        scanner = self._make_minimal_scanner()
        scanner._scan_cycle_count = 1  # not divisible by 3

        scan_tier2 = (scanner._scan_cycle_count % cfg.TIER2_SCAN_EVERY_N_CYCLES == 0)
        pairs_in_cycle = [
            (sym, info)
            for sym, info in scanner.pair_mgr.pairs.items()
            if info.tier == PairTier.TIER1
            or (info.tier == PairTier.TIER2 and scan_tier2)
        ]
        symbols = [s for s, _ in pairs_in_cycle]
        assert "BTCUSDT" in symbols
        assert "SOLUSDT" not in symbols  # Tier 2 skipped
        assert "XYZUSDT" not in symbols  # Tier 3 always excluded from main scan

    def test_tier2_included_on_nth_cycle(self, monkeypatch):
        """On cycles divisible by TIER2_SCAN_EVERY_N_CYCLES, Tier 2 pairs
        are included."""
        import config as cfg
        monkeypatch.setattr(cfg, "TIER2_SCAN_EVERY_N_CYCLES", 3)

        scanner = self._make_minimal_scanner()
        scanner._scan_cycle_count = 3  # divisible by 3

        scan_tier2 = (scanner._scan_cycle_count % cfg.TIER2_SCAN_EVERY_N_CYCLES == 0)
        pairs_in_cycle = [
            (sym, info)
            for sym, info in scanner.pair_mgr.pairs.items()
            if info.tier == PairTier.TIER1
            or (info.tier == PairTier.TIER2 and scan_tier2)
        ]
        symbols = [s for s, _ in pairs_in_cycle]
        assert "BTCUSDT" in symbols
        assert "SOLUSDT" in symbols   # Tier 2 included on 3rd cycle
        assert "XYZUSDT" not in symbols  # Tier 3 still excluded


# ---------------------------------------------------------------------------
# Scanner: Tier 2 excluded from SCALP channel
# ---------------------------------------------------------------------------


class TestScannerTier2SkipsScalp:
    """Verify _should_skip_channel() returns True for Tier 2 pairs on SCALP."""

    def _make_scanner(self):
        from src.scanner import Scanner
        from src.signal_quality import MarketState
        from collections import defaultdict

        scanner = Scanner.__new__(Scanner)
        scanner._scan_cycle_count = 0
        scanner._last_tier3_scan_time = 0.0
        scanner.paused_channels = set()
        scanner._cooldown_until = {}
        scanner._regime_history = {}
        scanner.circuit_breaker = None
        scanner._suppression_counters = defaultdict(int)

        # Minimal router mock
        scanner.router = MagicMock()
        scanner.router.active_signals = {}

        # Pair manager with one Tier 2 pair
        scanner.pair_mgr = MagicMock()
        scanner.pair_mgr.pairs = {
            "TIER2USDT": PairInfo("TIER2USDT", "spot", volume_24h_usd=1_000_000, tier=PairTier.TIER2),
            "TIER1USDT": PairInfo("TIER1USDT", "futures", volume_24h_usd=5_000_000, tier=PairTier.TIER1),
        }

        # Minimal scan context mock — use a real MarketState value
        ctx = MagicMock()
        ctx.pair_quality.passed = True
        ctx.market_state = MarketState.STRONG_TREND
        ctx.is_ranging = False
        ctx.adx_val = 30.0
        ctx.regime_result.regime.value = "TRENDING_UP"
        scanner._ctx = ctx

        return scanner

    def test_tier2_skips_scalp_channel(self):
        """_should_skip_channel returns True for a Tier 2 pair on SCALP."""
        scanner = self._make_scanner()
        ctx = scanner._ctx

        result = scanner._should_skip_channel("TIER2USDT", "360_SCALP", ctx)
        assert result is True, "Tier 2 pair must be skipped for SCALP channel"

    def test_tier2_allowed_on_swing_channel(self):
        """_should_skip_channel returns False for a Tier 2 pair on SWING."""
        scanner = self._make_scanner()
        ctx = scanner._ctx
        result = scanner._should_skip_channel("TIER2USDT", "360_SWING", ctx)
        assert result is False, "Tier 2 pair must NOT be skipped for SWING channel"

    def test_tier1_not_skipped_on_scalp(self):
        """_should_skip_channel returns False for a Tier 1 pair on SCALP."""
        scanner = self._make_scanner()
        ctx = scanner._ctx
        result = scanner._should_skip_channel("TIER1USDT", "360_SCALP", ctx)
        assert result is False, "Tier 1 pair must NOT be skipped for SCALP channel"


# ---------------------------------------------------------------------------
# SMCDetector.detect() — lookback/tolerance_pct overrides
# ---------------------------------------------------------------------------


class TestSMCDetectorLookbackOverride:
    """Verify that SMCDetector.detect() forwards custom lookback/tolerance
    to detect_liquidity_sweeps()."""

    def _make_candles(self, n=25, sweep_high=True) -> Dict[str, Any]:
        """Build a minimal candles dict with a detectable sweep on 5m."""
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0
        if sweep_high:
            # Bearish sweep: wick above 105, close back inside (within 0.15%)
            high[-1] = 107.0
            close[-1] = 105.12
        return {"5m": {"high": high, "low": low, "close": close, "open": close.copy()}}

    def test_detect_with_scalp_params_finds_sweep(self):
        """With lookback=20 and tolerance=0.15, detector finds the sweep."""
        detector = SMCDetector()
        candles = self._make_candles(n=25, sweep_high=True)
        result = detector.detect("BTCUSDT", candles, [], lookback=20, tolerance_pct=0.15)
        assert len(result.sweeps) >= 1

    def test_detect_with_tight_tolerance_misses_sweep(self):
        """With the default tight tolerance=0.05, the same sweep is missed
        (close is 0.114% above level — beyond the 0.05% window)."""
        detector = SMCDetector()
        candles = self._make_candles(n=25, sweep_high=True)
        result = detector.detect("BTCUSDT", candles, [], lookback=20, tolerance_pct=0.05)
        # close[-1] = 105.12, level = 105.0, tol = 105 * 0.0005 = 0.0525
        # close must be ≤ 105.0525 — 105.12 exceeds this → no sweep
        assert len(result.sweeps) == 0

    def test_detect_default_params_preserved(self):
        """Default detect() call (no overrides) uses lookback=50/tolerance=0.05."""
        detector = SMCDetector()
        # Only 25 candles — below default lookback=50 → no sweeps (insufficient data)
        candles = self._make_candles(n=25, sweep_high=True)
        result = detector.detect("BTCUSDT", candles, [])
        # 25 < 50 + 1 = 51 → skipped
        assert len(result.sweeps) == 0

    def test_detect_with_enough_candles_for_default(self):
        """With 55 candles and default params, detect() can process the data."""
        detector = SMCDetector()
        n = 55
        high = np.ones(n) * 105.0
        low = np.ones(n) * 95.0
        close = np.ones(n) * 100.0
        high[-1] = 107.0
        close[-1] = 105.04  # within default 0.05% tolerance
        candles = {"5m": {"high": high, "low": low, "close": close, "open": close.copy()}}
        result = detector.detect("BTCUSDT", candles, [])
        assert isinstance(result.sweeps, list)
