"""Tests for Phase 1: merged Telegram channel routing (config._build_channel_telegram_map)."""

from __future__ import annotations

import importlib
import os
from unittest import mock


def _build_map(**env_overrides):
    """Re-import config with specific env vars set and return CHANNEL_TELEGRAM_MAP."""
    env = {
        "TELEGRAM_SCALP_CHANNEL_ID": "scalp_id",
        "TELEGRAM_SWING_CHANNEL_ID": "swing_id",
        "TELEGRAM_SPOT_CHANNEL_ID": "spot_id",
        "TELEGRAM_GEM_CHANNEL_ID": "gem_id",
        "TELEGRAM_ACTIVE_CHANNEL_ID": "",
        "TELEGRAM_PORTFOLIO_CHANNEL_ID": "",
        **env_overrides,
    }
    with mock.patch.dict(os.environ, env, clear=False):
        import config as cfg_module
        importlib.reload(cfg_module)
        return cfg_module._build_channel_telegram_map()


class TestBuildChannelTelegramMap:
    def test_with_merged_env_vars_routes_scalp_swing_to_active(self):
        """When ACTIVE + PORTFOLIO are set, all SCALP/SWING → active, SPOT/GEM → portfolio."""
        env = {
            "TELEGRAM_ACTIVE_CHANNEL_ID": "active_id",
            "TELEGRAM_PORTFOLIO_CHANNEL_ID": "portfolio_id",
        }
        mapping = _build_map(**env)

        # All SCALP* and SWING → active_id
        for ch in ("360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD", "360_SCALP_VWAP", "360_SCALP_OBI", "360_SWING"):
            assert mapping[ch] == "active_id", f"{ch} should route to active_id"

        # SPOT and GEM → portfolio_id
        for ch in ("360_SPOT", "360_GEM"):
            assert mapping[ch] == "portfolio_id", f"{ch} should route to portfolio_id"

    def test_without_merged_env_vars_falls_back_to_individual(self):
        """When merged vars are empty, routing falls back to per-channel IDs."""
        mapping = _build_map(
            TELEGRAM_ACTIVE_CHANNEL_ID="",
            TELEGRAM_PORTFOLIO_CHANNEL_ID="",
        )
        assert mapping["360_SCALP"] == "scalp_id"
        assert mapping["360_SCALP_FVG"] == "scalp_id"
        assert mapping["360_SCALP_CVD"] == "scalp_id"
        assert mapping["360_SCALP_VWAP"] == "scalp_id"
        assert mapping["360_SCALP_OBI"] == "scalp_id"
        assert mapping["360_SWING"] == "swing_id"
        assert mapping["360_SPOT"] == "spot_id"
        assert mapping["360_GEM"] == "gem_id"

    def test_partial_merge_only_active_set(self):
        """Only ACTIVE set → SCALP/SWING route to active_id; SPOT/GEM fallback to individuals."""
        mapping = _build_map(
            TELEGRAM_ACTIVE_CHANNEL_ID="active_id",
            TELEGRAM_PORTFOLIO_CHANNEL_ID="",
        )
        for ch in ("360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD", "360_SCALP_VWAP", "360_SCALP_OBI", "360_SWING"):
            assert mapping[ch] == "active_id"
        assert mapping["360_SPOT"] == "spot_id"
        assert mapping["360_GEM"] == "gem_id"

    def test_partial_merge_only_portfolio_set(self):
        """Only PORTFOLIO set → SPOT/GEM route to portfolio_id; SCALP/SWING fallback."""
        mapping = _build_map(
            TELEGRAM_ACTIVE_CHANNEL_ID="",
            TELEGRAM_PORTFOLIO_CHANNEL_ID="portfolio_id",
        )
        for ch in ("360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD", "360_SCALP_VWAP", "360_SCALP_OBI", "360_SWING"):
            assert mapping[ch] == "scalp_id" if "SCALP" in ch else "swing_id"
        assert mapping["360_SPOT"] == "portfolio_id"
        assert mapping["360_GEM"] == "portfolio_id"

    def test_all_eight_channels_present(self):
        """The map always contains exactly 8 channel keys."""
        mapping = _build_map()
        expected_keys = {
            "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD",
            "360_SCALP_VWAP", "360_SCALP_OBI", "360_SWING",
            "360_SPOT", "360_GEM",
        }
        assert set(mapping.keys()) == expected_keys
