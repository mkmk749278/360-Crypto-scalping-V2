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
        **env_overrides,
    }
    with mock.patch.dict(os.environ, env, clear=False):
        import config as cfg_module
        importlib.reload(cfg_module)
        return cfg_module._build_channel_telegram_map()


class TestBuildChannelTelegramMap:
    def test_with_merged_env_var_routes_all_to_active(self):
        """When ACTIVE is set, all channels → active_id."""
        env = {
            "TELEGRAM_ACTIVE_CHANNEL_ID": "active_id",
        }
        mapping = _build_map(**env)

        for ch in ("360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD", "360_SCALP_VWAP", "360_SCALP_OBI", "360_SWING", "360_SPOT", "360_GEM"):
            assert mapping[ch] == "active_id", f"{ch} should route to active_id"

    def test_without_merged_env_vars_falls_back_to_individual(self):
        """When merged var is empty, routing falls back to per-channel IDs."""
        mapping = _build_map(
            TELEGRAM_ACTIVE_CHANNEL_ID="",
        )
        assert mapping["360_SCALP"] == "scalp_id"
        assert mapping["360_SCALP_FVG"] == "scalp_id"
        assert mapping["360_SCALP_CVD"] == "scalp_id"
        assert mapping["360_SCALP_VWAP"] == "scalp_id"
        assert mapping["360_SCALP_OBI"] == "scalp_id"
        assert mapping["360_SWING"] == "swing_id"
        assert mapping["360_SPOT"] == "spot_id"
        assert mapping["360_GEM"] == "gem_id"

    def test_active_set_routes_all_to_active(self):
        """ACTIVE set → all channels route to active_id."""
        mapping = _build_map(
            TELEGRAM_ACTIVE_CHANNEL_ID="active_id",
        )
        for ch in ("360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD", "360_SCALP_VWAP", "360_SCALP_OBI", "360_SWING", "360_SPOT", "360_GEM"):
            assert mapping[ch] == "active_id"

    def test_all_eight_channels_present(self):
        """The map always contains exactly 8 channel keys."""
        mapping = _build_map()
        expected_keys = {
            "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD",
            "360_SCALP_VWAP", "360_SCALP_OBI", "360_SWING",
            "360_SPOT", "360_GEM",
        }
        assert set(mapping.keys()) == expected_keys
