"""Regime-Adaptive Signal Scheduling — dynamic channel allowlists.

Determines which channels are allowed to fire for each pair based on
the current market regime.  QUIET pairs skip scalp channels to reduce
false triggers; RANGING/TRENDING pairs get priority scalp execution.

PR 07 Implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

from src.utils import get_logger

log = get_logger("regime_manager")


# Regime-channel scheduling matrix.
# Maps each regime to the set of channel families that are ALLOWED.
# Channels not in the allowed set for the current regime are skipped.
_REGIME_ALLOWED_CHANNELS: Dict[str, FrozenSet[str]] = {
    "TRENDING_UP": frozenset({
        "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD", "360_SCALP_OBI",
        "360_SWING", "360_SPOT", "360_GEM",
    }),
    "TRENDING_DOWN": frozenset({
        "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD", "360_SCALP_OBI",
        "360_SWING", "360_SPOT", "360_GEM",
    }),
    "RANGING": frozenset({
        "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD",
        "360_SCALP_VWAP", "360_SCALP_OBI",
        "360_SWING", "360_SPOT", "360_GEM",
    }),
    "VOLATILE": frozenset({
        "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD", "360_SCALP_OBI",
        "360_SPOT", "360_GEM",
        # SWING blocked in VOLATILE (chaotic, stops get swept)
    }),
    "QUIET": frozenset({
        # Only VWAP scalp variant is fully blocked; other scalp channels
        # are allowed but with elevated thresholds (handled by scanner).
        "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD", "360_SCALP_OBI",
        "360_SWING", "360_SPOT", "360_GEM",
    }),
}

# Channels that receive PRIORITY scheduling in specific regimes.
# Priority channels are evaluated first and with relaxed thresholds.
_REGIME_PRIORITY_CHANNELS: Dict[str, FrozenSet[str]] = {
    "TRENDING_UP": frozenset({"360_SCALP", "360_SCALP_FVG", "360_SWING"}),
    "TRENDING_DOWN": frozenset({"360_SCALP", "360_SCALP_FVG", "360_SWING"}),
    "RANGING": frozenset({"360_SCALP_VWAP", "360_SCALP", "360_SCALP_CVD"}),
    "VOLATILE": frozenset({"360_SCALP_OBI", "360_SCALP_CVD"}),
    "QUIET": frozenset({"360_SPOT", "360_GEM"}),
}


@dataclass
class RegimeSchedule:
    """Scheduling decision for a pair in a given regime."""

    regime: str
    allowed_channels: FrozenSet[str] = field(default_factory=frozenset)
    priority_channels: FrozenSet[str] = field(default_factory=frozenset)
    skipped_channels: List[str] = field(default_factory=list)


class RegimeManager:
    """Manages regime-adaptive channel scheduling.

    Determines which channels should be evaluated for each pair based
    on the current market regime, and tracks which channels were skipped.
    """

    def __init__(
        self,
        allowed_map: Optional[Dict[str, FrozenSet[str]]] = None,
        priority_map: Optional[Dict[str, FrozenSet[str]]] = None,
    ) -> None:
        self._allowed = allowed_map or _REGIME_ALLOWED_CHANNELS
        self._priority = priority_map or _REGIME_PRIORITY_CHANNELS
        self._skip_counts: Dict[str, int] = {}

    def get_schedule(self, regime: str) -> RegimeSchedule:
        """Return the channel schedule for the given regime.

        Parameters
        ----------
        regime:
            Market regime string (e.g. ``"TRENDING_UP"``).

        Returns
        -------
        RegimeSchedule
            Contains allowed and priority channels for the regime.
        """
        regime_upper = regime.upper() if regime else "RANGING"
        allowed = self._allowed.get(regime_upper, frozenset())
        priority = self._priority.get(regime_upper, frozenset())

        # If no explicit allowlist for this regime, allow everything
        if not allowed:
            allowed = frozenset({
                "360_SCALP", "360_SCALP_FVG", "360_SCALP_CVD",
                "360_SCALP_VWAP", "360_SCALP_OBI",
                "360_SWING", "360_SPOT", "360_GEM",
            })

        return RegimeSchedule(
            regime=regime_upper,
            allowed_channels=allowed,
            priority_channels=priority,
        )

    def is_channel_allowed(self, channel: str, regime: str) -> bool:
        """Check if a channel is allowed in the current regime."""
        schedule = self.get_schedule(regime)
        return channel in schedule.allowed_channels

    def is_channel_priority(self, channel: str, regime: str) -> bool:
        """Check if a channel has priority status in the current regime."""
        schedule = self.get_schedule(regime)
        return channel in schedule.priority_channels

    def filter_channels(
        self,
        channels: List[Any],
        regime: str,
    ) -> tuple[List[Any], List[str]]:
        """Filter and sort channels based on regime scheduling.

        Parameters
        ----------
        channels:
            List of channel objects (must have ``.config.name`` attribute).
        regime:
            Current market regime.

        Returns
        -------
        tuple[list, list[str]]
            ``(allowed_channels, skipped_channel_names)``
            Allowed channels are sorted with priority channels first.
        """
        schedule = self.get_schedule(regime)
        allowed = []
        skipped = []
        priority = []

        for chan in channels:
            name = chan.config.name
            if name in schedule.allowed_channels:
                if name in schedule.priority_channels:
                    priority.append(chan)
                else:
                    allowed.append(chan)
            else:
                skipped.append(name)
                self._skip_counts[name] = self._skip_counts.get(name, 0) + 1

        if skipped:
            log.debug(
                "Regime {} — skipped channels: {}",
                regime, ", ".join(skipped),
            )

        # Priority channels first, then remaining allowed channels
        return priority + allowed, skipped

    def get_skip_stats(self) -> Dict[str, int]:
        """Return cumulative skip counts per channel."""
        return dict(self._skip_counts)
