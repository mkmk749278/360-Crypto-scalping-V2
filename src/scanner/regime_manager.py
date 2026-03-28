"""Regime-Adaptive Signal Scheduler (PR07).

Determines which scalp channels are permitted to generate signals for a given
pair and market regime so that:

* QUIET pairs skip scalp channels (too little momentum → high false-signal rate).
* RANGING / QUIET → mean-reversion channels prioritised (RANGE_FADE, VWAP).
* TRENDING → trend-following channels prioritised (standard SCALP, FVG).
* VOLATILE → order-flow channels prioritised (OBI, CVD).

Usage::

    from src.scanner.regime_manager import RegimeChannelScheduler

    scheduler = RegimeChannelScheduler()
    allowed = scheduler.get_allowed_channels("RANGING", all_channel_names)
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Optional, Sequence

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime → blocked channels (hard gate)
# ---------------------------------------------------------------------------
# These mappings duplicate (and extend) the existing scanner constants so that
# the regime_manager is authoritative for regime-aware scheduling.  The
# scanner can delegate to this module instead of maintaining its own maps.

_HARD_BLOCKED: Dict[str, FrozenSet[str]] = {
    # VWAP signals are meaningless without sufficient trading volume
    "QUIET": frozenset({"360_SCALP_VWAP"}),
    # Swing trades in chaotic regimes produce excessive stop-outs
    "VOLATILE": frozenset({"360_SWING"}),
    "DIRTY_RANGE": frozenset({"360_SWING", "360_SPOT"}),
}

# ---------------------------------------------------------------------------
# Regime → priority channels (soft boost for confidence scoring)
# ---------------------------------------------------------------------------
_PRIORITY_CHANNELS: Dict[str, List[str]] = {
    "TRENDING_UP":   ["360_SCALP", "360_SCALP_FVG", "360_SWING"],
    "TRENDING_DOWN": ["360_SCALP", "360_SCALP_FVG", "360_SWING"],
    "RANGING":       ["360_SCALP", "360_SCALP_VWAP", "360_SCALP_CVD"],
    "QUIET":         ["360_SCALP"],          # Only trend-reversal scalp survives
    "VOLATILE":      ["360_SCALP_OBI", "360_SCALP_CVD"],
}


class RegimeChannelScheduler:
    """Determines which channels are allowed / prioritised for a given regime.

    Parameters
    ----------
    extra_blocked:
        Additional channel → blocked-regimes map to merge with the defaults.
        Useful for per-deployment overrides without patching this module.
    """

    def __init__(
        self,
        extra_blocked: Optional[Dict[str, Sequence[str]]] = None,
    ) -> None:
        # Build the internal blocked map: regime → frozenset(blocked_channels)
        self._blocked: Dict[str, FrozenSet[str]] = dict(_HARD_BLOCKED)
        if extra_blocked:
            for channel, blocked_regimes in extra_blocked.items():
                for regime in blocked_regimes:
                    existing = self._blocked.get(regime, frozenset())
                    self._blocked[regime] = existing | frozenset({channel})

    def is_channel_allowed(self, channel_name: str, regime: str) -> bool:
        """Return False when *channel_name* is hard-blocked for *regime*.

        Parameters
        ----------
        channel_name:
            The channel's name string (e.g. ``"360_SCALP_VWAP"``).
        regime:
            Current market regime.

        Returns
        -------
        bool
        """
        regime_upper = regime.upper() if regime else ""
        blocked = self._blocked.get(regime_upper, frozenset())
        if channel_name in blocked:
            log.debug(
                "RegimeChannelScheduler: %s hard-blocked in %s",
                channel_name, regime_upper,
            )
            return False
        return True

    def get_allowed_channels(
        self,
        regime: str,
        all_channels: Sequence[str],
    ) -> List[str]:
        """Return the subset of *all_channels* that are allowed for *regime*.

        Parameters
        ----------
        regime:
            Current market regime string.
        all_channels:
            Full list of channel names configured in the scanner.

        Returns
        -------
        list[str]
            Channels that may run in the given regime, in input order.
        """
        allowed = [c for c in all_channels if self.is_channel_allowed(c, regime)]
        blocked_count = len(all_channels) - len(allowed)
        if blocked_count > 0:
            log.info(
                "RegimeChannelScheduler [%s]: %d/%d channels allowed "
                "(%d hard-blocked)",
                regime, len(allowed), len(all_channels), blocked_count,
            )
        return allowed

    def get_priority_channels(self, regime: str) -> List[str]:
        """Return the channels that should be prioritised for *regime*.

        Priority channels receive a soft confidence boost in the scanner.
        The list is returned in priority order (most preferred first).

        Parameters
        ----------
        regime:
            Current market regime string.

        Returns
        -------
        list[str]
            Prioritised channel names.  May be empty if the regime has no
            specific priority mapping.
        """
        return list(_PRIORITY_CHANNELS.get(regime.upper() if regime else "", []))

    def log_skipped_pairs(
        self,
        skipped: List[str],
        regime: str,
        channel_name: str,
    ) -> None:
        """Log pairs that were skipped due to regime suppression.

        Parameters
        ----------
        skipped:
            List of pair symbols that produced no signal this cycle.
        regime:
            The regime that caused the skip.
        channel_name:
            The channel that was blocked.
        """
        if not skipped:
            return
        log.info(
            "Regime-suppressed [%s / %s]: %d pairs skipped: %s%s",
            channel_name, regime, len(skipped),
            ", ".join(skipped[:5]),
            f" … (+{len(skipped) - 5} more)" if len(skipped) > 5 else "",
        )
