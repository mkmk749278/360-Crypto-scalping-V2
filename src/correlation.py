"""Correlation-aware position limiting.

Prevents the system from filling all concurrent signal slots with
highly correlated positions (e.g., all LONG on BTC-correlated alts).
"""

from __future__ import annotations

import os
from typing import Dict, List, Set, Tuple

from src.utils import get_logger

log = get_logger("correlation")

# Maximum number of same-direction positions allowed within a single
# correlation group.  Env-overridable.
MAX_SAME_DIRECTION_PER_GROUP: int = int(
    os.getenv("MAX_SAME_DIRECTION_PER_GROUP", "3")
)

# Correlation groups – symbols that tend to move together.
# A symbol can appear in multiple groups.
CORRELATION_GROUPS: Dict[str, List[str]] = {
    "BTC_ECOSYSTEM": [
        "BTCUSDT", "BTCBUSD", "WBTCUSDT",
    ],
    "MAJOR_ALTS": [
        "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
        "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT",
        "NEARUSDT", "ATOMUSDT",
    ],
    "MEME": [
        "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT",
    ],
    "DEFI": [
        "UNIUSDT", "AAVEUSDT", "MKRUSDT", "COMPUSDT",
        "SUSHIUSDT", "CRVUSDT",
    ],
    "LAYER2": [
        "ARBUSDT", "OPUSDT", "STXUSDT", "IMXUSDT",
    ],
}

# Build reverse lookup: symbol → set of group names
_SYMBOL_TO_GROUPS: Dict[str, Set[str]] = {}
for _group_name, _symbols in CORRELATION_GROUPS.items():
    for _sym in _symbols:
        _SYMBOL_TO_GROUPS.setdefault(_sym, set()).add(_group_name)


def get_correlation_groups(symbol: str) -> Set[str]:
    """Return the set of correlation group names that *symbol* belongs to."""
    return _SYMBOL_TO_GROUPS.get(symbol, set())


def check_correlation_limit(
    symbol: str,
    direction: str,
    active_positions: Dict[str, Tuple[str, str]],
    max_per_group: int = MAX_SAME_DIRECTION_PER_GROUP,
) -> Tuple[bool, str]:
    """Check whether adding a new position would exceed correlation limits.

    Parameters
    ----------
    symbol:
        The symbol of the new signal (e.g. ``"SOLUSDT"``).
    direction:
        ``"LONG"`` or ``"SHORT"``.
    active_positions:
        Dict mapping signal_id → (symbol, direction) for all currently
        active signals.
    max_per_group:
        Maximum same-direction positions allowed per correlation group.

    Returns
    -------
    (allowed, reason)
        ``allowed`` is True if the position can be opened.
        ``reason`` explains why it was blocked (empty if allowed).
    """
    new_groups = get_correlation_groups(symbol)
    if not new_groups:
        return True, ""

    # Count existing same-direction positions per group
    group_counts: Dict[str, int] = {}
    for _sid, (pos_symbol, pos_direction) in active_positions.items():
        if pos_direction != direction:
            continue
        pos_groups = get_correlation_groups(pos_symbol)
        for g in pos_groups:
            group_counts[g] = group_counts.get(g, 0) + 1

    # Check if adding this signal would exceed the limit in any shared group
    for g in new_groups:
        current = group_counts.get(g, 0)
        if current >= max_per_group:
            return False, (
                f"Correlation limit: {current}/{max_per_group} {direction} positions "
                f"already open in group '{g}'"
            )

    return True, ""
