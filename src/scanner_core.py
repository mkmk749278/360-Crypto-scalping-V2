"""Scanner core — compatibility shim.

The Scanner implementation was moved from ``src/scanner.py`` into the
``src/scanner/`` subpackage (``src/scanner/__init__.py``) as part of
PR_14 (scanner decomposition).  This module re-exports everything from
the subpackage so that any code referencing ``src.scanner_core`` still
works.
"""
from __future__ import annotations

from src.scanner import *  # noqa: F401, F403
from src.scanner import (  # noqa: F401
    Scanner,
    ScanContext,
    _CHANNEL_GATE_PROFILE,
    _CHANNEL_PENALTY_WEIGHTS,
    _MTF_REGIME_CONFIG,
    _normalize_candle_dict,
    _RANGING_ADX_SUPPRESS_THRESHOLD,
    _REGIME_PENALTY_MULTIPLIER,
    _SCALP_CHANNELS,
    _SPREAD_CACHE_TTL,
    _SPREAD_FAIL_CACHE_TTL,
)
