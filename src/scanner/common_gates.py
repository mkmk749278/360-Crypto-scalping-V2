"""Common gate logic shared across all scalp channel strategies.

Extracts duplicated gating, suppression, and probability checks from
individual channel evaluators into a reusable module.  Channel-specific
extensions remain in each channel file.

PR 05 Implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.utils import get_logger

log = get_logger("common_gates")


# ---------------------------------------------------------------------------
# Gate result model
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Result of evaluating a single gate."""

    name: str
    passed: bool
    penalty: float = 0.0
    reason: str = ""


@dataclass
class GateCheckResult:
    """Aggregated result of all common gate checks for a signal."""

    passed: bool = True
    total_penalty: float = 0.0
    fired_gates: List[str] = field(default_factory=list)
    gate_results: List[GateResult] = field(default_factory=list)
    probability_score: float = 0.0

    def add_gate(self, result: GateResult) -> None:
        """Add a gate result to the aggregate."""
        self.gate_results.append(result)
        if not result.passed:
            self.passed = False
            self.fired_gates.append(result.name)
        elif result.penalty > 0:
            self.total_penalty += result.penalty
            self.fired_gates.append(f"{result.name}(penalty={result.penalty:.1f})")


# ---------------------------------------------------------------------------
# Common gate functions
# ---------------------------------------------------------------------------

def check_regime_compatibility(
    channel: str,
    regime: str,
    incompatible_map: Dict[str, List[str]],
) -> GateResult:
    """Check whether the channel is compatible with the current regime.

    Parameters
    ----------
    channel:
        Channel name (e.g. ``"360_SCALP"``).
    regime:
        Current market regime string.
    incompatible_map:
        Mapping of channel → blocked regimes.

    Returns
    -------
    GateResult
        Hard gate: ``passed=False`` if regime is in the blocked list.
    """
    blocked = incompatible_map.get(channel, [])
    regime_upper = regime.upper() if regime else ""
    if regime_upper in blocked:
        return GateResult(
            name="regime_compatibility",
            passed=False,
            reason=f"{channel} blocked in {regime_upper} regime",
        )
    return GateResult(name="regime_compatibility", passed=True)


def check_spread_gate(
    spread_pct: float,
    max_spread: float,
    regime: str = "",
) -> GateResult:
    """Check whether spread is within acceptable bounds.

    In QUIET regime, the spread tolerance is relaxed by 50% because
    spreads naturally widen in low-volume conditions.
    """
    effective_max = max_spread
    if regime.upper() == "QUIET":
        effective_max = max_spread * 1.5
    if spread_pct <= effective_max:
        return GateResult(name="spread", passed=True)
    return GateResult(
        name="spread",
        passed=False,
        reason=f"Spread {spread_pct:.4f} exceeds max {effective_max:.4f}",
    )


def check_volume_gate(
    volume_24h_usd: float,
    min_volume: float,
) -> GateResult:
    """Check whether 24h volume meets the minimum threshold."""
    if volume_24h_usd >= min_volume:
        return GateResult(name="volume", passed=True)
    return GateResult(
        name="volume",
        passed=False,
        reason=f"Volume ${volume_24h_usd:,.0f} below min ${min_volume:,.0f}",
    )


def compute_regime_penalty_multiplier(
    regime: str,
    channel: str,
    regime_multipliers: Dict[str, float],
    scalp_quiet_penalty: float = 1.8,
) -> float:
    """Compute the regime-specific penalty multiplier for soft gates.

    Parameters
    ----------
    regime:
        Current regime string.
    channel:
        Channel name.
    regime_multipliers:
        Mapping of regime → base multiplier.
    scalp_quiet_penalty:
        Override multiplier for SCALP channels in QUIET regime.
    """
    regime_key = regime.upper() if regime else ""
    if regime_key == "QUIET" and channel.startswith("360_SCALP"):
        return scalp_quiet_penalty
    return regime_multipliers.get(regime_key, 1.0)


def apply_soft_gate_penalty(
    gate_name: str,
    gate_failed: bool,
    base_penalty: float,
    regime_multiplier: float,
    channel_penalties: Optional[Dict[str, float]] = None,
    channel: str = "",
) -> GateResult:
    """Apply a regime-scaled soft penalty for a failed gate.

    Parameters
    ----------
    gate_name:
        Name of the gate (for logging/tracking).
    gate_failed:
        Whether the gate condition failed.
    base_penalty:
        Base confidence penalty points.
    regime_multiplier:
        Regime-specific scaling factor.
    channel_penalties:
        Per-channel penalty weight overrides.
    channel:
        Channel name for penalty lookup.
    """
    if not gate_failed:
        return GateResult(name=gate_name, passed=True, penalty=0.0)

    effective_base = base_penalty
    if channel_penalties and channel:
        effective_base = channel_penalties.get(channel, {}).get(gate_name, base_penalty)

    penalty = effective_base * regime_multiplier
    return GateResult(
        name=gate_name,
        passed=True,  # Soft gate — signal proceeds with penalty
        penalty=penalty,
        reason=f"Soft penalty: {penalty:.1f} (base={effective_base:.1f} × regime={regime_multiplier:.2f})",
    )


def run_common_gates(
    channel: str,
    regime: str,
    spread_pct: float,
    volume_24h_usd: float,
    max_spread: float,
    min_volume: float,
    incompatible_map: Optional[Dict[str, List[str]]] = None,
) -> GateCheckResult:
    """Run the common hard gates shared across all channels.

    Parameters
    ----------
    channel:
        Channel name.
    regime:
        Current market regime.
    spread_pct:
        Current bid-ask spread percentage.
    volume_24h_usd:
        24-hour trading volume in USD.
    max_spread:
        Maximum acceptable spread.
    min_volume:
        Minimum required 24h volume.
    incompatible_map:
        Regime-channel incompatibility mapping.

    Returns
    -------
    GateCheckResult
        Aggregated gate results with pass/fail and total penalty.
    """
    result = GateCheckResult()

    # 1. Regime compatibility (hard gate)
    if incompatible_map:
        regime_gate = check_regime_compatibility(channel, regime, incompatible_map)
        result.add_gate(regime_gate)
        if not regime_gate.passed:
            return result

    # 2. Spread gate (hard gate)
    spread_gate = check_spread_gate(spread_pct, max_spread, regime)
    result.add_gate(spread_gate)
    if not spread_gate.passed:
        return result

    # 3. Volume gate (hard gate)
    volume_gate = check_volume_gate(volume_24h_usd, min_volume)
    result.add_gate(volume_gate)
    if not volume_gate.passed:
        return result

    return result
