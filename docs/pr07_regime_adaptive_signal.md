# PR 07: Regime-Adaptive Signal Scheduling

**Objective:** Ensure scalp signals are delivered in the optimal regime window.

**Description:**
- QUIET pairs → skip scalp channels.
- RANGING/TRENDING → priority scalp.
- Ensures high-probability scalp signals without false triggers.

**Files to Update:**
- scanner/regime_manager.py (update regime checks)
- scanner/scalp_channels.py (integrate with filter module)

**Implementation Steps:**
1. Update regime checks to dynamically determine allowed channels.
2. Integrate with probability scoring filter.
3. Add logging for skipped/regime-suppressed pairs.