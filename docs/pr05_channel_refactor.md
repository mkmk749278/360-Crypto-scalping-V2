# PR 05: Refactor Channel Logic

**Objective:** Remove duplicated code across scalp channels.

**Description:**
- Common gating, suppression, and probability logic moved to shared module.
- Channel-specific extensions added for FVG, CVD, VWAP, OBI.

**Files to Update:**
- scanner/common_gates.py (new file)
- scanner/scalp_channels.py (refactor channels to use common_gates)
- utils/pair_metrics.py (update shared functions)

**Implementation Steps:**
1. Identify duplicated gating logic in all scalp channels.
2. Move logic to common_gates.py.
3. Each channel imports common gates and applies additional strategy-specific rules.
4. Ensure backward compatibility with logs.