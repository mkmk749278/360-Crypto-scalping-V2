# PR 08: Testing & Simulation Module

**Objective:** Simulate new filters, dynamic SL/TP, and batch scanning before live deployment.

**Description:**
- Replay historical 7–30 days of data.
- Output performance metrics: hit rate, SL, latency.
- Allows validation before pushing to live trading.

**Files to Update:**
- simulation/simulator.py (new module)
- scanner/scanner.py (add simulation hooks)

**Implementation Steps:**
1. Build Simulator class to replay historical data.
2. Feed historical signals through new filter, dynamic SL/TP, and batch scanning.
3. Output CSV/JSON with success rate and latency metrics.
4. Adjust thresholds if necessary based on results.