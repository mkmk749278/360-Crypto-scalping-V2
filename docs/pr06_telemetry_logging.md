# PR 06: Telemetry & Logging Enhancements

**Objective:** Improve monitoring for suppressed signals and latency.

**Description:**
- Add probability score logging.
- Log suppressed signals per reason (regime, pair quality, volatility).
- Telemetry dashboards updated.

**Files to Update:**
- scanner/scanner.py (update logging)
- utils/logging.py (new helper functions)

**Implementation Steps:**
1. Add log_suppressed_signal(pair, channel, reason, probability_score) function.
2. Include probability scores in telemetry.
3. Monitor latency spikes and WS reconnections.