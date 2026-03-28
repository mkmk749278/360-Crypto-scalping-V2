# PR 03: WebSocket & Scan Latency Optimization

**Objective:** Reduce delays and avoid missed scalp signals.

**Description:**
- Shard balancing for WebSocket streams (split spot/futures).
- Watchdog & reconnect improvements.
- Circuit breaker optimization for Tier 2 pairs.

**Files to Update:**
- ws_manager/ws_manager.py
- scanner/scanner.py (scan cycle improvements)

**Implementation Steps:**
1. Split WS streams to avoid single shard overload.
2. Reduce ping timeout errors with smarter reconnection logic.
3. Optimize scan loop: skip low-priority pairs when latency is high.
4. Add telemetry logs for WS health and latency.