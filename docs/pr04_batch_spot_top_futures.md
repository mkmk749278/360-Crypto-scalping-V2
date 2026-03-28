# PR 04: Batch Spot Scanning & Top Futures Prioritization

**Objective:** Stay within Binance API limits while maintaining timely scalp signals.

**Description:**
- Top 100 futures scanned in real-time.
- Spot pairs scanned in hourly batches.
- Scalp channel prioritized for real-time execution.

**Files to Update:**
- scanner/scanner.py
- utils/api_limits.py (new helper functions for batch scheduling)

**Implementation Steps:**
1. Create batch scheduler for spot pairs.
2. Always scan top 100 futures in real-time for scalp channels.
3. Prioritize scalp signals over other channels if latency is high.
4. Log batch execution and API/min usage.