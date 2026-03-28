# PR 01: High-Probability Filter Module

**Objective:** Introduce adaptive scoring per pair/channel to filter high-probability scalp signals.

**Description:**
- Score inputs: market regime, pair spread, liquidity, historical hit rate, volatility.
- Output: probability score per pair.
- Thresholding: signals only allowed above a dynamic probability threshold.

**Files to Update:**
- scanner/filter_module.py (new file)
- scanner/scalp_channels.py (update to use new filter)
- utils/pair_metrics.py (for pair scoring)

**Implementation Steps:**
1. Create filter_module.py with function get_pair_probability(pair_data) returning 0–100 score.
2. Update each scalp channel to call get_pair_probability() before generating signals.
3. Only allow signals if score > threshold (default: 70, adjustable per channel).
4. Add logging for suppressed signals with probability score.