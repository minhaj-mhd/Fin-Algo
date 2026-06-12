---
title: "Failed Approach: Unanimous Multi-Timeframe Conviction Strategy"
type: reference
status: active
updated: 2026-06-12
tags: []
---
# Failed Approach: Unanimous Multi-Timeframe Conviction Strategy

## Concept
The goal of this strategy was to enforce a strict alignment across all four model layers (15-Minute, 30-Minute, 1-Hour, and Daily) to identify exceptionally high-probability trades with unanimous momentum.

### Initial Hypothesis
If every timeframe model (Daily Gatekeeper, 1H, 30M, and 15M) is highly confident in the same direction, the intraday momentum should be strong enough to capture a high win-rate over short holding periods (15 minutes or 30 minutes).

## Iterations and Results

### Attempt 1: The Cascading Conviction Filter
**Condition:**
Enforced a perfect descending staircase of raw XGBoost conviction scores:
- `15m_score > 30m_score > 1h_score > daily_score > 0.05`

**Result:**
**0 Trades.** The raw probability outputs across different model layers are not uniformly distributed in a way that naturally forms this staircase. The requirement was overly restrictive, filtering out the entire market.

### Attempt 2: Relaxed Cascading Filter
**Condition:**
Same staircase condition but lowered the absolute minimum threshold to `0.01`:
- `15m_score > 30m_score > 1h_score > daily_score > 0.01`

**Result:**
**2 Trades total (1 Long, 1 Short).** While it allowed a couple of trades through, the aggregate PnL was negative (-0.92% for 15m hold, -1.91% for 30m hold). The cascading inequality (`A > B > C > D`) mathematically eliminates almost all viable setups because higher timeframes can easily spike a slightly higher probability score than an intermediate timeframe, invalidating the setup despite a strong unanimous directional consensus.

### Attempt 3: Strict Unanimous Direction + High Immediate Momentum (Rank-Based)
**Condition:**
Removed the `A > B > C > D` cascading requirement entirely. Instead:
- All four timeframes must be strictly positive (`> 0`).
- The 15-minute model must rank the ticker in the **Top 5** for that specific bar to ensure high immediate conviction.

**Result:**
**13 Trades total.**
- **15-Minute Hold:** Negative PnL (-1.39% total net return).
- **30-Minute Hold:** Positive PnL (+0.83% total net return) with a 66% win-rate on Short trades.

## Conclusion
While the third iteration (Top 5 15M Rank + Unanimous positive scores) showed some mild profitability on a 30-minute hold for shorts, the overall approach of requiring strict unanimous alignment across four separate timeframe layers is fundamentally flawed for short-duration intraday trading.

1. **Score Incompatibility:** Raw probability scores across different XGBoost models trained on different horizons cannot be directly compared via cascading inequalities.
2. **Lag Factor:** Requiring higher timeframes (like the Daily and 1H) to be fully aligned often means the move is already exhausted or highly consensus, making it difficult to extract edge in the immediate 15-minute window following the signal.
3. **Over-filtering:** The market rarely aligns perfectly across four time horizons. Enforcing this eliminates the vast majority of profitable, highly reactive 15-minute setups.

**Verdict:** Abandoned as a core standalone system. Future models should rely on blended ranking scores or let a higher timeframe act simply as a binary gatekeeper, rather than requiring unanimous, multi-layer absolute thresholds.
