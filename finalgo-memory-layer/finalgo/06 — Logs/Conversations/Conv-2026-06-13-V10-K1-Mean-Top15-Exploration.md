# 💬 Conversation Context: V10 K1 vs Mean Top 15 Conviction Exploration

## 📌 Metadata
- **Conversation ID**: cd48e7de-709e-462a-b27b-63d7493f8d78
- **Start Date**: 2026-06-13
- **Status**: 🔴 Concluded
- **Focus Area**: Research / Models (v10 1h)

## 🎯 Objectives
- [x] Write a script to backtest trading only on k1 when k1_conviction > mean(top 15) * (1 + threshold)
- [x] Run on OOS (Out-Of-Sample) datasets only for v10
- [x] Sweep multiple percentage thresholds to find the optimal relative magnitude
- [x] Summarize findings and determine the optimal percentage

## 💻 Active Code Files Modified
- [scripts/analysis/explore_v10_k1_mean_top15.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/analysis/explore_v10_k1_mean_top15.py)

## 📝 Compacted Session Log
- **Initial Analysis**: Bootstrapping session. Creating exploratory backtest script to evaluate if a relative magnitude check (k1 vs mean top 15) provides an edge on OOS data for v10.
- **Backtest Execution**: Ran a sweep from +0% to +200% over the mean of the top 15 conviction scores on the `2025-08` to `2026-05` OOS dataset (177k rows).
- **Findings**:
  - **Shorts scale beautifully**: The higher the K1 conviction relative to the top 15 mean, the better the short trades perform. At +50%, Short Win Rate hits 69.1% with +18.6 bps net (595 trades). At +75%, it hits 71.1% WR with +27.7 bps net.
  - **Longs break down at extremes**: Long trades see a mild improvement up to +50% (reaching 62.6% WR and +5.8 bps net), but completely collapse at +75% and beyond (going negative). This indicates that extreme long conviction in v10 is often spurious or chasing overextended breakouts.
  - **Optimal Threshold**: A baseline **+50%** threshold provides the best balanced uplift (Total WR jumps from 62.5% to 66.4%, Net bps from +8.4 to +13.3) while preserving 1,017 trades.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[02 — Models/1H/Model Card - v10 Native 1h]]
