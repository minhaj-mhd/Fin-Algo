---
title: "Conversation Context: Print v10 Final Tree"
type: log
status: active
updated: 2026-06-12
tags: []
---
# 💬 Conversation Context: Print v10 Final Tree

## 📌 Metadata
- **Conversation ID**: 97a0de4c-e037-44e0-9aa7-25be15769179
- **Start Date**: 2026-06-08
- **Status**: 🟢 Active
- **Focus Area**: Model Suite

## 🎯 Objectives
- [x] Print the v10 final tree

## 💻 Active Code Files Modified
- None

## 📝 Compacted Session Log
- **Initial Analysis**: The user requested to print the v10 final tree and noticed the depth of 5.
- **Execution**: Wrote and executed a Python script to extract the final trees (tree 50 for Long, tree 65 for Short) from the `v10_native_1h` XGBoost models. 
- **Experiment - Depth 4**: User requested to retrain v10 with `max_depth=4`. Modified `train_ranking_clean.py` to add a `1h_v3_d4` profile and ran the walk-forward evaluation.
- **Experiment - V12 and V13 Veto on V10 Host (Strict OOS: May 2026)**: Discovered that Jan-Apr 2026 was in-sample for the production models. Re-ran the exact backtest strictly on the unseen holdout month (May 2026).
- **Finding**: Even on perfectly unseen data, the validation thesis holds beautifully. On the LONG side, `v10` alone lost money (-6.7 bps net). But filtering with `v13 > p90` stripped away the noise, boosting Raw WR from 47.4% to 52.9% and Raw Gross edge from +3.3 bps to +15.2 bps, resulting in a **profitable +5.2 bps Net Edge**. On the SHORT side, `v13 > p90` lifted the baseline from a -2.1 bps net loss to a **+2.9 bps Net Edge**. `v13` is strictly superior to `v12` as a validation gate.
- **Experiment - Double Veto (V12 + V13)**: Tested requiring BOTH `v12 > p90` and `v13 > p90` to validate a trade.
- **Finding**: Adding `v12` as a second lock on top of `v13` slightly degraded performance. Long net edge dropped from +5.2 to +5.0 bps; Short net edge dropped from +2.9 to +2.4 bps. `v13_ndcg_raw` contains all the necessary signal; stacking `v12` just introduces noise and excludes valid trades.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[02 — Models/_Shared/Model Registry & File Structures]]
